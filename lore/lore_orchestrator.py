# lore/lore_orchestrator.py - FULLY INTEGRATED VERSION WITH ALL MODULES

"""
Lore Orchestrator - Unified Entry Point for All Lore Functionality

This module provides a single, comprehensive interface to all lore system components,
including politics, religion, and world lore management with full resource management.

FULLY INTEGRATED: Includes education, geopolitical, local lore, politics, religion, 
and world lore managers with all their specialized functionality.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple, Set, AsyncGenerator
from datetime import datetime
import json
from enum import Enum
from dataclasses import dataclass
import uuid
import os
import asyncpg
import random

# Core imports
from db.connection import get_db_connection_context

# Nyx governance
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance, with_governance_permission

# Agents SDK imports for new managers
from agents import Agent, function_tool, Runner, ModelSettings, trace, handoff
from agents.run import RunConfig
from agents.run_context import RunContextWrapper
from pydantic import BaseModel, Field

# Import the specialized manager input/output models
from lore.managers.education import (
    EducationalSystem, KnowledgeTradition, TeachingContent,
    KnowledgeExchangeResult, StreamingPhaseUpdate
)
from lore.managers.geopolitical import (
    GeographicRegion, PoliticalEntity, BorderDispute, ConflictSimulation,
    EconomicTradeSimulation, ClimateGeographyEffect, CovertOperation
)
from lore.managers.local_lore import (
    LocationDataInput, MythCreationInput, HistoryCreationInput, LandmarkCreationInput,
    UrbanMyth, LocalHistory, Landmark, NarrativeEvolution, MythTransmissionResult,
    NarrativeConnection, ConsistencyCheckResult, TouristDevelopment, TraditionDynamics,
    LegendVariant, LocationLoreResult, LoreEvolutionResult,
    NarrativeStyle, EvolutionType, ConnectionType
)

# Import models from politics module
from lore.managers.politics import (
    DiplomaticNegotiationResult, MediaCoverageItem, DiplomaticNegotiation,
    FactionAgentProxy
)

# Import models from religion module  
from lore.managers.religion import (
    DeityParams, PantheonParams, ReligiousPracticeParams, HolySiteParams,
    ReligiousTextParams, ReligiousOrderParams, ReligiousConflictParams,
    NationReligionDistribution, CompleteRitual, SectarianPosition
)

logger = logging.getLogger(__name__)

# Database connection
DB_DSN = os.getenv("DB_DSN")

# Singleton instance storage
_ORCHESTRATOR_INSTANCES: Dict[Tuple[int, int], "LoreOrchestrator"] = {}


@dataclass
class OrchestratorConfig:
    """Configuration for the Lore Orchestrator"""
    enable_matriarchal_theme: bool = True
    enable_governance: bool = True
    enable_metrics: bool = True
    enable_validation: bool = True
    enable_cache: bool = True
    cache_ttl: int = 3600
    cache_max_size: int = 1000
    max_parallel_operations: int = 10
    auto_initialize: bool = True
    resource_limits: Dict[str, Any] = None
    redis_url: Optional[str] = None
    max_size_mb: float = 100
    

class LoreOrchestrator:
    """
    Master orchestrator that provides unified access to all lore functionality.
    Acts as the single entry point for external systems to interact with lore.
    
    FULLY INTEGRATED with:
    - Educational system management
    - Geopolitical system management  
    - Local lore and urban myth management
    - Politics and diplomacy management
    - Religion and faith systems management
    - World lore and resource management
    - Canon system for canonical state management
    - Cache system for performance optimization
    - Registry system for manager coordination
    - Validation system for data integrity
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
        
        # Core systems
        self._canon_module = None
        self._cache_system = None
        self._registry_system = None
        self._canon_validation = None
        self._canonical_context_class = None
        
        # Specialized managers (lazy loaded)
        self._education_manager = None
        self._geopolitical_manager = None
        self._local_lore_manager = None
        self._politics_manager = None  # NEW
        self._religion_manager = None  # NEW
        self._world_lore_manager = None  # NEW
        
        # Integration components
        self._npc_integration = None
        self._conflict_integration = None
        self._context_enhancer = None
        
        # Framework components
        self._matriarchal_framework = None
        self._matriarchal_power_framework = None
        
        # System components
        self._lore_dynamics_system = None
        self._regional_culture_system = None
        
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
        
        # World coordination components (NEW)
        self._master_coordinator = None
        self._unified_trace_system = None
        self._content_validator = None
        self._relationship_mapper = None
        
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
                
                # Initialize cache if enabled
                if self.config.enable_cache:
                    await self._initialize_cache()
                
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
    
    async def _initialize_cache(self):
        """Initialize cache system."""
        cache = await self._get_cache_system()
        self._component_init_status['cache'] = True
        logger.info("Cache system initialized")
    
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
        from lore.metrics import metrics_manager
        self._component_init_status['metrics'] = True
        logger.info("Metrics collection initialized")
    
    # ===== EDUCATIONAL SYSTEM OPERATIONS =====
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="educational_operation",
        action_description="Performing educational system operation",
        id_from_context=lambda ctx: "lore_orchestrator"
    )
    async def add_educational_system(
        self, ctx,
        name: str,
        system_type: str,
        description: str,
        target_demographics: List[str],
        controlled_by: str,
        core_teachings: List[str],
        teaching_methods: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Add an educational system.
        
        Args:
            name: Name of the educational system
            system_type: Type of system
            description: Description
            target_demographics: Target demographics
            controlled_by: Controlling faction
            core_teachings: Core teachings
            teaching_methods: Teaching methods
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with system ID and status
        """
        manager = await self._get_education_manager()
        from lore.managers.education import add_educational_system as add_edu_system
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await add_edu_system(
            run_ctx, name, system_type, description, target_demographics,
            controlled_by, core_teachings, teaching_methods, **kwargs
        )
    
    async def add_knowledge_tradition(
        self, ctx,
        name: str,
        tradition_type: str,
        description: str,
        knowledge_domain: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Add a knowledge tradition.
        
        Args:
            name: Name of the tradition
            tradition_type: Type of tradition
            description: Description
            knowledge_domain: Domain of knowledge
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with tradition ID and status
        """
        manager = await self._get_education_manager()
        from lore.managers.education import add_knowledge_tradition as add_tradition
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await add_tradition(
            run_ctx, name, tradition_type, description, knowledge_domain, **kwargs
        )
    
    async def add_teaching_content(
        self, ctx,
        system_id: int,
        title: str,
        content_type: str,
        subject_area: str,
        description: str,
        target_age_group: str,
        key_points: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Add teaching content to an educational system.
        
        Args:
            system_id: ID of the educational system
            title: Content title
            content_type: Type of content
            subject_area: Subject area
            description: Description
            target_age_group: Target age group
            key_points: Key points
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with content ID and status
        """
        manager = await self._get_education_manager()
        from lore.managers.education import add_teaching_content as add_content
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await add_content(
            run_ctx, system_id, title, content_type, subject_area,
            description, target_age_group, key_points, **kwargs
        )
    
    async def generate_educational_systems(self, ctx=None) -> List[Dict[str, Any]]:
        """
        Generate educational systems.
        
        Returns:
            List of generated educational systems
        """
        manager = await self._get_education_manager()
        return await manager.generate_educational_systems(ctx)
    
    async def generate_knowledge_traditions(self, ctx=None) -> List[Dict[str, Any]]:
        """
        Generate knowledge traditions.
        
        Returns:
            List of generated knowledge traditions
        """
        manager = await self._get_education_manager()
        return await manager.generate_knowledge_traditions(ctx)
    
    async def stream_educational_development(
        self, ctx,
        system_name: str,
        system_type: str,
        matriarchy_level: int = 8
    ) -> AsyncGenerator[StreamingPhaseUpdate, None]:
        """
        Stream the development of a complete educational system.
        
        Args:
            system_name: Name of the system
            system_type: Type of system
            matriarchy_level: Level of matriarchy (1-10)
            
        Yields:
            StreamingPhaseUpdate objects for each phase
        """
        manager = await self._get_education_manager()
        from lore.managers.education import stream_educational_development as stream_edu
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        async for update in stream_edu(run_ctx, system_name, system_type, matriarchy_level):
            yield update
    
    async def exchange_knowledge_between_systems(
        self, ctx,
        source_system_id: int,
        target_system_id: int,
        knowledge_domain: str
    ) -> Dict[str, Any]:
        """
        Facilitate knowledge exchange between two educational systems.
        
        Args:
            source_system_id: Source system ID
            target_system_id: Target system ID
            knowledge_domain: Domain of knowledge to exchange
            
        Returns:
            Dictionary with exchange results
        """
        manager = await self._get_education_manager()
        from lore.managers.education import exchange_knowledge_between_systems as exchange_knowledge
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await exchange_knowledge(run_ctx, source_system_id, target_system_id, knowledge_domain)
    
    async def search_educational_systems(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search educational systems by semantic similarity.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching systems
        """
        manager = await self._get_education_manager()
        from lore.managers.education import search_educational_systems as search_edu
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await search_edu(run_ctx, query, limit)
    
    async def get_teaching_contents(
        self,
        system_id: int,
        subject_area: Optional[str] = None,
        include_restricted: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get teaching contents for an educational system.
        
        Args:
            system_id: System ID
            subject_area: Optional subject area filter
            include_restricted: Whether to include restricted content
            
        Returns:
            List of teaching contents
        """
        manager = await self._get_education_manager()
        from lore.managers.education import get_teaching_contents
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await get_teaching_contents(run_ctx, system_id, subject_area, include_restricted)
    
    # ===== GEOPOLITICAL OPERATIONS =====
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="geopolitical_operation",
        action_description="Performing geopolitical operation",
        id_from_context=lambda ctx: "lore_orchestrator"
    )
    async def add_geographic_region(
        self, ctx,
        name: str,
        region_type: str,
        description: str,
        **kwargs
    ) -> int:
        """
        Add a geographic region.
        
        Args:
            name: Region name
            region_type: Type of region
            description: Description
            **kwargs: Additional parameters (climate, resources, governing_faction, etc.)
            
        Returns:
            Region ID
        """
        manager = await self._get_geopolitical_manager()
        from lore.managers.geopolitical import GeopoliticalSystemManager
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        # Call the static function tool which properly handles all the parameters
        return await GeopoliticalSystemManager.add_geographic_region(
            run_ctx, name, region_type, description, **kwargs
        )
    
    async def generate_world_nations(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Generate world nations.
        
        Args:
            count: Number of nations to generate
            
        Returns:
            List of generated nations with all their properties
        """
        manager = await self._get_geopolitical_manager()
        from lore.managers.geopolitical import GeopoliticalSystemManager
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        # Call the static function tool
        return await GeopoliticalSystemManager.generate_world_nations(run_ctx, count)
    
    async def simulate_conflict(
        self,
        entity_ids: List[int],
        conflict_type: str,
        alliances: Optional[Dict[str, List[int]]] = None,
        duration_months: int = 12
    ) -> Dict[str, Any]:
        """
        Simulate conflict between entities (supports multi-party conflicts).
        
        Args:
            entity_ids: List of entity IDs involved (minimum 2)
            conflict_type: Type of conflict (war, trade_war, diplomatic, etc.)
            alliances: Optional alliance structure {alliance_name: [member_ids]}
            duration_months: Duration of conflict simulation in months
            
        Returns:
            Comprehensive conflict simulation results including outcomes
        """
        manager = await self._get_geopolitical_manager()
        from lore.managers.geopolitical import GeopoliticalSystemManager
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        # Call the static function tool
        return await GeopoliticalSystemManager.simulate_conflict(
            run_ctx, entity_ids, conflict_type, alliances, duration_months
        )
    
    async def resolve_border_dispute(
        self,
        dispute_id: int,
        resolution_approach: str
    ) -> Dict[str, Any]:
        """
        Resolve a border dispute.
        
        Args:
            dispute_id: ID of the dispute
            resolution_approach: Approach to use (diplomatic, military, arbitration, etc.)
            
        Returns:
            Resolution results with stability rating
        """
        manager = await self._get_geopolitical_manager()
        from lore.managers.geopolitical import GeopoliticalSystemManager
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        # Call the static function tool
        return await GeopoliticalSystemManager.resolve_border_dispute(
            run_ctx, dispute_id, resolution_approach
        )
    
    async def predict_geopolitical_evolution(
        self,
        entity_id: int,
        years_forward: int = 5,
        include_events: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Predict geopolitical evolution of an entity.
        
        Args:
            entity_id: Entity ID
            years_forward: Years to predict forward
            include_events: Whether to include events
            
        Yields:
            Evolution updates as they are generated
        """
        manager = await self._get_geopolitical_manager()
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        # This is a generator function, so we need to properly yield from it
        async for update in manager.predict_geopolitical_evolution(
            run_ctx, entity_id, years_forward, include_events
        ):
            yield update
    
    async def simulate_trade(
        self,
        nation1: str,
        nation2: str,
        trade_goods: List[str],
        trade_route: str
    ) -> Dict[str, Any]:
        """
        Simulate economic trade between nations.
        
        Args:
            nation1: First nation name
            nation2: Second nation name
            trade_goods: List of goods being traded
            trade_route: Trade route description
            
        Returns:
            Trade simulation results with economic impacts
        """
        manager = await self._get_geopolitical_manager()
        from lore.managers.geopolitical import GeopoliticalSystemManager
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        # Call the static function tool
        return await GeopoliticalSystemManager.simulate_trade(
            run_ctx, nation1, nation2, trade_goods, trade_route
        )
    
    async def simulate_geography_impact(
        self,
        region_name: str,
        terrain_features: List[str],
        climate_type: str
    ) -> Dict[str, Any]:
        """
        Simulate geography impact on political development.
        
        Args:
            region_name: Name of region
            terrain_features: List of terrain features (mountains, rivers, etc.)
            climate_type: Climate type (temperate, arid, tropical, etc.)
            
        Returns:
            Impact simulation results with political stability effects
        """
        manager = await self._get_geopolitical_manager()
        from lore.managers.geopolitical import GeopoliticalSystemManager
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        # Call the static function tool
        return await GeopoliticalSystemManager.simulate_geography_impact(
            run_ctx, region_name, terrain_features, climate_type
        )
    
    async def simulate_espionage(
        self,
        agent_name: str,
        target_nation: str,
        operation_type: str,
        secrecy_level: int
    ) -> Dict[str, Any]:
        """
        Simulate covert operations between nations.
        
        Args:
            agent_name: Name of the agent conducting the operation
            target_nation: Target nation name
            operation_type: Type of operation (intelligence_gathering, sabotage, etc.)
            secrecy_level: Secrecy level (1-10, higher = more secret)
            
        Returns:
            Espionage simulation results with mission outcome
        """
        manager = await self._get_geopolitical_manager()
        from lore.managers.geopolitical import GeopoliticalSystemManager
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        # Call the static function tool
        return await GeopoliticalSystemManager.simulate_espionage(
            run_ctx, agent_name, target_nation, operation_type, secrecy_level
        )
    
    # ===== LOCAL LORE OPERATIONS =====
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="local_lore_operation",
        action_description="Performing local lore operation",
        id_from_context=lambda ctx: "lore_orchestrator"
    )
    async def add_urban_myth(self, ctx, input: MythCreationInput) -> int:
        """
        Add an urban myth.
        
        Args:
            input: MythCreationInput with myth details
            
        Returns:
            Myth ID
        """
        manager = await self._get_local_lore_manager()
        from lore.managers.local_lore import add_urban_myth
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await add_urban_myth(run_ctx, input)
    
    async def add_local_history(self, ctx, input: HistoryCreationInput) -> int:
        """
        Add a local historical event.
        
        Args:
            input: HistoryCreationInput with event details
            
        Returns:
            History ID
        """
        manager = await self._get_local_lore_manager()
        from lore.managers.local_lore import add_local_history
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await add_local_history(run_ctx, input)
    
    async def add_landmark(self, ctx, input: LandmarkCreationInput) -> int:
        """
        Add a landmark.
        
        Args:
            input: LandmarkCreationInput with landmark details
            
        Returns:
            Landmark ID
        """
        manager = await self._get_local_lore_manager()
        from lore.managers.local_lore import add_landmark
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await add_landmark(run_ctx, input)
    
    async def evolve_myth(
        self, ctx,
        myth_id: int,
        evolution_type: EvolutionType,
        causal_factors: Optional[List[str]] = None
    ) -> NarrativeEvolution:
        """
        Evolve an urban myth.
        
        Args:
            myth_id: ID of the myth
            evolution_type: Type of evolution
            causal_factors: Causal factors
            
        Returns:
            NarrativeEvolution results
        """
        manager = await self._get_local_lore_manager()
        from lore.managers.local_lore import evolve_myth
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await evolve_myth(run_ctx, myth_id, evolution_type, causal_factors)
    
    async def connect_myth_history(self, ctx, myth_id: int, history_id: int) -> NarrativeConnection:
        """
        Connect a myth to a historical event.
        
        Args:
            myth_id: Myth ID
            history_id: History ID
            
        Returns:
            NarrativeConnection
        """
        manager = await self._get_local_lore_manager()
        from lore.managers.local_lore import connect_myth_history
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await connect_myth_history(run_ctx, myth_id, history_id)
    
    async def connect_history_landmark(self, ctx, history_id: int, landmark_id: int) -> NarrativeConnection:
        """
        Connect a historical event to a landmark.
        
        Args:
            history_id: History ID
            landmark_id: Landmark ID
            
        Returns:
            NarrativeConnection
        """
        manager = await self._get_local_lore_manager()
        from lore.managers.local_lore import connect_history_landmark
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await connect_history_landmark(run_ctx, history_id, landmark_id)
    
    async def ensure_narrative_consistency(
        self, ctx,
        location_id: int,
        auto_fix: bool = True
    ) -> ConsistencyCheckResult:
        """
        Ensure narrative consistency for a location.
        
        Args:
            location_id: Location ID
            auto_fix: Whether to auto-fix issues
            
        Returns:
            ConsistencyCheckResult
        """
        manager = await self._get_local_lore_manager()
        from lore.managers.local_lore import ensure_narrative_consistency
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await ensure_narrative_consistency(run_ctx, location_id, auto_fix)
    
    async def get_location_lore(self, ctx, location_id: int) -> LocationLoreResult:
        """
        Get all lore for a location.
        
        Args:
            location_id: Location ID
            
        Returns:
            LocationLoreResult
        """
        manager = await self._get_local_lore_manager()
        from lore.managers.local_lore import get_location_lore
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await get_location_lore(run_ctx, location_id)
    
    async def generate_location_lore(self, ctx, location_data: LocationDataInput) -> Dict[str, Any]:
        """
        Generate comprehensive lore for a location.
        
        Args:
            location_data: LocationDataInput
            
        Returns:
            Generation statistics
        """
        manager = await self._get_local_lore_manager()
        from lore.managers.local_lore import generate_location_lore
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await generate_location_lore(run_ctx, location_data)
    
    async def evolve_location_lore(
        self, ctx,
        location_id: int,
        event_description: str
    ) -> LoreEvolutionResult:
        """
        Evolve location lore based on an event.
        
        Args:
            location_id: Location ID
            event_description: Event description
            
        Returns:
            LoreEvolutionResult
        """
        manager = await self._get_local_lore_manager()
        from lore.managers.local_lore import evolve_location_lore
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await evolve_location_lore(run_ctx, location_id, event_description)
    
    async def generate_legend_variants(
        self, ctx,
        myth_id: int,
        variant_count: int = 3
    ) -> Dict[str, Any]:
        """
        Generate contradictory legend variants.
        
        Args:
            myth_id: Myth ID
            variant_count: Number of variants
            
        Returns:
            Generated variants
        """
        manager = await self._get_local_lore_manager()
        from lore.managers.local_lore import generate_legend_variants
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await generate_legend_variants(run_ctx, myth_id, variant_count)
    
    async def develop_tourist_attraction(self, ctx, myth_id: int) -> TouristDevelopment:
        """
        Develop a tourist attraction from a myth.
        
        Args:
            myth_id: Myth ID
            
        Returns:
            TouristDevelopment plan
        """
        manager = await self._get_local_lore_manager()
        from lore.managers.local_lore import develop_tourist_attraction
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await develop_tourist_attraction(run_ctx, myth_id)
    
    async def simulate_tradition_dynamics(self, ctx, myth_id: int) -> TraditionDynamics:
        """
        Simulate oral vs written tradition dynamics.
        
        Args:
            myth_id: Myth ID
            
        Returns:
            TraditionDynamics
        """
        manager = await self._get_local_lore_manager()
        from lore.managers.local_lore import simulate_tradition_dynamics
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await simulate_tradition_dynamics(run_ctx, myth_id)
    
    async def simulate_myth_transmission(
        self, ctx,
        myth_id: int,
        target_regions: List[str],
        transmission_steps: int = 3
    ) -> MythTransmissionResult:
        """
        Simulate myth transmission across regions.
        
        Args:
            myth_id: Myth ID
            target_regions: Target regions
            transmission_steps: Number of steps
            
        Returns:
            MythTransmissionResult
        """
        manager = await self._get_local_lore_manager()
        from lore.managers.local_lore import simulate_myth_transmission
        from agents import RunContextWrapper
        
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "manager": manager
        })
        
        return await simulate_myth_transmission(run_ctx, myth_id, target_regions, transmission_steps)
    
    # ===== POLITICS OPERATIONS (NEW SECTION) =====
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="political_operation",
        action_description="Performing political operation",
        id_from_context=lambda ctx: "lore_orchestrator"
    )
    async def add_nation(
        self, ctx,
        name: str,
        government_type: str,
        description: str,
        relative_power: int,
        matriarchy_level: int,
        **kwargs
    ) -> int:
        """Add a nation to the political landscape."""
        manager = await self._get_politics_manager()
        return await manager.add_nation(
            ctx, name, government_type, description, relative_power,
            matriarchy_level, **kwargs
        )
    
    async def add_international_relation(
        self, ctx,
        nation1_id: int,
        nation2_id: int,
        relationship_type: str,
        relationship_quality: int,
        description: str,
        **kwargs
    ) -> int:
        """Add or update international relations between nations."""
        manager = await self._get_politics_manager()
        return await manager.add_international_relation(
            ctx, nation1_id, nation2_id, relationship_type, 
            relationship_quality, description, **kwargs
        )
    
    async def get_all_nations(self, ctx) -> List[Dict[str, Any]]:
        """Get all nations in the world."""
        manager = await self._get_politics_manager()
        return await manager.get_all_nations(ctx)
    
    async def generate_initial_conflicts(self, ctx, count: int = 3) -> List[Dict[str, Any]]:
        """Generate initial conflicts between nations."""
        manager = await self._get_politics_manager()
        return await manager.generate_initial_conflicts(ctx, count)
    
    async def stream_crisis_events(self, ctx, conflict_id: int) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream real-time updates about an evolving crisis."""
        manager = await self._get_politics_manager()
        async for event in manager.stream_crisis_events(ctx, conflict_id):
            yield event
    
    async def simulate_diplomatic_negotiation(
        self, ctx, 
        nation1_id: int, 
        nation2_id: int, 
        issue: str
    ) -> Dict[str, Any]:
        """Simulate diplomatic negotiations between two nations."""
        manager = await self._get_politics_manager()
        return await manager.simulate_diplomatic_negotiation(ctx, nation1_id, nation2_id, issue)
    
    async def simulate_media_coverage(self, ctx, event_id: int) -> Dict[str, Any]:
        """Simulate media coverage of a political event from different perspectives."""
        manager = await self._get_politics_manager()
        return await manager.simulate_media_coverage(ctx, event_id)
    
    async def generate_domestic_issues(self, ctx, nation_id: int, count: int = 2) -> List[Dict[str, Any]]:
        """Generate domestic issues for a specific nation."""
        manager = await self._get_politics_manager()
        return await manager.generate_domestic_issues(ctx, nation_id, count)
    
    async def get_active_conflicts(self, ctx) -> List[Dict[str, Any]]:
        """Get all active conflicts."""
        manager = await self._get_politics_manager()
        return await manager.get_active_conflicts(ctx)
    
    async def get_nation_politics(self, ctx, nation_id: int) -> Dict[str, Any]:
        """Get comprehensive political information about a nation."""
        manager = await self._get_politics_manager()
        return await manager.get_nation_politics(ctx, nation_id)
    
    async def evolve_all_conflicts(self, ctx, days_passed: int = 30) -> Dict[str, Any]:
        """Evolve all active conflicts over time."""
        manager = await self._get_politics_manager()
        return await manager.evolve_all_conflicts(ctx, days_passed)
    
    async def simulate_political_reforms(self, ctx, nation_id: int) -> Dict[str, Any]:
        """Model how a nation's political system might evolve under pressure."""
        manager = await self._get_politics_manager()
        return await manager.simulate_political_reforms(ctx, nation_id)
    
    async def track_dynasty_lineage(
        self, ctx, 
        dynasty_id: int, 
        generations_to_advance: int = 1
    ) -> Dict[str, Any]:
        """Advance a dynasty by generations."""
        manager = await self._get_politics_manager()
        return await manager.track_dynasty_lineage(ctx, dynasty_id, generations_to_advance)
    
    async def initialize_faction_proxies(self, ctx) -> Dict[str, Any]:
        """Initialize agent proxies for all factions in the world."""
        manager = await self._get_politics_manager()
        return await manager.initialize_faction_proxies(ctx)
    
    async def execute_coup(self, ctx, nation_id: int, new_leader_id: int, reason: str) -> Dict[str, Any]:
        """Execute a coup in a nation."""
        manager = await self._get_politics_manager()
        return await manager.execute_coup(ctx, nation_id, new_leader_id, reason)
    
    # ===== RELIGION OPERATIONS (NEW SECTION) =====
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="religious_operation",
        action_description="Performing religious operation",
        id_from_context=lambda ctx: "lore_orchestrator"
    )
    async def add_deity(self, ctx, params: DeityParams) -> int:
        """Add a deity to the pantheon."""
        manager = await self._get_religion_manager()
        return await manager.add_deity(ctx, params)
    
    async def add_pantheon(self, ctx, params: PantheonParams) -> int:
        """Add a pantheon to the world."""
        manager = await self._get_religion_manager()
        return await manager.add_pantheon(ctx, params)
    
    async def add_religious_practice(self, ctx, params: ReligiousPracticeParams) -> int:
        """Add a religious practice."""
        manager = await self._get_religion_manager()
        return await manager.add_religious_practice(ctx, params)
    
    async def add_holy_site(self, ctx, params: HolySiteParams) -> int:
        """Add a holy site."""
        manager = await self._get_religion_manager()
        return await manager.add_holy_site(ctx, params)
    
    async def add_religious_text(self, ctx, params: ReligiousTextParams) -> int:
        """Add a religious text."""
        manager = await self._get_religion_manager()
        return await manager.add_religious_text(ctx, params)
    
    async def add_religious_order(self, ctx, params: ReligiousOrderParams) -> int:
        """Add a religious order."""
        manager = await self._get_religion_manager()
        return await manager.add_religious_order(ctx, params)
    
    async def add_religious_conflict(self, ctx, params: ReligiousConflictParams) -> int:
        """Add a religious conflict."""
        manager = await self._get_religion_manager()
        return await manager.add_religious_conflict(ctx, params)
    
    async def generate_pantheon(self, ctx) -> Dict[str, Any]:
        """Generate a complete pantheon."""
        manager = await self._get_religion_manager()
        return await manager.generate_pantheon(ctx)
    
    async def generate_religious_practices(self, ctx, pantheon_id: int) -> List[Dict[str, Any]]:
        """Generate religious practices for a pantheon."""
        manager = await self._get_religion_manager()
        return await manager.generate_religious_practices(ctx, pantheon_id)
    
    async def generate_complete_faith_system(self, ctx) -> Dict[str, Any]:
        """Generate a complete faith system with all components."""
        manager = await self._get_religion_manager()
        return await manager.generate_complete_faith_system(ctx)
    
    async def distribute_religions(self, ctx) -> List[Dict[str, Any]]:
        """Distribute religions across nations."""
        manager = await self._get_religion_manager()
        return await manager.distribute_religions(ctx)
    
    async def generate_ritual(
        self, ctx,
        pantheon_id: int,
        deity_id: Optional[int] = None,
        purpose: str = "blessing",
        formality_level: int = 5
    ) -> Dict[str, Any]:
        """Generate a detailed religious ritual."""
        manager = await self._get_religion_manager()
        return await manager.generate_ritual(ctx, pantheon_id, deity_id, purpose, formality_level)
    
    async def simulate_theological_dispute(
        self, ctx,
        pantheon_id: int,
        dispute_topic: str
    ) -> Dict[str, Any]:
        """Simulate a theological dispute between religious factions."""
        manager = await self._get_religion_manager()
        return await manager.simulate_theological_dispute(ctx, pantheon_id, dispute_topic)
    
    async def evolve_religion_from_culture(
        self, ctx,
        pantheon_id: int,
        nation_id: int,
        years: int = 50
    ) -> Dict[str, Any]:
        """Evolve a religion based on cultural interaction over time."""
        manager = await self._get_religion_manager()
        return await manager.evolve_religion_from_culture(ctx, pantheon_id, nation_id, years)
    
    async def generate_sectarian_development(
        self, ctx,
        pantheon_id: int,
        trigger_event: str
    ) -> Dict[str, Any]:
        """Generate a sectarian split within a religion."""
        manager = await self._get_religion_manager()
        return await manager.generate_sectarian_development(ctx, pantheon_id, trigger_event)
    
    async def get_nation_religion(self, ctx, nation_id: int) -> Dict[str, Any]:
        """Get comprehensive religious information about a nation."""
        manager = await self._get_religion_manager()
        return await manager.get_nation_religion(ctx, nation_id)
    
    # ===== WORLD LORE OPERATIONS (NEW SECTION) =====
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="world_lore_operation",
        action_description="Performing world lore operation",
        id_from_context=lambda ctx: "lore_orchestrator"
    )
    async def get_world_data(self, world_id: str) -> Optional[Dict[str, Any]]:
        """Get world data from cache or fetch if not available."""
        manager = await self._get_world_lore_manager()
        return await manager.get_world_data(world_id)
    
    async def set_world_data(
        self,
        world_id: str,
        data: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set world data in cache."""
        manager = await self._get_world_lore_manager()
        return await manager.set_world_data(world_id, data, tags)
    
    async def invalidate_world_data(
        self,
        world_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate world data cache."""
        manager = await self._get_world_lore_manager()
        await manager.invalidate_world_data(world_id, recursive)
    
    async def get_world_history(self, world_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get world history from cache or fetch if not available."""
        manager = await self._get_world_lore_manager()
        return await manager.get_world_history(world_id)
    
    async def set_world_history(
        self,
        world_id: str,
        history: List[Dict[str, Any]],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set world history in cache."""
        manager = await self._get_world_lore_manager()
        return await manager.set_world_history(world_id, history, tags)
    
    async def get_world_events(self, world_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get world events from cache or fetch if not available."""
        manager = await self._get_world_lore_manager()
        return await manager.get_world_events(world_id)
    
    async def set_world_events(
        self,
        world_id: str,
        events: List[Dict[str, Any]],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set world events in cache."""
        manager = await self._get_world_lore_manager()
        return await manager.set_world_events(world_id, events, tags)
    
    async def get_world_relationships(self, world_id: str) -> Optional[Dict[str, Any]]:
        """Get world relationships from cache or fetch if not available."""
        manager = await self._get_world_lore_manager()
        return await manager.get_world_relationships(world_id)
    
    async def set_world_relationships(
        self,
        world_id: str,
        relationships: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set world relationships in cache."""
        manager = await self._get_world_lore_manager()
        return await manager.set_world_relationships(world_id, relationships, tags)
    
    async def get_world_metadata(self, world_id: str) -> Optional[Dict[str, Any]]:
        """Get world metadata from cache or fetch if not available."""
        manager = await self._get_world_lore_manager()
        return await manager.get_world_metadata(world_id)
    
    async def set_world_metadata(
        self,
        world_id: str,
        metadata: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set world metadata in cache."""
        manager = await self._get_world_lore_manager()
        return await manager.set_world_metadata(world_id, metadata, tags)
    
    async def create_world_element(
        self,
        element_type: str,
        element_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create any world element using the canon system."""
        manager = await self._get_world_lore_manager()
        return await manager.create_world_element(element_type, element_data)
    
    async def get_world_lore(self, world_id: int) -> Dict[str, Any]:
        """Retrieve comprehensive world lore including cultures, religions, and history."""
        manager = await self._get_world_lore_manager()
        return await manager.get_world_lore(world_id)
    
    async def update_world_lore(self, world_id: int, updates: Dict[str, Any]) -> bool:
        """Update world lore with new information."""
        manager = await self._get_world_lore_manager()
        return await manager.update_world_lore(world_id, updates)
    
    async def get_cultural_context(self, culture_id: int) -> Dict[str, Any]:
        """Get detailed cultural context including traditions, customs, and beliefs."""
        manager = await self._get_world_lore_manager()
        return await manager.get_cultural_context(culture_id)
    
    async def get_religious_context(self, religion_id: int) -> Dict[str, Any]:
        """Get detailed religious context including beliefs, practices, and hierarchy."""
        manager = await self._get_world_lore_manager()
        return await manager.get_religious_context(religion_id)
    
    async def get_historical_events(
        self,
        world_id: int,
        time_period: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve historical events, optionally filtered by time period."""
        manager = await self._get_world_lore_manager()
        return await manager.get_historical_events(world_id, time_period)
    
    async def query_world_state(self, query: str) -> str:
        """Handle a natural language query about the world state."""
        manager = await self._get_world_lore_manager()
        return await manager.query_world_state(query)
    
    async def resolve_world_inconsistencies(self, world_id: str) -> str:
        """Identify and resolve any inconsistencies in the world lore."""
        manager = await self._get_world_lore_manager()
        return await manager.resolve_world_inconsistencies(world_id)
    
    async def generate_world_summary(
        self,
        world_id: str,
        include_history: bool = True,
        include_current_state: bool = True
    ) -> str:
        """Generate world documentation for history and current state."""
        manager = await self._get_world_lore_manager()
        return await manager.generate_world_summary(world_id, include_history, include_current_state)
    
    async def validate_world_consistency(self) -> Dict[str, Any]:
        """Validate world consistency and find issues."""
        manager = await self._get_world_lore_manager()
        return await manager.validate_world_consistency()
    
    # ===== WORLD COORDINATION OPERATIONS (NEW SECTION) =====
    
    async def coordinate_lore_task(
        self,
        task_description: str,
        subsystems: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Coordinate a task across multiple lore subsystems.
        
        Args:
            task_description: Description of the task
            subsystems: List of subsystems involved
            context: Optional context for the task
            
        Returns:
            Execution plan and results
        """
        coordinator = await self._get_master_coordinator()
        return await coordinator.coordinate_task(task_description, subsystems, context or {})
    
    async def validate_lore_consistency(
        self,
        content: Dict[str, Any],
        content_type: str
    ) -> Dict[str, Any]:
        """
        Validate the consistency of newly generated lore content.
        
        Args:
            content: The content to validate
            content_type: Type of content
            
        Returns:
            Validation results
        """
        coordinator = await self._get_master_coordinator()
        return await coordinator.validate_consistency(content, content_type)
    
    async def get_coordination_status(self) -> Dict[str, Any]:
        """Get the current status of the lore coordination system."""
        coordinator = await self._get_master_coordinator()
        return await coordinator.get_status()
    
    async def validate_content(
        self,
        ctx,
        content: Dict[str, Any],
        content_type: str
    ) -> Dict[str, Any]:
        """
        Validate lore content for consistency and quality.
        
        Args:
            ctx: Context
            content: Content to validate
            content_type: Type of content
            
        Returns:
            Validation results
        """
        validator = await self._get_content_validator()
        return await validator.validate_content(ctx, content, content_type)
    
    async def create_relationship_graph(
        self,
        elements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a relationship graph from lore elements.
        
        Args:
            elements: List of lore elements
            
        Returns:
            Relationship graph
        """
        mapper = await self._get_relationship_mapper()
        return await mapper.create_relationship_graph(elements)
    
    async def find_related_elements(
        self,
        element_id: str,
        element_type: str,
        depth: int = 1
    ) -> Dict[str, Any]:
        """
        Find lore elements related to the specified element.
        
        Args:
            element_id: ID of the element
            element_type: Type of element
            depth: Depth of relationship search
            
        Returns:
            Related elements and their relationships
        """
        mapper = await self._get_relationship_mapper()
        return await mapper.find_related_elements(element_id, element_type, depth)
    
    async def start_lore_trace(
        self,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a trace for a lore operation.
        
        Args:
            operation: Name of the operation
            metadata: Optional metadata
            
        Returns:
            Trace ID
        """
        trace_system = await self._get_unified_trace_system()
        return trace_system.start_trace(operation, metadata)
    
    async def add_trace_step(
        self,
        trace_id: str,
        step_name: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a step to an existing trace.
        
        Args:
            trace_id: ID of the trace
            step_name: Name of the step
            data: Optional step data
        """
        trace_system = await self._get_unified_trace_system()
        trace_system.add_trace_step(trace_id, step_name, data)
    
    async def export_trace(
        self,
        trace_id: str,
        format_type: str = "json"
    ) -> Dict[str, Any]:
        """
        Export trace information.
        
        Args:
            trace_id: ID of the trace
            format_type: Format for export
            
        Returns:
            Trace data
        """
        trace_system = await self._get_unified_trace_system()
        return trace_system.export_trace(trace_id, format_type)
    
    # ===== CANON OPERATIONS =====
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="canonical_operation",
        action_description="Performing canonical operation",
        id_from_context=lambda ctx: "lore_orchestrator"
    )
    async def find_or_create_npc(self, ctx, npc_name: str, **kwargs) -> int:
        """
        Find or create an NPC canonically with semantic similarity checking.
        
        Args:
            npc_name: Name of the NPC
            **kwargs: Additional NPC attributes (role, affiliations, etc.)
            
        Returns:
            NPC ID
        """
        canon = await self._get_canon_module()
        async with get_db_connection_context() as conn:
            return await canon.find_or_create_npc(ctx, conn, npc_name, **kwargs)
    
    async def find_or_create_location(self, location_name: str, **kwargs) -> str:
        """
        Find or create a location canonically.
        
        Args:
            location_name: Name of the location
            **kwargs: Additional location attributes
            
        Returns:
            Location name (canonical)
        """
        canon = await self._get_canon_module()
        ctx = self._create_canonical_context()
        async with get_db_connection_context() as conn:
            return await canon.find_or_create_location(ctx, conn, location_name, **kwargs)
    
    async def find_or_create_faction(self, faction_name: str, **kwargs) -> int:
        """
        Find or create a faction canonically.
        
        Args:
            faction_name: Name of the faction
            **kwargs: Additional faction attributes
            
        Returns:
            Faction ID
        """
        canon = await self._get_canon_module()
        ctx = self._create_canonical_context()
        async with get_db_connection_context() as conn:
            return await canon.find_or_create_faction(ctx, conn, faction_name, **kwargs)
    
    async def log_canonical_event(self, event_text: str, tags: List[str] = None, significance: int = 5):
        """
        Log a canonical event to establish world history.
        
        Args:
            event_text: Description of the event
            tags: Event tags
            significance: Significance level (1-10)
        """
        canon = await self._get_canon_module()
        ctx = self._create_canonical_context()
        async with get_db_connection_context() as conn:
            await canon.log_canonical_event(ctx, conn, event_text, tags, significance)
    
    async def create_journal_entry(self, entry_text: str, **kwargs) -> int:
        """
        Create a journal entry integrated with memory system.
        
        Args:
            entry_text: Journal entry text
            **kwargs: Additional metadata
            
        Returns:
            Journal entry ID
        """
        canon = await self._get_canon_module()
        ctx = self._create_canonical_context()
        async with get_db_connection_context() as conn:
            return await canon.add_journal_entry(ctx, conn, entry_text, **kwargs)
    
    async def update_player_stat(self, player_name: str, stat_name: str, new_value: int, reason: str):
        """
        Update a player stat canonically with memory integration.
        
        Args:
            player_name: Name of the player
            stat_name: Name of the stat
            new_value: New stat value
            reason: Reason for the change
        """
        canon = await self._get_canon_module()
        ctx = self._create_canonical_context()
        async with get_db_connection_context() as conn:
            await canon.update_player_stat_canonically(ctx, conn, player_name, stat_name, new_value, reason)
    
    async def find_or_create_social_link(self, **kwargs) -> int:
        """
        Create or find a social relationship between entities.
        
        Args:
            **kwargs: Relationship parameters
            
        Returns:
            Link ID
        """
        canon = await self._get_canon_module()
        ctx = self._create_canonical_context()
        async with get_db_connection_context() as conn:
            return await canon.find_or_create_social_link(ctx, conn, **kwargs)
    
    async def propose_and_enact_change(self, entity_type: str, entity_identifier: Dict[str, Any], 
                                       updates: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Propose and enact a canonical change through the lore system.
        
        Args:
            entity_type: Type of entity to change
            entity_identifier: How to identify the entity
            updates: Changes to make
            reason: Reason for the change
            
        Returns:
            Change results
        """
        lore_system = await self._get_lore_system()
        ctx = self._create_canonical_context()
        return await lore_system.propose_and_enact_change(ctx, entity_type, entity_identifier, updates, reason)
    
    # ===== CACHE OPERATIONS =====
    
    async def cache_get(self, namespace: str, key: str, user_id: Optional[int] = None, 
                       conversation_id: Optional[int] = None) -> Any:
        """
        Get an item from cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            user_id: Optional user ID for scoping
            conversation_id: Optional conversation ID for scoping
            
        Returns:
            Cached value or None
        """
        if not self.config.enable_cache:
            return None
            
        cache = await self._get_cache_system()
        return await cache.get(namespace, key, user_id or self.user_id, 
                              conversation_id or self.conversation_id)
    
    async def cache_set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None,
                       user_id: Optional[int] = None, conversation_id: Optional[int] = None, 
                       priority: int = 0):
        """
        Set an item in cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            user_id: Optional user ID for scoping
            conversation_id: Optional conversation ID for scoping
            priority: Cache priority (0-10)
        """
        if not self.config.enable_cache:
            return
            
        cache = await self._get_cache_system()
        await cache.set(namespace, key, value, ttl or self.config.cache_ttl,
                       user_id or self.user_id, conversation_id or self.conversation_id,
                       priority)
    
    async def cache_invalidate(self, namespace: str, key: str, user_id: Optional[int] = None,
                              conversation_id: Optional[int] = None):
        """
        Invalidate a cache entry.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            user_id: Optional user ID for scoping
            conversation_id: Optional conversation ID for scoping
        """
        if not self.config.enable_cache:
            return
            
        cache = await self._get_cache_system()
        await cache.invalidate(namespace, key, user_id or self.user_id,
                              conversation_id or self.conversation_id)
    
    async def cache_invalidate_pattern(self, namespace: str, pattern: str):
        """
        Invalidate cache entries matching a pattern.
        
        Args:
            namespace: Cache namespace
            pattern: Regex pattern to match keys
        """
        if not self.config.enable_cache:
            return
            
        cache = await self._get_cache_system()
        await cache.invalidate_pattern(namespace, pattern, self.user_id, self.conversation_id)
    
    async def get_cache_analytics(self) -> Dict[str, Any]:
        """
        Get cache performance analytics.
        
        Returns:
            Cache analytics data
        """
        if not self.config.enable_cache:
            return {}
            
        cache = await self._get_cache_system()
        return await cache.get_cache_analytics()
    
    async def optimize_cache(self) -> Dict[str, Any]:
        """
        Optimize cache using AI-driven analysis.
        
        Returns:
            Optimization recommendations and results
        """
        if not self.config.enable_cache:
            return {"status": "cache disabled"}
            
        cache = await self._get_cache_system()
        return await cache.optimize_cache()
    
    # ===== REGISTRY OPERATIONS =====
    
    async def get_manager(self, manager_key: str) -> Any:
        """
        Get a specialized manager by key.
        
        Args:
            manager_key: Manager identifier
            
        Returns:
            Manager instance
        """
        registry = await self._get_registry_system()
        result = await registry.get_manager(manager_key)
        return await registry._get_or_init_manager(manager_key)
    
    async def get_available_manager_tools(self) -> List[Dict[str, Any]]:
        """
        Get all available tools across all managers.
        
        Returns:
            List of available tools with metadata
        """
        registry = await self._get_registry_system()
        tools = await registry.get_available_tools()
        return [tool.dict() for tool in tools]
    
    async def discover_manager_relationships(self) -> Dict[str, List[str]]:
        """
        Get the relationship map between managers.
        
        Returns:
            Manager relationship map
        """
        registry = await self._get_registry_system()
        result = await registry.discover_manager_relationships()
        return result.relationships
    
    async def execute_cross_manager_operation(self, starting_manager: str, target_manager: str,
                                             operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an operation requiring coordination between managers.
        
        Args:
            starting_manager: Starting manager key
            target_manager: Target manager key
            operation: Operation to perform
            params: Operation parameters
            
        Returns:
            Operation results
        """
        registry = await self._get_registry_system()
        from lore.core.registry import CrossManagerHandoffParams
        
        handoff_params = CrossManagerHandoffParams(
            starting_manager=starting_manager,
            target_manager=target_manager,
            operation=operation,
            params=params
        )
        
        result = await registry.execute_cross_manager_handoff(handoff_params)
        return result.dict()
    
    # ===== VALIDATION OPERATIONS =====
    
    async def validate_canon_duplicate(self, entity_type: str, proposal: Dict[str, Any], 
                                      existing_id: int) -> bool:
        """
        Validate if a proposed entity is a duplicate of an existing one.
        
        Args:
            entity_type: Type of entity
            proposal: Proposed entity data
            existing_id: ID of potentially duplicate entity
            
        Returns:
            True if duplicate, False otherwise
        """
        validator = await self._get_canon_validation()
        
        async with get_db_connection_context() as conn:
            if entity_type.lower() == "npc":
                return await validator.confirm_is_duplicate_npc(conn, proposal, existing_id)
            else:
                # Generic validation for other entity types
                logger.warning(f"No specific duplicate validator for entity type: {entity_type}")
                return False
    
    # ===== CONTEXT OPERATIONS =====
    
    def create_canonical_context(self, **kwargs) -> Any:
        """
        Create a canonical context object for operations.
        
        Args:
            **kwargs: Additional context attributes
            
        Returns:
            CanonicalContext instance
        """
        context_class = self._get_canonical_context_class()
        return context_class(self.user_id, self.conversation_id, **kwargs)
    
    def _create_canonical_context(self, **kwargs) -> Any:
        """
        Internal method to create canonical context.
        
        Args:
            **kwargs: Additional context attributes
            
        Returns:
            CanonicalContext instance
        """
        return self.create_canonical_context(**kwargs)
    
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
    
    # ===== MATRIARCHAL POWER FRAMEWORK OPERATIONS =====
    
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
    
    # ===== LORE DYNAMICS OPERATIONS =====
    
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

    async def get_faction_religion(self, faction_id: int) -> Dict[str, Any]:
        """
        Get religious information for a faction.
        
        Args:
            faction_id: ID of the faction
            
        Returns:
            Religious affiliations and practices
        """
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get faction data
                faction = await conn.fetchrow("""
                    SELECT * FROM Factions WHERE id = $1
                """, faction_id)
                
                if not faction:
                    return {"error": "Faction not found"}
                
                # Get religious orders associated with faction
                orders = await conn.fetch("""
                    SELECT ro.* FROM ReligiousOrders ro
                    WHERE $1 = ANY(ro.notable_members)
                       OR ro.name LIKE '%' || $2 || '%'
                """, faction['name'], faction['name'])
                
                # Get holy sites controlled by faction
                holy_sites = await conn.fetch("""
                    SELECT hs.* FROM HolySites hs
                    JOIN Locations l ON hs.location_id = l.id
                    WHERE l.controlling_faction = $1
                """, faction_id)
                
                return {
                    "faction": dict(faction),
                    "religious_orders": [dict(o) for o in orders],
                    "controlled_holy_sites": [dict(s) for s in holy_sites]
                }
    
    async def get_location_full_context(self, location_id: int) -> Dict[str, Any]:
        """
        Get complete context for a location including lore, politics, and religion.
        
        Args:
            location_id: ID of the location
            
        Returns:
            Complete location context
        """
        # Get base location data
        location_context = await self.get_location_context(f"location_{location_id}")
        
        # Get local lore
        local_lore = await self.get_location_lore(
            self.create_run_context(),
            location_id
        )
        
        # Get religious sites at location
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                holy_sites = await conn.fetch("""
                    SELECT * FROM HolySites WHERE location_id = $1
                """, location_id)
                
                # Get political control
                political = await conn.fetchrow("""
                    SELECT l.*, f.name as faction_name, n.name as nation_name
                    FROM Locations l
                    LEFT JOIN Factions f ON l.controlling_faction = f.id
                    LEFT JOIN Nations n ON l.nation_id = n.id
                    WHERE l.id = $1
                """, location_id)
        
        return {
            "base_context": location_context,
            "local_lore": local_lore,
            "holy_sites": [dict(s) for s in holy_sites],
            "political_control": dict(political) if political else {}
        }
    
    async def simulate_cultural_religious_exchange(
        self,
        nation1_id: int,
        nation2_id: int,
        exchange_type: str = "peaceful",
        duration_years: int = 10
    ) -> Dict[str, Any]:
        """
        Simulate cultural and religious exchange between nations.
        
        Args:
            nation1_id: First nation ID
            nation2_id: Second nation ID
            exchange_type: Type of exchange ('peaceful', 'conquest', 'trade')
            duration_years: Duration of exchange
            
        Returns:
            Exchange results
        """
        # Get cultural diffusion
        cultural_results = await self.simulate_cultural_diffusion(
            self.create_run_context(),
            nation1_id,
            nation2_id,
            duration_years
        )
        
        # Get religious evolution
        nation1_religion = await self.get_nation_religion(
            self.create_run_context(),
            nation1_id
        )
        
        if nation1_religion and "primary_pantheon" in nation1_religion:
            pantheon_id = nation1_religion["primary_pantheon"]["id"]
            religious_results = await self.evolve_religion_from_culture(
                self.create_run_context(),
                pantheon_id,
                nation2_id,
                duration_years
            )
        else:
            religious_results = {"message": "No primary religion to evolve"}
        
        return {
            "cultural_exchange": cultural_results,
            "religious_evolution": religious_results,
            "exchange_type": exchange_type,
            "duration_years": duration_years
        }

    async def get_resource_usage_stats(self) -> Dict[str, Any]:
        """
        Get detailed resource usage statistics.
        
        Returns:
            Resource usage statistics
        """
        world_manager = await self._get_world_lore_manager()
        return await world_manager.get_resource_stats()
    
    async def optimize_world_resources(self) -> Dict[str, Any]:
        """
        Optimize world lore resource usage.
        
        Returns:
            Optimization results
        """
        world_manager = await self._get_world_lore_manager()
        await world_manager.optimize_resources()
        return {"status": "optimized", "timestamp": datetime.now().isoformat()}
    
    async def cleanup_world_resources(self) -> Dict[str, Any]:
        """
        Clean up unused world lore resources.
        
        Returns:
            Cleanup results
        """
        world_manager = await self._get_world_lore_manager()
        await world_manager.cleanup_resources()
        return {"status": "cleaned", "timestamp": datetime.now().isoformat()}
    
    async def resolve_world_inconsistencies_with_fixes(self, world_id: str = "main") -> Dict[str, Any]:
        """
        Identify and automatically resolve world inconsistencies.
        
        Args:
            world_id: ID of the world
            
        Returns:
            Resolution results
        """
        world_manager = await self._get_world_lore_manager()
        return await world_manager.resolve_world_inconsistencies(world_id)
    
    async def generate_world_documentation(
        self,
        world_id: str = "main",
        include_history: bool = True,
        include_current_state: bool = True,
        format_type: str = "markdown"
    ) -> str:
        """
        Generate comprehensive world documentation.
        
        Args:
            world_id: ID of the world
            include_history: Include historical timeline
            include_current_state: Include current state
            format_type: Output format ('markdown', 'html', 'json')
            
        Returns:
            Formatted documentation
        """
        world_manager = await self._get_world_lore_manager()
        doc = await world_manager.generate_world_summary(world_id, include_history, include_current_state)
        
        if format_type == "html":
            # Convert markdown to HTML
            import markdown
            return markdown.markdown(doc)
        elif format_type == "json":
            # Parse and return as JSON structure
            return json.dumps({"documentation": doc, "world_id": world_id})
        return doc
    
    async def query_world_natural_language(self, query: str, world_id: str = "main") -> str:
        """
        Query the world state using natural language.
        
        Args:
            query: Natural language query
            world_id: ID of the world
            
        Returns:
            Query response
        """
        world_manager = await self._get_world_lore_manager()
        return await world_manager.query_world_state(query)
    
    async def export_world_trace(self, trace_id: str, format_type: str = "json") -> Dict[str, Any]:
        """
        Export a trace of world operations.
        
        Args:
            trace_id: ID of the trace
            format_type: Export format
            
        Returns:
            Trace data
        """
        trace_system = await self._get_unified_trace_system()
        return trace_system.export_trace(trace_id, format_type)
    
    async def create_lore_relationship_graph(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a relationship graph from lore elements.
        
        Args:
            elements: List of lore elements
            
        Returns:
            Relationship graph
        """
        mapper = await self._get_relationship_mapper()
        return await mapper.create_relationship_graph(elements)
    
    async def find_lore_connections(
        self,
        element_id: str,
        element_type: str,
        depth: int = 2,
        connection_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Find deep connections between lore elements.
        
        Args:
            element_id: ID of the element
            element_type: Type of element
            depth: Search depth
            connection_types: Types of connections to find
            
        Returns:
            Connection graph
        """
        mapper = await self._get_relationship_mapper()
        connections = await mapper.find_related_elements(element_id, element_type, depth)
        
        # Filter by connection types if specified
        if connection_types:
            filtered = []
            for conn in connections.get("related_elements", []):
                if conn.get("relationship_type") in connection_types:
                    filtered.append(conn)
            connections["related_elements"] = filtered
        
        return connections
    
    async def validate_lore_element(self, element: Dict[str, Any], element_type: str) -> Dict[str, Any]:
        """
        Validate a lore element for consistency and quality.
        
        Args:
            element: Element data
            element_type: Type of element
            
        Returns:
            Validation results
        """
        validator = await self._get_content_validator()
        return await validator.validate_content(
            self.create_run_context(),
            element,
            element_type
        )

    async def search_religious_content(self, query: str, content_type: str = 'all', limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search religious content by semantic similarity.
        
        Args:
            query: Search query
            content_type: Type of content ('deity', 'pantheon', 'practice', 'text', 'order', 'all')
            limit: Maximum results
            
        Returns:
            List of matching religious content
        """
        from utils.embedding_service import get_embedding
        
        embed = await get_embedding(query)
        results = []
        
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                if content_type in ['deity', 'all']:
                    deities = await conn.fetch("""
                        SELECT *, embedding <=> $1 as distance
                        FROM Deities
                        ORDER BY distance
                        LIMIT $2
                    """, embed, limit if content_type == 'deity' else 5)
                    results.extend([{**dict(d), 'type': 'deity'} for d in deities])
                
                if content_type in ['pantheon', 'all']:
                    pantheons = await conn.fetch("""
                        SELECT *, embedding <=> $1 as distance
                        FROM Pantheons
                        ORDER BY distance
                        LIMIT $2
                    """, embed, limit if content_type == 'pantheon' else 5)
                    results.extend([{**dict(p), 'type': 'pantheon'} for p in pantheons])
                
                if content_type in ['practice', 'all']:
                    practices = await conn.fetch("""
                        SELECT *, embedding <=> $1 as distance
                        FROM ReligiousPractices
                        ORDER BY distance
                        LIMIT $2
                    """, embed, limit if content_type == 'practice' else 5)
                    results.extend([{**dict(p), 'type': 'practice'} for p in practices])
        
        # Sort by distance and limit
        results.sort(key=lambda x: x.get('distance', float('inf')))
        return results[:limit]

    async def get_all_pantheons(self) -> List[Dict[str, Any]]:
        """
        Get all pantheons in the world.
        
        Returns:
            List of pantheons
        """
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheons = await conn.fetch("""
                    SELECT * FROM Pantheons
                    ORDER BY id
                """)
                return [dict(p) for p in pantheons]
    
    async def get_pantheon_deities(self, pantheon_id: int) -> List[Dict[str, Any]]:
        """
        Get all deities in a pantheon.
        
        Args:
            pantheon_id: ID of the pantheon
            
        Returns:
            List of deities
        """
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                deities = await conn.fetch("""
                    SELECT * FROM Deities
                    WHERE pantheon_id = $1
                    ORDER BY rank DESC
                """, pantheon_id)
                return [dict(d) for d in deities]
    
    async def get_religious_practices_by_nation(self, nation_id: int) -> List[Dict[str, Any]]:
        """
        Get religious practices specific to a nation.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            List of regional religious practices
        """
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                practices = await conn.fetch("""
                    SELECT rrp.*, rp.name, rp.practice_type, rp.purpose
                    FROM RegionalReligiousPractice rrp
                    JOIN ReligiousPractices rp ON rrp.practice_id = rp.id
                    WHERE rrp.nation_id = $1
                    ORDER BY rrp.importance DESC
                """, nation_id)
                return [dict(p) for p in practices]
    
    async def get_holy_sites_by_pantheon(self, pantheon_id: int) -> List[Dict[str, Any]]:
        """
        Get all holy sites for a pantheon.
        
        Args:
            pantheon_id: ID of the pantheon
            
        Returns:
            List of holy sites
        """
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                sites = await conn.fetch("""
                    SELECT * FROM HolySites
                    WHERE pantheon_id = $1
                    ORDER BY id
                """, pantheon_id)
                return [dict(s) for s in sites]
    
    async def get_religious_texts_by_pantheon(self, pantheon_id: int) -> List[Dict[str, Any]]:
        """
        Get all religious texts for a pantheon.
        
        Args:
            pantheon_id: ID of the pantheon
            
        Returns:
            List of religious texts
        """
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                texts = await conn.fetch("""
                    SELECT * FROM ReligiousTexts
                    WHERE pantheon_id = $1
                    ORDER BY id
                """, pantheon_id)
                return [dict(t) for t in texts]
    
    async def get_religious_conflicts_by_pantheon(self, pantheon_id: int) -> List[Dict[str, Any]]:
        """
        Get religious conflicts involving a pantheon.
        
        Args:
            pantheon_id: ID of the pantheon
            
        Returns:
            List of religious conflicts
        """
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get pantheon name first
                pantheon_name = await conn.fetchval("""
                    SELECT name FROM Pantheons WHERE id = $1
                """, pantheon_id)
                
                if pantheon_name:
                    conflicts = await conn.fetch("""
                        SELECT * FROM ReligiousConflicts
                        WHERE $1 = ANY(parties_involved)
                        ORDER BY beginning_date DESC
                    """, pantheon_name)
                    return [dict(c) for c in conflicts]
                return []

    async def get_nation_issues(self, nation_id: int) -> List[Dict[str, Any]]:
        """
        Get all domestic issues for a specific nation.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            List of domestic issues
        """
        manager = await self._get_politics_manager()
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                issues = await conn.fetch("""
                    SELECT * FROM DomesticIssues
                    WHERE nation_id = $1
                    ORDER BY severity DESC
                """, nation_id)
                return [dict(issue) for issue in issues]
    
    async def get_faction_proxy(self, faction_id: int) -> Optional[Any]:
        """
        Get a faction agent proxy for autonomous faction behavior.
        
        Args:
            faction_id: ID of the faction
            
        Returns:
            FactionAgentProxy instance or None
        """
        manager = await self._get_politics_manager()
        if not manager.faction_proxies:
            await manager.initialize_faction_proxies(self.create_run_context())
        return manager.faction_proxies.get(faction_id)
    
    async def simulate_faction_reaction(self, faction_id: int, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate how a faction reacts to an event.
        
        Args:
            faction_id: ID of the faction
            event: Event data
            
        Returns:
            Faction reaction
        """
        proxy = await self.get_faction_proxy(faction_id)
        if not proxy:
            return {"error": f"Faction {faction_id} not found or not initialized"}
        return await proxy.react_to_event(event, self.create_run_context())
    
    async def create_dynasty(self, name: str, founding_date: str, ruling_nation: int, **kwargs) -> int:
        """
        Create a new dynasty.
        
        Args:
            name: Dynasty name
            founding_date: Founding date
            ruling_nation: ID of ruling nation
            **kwargs: Additional dynasty attributes
            
        Returns:
            Dynasty ID
        """
        manager = await self._get_politics_manager()
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                dynasty_id = await conn.fetchval("""
                    INSERT INTO Dynasties (
                        name, founding_date, ruling_nation, matriarch, 
                        patriarch, notable_members, family_traits
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id
                """, 
                name, founding_date, ruling_nation,
                kwargs.get('matriarch'), kwargs.get('patriarch'),
                kwargs.get('notable_members', []), kwargs.get('family_traits', []))
                
                return dynasty_id
    
    async def get_conflict_news(self, conflict_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get news articles about a specific conflict.
        
        Args:
            conflict_id: ID of the conflict
            limit: Maximum number of articles
            
        Returns:
            List of news articles
        """
        async with await self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                news = await conn.fetch("""
                    SELECT * FROM ConflictNews
                    WHERE conflict_id = $1
                    ORDER BY publication_date DESC
                    LIMIT $2
                """, conflict_id, limit)
                return [dict(article) for article in news]
    
    # ===== REGIONAL CULTURE OPERATIONS =====
    
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
        ctx = self._create_mock_context(nation_id=nation_id)
        return await culture_system.get_nation_culture(ctx, nation_id)
    
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
        ctx = self._create_mock_context(
            action='evolve_conflicts',
            days_passed=days_passed
        )
        
        if hasattr(conflict_system, 'evolve_all_conflicts'):
            return await conflict_system.evolve_all_conflicts(ctx, days_passed)
        else:
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
        ctx = self._create_mock_context(
            event_description=event_description[:100],
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
        
        # Cleanup cache if it exists
        if self._cache_system:
            # Cache doesn't have explicit cleanup but we can clear it
            await self._cache_system.clear_namespace("*")
        
        # Cleanup specialized managers
        for manager in [
            self._education_manager, 
            self._geopolitical_manager, 
            self._local_lore_manager,
            self._politics_manager,
            self._religion_manager,
            self._world_lore_manager
        ]:
            if manager:
                if hasattr(manager, 'cleanup'):
                    await manager.cleanup()
                elif hasattr(manager, 'close'):
                    await manager.close()
                elif hasattr(manager, 'stop'):
                    await manager.stop()
                # Also clear any agent tasks if they exist
                if hasattr(manager, 'maintenance_task') and manager.maintenance_task:
                    manager.maintenance_task.cancel()
                    try:
                        await manager.maintenance_task
                    except asyncio.CancelledError:
                        pass
        
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
            self._lore_dynamics_system,
            self._master_coordinator,
            self._unified_trace_system,
            self._content_validator,
            self._relationship_mapper
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
        """
        # Default context structure many governance decorators expect
        if 'context' not in attributes:
            attributes['context'] = {
                'user_id': self.user_id,
                'conversation_id': self.conversation_id
            }
        
        # Create and return mock object with attributes
        return type('MockContext', (object,), attributes)()
    
    # ===== COMPONENT GETTERS (LAZY INITIALIZATION) =====
    
    async def _get_canon_module(self):
        """
        Get or initialize the canon module.
        CRITICAL: This is lazy loaded as many other modules depend on it.
        """
        if not self._canon_module:
            # Import here to avoid circular dependencies
            from lore.core import canon
            self._canon_module = canon
            logger.info("Canon module loaded")
        return self._canon_module
    
    async def _get_cache_system(self):
        """Get or initialize the cache system."""
        if not self._cache_system:
            from lore.core.cache import LoreCache
            self._cache_system = LoreCache(
                max_size=self.config.cache_max_size,
                ttl=self.config.cache_ttl
            )
            logger.info("Cache system initialized")
        return self._cache_system
    
    async def _get_registry_system(self):
        """Get or initialize the manager registry."""
        if not self._registry_system:
            from lore.core.registry import ManagerRegistry
            self._registry_system = ManagerRegistry(self.user_id, self.conversation_id)
            logger.info("Registry system initialized")
        return self._registry_system
    
    async def _get_canon_validation(self):
        """Get or initialize the canon validation agent."""
        if not self._canon_validation:
            from lore.core.validation import CanonValidationAgent
            self._canon_validation = CanonValidationAgent()
            logger.info("Canon validation agent initialized")
        return self._canon_validation
    
    def _get_canonical_context_class(self):
        """Get the CanonicalContext class (lazy loaded)."""
        if not self._canonical_context_class:
            from lore.core.context import CanonicalContext
            self._canonical_context_class = CanonicalContext
            logger.info("CanonicalContext class loaded")
        return self._canonical_context_class
    
    async def _get_education_manager(self):
        """Get or initialize the education manager."""
        if not self._education_manager:
            from lore.managers.education import EducationalSystemManager
            self._education_manager = EducationalSystemManager(self.user_id, self.conversation_id)
            await self._education_manager.ensure_initialized()
            if self._governor:
                self._education_manager.set_governor(self._governor)
                await self._education_manager.register_with_governance(
                    AgentType.NARRATIVE_CRAFTER,
                    "education_manager",
                    "Manages educational systems and knowledge traditions",
                    "education",
                    DirectivePriority.MEDIUM
                )
            logger.info("Education manager initialized")
        return self._education_manager
    
    async def _get_geopolitical_manager(self):
        """Get or initialize the geopolitical manager."""
        if not self._geopolitical_manager:
            from lore.managers.geopolitical import GeopoliticalSystemManager
            self._geopolitical_manager = GeopoliticalSystemManager(self.user_id, self.conversation_id)
            await self._geopolitical_manager.ensure_initialized()
            if self._governor:
                self._geopolitical_manager.set_governor(self._governor)
                await self._geopolitical_manager.register_with_governance(
                    AgentType.NARRATIVE_CRAFTER,
                    "geopolitical_manager",
                    "Manages geopolitical systems and conflicts",
                    "geopolitics",
                    DirectivePriority.MEDIUM
                )
            logger.info("Geopolitical manager initialized")
        return self._geopolitical_manager
    
    async def _get_local_lore_manager(self):
        """Get or initialize the local lore manager."""
        if not self._local_lore_manager:
            from lore.managers.local_lore import LocalLoreManager
            self._local_lore_manager = LocalLoreManager(self.user_id, self.conversation_id)
            await self._local_lore_manager.ensure_initialized()
            if self._governor:
                self._local_lore_manager.set_governor(self._governor)
                await self._local_lore_manager.register_with_governance()
            logger.info("Local lore manager initialized")
        return self._local_lore_manager
    
    async def _get_politics_manager(self):
        """Get or initialize the politics manager."""
        if not self._politics_manager:
            from lore.managers.politics import WorldPoliticsManager
            self._politics_manager = WorldPoliticsManager(self.user_id, self.conversation_id)
            await self._politics_manager.ensure_initialized()
            if self._governor:
                await self._politics_manager.register_with_governance()
            logger.info("Politics manager initialized")
        return self._politics_manager
    
    async def _get_religion_manager(self):
        """Get or initialize the religion manager."""
        if not self._religion_manager:
            from lore.managers.religion import ReligionManager
            self._religion_manager = ReligionManager(self.user_id, self.conversation_id)
            await self._religion_manager.ensure_initialized()
            if self._governor:
                await self._religion_manager.register_with_governance()
            logger.info("Religion manager initialized")
        return self._religion_manager
    
    async def _get_world_lore_manager(self):
        """Get or initialize the world lore manager."""
        if not self._world_lore_manager:
            from lore.managers.world_lore_manager import WorldLoreManager
            self._world_lore_manager = WorldLoreManager(
                self.user_id, 
                self.conversation_id,
                max_size_mb=self.config.max_size_mb,
                redis_url=self.config.redis_url
            )
            await self._world_lore_manager.start()
            logger.info("World lore manager initialized")
        return self._world_lore_manager
    
    async def _get_master_coordinator(self):
        """Get or initialize the master coordination agent."""
        if not self._master_coordinator:
            from lore.managers.world_lore_manager import MasterCoordinationAgent
            world_lore_manager = await self._get_world_lore_manager()
            self._master_coordinator = MasterCoordinationAgent(world_lore_manager)
            await self._master_coordinator.initialize(self.user_id, self.conversation_id)
            logger.info("Master coordinator initialized")
        return self._master_coordinator
    
    async def _get_unified_trace_system(self):
        """Get or initialize the unified trace system."""
        if not self._unified_trace_system:
            from lore.managers.world_lore_manager import UnifiedTraceSystem
            self._unified_trace_system = UnifiedTraceSystem(self.user_id, self.conversation_id)
            logger.info("Unified trace system initialized")
        return self._unified_trace_system
    
    async def _get_content_validator(self):
        """Get or initialize the content validation tool."""
        if not self._content_validator:
            from lore.managers.world_lore_manager import ContentValidationTool
            world_lore_manager = await self._get_world_lore_manager()
            self._content_validator = ContentValidationTool(world_lore_manager)
            logger.info("Content validator initialized")
        return self._content_validator
    
    async def _get_relationship_mapper(self):
        """Get or initialize the relationship mapper."""
        if not self._relationship_mapper:
            from lore.managers.world_lore_manager import LoreRelationshipMapper
            world_lore_manager = await self._get_world_lore_manager()
            self._relationship_mapper = LoreRelationshipMapper(world_lore_manager)
            logger.info("Relationship mapper initialized")
        return self._relationship_mapper
    
    async def _get_lore_system(self):
        """Get or initialize the core lore system."""
        if not self._lore_system:
            from lore.core.lore_system import LoreSystem
            self._lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
            # Set governor if available
            if self._governor:
                self._lore_system.set_governor(self._governor)
            await self._lore_system.ensure_initialized()
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
        """Get or initialize regional culture system."""
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
        """Get or initialize matriarchal power framework."""
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
        """Get or initialize lore dynamics system."""
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
    
    def create_run_context(self, additional_context: Dict[str, Any] = None) -> RunContextWrapper:
        """
        Create a RunContextWrapper for agent operations.
        
        Args:
            additional_context: Additional context to include
            
        Returns:
            RunContextWrapper instance
        """
        context = {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        }
        if additional_context:
            context.update(additional_context)
        return RunContextWrapper(context=context)
    
    async def get_connection_pool(self) -> asyncpg.Pool:
        """
        Get a connection pool for database operations.
        
        Returns:
            Database connection pool
        """
        if not hasattr(self, '_pool'):
            self._pool = await asyncpg.create_pool(dsn=DB_DSN)
        return self._pool


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
    ctx = orchestrator._create_mock_context()
    return await orchestrator.evolve_world_with_event(ctx, event_description, **kwargs)


# ===== MODULE INITIALIZATION =====

def setup_lore_orchestrator():
    """Setup function for module initialization."""
    logging.basicConfig(level=logging.INFO)
    logger.info("FULLY INTEGRATED Lore Orchestrator loaded with all modules: education, geopolitical, local lore, politics, religion, and world lore management")


# Run setup on module import
setup_lore_orchestrator()
