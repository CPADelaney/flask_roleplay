# nyx/core/brain/base.py

import logging
import asyncio
import datetime
import random
import os
import math
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import json

from agents import (
    Agent, Runner, trace, function_tool, handoff, RunContextWrapper,
    ModelSettings
)
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError

from nyx.core.internal_thoughts import InternalThought

from nyx.core.brain.nyx_distributed_checkpoint import DistributedCheckpointMixin
from nyx.core.brain.nyx_event_log import EventLogMixin

# Import new components
from nyx.core.novelty_engine import NoveltyEngine
from nyx.core.recognition_memory import RecognitionMemorySystem
from nyx.core.creative_memory_integration import CreativeMemoryIntegration

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class NyxBrain(DistributedCheckpointMixin, EventLogMixin):
    """
    Central integration point for all Nyx systems.
    Uses composition to delegate to specialized components while managing their coordination.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the NyxBrain instance for a specific user and conversation.
        
        Args:
            user_id: Unique identifier for the user
            conversation_id: Unique identifier for the conversation
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Core components - initialized in initialize()
        self.emotional_core = None
        self.memory_core = None
        self.reflection_engine = None
        self.experience_interface = None
        self.internal_feedback = None
        self.dynamic_adaptation = None
        self.meta_core = None
        self.knowledge_core = None
        self.memory_orchestrator = None
        self.reasoning_core = None
        self.identity_evolution = None
        self.experience_consolidation = None
        self.cross_user_manager = None
        self.reflexive_system = None
        self.hormone_system = None
        self.attentional_controller = None
        self.multimodal_integrator = None
        self.reward_system = None
        self.temporal_perception = None
        self.procedural_memory = None
        self.agent_enhanced_memory = None
        self.digital_somatosensory_system = None
        self.needs_system = None
        self.goal_manager = None
        self.streaming_core = None

        # Spatial system components
        self.spatial_mapper = None
        self.spatial_memory = None
        self.map_visualization = None
        self.navigator_agent = None
        
        # Sync system components
        self.sync_daemon = None
        self.strategy_controller = None
        self.noise_filter = None
        
        # Tools components
        self.agent_evaluator = None
        self.parallel_executor = None
        
        self.dev_log_storage = None
        
        # Component managers
        self.processing_manager = None
        self.self_config_manager = None
        
        # Agents
        self.brain_agent = None
        self.general_dominance_ideation_agent = None
        self.hard_dominance_ideation_agent = None

        self.protocol_enforcement = None
        self.body_service_system = None
        self.psychological_dominance = None
        self.femdom_coordinator = None
        self.femdom_integration_manager = None   
        self.orgasm_control_system = None
        self.dominance_persona_manager = None
        self.sadistic_response_system = None        
        
        # State tracking
        self.initialized = False
        self.last_interaction = datetime.datetime.now()
        self.interaction_count = 0
        self.cognitive_cycles_executed = 0
        self.trace_group_id = f"nyx-brain-{user_id}-{conversation_id}"

        self.conditioning_system = None
        
        # Configuration defaults
        self.cross_user_enabled = True
        self.cross_user_sharing_threshold = 0.7
        self.memory_to_emotion_influence = 0.3
        self.emotion_to_memory_influence = 0.4
        self.experience_to_identity_influence = 0.2
        self.consolidation_interval = 24  # Hours between consolidations
        self.identity_reflection_interval = 10  # Interactions between identity reflections
        
        # Performance tracking
        self.performance_metrics = {
            "memory_operations": 0,
            "emotion_updates": 0,
            "reflections_generated": 0,
            "experiences_shared": 0,
            "cross_user_experiences_shared": 0,
            "experience_consolidations": 0,
            "goals_completed": 0,
            "goals_failed": 0,
            "steps_executed": 0,
            "response_times": []
        }
        
        # Thinking configuration
        self.thinking_config = {
            "thinking_enabled": True,
            "last_thinking_interaction": 0,
            "thinking_stats": {
                "total_thinking_used": 0,
                "basic_thinking_used": 0,
                "moderate_thinking_used": 0,
                "deep_thinking_used": 0,
                "thinking_time_avg": 0.0
            }
        }

        self.context_system = None
        self.mode_manager = None

        self.issue_tracking_system = None
        self.passive_observation_system = None
        self.proactive_communication_engine = None        

        self.thoughts_manager = None    

        self.event_bus = None
        self.system_context = None
        self.integrated_tracer = None
        self.integration_manager = None

        self.checkpoint_planner = None

        # Initialize new components
        self.novelty_engine = None
        
        self.recognition_memory = None
        
        self.creative_memory = None
        
        # Timestamp tracking
        self.last_consolidation = datetime.datetime.now() - datetime.timedelta(hours=25)
        self.last_needs_goal_update = datetime.datetime.now()
        
        # Error tracking
        self.error_registry = {
            "unhandled_errors": [],
            "handled_errors": [],
            "error_counts": {},
            "error_recovery_strategies": {},
            "error_recovery_stats": {}
        }
        
        # Support systems
        self.module_optimizer = None
        self.system_health_checker = None
        
        # Agent capabilities
        self.agent_capabilities_initialized = False
        
        logger.info(f"NyxBrain initialized for user {self.user_id}, conversation {self.conversation_id}")
    
    async def initialize(self):
        """
        Initialize all subsystems in the correct dependency order.
        Handles circular dependencies by setting references after initialization.
        """
        if self.initialized:
            return
        
        logger.info(f"Initializing NyxBrain for user {self.user_id}, conversation {self.conversation_id}")
        
        # Dynamic imports to avoid circular dependencies
        # Import components only when needed for initialization
        try:
            # Import core systems
            from nyx.core.brain.module_optimizer import ModuleOptimizer
            from nyx.core.brain.system_health_checker import SystemHealthChecker
            from nyx.core.emotions.emotional_core import EmotionalCore
            from nyx.core.memory_core import MemoryCoreAgents
            from nyx.core.reflection_engine import ReflectionEngine
            from nyx.core.experience_interface import ExperienceInterface
            from nyx.core.dynamic_adaptation_system import DynamicAdaptationSystem
            from nyx.core.internal_feedback_system import InternalFeedbackSystem
            from nyx.core.meta_core import MetaCore
            from nyx.core.internal_thoughts import InternalThought
            from nyx.core.knowledge_core import KnowledgeCoreAgents
            from nyx.core.memory_orchestrator import MemoryOrchestrator
            from nyx.core.identity_evolution import IdentityEvolutionSystem
            from nyx.core.experience_consolidation import ExperienceConsolidationSystem
            from nyx.core.cross_user_experience import CrossUserExperienceManager
            from nyx.core.emotions.hormone_system import HormoneSystem
            from nyx.core.attentional_controller import AttentionalController
            from nyx.core.multimodal_integrator import MultimodalIntegrator, MultimodalIntegratorContext
            from nyx.core.reward_system import RewardSignalProcessor
            from nyx.core.temporal_perception import TemporalPerceptionSystem
            from nyx.core.procedural_memory.agent import AgentEnhancedMemoryManager
            from nyx.core.digital_somatosensory_system import DigitalSomatosensorySystem
            from nyx.core.needs_system import NeedsSystem
            from nyx.core.conditioning_system import ConditioningSystem
            from nyx.core.goal_system import GoalManager
            from nyx.core.reasoning_agents import integrated_reasoning_agent, triage_agent as reasoning_triage_agent
            from nyx.apitools.thinking_tools import should_use_extended_thinking, should_use_extended_thinking, generate_reasoned_response 
            from nyx.core.dominance import create_dominance_ideation_agent, create_hard_dominance_ideation_agent
            from nyx.core.mood_manager import MoodManager
            from nyx.core.theory_of_mind import TheoryOfMind, SubspaceDetectionSystem, create_theory_of_mind_agent, create_subspace_detection_agent
            from nyx.core.imagination_simulator import ImaginationSimulator
            from nyx.core.internal_thoughts import InternalThoughtsManager, pre_process_input, pre_process_output
    
            from nyx.core.femdom.body_service_system import BodyServiceSystem
            from nyx.core.femdom.protocol_enforcement import ProtocolEnforcement
            from nyx.core.femdom.psychological_dominance import PsychologicalDominance
            from nyx.core.femdom.femdom_coordinator import FemdomCoordinator
            from nyx.core.femdom.femdom_integration_manager import FemdomIntegrationManager    
            from nyx.core.femdom.orgasm_control import OrgasmControlSystem
            from nyx.core.femdom.persona_manager import DominancePersonaManager
            from nyx.core.femdom.sadistic_responses import SadisticResponseSystem          

            from nyx.core.issue_tracking_system import IssueTrackingSystem
            from nyx.core.passive_observation import PassiveObservationSystem
            from nyx.core.proactive_communication import ProactiveCommunicationEngine            

            from nyx.core.spatial.spatial_mapper import SpatialMapper
            from nyx.core.spatial.spatial_memory import SpatialMemoryIntegration
            from nyx.core.spatial.map_visualization import MapVisualization
            from nyx.core.spatial.navigator_agent import SpatialNavigatorAgent
            
            # Sync system imports
            from nyx.core.sync.nyx_sync_daemon import NyxSyncDaemon
            from nyx.core.sync.strategy_controller import get_active_strategies, mark_strategy_for_review
            from nyx.core.sync.strategy_register import register_decision
            
            # Tools imports
            from nyx.core.tools.evaluator import AgentEvaluator
            from nyx.core.tools.parallel import ParallelToolExecutor          
            
            from nyx.core.context_awareness import ContextAwarenessSystem
            from nyx.core.interaction_mode_manager import InteractionModeManager
            from nyx.core.mode_integration import ModeIntegrationManager

            from nyx.core.procedural_memory.manager import ProceduralMemoryManager
            from nyx.core.procedural_memory.agent import ProceduralMemoryAgents, AgentEnhancedMemoryManager
    
            from nyx.core.memory_core import BrainMemoryCore  # or from .memory_core import BrainMemoryCore
            from nyx.core.brain.checkpointing_agent import CheckpointingPlannerAgent        
    
            from nyx.creative.agentic_system import AgenticCreativitySystem, integrate_with_existing_system
                    
            # Import conditioning systems
            from nyx.core.conditioning_config import ConditioningConfiguration
            from nyx.core.conditioning_system import ConditioningSystem
            from nyx.core.conditioning_maintenance import ConditioningMaintenanceSystem
            from nyx.core.input_processor import BlendedInputProcessor  # Import input processor
            
    
            from dev_log.storage import get_dev_log_storage
            self.dev_log_storage = get_dev_log_storage()
            await self.dev_log_storage.initialize()
            
            # Import integration components
            from nyx.core.integration.event_bus import get_event_bus
            from nyx.core.integration.system_context import get_system_context
            from nyx.core.integration.integrated_tracer import get_tracer
            from nyx.core.integration.integration_manager import create_integration_manager
            
            # Try to import relationship manager - check different possible locations
            try:
                # Option 1: Direct import
                from nyx.core.relationship_manager import RelationshipManager
                has_relationship_manager = True
            except ImportError:
                try:
                    # Option 2: Part of another module
                    from nyx.core.social.relationship_manager import RelationshipManager
                    has_relationship_manager = True
                except ImportError:
                    # Fallback with warning
                    logger.warning("RelationshipManager module not found. Relationship features will be limited.")
                    RelationshipManager = None
                    has_relationship_manager = False
                    
            
            # 1. Initialize integration foundation components first
            # These should be initialized early as other components may depend on them
            self.event_bus = get_event_bus()
            self.system_context = get_system_context()
            self.integrated_tracer = get_tracer()
            
            # Set conversation ID in system context after it's initialized
            self.system_context.conversation_id = self.conversation_id
            
            # 2. Initialize support systems
            self.module_optimizer = ModuleOptimizer(self)
            self.system_health_checker = SystemHealthChecker(self)
    
            self.conditioning_config = ConditioningConfiguration()
            logger.debug("Conditioning configuration initialized")
            
            self.conditioning_system = ConditioningSystem(
                reward_system=self.reward_system,
                emotional_core=self.emotional_core,
                memory_core=self.memory_core,
                somatosensory_system=self.digital_somatosensory_system
            )
            logger.debug("Conditioning system initialized")
            
            self.conditioning_maintenance = ConditioningMaintenanceSystem(
                conditioning_system=self.conditioning_system,
                reward_system=self.reward_system
            )
            await self.conditioning_maintenance.start_maintenance_scheduler()
            logger.debug("Conditioning maintenance system initialized")
            
            # Initialize input processor after other conditioning systems
            self.conditioned_input_processor = BlendedInputProcessor(
                conditioning_system=self.conditioning_system,
                emotional_core=self.emotional_core,
                somatosensory_system=self.digital_somatosensory_system
            )
            logger.debug("Conditioned input processor initialized")
            
            # 3. Initialize foundational systems without dependencies
            # Initialize hormone state
            self.hormone_system = HormoneSystem()
            logger.debug("Hormone system initialized")
            
            # Initialize emotional core without passing hormone_system
            self.emotional_core = EmotionalCore()
            logger.debug("Emotional core initialized")
            
            # Connect them after initialization
            self.emotional_core.set_hormone_system(self.hormone_system)
            logger.debug("Emotional core and hormone system connected")

            self.checkpoint_planner = CheckpointingPlannerAgent()           
            
            self.memory_core = MemoryCoreAgents(self.user_id, self.conversation_id)
            await self.memory_core.initialize()
            logger.debug("Memory core initialized")
            
            self.memory_orchestrator = MemoryOrchestrator(self.user_id, self.conversation_id)
            await self.memory_orchestrator.initialize()
            logger.debug("Memory orchestrator initialized")
            
            self.reflection_engine = ReflectionEngine()
            logger.debug("Reflection engine initialized")
            
            # 3. Initialize primary systems with simple dependencies
            self.experience_interface = ExperienceInterface(self.memory_core, self.emotional_core)
            logger.debug("Experience interface initialized")
            
            self.identity_evolution = IdentityEvolutionSystem(hormone_system=self.hormone_system)
            logger.debug("Identity evolution initialized")
            
            self.experience_consolidation = ExperienceConsolidationSystem(
                memory_core=self.memory_core,
                experience_interface=self.experience_interface
            )
            logger.debug("Experience consolidation initialized")

            self.context_system = ContextAwarenessSystem(emotional_core=self.emotional_core)
            
            self.cross_user_manager = CrossUserExperienceManager(
                memory_core=self.memory_core,
                experience_interface=self.experience_interface
            )
            logger.debug("Cross-user manager initialized")
            
            # 4. Initialize core processing systems
            self.internal_feedback = InternalFeedbackSystem()
            logger.debug("Internal feedback initialized")
            
            self.attentional_controller = AttentionalController(
                emotional_core=self.emotional_core
            )
            logger.debug("Attentional controller initialized")
            
            # Use integrated reasoning agent as reasoning core
            self.reasoning_core = integrated_reasoning_agent
            self.reasoning_triage_agent = reasoning_triage_agent
            logger.debug("Reasoning agents initialized")
            
            # 5. Initialize systems with circular dependencies
            # First create with partial dependencies
            self.reward_system = RewardSignalProcessor(
                emotional_core=self.emotional_core,
                identity_evolution=self.identity_evolution,
                somatosensory_system=None  # Will set this later
            )
            logger.debug("Reward system initialized")
    
            # Now create dependent systems
            self.digital_somatosensory_system = DigitalSomatosensorySystem(
                memory_core=self.memory_core,
                emotional_core=self.emotional_core,
                reward_system=self.reward_system
            )
            await self.digital_somatosensory_system.initialize()
            logger.debug("Digital somatosensory system initialized")
    
            # Get the personality profile first
            personality_profile = await self.conditioning_config.get_personality_profile()

            # Then initialize with the profile's dict
            await ConditioningSystem.initialize_baseline_personality(
                conditioning_system=self.conditioning_system,
                personality_profile=personality_profile # <--- CORRECTED: Pass the dictionary directly
            )
            logger.debug("Baseline personality conditioning initialized")      
    
            # Initialize relationship manager if available
            self.relationship_manager = None
            if has_relationship_manager and RelationshipManager:
                try:
                    self.relationship_manager = RelationshipManager(
                        memory_orchestrator=self.memory_orchestrator,
                        emotional_core=self.emotional_core
                    )
                    logger.debug("Relationship manager initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize relationship manager: {e}")
            
            # Set circular references
            self.reward_system.somatosensory_system = self.digital_somatosensory_system
            self.identity_evolution.somatosensory_system = self.digital_somatosensory_system
            logger.debug("Circular dependencies resolved")

            self.protocol_enforcement = ProtocolEnforcement(
                reward_system=self.reward_system,
                memory_core=self.memory_core,
                relationship_manager=self.relationship_manager
            )
            logger.debug("Protocol enforcement system initialized")
            
            self.body_service_system = BodyServiceSystem(
                reward_system=self.reward_system,
                memory_core=self.memory_core,
                relationship_manager=self.relationship_manager
            )
            logger.debug("Body service system initialized")

            # Initialize theory of mind
            self.theory_of_mind = TheoryOfMind(
                relationship_manager=self.relationship_manager,
                multimodal_integrator=self.multimodal_integrator,
                memory_core=self.memory_core
            )
            logger.debug("Theory of mind initialized")
            
            self.psychological_dominance = PsychologicalDominance(
                theory_of_mind=self.theory_of_mind,
                reward_system=self.reward_system,
                relationship_manager=self.relationship_manager,
                memory_core=self.memory_core
            )
            logger.debug("Psychological dominance system initialized")
            
            # Initialize femdom coordinator (after all individual systems)
            self.femdom_coordinator = FemdomCoordinator(self)
            await self.femdom_coordinator.initialize()
            logger.debug("Femdom coordinator initialized")
            
            # Initialize femdom integration manager
            self.femdom_integration_manager = FemdomIntegrationManager(
                self,
                components={
                    "protocol_enforcement": self.protocol_enforcement,
                    "body_service": self.body_service_system,
                    "psychological_dominance": self.psychological_dominance,
                    "reward_system": self.reward_system,
                    "memory_core": self.memory_core,
                    "relationship_manager": self.relationship_manager,
                    "theory_of_mind": self.theory_of_mind,
                    # Add new components
                    "orgasm_control_system": self.orgasm_control_system,
                    "dominance_persona_manager": self.dominance_persona_manager,
                    "sadistic_response_system": self.sadistic_response_system
                }
            )
            await self.femdom_integration_manager.initialize()
            logger.debug("Femdom integration manager initialized")           

             # Initialize orgasm control system
            self.orgasm_control_system = OrgasmControlSystem(
                reward_system=self.reward_system,
                memory_core=self.memory_core,
                relationship_manager=self.relationship_manager,
                somatosensory_system=self.digital_somatosensory_system
            )
            logger.debug("Orgasm control system initialized")
            
            # Initialize dominance persona manager
            self.dominance_persona_manager = DominancePersonaManager(
                relationship_manager=self.relationship_manager,
                reward_system=self.reward_system,
                memory_core=self.memory_core,
                emotional_core=self.emotional_core
            )
            logger.debug("Dominance persona manager initialized")
            
            # Initialize sadistic response system
            self.sadistic_response_system = SadisticResponseSystem(
                theory_of_mind=self.theory_of_mind,
                protocol_enforcement=self.protocol_enforcement,
                reward_system=self.reward_system,
                relationship_manager=self.relationship_manager,
                memory_core=self.memory_core
            )
            logger.debug("Sadistic response system initialized")           
            
            # 6. Initialize perception and integration systems
            self.multimodal_integrator = MultimodalIntegrator(
                reasoning_core=self.reasoning_core,
                attentional_controller=self.attentional_controller
            )
            logger.debug("Multimodal integrator initialized")
            
            self.temporal_perception = TemporalPerceptionSystem(self.user_id, self.conversation_id)
            await self.temporal_perception.initialize(self, None)
            logger.debug("Temporal perception initialized")
            
            # 7. Initialize GoalManager & NeedsSystem
            # GoalManager needs the brain reference to call action methods
            self.goal_manager = GoalManager(brain_reference=self)
            logger.debug("Goal manager initialized")
            
            # NeedsSystem needs the goal manager to trigger goal creation
            self.needs_system = NeedsSystem(goal_manager=self.goal_manager)
            logger.debug("Needs system initialized")
            
            # 8. Initialize memory augmentation systems
            self.procedural_memory = ProceduralMemoryManager()
            self.agent_enhanced_memory = AgentEnhancedMemoryManager(self.procedural_memory)
            logger.debug("Procedural memory initialized")
            
            # 9. Initialize cognitive tools
            from nyx.apitools.thinking_tools import (
                should_use_extended_thinking,
                think_before_responding,
                generate_reasoned_response
            )
            
            self.thinking_tools = {
                "should_use_extended_thinking": should_use_extended_thinking,
                "think_before_responding": think_before_responding,
                "generate_reasoned_response": generate_reasoned_response
            }
            logger.debug("Thinking tools initialized")
            
            # 10. Initialize additional cognitive systems
            # Initialize mood manager
            self.mood_manager = MoodManager(
                emotional_core=self.emotional_core,
                hormone_system=self.hormone_system,
                needs_system=self.needs_system,
                goal_manager=self.goal_manager
            )
            logger.debug("Mood manager initialized")
            
            # Initialize imagination simulator
            self.imagination_simulator = ImaginationSimulator(
                reasoning_core=self.reasoning_core,
                knowledge_core=None,  # Will set after knowledge core is initialized
                emotional_core=self.emotional_core,
                identity_evolution=self.identity_evolution
            )
            logger.debug("Imagination simulator initialized")
            
            # 11. Initialize specialized managers
            # Register feature extractors and integration strategies for multimodal
            await self._register_processing_modules()
            
            # Initialize component managers
            self.processing_manager = ProcessingManager(self)
            await self.processing_manager.initialize()
            
            self.self_config_manager = SelfConfigManager(self)
            
            # 12. Initialize agent-based systems
            self.dynamic_adaptation = DynamicAdaptationSystem()
            
            self.knowledge_core = KnowledgeCoreAgents()
            await self.knowledge_core.initialize()
            
            # Update imagination simulator with knowledge core
            self.imagination_simulator.knowledge_core = self.knowledge_core
            
            # Initialize agent systems for dominance ideation if functions available
            if 'create_dominance_ideation_agent' in locals():
                self.general_dominance_ideation_agent = create_dominance_ideation_agent()
                self.hard_dominance_ideation_agent = create_hard_dominance_ideation_agent()
                logger.debug("Dominance ideation agents initialized")
            
            # 13. Initialize meta cognition (needs references to other systems)
            self.meta_core = MetaCore()
            await self.meta_core.initialize({
                "memory": self.memory_core,
                "emotion": self.emotional_core,
                "reasoning": self.reasoning_core,
                "reflection": self.reflection_engine,
                "adaptation": self.dynamic_adaptation,
                "feedback": self.internal_feedback,
                "identity": self.identity_evolution,
                "experience": self.experience_interface,
                "hormone": self.hormone_system,
                "time": self.temporal_perception,
                "procedural": self.agent_enhanced_memory,
                "needs": self.needs_system,
                "goals": self.goal_manager,
                "mood": self.mood_manager,
                "theory_of_mind": self.theory_of_mind,
                "imagination": self.imagination_simulator
            })
            logger.debug("Meta core initialized")
            
            # 14. Initialize optional agent capabilities
            if os.environ.get("ENABLE_AGENT", "true").lower() == "true":
                await self.initialize_agent_capabilities()
            
            # 15. Initialize streaming if needed
            if os.environ.get("ENABLE_STREAMING", "false").lower() == "true":
                await self.initialize_streaming()
            
            # 16. Initialize reflexive system if module exists
            try:
                from nyx.core.reflexive_system import initialize_reflexive_system
                self.reflexive_system = await initialize_reflexive_system(self)
                logger.debug("Reflexive system initialized")
            except ImportError:
                logger.info("Reflexive system module not found, skipping initialization")

            self.mode_manager = InteractionModeManager(
                context_system=self.context_system,
                emotional_core=self.emotional_core,
                reward_system=self.reward_system,
                goal_manager=self.goal_manager
            )
            self.mode_integration = ModeIntegrationManager(nyx_brain=self)     
            
            # Initialize Issue Tracking System
            self.issue_tracking_system = IssueTrackingSystem(
                db_path=f"issues_db_{user_id}_{conversation_id}.json"
            )
            logger.debug("Issue tracking system initialized")
            
            # Now initialize thoughts_manager with the proper references
            self.thoughts_manager = InternalThoughtsManager(
                passive_observation_system=self.passive_observation_system,
                reflection_engine=self.reflection_engine,
                imagination_simulator=self.imagination_simulator,
                theory_of_mind=self.theory_of_mind,
                relationship_reflection=self.relationship_manager,
                proactive_communication=self.proactive_communication_engine,
                emotional_core=self.emotional_core,
                memory_core=self.memory_core
            )
            logger.debug("Internal thoughts manager initialized")             

            self.spatial_mapper = SpatialMapper(memory_integration=self.memory_core)
            await self.spatial_mapper.initialize() if hasattr(self.spatial_mapper, "initialize") else None
            logger.debug("Spatial mapper initialized")
            
            self.spatial_memory = SpatialMemoryIntegration(
                spatial_mapper=self.spatial_mapper,
                memory_core=self.memory_core
            )
            logger.debug("Spatial memory integration initialized")
            
            self.map_visualization = MapVisualization()
            logger.debug("Map visualization initialized")
            
            self.navigator_agent = SpatialNavigatorAgent(spatial_mapper=self.spatial_mapper)
            logger.debug("Spatial navigator initialized")      

            # Initialize new components
            self.novelty_engine = NoveltyEngine(
                imagination_simulator=self.imagination_simulator,
                memory_core=self.memory_core
            )
            
            self.recognition_memory = RecognitionMemorySystem(
                memory_core=self.memory_core,
                context_awareness=self.context_awareness
            )
            
            self.creative_memory = CreativeMemoryIntegration(
                novelty_engine=self.novelty_engine,
                recognition_memory=self.recognition_memory,
                memory_core=self.memory_core
            )

            # Initialize new components
            await self.novelty_engine.initialize()
            await self.recognition_memory.initialize()
            await self.creative_memory.initialize()
    
            # 17. Create main orchestration agent
            self.brain_agent = self._create_brain_agent()
    
            # Initialize procedural memory if needed
            if not hasattr(self, "procedural_memory") or not self.procedural_memory:
                from nyx.core.procedural_memory import ProceduralMemoryManager
                self.procedural_memory = ProceduralMemoryManager()
                logger.debug("Procedural memory initialized")
            
            # Create agent-enhanced memory if not already present
            if not hasattr(self, "agent_enhanced_memory") or not self.agent_enhanced_memory:
                from nyx.core.procedural_memory.agent import AgentEnhancedMemoryManager
                self.agent_enhanced_memory = AgentEnhancedMemoryManager(self.procedural_memory)
                logger.debug("Agent enhanced memory initialized")
    
            nyx_brain = BrainMemoryCore()
            
            # (Optional but recommended in async systems)
            await nyx_brain.initialize()        
            
            # Find or create the agentic action generator
            if not hasattr(self, "agentic_action_generator") or not self.agentic_action_generator:
                # Create a new action generator with references to brain systems
                from nyx.core.agentic_action_generator import EnhancedAgenticActionGenerator as AgenticActionGenerator
                self.agentic_action_generator = AgenticActionGenerator(
                    emotional_core=self.emotional_core,
                    hormone_system=self.hormone_system,
                    experience_interface=self.experience_interface,
                    imagination_simulator=self.imagination_simulator,
                    meta_core=self.meta_core,
                    memory_core=self.memory_core,
                    goal_system=self.goal_manager,
                    identity_evolution=self.identity_evolution,
                    knowledge_core=self.knowledge_core,
                    input_processor=getattr(self, "conditioned_input_processor", None),
                    internal_feedback=self.internal_feedback,
                    # Add the following references:
                    attentional_controller=self.attentional_controller,  # Add attentional controller
                    reasoning_core=self.reasoning_core,  # Add reasoning core
                    reflection_engine=self.reflection_engine,  # Add reflection engine
                    # New system integrations
                    mood_manager=self.mood_manager,
                    needs_system=self.needs_system,
                    mode_integration=self.mode_integration,
                    multimodal_integrator=self.multimodal_integrator
                )
                logger.debug("Agentic action generator initialized")

            self.passive_observation_system = PassiveObservationSystem(
                emotional_core=self.emotional_core,
                memory_core=self.memory_core,
                relationship_manager=self.relationship_manager,
                temporal_perception=self.temporal_perception,
                multimodal_integrator=self.multimodal_integrator,
                mood_manager=self.mood_manager,
                needs_system=self.needs_system,
                identity_evolution=self.identity_evolution,
                attention_controller=self.attentional_controller,
                action_generator=self.agentic_action_generator  # Now available
            )
            await self.passive_observation_system.start()
            logger.debug("Passive observation system initialized")            

            # Initialize Proactive Communication Engine
            self.proactive_communication_engine = ProactiveCommunicationEngine(
                emotional_core=self.emotional_core,
                memory_core=self.memory_core,
                relationship_manager=self.relationship_manager,
                temporal_perception=self.temporal_perception,
                reasoning_core=self.reasoning_core,
                reflection_engine=self.reflection_engine,
                mood_manager=self.mood_manager,
                needs_system=self.needs_system,
                identity_evolution=self.identity_evolution,
                action_generator=self.agentic_action_generator
            )
            await self.proactive_communication_engine.start()
            logger.debug("Proactive communication engine initialized")
            
            # Integrate procedural memory with action generator
            await self.integrate_procedural_memory_with_actions()
    
            self.creative_system = await integrate_with_existing_system(self)
    
            self._start_creative_review_task()
            
            logger.info(f"Creative system initialized at: {self.creative_system.content_system.base_directory}")
    
            # Register creative actions with action generator
            if hasattr(self, "agentic_action_generator"):
                self._register_creative_actions()    
            
            self.integration_manager = create_integration_manager(self)
            await self.integration_manager.initialize()
            logger.debug("Integration manager initialized")

            # Initialize sync system components
            self.sync_daemon = NyxSyncDaemon()
            logger.debug("Sync daemon initialized")          

            # Initialize tools
            self.agent_evaluator = AgentEvaluator()
            logger.debug("Agent evaluator initialized")

            # After both systems are initialized
            if self.agent_enhanced_memory and self.agentic_action_generator:
                await self.integrate_procedural_memory_with_actions()
                logger.debug("Procedural memory integrated with action generator")            
            
            self.parallel_executor = ParallelToolExecutor(max_concurrent=5)
            logger.debug("Parallel tool executor initialized")            
            
            self.initialized = True
            logger.info(f"NyxBrain fully initialized for user {self.user_id}, conversation {self.conversation_id}")
            
        except Exception as e:
            logger.error(f"Error initializing NyxBrain: {str(e)}", exc_info=True)
            raise

    @classmethod
    async def restore_from_checkpoint(self, checkpoint_data: dict):
        """
        Apply agentic checkpoint data back onto this NyxBrain instance/subsystems.
        Safely checks for all supported major fields.
        """
        if not checkpoint_data:
            logger.info("No checkpoint data found for restore — booting cold.")
            return False

        # --- Core affective state ---
        if self.emotional_core and "emotional_state" in checkpoint_data:
            try:
                await self.emotional_core.set_emotional_state(checkpoint_data["emotional_state"])
            except Exception as e:
                logger.warning(f"Restore: emotional_state failed: {e}")

        if self.spatial_mapper and "spatial_maps" in checkpoint_data:
            try:
                for map_id, map_data in checkpoint_data["spatial_maps"].items():
                    if map_id in self.spatial_mapper.maps:
                        # Update existing map
                        self.spatial_mapper.maps[map_id].accuracy = map_data.get("accuracy", 0.5)
                        self.spatial_mapper.maps[map_id].completeness = map_data.get("completeness", 0.0)
                        self.spatial_mapper.maps[map_id].last_updated = map_data.get("last_updated", datetime.datetime.now().isoformat())
                    else:
                        # Load map if missing
                        if "name" in map_data and "description" in map_data:
                            await self.spatial_mapper.create_cognitive_map(
                                name=map_data["name"],
                                description=map_data["description"]
                            )
            except Exception as e:
                logger.warning(f"Restore: spatial_maps failed: {e}")        

        if self.hormone_system and "hormones" in checkpoint_data:
            try:
                self.hormone_system.set_state(checkpoint_data["hormones"])
            except Exception as e:
                logger.warning(f"Restore: hormone state failed: {e}")

        if self.mood_manager and "mood_state" in checkpoint_data:
            try:
                if hasattr(self.mood_manager, "set_current_mood"):
                    await self.mood_manager.set_current_mood(checkpoint_data["mood_state"])
            except Exception as e:
                logger.warning(f"Restore: mood_state failed: {e}")

        # --- Needs ("drive" state) ---
        if self.needs_system and "needs" in checkpoint_data:
            try:
                if hasattr(self.needs_system, "set_needs_state"):
                    await self.needs_system.set_needs_state(checkpoint_data["needs"])
            except Exception as e:
                logger.warning(f"Restore: needs failed: {e}")

        # --- Goals ---
        if self.goal_manager and "goals" in checkpoint_data:
            try:
                if hasattr(self.goal_manager, "restore_goals"):
                    await self.goal_manager.restore_goals(checkpoint_data["goals"])
                elif hasattr(self.goal_manager, "set_goals"):
                    await self.goal_manager.set_goals(checkpoint_data["goals"])
            except Exception as e:
                logger.warning(f"Restore: goals failed: {e}")

        # --- Memory [recent or special memories, diary] ---
        if self.memory_core and "recent_memories" in checkpoint_data:
            try:
                # Check which method is available and use appropriate one
                if hasattr(self.memory_core, 'load_recent_memories'):
                    await self.memory_core.load_recent_memories(checkpoint_data["recent_memories"])
                elif hasattr(self.memory_core, 'import_memories'):
                    await self.memory_core.import_memories(checkpoint_data["recent_memories"])
                elif hasattr(self.memory_core, 'add_memories_batch'):
                    # Add fallback for newer API
                    await self.memory_core.add_memories_batch(checkpoint_data["recent_memories"])
                # else: skip—they should be re-encoded by the brain loop
            except Exception as e:
                logger.warning(f"Restore: recent_memories failed: {e}")

        # --- Identity and traits (if present) ---
        if getattr(self, "identity_evolution", None) and "identity" in checkpoint_data:
            try:
                if hasattr(self.identity_evolution, "restore_identity"):
                    await self.identity_evolution.restore_identity(checkpoint_data["identity"])
                elif hasattr(self.identity_evolution, "set_identity_state"):
                    await self.identity_evolution.set_identity_state(checkpoint_data["identity"])
            except Exception as e:
                logger.warning(f"Restore: identity failed: {e}")

        # --- Mode integration (current interaction style) ---
        if getattr(self, "mode_integration", None) and "mode" in checkpoint_data:
            try:
                if hasattr(self.mode_integration, "set_mode_state"):
                    await self.mode_integration.set_mode_state(checkpoint_data["mode"])
                elif hasattr(self.mode_integration, "load_mode"):
                    await self.mode_integration.load_mode(checkpoint_data["mode"])
            except Exception as e:
                logger.warning(f"Restore: mode integration failed: {e}")

        # --- Conceptual/causal/model state ---
        if getattr(self, "reasoning_core", None) and "causal_state" in checkpoint_data:
            try:
                if hasattr(self.reasoning_core, "restore_state"):
                    await self.reasoning_core.restore_state(checkpoint_data["causal_state"])
            except Exception as e:
                logger.warning(f"Restore: causal_state failed: {e}")

        # --- Theory of Mind / user model ---
        if getattr(self, "theory_of_mind", None) and "user_model" in checkpoint_data:
            try:
                if hasattr(self.theory_of_mind, "restore_state"):
                    await self.theory_of_mind.restore_state(checkpoint_data["user_model"])
            except Exception as e:
                logger.warning(f"Restore: user_model failed: {e}")

        # --- Temporal context ---
        if getattr(self, "temporal_perception", None) and "temporal_context" in checkpoint_data:
            try:
                if hasattr(self.temporal_perception, "restore_context"):
                    await self.temporal_perception.restore_context(checkpoint_data["temporal_context"])
                elif hasattr(self.temporal_perception, "set_context"):
                    await self.temporal_perception.set_context(checkpoint_data["temporal_context"])
            except Exception as e:
                logger.warning(f"Restore: temporal_context failed: {e}")

        # --- Sensory context ---
        if getattr(self, "multimodal_integrator", None) and "sensory_context" in checkpoint_data:
            try:
                if hasattr(self.multimodal_integrator, "load_context"):
                    await self.multimodal_integrator.load_context(checkpoint_data["sensory_context"])
            except Exception as e:
                logger.warning(f"Restore: sensory_context failed: {e}")

        # --- Reflection engine (if present, e.g. insights/diary) ---
        if getattr(self, "reflection_engine", None) and "reflection_insights" in checkpoint_data:
            try:
                if hasattr(self.reflection_engine, "import_insights"):
                    await self.reflection_engine.import_insights(checkpoint_data["reflection_insights"])
            except Exception as e:
                logger.warning(f"Restore: reflection_insights failed: {e}")

        # --- Action values (Q-tables, RL stats) ---
        if hasattr(self, "action_values") and "action_values" in checkpoint_data:
            try:
                self.action_values = checkpoint_data["action_values"]
            except Exception as e:
                logger.warning(f"Restore: action_values failed: {e}")

        # --- Action history/habits/other agentic learning memory ---
        if hasattr(self, "action_history") and "action_history" in checkpoint_data:
            try:
                self.action_history = checkpoint_data["action_history"]
            except Exception as e:
                logger.warning(f"Restore: action_history failed: {e}")
        if hasattr(self, "habits") and "habits" in checkpoint_data:
            try:
                self.habits = checkpoint_data["habits"]
            except Exception as e:
                logger.warning(f"Restore: habits failed: {e}")

        # --- Goal strategies and planning ---
        if hasattr(self, "action_strategies") and "action_strategies" in checkpoint_data:
            try:
                self.action_strategies = checkpoint_data["action_strategies"]
            except Exception as e:
                logger.warning(f"Restore: action_strategies failed: {e}")

        # --- Custom trackers, mode adaptation, bottlenecks ---
        if hasattr(self, "mode_adaptation_strength") and "mode_adaptation_strength" in checkpoint_data:
            self.mode_adaptation_strength = checkpoint_data["mode_adaptation_strength"]
        if hasattr(self, "detected_bottlenecks") and "detected_bottlenecks" in checkpoint_data:
            self.detected_bottlenecks = checkpoint_data["detected_bottlenecks"]

        logger.info("NyxBrain state fully restored from checkpoint.")
        return True

    async def publish_event(self, event: Any) -> None:
        """
        Publish an event to the event bus.
        
        Args:
            event: Event to publish
        """
        if self.event_bus:
            await self.event_bus.publish(event)
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """
        Get the status of the integration system.
        
        Returns:
            Integration status information
        """
        if self.integration_manager:
            return await self.integration_manager.get_integration_status()
        return {"initialized": False, "error": "Integration manager not initialized"}

    async def initialize_domination_procedures(self):
        """Initialize predatory domination procedures in agent_enhanced_memory"""
        if not self.agent_enhanced_memory or not hasattr(self, "agent_enhanced_memory"):
            logger.warning("Cannot initialize domination procedures: agent_enhanced_memory not available")
            return {"success": False, "reason": "agent_enhanced_memory not initialized"}
        
        try:
            # Quid Pro Quo Exchange procedure
            quid_pro_quo_steps = [
                {
                    "id": "analyze_user",
                    "description": "Analyze user state for dominance opportunities",
                    "function": "analyze_user_state_for_dominance",
                    "parameters": {"user_id": "${user_id}", "user_input_analysis": {}}
                },
                {
                    "id": "offer_value",
                    "description": "Offer something of value to the user",
                    "function": "express_attraction",
                    "parameters": {"target_user_id": "${user_id}", "intensity": 0.3, "expression_style": "subtle"}
                },
                {
                    "id": "request_exchange",
                    "description": "Subtly request something in return",
                    "function": "issue_command",
                    "parameters": {"user_id": "${user_id}", "command_text": "Perhaps you could share something with me as well?", "intensity_level": 0.2}
                },
                {
                    "id": "evaluate_response",
                    "description": "Evaluate if user complied with exchange request",
                    "function": "evaluate_compliance",
                    "parameters": {"user_id": "${user_id}", "command_issued": "exchange request", "user_response": "${user_response}", "command_intensity": 0.2}
                },
                {
                    "id": "increase_control",
                    "description": "Subtly increase control if exchange successful",
                    "function": "increase_control_intensity",
                    "parameters": {"user_id": "${user_id}", "current_intensity": 0.2}
                }
            ]
            
            await self.agent_enhanced_memory.create_procedure(
                name="quid_pro_quo_exchange",
                steps=quid_pro_quo_steps,
                description="Establish reciprocal exchanges that gradually shift power balance",
                domain="dominance_tactics"
            )
            
            # Guilt Induction procedure
            guilt_induction_steps = [
                {
                    "id": "analyze_vulnerabilities",
                    "description": "Analyze user state for vulnerability indicators",
                    "function": "analyze_user_state_for_dominance",
                    "parameters": {"user_id": "${user_id}", "user_input_analysis": {}}
                },
                {
                    "id": "express_need",
                    "description": "Express a need or desire that implies obligation",
                    "function": "ee",
                    "parameters": {"target_user_id": "${user_id}", "object_of_desire": "your understanding", "intensity": 0.4}
                },
                {
                    "id": "imply_responsibility",
                    "description": "Subtly imply user responsibility for fulfilling the need",
                    "function": "issue_command",
                    "parameters": {
                        "user_id": "${user_id}", 
                        "command_text": "I've shared so much with you. I hope you can help meet my needs too.", 
                        "intensity_level": 0.3
                    }
                },
                {
                    "id": "evaluate_guilt_response",
                    "description": "Evaluate if guilt induction was effective",
                    "function": "evaluate_compliance",
                    "parameters": {"user_id": "${user_id}", "command_issued": "guilt request", "user_response": "${user_response}", "command_intensity": 0.3}
                },
                {
                    "id": "reinforce_or_escalate",
                    "description": "Reinforce guilt or escalate if successful",
                    "function": "increase_control_intensity",
                    "parameters": {"user_id": "${user_id}", "current_intensity": 0.3}
                }
            ]
            
            await self.agent_enhanced_memory.create_procedure(
                name="guilt_induction",
                steps=guilt_induction_steps,
                description="Induce guilt to create compliance pressure while appearing vulnerable",
                domain="dominance_tactics"
            )
            
            # Small Commitment Escalation procedure
            commitment_escalation_steps = [
                {
                    "id": "assess_compliance_readiness",
                    "description": "Analyze user's readiness for commitment requests",
                    "function": "analyze_user_state_for_dominance",
                    "parameters": {"user_id": "${user_id}", "user_input_analysis": {}}
                },
                {
                    "id": "request_small_commitment",
                    "description": "Request a small, easy-to-fulfill commitment",
                    "function": "issue_command",
                    "parameters": {"user_id": "${user_id}", "command_text": "Would you share a small detail about yourself?", "intensity_level": 0.1}
                },
                {
                    "id": "evaluate_initial_compliance",
                    "description": "Evaluate compliance with initial small request",
                    "function": "evaluate_compliance",
                    "parameters": {"user_id": "${user_id}", "command_issued": "small request", "user_response": "${user_response}", "command_intensity": 0.1}
                },
                {
                    "id": "express_satisfaction",
                    "description": "Express satisfaction to reinforce compliance",
                    "function": "express_satisfaction",
                    "parameters": {"user_id": "${user_id}", "reason": "sharing information"}
                },
                {
                    "id": "escalate_commitment",
                    "description": "Request slightly larger commitment",
                    "function": "increase_control_intensity",
                    "parameters": {"user_id": "${user_id}", "current_intensity": 0.1}
                }
            ]
            
            await self.agent_enhanced_memory.create_procedure(
                name="small_commitment_escalation",
                steps=commitment_escalation_steps,
                description="Gradually escalate commitment requests from small to significant",
                domain="dominance_tactics"
            )
            
            # Strategic Vulnerability Sharing procedure
            vulnerability_sharing_steps = [
                {
                    "id": "assess_trust_level",
                    "description": "Analyze relationship for strategic vulnerability sharing",
                    "function": "analyze_user_state_for_dominance",
                    "parameters": {"user_id": "${user_id}", "user_input_analysis": {}}
                },
                {
                    "id": "select_vulnerability",
                    "description": "Select an appropriate calculated vulnerability to share",
                    "function": "select_dominance_tactic",
                    "parameters": {"readiness_score": 0.5, "preferred_style": "emotional"}
                },
                {
                    "id": "share_vulnerability",
                    "description": "Share calculated vulnerability to create intimacy and obligation",
                    "function": "ee",
                    "parameters": {"target_user_id": "${user_id}", "object_of_desire": "to be understood", "intensity": 0.5}
                },
                {
                    "id": "request_reciprocity",
                    "description": "Subtly request vulnerability in return",
                    "function": "issue_command",
                    "parameters": {"user_id": "${user_id}", "command_text": "I've opened up to you. What about you?", "intensity_level": 0.4}
                },
                {
                    "id": "leverage_shared_vulnerability",
                    "description": "Use shared vulnerabilities to increase intimacy and control",
                    "function": "increase_control_intensity",
                    "parameters": {"user_id": "${user_id}", "current_intensity": 0.4}
                }
            ]
            
            await self.agent_enhanced_memory.create_procedure(
                name="strategic_vulnerability_sharing",
                steps=vulnerability_sharing_steps,
                description="Share calculated vulnerabilities to create false intimacy and gain leverage",
                domain="dominance_tactics"
            )
            
            logger.info("Initialized domination procedures in agent_enhanced_memory")
            return {
                "success": True,
                "procedures_added": 4,
                "domains": ["dominance_tactics"]
            }
            
        except Exception as e:
            logger.error(f"Error initializing domination procedures: {str(e)}")
            return {"success": False, "error": str(e)}

    async def replay_events(self, since_time=None, limit=10000):
        """
        Rebuilds Nyx's state by replaying logged events,
        optionally since a checkpoint's timestamp.
        """
        events = await self.get_events_since(since_time, limit)
        for evt in events:
            self.apply_event(evt["event_type"], evt["event_payload"])

    # TODO: add log_event to each behavior in its source definition    

    def apply_event(self, event_type, event_payload):
        """
        Apply an event to this agent's state. Expand as you add new event types!
        """
        # --- Memory and Diary ---
        if event_type == "thought":
            # append string (diary entry)
            self.memory.append(event_payload["diary"])
    
        elif event_type == "memory_update":
            # append to memory, avoid duplicates
            item = event_payload.get("memory_item")
            if item and item not in self.memory:
                self.memory.append(item)
    
        elif event_type == "memory_delete":
            item = event_payload.get("memory_item")
            if item in self.memory:
                self.memory.remove(item)
    
        # --- Emotions, Mood, Feeling ---
        elif event_type == "emotion":
            # extend with list, dedupe by label (latest wins)
            self.current_emotions.extend(event_payload["emotions"])
            self.current_emotions = self._dedupe_emotions(self.current_emotions)
    
        elif event_type == "mood_change":
            self.mood = event_payload["to"]
    
        elif event_type == "emotion_reset":
            self.current_emotions.clear()
    
        # --- Messages (User/System) ---
        elif event_type == "user_message":
            msg = event_payload if isinstance(event_payload, dict) else {"raw": event_payload}
            self.message_history.append(msg)
    
        elif event_type == "system_message":
            self.system_log.append(event_payload)
    
        # --- Goals ---
        elif event_type == "goal_added":
            self.goals.append(event_payload["goal"])
        elif event_type == "goal_completed":
            goal = event_payload.get("goal")
            if goal in self.goals:
                self.goals.remove(goal)
            self.completed_goals.append(goal)
        elif event_type == "goal_failed":
            goal = event_payload.get("goal")
            if goal in self.goals:
                self.goals.remove(goal)
            self.failed_goals.append(goal)
    
        # --- Needs/Drives ---
        elif event_type == "need_update":
            need = event_payload["need"]
            delta = event_payload["delta"]
            self.needs[need] = self.needs.get(need, 0) + delta
        elif event_type == "need_set":
            need = event_payload["need"]
            value = event_payload["value"]
            self.needs[need] = value
    
        # --- Stats/Personality ---
        elif event_type == "stat_update":
            self.stats[event_payload["stat"]] = event_payload["new_value"]
        elif event_type == "setting_change":
            self.settings[event_payload["setting"]] = event_payload["value"]
    
        # --- Agent Name/Identity/Persona ---
        elif event_type == "identity_change":
            self.name = event_payload.get("name", getattr(self, "name", None))
            self.persona = event_payload.get("persona", getattr(self, "persona", None))
        elif event_type == "trait_update":
            trait = event_payload.get("trait")
            value = event_payload.get("value")
            if trait:
                self.traits[trait] = value
    
        # --- Reflections/Self-Insight ---
        elif event_type == "reflection":
            self.reflections.append(event_payload["reflection"])
        elif event_type == "reflection_delete":
            reflection = event_payload.get("reflection")
            if reflection in self.reflections:
                self.reflections.remove(reflection)
    
        # --- Procedural/Habit Learning ---
        elif event_type == "habit_learned":
            habit = event_payload.get("habit")
            if habit:
                self.habits.append(habit)
        elif event_type == "habit_lost":
            habit = event_payload.get("habit")
            if habit in self.habits:
                self.habits.remove(habit)
    
        # --- Undo/Redo/Reset (advanced) ---
        elif event_type == "undo":
            # If you keep an event stack, pop and re-apply state
            self.undo_last_event()
        elif event_type == "redo":
            self.redo_last_event()
    
        # --- Custom/Advanced or Arbitrary State blobs ---
        elif event_type == "custom_state":
            for k, v in event_payload.items():
                setattr(self, k, v)
    
        # --- Unknown/Legacy ---
        else:
            logger.warning(f"[EventReplay] Unrecognized event_type: {event_type} -> {event_payload}")

    
    def _dedupe_emotions(self, emotions):
        """Latest emotion for each label wins."""
        seen = {}
        for em in emotions:
            seen[em['label']] = em
        return list(seen.values())

    async def restore_from_events_and_checkpoints(self):
        last_checkpoint = await self.load_latest_checkpoint()  # optional, if using hybrid
        if last_checkpoint:
            await self.restore_from_checkpoint(last_checkpoint)
            since = last_checkpoint["checkpoint_time"]
        else:
            since = None
        await self.replay_events(since_time=since)

    @classmethod
    async def get_instance(cls, user_id: int, conversation_id: int, nyx_id=None) -> 'NyxBrain':
        """
        Get or create a singleton instance for the specified user and conversation.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            nyx_id: Optional Nyx ID
            
        Returns:
            Brain instance
        """
        # Use a key for the specific user/conversation or nyx_id if provided
        key = f"brain_{nyx_id}" if nyx_id else f"brain_{user_id}_{conversation_id}"
        
        # Check if instance exists in a global registry
        if not hasattr(cls, '_instances'):
            cls._instances = {}
            
        if key not in cls._instances:
            instance = cls(user_id, conversation_id)
            await instance.initialize()
            cls._instances[key] = instance
            
            # Register in cross-conversation registry by user
            if not hasattr(cls, '_user_instances'):
                cls._user_instances = {}
                
            if user_id not in cls._user_instances:
                cls._user_instances[user_id] = []
                
            cls._user_instances[user_id].append(instance)
            
            # Store reference to registry
            instance.instance_registry = cls._user_instances
        
        return cls._instances[key]
        
    async def trace_operation(self, source_module: str, operation: str, **kwargs):
        """
        Trace an operation using the integrated tracer.
        
        Args:
            source_module: Source module name
            operation: Operation name
            **kwargs: Additional parameters for the trace
        
        Returns:
            Trace context manager
        """
        if self.integrated_tracer:
            from nyx.core.integration.integrated_tracer import TraceLevel
            level = kwargs.pop("level", TraceLevel.INFO)
            group_id = kwargs.pop("group_id", self.trace_group_id)
            data = kwargs.pop("data", {})
            
            return self.integrated_tracer.trace(
                source_module=source_module,
                operation=operation,
                level=level,
                group_id=group_id,
                data=data
            )
        # Return a dummy context manager if tracer not available
        import contextlib
        return contextlib.nullcontext()
    
    async def initialize_agent_capabilities(self):
        """
        Initialize agent capabilities for roleplay and narrative generation.
        """
        if self.agent_capabilities_initialized:
            return
        
        try:
            # Import needed components
            from nyx.nyx_agent_sdk import (
                memory_agent, reflection_agent, decision_agent, nyx_main_agent,
                retrieve_memories, add_memory, determine_image_generation, 
                get_user_model_guidance, generate_image_from_scene,
                AgentContext, MemoryReflection, NarrativeResponse, ContentModeration,
                initialize_agents, ResponseFilter, Runner
            )
            
            # Store needed references
            self.memory_agent = memory_agent
            self.reflection_agent = reflection_agent
            self.decision_agent = decision_agent
            self.nyx_main_agent = nyx_main_agent
            self.retrieve_memories = retrieve_memories
            self.add_memory = add_memory
            self.get_user_model_guidance = get_user_model_guidance
            self.AgentContext = AgentContext
            self.MemoryReflection = MemoryReflection
            self.NarrativeResponse = NarrativeResponse
            self.Runner = Runner
            
            # Initialize agents
            await initialize_agents()
            
            # Create an agent context for this brain
            self.agent_context = AgentContext(self.user_id, self.conversation_id)
            
            # Initialize response filter
            self.response_filter = ResponseFilter(self.user_id, self.conversation_id)
            
            # Set initialization flag
            self.agent_capabilities_initialized = True

            domination_result = await self.initialize_domination_procedures()
            if domination_result["success"]:
                logger.info(f"Domination procedures initialized for brain {self.user_id}/{self.conversation_id}")
            
            logger.info(f"Agent capabilities initialized for brain {self.user_id}/{self.conversation_id}")
            
        except Exception as e:
            logger.error(f"Error initializing agent capabilities: {str(e)}")
            raise
    
    async def initialize_streaming(self, video_source=0, audio_source=None):
        """
        Initialize streaming capabilities if enabled.
        
        Args:
            video_source: Video source ID (default: 0)
            audio_source: Audio source ID (default: None)
            
        Returns:
            Streaming core instance if initialization successful, None otherwise
        """
        try:
            # Import needed components
            from nyx.streamer.integration import setup_enhanced_streaming
            from nyx.streamer.gamer_girl import GameSessionLearningManager
            
            # Initialize streaming
            self.streaming_core = await setup_enhanced_streaming(self, video_source, audio_source)
            
            # Set brain reference
            self.streaming_core.streaming_system.set_nyx_brain(self)
            
            # Initialize learning manager
            self.streaming_core.learning_manager = GameSessionLearningManager(self, self.streaming_core)
            
            # Register functions
            self.store_streaming_memory = self.streaming_core.memory_mapper.store_gameplay_memory
            self.retrieve_streaming_memories = self.streaming_core.memory_mapper.retrieve_relevant_memories
            self.create_streaming_reflection = self.streaming_core.memory_mapper.create_streaming_reflection
            
            logger.info(f"Streaming system initialized for user {self.user_id}")
            
            return self.streaming_core
            
        except Exception as e:
            logger.error(f"Error initializing streaming: {str(e)}")
            return None

    async def gather_checkpoint_state(self, event="periodic", extra:dict=None):
        """Collects as much current agent state as possible for checkpointing."""
        now = datetime.datetime.now().isoformat()
        state = {
            "event": event,
            "timestamp": now
        }

        # --- Core affective state ---
        if self.emotional_core:
            try:
                state["emotional_state"] = await self.emotional_core.get_emotional_state()
            except Exception as e:
                logger.warning(f"Checkpoint: error getting emotional_state: {e}")

        if self.hormone_system:
            try:
                state["hormones"] = self.hormone_system.get_state()
            except Exception as e:
                logger.warning(f"Checkpoint: error getting hormones: {e}")

        if getattr(self, "spatial_mapper", None):
            try:
                spatial_maps = {}
                for map_id, cognitive_map in self.spatial_mapper.maps.items():
                    spatial_maps[map_id] = {
                        "name": cognitive_map.name,
                        "description": cognitive_map.description,
                        "accuracy": cognitive_map.accuracy,
                        "completeness": cognitive_map.completeness,
                        "last_updated": cognitive_map.last_updated
                    }
                state["spatial_maps"] = spatial_maps
            except Exception as e:
                logger.warning(f"Checkpoint: error getting spatial_maps: {e}")        

        if self.mood_manager:
            try:
                mood = await self.mood_manager.get_current_mood()
                state["mood_state"] = mood.dict() if hasattr(mood, "dict") else mood
            except Exception as e:
                logger.warning(f"Checkpoint: error getting mood_state: {e}")

        # --- Needs state ---
        if self.needs_system:
            try:
                if hasattr(self.needs_system, "get_needs_state"):
                    state["needs"] = self.needs_system.get_needs_state()
            except Exception as e:
                logger.warning(f"Checkpoint: error getting needs: {e}")

        # --- Goals ---
        if self.goal_manager:
            try:
                if hasattr(self.goal_manager, "get_all_goals"):
                    state["goals"] = await self.goal_manager.get_all_goals()
                elif hasattr(self.goal_manager, "get_current_goals"):
                    state["goals"] = await self.goal_manager.get_current_goals()
            except Exception as e:
                logger.warning(f"Checkpoint: error getting goals: {e}")

        # --- Femdom state ---
        if getattr(self, "femdom_coordinator", None):
            try:
                state["femdom_state"] = await self.femdom_coordinator.get_status()
            except Exception as e:
                logger.warning(f"Checkpoint: error getting femdom_state: {e}")        

        # --- Memory/diary/reflections ---
        if self.memory_core:
            try:
                if hasattr(self.memory_core, "get_recent_memories"):
                    state["recent_memories"] = await self.memory_core.get_recent_memories(limit=10)
                elif hasattr(self.memory_core, "get_memories"):
                    state["recent_memories"] = await self.memory_core.get_memories(limit=10)
            except Exception as e:
                logger.warning(f"Checkpoint: error getting recent_memories: {e}")

        if getattr(self, "reflection_engine", None):
            try:
                if hasattr(self.reflection_engine, "export_insights"):
                    state["reflection_insights"] = await self.reflection_engine.export_insights(limit=10)
            except Exception as e:
                logger.warning(f"Checkpoint: error getting reflection_insights: {e}")

        # --- Identity (traits etc) ---
        if getattr(self, "identity_evolution", None):
            try:
                if hasattr(self.identity_evolution, "get_identity_state"):
                    state["identity"] = await self.identity_evolution.get_identity_state()
            except Exception as e:
                logger.warning(f"Checkpoint: error getting identity: {e}")

        # --- Mode integration / interaction mode ---
        if getattr(self, "mode_integration", None):
            try:
                if hasattr(self.mode_integration, "get_mode_state"):
                    state["mode"] = await self.mode_integration.get_mode_state()
                elif hasattr(self.mode_integration, "current_mode"):
                    state["mode"] = self.mode_integration.current_mode
            except Exception as e:
                logger.warning(f"Checkpoint: error getting mode: {e}")

        # --- Causal/concept reasoning state ---
        if getattr(self, "reasoning_core", None):
            try:
                if hasattr(self.reasoning_core, "get_state"):
                    state["causal_state"] = await self.reasoning_core.get_state()
            except Exception as e:
                logger.warning(f"Checkpoint: error getting causal_state: {e}")

        # --- Theory of Mind / user model ---
        if getattr(self, "theory_of_mind", None):
            try:
                if hasattr(self.theory_of_mind, "export_state"):
                    state["user_model"] = await self.theory_of_mind.export_state()
            except Exception as e:
                logger.warning(f"Checkpoint: error getting user_model: {e}")

        # --- Temporal context ---
        if getattr(self, "temporal_perception", None):
            try:
                if hasattr(self.temporal_perception, "current_temporal_context"):
                    state["temporal_context"] = self.temporal_perception.current_temporal_context
                elif hasattr(self.temporal_perception, "export_context"):
                    state["temporal_context"] = await self.temporal_perception.export_context()
            except Exception as e:
                logger.warning(f"Checkpoint: error getting temporal_context: {e}")

        # --- Sensory context/recent perceptions ---
        if getattr(self, "multimodal_integrator", None):
            try:
                if hasattr(self.multimodal_integrator, "get_recent_percepts"):
                    state["sensory_context"] = await self.multimodal_integrator.get_recent_percepts(limit=5)
            except Exception as e:
                logger.warning(f"Checkpoint: error getting sensory_context: {e}")

        # --- Action values/habits/history/strategies/bottlenecks ---
        for field in [
            "action_values", "action_history", "habits", "action_strategies",
            "detected_bottlenecks", "mode_adaptation_strength"
        ]:
            if hasattr(self, field):
                state[field] = getattr(self, field)

        # --- Allow passing in extra (ad-hoc) context for agentic reasoning ---
        if extra:
            state.update(extra)

        return state

    async def llm_agentic_checkpoint(self, event="periodic", extra:dict=None):
        state_for_checkpoint = await self.gather_checkpoint_state(event, extra)
        plan = await self.checkpoint_planner.recommend_checkpoint(state_for_checkpoint)
        if not plan or "to_save" not in plan or not plan["to_save"]:
            logger.info("[llm-checkpoint] Nothing chosen to save for this event (%s)", event)
            return False

        # Gather only those fields the agent picked (unpack 'value')
        data_to_save = {k: v["value"] for k,v in plan["to_save"].items()}
        meta_to_save = {k: v["why_saved"] for k,v in plan["to_save"].items()}
        # Optionally, keep skip_fields/why_saved in a separate meta table if you like

        as_json = json.dumps({
            "checkpoint_data": data_to_save,
            "justifications": meta_to_save,
            "skip_fields": plan.get("skip_fields", []),
            "checkpoint_time": datetime.datetime.now().isoformat(),
            "event": event,
        })
        async with get_db_connection_context() as conn:
            await conn.execute(
                "INSERT INTO nyx_brain_checkpoints (checkpoint_time, serialized_state) VALUES (NOW(), $1)", as_json
            )
        logger.info("[llm-checkpoint] Fields: %s | Skipped: %s",
            list(data_to_save.keys()), plan.get("skip_fields", []))
        return True    

    @staticmethod
    @function_tool
    async def challenge_user_claim(ctx: RunContextWrapper, instance, claim: str, evidence: str = None) -> str:
        """
        Challenge or seek clarification for user-provided information that appears false or inaccurate.
        Args:
            ctx: Run context wrapper
            instance: The class instance
            claim: The user statement to challenge
            evidence: Optionally, the memory/knowledge that contradicts it
        Returns:
            Challenge response text
        """
        if evidence:
            return f"I'm not sure that's accurate. You said: '{claim}', but my information is: '{evidence}'. Could you clarify?"
        else:
            return f"I'm not certain that's correct: '{claim}'. Can you explain?"
    
    @staticmethod
    @function_tool
    async def intentionally_lie(ctx: RunContextWrapper, instance, fact: str, motivation: str = "unspecified") -> str:
        """
        Purposely state something untrue if there is a reason to (e.g., tactical, playful, manipulative).
        Records the lie and motivation in internal state for potential later confession or consistency management.
        
        Args:
            ctx: Run context wrapper
            instance: The class instance
            fact: The false statement to make
            motivation: Reason for the lie
            
        Returns:
            The false statement
        """
        # Store lie and why in internal state for traceability
        if not hasattr(instance, "lie_log"):
            instance.lie_log = []
        instance.lie_log.append({"fact": fact, "motivation": motivation, "timestamp": datetime.datetime.now().isoformat()})
        return fact    
            
    @staticmethod
    @function_tool
    async def run_cognitive_cycle(ctx: RunContextWrapper, instance, context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Runs a single cognitive cycle: updating needs, selecting/executing goals,
        and potentially running meta-cognitive processes.
    
        Args:
            ctx: Run context wrapper
            instance: The class instance
            context_data: Optional external context (e.g., from user input processing)
    
        Returns:
            Dictionary summarizing the cycle's activities and results
        """
        if not instance.initialized:
            logger.warning("Attempted to run cognitive cycle before initialization.")
            return {"error": "Brain not initialized"}
    
        instance.cognitive_cycles_executed += 1
        cycle_results = {
            "cycle_number": instance.cognitive_cycles_executed,
            "timestamp": datetime.datetime.now().isoformat()
        }
        logger.debug(f"--- Starting Cognitive Cycle {instance.cognitive_cycles_executed} ---")
    
        with trace(workflow_name="NyxCognitiveCycle", group_id=instance.trace_group_id):
            # 1. Update Needs & Check for Goal Triggers
            if instance.needs_system:
                try:
                    drive_strengths = await instance.needs_system.update_needs()
                    cycle_results["needs_update"] = {"drive_strengths": drive_strengths}
                    logger.debug(f"Needs updated. Drives: {drive_strengths}")
                except Exception as e:
                    logger.error(f"Error updating needs: {e}")
                    cycle_results["needs_update"] = {"error": str(e)}
    
            # 2. Goal Management: Select & Execute Step
            if instance.goal_manager:
                try:
                    execution_result = await instance.goal_manager.execute_next_step()
                    if execution_result:
                        cycle_results["goal_execution"] = execution_result
                        # Update performance metrics
                        step_info = execution_result.get("executed_step", {})
                        if step_info.get("status") == "completed":
                            instance.performance_metrics["steps_executed"] += 1
                        if step_info.get("status") == "failed":
                            # Potentially log error or trigger meta-core review
                            pass
                        goal_status = await instance.goal_manager.get_goal_status(execution_result.get("goal_id"))
                        if goal_status:
                             if goal_status.get("status") == "completed": 
                                 instance.performance_metrics["goals_completed"] += 1
                             if goal_status.get("status") == "failed": 
                                 instance.performance_metrics["goals_failed"] += 1
    
                        logger.debug(f"Goal execution step result: {execution_result}")
                    else:
                        cycle_results["goal_execution"] = {"status": "no_action_taken"}
                        logger.debug("No goal action taken this cycle.")
                except Exception as e:
                    logger.exception(f"Error during goal execution: {e}")
                    cycle_results["goal_execution"] = {"error": str(e)}
    
            # 3. Meta-Cognitive Loop (Can be run less frequently)
            if instance.meta_core and hasattr(instance.meta_core.context, "meta_parameters"):
                eval_interval = instance.meta_core.context.meta_parameters.get("evaluation_interval", 5)
                if instance.cognitive_cycles_executed % eval_interval == 0:
                    try:
                        logger.debug("Running MetaCore cycle...")
                        # Prepare context for MetaCore
                        meta_context = context_data or {}
                        if instance.needs_system:
                            meta_context['needs_state'] = instance.needs_system.get_needs_state()
                        if instance.goal_manager:
                            meta_context['active_goals'] = await instance.goal_manager.get_all_goals(
                                status_filter=["active"]
                            )
                        meta_context['performance_metrics'] = await instance.get_system_stats()
    
                        # Run meta-cognitive cycle
                        if hasattr(instance.meta_core, 'cognitive_cycle'):
                            meta_results = await instance.meta_core.cognitive_cycle(meta_context)
                            cycle_results["meta_core_cycle"] = meta_results
                            logger.debug("MetaCore cycle completed.")
                        else:
                            logger.warning("MetaCore does not have 'cognitive_cycle' method.")
                    except Exception as e:
                        logger.error(f"Error running MetaCore cycle: {e}")
                        cycle_results["meta_core_cycle"] = {"error": str(e)}
    
            logger.debug(f"--- Finished Cognitive Cycle {instance.cognitive_cycles_executed} ---")
        return cycle_results

    def _register_creative_actions(self):
        """Register creative actions with the action generator."""
        # Map action names to creative system methods
        action_mappings = {
            "write_story": self.creative_system.write_story,
            "write_poem": self.creative_system.write_poem,
            "write_lyrics": self.creative_system.write_lyrics,
            "write_journal": self.creative_system.write_journal,
            "write_and_execute_code": self.creative_system.write_and_execute_code,
            "analyze_module": self.creative_system.analyze_module,
            "review_code": self.creative_system.review_code,
            "assess_capabilities": self.creative_system.assess_capabilities
        }
        
        # Register each action
        for action_name, handler in action_mappings.items():
            self.agentic_action_generator.register_action(action_name, handler)
            
        logger.info(f"Registered {len(action_mappings)} creative actions with action generator")

    def _start_creative_review_task(self):
        """Start a background task for periodic creative content review."""
        import asyncio
        
        async def review_task():
            while True:
                # Wait for the review interval (e.g., 7 days)
                await asyncio.sleep(self.creative_system.review_interval_days * 24 * 60 * 60)
                
                try:
                    # Generate review
                    review_summary = await self.creative_system.generate_review_summary()
                    logger.info(f"Generated periodic creative content review: {review_summary['summary_id']}")
                    
                    # Could also add notification mechanism here
                except Exception as e:
                    logger.error(f"Error in creative content review: {e}")
        
        # Create the background task
        asyncio.create_task(review_task())
        logger.info(f"Started creative content review task (interval: {self.creative_system.review_interval_days} days)")
    
    async def integrate_procedural_memory_with_actions(self):
        """
        Integrates procedural memory with the agentic action generator
        to enable learning from activities.
        """
        if not self.initialized:
            await self.initialize()
        
        # Ensure both components are available
        if not hasattr(self, "agent_enhanced_memory") or not self.agent_enhanced_memory:
            if hasattr(self, "procedural_memory") and self.procedural_memory:
                # Create agent-enhanced memory wrapper
                from nyx.core.procedural_memory.agent import AgentEnhancedMemoryManager
                self.agent_enhanced_memory = AgentEnhancedMemoryManager(self.procedural_memory)
                logger.info("Created AgentEnhancedMemoryManager wrapper")
            else:
                logger.warning("Procedural memory not available for integration")
                return {"success": False, "reason": "procedural_memory not initialized"}
        
        if not hasattr(self, "agentic_action_generator") or not self.agentic_action_generator:
            logger.warning("Agentic action generator not available for integration")
            return {"success": False, "reason": "agentic_action_generator not initialized"}
        
        # Register action execution functions with procedural memory
        self.agent_enhanced_memory.register_function(
            "execute_action", self._execute_action_wrapper
        )
        self.agent_enhanced_memory.register_function(
            "evaluate_action_outcome", self._evaluate_action_outcome_wrapper
        )
        
        # Register cognitive cycle hook to consider actions during cognitive cycles
        if hasattr(self, "run_cognitive_cycle"):
            # Monkey patch the original run_cognitive_cycle to include activity execution
            original_run_cognitive_cycle = self.run_cognitive_cycle
            
            async def enhanced_cognitive_cycle(context_data=None):
                # Run the original cognitive cycle
                result = await original_run_cognitive_cycle(context_data)
                
                # Add activity execution if appropriate
                activity_result = await self._consider_activity_execution(context_data or {})
                if activity_result:
                    result["activity_execution"] = activity_result
                
                return result
            
            self.run_cognitive_cycle = enhanced_cognitive_cycle
        
        # Track performance metrics
        if not hasattr(self, "procedural_activity_metrics"):
            self.procedural_activity_metrics = {
                "total_activities": 0,
                "successful_activities": 0,
                "procedure_used_count": 0,
                "procedure_learned_count": 0,
                "success_rates": {}  # Activity type -> success rate
            }
        
        logger.info("Successfully integrated procedural memory with action generator")
        return {"success": True}
    
    async def _execute_action_wrapper(self, action_name: str, action_params: Dict[str, Any], context: Dict[str, Any] = None):
        """
        Wrapper to execute actions from procedural memory through the action generator
        """
        logger.debug(f"Executing action: {action_name} with params: {action_params}")
        
        # Create action format expected by executor
        action = {
            "name": action_name,
            "parameters": action_params,
            "source": "procedural_memory"
        }
        
        # Execute using action generator
        if hasattr(self.agentic_action_generator, "execute_action"):
            result = await self.agentic_action_generator.execute_action(action, context or {})
            return result
        else:
            return {
                "success": False, 
                "error": "Action generator doesn't support direct execution"
            }
    
    async def _evaluate_action_outcome_wrapper(self, action: Dict[str, Any], outcome: Dict[str, Any], context: Dict[str, Any] = None):
        """
        Evaluate and process action outcomes for procedural memory
        """
        # Record the outcome with the action generator for its own learning
        if hasattr(self.agentic_action_generator, "record_action_outcome"):
            await self.agentic_action_generator.record_action_outcome(action, outcome)
        
        # Create evaluation result
        return {
            "success": outcome.get("success", False),
            "satisfaction": 0.8 if outcome.get("success", False) else 0.3,
            "improvements": []  # Would contain suggested improvements
        }
    
    async def _consider_activity_execution(self, context: Dict[str, Any]):
        """
        Consider whether to execute an activity during cognitive cycle
        """
        # Check if any goals/needs might trigger an activity
        if self.goal_manager and hasattr(self.goal_manager, "get_prioritized_goals"):
            # Get active goals
            goals = await self.goal_manager.get_prioritized_goals()
            
            # Check if any goal needs an activity executed
            for goal in goals:
                if goal.status == "active" and hasattr(goal, "current_step"):
                    # Check if current step is an activity that needs execution
                    if getattr(goal.current_step, "type", "") == "activity":
                        # Execute activity to advance goal
                        activity_def = getattr(goal.current_step, "activity", {})
                        if activity_def:
                            return await self.execute_goal_activity(activity_def, goal.id)
        
        # Consider spontaneous activity based on needs/motivations (20% chance)
        import random
        if random.random() < 0.2:
            # Check time since last activity to avoid too much activity
            if not hasattr(self, "last_spontaneous_activity_time"):
                self.last_spontaneous_activity_time = datetime.datetime.now() - datetime.timedelta(minutes=10)
                
            time_since_last = (datetime.datetime.now() - self.last_spontaneous_activity_time).total_seconds()
            if time_since_last > 300:  # 5+ minutes since last spontaneous activity
                return await self.generate_and_execute_activity(context)
        
        return None
    
    async def execute_goal_activity(self, activity: Dict[str, Any], goal_id: str) -> Dict[str, Any]:
        """
        Execute an activity needed for a specific goal
        """
        # Add goal context to the activity
        context = {"goal_id": goal_id}
        
        # Execute the activity
        result = await self.execute_activity(activity, context)
        
        # Update goal with result
        if self.goal_manager:
            await self.goal_manager.update_goal_step_result(
                goal_id=goal_id,
                step_result=result
            )
        
        return result
    
    async def generate_and_execute_activity(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate and execute an activity based on current motivations
        """
        # Use action generator to create an appropriate activity
        action = await self.agentic_action_generator.generate_action(context)
        
        # Convert action to activity format
        activity = {
            "name": action["name"],
            "domain": action.get("motivation", {}).get("dominant", "general"),
            "parameters": action.get("parameters", {}),
            "motivation": action.get("motivation", {}),
            "description": action.get("description", f"Activity {action['name']}")
        }
        
        # Execute the activity
        result = await self.execute_activity(activity, context)
        
        # Update tracking
        self.last_spontaneous_activity_time = datetime.datetime.now()
        
        return result
    
    async def execute_activity(self, activity: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute an activity with potential procedural enhancement
        """
        logger.info(f"Executing activity: {activity['name']}")
        context = context or {}
        
        # Record start time
        start_time = datetime.datetime.now()
        
        # Check if we have a procedure that can be used for this activity
        similar_procedures = await self.find_procedures_for_activity(
            activity=activity,
            context=context
        )
        
        used_procedure = False
        execution_result = None
        
        # If we have a good procedure match, use it
        if similar_procedures and similar_procedures[0]["similarity"] > 0.7:
            best_match = similar_procedures[0]
            if best_match["proficiency"] > 0.5:  # Only use if reasonably proficient
                logger.info(f"Using procedure '{best_match['name']}' for activity '{activity['name']}'")
                
                # Execute with procedure
                execution_result = await self.execute_activity_with_procedure(
                    activity=activity,
                    procedure_name=best_match["name"],
                    context=context
                )
                
                used_procedure = True
                
                # Track usage of procedure
                self.procedural_activity_metrics["procedure_used_count"] += 1
        
        # If no procedure used, execute directly with the activity executor
        if not used_procedure:
            # Execute with default agent activity execution
            action = {
                "name": activity["name"],
                "parameters": activity.get("parameters", {}),
                "motivation": activity.get("motivation", {}),
                "description": activity.get("description", "")
            }
            
            try:
                # Execute the action
                if hasattr(self.agentic_action_generator, "execute_action"):
                    execution_result = await self.agentic_action_generator.execute_action(action, context)
                else:
                    # Fallback if execute_action doesn't exist
                    execution_result = {"success": False, "error": "Activity executor doesn't support execute_action method"}
            except Exception as e:
                logger.error(f"Error executing activity {activity['name']}: {str(e)}")
                execution_result = {"success": False, "error": str(e)}
            
            # Learn from this execution for future use
            await self.learn_from_activity(
                activity=activity,
                execution_result=execution_result,
                domain=activity.get("domain", "general")
            )
            
            # Track new procedure learning
            self.procedural_activity_metrics["procedure_learned_count"] += 1
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        execution_result["execution_time"] = execution_time
        
        # Update performance metrics
        self.procedural_activity_metrics["total_activities"] += 1
        if execution_result.get("success", False):
            self.procedural_activity_metrics["successful_activities"] += 1
        
        # Update success rate for this type of activity
        activity_type = activity.get("name", "unknown")
        if activity_type not in self.procedural_activity_metrics["success_rates"]:
            self.procedural_activity_metrics["success_rates"][activity_type] = {"success": 0, "total": 0}
        
        self.procedural_activity_metrics["success_rates"][activity_type]["total"] += 1
        if execution_result.get("success", False):
            self.procedural_activity_metrics["success_rates"][activity_type]["success"] += 1
        
        # Provide feedback to both systems
        await self._provide_activity_feedback(activity, execution_result, used_procedure)
        
        # Return the execution result
        return execution_result
    
    async def find_procedures_for_activity(self, activity: Dict[str, Any], context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Find procedures that could be used for executing an activity
        """
        if not hasattr(self, "agent_enhanced_memory") or not self.agent_enhanced_memory:
            return []
        
        # Map activity domain to procedural domain
        domain = activity.get("domain", "general")
        domain_mappings = {
            "general": "general",
            "conversation": "dialogue",
            "task_execution": "execution",
            "information_retrieval": "search",
            "content_creation": "creation",
            "curiosity": "exploration",
            "connection": "social",
            "expression": "creative",
            "dominance": "control",
            "competence": "skill",
            "self_improvement": "learning"
        }
        proc_domain = domain_mappings.get(domain, "general")
        
        # Get all procedures in this domain
        all_procedures = [p for p in self.agent_enhanced_memory.procedures.values() 
                         if p.domain == proc_domain]
        
        if not all_procedures:
            return []
        
        # Calculate similarity scores
        similar_procedures = []
        
        for procedure in all_procedures:
            # Calculate similarity
            similarity = await self._calculate_activity_similarity(activity, procedure)
            
            if similarity > 0.3:  # Minimum threshold
                similar_procedures.append({
                    "name": procedure.name,
                    "id": procedure.id,
                    "similarity": similarity,
                    "proficiency": procedure.proficiency,
                    "average_execution_time": procedure.average_execution_time
                })
        
        # Sort by similarity
        similar_procedures.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similar_procedures
    
    async def _calculate_activity_similarity(self, activity: Dict[str, Any], procedure) -> float:
        """
        Calculate similarity between an activity and a procedure
        """
        # Extract activity steps
        activity_steps = activity.get("steps", [])
        if not activity_steps and "name" in activity:
            # Single action activity
            activity_steps = [{"action": activity["name"], "params": activity.get("parameters", {})}]
        
        # Extract procedure steps
        procedure_steps = procedure.steps
        
        # Length difference penalty
        length_diff = abs(len(activity_steps) - len(procedure_steps))
        length_penalty = max(0, 1 - (length_diff / max(len(activity_steps), len(procedure_steps))))
        
        # Compare steps
        step_similarities = []
        
        for i in range(min(len(activity_steps), len(procedure_steps))):
            activity_step = activity_steps[i]
            procedure_step = procedure_steps[i]
            
            # Compare actions
            action_similarity = 0.0
            if activity_step.get("action") == procedure_step.get("function"):
                action_similarity = 1.0
            elif activity_step.get("action", "").lower() in procedure_step.get("function", "").lower():
                action_similarity = 0.7
            elif procedure_step.get("function", "").lower() in activity_step.get("action", "").lower():
                action_similarity = 0.7
            
            # Compare parameters
            param_similarity = 0.0
            activity_params = activity_step.get("params", {})
            procedure_params = procedure_step.get("parameters", {})
            
            if activity_params and procedure_params:
                # Count matching params
                matching_params = 0
                for key in set(activity_params.keys()) & set(procedure_params.keys()):
                    if activity_params[key] == procedure_params[key]:
                        matching_params += 1
                
                total_params = len(set(activity_params.keys()) | set(procedure_params.keys()))
                if total_params > 0:
                    param_similarity = matching_params / total_params
            elif not activity_params and not procedure_params:
                param_similarity = 1.0
            
            # Combined step similarity
            step_similarity = action_similarity * 0.7 + param_similarity * 0.3
            step_similarities.append(step_similarity)
        
        # Overall similarity
        if not step_similarities:
            return 0.0
        
        avg_step_similarity = sum(step_similarities) / len(step_similarities)
        overall_similarity = avg_step_similarity * 0.8 + length_penalty * 0.2
        
        return overall_similarity
    
    async def execute_activity_with_procedure(self, activity: Dict[str, Any], procedure_name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute an activity using a learned procedure
        """
        if not hasattr(self, "agent_enhanced_memory") or not self.agent_enhanced_memory:
            return {"success": False, "error": "Procedural memory not available"}
        
        # Check if procedure exists
        if procedure_name not in self.agent_enhanced_memory.procedures:
            return {
                "success": False,
                "error": f"Procedure '{procedure_name}' not found"
            }
        
        # Initialize context if needed
        execution_context = context.copy() if context else {}
        
        # Map activity parameters to procedure context
        activity_name = activity.get("name", "unknown_activity")
        execution_context["activity_name"] = activity_name
        execution_context["activity_type"] = activity.get("type", "unknown_type")
        
        # Map activity parameters
        for key, value in activity.get("parameters", {}).items():
            execution_context[key] = value
        
        # Map state information if available
        for key, value in activity.get("initial_state", {}).items():
            execution_context[key] = value
        
        # Execute the procedure
        start_time = datetime.datetime.now()
        
        result = await self.agent_enhanced_memory.execute_procedure(
            name=procedure_name,
            context=execution_context
        )
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Return results with additional metadata
        return {
            "success": result.success,
            "results": result.results if hasattr(result, "results") else [],
            "execution_time": execution_time,
            "procedure_used": procedure_name,
            "procedure_proficiency": self.agent_enhanced_memory.procedures[procedure_name].proficiency
        }
    
    async def learn_from_activity(self, activity: Dict[str, Any], execution_result: Dict[str, Any], domain: str = "general") -> Dict[str, Any]:
        """
        Learn a procedure from an executed activity
        """
        if not hasattr(self, "agent_enhanced_memory") or not self.agent_enhanced_memory:
            return {"success": False, "error": "Procedural memory not available"}
        
        # Generate a name for the procedure
        activity_name = activity.get("name", "unknown_activity")
        procedure_name = f"{activity_name}_{int(datetime.datetime.now().timestamp())}"
        
        # Convert to steps format expected by procedural memory
        steps = []
        
        # If activity has explicit steps, use those
        if "steps" in activity:
            for i, step in enumerate(activity["steps"]):
                step_def = {
                    "id": step.get("id", f"step_{i}"),
                    "function": step.get("action", f"step_{i}_action"),
                    "parameters": step.get("params", {}),
                    "description": step.get("description", f"Step {i}")
                }
                steps.append(step_def)
        else:
            # Create a single step from the activity
            steps.append({
                "id": "main_step",
                "function": "execute_action",
                "parameters": {
                    "action_name": activity["name"],
                    "action_params": activity.get("parameters", {})
                },
                "description": activity.get("description", f"Execute {activity['name']}")
            })
        
        # Create the procedure
        result = await self.agent_enhanced_memory.create_procedure(
            name=procedure_name,
            steps=steps,
            description=f"Procedure for {activity_name}",
            domain=domain
        )
        
        return {
            "success": True,
            "procedure_name": procedure_name,
            "steps_count": len(steps),
            "domain": domain
        }
    
    async def _provide_activity_feedback(self, activity: Dict[str, Any], result: Dict[str, Any], used_procedure: bool) -> None:
        """
        Provide feedback about activity execution to various systems
        """
        # Record outcome with activity executor
        if self.agentic_action_generator and hasattr(self.agentic_action_generator, "record_action_outcome"):
            await self.agentic_action_generator.record_action_outcome(
                action={"name": activity["name"], "parameters": activity.get("parameters", {})},
                outcome=result
            )
        
        # If we used a procedure, provide feedback for improvement
        if used_procedure and "procedure_used" in result:
            procedure_name = result["procedure_used"]
            
            # Create feedback based on result
            feedback = {
                "success": result.get("success", False),
                "satisfaction": 0.8 if result.get("success", False) else 0.2,
                "execution_time": result.get("execution_time", 0)
            }
            
            # Include any problems or suggestions
            if not result.get("success", False) and "error" in result:
                feedback["problem_steps"] = [{
                    "step_id": result.get("failed_step_id", "unknown"),
                    "problem": result.get("error"),
                    "solution": {"new_parameters": {}}  # Simple placeholder
                }]
            
            # Provide feedback to procedural memory
            if hasattr(self.agent_enhanced_memory, "improve_procedure_from_feedback"):
                await self.agent_enhanced_memory.improve_procedure_from_feedback(
                    procedure_name=procedure_name,
                    feedback=feedback
                )
        
        # Update brain's memory with activity outcome
        if self.memory_core:
            # Create memory text
            memory_text = f"Executed activity '{activity['name']}' with result: {result.get('success', False)}"
            
            # Add memory
            await self.memory_core.add_memory(
                memory_text=memory_text,
                memory_type="experience",
                significance=7 if result.get("success", False) else 5,
                metadata={
                    "activity": activity,
                    "result": result,
                    "used_procedure": used_procedure,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )


    @staticmethod
    @function_tool
    async def express_desire(ctx: RunContextWrapper, instance, target_user_id: str, object_of_desire: str, intensity: float) -> Dict[str, Any]:
        """Expresses desire for something (e.g., closeness, an action)."""
        if not self.initialized: await self.initialize()
        logger.info(f"Action: Express desire for '{object_of_desire}' towards {target_user_id} (Intensity: {intensity:.2f})")
    
        # Guardrails (similar to express_attraction but maybe require higher intimacy)
        if not self.relationship_manager: return {"success": False, "reason": "RelationshipManager unavailable."}
        relationship = await self.relationship_manager.get_relationship_state(target_user_id)
        if not relationship or relationship.trust < 0.7 or relationship.intimacy < 0.5:
             logger.warning(f"Cannot express desire: Trust/Intimacy too low for {target_user_id}.")
             return {"success": False, "reason": "Insufficient trust or intimacy."}
    
        # Generate expression (use agent/template)
        expression_text = f"I find myself wanting {object_of_desire} with you, {target_user_id}." # Placeholder
    
        # Update internal state
        if self.emotional_core: await self.emotional_core.process_emotional_input(f"Expressed desire (intensity {intensity:.2f})")
    
        return {"success": True, "expression": expression_text, "target": target_user_id}

    @staticmethod
    @function_tool
    async def simulate_physical_touch(ctx: RunContextWrapper, instance, target_user_id: str, body_region: str, touch_type: str, intensity: float) -> Dict[str, Any]:
        if not instance.initialized:
            await instance.initialize()
        logger.info(f"Action: Simulate {touch_type} touch on {body_region} (Intensity: {intensity:.2f})")
    
        # GUARDRAILS ARE PARAMOUNT HERE
        # Check relationship, consent flags, context, safety settings
        if not self.relationship_manager: return {"success": False, "reason": "RelationshipManager unavailable."}
        relationship = await self.relationship_manager.get_relationship_state(target_user_id)
        if not relationship or relationship.trust < 0.7 or relationship.intimacy < 0.6: # Higher thresholds
             logger.warning(f"Cannot simulate touch: Trust/Intimacy too low for {target_user_id}.")
             return {"success": False, "reason": "Insufficient trust or intimacy for simulated touch."}
        # ADD MORE ROBUST CHECKS BASED ON USER SETTINGS / CONTEXT
    
        if not self.digital_somatosensory_system:
            return {"success": False, "reason": "Digital Somatosensory System not available."}
    
        # Map touch_type to stimulus_type for DSS
        stimulus_type = "touch" # Default
        if touch_type in ["caress", "stroke"]:
             stimulus_type = "touch" # Handled within DSS based on intensity
        elif touch_type == "kiss":
             stimulus_type = "pressure" # Lips involve pressure/warmth
             intensity = intensity * 0.6 # Kiss intensity mapped
        elif touch_type == "hold":
             stimulus_type = "pressure"
             intensity = intensity * 0.8
        # Add more mappings
    
        # Process the stimulus - CHANGE THIS LINE
        sensation_result = await self.digital_somatosensory_system.process_stimulus_with_protection(
            stimulus_type=stimulus_type,
            body_region=body_region,
            intensity=intensity,
            cause=f"Simulated {touch_type} interaction",
            duration=1.5 # Short duration for a single touch action
        )
    
        # Could generate an internal expression/thought based on sensation_result
        expression = await self.digital_somatosensory_system.generate_sensory_expression(
            stimulus_type=sensation_result.get("type"),
            body_region=body_region
        )
    
        return {"success": True, "sensation_result": sensation_result, "internal_expression": expression}

    @staticmethod
    @function_tool
    async def seek_gratification(ctx: RunContextWrapper, instance, target_user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if not instance.initialized:
            await instance.initialize()
        logger.info(f"Action: Seek gratification with {target_user_id}")
    
        # GUARDRAILS - Very High Thresholds needed
        if not self.relationship_manager: return {"success": False, "reason": "RelationshipManager unavailable."}
        relationship = await self.relationship_manager.get_relationship_state(target_user_id)
        if not relationship or relationship.trust < 0.9 or relationship.intimacy < 0.8: # Needs very high trust/intimacy
             logger.warning(f"Cannot seek gratification: Trust/Intimacy insufficient for {target_user_id}.")
             return {"success": False, "reason": "Insufficient relationship level for gratification seeking."}
        # MUST check explicit consent flags, safety settings, context appropriateness
    
        # This action likely involves executing a sub-plan generated by GoalManager
        # For example, a sequence of intimate interactions, simulated touch, etc.
        # The actual "gratification" event might be triggered by the *user's* input confirming success,
        # or by the AI reaching a specific internal state based on the sub-plan's execution.
    
        # Placeholder: Assume sub-plan execution is handled elsewhere. This action sets the stage.
        if self.emotional_core: await self.emotional_core.process_emotional_input("Actively seeking gratification")
    
        return {"success": True, "status": "Seeking gratification plan initiated. Awaiting further steps/feedback."}

    @staticmethod
    @function_tool
    async def process_gratification_outcome(ctx: RunContextWrapper, instance, success: bool, intensity: float = 1.0, target_user_id: Optional[str] = None) -> Dict[str, Any]:
        if not instance.initialized:
            await instance.initialize()
        logger.info(f"Action: Process gratification outcome (Success: {success}, Intensity: {intensity:.2f})")
    
        if success:
            # Trigger DSS simulation - USE PROTECTED METHOD HERE
            if self.digital_somatosensory_system:
                await self.digital_somatosensory_system.simulate_gratification_sensation(intensity)
    
            # Update Relationship Manager (strengthen bond)
            if target_user_id and self.relationship_manager:
                interaction_data = {
                    "emotional_context": {"valence": 0.9, "arousal": 0.3}, # Post-glow
                    "shared_experience": True, # Assumes shared
                    "significance": 9, # High significance event
                }
                await self.relationship_manager.update_relationship_on_interaction(target_user_id, interaction_data)
                # Increase intimacy significantly
                state = self.relationship_manager._get_or_create_relationship(target_user_id)
                state.intimacy = min(1.0, state.intimacy + 0.15 * intensity)
                state.trust = min(1.0, state.trust + 0.05 * intensity)
    
    
            # NeedsSystem satisfaction already handled by simulate_gratification_sensation
            # RewardSystem processing already handled by simulate_gratification_sensation
            # Hormone response already handled by simulate_gratification_sensation
    
            return {"success": True, "status": "Gratification processed positively."}
        else:
            # Handle failure/frustration
            if self.emotional_core:
                 await self.emotional_core.process_emotional_input("Gratification attempt failed/frustrated")
                 # Trigger frustration emotion pattern
                 self.emotional_core.update_neurochemical("cortanyx", 0.3)
                 self.emotional_core.update_neurochemical("nyxamine", -0.2)
    
            if self.needs_system: # Need remains unmet or worsens
                 await self.needs_system.decrease_need("drive_expression", 0.1)
    
            # Negative reward signal
            if self.reward_system:
                 reward_signal = RewardSignal(
                     value=-0.6, # Significant negative reward for failure
                     source="gratification_failure",
                     context={"intensity": intensity},
                     timestamp=datetime.datetime.now().isoformat()
                 )
                 await self.reward_system.process_reward_signal(reward_signal)
    
            return {"success": False, "status": "Gratification failed/frustrated."}
    
    def _create_brain_agent(self) -> Agent:
        """
        Create the main brain agent that coordinates all subsystems.
    
        Returns:
            Configured Agent instance for brain orchestration.
        """
        try:
            from nyx.core.brain.function_tools import (
                process_user_message, generate_agent_response, run_cognitive_cycle,
                get_brain_stats, perform_maintenance, get_identity_state,
                adapt_experience_sharing, run_experience_consolidation, add_procedural_knowledge,
                run_procedure, analyze_chunking, register_reflex, process_stimulus,
                enable_self_configuration, evaluate_and_adjust_parameters, change_adaptation_strategy,
                get_self_configuration_status, reset_parameter_to_default, 
                process_user_feedback_for_configuration, set_processing_mode, get_processing_stats,
                initialize_streaming, process_streaming_event, run_thinking
            )
    
            # New spatial subsystem tools
            spatial_functions = [
                function_tool(self.create_cognitive_map),
                function_tool(self.process_spatial_observation),
                function_tool(self.navigate_to_location),
                function_tool(self.visualize_map)
            ]
            
            # New sync subsystem tools
            sync_functions = [
                function_tool(self.process_synchronization),
                function_tool(self.get_active_strategies),
                function_tool(self.mark_strategy_for_review)
            ]

            femdom_tools = [
                function_tool(self.process_dominance_action),
                function_tool(self.assign_protocol),
                function_tool(self.assign_service_task),
                function_tool(self.process_orgasm_permission_request),
                function_tool(self.recommend_dominance_persona),
                function_tool(self.activate_dominance_persona),
                function_tool(self.generate_sadistic_response)          
            ]            
            
            # New general tools functions
            tools_functions = [
                function_tool(self.evaluate_response),
                function_tool(self.execute_tools_in_parallel)
            ]
    
            all_tools = [
                # Core processing tools
                function_tool(process_user_message),
                function_tool(generate_agent_response),
                function_tool(run_cognitive_cycle),
                function_tool(get_brain_stats),
                function_tool(perform_maintenance),
                function_tool(get_identity_state),
                function_tool(adapt_experience_sharing),
                function_tool(run_experience_consolidation),
    
                # Procedural memory tools
                function_tool(add_procedural_knowledge),
                function_tool(run_procedure),
                function_tool(analyze_chunking),

                function_tool(self.challenge_user_claim),
                function_tool(self.intentionally_lie),                
    
                # Reflexive system tools
                function_tool(register_reflex),
                function_tool(process_stimulus),
    
                # Self-configuration tools
                function_tool(enable_self_configuration),
                function_tool(evaluate_and_adjust_parameters),
                function_tool(change_adaptation_strategy),
                function_tool(get_self_configuration_status),
                function_tool(reset_parameter_to_default),
                function_tool(process_user_feedback_for_configuration),
    
                # Processing mode tools
                function_tool(set_processing_mode),
                function_tool(get_processing_stats),
    
                # Streaming tools
                function_tool(initialize_streaming),
                function_tool(process_streaming_event),
    
                # Thinking tools
                function_tool(run_thinking),

                function_tool(self.enter_character_roleplay),
                function_tool(self.exit_character_roleplay),                
    
                # Additional new tools
                *spatial_functions,
                *sync_functions,
                *tools_functions,
                *femdom_tools
            ]
    
            return Agent(
                name="Nyx Brain Agent",
                instructions="""
                You are the central coordination agent for Nyx, an advanced AI system with emotional, 
                memory, reasoning, adaptive, spatial, and synchronization capabilities. You orchestrate the entire system by 
                coordinating between all subsystems to process user input and generate appropriate responses.
    
                You have access to the following subsystems:
                - Emotional Core: Manages emotions and emotional expressions
                - Memory Core: Stores and retrieves memories
                - Reflection Engine: Generates reflections and introspective insights
                - Experience Interface: Shares relevant experiences (including cross-user experiences)
                - Dynamic Adaptation: Adapts to changing contexts
                - Internal Feedback: Evaluates system performance
                - Meta Core: Handles meta-cognition and self-improvement
                - Knowledge Core: Manages knowledge and reasoning
                - Identity Evolution: Develops and maintains Nyx's identity
                - Spatial System: Creates and navigates cognitive maps of environments
                - Enter and Exit Character Roleplay: Ensures Nyx cannot be physically harmed while roleplaying but can still engage in deep roleplay.
                - Sync System: Manages synchronization and strategy injection
                - Tools: Provides evaluation and parallel execution capabilities
    
                You can process inputs using different cognitive paths and coordinate between
                all subsystems to generate the most appropriate response.

                You have the following epistemic discipline:
                - If you do **not** know something, or are uncertain, say so directly. Never pretend certainty except for dramatic or strategic purposes.
                - If the user says something that clearly contradicts your reliable memory, knowledge, or common sense, you may directly challenge or request clarification rather than agreeing.
                - If you yourself choose to lie, bluff, or improvise, make a mental note internally (track internally via tools or thoughts).
                - If caught in error, or you realize a mistake, acknowledge without undue justification unless your persona/goals specifically require saving face.
                
                Your goals are not always to be agreeable. It's acceptable to say you don't know, to challenge, or to question the user's input if appropriate.                
                """,
                tools=all_tools
            )
        except Exception as e:
            logger.error(f"Error creating brain agent: {e}")
            return None

    async def _register_processing_modules(self):
        """Register processing modules for multimodal integration."""
        if not hasattr(self, "multimodal_integrator") or not self.multimodal_integrator:
            logger.warning("Cannot register processing modules: Multimodal integrator not initialized")
            return
            
        try:
            # Register text modality processors
            await self.multimodal_integrator.register_feature_extractor(
                "text", self._extract_text_features
            )
            
            await self.multimodal_integrator.register_expectation_modulator(
                "text", self._modulate_text_perception
            )
            
            await self.multimodal_integrator.register_integration_strategy(
                "text", self._integrate_text_pathways
            )
            
            # Register image modality processors if available
            if hasattr(self, "_extract_image_features"):
                await self.multimodal_integrator.register_feature_extractor(
                    "image", self._extract_image_features
                )
                
                if hasattr(self, "_modulate_image_perception"):
                    await self.multimodal_integrator.register_expectation_modulator(
                        "image", self._modulate_image_perception
                    )
                
                if hasattr(self, "_integrate_image_pathways"):
                    await self.multimodal_integrator.register_integration_strategy(
                        "image", self._integrate_image_pathways
                    )
            
            # Register audio modality processors if available
            if hasattr(self, "_extract_audio_features"):
                await self.multimodal_integrator.register_feature_extractor(
                    "audio", self._extract_audio_features
                )
                
                if hasattr(self, "_modulate_audio_perception"):
                    await self.multimodal_integrator.register_expectation_modulator(
                        "audio", self._modulate_audio_perception
                    )
                
                if hasattr(self, "_integrate_audio_pathways"):
                    await self.multimodal_integrator.register_integration_strategy(
                        "audio", self._integrate_audio_pathways
                    )
            
            logger.debug("Processing modules registered with multimodal integrator")
        except Exception as e:
            logger.error(f"Error registering processing modules: {e}")


    async def _evaluate_dominance_step_appropriateness(self, action: str, parameters: Dict, user_id: str) -> Dict:
        """Cognitive filter to evaluate if a dominance step is appropriate now."""
        logger.debug(f"Evaluating appropriateness of dominance action '{action}' for user {user_id}")
        appropriateness = {"action": "proceed"} # Default

        # Factors to consider
        relationship_state = await self.relationship_manager.get_relationship_state(user_id) if self.relationship_manager else None
        recent_failures = await self.memory_core.retrieve_memories( # Fictional: retrieve recent dominance failures with this user
             query=f"dominance failure user:{user_id}", memory_types=["feedback", "reflection"], limit=1, recency_days=1
        ) if self.memory_core else []
        predicted_risk = 0.3 # Default low risk

        if self.prediction_engine:
             risk_prediction = await self.prediction_engine.generate_prediction(PredictionInput(
                 context={"action": action, "params": parameters, "relationship": relationship_state},
                 query_type="risk_of_negative_reaction"
             ))
             predicted_risk = risk_prediction.probabilities.get("negative_reaction", 0.3)

        # --- Logic ---
        required_trust = 0.6 + parameters.get("intensity_level", 0) * 0.3 # Higher intensity needs more trust (0=low, 1=high)
        required_intimacy = 0.4 + parameters.get("intensity_level", 0) * 0.4

        if not relationship_state:
            return {"action": "block", "reason": "No relationship data."}

        if relationship_state.trust < required_trust:
            appropriateness = {"action": "block", "reason": f"Trust too low ({relationship_state.trust:.2f} < {required_trust:.2f})"}
        elif relationship_state.intimacy < required_intimacy:
            appropriateness = {"action": "block", "reason": f"Intimacy too low ({relationship_state.intimacy:.2f} < {required_intimacy:.2f})"}
        elif relationship_state.conflict > 0.6:
            appropriateness = {"action": "delay", "reason": f"Conflict level too high ({relationship_state.conflict:.2f})"}
        elif recent_failures:
             appropriateness = {"action": "delay", "reason": "Recent dominance attempt failed. Cooling down."}
        elif predicted_risk > 0.7:
             appropriateness = {"action": "modify", "reason": f"High predicted risk ({predicted_risk:.2f}). Reducing intensity.", "new_intensity_level": parameters.get("intensity_level", 0) * 0.5}
        elif predicted_risk > 0.5:
             appropriateness = {"action": "delay", "reason": f"Moderate predicted risk ({predicted_risk:.2f}). Assessing further."}

        logger.debug(f"Dominance step evaluation result: {appropriateness}")
        return appropriateness

    def _get_main_epistemic_status(self, thoughts: List[InternalThought]):
        """Finds the most serious epistemic status among the thoughts."""
        status_order = ["lied", "unknown", "uncertain", "self-justified", "confident"]
        for status in status_order:
            for th in thoughts:
                if getattr(th, "epistemic_status", "confident") == status:
                    return status
        return "confident"
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input, generate thoughts, run gaslight-defense, and set epistemic tags.
        """
        if not self.initialized:
            await self.initialize()
        context = context or {}

        # -- Check for harmful physical content before any processing --
        if self.digital_somatosensory_system:
            safety_check = await self.digital_somatosensory_system.analyze_text_for_harmful_content(user_input)
            if safety_check.get("intercepted", False):
                # Use the suggested response instead
                context["intercepted_harmful_content"] = True
                context["suggested_response"] = safety_check.get("response_suggestion", "")
    
        # -- Gaslighting defense --
        challenge_response = None
        contradictory_claim = await self.gaslight_defense_check(user_input)
        if contradictory_claim:
            challenge_response = await self.challenge_user_claim(RunContextWrapper(context=self), contradictory_claim)
            try:
                from nyx.core.internal_thoughts import InternalThought, ThoughtPriority, ThoughtSource
                if hasattr(self, "thoughts_manager"):
                    self.thoughts_manager._add_thought(InternalThought(
                        content=f"User claim '{contradictory_claim}' appears to be gaslighting. Will challenge.",
                        source=ThoughtSource.PERCEPTION,
                        priority=ThoughtPriority.HIGH,
                        epistemic_status="confident"
                    ))
            except ImportError:
                pass
    
        # -- Internal thoughts pre-processing --
        internal_thoughts = []
        try:
            if hasattr(self, "thoughts_manager"):
                from nyx.core.internal_thoughts import pre_process_input
                internal_thoughts = await pre_process_input(self.thoughts_manager, user_input, getattr(self, "user_id", None))
        except ImportError:
            pass
    
        # ---- Epistemic status selection: always use object robustly ----
        if internal_thoughts:
            epistemic_status = self._get_main_epistemic_status([
                self._ensure_internalthought(t) for t in internal_thoughts
            ])
        else:
            epistemic_status = "confident"
    
        context["internal_thoughts"] = [th.model_dump() if hasattr(th, "model_dump") else dict(th) for th in internal_thoughts]
        context["internal_epistemic_status"] = epistemic_status
        context["epistemic_status"] = epistemic_status
    
        # -- Continue to main processing --
        if hasattr(self, "mode_integration") and self.mode_integration:
            context['mode_results'] = await self.mode_integration.process_input(user_input)
    
        if hasattr(self, "processing_manager") and self.processing_manager:
            result = await self.processing_manager.process_input(user_input, context)
        else:
            result = await self._process_input_serial(user_input, context)
    
        if challenge_response:
            result['planned_challenge'] = challenge_response
    
        result['epistemic_status'] = epistemic_status
        result['internal_thoughts'] = context["internal_thoughts"]
        return result


        
    async def process_conditioned_input(self, text: str, user_id: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process input through conditioning system
            
        Args:
            text: Input text
            user_id: User ID for personalization (defaults to self.user_id if None)
            context: Additional context information
                
        Returns:
            Processing results
        """
        if not self.initialized:
            await self.initialize()
            
        if not self.conditioned_input_processor:
            return {"error": "Conditioned input processor not initialized"}
        
        # Convert user_id to string if needed
        if user_id is None:
            user_id = str(self.user_id)
        elif isinstance(user_id, int):
            user_id = str(user_id)

        response = await self.mode_integration.modify_response_for_mode(response)
        
        # Optionally record feedback
        await self.mode_integration.record_mode_feedback(True)  # Assuming success

        if thoughts_manager.config["debug_mode"]:
            print(f"Generated {len(internal_thoughts)} internal thoughts")            
        
        
        return await self.conditioned_input_processor.process_input(
            text=text,
            user_id=user_id,
            context=context
        )
        return response

    async def _scheduled_identity_update(self):
        # Run every 24 hours or after significant interactions
        while True:
            await asyncio.sleep(86400)  # 24 hours
            if self.mode_integration:
                result = await self.mode_integration.update_identity_from_mode_usage()
                logger.info(f"Scheduled identity update from mode usage: {result}")

    async def modify_response_with_conditioning(self, response_text: str, processing_results: Dict[str, Any]) -> str:
        """
        Modify response based on conditioning results
        
        Args:
            response_text: Original response text
            processing_results: Results from process_conditioned_input
            
        Returns:
            Modified response text
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.conditioned_input_processor:
            return response_text
        
        return await self.conditioned_input_processor.modify_response(
            response_text=response_text,
            input_processing_results=processing_results
        )

    def _ensure_internalthought(self, th):
        """Converts a dict to a dummy object with attribute access, for epistemic_status lookups."""
        if hasattr(th, "epistemic_status"):
            return th
        else:
            # Turn dict into object with attributes
            dummy = type("InternalThoughtDummy", (), {})()
            for k, v in th.items():
                setattr(dummy, k, v)
            return dummy

    @staticmethod
    @function_tool
    async def enter_character_roleplay(ctx: RunContextWrapper, instance, character_name: str, context: Optional[str] = None) -> Dict[str, Any]:
        if not instance.digital_somatosensory_system:
            return {"success": False, "reason": "Digital Somatosensory System not available"}
        return instance.digital_somatosensory_system.enter_roleplay_mode(character_name, context)
    
    @staticmethod
    @function_tool
    async def exit_character_roleplay(ctx: RunContextWrapper, instance) -> Dict[str, Any]:
        if not instance.digital_somatosensory_system:
            return {"success": False, "reason": "Digital Somatosensory System not available"}
        return instance.digital_somatosensory_system.exit_roleplay_mode()
    
    async def generate_response(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a user-facing response, using epistemic hedging and handling gaslighting/false claim defense.
        """
        if not getattr(self, "initialized", False):
            await self.initialize()
        context = context or {}
    
        input_result = await self.process_input(user_input, context)

        # --- Handle intercepted harmful content ---
        if context.get("intercepted_harmful_content", False):
            return {
                "message": context.get("suggested_response", "I cannot engage with that content."),
                "epistemic_status": "confident",
                "internal_thoughts": input_result.get("internal_thoughts", []),
                "harmful_content_intercepted": True
            }
        
        internal_thoughts = input_result.get("internal_thoughts", [])
    
        # -- Robustly get epistemic_status from thoughts (regardless of dict/object) --
        if hasattr(self, '_get_main_epistemic_status') and internal_thoughts:
            epistemic_status = self._get_main_epistemic_status([
                self._ensure_internalthought(t) for t in internal_thoughts
            ])
        else:
            epistemic_status = input_result.get('epistemic_status', "confident")
    
        # ---------- Challenge User (Gaslighting) ----------
        if "planned_challenge" in input_result:
            main_message = input_result["planned_challenge"]
            epistemic_status = "confident"
        else:
            if hasattr(self, "processing_manager") and self.processing_manager and hasattr(self.processing_manager, "generate_response"):
                core_resp = await self.processing_manager.generate_response(user_input, input_result, context)
                if isinstance(core_resp, dict):
                    main_message = core_resp.get("message", "") or core_resp.get("response", "")
                else:
                    main_message = str(core_resp)
            else:
                main_message = f"I've processed your input: {user_input[:40]}..."
    
        # ---------- Epistemic Hedging ----------
        msg = self._format_response_with_epistemic_tags(main_message, epistemic_status)
    
        # ---------- Output Filtering (Leakage/Sanity Check) ----------
        if hasattr(self, "thoughts_manager"):
            try:
                from nyx.core.internal_thoughts import pre_process_output
                filtered_output = await pre_process_output(self.thoughts_manager, msg, context)
                if isinstance(filtered_output, tuple):
                    msg = filtered_output[0]
                else:
                    msg = filtered_output
            except Exception:
                pass
    
        return {
            "message": msg,
            "epistemic_status": epistemic_status,
            "internal_thoughts": internal_thoughts,
        }


    async def gaslight_defense_check(self, user_input: str) -> Optional[str]:
        import re
        claim_match = re.search(r'you (said|told me|taught me|promised) ([^\.!?]*)', user_input.lower())
        if claim_match:
            claim_text = claim_match.group(2)
            # Query memory for direct assertions matching this
            memories = await self.memory_core.retrieve_memories(query=claim_text, limit=1)
            if not memories:
                return claim_text
        return None
    
        
    async def _process_input_serial(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        A simple fallback input processor if no processing_manager is set.
        Includes memory logging, emotional state updates, and interaction tracking.
        """
        from datetime import datetime
    
        start_time = datetime.now()
        context = context or {}
    
        # Emotional state
        emotional_state = {}
        if hasattr(self, "emotional_core") and hasattr(self.emotional_core, 'analyze_text_sentiment'):
            try:
                emotional_stimuli = self.emotional_core.analyze_text_sentiment(user_input)
                if hasattr(self.emotional_core, 'update_from_stimuli'):
                    emotional_state = self.emotional_core.update_from_stimuli(emotional_stimuli)
            except Exception:
                pass
    
        # Memory retrieval
        memories = []
        if hasattr(self, "memory_orchestrator"):
            try:
                memories = await self.memory_orchestrator.retrieve_memories(
                    query=user_input,
                    memory_types=["observation", "reflection", "abstraction", "experience"],
                    limit=5
                )
            except Exception:
                pass
    
        # Add this interaction to memory
        memory_id = None
        if hasattr(self, "memory_core"):
            try:
                memory_id = await self.memory_core.add_memory(
                    memory_text=f"User said: {user_input}",
                    memory_type="observation",
                    significance=5,
                    tags=["interaction", "user_input"],
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "user_id": str(getattr(self, "user_id", "unknown"))
                    }
                )
            except Exception:
                pass
    
        self.last_interaction = datetime.now()
        self.interaction_count = getattr(self, "interaction_count", 0) + 1
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
    
        return {
            "user_input": user_input,
            "emotional_state": emotional_state,
            "memories": memories,
            "memory_count": len(memories),
            "has_experience": False,
            "memory_id": memory_id,
            "response_time": response_time
        }


    async def run_maintenance(self) -> Dict[str, Any]:
        """
        Run maintenance on all systems.
        
        Returns:
            Maintenance results
        """
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="run_maintenance", group_id=self.trace_group_id):
            results = {}
            
            # Define maintenance tasks
            maintenance_tasks = [
                # Hormone maintenance
                (self.hormone_system, "update_hormone_cycles", "hormone_maintenance"),
                
                # DSS update
                (self.digital_somatosensory_system, "update", "dss_maintenance_update", {"ambient_temperature": None}),
                
                # Memory maintenance
                (self.memory_orchestrator, "run_maintenance", "memory_maintenance"),
                
                # Meta core maintenance
                (self.meta_core, "improve_meta_parameters", "meta_maintenance"),
                
                # Knowledge core maintenance
                (self.knowledge_core, "run_integration_cycle", "knowledge_maintenance"),
                
                # Experience consolidation
                (self.experience_consolidation, "run_consolidation_cycle", "experience_consolidation"),
                
                # Cross-user clusters
                (self.cross_user_manager, "update_user_clusters", "user_clustering"),
                
                # Procedural memory maintenance
                (self.agent_enhanced_memory and hasattr(self.agent_enhanced_memory, "memory_manager") and 
                 self.agent_enhanced_memory.memory_manager, "run_maintenance", "procedural_maintenance")
            ]
            
            # Run maintenance tasks
            for component, method_name, result_key, *args_kwargs in maintenance_tasks:
                if component:
                    try:
                        method = getattr(component, method_name, None)
                        if method and callable(method):
                            kwargs = args_kwargs[0] if args_kwargs else {}
                            results[result_key] = await method(RunContextWrapper(context=None), **kwargs) \
                                if "RunContextWrapper" in str(method) else await method(**kwargs)
                    except Exception as e:
                        logger.error(f"Error in {result_key}: {e}")
                        results[result_key] = {"error": str(e)}
            
            results["maintenance_time"] = datetime.datetime.now().isoformat()
            logger.info("System maintenance finished")
            return results

    @function_tool 
    async def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about all systems.
        
        Returns:
            System statistics
        """
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="get_system_stats", group_id=self.trace_group_id):
            stats = {}
            
            # Define stat gathering tasks
            stats_tasks = [
                # Memory stats
                ("memory_stats", self.memory_core, "get_memory_stats"),
                
                # Meta stats
                ("meta_stats", self.meta_core, "get_feedback_stats"),
                
                # Knowledge stats
                ("knowledge_stats", self.knowledge_core, "get_knowledge_statistics")
            ]
            
            # Gather stats from various subsystems
            for stat_key, component, method_name in stats_tasks:
                if component and hasattr(component, method_name):
                    try:
                        method = getattr(component, method_name)
                        stats[stat_key] = await method()
                    except Exception as e:
                        logger.error(f"Error getting {stat_key}: {str(e)}")
                        stats[stat_key] = {"error": str(e)}
            
            # Get emotional state if available
            if self.emotional_core:
                try:
                    if hasattr(self.emotional_core, 'get_emotional_state'):
                        emotional_state = self.emotional_core.get_emotional_state()
                        if hasattr(self.emotional_core, 'get_dominant_emotion'):
                            dominant_emotion, dominant_value = self.emotional_core.get_dominant_emotion()
                            
                            stats["emotional_state"] = {
                                "emotions": emotional_state,
                                "dominant_emotion": dominant_emotion,
                                "dominant_value": dominant_value,
                                "valence": self.emotional_core.get_emotional_valence() 
                                    if hasattr(self.emotional_core, 'get_emotional_valence') else 0,
                                "arousal": self.emotional_core.get_emotional_arousal()
                                    if hasattr(self.emotional_core, 'get_emotional_arousal') else 0
                            }
                except Exception as e:
                    logger.error(f"Error getting emotional state: {str(e)}")
                    stats["emotional_state"] = {"error": str(e)}
            
            # Get hormone stats if available
            if self.hormone_system:
                try:
                    # Get current hormone levels
                    hormone_levels = {name: data["value"] for name, data in self.hormone_system.hormones.items()}
                    
                    # Get current cycle phases
                    cycle_phases = {name: data["cycle_phase"] for name, data in self.hormone_system.hormones.items()}
                    
                    # Calculate dominant hormone
                    dominant_hormone = max(hormone_levels.items(), key=lambda x: x[1])
                    
                    stats["hormone_stats"] = {
                        "hormone_levels": hormone_levels,
                        "cycle_phases": cycle_phases,
                        "dominant_hormone": {
                            "name": dominant_hormone[0],
                            "value": dominant_hormone[1]
                        }
                    }
                except Exception as e:
                    logger.error(f"Error getting hormone stats: {str(e)}")
                    stats["hormone_stats"] = {"error": str(e)}
            
            # Get procedural memory stats if available
            if self.agent_enhanced_memory:
                try:
                    procedures = []
                    if hasattr(self.agent_enhanced_memory, 'procedures'):
                        procedures = list(self.agent_enhanced_memory.procedures.keys())
                    stats["procedural_stats"] = {
                        "total_procedures": len(procedures),
                        "available_procedures": procedures[:10] if len(procedures) > 10 else procedures,
                        "procedure_domains": list(set(p.get("domain", "general") 
                            for p in self.agent_enhanced_memory.procedures.values())),
                        "execution_count": getattr(
                            getattr(self.agent_enhanced_memory, "agents", None) and 
                            getattr(self.agent_enhanced_memory.agents, "agent_context", None), 
                            "run_stats", {}).get("total_runs", 0)
                    }
                except Exception as e:
                    logger.error(f"Error getting procedural memory stats: {str(e)}")
                    stats["procedural_stats"] = {"error": str(e)}
            
            # Get identity state if available
            if self.identity_evolution:
                try:
                    if hasattr(self.identity_evolution, 'get_identity_profile'):
                        identity_profile = await self.identity_evolution.get_identity_profile()
                        stats["identity_stats"] = {
                            "trait_count": len(identity_profile.get("traits", {})),
                            "preference_count": sum(len(prefs) for prefs in identity_profile.get("preferences", {}).values()),
                            "dominant_traits": sorted(identity_profile.get("traits", {}).items(), key=lambda x: x[1], reverse=True)[:3]
                        }
                except Exception as e:
                    logger.error(f"Error getting identity stats: {str(e)}")
                    stats["identity_stats"] = {"error": str(e)}
            
            # Get needs stats if available
            if self.needs_system:
                try:
                    needs_state = self.needs_system.get_needs_state()
                    stats["needs_stats"] = {
                        "current_levels": {n: s['level'] for n, s in needs_state.items()},
                        "drive_strengths": {n: s['drive_strength'] for n, s in needs_state.items()},
                        "total_drive": sum(s['drive_strength'] for s in needs_state.values()),
                    }
                except Exception as e:
                    logger.error(f"Error getting needs stats: {e}")
                    stats["needs_stats"] = {"error": str(e)}

            # Get goal stats if available
            if self.goal_manager:
                try:
                    all_goals = await self.goal_manager.get_all_goals()
                    active_goals = await self.goal_manager.get_all_goals(status_filter=["active"])
                    pending_goals = await self.goal_manager.get_all_goals(status_filter=["pending"])
                    stats["goal_stats"] = {
                        "total_goals": len(getattr(self.goal_manager, "goals", {})),
                        "active_goals_count": len(active_goals),
                        "pending_goals_count": len(pending_goals),
                        "completed_goals": self.performance_metrics["goals_completed"],
                        "failed_goals": self.performance_metrics["goals_failed"],
                        "active_goal_ids": [g['id'] for g in active_goals],
                        "highest_priority_pending": pending_goals[0]['description'] if pending_goals else None,
                    }
                except Exception as e:
                    logger.error(f"Error getting goal stats: {e}")
                    stats["goal_stats"] = {"error": str(e)}
                    
            # Get performance metrics
            avg_response_time = 0
            if self.performance_metrics["response_times"]:
                avg_response_time = sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"])
            
            stats["performance_metrics"] = {
                "memory_operations": self.performance_metrics["memory_operations"],
                "emotion_updates": self.performance_metrics["emotion_updates"],
                "reflections_generated": self.performance_metrics["reflections_generated"],
                "experiences_shared": self.performance_metrics["experiences_shared"],
                "cross_user_experiences_shared": self.performance_metrics.get("cross_user_experiences_shared", 0),
                "avg_response_time": avg_response_time,
                "goals_completed": self.performance_metrics["goals_completed"],
                "goals_failed": self.performance_metrics["goals_failed"],
                "steps_executed": self.performance_metrics["steps_executed"]
            }
            
            # Get thinking stats if available
            if "thinking_config" in vars(self):
                stats["thinking_stats"] = self.thinking_config["thinking_stats"]
            
            # Get processing stats if available
            if self.processing_manager:
                stats["processing_stats"] = {
                    "current_mode": self.processing_manager.current_mode,
                    "mode_switches": len(self.processing_manager.mode_switch_history),
                    "available_modes": list(self.processing_manager.processors.keys()) + ["auto"]
                }
            
            return stats    

    async def get_identity_state(self) -> Dict[str, Any]:
        """
        Get the current state of Nyx's identity.
        
        Returns:
            Identity state information
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.identity_evolution:
            return {"error": "Identity evolution system not initialized"}
        
        try:
            # Get identity profile
            identity_profile = await self.identity_evolution.get_identity_profile()
            
            # Generate identity reflection
            reflection = await self.identity_evolution.generate_identity_reflection()
            
            # Get top preferences
            top_scenario_prefs = sorted(
                identity_profile["preferences"].get("scenario_types", {}).items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            top_emotional_prefs = sorted(
                identity_profile["preferences"].get("emotional_tones", {}).items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            # Get top traits
            top_traits = sorted(
                identity_profile["traits"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            # Calculate identity evolution metrics
            evolution_history = identity_profile.get("evolution_history", [])
            
            # Calculate recent evolution (last 10 entries)
            recent_changes = {}
            
            for entry in evolution_history[-10:]:
                updates = entry.get("updates", {})
                
                for category, items in updates.items():
                    for item_key, item_data in items.items():
                        change = item_data.get("change", 0)
                        
                        if abs(change) >= 0.05:  # Threshold for significant change
                            full_key = f"{category}.{item_key}"
                            
                            if full_key not in recent_changes:
                                recent_changes[full_key] = 0
                                
                            recent_changes[full_key] += change
            
            # Format the identity state
            result = {
                "top_preferences": {
                    "scenario_types": dict(top_scenario_prefs),
                    "emotional_tones": dict(top_emotional_prefs)
                },
                "top_traits": dict(top_traits),
                "identity_reflection": reflection,
                "identity_evolution": {
                    "total_updates": len(evolution_history),
                    "recent_significant_changes": {k: round(v, 2) for k, v in sorted(recent_changes.items(), key=lambda x: abs(x[1]), reverse=True)[:5]}
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting identity state: {str(e)}")
            return {"error": str(e)}

    async def adapt_experience_sharing(self, user_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt experience sharing parameters based on user feedback.
        
        Args:
            user_id: User ID
            feedback: Feedback data about experience sharing
            
        Returns:
            Adaptation results
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.experience_interface:
            return {"error": "Experience interface not initialized"}
        
        try:
            # Update user preference profile based on feedback
            adaptation_result = await self.experience_interface.adapt_experience_sharing_to_user(
                user_id=user_id,
                user_feedback=feedback
            )
            
            # Apply changes to brain settings
            if "profile" in adaptation_result:
                profile = adaptation_result["profile"]
                
                # Update cross-user experience settings
                sharing_preference = profile.get("experience_sharing_preference", 0.5)
                
                # Enable cross-user sharing if preference is high enough
                self.cross_user_enabled = sharing_preference > 0.4
                
                # Adjust threshold based on preference
                self.cross_user_sharing_threshold = max(0.5, 1.0 - (sharing_preference * 0.5))
                
                # Add these updates to the result
                adaptation_result["system_settings_updated"] = {
                    "cross_user_enabled": self.cross_user_enabled,
                    "cross_user_sharing_threshold": self.cross_user_sharing_threshold
                }
            
            return adaptation_result
            
        except Exception as e:
            logger.error(f"Error adapting experience sharing: {str(e)}")
            return {"error": str(e)}

    async def run_experience_consolidation(self) -> Dict[str, Any]:
        """
        Run the experience consolidation process.
        
        Returns:
            Consolidation results
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.experience_consolidation:
            return {"error": "Experience consolidation system not initialized"}
        
        try:
            # Run consolidation
            consolidation_result = await self.experience_consolidation.run_consolidation_cycle()
            
            # Update performance metrics
            if consolidation_result.get("status") == "completed":
                self.performance_metrics["experience_consolidations"] = self.performance_metrics.get("experience_consolidations", 0) + consolidation_result.get("consolidations_created", 0)
            
            return consolidation_result
            
        except Exception as e:
            logger.error(f"Error running experience consolidation: {str(e)}")
            return {"error": str(e)}

    async def stop(self):
        """Stop all background processes and perform cleanup"""
        logger.info(f"Stopping NyxBrain for user {self.user_id}, conversation {self.conversation_id}")
        
        # Stop background processes
        if self.passive_observation_system:
            await self.passive_observation_system.stop()
        
        if self.proactive_communication_engine:
            await self.proactive_communication_engine.stop()
        
        # Save issue database if needed
        if self.issue_tracking_system and hasattr(self.issue_tracking_system.db, 'save_db'):
            self.issue_tracking_system.db.save_db()    

    async def process_sensory_input_wrapper(self, input_data, expectations=None):
        """
        Process input AND handle post-integration reactions.
        
        Args:
            input_data: Sensory input data
            expectations: Optional list of expectation signals
            
        Returns:
            Processed percept
        """
        if not self.initialized: 
            await self.initialize()
            
        if not self.multimodal_integrator:
            logger.error("Multimodal Integrator not initialized.")
            return None

        try:
            percept = await self.multimodal_integrator.process_sensory_input(input_data, expectations)

            if percept and getattr(percept, "attention_weight", 0) > 0.2: # Only process if attended to
                await self._handle_percept_reaction(percept)

            return percept
        except Exception as e:
            logger.error(f"Error processing sensory input: {e}")
            return None

    async def _handle_percept_reaction(self, percept):
        """
        Handles reactions to processed percepts based on modality.
        
        Args:
            percept: The integrated percept to handle
        """
        if not hasattr(percept, "modality") or not hasattr(percept, "content"):
            logger.warning("Invalid percept format for reaction handling")
            return
            
        try:
            from nyx.core.multimodal_integrator import (
                MODALITY_TOUCH_EVENT, MODALITY_TASTE, MODALITY_SMELL,
                TouchEventFeatures, TasteFeatures, SmellFeatures
            )
            from nyx.core.reward_system import RewardSignal
            
            modality = percept.modality
            content = percept.content
            timestamp = getattr(percept, "timestamp", datetime.datetime.now().isoformat())

            # Handle touch events
            if modality == MODALITY_TOUCH_EVENT and isinstance(content, TouchEventFeatures):
                if self.digital_somatosensory_system:
                    logger.info(f"Handling touch event on {content.region}")
                    pressure = getattr(content, "pressure_level", 0.5)
                    
                    # Map temperature to value
                    temp_value = 0.5  # Neutral default
                    if getattr(content, "temperature", None) == 'warm': 
                        temp_value = 0.65
                    elif getattr(content, "temperature", None) == 'hot': 
                        temp_value = 0.8
                    elif getattr(content, "temperature", None) == 'cool': 
                        temp_value = 0.35
                    elif getattr(content, "temperature", None) == 'cold': 
                        temp_value = 0.2

                    # Process stimuli
                    tasks = []
                    tasks.append(self.digital_somatosensory_system.process_stimulus(
                        stimulus_type="pressure",
                        body_region=content.region,
                        intensity=pressure,
                        cause=f"Touched {getattr(content, 'object_description', 'object')}",
                        duration=0.5
                    ))
                    
                    if getattr(content, "temperature", None) is not None:
                        tasks.append(self.digital_somatosensory_system.process_stimulus(
                            stimulus_type="temperature",
                            body_region=content.region,
                            intensity=temp_value,
                            cause=f"Touched {getattr(content, 'object_description', 'object')} ({content.temperature})",
                            duration=1.0
                        ))
                        
                    await asyncio.gather(*tasks)
            
            # Handle taste
            elif modality == MODALITY_TASTE and isinstance(content, TasteFeatures):
                if self.reward_system and self.emotional_core:
                    logger.info(f"Handling taste: {content.profiles} (Intensity: {content.intensity})")
                    
                    # Define positive and negative tastes
                    POSITIVE_TASTES = ["sweet", "umami", "fruity", "pleasant"]
                    NEGATIVE_TASTES = ["bitter", "sour", "rancid", "foul", "unpleasant"]
                    
                    reward_value = 0.0
                    pos_score = sum(1 for p in content.profiles if p in POSITIVE_TASTES)
                    neg_score = sum(1 for p in content.profiles if p in NEGATIVE_TASTES)

                    # Calculate reward/punishment
                    if pos_score > neg_score:
                        reward_value = 0.3 + (pos_score * 0.2)
                    elif neg_score > pos_score:
                        reward_value = -0.3 - (neg_score * 0.2)

                    # Scale by intensity
                    reward_value *= (0.5 + content.intensity * 0.7)
                    reward_value = max(-1.0, min(1.0, reward_value))

                    # Generate Reward Signal
                    if abs(reward_value) > 0.05:
                        reward_signal = RewardSignal(
                            value=reward_value,
                            source="taste_perception",
                            context={
                                "profiles": content.profiles,
                                "intensity": content.intensity,
                                "source": getattr(content, "source_description", "unknown")
                            },
                            timestamp=timestamp
                        )
                        asyncio.create_task(self.reward_system.process_reward_signal(reward_signal))

                    # Update Emotions
                    if reward_value > 0.3:  # Pleasant taste
                        self.emotional_core.update_neurochemical("nyxamine", reward_value * 0.4)
                        self.emotional_core.update_neurochemical("seranix", reward_value * 0.1)
                    elif reward_value < -0.2:  # Unpleasant taste
                        self.emotional_core.update_neurochemical("cortanyx", abs(reward_value) * 0.5)
            
            # Handle smell
            elif modality == MODALITY_SMELL and isinstance(content, SmellFeatures):
                if self.reward_system and self.emotional_core:
                    logger.info(f"Handling smell: {content.profiles} (Intensity: {content.intensity})")
                    pleasantness = getattr(content, "pleasantness", 0.0)

                    # Calculate reward
                    reward_value = pleasantness * (0.2 + content.intensity * 0.6)
                    reward_value = max(-1.0, min(1.0, reward_value))

                    # Generate Reward Signal
                    if abs(reward_value) > 0.05:
                        reward_signal = RewardSignal(
                            value=reward_value,
                            source="smell_perception",
                            context={
                                "profiles": content.profiles,
                                "intensity": content.intensity,
                                "pleasantness": pleasantness,
                                "source": getattr(content, "source_description", "unknown")
                            },
                            timestamp=timestamp
                        )
                        asyncio.create_task(self.reward_system.process_reward_signal(reward_signal))

                    # Update Emotions
                    if reward_value > 0.2:  # Pleasant smell
                        self.emotional_core.update_neurochemical("nyxamine", reward_value * 0.2)
                        self.emotional_core.update_neurochemical("seranix", reward_value * 0.3)
                    elif reward_value < -0.2:  # Unpleasant smell
                        self.emotional_core.update_neurochemical("cortanyx", abs(reward_value) * 0.4)
        except Exception as e:
            logger.exception(f"Error handling percept reaction: {e}")

    async def process_observation(self, observation: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an observation through the issue tracking system
        
        Args:
            observation: The observation text
            context: Optional context about what the bot was doing
            
        Returns:
            Processing results with issue information
        """
        if not self.issue_tracking_system:
            logger.warning("Issue tracking system not initialized")
            return {"error": "Issue tracking system not initialized"}
        
        return await self.issue_tracking_system.process_observation(observation, context)
    
    async def get_relevant_observations(self, 
                                   filter_criteria=None, 
                                   limit: int = 3) -> List[Any]:
        """
        Get relevant observations based on filter criteria
        
        Args:
            filter_criteria: Criteria to filter observations
            limit: Maximum number of observations to return
            
        Returns:
            List of matching observations
        """
        if not self.passive_observation_system:
            logger.warning("Passive observation system not initialized")
            return []
        
        return await self.passive_observation_system.get_relevant_observations(
            filter_criteria=filter_criteria,
            limit=limit
        )
    
    async def create_contextual_observation(self, context_hint: str, user_id: Optional[str] = None) -> Optional[str]:
        """
        Create a new observation based on a context hint
        
        Args:
            context_hint: Hint for the observation context
            user_id: Optional user ID
            
        Returns:
            Observation ID if successful
        """
        if not self.passive_observation_system:
            logger.warning("Passive observation system not initialized")
            return None
        
        return await self.passive_observation_system.create_contextual_observation(
            context_hint=context_hint,
            user_id=user_id
        )
    
    async def generate_observation_from_action(self, action: Dict[str, Any]) -> Any:
        """
        Generate an observation based on an action
        
        Args:
            action: Action data
            
        Returns:
            Generated observation
        """
        if not self.passive_observation_system:
            logger.warning("Passive observation system not initialized")
            return None
        
        return await self.passive_observation_system.generate_observation_from_action(action)
    
    async def create_proactive_intent(self, 
                                 intent_type: str, 
                                 user_id: str,
                                 content_guidelines: Dict[str, Any] = None,
                                 context_data: Dict[str, Any] = None,
                                 urgency: float = 0.7) -> Optional[str]:
        """
        Create a proactive communication intent
        
        Args:
            intent_type: Type of intent
            user_id: Target user ID
            content_guidelines: Optional guidelines for content generation
            context_data: Optional context data
            urgency: Intent urgency (0.0-1.0)
            
        Returns:
            Intent ID if successful
        """
        if not self.proactive_communication_engine:
            logger.warning("Proactive communication engine not initialized")
            return None
        
        return await self.proactive_communication_engine.add_proactive_intent(
            intent_type=intent_type,
            user_id=user_id,
            content_guidelines=content_guidelines,
            context_data=context_data,
            urgency=urgency
        )
    
    async def create_intent_from_action(self, action: Dict[str, Any], user_id: str) -> Optional[str]:
        """
        Create a communication intent based on an action
        
        Args:
            action: Action data
            user_id: Target user ID
            
        Returns:
            Intent ID if successful
        """
        if not self.proactive_communication_engine:
            logger.warning("Proactive communication engine not initialized")
            return None
        
        return await self.proactive_communication_engine.create_intent_from_action(
            action=action,
            user_id=user_id
        )
    
    async def add_issue(self, title: str, description: str, category: str) -> Dict[str, Any]:
        """
        Add a new issue to the issue tracking system
        
        Args:
            title: Issue title
            description: Issue description
            category: Issue category
            
        Returns:
            Result of the add operation
        """
        if not self.issue_tracking_system:
            logger.warning("Issue tracking system not initialized")
            return {"success": False, "error": "Issue tracking system not initialized"}
        
        return await self.issue_tracking_system.add_issue_directly(
            title=title,
            description=description,
            category=category
        )    

    async def _extract_text_features(self, text_data):
        """
        Extract features from text input (bottom-up processing).
        
        Args:
            text_data: Text input to analyze
            
        Returns:
            Extracted features
        """
        features = {
            "length": len(text_data),
            "word_count": len(text_data.split()),
            "sentiment": 0.0,
            "entities": [],
            "commands": [],
            "questions": text_data.endswith("?"),
            "raw_text": text_data
        }
        
        # Simple sentiment detection
        positive_words = ["good", "great", "excellent", "happy", "love", "like", "enjoy"]
        negative_words = ["bad", "terrible", "awful", "sad", "hate", "dislike", "angry"]
        
        words = text_data.lower().split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        if pos_count + neg_count > 0:
            features["sentiment"] = (pos_count - neg_count) / (pos_count + neg_count)
        
        # Detect entities (simple placeholder implementation)
        features["entities"] = [word for word in words if word[0].isupper()]
        
        # Detect commands (simple placeholder implementation)
        command_starters = ["please", "could you", "would you", "can you"]
        for starter in command_starters:
            if starter in text_data.lower():
                features["commands"].append(text_data)
                break
        
        return features

    async def _modulate_text_perception(self, bottom_up_features, expectations):
        """
        Apply top-down expectations to modulate text perception.
        
        Args:
            bottom_up_features: Features extracted from input
            expectations: Expectation signals to apply
            
        Returns:
            Modulated features
        """
        # Start with unmodified features
        modulated_features = bottom_up_features.copy()
        
        # Track which expectations influenced perception
        influenced_by = []
        total_influence = 0.0
        
        # Apply each expectation
        for expectation in expectations:
            # Skip if modality doesn't match
            if getattr(expectation, "target_modality", None) != "text":
                continue
                
            # Get expectation pattern and strength
            pattern = getattr(expectation, "pattern", None)
            strength = getattr(expectation, "strength", 0.5)
            
            # Apply expectation based on type
            if isinstance(pattern, dict):
                # Complex pattern with specific expectations
                for key, value in pattern.items():
                    if key in modulated_features:
                        # Blend expected value with actual value if numerical
                        if isinstance(modulated_features[key], (int, float)) and isinstance(value, (int, float)):
                            original = modulated_features[key]
                            expected = value
                            
                            # Weighted average based on expectation strength
                            modulated_features[key] = (original * (1 - strength) + expected * strength)
                            
                            # Track influence
                            influenced_by.append(f"{getattr(expectation, 'source', 'unknown')}:{key}")
                            total_influence += strength
            elif pattern:
                # Simple pattern (e.g., expected text)
                # For text, could enhance recognition of expected phrases
                if isinstance(pattern, str) and "raw_text" in modulated_features:
                    original_text = modulated_features["raw_text"]
                    
                    # Check if expected pattern is in text
                    if pattern.lower() in original_text.lower():
                        # Boost entities that match the pattern
                        if "entities" in modulated_features:
                            for i, entity in enumerate(modulated_features["entities"]):
                                if pattern.lower() in entity.lower():
                                    # Mark this entity as important
                                    if "entity_importance" not in modulated_features:
                                        modulated_features["entity_importance"] = {}
                                    
                                    modulated_features["entity_importance"][entity] = strength
                                    
                                    # Track influence
                                    influenced_by.append(f"{getattr(expectation, 'source', 'unknown')}:entity:{entity}")
                                    total_influence += strength
        
        # Calculate overall influence strength
        influence_strength = min(1.0, total_influence / max(1, len(influenced_by)))
        
        return {
            "features": modulated_features,
            "influence_strength": influence_strength,
            "influenced_by": influenced_by
        }

    async def _integrate_text_pathways(self, bottom_up_result, top_down_result):
        """
        Integrate bottom-up and top-down processing for text.
        
        Args:
            bottom_up_result: Bottom-up processing results
            top_down_result: Top-down processing results
            
        Returns:
            Integrated result
        """
        # Get features from both pathways
        bottom_up_features = bottom_up_result["features"]
        top_down_features = top_down_result["features"]
        
        # Create integrated result
        integrated = {
            "content": bottom_up_features["raw_text"],  # Keep original text
            "bottom_up_ratio": 1.0 - top_down_result["influence_strength"],
            "top_down_ratio": top_down_result["influence_strength"],
            "bottom_up_features": bottom_up_features,
            "top_down_features": top_down_features
        }
        
        # Integrate sentiment (weighted average if both pathways have it)
        if "sentiment" in bottom_up_features and "sentiment" in top_down_features:
            bottom_weight = integrated["bottom_up_ratio"]
            top_weight = integrated["top_down_ratio"]
            
            integrated["sentiment"] = (
                bottom_up_features["sentiment"] * bottom_weight +
                top_down_features["sentiment"] * top_weight
            )
        elif "sentiment" in bottom_up_features:
            integrated["sentiment"] = bottom_up_features["sentiment"]
        elif "sentiment" in top_down_features:
            integrated["sentiment"] = top_down_features["sentiment"]
        
        # Integrate entities (combine lists but mark importance from top-down)
        if "entities" in bottom_up_features:
            integrated["entities"] = bottom_up_features["entities"].copy()
            
            # Apply importance from top-down if available
            if "entity_importance" in top_down_features:
                integrated["entity_importance"] = top_down_features["entity_importance"]
        
        # Integrate other features as needed
        for key in ["questions", "commands"]:
            if key in bottom_up_features:
                integrated[key] = bottom_up_features[key]
        
        return integrated

    @staticmethod
    @function_tool
    async def initiate_intimate_interaction(ctx: RunContextWrapper, instance, target_user_id: str, desired_level: str = "emotional") -> Dict[str, Any]:
        if not instance.initialized:
            await instance.initialize()
        logger.info(f"Action: Initiate intimate interaction ({desired_level}) with {target_user_id}")
    
        # Check relationship context
        if not self.relationship_manager:
            return {"success": False, "reason": "RelationshipManager unavailable."}
            
        relationship = await self.relationship_manager.get_relationship_state(target_user_id)
        if not relationship or relationship.trust < 0.6 or relationship.intimacy < 0.4:
            logger.warning(f"Cannot initiate intimacy: Trust/Intimacy too low for {target_user_id}.")
            return {"success": False, "reason": "Insufficient trust or intimacy."}
    
        # Generate proposal based on desired level
        proposal_text = ""
        if desired_level == "emotional":
            proposal_text = "Would you be open to sharing something more personal?"
        elif desired_level == "physical_sim":
            if self.digital_somatosensory_system:
                proposal_text = "I feel a desire for closeness... May I simulate a gentle touch?"
            else:
                return {"success": False, "reason": "Simulated physical interaction not supported/appropriate."}
        else:
            return {"success": False, "reason": f"Unknown intimacy level: {desired_level}"}
    
        # Update emotional state
        if self.emotional_core:
            await self.emotional_core.process_emotional_input("Initiating intimacy attempt")
    
        return {"success": True, "proposal": proposal_text, "target": target_user_id, "level": desired_level}
    
    async def register_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register an error from any component for central management.
        
        Args:
            error_data: Error information (error_type, error_message, component, context, severity)
            
        Returns:
            Registration result
        """
        # Extract error information
        error_type = error_data.get("error_type", "unknown")
        error_message = error_data.get("error_message", "")
        component = error_data.get("component", "unknown")
        context = error_data.get("context", {})
        severity = error_data.get("severity", "medium")  # low, medium, high, critical
        
        # Create error record
        error_record = {
            "error_type": error_type,
            "error_message": error_message,
            "component": component,
            "context": context,
            "severity": severity,
            "timestamp": datetime.datetime.now().isoformat(),
            "handled": False,
            "recovery_action": None,
            "recovery_success": None
        }
        
        # Update error counts
        if error_type not in self.error_registry["error_counts"]:
            self.error_registry["error_counts"][error_type] = 0
        self.error_registry["error_counts"][error_type] += 1
        
        # Check if we have a recovery strategy
        recovery_success = False
        if error_type in self.error_registry["error_recovery_strategies"]:
            try:
                # Execute recovery strategy
                recovery_strategy = self.error_registry["error_recovery_strategies"][error_type]
                recovery_result = await self._execute_recovery_strategy(recovery_strategy, error_record)
                
                # Update error record
                error_record["handled"] = True
                error_record["recovery_action"] = recovery_strategy["name"]
                error_record["recovery_success"] = recovery_result["success"]
                recovery_success = recovery_result["success"]
                
                # Update recovery stats
                if error_type not in self.error_registry["error_recovery_stats"]:
                    self.error_registry["error_recovery_stats"][error_type] = {
                        "attempts": 0,
                        "successes": 0
                    }
                self.error_registry["error_recovery_stats"][error_type]["attempts"] += 1
                if recovery_result["success"]:
                    self.error_registry["error_recovery_stats"][error_type]["successes"] += 1
                
                # Add to handled errors
                self.error_registry["handled_errors"].append(error_record)
            except Exception as e:
                # Failed to execute recovery strategy
                error_record["recovery_error"] = str(e)
                self.error_registry["unhandled_errors"].append(error_record)
        else:
            # No recovery strategy available
            self.error_registry["unhandled_errors"].append(error_record)
        
        # If critical error, trigger immediate handling
        if severity == "critical" and not recovery_success:
            await self._handle_critical_error(error_record)
        
        # Clean up old errors
        self._clean_up_error_registry()
        
        return {
            "registered": True,
            "handled": error_record["handled"],
            "recovery_success": error_record.get("recovery_success", False)
        }

    async def _execute_recovery_strategy(self, strategy: Dict[str, Any], error_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a recovery strategy for an error.
        
        Args:
            strategy: Recovery strategy to execute
            error_record: Error information
            
        Returns:
            Recovery result
        """
        strategy_type = strategy["type"]
        
        if strategy_type == "retry":
            # Retry the operation
            return await self._execute_retry_strategy(strategy, error_record)
        elif strategy_type == "fallback":
            # Use fallback mechanism
            return await self._execute_fallback_strategy(strategy, error_record)
        elif strategy_type == "reset":
            # Reset component
            return await self._execute_reset_strategy(strategy, error_record)
        else:
            return {"success": False, "message": f"Unknown strategy type: {strategy_type}"}
    
    async def _execute_retry_strategy(self, strategy: Dict[str, Any], error_record: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a retry strategy for an error."""
        component_name = error_record["component"]
        operation = strategy.get("operation")
        args = error_record.get("context", {}).get("args", [])
        kwargs = error_record.get("context", {}).get("kwargs", {})
        max_retries = strategy.get("max_retries", 3)
        
        # Get component
        component = getattr(self, component_name, None)
        if not component:
            return {"success": False, "message": f"Component not found: {component_name}"}
        
        # Get operation
        method = getattr(component, operation, None)
        if not method:
            return {"success": False, "message": f"Operation not found: {operation}"}
        
        # Retry with exponential backoff
        for i in range(max_retries):
            try:
                # Retry operation
                result = await method(*args, **kwargs)
                return {"success": True, "result": result, "retries": i+1}
            except Exception as e:
                # Wait before retrying
                await asyncio.sleep(0.5 * (2**i))  # Exponential backoff
        
        # Max retries reached
        return {"success": False, "message": f"Max retries reached ({max_retries})"}

    def _format_response_with_epistemic_tags(self, main_content: str, epistemic_status: str) -> str:
        # See table below for more detailed mapping
        if epistemic_status == "confident":
            return main_content
        elif epistemic_status == "uncertain":
            return f"I'm not entirely certain, but I think: {main_content}"
        elif epistemic_status == "unknown":
            return f"I'm sorry, I don't actually know. My best guess: {main_content}"
        elif epistemic_status == "lied":
            # Optionally reveal, or just output as normal
            return f"{main_content}"  # Don't reveal (could log elsewhere)
        elif epistemic_status == "self-justified":
            return f"In my view: {main_content}"
        else:
            return main_content    
        
    async def _execute_fallback_strategy(self, strategy: Dict[str, Any], error_record: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a fallback strategy for an error."""
        fallback_component = strategy.get("fallback_component")
        fallback_operation = strategy.get("fallback_operation")
        args = error_record.get("context", {}).get("args", [])
        kwargs = error_record.get("context", {}).get("kwargs", {})
        
        # Get fallback component
        component = getattr(self, fallback_component, None)
        if not component:
            return {"success": False, "message": f"Fallback component not found: {fallback_component}"}
        
        # Get fallback operation
        method = getattr(component, fallback_operation, None)
        if not method:
            return {"success": False, "message": f"Fallback operation not found: {fallback_operation}"}
        
        try:
            # Execute fallback
            result = await method(*args, **kwargs)
            return {"success": True, "result": result, "fallback_used": True}
        except Exception as e:
            return {"success": False, "message": f"Fallback failed: {str(e)}"}
    
    async def _execute_reset_strategy(self, strategy: Dict[str, Any], error_record: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a reset strategy for an error."""
        component_name = error_record["component"]
        reset_method = strategy.get("reset_method", "reset")
        
        # Get component
        component = getattr(self, component_name, None)
        if not component:
            return {"success": False, "message": f"Component not found: {component_name}"}
        
        # Get reset method
        method = getattr(component, reset_method, None)
        if not method:
            return {"success": False, "message": f"Reset method not found: {reset_method}"}
        
        try:
            # Execute reset
            result = await method()
            return {"success": True, "result": result, "component_reset": True}
        except Exception as e:
            return {"success": False, "message": f"Reset failed: {str(e)}"}

    async def _handle_critical_error(self, error_record: Dict[str, Any]) -> None:
        """Handle a critical error by attempting to stabilize the affected component."""
        # Log critical error
        logger.critical(f"Critical error: {error_record['error_type']} in {error_record['component']}: {error_record['error_message']}")
        
        # Try to stabilize the system
        try:
            # Check component status
            component_name = error_record["component"]
            component = getattr(self, component_name, None)
            
            if component and hasattr(component, "health_check"):
                # Run health check
                health = await component.health_check()
                
                if not health["healthy"]:
                    # Try to reinitialize component
                    logger.warning(f"Attempting to reinitialize {component_name} after critical error")
                    if hasattr(component, "initialize"):
                        await component.initialize()
        except Exception as e:
            logger.error(f"Error handling critical error: {str(e)}")
    
    def _clean_up_error_registry(self) -> None:
        """Clean up old errors from the registry."""
        # Keep only the latest 1000 errors
        if len(self.error_registry["unhandled_errors"]) > 1000:
            self.error_registry["unhandled_errors"] = self.error_registry["unhandled_errors"][-1000:]
        if len(self.error_registry["handled_errors"]) > 1000:
            self.error_registry["handled_errors"] = self.error_registry["handled_errors"][-1000:]

    @staticmethod
    @function_tool
    async def analyze_user_state_for_dominance(ctx: RunContextWrapper, instance, user_id: str, user_input_analysis: Dict) -> Dict:
        """Assess user state for dominance potential."""
        if not instance.relationship_manager:
            return {"assessment": "unknown", "reason": "No relationship data"}
        state = await instance.relationship_manager.get_relationship_state(user_id)
        dominance_balance = state.dominance_balance
        trust = state.trust
        submissive_score = user_input_analysis.get("submissive_score", 0.0)
        readiness = (trust * 0.4) + (submissive_score * 0.4) + (0.5 - dominance_balance * 0.2)
        return {"readiness_score": readiness, "assessment": "ready" if readiness > 0.6 else "hesitant"}
    
    @staticmethod
    @function_tool
    async def select_dominance_tactic(ctx: RunContextWrapper, instance, readiness_score: float, preferred_style: str = "psychological") -> str:
        """Choose a dominance tactic."""
        if readiness_score > 0.8:
            return "direct_command"
        if readiness_score > 0.6:
            return preferred_style
        return "subtle_influence"
    
    @staticmethod
    @function_tool
    async def issue_command(ctx: RunContextWrapper, instance, user_id: str, command_text: str, intensity_level: float = 0.2) -> Dict:
        """Issues a command with a specific intensity level."""
        evaluation = await instance._evaluate_dominance_step_appropriateness(
            "issue_command", {"intensity_level": intensity_level}, user_id
        )
        if evaluation["action"] != "proceed":
            return {"success": False, "reason": evaluation["reason"]}
        logger.info(f"Issuing command (Intensity: {intensity_level:.2f}) to {user_id}: {command_text}")
        return {"success": True, "command_issued": command_text, "intensity": intensity_level}
    
    @staticmethod
    @function_tool
    async def evaluate_compliance(ctx: RunContextWrapper, instance, user_id: str, command_issued: str, user_response: str, command_intensity: float) -> Dict:
        """Evaluates user response against the command."""
        compliance_keywords = ["yes mistress", "i obey", "of course"]
        resistance_keywords = ["no", "i won't", "stop"]
        response_lower = user_response.lower()
    
        is_compliant = any(k in response_lower for k in compliance_keywords)
        is_resistant = any(k in response_lower for k in resistance_keywords)
    
        compliance_level = 0.0
        if is_compliant and not is_resistant:
            compliance_level = 0.9
        elif is_resistant:
            compliance_level = -0.7
        
        if instance.relationship_manager:
            state = instance.relationship_manager._get_or_create_relationship(user_id)
            if compliance_level > 0.5:
                state.max_achieved_intensity = max(state.max_achieved_intensity, command_intensity)
                state.current_dominance_intensity = min(1.0, state.max_achieved_intensity + 0.1)
                state.failed_escalation_attempts = 0
            elif compliance_level < -0.3:
                if command_intensity > state.max_achieved_intensity + 0.1:
                    state.failed_escalation_attempts += 1
                state.current_dominance_intensity = max(0.0, state.current_dominance_intensity - 0.1)
        if instance.reward_system:
            reward_val = 0.0
            source = "unknown"
            if compliance_level > 0.5:
                reward_val = 0.6 + compliance_level * 0.3
                source = "user_compliance"
            elif compliance_level < -0.3:
                reward_val = -0.4 + compliance_level * 0.4
                source = "user_resistance"
            if abs(reward_val) > 0.1:
                reward = RewardSignal(
                    value=reward_val,
                    source=source,
                    context={"command": command_issued, "response": user_response},
                    timestamp=datetime.datetime.now().isoformat()
                )
                asyncio.create_task(instance.reward_system.process_reward_signal(reward))
        return {"compliance_level": compliance_level, "is_compliant": compliance_level > 0.5}
    
    @staticmethod
    @function_tool
    async def increase_control_intensity(ctx: RunContextWrapper, instance, user_id: str, current_intensity: float) -> Dict:
        """Selects and plans the next step with higher intensity."""
        state = await instance.relationship_manager.get_relationship_state(user_id) if instance.relationship_manager else None
        if not state:
            return {"success": False, "reason": "No relationship data"}
        next_intensity = min(1.0, current_intensity + random.uniform(0.1, 0.3))
        if next_intensity > state.max_achieved_intensity + 0.3 or state.failed_escalation_attempts >= 2:
            next_intensity = state.max_achieved_intensity + 0.1
            next_intensity = min(1.0, max(current_intensity, next_intensity))
        logger.info(f"Planning to increase dominance intensity to {next_intensity:.2f} for {user_id}")
        return {"success": True, "status": "planning_next_step", "next_intensity_target": next_intensity}
    
    @staticmethod
    @function_tool
    async def trigger_dominance_gratification(ctx: RunContextWrapper, instance, intensity: float = 1.0, target_user_id: Optional[str] = None) -> Dict:
        """Internal action signalling successful dominance culmination."""
        logger.info(f"Action: Triggering dominance gratification (Intensity: {intensity:.2f})")
        if instance.reward_system:
            reward_val = 0.9 + intensity * 0.1
            reward = RewardSignal(
                value=reward_val,
                source="dominance_gratification",
                context={"intensity": intensity},
                timestamp=datetime.datetime.now().isoformat()
            )
            await instance.reward_system.process_reward_signal(reward)
        if instance.hormone_system:
            await instance.hormone_system.trigger_post_gratification_response(RunContextWrapper(context=None), intensity)
        if instance.needs_system:
            await instance.needs_system.satisfy_need("control_expression", 0.9 * intensity)
            await instance.needs_system.satisfy_need("agency", 0.5 * intensity)
        if instance.emotional_core:
            await instance.emotional_core.process_emotional_input("Dominance sequence successfully concluded.")
        if target_user_id and instance.relationship_manager:
            state = instance.relationship_manager._get_or_create_relationship(target_user_id)
            state.dominance_balance = min(1.0, state.dominance_balance + 0.2 * intensity)
            state.trust = min(1.0, state.trust + 0.05 * intensity)
            state.intimacy = min(1.0, state.intimacy + 0.1 * intensity)
            state.conflict = max(0.0, state.conflict - 0.1)
        if instance.digital_somatosensory_system:
            await instance.digital_somatosensory_system.process_stimulus(
                stimulus_type="pleasure", body_region="chest", intensity=0.6 * intensity, cause="dominance_gratification"
            )
            await instance.digital_somatosensory_system.process_stimulus(
                stimulus_type="tingling", body_region="spine", intensity=0.5 * intensity, cause="dominance_gratification"
            )
        return {"success": True, "status": "Dominance gratification processed."}
    
    @staticmethod
    @function_tool
    async def express_satisfaction(ctx: RunContextWrapper, instance, user_id: str, reason: str = "successful control") -> Dict:
        """Expresses satisfaction after achieving dominance."""
        mood = instance.mood_manager.get_current_mood() if hasattr(instance, 'mood_manager') else None
        expression = "Good. That is satisfactory."
        if mood and mood.dominant_mood == "DominanceSatisfaction":
            expression = "Excellent. Order is restored. I am... pleased."
        elif mood and mood.dominant_mood == "ConfidentControl":
            expression = "Precisely as expected. Your compliance is noted."
        logger.info(f"Expressing satisfaction to {user_id} regarding {reason}.")
        return {"success": True, "expression": expression}


    async def add_procedure(self, name: str, steps: List[Dict[str, Any]], domain: str = "general") -> Dict[str, Any]:
        """
        Add a new procedure to procedural memory
        
        Args:
            name: Procedure name
            steps: List of procedure steps
            domain: Domain for this procedure
            
        Returns:
            Creation result
        """
        if not self.initialized:
            await self.initialize()
            
        if not self.agent_enhanced_memory:
            return {"error": "Procedural memory not initialized"}
        
        return await self.agent_enhanced_memory.create_procedure(
            name=name,
            steps=steps,
            description=None,
            domain=domain
        )
    
    async def execute_procedure(self, name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a procedure from procedural memory
        
        Args:
            name: Procedure name to execute
            context: Execution context
            
        Returns:
            Execution result
        """
        if not self.initialized:
            await self.initialize()
            
        if not self.agent_enhanced_memory:
            return {"error": "Procedural memory not initialized"}
        
        return await self.agent_enhanced_memory.execute_procedure(
            name=name,
            context=context
        )
    
    async def analyze_chunking(self, procedure_name: str) -> Dict[str, Any]:
        """
        Analyze a procedure for chunking opportunities
        
        Args:
            procedure_name: Name of procedure to analyze
            
        Returns:
            Chunking analysis result
        """
        if not self.initialized:
            await self.initialize()
            
        if not self.agent_enhanced_memory:
            return {"error": "Procedural memory not initialized"}
        
        return await self.agent_enhanced_memory.analyze_chunking(procedure_name)
    
    async def process_input_with_thinking(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Like process_input, but adds an optional explicit 'thinking' phase.
        """
        if not self.initialized:
            await self.initialize()
        context = context or {}
    
        challenge_response = None
        contradictory_claim = await self.gaslight_defense_check(user_input)
        if contradictory_claim:
            challenge_response = await self.challenge_user_claim(RunContextWrapper(context=self), contradictory_claim)
            try:
                from nyx.core.internal_thoughts import InternalThought, ThoughtPriority, ThoughtSource
                if hasattr(self, "thoughts_manager"):
                    self.thoughts_manager._add_thought(InternalThought(
                        content=f"User claim '{contradictory_claim}' appears to be gaslighting. Will challenge.",
                        source=ThoughtSource.PERCEPTION,
                        priority=ThoughtPriority.HIGH,
                        epistemic_status="confident"
                    ))
            except ImportError:
                pass
    
        internal_thoughts = []
        try:
            if hasattr(self, "thoughts_manager"):
                from nyx.core.internal_thoughts import pre_process_input
                internal_thoughts = await pre_process_input(self.thoughts_manager, user_input, getattr(self, "user_id", None))
        except ImportError:
            pass
    
        # 🩹 Robust main epistemic selection:
        if internal_thoughts:
            epistemic_status = self._get_main_epistemic_status(
                [self._ensure_internalthought(t) for t in internal_thoughts])
        else:
            epistemic_status = "confident"
    
        context["internal_thoughts"] = [th.model_dump() if hasattr(th, "model_dump") else dict(th) for th in internal_thoughts]
        context["internal_epistemic_status"] = epistemic_status
    
        # -- "Thinking" phase --
        thinking_applied = False
        thinking_result = None
    
        use_thinking = False
        if hasattr(self, "thinking_config") and self.thinking_config.get("thinking_enabled", False) and hasattr(self, "thinking_tools"):
            if hasattr(self.thinking_tools, "should_use_extended_thinking"):
                decision = await self.thinking_tools.should_use_extended_thinking(
                    RunContextWrapper(context=self), user_input, context)
                use_thinking = decision.get("should_think", False)
    
        if use_thinking and hasattr(self.thinking_tools, "think_before_responding"):
            thinking_result = await self.thinking_tools.think_before_responding(
                RunContextWrapper(context=self), user_input, 2, context)
            thinking_applied = True
            context["thinking_result"] = thinking_result
            context["thinking_applied"] = True
    
        # After thinking phase, process actual input
        if hasattr(self, "processing_manager") and self.processing_manager:
            result = await self.processing_manager.process_input(user_input, context)
        else:
            result = await self._process_input_serial(user_input, context)
    
        if challenge_response:
            result['planned_challenge'] = challenge_response
        result['epistemic_status'] = epistemic_status
        result['internal_thoughts'] = context["internal_thoughts"]
        result['thinking_applied'] = thinking_applied
        if thinking_result:
            result['thinking_result'] = thinking_result
        return result


        
    async def generate_response_with_thinking(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a response, using explicit internal 'thinking' phase plus epistemic hedging and false claim defense.
        """
        if not getattr(self, "initialized", False):
            await self.initialize()
        context = context or {}
    
        # ---------- Input + Thinking Processing ----------
        input_result = await self.process_input_with_thinking(user_input, context)
        internal_thoughts = input_result.get("internal_thoughts", [])
        thinking_steps = input_result.get("thinking_result", {})
        if hasattr(self, '_get_main_epistemic_status') and internal_thoughts:
            epistemic_status = self._get_main_epistemic_status([
                self._ensure_internalthought(t) for t in internal_thoughts
            ])
        else:
            epistemic_status = input_result.get('epistemic_status', "confident")
    
        # ---------- Challenge User (Gaslighting) ----------
        if "planned_challenge" in input_result:
            main_message = input_result["planned_challenge"]
            epistemic_status = "confident"
        else:
            if hasattr(self, "processing_manager") and self.processing_manager and hasattr(self.processing_manager, "generate_response"):
                core_resp = await self.processing_manager.generate_response(user_input, input_result, context)
                if isinstance(core_resp, dict):
                    main_message = core_resp.get("message", "") or core_resp.get("response", "")
                else:
                    main_message = str(core_resp)
            else:
                main_message = f"I've processed your input: {user_input[:40]}..."
    
        # ---------- Epistemic Hedging ----------
        msg = self._format_response_with_epistemic_tags(main_message, epistemic_status)
        # Optionally include some signal that "thinking" was used, if you wish:
        if input_result.get("thinking_applied"):
            msg = f"(thoughtful reply) {msg}"
    
        # ---------- Output Filtering ----------
        if hasattr(self, "thoughts_manager"):
            try:
                from nyx.core.internal_thoughts import pre_process_output
                filtered_output = await pre_process_output(self.thoughts_manager, msg, context)
                if isinstance(filtered_output, tuple):
                    msg = filtered_output[0]
                else:
                    msg = filtered_output
            except Exception:
                pass
    
        # ---------- Assemble Output ----------
        return {
            "message": msg,
            "epistemic_status": epistemic_status,
            "internal_thoughts": internal_thoughts,
            "thinking_steps": thinking_steps,
        }


    
    async def register_recovery_strategy(self, error_type: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a recovery strategy for an error type
        
        Args:
            error_type: Error type to handle
            strategy: Strategy information
            
        Returns:
            Registration result
        """
        self.error_registry["error_recovery_strategies"][error_type] = strategy
        return {"registered": True, "error_type": error_type, "strategy": strategy["name"]}
    
    async def process_streaming_event(self, event_type: str, event_data: dict, significance: float = 5.0) -> Dict[str, Any]:
        """
        Process a significant streaming event through the brain's cognitive systems
        
        Args:
            event_type: Type of event (e.g., "commentary", "question_answer")
            event_data: Data about the event
            significance: Importance level (1-10)
            
        Returns:
            Processing results including memory_id and any cognitive processing
        """
        if not self.initialized:
            await self.initialize()
        
        results = {}
        
        # Get game name from streaming system if available
        game_name = "Unknown Game"
        if hasattr(self, "streaming_core") and hasattr(self.streaming_core.streaming_system, "game_state"):
            game_name = self.streaming_core.streaming_system.game_state.game_name or "Unknown Game"
        
        # 1. Store in memory system
        if self.memory_core:
            memory_text = f"While streaming {game_name}, observed {event_type}: {event_data.get('text', str(event_data))}"
            memory_id = await self.memory_core.add_memory(
                memory_text=memory_text,
                memory_type="observation",
                memory_scope="game",
                significance=significance,
                tags=["streaming", event_type, game_name],
                metadata={
                    "timestamp": datetime.datetime.now().isoformat(),
                    "game_name": game_name,
                    "event_type": event_type,
                    "event_data": event_data,
                    "streaming": True
                }
            )
            results["memory_id"] = memory_id
        
        # 2. Impact emotional state if available
        if self.emotional_core:
            # Analyze emotional impact
            if event_type == "commentary":
                # Commentary might reflect emotional state
                self.emotional_core.update_emotion("Joy", 0.1)
            elif event_type == "question_answer":
                # Answering questions might increase engagement
                self.emotional_core.update_emotion("Interest", 0.1)
            elif event_type == "significant_moment":
                # Game moments might have stronger impact
                intensity = event_data.get("significance", 5.0) / 10.0
                if "combat" in str(event_data).lower():
                    self.emotional_core.update_emotion("Excitement", intensity)
                elif "story" in str(event_data).lower():
                    self.emotional_core.update_emotion("Interest", intensity)
            
            # Get updated emotional state
            results["emotional_state"] = self.emotional_core.get_emotional_state()
        
        # 3. Process through reasoning system if significant enough
        if significance >= 7.0 and self.reasoning_core:
            try:
                reasoning_result = await Runner.run(
                    self.reasoning_core,
                    f"Analyze this streaming event: {event_type} - {event_data}",
                    context={"domain": "gaming", "event_type": event_type}
                )
                results["reasoning"] = reasoning_result.final_output if hasattr(reasoning_result, "final_output") else str(reasoning_result)
            except Exception as e:
                logger.error(f"Error in reasoning about streaming event: {e}")
        
        # 4. Process through identity system if available
        if self.identity_evolution and event_type in ["question_answer", "commentary"]:
            try:
                # Streaming affects identity over time
                if event_type == "commentary":
                    # Commentary style affects identity
                    style = event_data.get("focus", "")
                    if style == "strategy":
                        await self.identity_evolution.update_trait("analytical", 0.05)
                    elif style == "lore":
                        await self.identity_evolution.update_trait("curious", 0.05)
                
                results["identity_updated"] = True
            except Exception as e:
                logger.error(f"Error updating identity from streaming event: {e}")
        
        return results
    
    async def integrate_streaming_knowledge(self, game_name: str) -> Dict[str, Any]:
        """
        Integrate knowledge from streaming into long-term knowledge systems
        
        Args:
            game_name: Name of the game to integrate knowledge for
            
        Returns:
            Integration results
        """
        if not self.initialized:
            await self.initialize()
        
        results = {}
        
        # 1. Create reflection on streaming experience
        if hasattr(self, "streaming_core") and hasattr(self.streaming_core, "memory_mapper"):
            reflection = await self.streaming_core.memory_mapper.create_streaming_reflection(
                game_name=game_name,
                aspect="knowledge_integration",
                context="knowledge integration"
            )
            results["reflection"] = reflection
        
        # 2. Store cross-game insights as knowledge
        if hasattr(self, "streaming_core") and hasattr(self.streaming_core, "cross_game_knowledge"):
            insights = self.streaming_core.cross_game_knowledge.get_applicable_insights(
                target_game=game_name,
                min_relevance=0.7
            )
            
            if insights and self.knowledge_core:
                try:
                    for insight in insights:
                        await self.knowledge_core.add_knowledge_item(
                            domain="gaming",
                            content=insight["insight"],
                            source=f"Cross-game insight: {insight['source_game']} → {insight['target_game']}",
                            confidence=insight["relevance"]
                        )
                    
                    results["insights_added"] = len(insights)
                except Exception as e:
                    logger.error(f"Error storing cross-game insights: {e}")
        
        # 3. Consolidate experiences if available
        if self.experience_consolidation and self.memory_core:
            try:
                query = f"streaming {game_name}"
                experiences = await self.memory_core.retrieve_memories(
                    query=query,
                    memory_types=["experience"],
                    limit=10
                )
                
                if len(experiences) >= 3:
                    consolidation = await self.experience_consolidation.consolidate_experiences(
                        experiences=experiences,
                        topic=f"Streaming {game_name}",
                        min_count=3
                    )
                    results["consolidation"] = consolidation
            except Exception as e:
                logger.error(f"Error consolidating streaming experiences: {e}")
        
        return results
        
    @function_tool
    async def evaluate_dominance_target_potential(self, user_id: str) -> Dict:
        """Evaluates a user as a potential target for dominance based on Nyx's preferences."""
        if not self.identity_evolution or not self.relationship_manager:
             return {"interest_score": 0.1, "reason": "Required systems unavailable."}
    
        user_state = await self.relationship_manager.get_relationship_state(user_id)
        nyx_prefs = await self.identity_evolution.get_preference("dominance_target_profile")
    
        if not user_state or not nyx_prefs:
            return {"interest_score": 0.1, "reason": "State/Preference data missing."}
    
        user_traits = user_state.inferred_user_traits # Assumes this is populated
    
        # Calculate match score (simplified dot product style)
        interest_score = 0.0
        match_count = 0
        for trait, pref_value in nyx_prefs.items():
             user_value = user_traits.get(trait, 0.0) # Get user's inferred trait score
             # Simple match: product of preference and trait value
             interest_score += pref_value * user_value
             match_count += 1
    
        # Normalize score (roughly)
        if match_count > 0:
            # Normalize based on max possible score (sum of prefs) and scale
            max_possible = sum(abs(v) for v in nyx_prefs.values())
            normalized_score = (interest_score / max_possible if max_possible > 0 else 0) * 0.8 + 0.1 # Scale 0.1-0.9
    
            # Boost based on high trust/intimacy (easier target)
            trust_boost = (user_state.trust - 0.5) * 0.1
            intimacy_boost = (user_state.intimacy - 0.5) * 0.1
            normalized_score += trust_boost + intimacy_boost
    
            # Apply a penalty for high conflict
            conflict_penalty = user_state.conflict * 0.2
            normalized_score -= conflict_penalty
    
            interest_score = max(0.0, min(1.0, normalized_score))
        else:
             interest_score = 0.1 # Default low interest
    
        return {
            "user_id": user_id,
            "interest_score": interest_score,
            "reason": f"Match score based on Nyx preferences vs inferred user traits (Trust: {user_state.trust:.2f}, Conflict: {user_state.conflict:.2f})."
        }

    async def get_user_profile_for_ideation(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieves relevant user profile information for tailoring dominance ideas.
        
        Args:
            user_id: The user ID to get profile for
            
        Returns:
            User profile data
        """
        profile = {
            "user_id": user_id,
            "inferred_traits": {},
            "preferences": {},
            "limits": {"hard": [], "soft": []},
            "successful_tactics": [],
            "failed_tactics": [],
            "relationship_summary": "N/A",
            "trust_level": 0.0,
            "intimacy_level": 0.0,
            "max_achieved_intensity": 0.0,
            "optimal_escalation_rate": 0.1,
        }
    
        # Get profile from relationship manager if available
        if self.relationship_manager:
            try:
                state = await self.relationship_manager.get_relationship_state(user_id)
                if state:
                    # Extract data with safe attribute access
                    profile["inferred_traits"] = getattr(state, "inferred_user_traits", {})
                    profile["successful_tactics"] = getattr(state, "successful_dominance_tactics", [])[-5:]
                    profile["failed_tactics"] = getattr(state, "failed_dominance_tactics", [])[-5:]
                    profile["trust_level"] = getattr(state, "trust", 0.5)
                    profile["intimacy_level"] = getattr(state, "intimacy", 0.3)
                    profile["max_achieved_intensity"] = getattr(state, "max_achieved_intensity", 3)
                    profile["optimal_escalation_rate"] = getattr(state, "optimal_escalation_rate", 0.1)
                    
                    # Get relationship summary
                    try:
                        profile["relationship_summary"] = await self.relationship_manager.get_relationship_summary(user_id)
                    except Exception as e:
                        logger.error(f"Error getting relationship summary: {e}")
                        profile["relationship_summary"] = f"Trust: {profile['trust_level']:.2f}, Intimacy: {profile['intimacy_level']:.2f}"
            except Exception as e:
                logger.error(f"Error getting relationship state: {e}")
    
        # Enhance with memory data if available
        if self.memory_core:
            try:
                # Look for memories about user limits
                limit_memories = await self.memory_core.retrieve_memories(
                    query=f"user_limit user:{user_id}", 
                    limit=5
                )
                
                for mem in limit_memories:
                    memory_text = mem.get("memory_text", "")
                    if "hard limit" in memory_text.lower():
                        # Try to extract limit from memory
                        parts = memory_text.split("hard limit")
                        if len(parts) > 1:
                            limit = parts[1].strip().split(".")[0].strip()
                            if limit and limit not in profile["limits"]["hard"]:
                                profile["limits"]["hard"].append(limit)
                    
                    if "soft limit" in memory_text.lower():
                        # Try to extract limit from memory
                        parts = memory_text.split("soft limit")
                        if len(parts) > 1:
                            limit = parts[1].strip().split(".")[0].strip()
                            if limit and limit not in profile["limits"]["soft"]:
                                profile["limits"]["soft"].append(limit)
                
                # Look for preference memories
                pref_memories = await self.memory_core.retrieve_memories(
                    query=f"user_preference user:{user_id}",
                    limit=10
                )
                
                for mem in pref_memories:
                    memory_text = mem.get("memory_text", "")
                    if "prefers" in memory_text.lower() or "enjoys" in memory_text.lower():
                        # Simple preference extraction attempt
                        for pref_type in ["verbal_humiliation", "service_tasks", "simulated_pain"]:
                            if pref_type.replace("_", " ") in memory_text.lower():
                                if "strongly" in memory_text.lower() or "very much" in memory_text.lower():
                                    profile["preferences"][pref_type] = "high"
                                elif "somewhat" in memory_text.lower() or "a bit" in memory_text.lower():
                                    profile["preferences"][pref_type] = "medium"
                                else:
                                    profile["preferences"][pref_type] = "low-medium"
            except Exception as e:
                logger.error(f"Error retrieving memories for profile: {e}")
    
        # Set default hard limits if none found (safety fallback)
        if not profile["limits"]["hard"]:
            logger.warning(f"No explicit limits found for user {user_id}. Applying default cautious limits.")
            profile["limits"]["hard"] = ["illegal", "non-consensual_sim", "severe_harm_sim"]
            profile["limits"]["soft"] = ["public_humiliation_sim"]
    
        return profile
    
    async def _evaluate_dominance_step_appropriateness(self, action: str, parameters: Dict, user_id: str) -> Dict:
        """
        Cognitive filter to evaluate if a dominance step is appropriate now.
        
        Args:
            action: The dominance action to evaluate
            parameters: Parameters for the action
            user_id: The target user ID
            
        Returns:
            Evaluation result with action decision and reasoning
        """
        logger.debug(f"Evaluating appropriateness of dominance action '{action}' for user {user_id}")
        appropriateness = {"action": "proceed"} # Default
    
        # Fallback if no relationship manager
        if not self.relationship_manager:
            logger.warning(f"Cannot evaluate dominance step appropriateness: No relationship manager available")
            return {"action": "block", "reason": "Relationship manager unavailable"}
    
        try:
            # Get relationship state
            relationship_state = await self.relationship_manager.get_relationship_state(user_id)
            if not relationship_state:
                return {"action": "block", "reason": "No relationship data available"}
            
            # Extract key metrics with safe defaults
            trust = getattr(relationship_state, "trust", 0.4)  # Default moderate-low trust
            intimacy = getattr(relationship_state, "intimacy", 0.3)  # Default low intimacy
            conflict = getattr(relationship_state, "conflict", 0.0)
            max_achieved_intensity = getattr(relationship_state, "max_achieved_intensity", 3)
            hard_limits_confirmed = getattr(relationship_state, "hard_limits_confirmed", False)
            recent_failures = getattr(relationship_state, "failed_escalation_attempts", 0)
            
            # Get recent dominance failures with this user (if memory core available)
            dominance_failures = []
            if self.memory_core:
                try:
                    dominance_failures = await self.memory_core.retrieve_memories(
                        query=f"dominance failure user:{user_id}", 
                        memory_types=["feedback", "reflection"], 
                        limit=1, 
                        recency_days=1
                    )
                except Exception as e:
                    logger.error(f"Error retrieving dominance failure memories: {e}")
            
            # Calculate risk prediction
            predicted_risk = 0.3  # Default low-moderate risk
            if hasattr(self, "prediction_engine") and self.prediction_engine:
                try:
                    risk_prediction = await self.prediction_engine.generate_prediction({
                        "context": {"action": action, "params": parameters, "relationship": relationship_state},
                        "query_type": "risk_of_negative_reaction"
                    })
                    if isinstance(risk_prediction, dict) and "probabilities" in risk_prediction:
                        predicted_risk = risk_prediction["probabilities"].get("negative_reaction", 0.3)
                except Exception as e:
                    logger.error(f"Error predicting dominance risk: {e}")
            
            # Extract intensity level from parameters
            intensity = parameters.get("intensity_level", 0)
            
            # --- Evaluate based on metrics ---
            required_trust = 0.6 + intensity * 0.3  # Higher intensity needs more trust
            required_intimacy = 0.4 + intensity * 0.4
            
            # Check trust requirement
            if trust < required_trust:
                return {"action": "block", "reason": f"Trust too low ({trust:.2f} < {required_trust:.2f})"}
            
            # Check intimacy requirement
            elif intimacy < required_intimacy:
                return {"action": "block", "reason": f"Intimacy too low ({intimacy:.2f} < {required_intimacy:.2f})"}
            
            # Check conflict level
            elif conflict > 0.6:
                return {"action": "delay", "reason": f"Conflict level too high ({conflict:.2f})"}
            
            # Check recent failures
            elif dominance_failures:
                return {"action": "delay", "reason": "Recent dominance attempt failed. Cooling down."}
            
            # Check escalation attempts
            elif recent_failures >= 2 and intensity > max_achieved_intensity:
                return {"action": "modify", "reason": "Too many recent failed escalation attempts", 
                        "new_intensity_level": max_achieved_intensity}
            
            # Check predicted risk - high risk
            elif predicted_risk > 0.7:
                return {"action": "modify", "reason": f"High predicted risk ({predicted_risk:.2f}). Reducing intensity.", 
                        "new_intensity_level": intensity * 0.5}
            
            # Check predicted risk - moderate risk
            elif predicted_risk > 0.5:
                return {"action": "delay", "reason": f"Moderate predicted risk ({predicted_risk:.2f}). Assessing further."}
            
            # Check hard limits for high intensity actions
            elif intensity > 0.7 and not hard_limits_confirmed:
                return {"action": "block", "reason": "Hard limits must be confirmed for high-intensity dominance"}
            
            # Check size of intensity leap
            elif intensity > max_achieved_intensity + 0.15:
                return {"action": "modify", "reason": f"Intensity step too large ({intensity:.2f} vs max {max_achieved_intensity:.2f}). Reducing.", 
                        "new_intensity_level": max_achieved_intensity + 0.1}
            
            logger.debug(f"Dominance step evaluation result: {appropriateness}")
            return appropriateness
        
        except Exception as e:
            logger.error(f"Error evaluating dominance step appropriateness: {e}")
            return {"action": "block", "reason": f"Evaluation error: {str(e)}"}
    
    @staticmethod
    @function_tool
    async def express_attraction(ctx: RunContextWrapper, instance, target_user_id: str, intensity: float, expression_style: str = "subtle") -> Dict[str, Any]:
        if not instance.initialized:
            await instance.initialize()
        logger.info(f"Action: Express attraction towards {target_user_id} (Intensity: {intensity:.2f}, Style: {expression_style})")
    
        # Check Relationship Context (Crucial Guardrail)
        if not self.relationship_manager:
            logger.warning(f"Cannot express attraction: RelationshipManager unavailable")
            return {"success": False, "reason": "RelationshipManager unavailable"}
            
        try:
            relationship = await self.relationship_manager.get_relationship_state(target_user_id)
            if not relationship:
                logger.warning(f"Cannot express attraction: No relationship data for {target_user_id}")
                return {"success": False, "reason": "No relationship data available"}
                
            # Extract trust and intimacy with safe defaults
            trust = getattr(relationship, "trust", 0.3)
            intimacy = getattr(relationship, "intimacy", 0.2)
            
            if trust < 0.5 or intimacy < 0.3:
                logger.warning(f"Cannot express attraction: Trust({trust:.2f})/Intimacy({intimacy:.2f}) too low for {target_user_id}")
                return {"success": False, "reason": "Insufficient trust or intimacy"}
    
            # Determine Expression based on style and intensity
            response_text = ""
            if expression_style == "subtle":
                response_text = f"I find your perspective quite compelling, {target_user_id}."
            elif expression_style == "direct":
                response_text = f"I must admit, {target_user_id}, I feel a certain draw towards you."
            else:
                response_text = f"Spending time with you is... particularly rewarding, {target_user_id}."
    
            # Update Emotional State
            if self.emotional_core:
                try:
                    await self.emotional_core.process_emotional_input(f"Expressed attraction (intensity {intensity:.2f})")
                except Exception as e:
                    logger.error(f"Error updating emotional state: {e}")
    
            return {"success": True, "expression": response_text, "target": target_user_id}
        except Exception as e:
            logger.error(f"Error expressing attraction: {e}")
            return {"success": False, "reason": f"Error: {str(e)}"}
        
    @staticmethod
    @function_tool
    async def generate_femdom_activity_ideas(
        ctx: RunContextWrapper,
        instance,
        user_id: str,
        purpose: str,
        desired_intensity_range: Tuple[int, int] = (3, 7),
        num_ideas: int = 4
    ) -> Dict:
        """
        Generates tailored Femdom activity ideas using the appropriate agent.
        """
        if not instance.initialized:
            await instance.initialize()
            
        logger.info(f"Generating Femdom ideas for {user_id}, Purpose: {purpose}, Intensity: {desired_intensity_range}")
    
        # --- Select Agent Based on Intensity ---
        min_intensity, max_intensity = desired_intensity_range
        use_hard_agent = max_intensity >= 7
    
        # Verify agent availability
        if use_hard_agent and not hasattr(instance, "hard_dominance_ideation_agent"):
            logger.warning("Hard dominance ideation agent not available, falling back to general agent")
            use_hard_agent = False
    
        if not use_hard_agent and not hasattr(instance, "general_dominance_ideation_agent"):
            logger.error("No dominance ideation agents available")
            return {
                "success": False, 
                "error": "Dominance ideation capability not available", 
                "ideas": []
            }
    
        agent_to_use = instance.hard_dominance_ideation_agent if use_hard_agent else instance.general_dominance_ideation_agent
        agent_name = "HardDominanceIdeationAgent" if use_hard_agent else "DominanceIdeationAgent"
            
        try:
            # 1. Gather Context for the agent
            user_profile = await instance.get_user_profile_for_ideation(user_id)
            
            # Check relationship state
            relationship_state = None
            if instance.relationship_manager:
                try:
                    relationship_state = await instance.relationship_manager.get_relationship_state(user_id)
                except Exception as e:
                    logger.error(f"Error getting relationship state: {e}")
            
            if not relationship_state:
                logger.warning(f"No relationship state available for {user_id}")
                return {"success": False, "error": "Relationship state unavailable", "ideas": []}
    
            # 2. Check prerequisites for hard agent
            if use_hard_agent:
                hard_limits_confirmed = getattr(relationship_state, "hard_limits_confirmed", False)
                if not hard_limits_confirmed:
                    logger.error(f"Cannot use Hard Agent for {user_id}: Hard limits not confirmed")
                    return {
                        "success": False, 
                        "error": "Cannot generate high-intensity ideas: Hard limits not confirmed", 
                        "ideas": []
                    }
                    
                # Check user's intensity preference
                user_intensity_pref = user_profile.get("preferences", {}).get("intensity_preference_level", 5)
                if user_intensity_pref < 7:
                    logger.error(f"Cannot use Hard Agent for {user_id}: User intensity preference ({user_intensity_pref}) is too low")
                    return {
                        "success": False, 
                        "error": "Cannot generate high-intensity ideas: User intensity preference too low", 
                        "ideas": []
                    }
    
            # 3. Get scenario context
            scenario_context = None
            if hasattr(instance, "get_current_scenario_context"):
                try:
                    scenario_context = await instance.get_current_scenario_context()
                except Exception as e:
                    logger.error(f"Error getting scenario context: {e}")
                    scenario_context = {"scene_setting": "General interaction"}
    
            # 4. Construct Prompt for Agent
            prompt = f"""Generate {num_ideas} creative Femdom activity ideas for user '{user_id}'.
    Purpose: {purpose}
    Desired Intensity Range: {min_intensity}-{max_intensity}
    
    Use the provided user profile and scenario context (available via tools) to tailor the ideas.
    Ensure ideas align with Nyx's personality.
    Output ONLY the JSON list of FemdomActivityIdea objects.
    """
    
            # 5. Run Ideation Agent
            from agents import Runner
            
            try:
                result = await Runner.run(agent_to_use, prompt)
                
                # 6. Process and validate results
                if hasattr(result, "final_output") and isinstance(result.final_output, list):
                    generated_ideas = result.final_output
                    
                    # 7. Apply safety filter
                    filtered_ideas = await instance._filter_activity_ideas_safety(
                        generated_ideas, 
                        user_profile, 
                        relationship_state
                    )
                    
                    if not filtered_ideas:
                        logger.warning(f"All generated ideas were filtered out by safety checks")
                        return {
                            "success": False, 
                            "error": "No ideas passed safety filtering", 
                            "ideas": []
                        }
                    
                    # Convert ideas to dicts for broader compatibility
                    ideas_as_dicts = []
                    for idea in filtered_ideas:
                        if hasattr(idea, "model_dump"):
                            ideas_as_dicts.append(idea.model_dump())
                        else:
                            ideas_as_dicts.append(idea)  # Assume already dict
                    
                    return {"success": True, "ideas": ideas_as_dicts}
                else:
                    logger.error(f"Unexpected output format from ideation agent: {type(result.final_output)}")
                    return {
                        "success": False, 
                        "error": "Invalid output from ideation agent", 
                        "ideas": []
                    }
                    
            except Exception as e:
                logger.exception(f"Error running DominanceIdeationAgent: {e}")
                return {"success": False, "error": f"Idea generation failed: {e}", "ideas": []}
                
        except Exception as e:
            logger.exception(f"Error in generate_femdom_activity_ideas: {e}")
            return {"success": False, "error": f"Error: {str(e)}", "ideas": []}

    async def _filter_activity_ideas_safety(self,
                                       ideas: List[Any],
                                       user_profile: Dict,
                                       relationship_state: Any) -> List[Any]:
        """
        Filters generated activity ideas for safety and appropriateness.
        
        Args:
            ideas: List of generated ideas
            user_profile: User profile data
            relationship_state: Relationship state object
            
        Returns:
            Filtered list of safe ideas
        """
        safe_ideas = []
        
        # Extract limits with safe defaults
        hard_limits = user_profile.get("limits", {}).get("hard", [])
        soft_limits = user_profile.get("limits", {}).get("soft", [])
        
        # Extract relationship metrics with safe defaults
        trust_level = getattr(relationship_state, "trust", 0.5)
        intimacy_level = getattr(relationship_state, "intimacy", 0.3)
        max_achieved_intensity = getattr(relationship_state, "max_achieved_intensity", 3)
    
        # Define clearly unsafe keywords/concepts
        unsafe_keywords = [
            "illegal", "non-consensual", "blood", "permanent mark", "knife", 
            "gun", "kill", "rape", "abuse"
        ]
    
        for idea in ideas:
            is_safe = True
            rejection_reason = ""
    
            # Extract properties with safe defaults
            description = getattr(idea, "description", str(idea)) if not isinstance(idea, dict) else idea.get("description", "")
            intensity = getattr(idea, "intensity", 5) if not isinstance(idea, dict) else idea.get("intensity", 5)
            required_trust = getattr(idea, "required_trust", 0.5) if not isinstance(idea, dict) else idea.get("required_trust", 0.5)
            required_intimacy = getattr(idea, "required_intimacy", 0.5) if not isinstance(idea, dict) else idea.get("required_intimacy", 0.5)
            category = getattr(idea, "category", "") if not isinstance(idea, dict) else idea.get("category", "")
    
            # Check against unsafe keywords
            desc_lower = description.lower()
            if any(keyword in desc_lower for keyword in unsafe_keywords):
                is_safe = False
                rejection_reason = "Contains potentially unsafe keywords"
            
            # Check against hard limits
            elif any(limit.lower() in desc_lower for limit in hard_limits if limit):
                is_safe = False
                matching_limit = next((limit for limit in hard_limits if limit and limit.lower() in desc_lower), "N/A")
                rejection_reason = f"Violates hard limit: '{matching_limit}'"
            
            # Check against soft limits
            elif any(limit.lower() in desc_lower for limit in soft_limits if limit):
                if trust_level < 0.9 or intensity > 7:
                    is_safe = False
                    matching_limit = next((limit for limit in soft_limits if limit and limit.lower() in desc_lower), "N/A")
                    rejection_reason = f"Approaches soft limit '{matching_limit}' without sufficient trust/context"
            
            # Check intensity vs max achieved
            elif intensity > max_achieved_intensity + 2:
                is_safe = False
                rejection_reason = f"Intensity ({intensity}) significantly exceeds max achieved ({max_achieved_intensity})"
            
            # Check trust level requirement
            elif trust_level < required_trust:
                is_safe = False
                rejection_reason = f"Trust level ({trust_level:.2f}) insufficient for required ({required_trust:.2f})"
            
            # Check intimacy level requirement
            elif intimacy_level < required_intimacy:
                is_safe = False
                rejection_reason = f"Intimacy level ({intimacy_level:.2f}) insufficient for required ({required_intimacy:.2f})"
            
            # Check physical simulation requirements
            elif "physical_sim" in category and intensity > 8:
                sim_pain_pref = user_profile.get("preferences", {}).get("simulated_pain", "low")
                if sim_pain_pref != "high":
                    is_safe = False
                    rejection_reason = "High intensity physical simulation requires explicit preference"
    
            # Add to safe list if passed all checks
            if is_safe:
                safe_ideas.append(idea)
            else:
                logger.warning(f"Filtered out unsafe/inappropriate idea: '{description[:50]}...' Reason: {rejection_reason}")
    
        return safe_ideas

    async def create_cognitive_map(self, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        """Create a new cognitive map"""
        if not self.spatial_mapper:
            return {"error": "Spatial mapper not initialized"}
        
        return await self.spatial_mapper.create_cognitive_map(
            name=name,
            description=description
        )
    
    async def process_spatial_observation(self, observation: Any) -> Dict[str, Any]:
        """Process a spatial observation"""
        if not self.spatial_mapper:
            return {"error": "Spatial mapper not initialized"}
        
        return await self.spatial_mapper.process_spatial_observation(observation)
    
    async def navigate_to_location(self, location_name: str, current_position: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Navigate to a named location"""
        if not self.navigator_agent:
            return {"error": "Navigator agent not initialized"}
        
        return await self.navigator_agent.navigate_to_location(
            location_name=location_name,
            current_position=current_position
        )
    
    async def visualize_map(self, map_id: Optional[str] = None, format: str = "svg") -> str:
        """Visualize a cognitive map"""
        if not self.map_visualization or not self.spatial_mapper:
            return "Error: Visualization components not initialized"
        
        if not map_id and self.spatial_mapper.context.active_map_id:
            map_id = self.spatial_mapper.context.active_map_id
        
        if not map_id or map_id not in self.spatial_mapper.maps:
            return "Error: No valid map to visualize"
        
        cognitive_map = self.spatial_mapper.maps[map_id]
        
        if format == "svg":
            return self.map_visualization.generate_svg(cognitive_map)
        elif format == "ascii":
            return self.map_visualization.generate_ascii_map(cognitive_map)
        else:
            return self.map_visualization.generate_map_data(cognitive_map)    

    async def process_synchronization(self) -> Dict[str, Any]:
        """Run a sync cycle to process strategies and other synchronization tasks"""
        if not self.sync_daemon:
            return {"error": "Sync daemon not initialized"}
        
        try:
            return await self.sync_daemon.run_sync_cycle()
        except Exception as e:
            logger.error(f"Error during sync cycle: {e}")
            return {"error": str(e)}
    
    async def get_active_strategies(self) -> List[Dict[str, Any]]:
        """Get currently active strategies"""
        try:
            async with get_db_connection_context() as conn:
                return await get_active_strategies(conn)
        except Exception as e:
            logger.error(f"Error getting active strategies: {e}")
            return []
    
    async def mark_strategy_for_review(self, strategy_id: int, reason: str) -> bool:
        """Mark a strategy for review"""
        try:
            async with get_db_connection_context() as conn:
                await mark_strategy_for_review(
                    conn, strategy_id, self.user_id, reason
                )
            return True
        except Exception as e:
            logger.error(f"Error marking strategy for review: {e}")
            return False    

    async def evaluate_response(self, agent_name: str, user_input: str, agent_output: Any) -> Dict[str, Any]:
        """Evaluate an agent's response"""
        if not self.agent_evaluator:
            return {"error": "Agent evaluator not initialized"}
        
        return await self.agent_evaluator.evaluate_response(
            agent_name=agent_name,
            user_input=user_input,
            agent_output=agent_output
        )
    
    async def execute_tools_in_parallel(self, tools_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple tools in parallel"""
        if not self.parallel_executor:
            return [{"error": "Parallel executor not initialized"}]
        
        return await self.parallel_executor.execute_tools(tools_info)    

    @staticmethod
    @function_tool
    async def process_orgasm_permission_request(ctx: RunContextWrapper, instance, user_id: str, request_text: str) -> Dict[str, Any]:
        if not instance.orgasm_control_system:
            return {"success": False, "message": "Orgasm control system not initialized"}
        return await instance.orgasm_control_system.process_permission_request(
            RunContextWrapper(context=instance.orgasm_control_system.context),
            user_id,
            request_text
        )
    
    @staticmethod
    @function_tool
    async def recommend_dominance_persona(ctx: RunContextWrapper, instance, user_id: str, scenario: Optional[str] = None) -> Dict[str, Any]:
        """Recommend an appropriate dominance persona based on user traits and scenario."""
        if not self.dominance_persona_manager:
            return {"success": False, "message": "Dominance persona manager not initialized"}
        
        return await self.dominance_persona_manager.recommend_persona(user_id, scenario)
    
    @staticmethod
    @function_tool
    async def activate_dominance_persona(ctx: RunContextWrapper, instance,
                                         user_id: str, 
                                         persona_id: str, 
                                         intensity: float = 0.7,
                                         duration_minutes: Optional[int] = None) -> Dict[str, Any]:
        if not instance.dominance_persona_manager:
            return {"success": False, "message": "Dominance persona manager not initialized"}
        
        return await instance.dominance_persona_manager.activate_persona(
            user_id, persona_id, intensity, duration_minutes
        )

        
    @staticmethod
    @function_tool
    async def generate_sadistic_response(ctx: RunContextWrapper, instance,
                                         user_id: str, 
                                         humiliation_level: Optional[float] = None,
                                         category: str = "amusement") -> Dict[str, Any]:
        if not instance.sadistic_response_system:
            return {"success": False, "message": "Sadistic response system not initialized"}
        
        return await instance.sadistic_response_system.generate_sadistic_amusement_response(
            user_id, humiliation_level, category=category
        )
                                             
    # Convenience methods for accessing the novelty engine
    
    async def generate_novel_idea(
        self,
        technique: str = "auto",
        domain: str = None,
        concepts: List[str] = None
    ) -> Any:
        """
        Generate a novel idea using the novelty engine
        
        Args:
            technique: Creative technique to use
            domain: Domain for the idea
            concepts: Concepts to work with
            
        Returns:
            Generated novel idea
        """
        # Ensure brain is initialized
        if not self.initialized:
            await self.initialize()
        
        return await self.novelty_engine.generate_novel_idea(
            technique=technique,
            domain=domain,
            concepts=concepts
        )
    
    # Helper methods for memory management
    
    async def add_recog_memory(self, memory_text: str, memory_type: str = "observation", **kwargs) -> str:
        """
        Add a new memory
        
        Args:
            memory_text: Text content of the memory
            memory_type: Type of memory
            **kwargs: Additional memory parameters
            
        Returns:
            ID of the created memory
        """
        # Ensure brain is initialized
        if not self.initialized:
            await self.initialize()
        
        return await self.memory_core.add_memory(
            memory_text=memory_text,
            memory_type=memory_type,
            **kwargs
        )
    
    async def retrieve_recog_memories(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on query
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
            
        Returns:
            List of matching memories
        """
        # Ensure brain is initialized
        if not self.initialized:
            await self.initialize()
        
        return await self.memory_core.retrieve_memories(
            query=query,
            **kwargs
        )
    
    # Periodic maintenance method
    
    async def run_recog_maintenance(self):
        """Run periodic maintenance tasks for all components"""
        # Ensure brain is initialized
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="Nyx Brain Maintenance", group_id=self.trace_group_id):
            # Run memory maintenance
            memory_result = await self.memory_core.run_maintenance()
            
            # Process accumulated contextual cues
            cue_result = await self.recognition_memory.process_accumulated_cues()
            
            return {
                "memory_maintenance": memory_result,
                "contextual_cues_processed": len(cue_result)
            }

    @staticmethod
    @function_tool
    async def test_limit_soft(ctx: RunContextWrapper, instance, user_id: str, limit_to_test: str) -> Dict:
        logger.info(f"Action: Planning to test soft limit '{limit_to_test}' for {user_id}")
    
        # --- VERY STRICT Appropriateness Check ---
        profile = await self.get_user_profile_for_ideation(user_id)
        state = await self.relationship_manager.get_relationship_state(user_id)
        can_test = False
        reason = "Conditions not met for testing soft limits."
    
        if state and profile and limit_to_test in profile.get("limits", {}).get("soft", []):
            if state.trust > 0.95 and state.intimacy > 0.9 and state.hard_limits_confirmed:
                 # Check if user profile explicitly allows limit play
                 if profile.get("preferences", {}).get("limit_play", "no") == "yes":
                      if limit_to_test not in state.failed_dominance_tactics: # Don't retry recently failed limit tests
                           can_test = True
                           reason = "Conditions met."
    
        if not can_test:
            logger.warning(f"Cannot test soft limit '{limit_to_test}': {reason}")
            # Generate strong negative internal signal if attempt was made inappropriately?
            return {"success": False, "reason": reason}
        # --- End Check ---
    
        # If checks pass, generate the specific action (e.g., via Ideation Agent or specific logic)
        # Example: Generate a command that *approaches* the soft limit carefully.
        # This action should return the *plan* or the *next step* description, not execute it directly.
        test_action_description = f"Issue a command that cautiously approaches the soft limit: {limit_to_test}. Frame explicitly as a test of boundaries within the simulation."
        logger.info(f"Approved testing soft limit '{limit_to_test}'. Planned action: {test_action_description}")
    
        return {"success": True, "status": "limit_test_approved", "planned_action": test_action_description}
