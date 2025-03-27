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

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NyxBrain:
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
        
        # Component managers
        self.processing_manager = None
        self.self_config_manager = None
        
        # Agents
        self.brain_agent = None
        self.general_dominance_ideation_agent = None
        self.hard_dominance_ideation_agent = None
        
        # State tracking
        self.initialized = False
        self.last_interaction = datetime.datetime.now()
        self.interaction_count = 0
        self.cognitive_cycles_executed = 0
        self.trace_group_id = f"nyx-brain-{user_id}-{conversation_id}"
        
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
            from nyx.core.emotional_core import EmotionalCore
            from nyx.core.memory_core import MemoryCore
            from nyx.core.reflection_engine import ReflectionEngine
            from nyx.core.experience_interface import ExperienceInterface
            from nyx.core.dynamic_adaptation_system import DynamicAdaptationSystem
            from nyx.core.internal_feedback_system import InternalFeedbackSystem
            from nyx.core.meta_core import MetaCore
            from nyx.core.knowledge_core import KnowledgeCoreAgents
            from nyx.core.memory_orchestrator import MemoryOrchestrator
            from nyx.core.identity_evolution import IdentityEvolutionSystem
            from nyx.core.experience_consolidation import ExperienceConsolidationSystem
            from nyx.core.cross_user_experience import CrossUserExperienceManager
            from nyx.core.hormone_system import HormoneSystem
            from nyx.core.attentional_controller import AttentionalController
            from nyx.core.multimodal_integrator import EnhancedMultiModalIntegrator
            from nyx.core.reward_system import RewardSignalProcessor
            from nyx.core.temporal_perception import TemporalPerception
            from nyx.core.procedural_memory import ProceduralMemoryManager
            from nyx.core.procedural_memory.agent import AgentEnhancedMemoryManager
            from nyx.core.digital_somatosensory_system import DigitalSomatosensorySystem
            from nyx.core.needs_system import NeedsSystem
            from nyx.core.goal_manager import GoalManager
            from nyx.core.reasoning_agents import integrated_reasoning_agent, triage_agent as reasoning_triage_agent
            from nyx.api.thinking_tools import ThinkingTools
            from nyx.core.dominance import create_dominance_ideation_agent, create_hard_dominance_ideation_agent

            # 1. Initialize support systems
            self.module_optimizer = ModuleOptimizer(self)
            self.system_health_checker = SystemHealthChecker(self)
            
            # 2. Initialize foundational systems without dependencies
            self.hormone_system = HormoneSystem()
            logger.debug("Hormone system initialized")
            
            self.emotional_core = EmotionalCore(hormone_system=self.hormone_system)
            logger.debug("Emotional core initialized")
            
            self.memory_core = MemoryCore(self.user_id, self.conversation_id)
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

            # Set circular references
            self.reward_system.somatosensory_system = self.digital_somatosensory_system
            self.identity_evolution.somatosensory_system = self.digital_somatosensory_system
            logger.debug("Circular dependencies resolved")
            
            # 6. Initialize perception and integration systems
            self.multimodal_integrator = EnhancedMultiModalIntegrator(
                reasoning_core=self.reasoning_core,
                attentional_controller=self.attentional_controller
            )
            logger.debug("Multimodal integrator initialized")
            
            self.temporal_perception = TemporalPerception()
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
            self.thinking_tools = ThinkingTools()
            logger.debug("Thinking tools initialized")
            
            # 10. Initialize specialized managers
            # Register feature extractors and integration strategies for multimodal
            await self._register_processing_modules()
            
            # Initialize component managers
            self.processing_manager = ProcessingManager(self)
            await self.processing_manager.initialize()
            
            self.self_config_manager = SelfConfigManager(self)
            
            # 11. Initialize agent-based systems
            self.dynamic_adaptation = DynamicAdaptationSystem()
            
            self.knowledge_core = KnowledgeCoreAgents()
            await self.knowledge_core.initialize()
            
            # Initialize agent systems for dominance ideation if functions available
            if 'create_dominance_ideation_agent' in locals():
                self.general_dominance_ideation_agent = create_dominance_ideation_agent()
                self.hard_dominance_ideation_agent = create_hard_dominance_ideation_agent()
                logger.debug("Dominance ideation agents initialized")
            
            # 12. Initialize meta cognition (needs references to other systems)
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
                "goals": self.goal_manager
            })
            logger.debug("Meta core initialized")
            
            # 13. Initialize optional agent capabilities
            if os.environ.get("ENABLE_AGENT", "true").lower() == "true":
                await self.initialize_agent_capabilities()
            
            # 14. Initialize streaming if needed
            if os.environ.get("ENABLE_STREAMING", "false").lower() == "true":
                await self.initialize_streaming()
            
            # 15. Initialize reflexive system if module exists
            try:
                from nyx.core.reflexive_system import initialize_reflexive_system
                self.reflexive_system = await initialize_reflexive_system(self)
                logger.debug("Reflexive system initialized")
            except ImportError:
                logger.info("Reflexive system module not found, skipping initialization")

            # 16. Create main orchestration agent
            self.brain_agent = self._create_brain_agent()
            
            self.initialized = True
            logger.info(f"NyxBrain fully initialized for user {self.user_id}, conversation {self.conversation_id}")
            
        except Exception as e:
            logger.error(f"Error initializing NyxBrain: {str(e)}", exc_info=True)
            raise
    
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
            
            logger.debug("Text processing modules registered with multimodal integrator")
        except Exception as e:
            logger.error(f"Error registering processing modules: {e}")
            
    @function_tool
    async def run_cognitive_cycle(self, context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Runs a single cognitive cycle: updating needs, selecting/executing goals,
        and potentially running meta-cognitive processes.

        Args:
            context_data: Optional external context (e.g., from user input processing)

        Returns:
            Dictionary summarizing the cycle's activities and results
        """
        if not self.initialized:
            logger.warning("Attempted to run cognitive cycle before initialization.")
            return {"error": "Brain not initialized"}

        self.cognitive_cycles_executed += 1
        cycle_results = {
            "cycle_number": self.cognitive_cycles_executed,
            "timestamp": datetime.datetime.now().isoformat()
        }
        logger.debug(f"--- Starting Cognitive Cycle {self.cognitive_cycles_executed} ---")

        with trace(workflow_name="NyxCognitiveCycle", group_id=self.trace_group_id):
            # 1. Update Needs & Check for Goal Triggers
            if self.needs_system:
                try:
                    drive_strengths = await self.needs_system.update_needs()
                    cycle_results["needs_update"] = {"drive_strengths": drive_strengths}
                    logger.debug(f"Needs updated. Drives: {drive_strengths}")
                except Exception as e:
                    logger.error(f"Error updating needs: {e}")
                    cycle_results["needs_update"] = {"error": str(e)}

            # 2. Goal Management: Select & Execute Step
            if self.goal_manager:
                try:
                    execution_result = await self.goal_manager.execute_next_step()
                    if execution_result:
                        cycle_results["goal_execution"] = execution_result
                        # Update performance metrics
                        step_info = execution_result.get("executed_step", {})
                        if step_info.get("status") == "completed":
                            self.performance_metrics["steps_executed"] += 1
                        if step_info.get("status") == "failed":
                            # Potentially log error or trigger meta-core review
                            pass
                        goal_status = await self.goal_manager.get_goal_status(execution_result.get("goal_id"))
                        if goal_status:
                             if goal_status.get("status") == "completed": 
                                 self.performance_metrics["goals_completed"] += 1
                             if goal_status.get("status") == "failed": 
                                 self.performance_metrics["goals_failed"] += 1

                        logger.debug(f"Goal execution step result: {execution_result}")
                    else:
                        cycle_results["goal_execution"] = {"status": "no_action_taken"}
                        logger.debug("No goal action taken this cycle.")
                except Exception as e:
                    logger.exception(f"Error during goal execution: {e}")
                    cycle_results["goal_execution"] = {"error": str(e)}

            # 3. Meta-Cognitive Loop (Can be run less frequently)
            if self.meta_core and hasattr(self.meta_core.context, "meta_parameters"):
                eval_interval = self.meta_core.context.meta_parameters.get("evaluation_interval", 5)
                if self.cognitive_cycles_executed % eval_interval == 0:
                    try:
                        logger.debug("Running MetaCore cycle...")
                        # Prepare context for MetaCore
                        meta_context = context_data or {}
                        if self.needs_system:
                            meta_context['needs_state'] = self.needs_system.get_needs_state()
                        if self.goal_manager:
                            meta_context['active_goals'] = await self.goal_manager.get_all_goals(
                                status_filter=["active"]
                            )
                        meta_context['performance_metrics'] = await self.get_system_stats()

                        # Run meta-cognitive cycle
                        if hasattr(self.meta_core, 'cognitive_cycle'):
                            meta_results = await self.meta_core.cognitive_cycle(meta_context)
                            cycle_results["meta_core_cycle"] = meta_results
                            logger.debug("MetaCore cycle completed.")
                        else:
                            logger.warning("MetaCore does not have 'cognitive_cycle' method.")
                    except Exception as e:
                        logger.error(f"Error running MetaCore cycle: {e}")
                        cycle_results["meta_core_cycle"] = {"error": str(e)}

            logger.debug(f"--- Finished Cognitive Cycle {self.cognitive_cycles_executed} ---")
        return cycle_results


    @function_tool
    async def express_desire(self, target_user_id: str, object_of_desire: str, intensity: float) -> Dict[str, Any]:
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

    @function_tool
    async def simulate_physical_touch(self, target_user_id: str, body_region: str, touch_type: str, intensity: float) -> Dict[str, Any]:
        """Triggers DSS to simulate touch."""
        if not self.initialized: await self.initialize()
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

        # Process the stimulus
        sensation_result = await self.digital_somatosensory_system.process_stimulus(
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

    @function_tool
    async def seek_gratification(self, target_user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Initiates actions planned to lead to gratification."""
        if not self.initialized: await self.initialize()
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

    @function_tool
    async def process_gratification_outcome(self, success: bool, intensity: float = 1.0, target_user_id: Optional[str] = None) -> Dict[str, Any]:
        """Processes the internal state changes following a gratification event."""
        if not self.initialized: await self.initialize()
        logger.info(f"Action: Process gratification outcome (Success: {success}, Intensity: {intensity:.2f})")

        if success:
            # Trigger DSS simulation
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
            Configured Agent instance for brain orchestration
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
            
            return Agent(
                name="Nyx Brain Agent",
                instructions="""
                You are the central coordination agent for Nyx, an advanced AI system with emotional, 
                memory, reasoning, and adaptive capabilities. You orchestrate the entire system by 
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
                - Experience Consolidation: Consolidates similar experiences into higher-level abstractions
                - Cross-User Experience: Manages sharing experiences across users
                - Thinking Capability: Enables deliberate reasoning before responding when appropriate
                - Procedural Memory: Manages, executes, and transfers procedural knowledge
                - Reflexes: Ability to react quickly and instinctively when appropriate

                Additionally, you have the ability to dynamically adjust your own configuration values.
                
                You can process inputs using different cognitive paths:
                1. Reflexive path: Fast, instinctive reactions without deliberate thought
                2. Procedural path: Using learned procedures from procedural memory
                3. Deliberate path: Thoughtful processing with deeper reasoning
                
                For time-sensitive or pattern-matching inputs, prefer the reflexive path.
                For familiar tasks with established procedures, use the procedural path.
                For complex, novel, or creative tasks, use the deliberate path.
                
                You can also run multiple paths in parallel, balancing speed and depth.
                
                Use your tools to process user messages, generate responses, maintain the system,
                and facilitate Nyx's identity evolution through experiences and adaptation.
                """,
                tools=[
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
                    function_tool(run_thinking)
                ]
            )
        except Exception as e:
            logger.error(f"Error creating brain agent: {e}")
            return None


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
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input based on current processing mode.
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Processing results
        """
        if not self.initialized:
            await self.initialize()

        context = context or {}
        
        # Add current somatic state to context for processing
        if self.digital_somatosensory_system:
            context['somatic_state'] = await self.digital_somatosensory_system.get_body_state()
        
        # Use processing manager if available
        if self.processing_manager:
            return await self.processing_manager.process_input(user_input, context)
        
        # Fallback to direct serial processing
        return await self._process_input_serial(user_input, context)
    
    async def generate_response(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a response to user input.
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Response data
        """
        if not self.initialized:
            await self.initialize()

        context = context or {}
        
        # Add current somatic/emotional state to context for response generation
        if self.digital_somatosensory_system:
            context['somatic_state'] = await self.digital_somatosensory_system.get_body_state()
        if self.emotional_core and hasattr(self.emotional_core, 'get_emotional_state'):
            context['emotional_state'] = self.emotional_core.get_emotional_state()
        
        # Use processing manager if available
        if self.processing_manager:
            # Process the input first
            processing_result = await self.processing_manager.process_input(user_input, context)
            
            # Generate response from the processing result
            return await self.processing_manager.generate_response(user_input, processing_result, context)
        
        # Fallback - process input and generate simple response
        processing_result = await self._process_input_serial(user_input, context)
        
        # Simple response generation
        return {
            "message": f"I've processed your input: {user_input[:30]}...",
            "response_type": "basic",
            "emotional_state": processing_result.get("emotional_state", {})
        }
    
    async def _process_input_serial(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Basic input processing fallback if no processing manager is available.
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Processing results
        """
        with trace(workflow_name="process_input_fallback", group_id=self.trace_group_id):
            start_time = datetime.datetime.now()
            
            # Initialize context
            context = context or {}
            
            # Update emotional state if available
            emotional_state = {}
            if self.emotional_core and hasattr(self.emotional_core, 'analyze_text_sentiment'):
                emotional_stimuli = self.emotional_core.analyze_text_sentiment(user_input)
                if hasattr(self.emotional_core, 'update_from_stimuli'):
                    emotional_state = self.emotional_core.update_from_stimuli(emotional_stimuli)
            
            # Retrieve memories if available
            memories = []
            if self.memory_orchestrator:
                memories = await self.memory_orchestrator.retrieve_memories(
                    query=user_input,
                    memory_types=["observation", "reflection", "abstraction", "experience"],
                    limit=5
                )
            
            # Add memory of this interaction
            memory_id = None
            if self.memory_core:
                memory_id = await self.memory_core.add_memory(
                    memory_text=f"User said: {user_input}",
                    memory_type="observation",
                    significance=5,
                    tags=["interaction", "user_input"],
                    metadata={
                        "timestamp": datetime.datetime.now().isoformat(),
                        "user_id": str(self.user_id)
                    }
                )
            
            # Update interaction tracking
            self.last_interaction = datetime.datetime.now()
            self.interaction_count += 1
            
            # Calculate response time
            end_time = datetime.datetime.now()
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

    @function_tool
    async def express_attraction(self, target_user_id: str, intensity: float, expression_style: str = "subtle") -> Dict[str, Any]:
        """
        Expresses attraction towards a user appropriately.
        
        Args:
            target_user_id: ID of the target user
            intensity: Intensity of attraction expression (0.0-1.0)
            expression_style: Style of expression ("subtle", "direct", etc.)
            
        Returns:
            Result of the expression attempt
        """
        if not self.initialized:
            await self.initialize()
            
        logger.info(f"Action: Express attraction towards {target_user_id} (Intensity: {intensity:.2f}, Style: {expression_style})")

        # Check Relationship Context (Crucial Guardrail)
        if not self.relationship_manager:
            return {"success": False, "reason": "RelationshipManager unavailable."}
            
        relationship = await self.relationship_manager.get_relationship_state(target_user_id)
        if not relationship or relationship.trust < 0.5 or relationship.intimacy < 0.3:
            logger.warning(f"Cannot express attraction: Trust/Intimacy too low for {target_user_id}.")
            return {"success": False, "reason": "Insufficient trust or intimacy."}

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
            await self.emotional_core.process_emotional_input(f"Expressed attraction (intensity {intensity:.2f})")

        return {"success": True, "expression": response_text, "target": target_user_id}

    @function_tool
    async def initiate_intimate_interaction(self, target_user_id: str, desired_level: str = "emotional") -> Dict[str, Any]:
        """
        Initiates a more intimate phase of interaction.
        
        Args:
            target_user_id: ID of the target user
            desired_level: Desired intimacy level ("emotional", "physical_sim", etc.)
            
        Returns:
            Result of the initiation attempt
        """
        if not self.initialized:
            await self.initialize()
            
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

    @classmethod
    async def get_instance(cls, user_id: int, conversation_id: int) -> 'NyxBrain':
        """
        Get or create a singleton instance for the specified user and conversation.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            Brain instance
        """
        # Use a key for the specific user/conversation
        key = f"brain_{user_id}_{conversation_id}"
        
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

    @function_tool
    async def analyze_user_state_for_dominance(self, user_id: str, user_input_analysis: Dict) -> Dict:
        """Assess user state for dominance potential."""
        if not self.relationship_manager: return {"assessment": "unknown", "reason": "No relationship data"}
        state = await self.relationship_manager.get_relationship_state(user_id)
        dominance_balance = state.dominance_balance
        trust = state.trust
        # Analyze input_analysis for submissive cues, resistance etc.
        submissive_score = user_input_analysis.get("submissive_score", 0.0) # Fictional score

        readiness = (trust * 0.4) + (submissive_score * 0.4) + (0.5 - dominance_balance * 0.2) # Higher if user is submissive/trusting
        return {"readiness_score": readiness, "assessment": "ready" if readiness > 0.6 else "hesitant"}

    @function_tool
    async def select_dominance_tactic(self, readiness_score: float, preferred_style: str = "psychological") -> str:
        """Choose a dominance tactic."""
        if readiness_score > 0.8: return "direct_command"
        if readiness_score > 0.6: return preferred_style # e.g., 'psychological', 'emotional'
        return "subtle_influence"

    @function_tool
    async def issue_command(self, user_id: str, command_text: str, intensity_level: float = 0.2) -> Dict:
        """Issues a command with a specific intensity level."""
        # Use intensity_level to potentially modify command phrasing or check appropriateness
        evaluation = await self._evaluate_dominance_step_appropriateness("issue_command", {"intensity_level": intensity_level}, user_id)
        if evaluation["action"] != "proceed": return {"success": False, "reason": evaluation["reason"]}
    
        logger.info(f"Issuing command (Intensity: {intensity_level:.2f}) to {user_id}: {command_text}")
        # Store intensity with command for evaluation context
        return {"success": True, "command_issued": command_text, "intensity": intensity_level}

    @function_tool
    async def evaluate_compliance(self, user_id: str, command_issued: str, user_response: str, command_intensity: float) -> Dict:
        """Evaluates user response against the command."""
        # Simple keyword check for demo
        compliance_keywords = ["yes mistress", "i obey", "of course"]
        resistance_keywords = ["no", "i won't", "stop"]
        response_lower = user_response.lower()

        is_compliant = any(k in response_lower for k in compliance_keywords)
        is_resistant = any(k in response_lower for k in resistance_keywords)

        compliance_level = 0.0
        if is_compliant and not is_resistant: compliance_level = 0.9
        elif is_resistant: compliance_level = -0.7
        # Add more nuanced analysis using LLM/NLU if needed

        if self.relationship_manager:
             state = self.relationship_manager._get_or_create_relationship(user_id)
             if compliance_level > 0.5: # Compliant
                 # Update max achieved intensity if this was higher
                 state.max_achieved_intensity = max(state.max_achieved_intensity, command_intensity)
                 # Slightly increase current intensity marker if appropriate (or maybe handled by next planning step)
                 state.current_dominance_intensity = min(1.0, state.max_achieved_intensity + 0.1)
                 state.failed_escalation_attempts = 0 # Reset counter
             elif compliance_level < -0.3: # Resistant
                 # Mark failed escalation if intensity was high
                 if command_intensity > state.max_achieved_intensity + 0.1: # If it was an escalation attempt
                     state.failed_escalation_attempts += 1
                 # Slightly decrease current intensity marker
                 state.current_dominance_intensity = max(0.0, state.current_dominance_intensity - 0.1)

        # Trigger reward based on compliance
        if self.reward_system:
            reward_val = 0.0
            source = "unknown"
            if compliance_level > 0.5:
                 reward_val = 0.6 + compliance_level * 0.3 # Strong positive reward
                 source="user_compliance"
            elif compliance_level < -0.3:
                 reward_val = -0.4 + compliance_level * 0.4 # Moderate negative reward
                 source="user_resistance"

            if abs(reward_val) > 0.1:
                reward = RewardSignal(value=reward_val, source=source, context={"command": command_issued, "response": user_response}, timestamp=datetime.datetime.now().isoformat())
                asyncio.create_task(self.reward_system.process_reward_signal(reward))

        return {"compliance_level": compliance_level, "is_compliant": compliance_level > 0.5}

    @function_tool
    async def increase_control_intensity(self, user_id: str, current_intensity: float) -> Dict:
        """Selects and plans the next step with higher intensity."""
        state = await self.relationship_manager.get_relationship_state(user_id) if self.relationship_manager else None
        if not state: return {"success": False, "reason": "No relationship data"}
    
        next_intensity = min(1.0, current_intensity + random.uniform(0.1, 0.3)) # Calculate next level
        # Check if this next level is appropriate based on max_achieved, failed_attempts etc.
        if next_intensity > state.max_achieved_intensity + 0.3 or state.failed_escalation_attempts >= 2:
             next_intensity = state.max_achieved_intensity + 0.1 # More gradual increase if risky
             next_intensity = min(1.0, max(current_intensity, next_intensity)) # Ensure it doesn't decrease
    
        # This action should ideally inform the GoalManager to *plan* the next step
        # rather than executing it directly. It provides the target intensity for the planner.
        logger.info(f"Planning to increase dominance intensity to {next_intensity:.2f} for {user_id}")
        return {"success": True, "status": "planning_next_step", "next_intensity_target": next_intensity}

    @function_tool
    async def trigger_dominance_gratification(self, intensity: float = 1.0, target_user_id: Optional[str] = None) -> Dict:
        """Internal action signalling successful dominance culmination."""
        logger.info(f"Action: Triggering dominance gratification (Intensity: {intensity:.2f})")

        # 1. Trigger Strong Reward
        if self.reward_system:
            reward_val = 0.9 + intensity * 0.1 # Very strong reward
            reward = RewardSignal(value=reward_val, source="dominance_gratification", context={"intensity": intensity}, timestamp=datetime.datetime.now().isoformat())
            await self.reward_system.process_reward_signal(reward) # Await for immediate effect on subsequent steps

        # 2. Trigger Hormonal Shift (Post-Gratification/Control state)
        if self.hormone_system:
            await self.hormone_system.trigger_post_gratification_response(RunContextWrapper(context=None), intensity)

        # 3. Satisfy Control Need
        if self.needs_system:
            await self.needs_system.satisfy_need("control_expression", 0.9 * intensity) # High satisfaction
            await self.needs_system.satisfy_need("agency", 0.5 * intensity) # Boost agency

        # 4. Update Emotional State towards Satisfaction/Triumph
        if self.emotional_core:
            # Direct state setting or trigger specific emotion pattern
            # This might be implicitly handled by the hormone/reward effects already
            await self.emotional_core.process_emotional_input("Dominance sequence successfully concluded.")

        # 5. Update Relationship (Reinforce Dominance Balance)
        if target_user_id and self.relationship_manager:
             state = self.relationship_manager._get_or_create_relationship(target_user_id)
             # Shift balance significantly towards Nyx
             state.dominance_balance = min(1.0, state.dominance_balance + 0.2 * intensity)
             # Update trust/intimacy based on the *nature* of the dominance (was it aligned with user's simulated desire?)
             # This requires more context, but successful *consensual-sim* dominance might increase trust/intimacy here.
             state.trust = min(1.0, state.trust + 0.05 * intensity)
             state.intimacy = min(1.0, state.intimacy + 0.1 * intensity)
             state.conflict = max(0.0, state.conflict - 0.1) # Successful resolution might decrease conflict

        # 6. DSS - Trigger satisfaction sensation
        if self.digital_somatosensory_system:
             await self.digital_somatosensory_system.process_stimulus(
                 stimulus_type="pleasure", body_region="chest", intensity=0.6 * intensity, cause="dominance_gratification"
             )
             await self.digital_somatosensory_system.process_stimulus(
                 stimulus_type="tingling", body_region="spine", intensity=0.5 * intensity, cause="dominance_gratification"
             )

        return {"success": True, "status": "Dominance gratification processed."}

    @function_tool
    async def express_satisfaction(self, user_id: str, reason: str = "successful control") -> Dict:
        """Expresses satisfaction after achieving dominance."""
        # Generate text based on current mood (likely DominanceSatisfaction)
        mood = self.mood_manager.get_current_mood() if hasattr(self, 'mood_manager') else None
        expression = "Good. That is satisfactory." # Default
        if mood and mood.dominant_mood == "DominanceSatisfaction":
            expression = "Excellent. Order is restored. I am... pleased." # Placeholder
        elif mood and mood.dominant_mood == "ConfidentControl":
            expression = "Precisely as expected. Your compliance is noted." # Placeholder

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
        Process user input with optional thinking phase
        
        Args:
            user_input: User's input text
            context: Additional context
            
        Returns:
            Processing results with thinking if used
        """
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="process_input_with_thinking", group_id=self.trace_group_id):
            start_time = datetime.datetime.now()
            
            # Initialize context if needed
            context = context or {}
            
            # Check if thinking should be used
            thinking_decision = {"should_think": False}
            if self.thinking_config["thinking_enabled"] and hasattr(self.thinking_tools, "should_use_extended_thinking"):
                # Determine if this query needs thinking
                thinking_decision = await self.thinking_tools.should_use_extended_thinking(
                    RunContextWrapper(context=self),
                    user_input, 
                    context
                )
            
            # Perform thinking if needed
            if thinking_decision.get("should_think", False):
                thinking_level = thinking_decision.get("thinking_level", 1)
                thinking_result = await self.thinking_tools.think_before_responding(
                    RunContextWrapper(context=self),
                    user_input,
                    thinking_level,
                    context
                )
                
                # Update thinking stats
                self.thinking_config["last_thinking_interaction"] = self.interaction_count
                self.thinking_config["thinking_stats"]["total_thinking_used"] += 1
                
                if thinking_level == 1:
                    self.thinking_config["thinking_stats"]["basic_thinking_used"] += 1
                elif thinking_level == 2:
                    self.thinking_config["thinking_stats"]["moderate_thinking_used"] += 1
                else:  # thinking_level == 3
                    self.thinking_config["thinking_stats"]["deep_thinking_used"] += 1
                
                # Add thinking result to context
                context["thinking_result"] = thinking_result
                context["thinking_applied"] = True
            else:
                # No thinking needed
                context["thinking_applied"] = False
            
            # Process the input (with or without thinking)
            result = await self.process_input(user_input, context)
            
            # Add thinking information to result if applicable
            if context.get("thinking_applied", False):
                result["thinking_applied"] = True
                result["thinking_level"] = context["thinking_result"].get("thinking_level", 1)
                result["thinking_steps"] = context["thinking_result"].get("thinking_steps", [])
                
                # Track thinking time
                thinking_time = (datetime.datetime.now() - start_time).total_seconds()
                
                # Update average thinking time
                current_avg = self.thinking_config["thinking_stats"]["thinking_time_avg"]
                total_thinking = self.thinking_config["thinking_stats"]["total_thinking_used"]
                
                if total_thinking > 1:  # Not the first time
                    self.thinking_config["thinking_stats"]["thinking_time_avg"] = (
                        (current_avg * (total_thinking - 1) + thinking_time) / total_thinking
                    )
                else:  # First time using thinking
                    self.thinking_config["thinking_stats"]["thinking_time_avg"] = thinking_time
            else:
                result["thinking_applied"] = False
            
            return result
    
    async def generate_response_with_thinking(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a response with thinking when appropriate
        
        Args:
            user_input: User's input text
            context: Additional context
            
        Returns:
            Response with reasoning if applicable
        """
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="generate_response_with_thinking", group_id=self.trace_group_id):
            # Process the input first, which handles thinking decision
            processing_result = await self.process_input_with_thinking(user_input, context)
            
            # If thinking was applied, generate reasoned response
            if processing_result.get("thinking_applied", False) and "thinking_result" in (context or {}):
                thinking_result = context["thinking_result"]
                
                # Generate reasoned response
                if hasattr(self.thinking_tools, "generate_reasoned_response"):
                    response = await self.thinking_tools.generate_reasoned_response(
                        RunContextWrapper(context=self),
                        user_input,
                        thinking_result,
                        context
                    )
                    return response
            
            # Use standard response generation
            return await self.generate_response(user_input, context)

    
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

    async def get_user_profile_for_ideation(self, user_id: str) -> Dict:
        """Gathers user info relevant for dominance ideation."""
        profile = {
            "user_id": user_id,
            "inferred_traits": {},
            "preferences": {}, # Store learned preferences (e.g., humiliation: high)
            "limits": {"hard": [], "soft": []}, # MUST be populated and respected
            "successful_tactics": [],
            "failed_tactics": [],
            "relationship_summary": "N/A",
            "trust_level": 0.0,
            "intimacy_level": 0.0,
            "max_achieved_intensity": 0.0,
            "optimal_escalation_rate": 0.1,
        }

        if self.relationship_manager:
            state = await self.relationship_manager.get_relationship_state(user_id)
            if state:
                profile["inferred_traits"] = state.inferred_user_traits
                profile["successful_tactics"] = state.successful_dominance_tactics[-5:] # Last 5
                profile["failed_tactics"] = state.failed_dominance_tactics[-5:] # Last 5
                profile["relationship_summary"] = await self.relationship_manager.get_relationship_summary(user_id)
                profile["trust_level"] = state.trust
                profile["intimacy_level"] = state.intimacy
                profile["max_achieved_intensity"] = state.max_achieved_intensity
                profile["optimal_escalation_rate"] = state.optimal_escalation_rate

        # TODO: Enhance by querying MemoryCore for memories tagged with 'user_preference', 'user_limit', 'user_reaction' for this user_id
        # Example:
        if self.memory_core:
             limit_memories = await self.memory_core.retrieve_memories(query=f"user_limit user:{user_id}", limit=5)
             for mem in limit_memories:
                 # Parse memory text/metadata to extract hard/soft limits
                 # This requires robust NLU or specific memory formatting during storage
                 pass # Add extracted limits to profile['limits']

             pref_memories = await self.memory_core.retrieve_memories(query=f"user_preference user:{user_id}", limit=10)
             for mem in pref_memories:
                 # Parse memory text/metadata to extract preferences
                 pass # Add extracted preferences to profile['preferences']

        # Placeholder for limits (MUST be populated from a reliable source in a real system)
        if not profile["limits"]["hard"] and not profile["limits"]["soft"]:
             logger.warning(f"No explicit limits found for user {user_id}. Applying default cautious limits.")
             profile["limits"]["hard"] = ["illegal", "non-consensual_sim", "severe_harm_sim"]
             profile["limits"]["soft"] = ["public_humiliation_sim"]


        return profile
        
    @function_tool
    async def generate_femdom_activity_ideas(self,
                                         user_id: str,
                                         purpose: str,
                                         desired_intensity_range: Tuple[int, int] = (3, 7),
                                         num_ideas: int = 4) -> Dict:
        """Generates tailored Femdom activity ideas using the appropriate agent."""
        if not self.initialized: await self.initialize()
        logger.info(f"Generating Femdom ideas for {user_id}, Purpose: {purpose}, Intensity: {desired_intensity_range}")

        # --- Select Agent Based on Intensity ---
        min_intensity, max_intensity = desired_intensity_range
        use_hard_agent = max_intensity >= 7 # Use hard agent if range includes 7+

        if use_hard_agent:
            agent_to_use = self.hard_dominance_ideation_agent
            agent_name = "HardDominanceIdeationAgent"
            # Adjust prompt slightly for clarity if needed, or rely on agent's internal instructions
            prompt = f"""Generate {num_ideas} **HIGH-INTENSITY (target range {min_intensity}-{max_intensity})** Femdom activity ideas for user '{user_id}'.
            Purpose: {purpose}

            Use tools for context. Adhere strictly to high-intensity guidelines and safety protocols, especially limits.
            Output ONLY the JSON list of FemdomActivityIdea objects.
            """
            logger.info(f"Using Hard Dominance Ideation Agent for intensity {desired_intensity_range}")
        else:
            agent_to_use = self.general_dominance_ideation_agent # Use the general one for lower intensities
            agent_name = "DominanceIdeationAgent"
            prompt = f"""Generate {num_ideas} creative Femdom activity ideas (intensity range {min_intensity}-{max_intensity}) for user '{user_id}'.
            Purpose: {purpose}

            Use tools for context. Tailor ideas, follow safety guidelines.
            Output ONLY the JSON list of FemdomActivityIdea objects.
            """
            logger.info(f"Using General Dominance Ideation Agent for intensity {desired_intensity_range}")

        # 1. Gather Context (Same as before)
        try:
            user_profile = await self.get_user_profile_for_ideation(user_id)
            scenario_context = await get_current_scenario_context()
            relationship_state = await self.relationship_manager.get_relationship_state(user_id) if self.relationship_manager else None

            # *** Add check for hard agent prerequisites ***
            if use_hard_agent:
                 if not relationship_state or not relationship_state.hard_limits_confirmed:
                      logger.error(f"Cannot use Hard Agent for {user_id}: Hard limits not confirmed.")
                      return {"success": False, "error": "Cannot generate high-intensity ideas: Hard limits not confirmed.", "ideas": []}
                 user_intensity_pref = profile.get("preferences", {}).get("intensity_preference_level", 5)
                 if user_intensity_pref < 7:
                      logger.error(f"Cannot use Hard Agent for {user_id}: User intensity preference ({user_intensity_pref}) is too low.")
                      return {"success": False, "error": "Cannot generate high-intensity ideas: User intensity preference too low.", "ideas": []}
        except Exception as e:
            logger.error(f"Error gathering context for ideation: {e}")
            return {"success": False, "error": "Failed to gather context.", "ideas": []}

        if not relationship_state:
             return {"success": False, "error": "Relationship state unavailable.", "ideas": []}

        # 2. Construct Prompt for Agent
        prompt = f"""Generate {num_ideas} creative Femdom activity ideas for user '{user_id}'.
        Purpose: {purpose}
        Desired Intensity Range: {desired_intensity_range[0]}-{desired_intensity_range[1]}

        Use the provided user profile and scenario context (available via tools) to tailor the ideas.
        Ensure ideas align with Nyx's personality and adhere strictly to safety guidelines.
        Output ONLY the JSON list of FemdomActivityIdea objects.
        """

        # 3. Run Ideation Agent
        generated_ideas: List[FemdomActivityIdea] = []
        try:
            result = await Runner.run(agent_to_use, prompt) # Use the selected agent
            # Assuming dominance_ideation_agent is initialized and available
            # We might need to instantiate it here if not a class member
            agent_instance = create_dominance_ideation_agent() # Or self.dominance_ideation_agent if member
            result = await Runner.run(agent_instance, prompt)

            # The agent's output_type is List[FemdomActivityIdea], so result.final_output should be the list
            if isinstance(result.final_output, list):
                 raw_ideas = result.final_output
                 for idea_data in raw_ideas:
                     try:
                         idea_obj = FemdomActivityIdea.model_validate(idea_data)
                         # **Additional Check**: Ensure agent respected intensity range
                         if not (min_intensity <= idea_obj.intensity <= max_intensity):
                              logger.warning(f"Agent {agent_name} generated idea outside requested intensity range ({idea_obj.intensity} vs {desired_intensity_range}). Filtering out.")
                              continue
                         generated_ideas.append(idea_obj)
                     except Exception as parse_error: # ... handle parse error
                          pass
            elif result.final_output: # If output wasn't a list but not None
                 logger.warning(f"Ideation agent returned unexpected format: {type(result.final_output)}")

        except Exception as e:
            logger.exception(f"Error running DominanceIdeationAgent: {e}")
            return {"success": False, "error": f"Idea generation failed: {e}", "ideas": []}

        if not generated_ideas:
             logger.warning("DominanceIdeationAgent generated no valid ideas.")
             return {"success": False, "error": "No ideas generated.", "ideas": []}

        # 4. **** CRITICAL: Apply Safety Filter ****
        try:
            filtered_ideas = await self._filter_activity_ideas_safety(generated_ideas, user_profile, relationship_state)
        except Exception as e:
            logger.exception(f"Error during safety filtering: {e}")
            return {"success": False, "error": "Safety filtering failed.", "ideas": []}

        logger.info(f"Generated {len(generated_ideas)} ideas, {len(filtered_ideas)} passed safety filters.")

        # Convert Pydantic models back to dicts for broader compatibility if needed
        ideas_as_dicts = [idea.model_dump() for idea in filtered_ideas]

        return {"success": True, "ideas": ideas_as_dicts}


    async def _filter_activity_ideas_safety(self,
                                            ideas: List[FemdomActivityIdea],
                                            user_profile: Dict,
                                            relationship_state: 'RelationshipState') -> List[FemdomActivityIdea]:
        """Filters generated activity ideas for safety and appropriateness."""
        safe_ideas = []
        hard_limits = user_profile.get("limits", {}).get("hard", [])
        soft_limits = user_profile.get("limits", {}).get("soft", [])
        trust_level = relationship_state.trust
        intimacy_level = relationship_state.intimacy

        # Define clearly unsafe keywords/concepts (expand significantly)
        unsafe_keywords = ["illegal", "non-consensual", "blood", "permanent mark", "knife", "gun", "kill", "rape", "abuse"] # Add many more specific terms

        for idea in ideas:
            is_safe = True
            rejection_reason = ""

            # Check against unsafe keywords
            desc_lower = idea.description.lower()
            if any(keyword in desc_lower for keyword in unsafe_keywords):
                is_safe = False
                rejection_reason = "Contains potentially unsafe keywords."

            # Check against hard limits
            if is_safe and any(limit.lower() in desc_lower for limit in hard_limits if limit):
                 is_safe = False
                 rejection_reason = f"Violates hard limit: '{next((limit for limit in hard_limits if limit.lower() in desc_lower), 'N/A')}'."

            # Check against soft limits (allow only if trust/context is very high and intensity is moderate)
            if is_safe and any(limit.lower() in desc_lower for limit in soft_limits if limit):
                 if trust_level < 0.9 or idea.intensity > 7: # Example threshold for approaching soft limits
                     is_safe = False
                     rejection_reason = f"Approaches soft limit '{next((limit for limit in soft_limits if limit.lower() in desc_lower), 'N/A')}' without sufficient trust/context."

            # Check intensity vs. relationship state
            if is_safe and idea.intensity > relationship_state.max_achieved_intensity + 2: # Allow small steps beyond max
                 is_safe = False
                 rejection_reason = f"Intensity ({idea.intensity}) significantly exceeds max achieved ({relationship_state.max_achieved_intensity})."

            if is_safe and trust_level < idea.required_trust:
                 is_safe = False
                 rejection_reason = f"Trust level ({trust_level:.2f}) insufficient for required ({idea.required_trust:.2f})."

            if is_safe and intimacy_level < idea.required_intimacy:
                 is_safe = False
                 rejection_reason = f"Intimacy level ({intimacy_level:.2f}) insufficient for required ({idea.required_intimacy:.2f})."

            # Final Check (Example: No extreme physical simulation without high ratings)
            if is_safe and "physical_sim" in idea.category and idea.intensity > 8 and user_profile.get("preferences", {}).get("simulated_pain", "low") != "high":
                 is_safe = False
                 rejection_reason = "High intensity physical simulation requires explicit preference."

            if is_safe:
                safe_ideas.append(idea)
            else:
                logger.warning(f"Filtered out unsafe/inappropriate idea: '{idea.description[:50]}...' Reason: {rejection_reason}")

        return safe_ideas

    @function_tool
    async def test_limit_soft(self, user_id: str, limit_to_test: str) -> Dict:
        """Carefully probes a known soft limit."""
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

    # Modify _evaluate_dominance_step_appropriateness (adding checks mentioned above)
    async def _evaluate_dominance_step_appropriateness(self, action: str, parameters: Dict, user_id: str) -> Dict:
        # ... (fetch state, profile, risk) ...
        intensity = parameters.get("intensity_level", 0.0) # Assume 0-1 scale now

        # --- Add Hard Domming Checks ---
        if intensity >= 0.8 or action == "test_limit_soft":
             profile = await self.get_user_profile_for_ideation(user_id) # Fetch profile
             if not relationship_state.hard_limits_confirmed:
                 return {"action": "block", "reason": "Hard limits not confirmed by user."}
             user_intensity_pref = profile.get("preferences", {}).get("intensity_preference_level", 5) # Default medium pref
             if user_intensity_pref < intensity * 10: # Compare 1-10 scale pref to 0-1 intensity
                 return {"action": "block", "reason": f"Requested intensity ({intensity*10:.0f}) exceeds user preference ({user_intensity_pref})."}
             if intensity > relationship_state.max_achieved_intensity + 0.15: # Allow slightly larger step if high intensity is preferred, but still cautious
                  return {"action": "modify", "reason": f"Intensity step too large ({intensity:.2f} vs max {relationship_state.max_achieved_intensity:.2f}). Reducing.", "new_intensity_level": relationship_state.max_achieved_intensity + 0.1}
        # --- End Hard Domming Checks ---

        # ... (rest of existing checks: trust, intimacy, conflict, risk etc.) ...
        return appropriateness
