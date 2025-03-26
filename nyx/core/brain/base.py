# nyx/core/brain/base.py
import logging
import asyncio
import datetime
import random
import os
import math
from typing import Dict, List, Any, Optional, Tuple, Union

from agents import Agent, Runner, trace, function_tool, handoff, RunContextWrapper
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError

from nyx.core.brain.processing.manager import ProcessingManager
from nyx.core.brain.adaptation.self_config import SelfConfigManager
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

logger = logging.getLogger(__name__)

class NyxBrain:
    """
    Central integration point for all Nyx systems.
    Uses composition to delegate to specialized components.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
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
        
        # Component managers
        self.processing_manager = None
        self.self_config_manager = None
        
        # State tracking
        self.initialized = False
        self.last_interaction = datetime.datetime.now()
        self.interaction_count = 0
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
        
        # Error tracking
        self.error_registry = {
            "unhandled_errors": [],
            "handled_errors": [],
            "error_counts": {},
            "error_recovery_strategies": {},
            "error_recovery_stats": {}
        }
        
        # Agent system
        self.agent_capabilities_initialized = False
        
        logger.info(f"NyxBrain initialized for user {self.user_id}, conversation {self.conversation_id}")
    
    async def initialize(self):
        """Initialize all subsystems"""
        if self.initialized:
            return
        
        logger.info(f"Initializing NyxBrain for user {self.user_id}, conversation {self.conversation_id}")
        
        # Dynamic imports to avoid circular dependencies
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
        from nyx.core.reasoning_agents import integrated_reasoning_agent, triage_agent as reasoning_triage_agent
        from nyx.api.thinking_tools import ThinkingTools
        
        try:
            # Initialize hormone system first (needed by other systems)
            self.hormone_system = HormoneSystem()
            
            # Initialize emotional core with hormone system
            self.emotional_core = EmotionalCore(hormone_system=self.hormone_system)
            
            # Initialize memory system
            self.memory_core = MemoryCore(self.user_id, self.conversation_id)
            await self.memory_core.initialize()
            
            # Initialize memory orchestrator
            self.memory_orchestrator = MemoryOrchestrator(self.user_id, self.conversation_id)
            await self.memory_orchestrator.initialize()
            
            # Initialize reflection engine
            self.reflection_engine = ReflectionEngine()
            
            # Initialize experience interface
            self.experience_interface = ExperienceInterface(self.memory_core, self.emotional_core)
            
            # Initialize identity evolution system with hormone system reference
            self.identity_evolution = IdentityEvolutionSystem(hormone_system=self.hormone_system)
            
            # Initialize experience consolidation system
            self.experience_consolidation = ExperienceConsolidationSystem(
                memory_core=self.memory_core,
                experience_interface=self.experience_interface
            )
            
            # Initialize cross-user experience manager
            self.cross_user_manager = CrossUserExperienceManager(
                memory_core=self.memory_core,
                experience_interface=self.experience_interface
            )
            
            # Initialize internal feedback system
            self.internal_feedback = InternalFeedbackSystem()
            
            # Initialize attention and perception systems
            self.attentional_controller = AttentionalController(
                emotional_core=self.emotional_core
            )
            
            # Use integrated reasoning agent as reasoning core
            self.reasoning_core = integrated_reasoning_agent
            self.reasoning_triage_agent = reasoning_triage_agent
            
            # Initialize multimodal integrator
            self.multimodal_integrator = EnhancedMultiModalIntegrator(
                reasoning_core=self.reasoning_core,
                attentional_controller=self.attentional_controller
            )
            
            # Initialize reward system
            self.reward_system = RewardSignalProcessor(
                emotional_core=self.emotional_core,
                identity_evolution=self.identity_evolution
            )
            
            # Initialize temporal perception
            self.temporal_perception = TemporalPerception()
            await self.temporal_perception.initialize(self, None)
            
            # Initialize procedural memory
            self.procedural_memory = ProceduralMemoryManager()
            self.agent_enhanced_memory = AgentEnhancedMemoryManager(self.procedural_memory)
            
            # Initialize thinking tools
            self.thinking_tools = ThinkingTools()
            
            # Register feature extractors and integration strategies for multimodal
            await self._register_processing_modules()
            
            # Initialize component managers
            self.processing_manager = ProcessingManager(self)
            await self.processing_manager.initialize()
            
            self.self_config_manager = SelfConfigManager(self)
            
            # Initialize dynamic adaptation system
            self.dynamic_adaptation = DynamicAdaptationSystem()
            
            # Initialize knowledge core
            self.knowledge_core = KnowledgeCoreAgents()
            await self.knowledge_core.initialize()
            
            # Initialize meta core last, as it needs references to other systems
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
                "procedural": self.agent_enhanced_memory
            })
            
            # Initialize agent capabilities if needed
            if os.environ.get("ENABLE_AGENT", "true").lower() == "true":
                await self.initialize_agent_capabilities()
            
            # Initialize streaming if needed
            if os.environ.get("ENABLE_STREAMING", "false").lower() == "true":
                await self.initialize_streaming()
            
            # Initialize reflexive system if module exists
            try:
                from nyx.core.reflexive_system import initialize_reflexive_system
                await initialize_reflexive_system(self)
            except ImportError:
                logger.info("Reflexive system module not found, skipping initialization")
            
            # Create main brain agent with all function tools
            self.brain_agent = self._create_brain_agent()
            
            self.initialized = True
            logger.info(f"NyxBrain fully initialized for user {self.user_id}, conversation {self.conversation_id}")
            
        except Exception as e:
            logger.error(f"Error initializing NyxBrain: {str(e)}")
            raise
    
    async def initialize_agent_capabilities(self):
        """Initialize the agent capabilities for roleplay and narrative"""
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
        """Initialize streaming capabilities if needed"""
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
        """Register processing modules for multimodal integration"""
        # Register text modality processors
        if hasattr(self, "multimodal_integrator"):
            await self.multimodal_integrator.register_feature_extractor(
                "text", self._extract_text_features
            )
            
            await self.multimodal_integrator.register_expectation_modulator(
                "text", self._modulate_text_perception
            )
            
            await self.multimodal_integrator.register_integration_strategy(
                "text", self._integrate_text_pathways
            )
    
    async def _extract_text_features(self, text_data):
        """Extract features from text input (bottom-up processing)"""
        features = {
            "length": len(text_data),
            "word_count": len(text_data.split()),
            "sentiment": 0.0,  # Placeholder for actual sentiment analysis
            "entities": [],  # Placeholder for named entity recognition
            "commands": [],  # Placeholder for command recognition
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
        """Apply top-down expectations to modulate text perception"""
        # Start with unmodified features
        modulated_features = bottom_up_features.copy()
        
        # Track which expectations influenced perception
        influenced_by = []
        total_influence = 0.0
        
        # Apply each expectation
        for expectation in expectations:
            # Skip if modality doesn't match
            if expectation.target_modality != "text":
                continue
                
            # Get expectation pattern and strength
            pattern = expectation.pattern
            strength = expectation.strength
            
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
                            influenced_by.append(f"{expectation.source}:{key}")
                            total_influence += strength
            else:
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
                                    influenced_by.append(f"{expectation.source}:entity:{entity}")
                                    total_influence += strength
        
        # Calculate overall influence strength
        influence_strength = min(1.0, total_influence / max(1, len(influenced_by)))
        
        return {
            "features": modulated_features,
            "influence_strength": influence_strength,
            "influenced_by": influenced_by
        }
    
    async def _integrate_text_pathways(self, bottom_up_result, top_down_result):
        """Integrate bottom-up and top-down processing for text"""
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
    
    def _create_brain_agent(self) -> Agent:
        """Create the main brain agent that coordinates all subsystems"""
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
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input based on current processing mode
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Processing results
        """
        if not self.initialized:
            await self.initialize()
        
        # Use processing manager if available
        if self.processing_manager:
            return await self.processing_manager.process_input(user_input, context)
        
        # Fallback to direct serial processing
        return await self._process_input_serial(user_input, context)
    
    async def generate_response(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a response to user input
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Response data
        """
        if not self.initialized:
            await self.initialize()
        
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
        Basic input processing fallback if no processing manager is available
        
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
            if self.emotional_core:
                emotional_stimuli = self.emotional_core.analyze_text_sentiment(user_input)
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
        Run maintenance on all systems
        
        Returns:
            Maintenance results
        """
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="run_maintenance", group_id=self.trace_group_id):
            results = {}
            
            # Run hormone maintenance
            if self.hormone_system:
                try:
                    hormone_result = await self.hormone_system.update_hormone_cycles(RunContextWrapper(context=None))
                    results["hormone_maintenance"] = hormone_result
                    
                    # Update identity from hormones if identity evolution is available
                    if self.identity_evolution:
                        identity_update = await self.identity_evolution.update_identity_from_hormones(RunContextWrapper(context=None))
                        results["hormone_identity_update"] = identity_update
                except Exception as e:
                    logger.error(f"Error in hormone maintenance: {str(e)}")
                    results["hormone_maintenance"] = {"error": str(e)}
            
            # Run memory maintenance
            if self.memory_orchestrator:
                try:
                    memory_result = await self.memory_orchestrator.run_maintenance()
                    results["memory_maintenance"] = memory_result
                except Exception as e:
                    logger.error(f"Error in memory maintenance: {str(e)}")
                    results["memory_maintenance"] = {"error": str(e)}
            
            # Run meta core maintenance if available
            if self.meta_core:
                try:
                    meta_result = await self.meta_core.improve_meta_parameters()
                    results["meta_maintenance"] = meta_result
                except Exception as e:
                    logger.error(f"Error in meta maintenance: {str(e)}")
                    results["meta_maintenance"] = {"error": str(e)}
            
            # Run knowledge core maintenance if available
            if self.knowledge_core:
                try:
                    knowledge_result = await self.knowledge_core.run_integration_cycle()
                    results["knowledge_maintenance"] = knowledge_result
                except Exception as e:
                    logger.error(f"Error in knowledge maintenance: {str(e)}")
                    results["knowledge_maintenance"] = {"error": str(e)}
            
            # Run experience consolidation if available
            if self.experience_consolidation:
                try:
                    consolidation_result = await self.experience_consolidation.run_consolidation_cycle()
                    results["experience_consolidation"] = consolidation_result
                except Exception as e:
                    logger.error(f"Error in experience consolidation: {str(e)}")
                    results["experience_consolidation"] = {"error": str(e)}
            
            # Update cross-user clusters if available
            if self.cross_user_manager:
                try:
                    cluster_result = await self.cross_user_manager.update_user_clusters()
                    results["user_clustering"] = cluster_result
                except Exception as e:
                    logger.error(f"Error updating user clusters: {str(e)}")
                    results["user_clustering"] = {"error": str(e)}
            
            # Run procedural memory maintenance if available
            if self.agent_enhanced_memory:
                try:
                    procedural_result = await self.agent_enhanced_memory.memory_manager.run_maintenance()
                    results["procedural_maintenance"] = procedural_result
                except Exception as e:
                    logger.error(f"Error in procedural maintenance: {str(e)}")
                    results["procedural_maintenance"] = {"error": str(e)}
            
            results["maintenance_time"] = datetime.datetime.now().isoformat()
            return results
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about all systems
        
        Returns:
            System statistics
        """
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="get_system_stats", group_id=self.trace_group_id):
            stats = {}
            
            # Get memory stats if available
            if self.memory_orchestrator and self.memory_core:
                try:
                    memory_stats = await self.memory_core.get_memory_stats()
                    stats["memory_stats"] = memory_stats
                except Exception as e:
                    logger.error(f"Error getting memory stats: {str(e)}")
                    stats["memory_stats"] = {"error": str(e)}
            
            # Get emotional state if available
            if self.emotional_core:
                try:
                    emotional_state = self.emotional_core.get_emotional_state()
                    dominant_emotion, dominant_value = self.emotional_core.get_dominant_emotion()
                    
                    stats["emotional_state"] = {
                        "emotions": emotional_state,
                        "dominant_emotion": dominant_emotion,
                        "dominant_value": dominant_value,
                        "valence": self.emotional_core.get_emotional_valence(),
                        "arousal": self.emotional_core.get_emotional_arousal()
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
            
            # Get performance metrics
            avg_response_time = sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"]) if self.performance_metrics["response_times"] else 0
            
            stats["performance_metrics"] = {
                "memory_operations": self.performance_metrics["memory_operations"],
                "emotion_updates": self.performance_metrics["emotion_updates"],
                "reflections_generated": self.performance_metrics["reflections_generated"],
                "experiences_shared": self.performance_metrics["experiences_shared"],
                "cross_user_experiences_shared": self.performance_metrics.get("cross_user_experiences_shared", 0),
                "avg_response_time": avg_response_time
            }
            
            # Get meta stats if available
            if self.meta_core:
                try:
                    meta_stats = await self.meta_core.get_feedback_stats()
                    stats["meta_stats"] = meta_stats
                except Exception as e:
                    logger.error(f"Error getting meta stats: {str(e)}")
                    stats["meta_stats"] = {"error": str(e)}
            
            # Get knowledge stats if available
            if self.knowledge_core:
                try:
                    knowledge_stats = await self.knowledge_core.get_knowledge_statistics()
                    stats["knowledge_stats"] = knowledge_stats
                except Exception as e:
                    logger.error(f"Error getting knowledge stats: {str(e)}")
                    stats["knowledge_stats"] = {"error": str(e)}
            
            # Get procedural memory stats if available
            if self.agent_enhanced_memory:
                try:
                    procedures = list(self.agent_enhanced_memory.procedures.keys())
                    stats["procedural_stats"] = {
                        "total_procedures": len(procedures),
                        "available_procedures": procedures[:10] if len(procedures) > 10 else procedures,
                        "procedure_domains": list(set(p.get("domain", "general") for p in self.agent_enhanced_memory.procedures.values())),
                        "execution_count": getattr(self.agent_enhanced_memory.agents.agent_context, "run_stats", {}).get("total_runs", 0)
                    }
                except Exception as e:
                    logger.error(f"Error getting procedural memory stats: {str(e)}")
                    stats["procedural_stats"] = {"error": str(e)}
            
            # Get identity state if available
            if self.identity_evolution:
                try:
                    identity_profile = await self.identity_evolution.get_identity_profile()
                    stats["identity_stats"] = {
                        "trait_count": len(identity_profile.get("traits", {})),
                        "preference_count": sum(len(prefs) for prefs in identity_profile.get("preferences", {}).values()),
                        "dominant_traits": sorted(identity_profile.get("traits", {}).items(), key=lambda x: x[1], reverse=True)[:3]
                    }
                except Exception as e:
                    logger.error(f"Error getting identity stats: {str(e)}")
                    stats["identity_stats"] = {"error": str(e)}
            
            # Get experience stats if available
            if self.experience_interface:
                try:
                    stats["experience_stats"] = {
                        "experiences_shared": self.performance_metrics["experiences_shared"],
                        "cross_user_experiences_shared": self.performance_metrics.get("cross_user_experiences_shared", 0),
                        "consolidations_performed": self.performance_metrics.get("experience_consolidations", 0)
                    }
                except Exception as e:
                    logger.error(f"Error getting experience stats: {str(e)}")
                    stats["experience_stats"] = {"error": str(e)}
            
            # Get reflection stats
            stats["reflection_stats"] = {
                "reflections_generated": self.performance_metrics["reflections_generated"]
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
    
    async def get_identity_state(self) -> Dict[str, Any]:
        """
        Get the current state of Nyx's identity
        
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
        Adapt experience sharing parameters based on user feedback
        
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
        Run the experience consolidation process
        
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
    
    # nyx/core/brain/base.py (continued)
    async def register_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register an error from any component for central management
        
        Args:
            error_data: Error information
            
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
        Execute a recovery strategy for an error
        
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
        """Execute a retry strategy"""
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
        """Execute a fallback strategy"""
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
        """Execute a reset strategy"""
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
        """Handle a critical error"""
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
        """Clean up old errors from the registry"""
        # Keep only the latest 1000 errors
        if len(self.error_registry["unhandled_errors"]) > 1000:
            self.error_registry["unhandled_errors"] = self.error_registry["unhandled_errors"][-1000:]
        if len(self.error_registry["handled_errors"]) > 1000:
            self.error_registry["handled_errors"] = self.error_registry["handled_errors"][-1000:]
    
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
    
    @classmethod
    async def get_instance(cls, user_id: int, conversation_id: int) -> 'NyxBrain':
        """
        Get or create a singleton instance for the specified user and conversation
        
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
