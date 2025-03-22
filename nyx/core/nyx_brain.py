# nyx/core/nyx_brain.py

import logging
import asyncio
import json
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

from agents import Agent, Runner, trace, function_tool, handoff
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
from pydantic import BaseModel, Field

# Import core systems
from nyx.core.emotional_core import EmotionalCore
from nyx.core.memory_core import MemoryCore
from nyx.core.reflection_engine import ReflectionEngine
from nyx.core.experience_interface import ExperienceInterface
from nyx.core.dynamic_adaptation_system import DynamicAdaptationSystem
from nyx.core.internal_feedback_system import InternalFeedbackSystem
from nyx.core.meta_core import MetaCore
from nyx.core.knowledge_core import KnowledgeCoreAgents
from nyx.core.memory_orchestrator import MemoryOrchestrator
from nyx.core.reasoning_agents import integrated_reasoning_agent, triage_agent as reasoning_triage_agent
from nyx.core.component_influence_matrix import ComponentInfluenceMatrix

# Import function tools
from nyx.api.function_tools import (
    add_memory, retrieve_memories, create_reflection, create_abstraction,
    construct_narrative, retrieve_experiences, share_experience,
    get_emotional_state, update_emotion, set_emotion,
    process_input, generate_response, run_maintenance, get_system_stats,
    adapt_to_context, evaluate_response
)

logger = logging.getLogger(__name__)

# =============== Pydantic Models for Input/Output ===============

class UserInput(BaseModel):
    """Input from a user to be processed"""
    user_id: int = Field(..., description="User ID")
    text: str = Field(..., description="User's input text")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class ProcessResult(BaseModel):
    """Result of processing user input"""
    user_input: str = Field(..., description="Original user input")
    emotional_state: Dict[str, Any] = Field(..., description="Current emotional state")
    memories: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved memories")
    memory_count: int = Field(0, description="Number of memories retrieved")
    has_experience: bool = Field(False, description="Whether an experience was found")
    experience_response: Optional[str] = Field(None, description="Experience response if available")
    memory_id: Optional[str] = Field(None, description="ID of stored memory")
    response_time: float = Field(0.0, description="Processing time in seconds")
    context_change: Optional[Dict[str, Any]] = Field(None, description="Context change detection")

class ResponseResult(BaseModel):
    """Result of generating a response"""
    message: str = Field(..., description="Main response message")
    response_type: str = Field(..., description="Type of response")
    emotional_state: Dict[str, Any] = Field(..., description="Current emotional state")
    emotional_expression: Optional[str] = Field(None, description="Emotional expression if any")
    memories_used: List[str] = Field(default_factory=list, description="IDs of memories used")
    memory_count: int = Field(0, description="Number of memories used")
    evaluation: Optional[Dict[str, Any]] = Field(None, description="Response evaluation if available")

# =============== Brain Function Tools ===============

@function_tool
async def process_user_message(ctx, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process a user message with all cognitive systems
    
    Args:
        user_input: User's message text
        context: Optional additional context
    
    Returns:
        Processing results with emotional state, memories, etc.
    """
    brain = ctx.context
    
    # Process through the full system
    result = await brain.process_input(user_input, context)
    return result

@function_tool
async def generate_agent_response(ctx, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate a complete response to the user
    
    Args:
        user_input: User's message text
        context: Optional additional context
    
    Returns:
        Complete response with message, emotional expression, etc.
    """
    brain = ctx.context
    
    # Generate a full response
    response = await brain.generate_response(user_input, context)
    return response

@function_tool
async def run_cognitive_cycle(ctx, context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run a meta-cognitive cycle
    
    Args:
        context_data: Additional context for the cognitive cycle
    
    Returns:
        Results of the cognitive cycle
    """
    brain = ctx.context
    
    # Run meta-cognitive cycle
    if brain.meta_core:
        result = await brain.meta_core.cognitive_cycle(context_data)
        return result
    
    return {"error": "Meta core not initialized"}

@function_tool
async def get_brain_stats(ctx) -> Dict[str, Any]:
    """
    Get comprehensive statistics about all systems
    
    Returns:
        Statistics for all systems
    """
    brain = ctx.context
    
    # Get comprehensive stats
    stats = await brain.get_system_stats()
    return stats

@function_tool
async def perform_maintenance(ctx) -> Dict[str, Any]:
    """
    Run maintenance on all systems
    
    Returns:
        Maintenance results
    """
    brain = ctx.context
    
    # Run maintenance on all systems
    result = await brain.run_maintenance()
    return result

@function_tool
async def get_influence_matrix_report(ctx) -> Dict[str, Any]:
    """
    Get a report on the current state of the component influence matrix
    
    Returns:
        Influence matrix report data
    """
    brain = ctx.context
    
    if brain.influence_matrix:
        report = brain.influence_matrix.get_influence_report()
        return report
    
    return {"error": "Influence matrix not initialized"}

@function_tool
async def optimize_influence_matrix(ctx) -> Dict[str, Any]:
    """
    Run optimization on the component influence matrix
    
    Returns:
        Optimization results
    """
    brain = ctx.context
    
    if brain.influence_matrix:
        result = await brain.optimize_influence_matrix()
        return result
    
    return {"error": "Influence matrix not initialized"}

# =============== Main Brain Class ===============

class NyxBrain:
    """
    Central integration point for all Nyx systems.
    Handles cross-component communication and provides a unified API using the OpenAI Agents SDK.
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
        
        # Component influence matrix - initialized in initialize()
        self.influence_matrix = None
        
        # State tracking
        self.initialized = False
        self.last_interaction = datetime.datetime.now()
        self.interaction_count = 0
        
        # Legacy influence parameters - kept for backward compatibility
        # but actual values will come from influence matrix
        self.memory_to_emotion_influence = 0.3
        self.emotion_to_memory_influence = 0.4
        
        # Performance monitoring
        self.performance_metrics = {
            "memory_operations": 0,
            "emotion_updates": 0,
            "reflections_generated": 0,
            "experiences_shared": 0,
            "response_times": []
        }
        
        # Caching
        self.context_cache = {}
        
        # Main brain agent - initialized in initialize()
        self.brain_agent = None
        
        # Component activity tracking for influence learning
        self.component_activity = {}
        
        # Trace group ID for connecting traces
        self.trace_group_id = f"nyx-brain-{user_id}-{conversation_id}"
    
    @classmethod
    async def get_instance(cls, user_id: int, conversation_id: int) -> 'NyxBrain':
        """Get or create a singleton instance for the specified user and conversation"""
        # Use a key for the specific user/conversation
        key = f"brain_{user_id}_{conversation_id}"
        
        # Check if instance exists in a global registry
        if not hasattr(cls, '_instances'):
            cls._instances = {}
            
        if key not in cls._instances:
            instance = cls(user_id, conversation_id)
            await instance.initialize()
            cls._instances[key] = instance
        
        return cls._instances[key]
    
    async def initialize(self):
        """Initialize all subsystems"""
        if self.initialized:
            return
        
        logger.info(f"Initializing NyxBrain for user {self.user_id}, conversation {self.conversation_id}")
        
        # Initialize influence matrix first
        self.influence_matrix = ComponentInfluenceMatrix()
        
        # Initialize core components
        self.emotional_core = EmotionalCore()
        
        # Initialize memory system
        self.memory_core = MemoryCore(self.user_id, self.conversation_id)
        await self.memory_core.initialize()
        
        # Initialize memory orchestrator
        self.memory_orchestrator = MemoryOrchestrator(self.user_id, self.conversation_id)
        await self.memory_orchestrator.initialize()
        
        # Initialize reflection engine
        self.reflection_engine = ReflectionEngine()
        
        # Initialize experience interface with memory core and emotional core
        self.experience_interface = ExperienceInterface(self.memory_core, self.emotional_core)
        
        # Initialize internal feedback system
        self.internal_feedback = InternalFeedbackSystem()
        
        # Initialize dynamic adaptation system
        self.dynamic_adaptation = DynamicAdaptationSystem()
        
        # Initialize knowledge core
        self.knowledge_core = KnowledgeCoreAgents()
        await self.knowledge_core.initialize()
        
        # Use integrated reasoning agent as reasoning core
        self.reasoning_core = integrated_reasoning_agent
        
        # Initialize meta core last, as it needs references to other systems
        self.meta_core = MetaCore()
        await self.meta_core.initialize({
            "memory": self.memory_core,
            "emotion": self.emotional_core,
            "reasoning": self.reasoning_core,
            "reflection": self.reflection_engine,
            "adaptation": self.dynamic_adaptation,
            "feedback": self.internal_feedback
        })
        
        # Initialize component activity tracking
        self.component_activity = {
            "memory": 0.0,
            "emotion": 0.0,
            "reasoning": 0.0,
            "reflection": 0.0,
            "adaptation": 0.0,
            "experience": 0.0,
            "meta": 0.0,
            "knowledge": 0.0,
            "feedback": 0.0
        }
        
        # Initialize main brain agent
        self.brain_agent = self._create_brain_agent()
        
        self.initialized = True
        logger.info(f"NyxBrain initialized for user {self.user_id}, conversation {self.conversation_id}")
    
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
            - Experience Interface: Shares relevant experiences
            - Dynamic Adaptation: Adapts to changing contexts
            - Internal Feedback: Evaluates system performance
            - Meta Core: Handles meta-cognition and self-improvement
            - Knowledge Core: Manages knowledge and reasoning
            - Component Influence Matrix: Manages dynamic interactions between components
            
            Use your tools to process user messages, generate responses, and maintain the system.
            """,
            tools=[
                process_user_message,
                generate_agent_response,
                run_cognitive_cycle,
                get_brain_stats,
                perform_maintenance,
                get_influence_matrix_report,
                optimize_influence_matrix
            ]
        )
    
    async def process_input(self, 
                          user_input: str, 
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input and update all systems.
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Processing results with relevant memories, emotional state, etc.
        """
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="process_input", group_id=self.trace_group_id):
            start_time = datetime.datetime.now()
            
            # Reset component activity tracking for this interaction
            self._reset_component_activity()
            
            # Run meta-cognitive cycle if available
            meta_result = {}
            if self.meta_core:
                try:
                    meta_result = await self.meta_core.cognitive_cycle(context or {})
                    # Track meta core activity
                    self.component_activity["meta"] = 0.7
                except Exception as e:
                    logger.error(f"Error in meta-cognitive cycle: {str(e)}")
                    meta_result = {"error": str(e)}
                    self.component_activity["meta"] = 0.1
            
            # Update interaction tracking
            self.last_interaction = datetime.datetime.now()
            self.interaction_count += 1
            
            # Initialize context
            context = context or {}
            # Add user input to context
            context["user_input"] = user_input
            
            # Process emotional impact of input
            emotional_stimuli = self.emotional_core.analyze_text_sentiment(user_input)
            emotional_state = self.emotional_core.update_from_stimuli(emotional_stimuli)
            
            # Track emotion core activity
            emotional_stimuli_magnitude = sum(emotional_stimuli.values())
            self.component_activity["emotion"] = 0.3 + (0.7 * min(1.0, emotional_stimuli_magnitude))
            
            # Add emotional state to context for memory retrieval
            context["emotional_state"] = emotional_state
            
            # Get dynamic influence weight from emotion to memory
            emotion_to_memory_influence = 0.4  # Default value
            if self.influence_matrix:
                emotion_to_memory_influence = self.influence_matrix.calculate_contextual_influence(
                    "emotion", "memory", context
                )
                
            # Apply emotion-to-memory influence when retrieving memories
            # by including emotional state in context with appropriate weight
            retrieval_context = context.copy()
            if emotion_to_memory_influence > 0.3:  # Only apply if influence is significant
                # Enhance emotional context based on influence level
                retrieval_context["emotional_weight"] = emotion_to_memory_influence
            
            # Retrieve relevant memories using memory orchestrator
            memories = await self.memory_orchestrator.retrieve_memories(
                query=user_input,
                memory_types=context.get("memory_types", ["observation", "reflection", "abstraction", "experience"]), 
                limit=context.get("memory_limit", 5)
            )
            
            # Track memory activity based on retrieval success
            self.component_activity["memory"] = 0.4 + (0.6 * min(1.0, len(memories) / 10))
            self.performance_metrics["memory_operations"] += 1
            
            # Get dynamic influence weight from memory to emotion
            memory_to_emotion_influence = 0.3  # Default value
            if self.influence_matrix:
                memory_to_emotion_influence = self.influence_matrix.calculate_contextual_influence(
                    "memory", "emotion", context
                )
            
            # Update emotional state based on retrieved memories
            if memories:
                memory_emotional_impact = await self._calculate_memory_emotional_impact(memories, context)
                # Apply memory-to-emotion influence with dynamic weight
                for emotion, value in memory_emotional_impact.items():
                    self.emotional_core.update_emotion(emotion, value * memory_to_emotion_influence)
                
                # Get updated emotional state
                emotional_state = self.emotional_core.get_emotional_state()
            
            self.performance_metrics["emotion_updates"] += 1
            
            # Determine experience sharing based on dynamic influence
            experience_to_response_influence = 0.5  # Default value
            if self.influence_matrix:
                experience_to_response_influence = self.influence_matrix.calculate_contextual_influence(
                    "experience", "reasoning", context
                )
            
            # Check if experience sharing is appropriate
            should_share_experience = self._should_share_experience(user_input, context) or experience_to_response_influence > 0.7
            experience_result = None
            
            if should_share_experience:
                # Retrieve and format experience
                experience_result = await self.experience_interface.share_experience_enhanced(
                    query=user_input,
                    context_data=context
                )
                self.performance_metrics["experiences_shared"] += 1
                
                # Track experience activity
                if experience_result and experience_result.get("has_experience", False):
                    self.component_activity["experience"] = 0.8
                else:
                    self.component_activity["experience"] = 0.2
            
            # Add memory of this interaction
            memory_text = f"User said: {user_input}"
            
            memory_id = await self.memory_core.add_memory(
                memory_text=memory_text,
                memory_type="observation",
                memory_scope="game",
                significance=5,
                tags=["interaction", "user_input"],
                metadata={
                    "emotional_context": self.emotional_core.get_formatted_emotional_state(),
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
            
            # Check for reasoning requirements
            if self._is_reasoning_query(user_input):
                self.component_activity["reasoning"] = 0.8
            else:
                self.component_activity["reasoning"] = 0.3
            
            # Check for context change using dynamic adaptation
            context_change_result = None
            if self.dynamic_adaptation:
                # Prepare context for change detection
                context_for_adaptation = {
                    "user_input": user_input,
                    "emotional_state": emotional_state,
                    "memories_retrieved": len(memories),
                    "has_experience": experience_result["has_experience"] if experience_result else False,
                    "interaction_count": self.interaction_count
                }
                
                # Get dynamic influence for adaptation
                adaptation_influence = 0.5  # Default value
                if self.influence_matrix:
                    adaptation_influence = self.influence_matrix.calculate_contextual_influence(
                        "adaptation", "meta", context
                    )
                
                # Only run adaptation if influence is sufficient
                if adaptation_influence > 0.3:
                    # Detect context change
                    context_change_result = await self.dynamic_adaptation.detect_context_change(context_for_adaptation)
                    
                    # Track adaptation activity
                    if context_change_result and getattr(context_change_result, "significant_change", False):
                        self.component_activity["adaptation"] = 0.9
                    else:
                        self.component_activity["adaptation"] = 0.4
            
            # Calculate response time
            end_time = datetime.datetime.now()
            response_time = (end_time - start_time).total_seconds()
            self.performance_metrics["response_times"].append(response_time)
            
            # Update influence matrix with component activity and estimated outcomes
            interaction_outcomes = {
                "success_rate": context.get("success_rate", 0.5),
                "response_quality": context.get("response_quality", 0.5),
                "user_satisfaction": context.get("user_satisfaction", 0.5)
            }
            
            await self._update_influence_matrix(self.component_activity, interaction_outcomes)
            
            # Return processing results in a structured format using the ProcessResult model
            result = {
                "user_input": user_input,
                "emotional_state": emotional_state,
                "memories": memories,
                "memory_count": len(memories),
                "has_experience": experience_result["has_experience"] if experience_result else False,
                "experience_response": experience_result["response_text"] if experience_result and experience_result["has_experience"] else None,
                "memory_id": memory_id,
                "response_time": response_time,
                "context_change": context_change_result,
                "meta_result": meta_result,
                "component_activity": self.component_activity
            }
            
            return result
    
    async def generate_response(self, 
                             user_input: str, 
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a complete response to user input.
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Response data including main message and supporting information
        """
        with trace(workflow_name="generate_response", group_id=self.trace_group_id):
            # Process the input first
            processing_result = await self.process_input(user_input, context)
            
            # Enhanced context with processing results
            enhanced_context = context.copy() if context else {}
            enhanced_context.update({
                "processing_result": processing_result,
                "component_activity": self.component_activity
            })
            
            # Determine response strategy using dynamic influence weights
            experience_influence = 0.5  # Default value
            reasoning_influence = 0.5  # Default value
            
            if self.influence_matrix:
                experience_influence = self.influence_matrix.calculate_contextual_influence(
                    "experience", "reasoning", enhanced_context
                )
                reasoning_influence = self.influence_matrix.calculate_contextual_influence(
                    "reasoning", "emotion", enhanced_context
                )
            
            # Determine if experience response should be used
            if processing_result["has_experience"] and experience_influence > 0.4:
                main_response = processing_result["experience_response"]
                response_type = "experience"
                self.component_activity["experience"] = 0.9
            else:
                # For reasoning-related queries, use the reasoning agents
                if self._is_reasoning_query(user_input) and reasoning_influence > 0.3:
                    try:
                        reasoning_result = await Runner.run(
                            reasoning_triage_agent,
                            user_input
                        )
                        main_response = reasoning_result.final_output
                        response_type = "reasoning"
                        self.component_activity["reasoning"] = 0.9
                    except Exception as e:
                        logger.error(f"Error in reasoning response: {str(e)}")
                        # Fallback to standard response
                        main_response = "I understand your question and would like to reason through it with you."
                        response_type = "standard"
                        self.component_activity["reasoning"] = 0.3
                else:
                    # No specific experience to share, generate standard response
                    # In a real implementation, this would be more sophisticated
                    main_response = "I acknowledge your message and have processed it through my systems."
                    response_type = "standard"
            
            # Determine if emotion should be expressed
            emotional_expression_influence = 0.5  # Default
            if self.influence_matrix:
                emotional_expression_influence = self.influence_matrix.calculate_contextual_influence(
                    "emotion", "experience", enhanced_context
                )
            
            should_express_emotion = (
                self.emotional_core.should_express_emotion() or 
                emotional_expression_influence > 0.7
            )
            
            emotional_expression = None
            
            if should_express_emotion:
                try:
                    expression_result = await self.emotional_core.generate_emotional_expression(force=False)
                    if expression_result.get("expressed", False):
                        emotional_expression = expression_result.get("expression", "")
                        self.component_activity["emotion"] = 0.8
                except Exception as e:
                    logger.error(f"Error generating emotional expression: {str(e)}")
                    emotional_expression = self.emotional_core.get_expression_for_emotion()
                    self.component_activity["emotion"] = 0.4
            
            # Package the response
            response_data = {
                "message": main_response,
                "response_type": response_type,
                "emotional_state": processing_result["emotional_state"],
                "emotional_expression": emotional_expression,
                "memories_used": [m["id"] for m in processing_result["memories"]],
                "memory_count": processing_result["memory_count"],
                "component_activity": self.component_activity
            }
            
            # Add memory of this response
            await self.memory_core.add_memory(
                memory_text=f"I responded: {main_response}",
                memory_type="observation",
                memory_scope="game",
                significance=5,
                tags=["interaction", "nyx_response"],
                metadata={
                    "emotional_context": self.emotional_core.get_formatted_emotional_state(),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "response_type": response_type
                }
            )
            
            # Get feedback influence
            feedback_influence = 0.5  # Default
            if self.influence_matrix:
                feedback_influence = self.influence_matrix.calculate_contextual_influence(
                    "feedback", "adaptation", enhanced_context
                )
            
            # Evaluate the response if internal feedback system is available and influence is sufficient
            if self.internal_feedback and feedback_influence > 0.3:
                try:
                    evaluation = await self.internal_feedback.critic_evaluate(
                        aspect="effectiveness",
                        content={"text": main_response, "type": response_type},
                        context={"user_input": user_input}
                    )
                    
                    # Add evaluation to response data
                    response_data["evaluation"] = evaluation
                    
                    # Track feedback activity
                    self.component_activity["feedback"] = 0.7
                except Exception as e:
                    logger.error(f"Error evaluating response: {str(e)}")
                    self.component_activity["feedback"] = 0.2
            
            # Update influence matrix again with final component activity
            interaction_outcomes = {
                "success_rate": response_data.get("evaluation", {}).get("success_rate", 0.5),
                "response_quality": response_data.get("evaluation", {}).get("quality_score", 0.5),
                "user_satisfaction": response_data.get("evaluation", {}).get("user_satisfaction", 0.5)
            }
            
            await self._update_influence_matrix(self.component_activity, interaction_outcomes)
            
            return response_data
    
    async def generate_enhanced_response(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate an enhanced response to user input with adaptation.
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Enhanced response data with adaptation
        """
        with trace(workflow_name="generate_enhanced_response", group_id=self.trace_group_id):
            # Generate standard response
            response_data = await self.generate_response(user_input, context)
            
            # Check if adaptation should be applied based on influence matrix
            adaptation_threshold = 0.5  # Default
            if self.influence_matrix:
                adaptation_threshold = self.influence_matrix.calculate_contextual_influence(
                    "adaptation", "meta", context or {}
                )
            
            # Add adaptive behavior from dynamic adaptation system
            if self.dynamic_adaptation and adaptation_threshold > 0.4:
                try:
                    # Create adaptable context
                    adaptable_context = {
                        "user_input": user_input,
                        "response": response_data["message"],
                        "interaction_type": context.get("interaction_type", "general") if context else "general",
                    }
                    
                    # Detect context change
                    change_result = await self.dynamic_adaptation.detect_context_change(adaptable_context)
                    
                    # Monitor performance
                    performance = await self.dynamic_adaptation.monitor_performance({
                        "success_rate": context.get("success_rate", 0.5) if context else 0.5,
                        "user_satisfaction": context.get("user_satisfaction", 0.5) if context else 0.5,
                        "efficiency": context.get("efficiency", 0.5) if context else 0.5,
                        "response_quality": context.get("response_quality", 0.5) if context else 0.5
                    })
                    
                    # Add adaptation data to response
                    response_data["adaptation"] = {
                        "context_change": change_result,
                        "performance": performance
                    }
                    
                    # Track adaptation activity
                    self.component_activity["adaptation"] = 0.8
                    
                    # If significant change, select strategy
                    if change_result.significant_change:  # Updated to use the proper attribute
                        strategy = await self.dynamic_adaptation.select_strategy(adaptable_context, performance)
                        response_data["adaptation"]["strategy"] = strategy
                except Exception as e:
                    logger.error(f"Error in adaptation: {str(e)}")
                    response_data["adaptation"] = {"error": str(e)}
                    self.component_activity["adaptation"] = 0.2
            
            # Determine if reflection should be included based on influence
            reflection_threshold = 0.5  # Default
            if self.influence_matrix:
                reflection_threshold = self.influence_matrix.calculate_contextual_influence(
                    "reflection", "meta", context or {}
                )
            
            # Generate a meta-cognitive reflection if appropriate
            if (self.interaction_count % 10 == 0 or reflection_threshold > 0.7) and self.reflection_engine:
                try:
                    system_stats = await self.get_system_stats()
                    introspection = await self.reflection_engine.generate_introspection(
                        memory_stats=system_stats["memory_stats"],
                        player_model=context.get("player_model")
                    )
                    response_data["introspection"] = introspection
                    
                    # Track reflection activity
                    self.component_activity["reflection"] = 0.8
                except Exception as e:
                    logger.error(f"Error generating introspection: {str(e)}")
                    self.component_activity["reflection"] = 0.2
            
            # Update influence matrix with final activities
            await self._update_influence_matrix(self.component_activity, {
                "success_rate": 0.5,  # Default as we don't have real feedback yet
                "response_quality": 0.5,
                "user_satisfaction": 0.5
            })
            
            return response_data
    
    async def create_reflection(self, 
                             topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a reflection on memories.
        
        Args:
            topic: Optional topic to focus reflection on
            
        Returns:
            Reflection data
        """
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="create_reflection", group_id=self.trace_group_id):
            # Track reflection activity
            self.component_activity["reflection"] = 0.9
            
            # Get the memory-to-reflection influence
            memory_to_reflection_influence = 0.5  # Default
            if self.influence_matrix:
                memory_to_reflection_influence = self.influence_matrix.get_influence("memory", "reflection")
            
            # Enhance retrieval parameters based on influence
            memory_limit = 5  # Default
            if memory_to_reflection_influence > 0.7:
                memory_limit = 8  # Retrieve more memories if influence is high
            elif memory_to_reflection_influence < 0.3:
                memory_limit = 3  # Retrieve fewer memories if influence is low
            
            # Use orchestrator for reflection creation with dynamic parameters
            reflection_result = await self.memory_orchestrator.create_reflection(
                topic=topic,
                limit=memory_limit
            )
            self.performance_metrics["reflections_generated"] += 1
            
            # Track memory activity
            self.component_activity["memory"] = 0.7
            
            # Update influence matrix
            await self._update_influence_matrix(self.component_activity, {
                "success_rate": 0.7,  # Reflections typically successful
                "response_quality": 0.6,
                "user_satisfaction": 0.6
            })
            
            return reflection_result
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """Run maintenance on all subsystems"""
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="run_maintenance", group_id=self.trace_group_id):
            results = {}
            
            # Optimize influence matrix first
            if self.influence_matrix:
                try:
                    matrix_result = await self.optimize_influence_matrix()
                    results["influence_matrix_maintenance"] = matrix_result
                except Exception as e:
                    logger.error(f"Error in influence matrix maintenance: {str(e)}")
                    results["influence_matrix_maintenance"] = {"error": str(e)}
            
            # Run memory maintenance
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
            
            results["maintenance_time"] = datetime.datetime.now().isoformat()
            return results
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about all systems"""
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="get_system_stats", group_id=self.trace_group_id):
            # Get memory stats
            memory_stats = await self.memory_orchestrator.memory_core.get_memory_stats()
            
            # Get emotional state
            emotional_state = self.emotional_core.get_emotional_state()
            dominant_emotion, dominant_value = self.emotional_core.get_dominant_emotion()
            
            # Get performance metrics
            avg_response_time = sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"]) if self.performance_metrics["response_times"] else 0
            
            # Generate introspection text
            introspection = await self.reflection_engine.generate_introspection(
                memory_stats=memory_stats,
                player_model=None  # Player model would be provided in real implementation
            )
            
            # Get meta core stats if available
            meta_stats = {}
            if self.meta_core:
                try:
                    meta_stats = await self.meta_core.get_feedback_stats()
                except Exception as e:
                    logger.error(f"Error getting meta stats: {str(e)}")
                    meta_stats = {"error": str(e)}
            
            # Get knowledge core stats if available
            knowledge_stats = {}
            if self.knowledge_core:
                try:
                    knowledge_stats = await self.knowledge_core.get_knowledge_statistics()
                except Exception as e:
                    logger.error(f"Error getting knowledge stats: {str(e)}")
                    knowledge_stats = {"error": str(e)}
            
            # Get influence matrix stats if available
            influence_stats = {}
            if self.influence_matrix:
                try:
                    influence_stats = self.influence_matrix.get_influence_report()
                except Exception as e:
                    logger.error(f"Error getting influence stats: {str(e)}")
                    influence_stats = {"error": str(e)}
            
            return {
                "memory_stats": memory_stats,
                "emotional_state": {
                    "emotions": emotional_state,
                    "dominant_emotion": dominant_emotion,
                    "dominant_value": dominant_value,
                    "valence": self.emotional_core.get_emotional_valence(),
                    "arousal": self.emotional_core.get_emotional_arousal()
                },
                "interaction_stats": {
                    "interaction_count": self.interaction_count,
                    "last_interaction": self.last_interaction.isoformat()
                },
                "performance_metrics": {
                    "memory_operations": self.performance_metrics["memory_operations"],
                    "emotion_updates": self.performance_metrics["emotion_updates"],
                    "reflections_generated": self.performance_metrics["reflections_generated"],
                    "experiences_shared": self.performance_metrics["experiences_shared"],
                    "avg_response_time": avg_response_time
                },
                "introspection": introspection,
                "meta_stats": meta_stats,
                "knowledge_stats": knowledge_stats,
                "influence_stats": influence_stats,
                "component_activity": self.component_activity
            }
    
    def _should_share_experience(self, user_input: str, context: Dict[str, Any]) -> bool:
        """Determine if we should share an experience based on input and context"""
        # Get dynamic influence weight for experience sharing
        experience_share_threshold = 0.5  # Default
        if self.influence_matrix and context:
            experience_share_threshold = self.influence_matrix.calculate_contextual_influence(
                "experience", "memory", context
            )
        
        # Check for explicit experience requests
        explicit_request = any(phrase in user_input.lower() for phrase in 
                             ["remember", "recall", "tell me about", "have you done", 
                              "previous", "before", "past", "experience", "what happened"])
        
        if explicit_request:
            return True
        
        # Check if it's a question that could benefit from experience sharing
        is_question = user_input.endswith("?") or user_input.lower().startswith(("what", "how", "when", "where", "why", "who", "can", "could", "do", "did"))
        
        # Adjust based on dynamic threshold
        if is_question and experience_share_threshold > 0.6:
            return True
        
        # Use explicit context setting if available
        if "share_experiences" in context and context["share_experiences"]:
            return True
        
        # Default to not sharing experiences unless explicitly requested
        return False
    
    def _is_reasoning_query(self, user_input: str) -> bool:
        """Determine if a query is likely to need reasoning capabilities"""
        reasoning_indicators = [
            "why", "how come", "explain", "what if", "cause", "reason", "logic", 
            "analyze", "understand", "think through", "consider", "would happen",
            "hypothetical", "scenario", "reasoning", "connect", "relationship",
            "causality", "counterfactual", "consequence", "impact", "effect"
        ]
        
        return any(indicator in user_input.lower() for indicator in reasoning_indicators)
    
    async def _calculate_memory_emotional_impact(self, 
                                          memories: List[Dict[str, Any]], 
                                          context: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate emotional impact from relevant memories using dynamic influence weights
        
        Args:
            memories: List of retrieved memories
            context: Current context
            
        Returns:
            Dictionary of emotion impacts
        """
        impact = {}
        
        # Get dynamic memory-to-emotion influence
        memory_emotion_influence = 0.3  # Default
        if self.influence_matrix:
            memory_emotion_influence = self.influence_matrix.calculate_contextual_influence(
                "memory", "emotion", context
            )
        
        # Scale impact factor based on dynamic influence
        base_impact_scale = 0.1
        impact_scale = base_impact_scale * (memory_emotion_influence / 0.3)  # Scale based on influence
        
        for memory in memories:
            # Extract emotional context
            emotional_context = memory.get("metadata", {}).get("emotional_context", {})
            
            if not emotional_context:
                continue
                
            # Get primary emotion
            primary_emotion = emotional_context.get("primary_emotion")
            primary_intensity = emotional_context.get("primary_intensity", 0.5)
            
            if primary_emotion:
                # Calculate impact based on relevance and recency
                relevance = memory.get("relevance", 0.5)
                
                # Get timestamp if available
                timestamp_str = memory.get("metadata", {}).get("timestamp")
                recency_factor = 1.0
                if timestamp_str:
                    try:
                        timestamp = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        days_old = (datetime.datetime.now() - timestamp).days
                        recency_factor = max(0.5, 1.0 - (days_old / 30))  # Decay over 30 days, minimum 0.5
                    except (ValueError, TypeError):
                        # If timestamp can't be parsed, use default
                        pass
                
                # Calculate final impact value with dynamic scaling
                impact_value = primary_intensity * relevance * recency_factor * impact_scale
                
                # Add to impact dict
                if primary_emotion not in impact:
                    impact[primary_emotion] = 0
                impact[primary_emotion] += impact_value
        
        return impact
    
    def _reset_component_activity(self):
        """Reset component activity tracking for a new interaction"""
        self.component_activity = {
            "memory": 0.0,
            "emotion": 0.0,
            "reasoning": 0.0,
            "reflection": 0.0,
            "adaptation": 0.0,
            "experience": 0.0,
            "meta": 0.0,
            "knowledge": 0.0,
            "feedback": 0.0
        }
    
    async def _update_influence_matrix(self, 
                                 component_activity: Dict[str, float], 
                                 outcomes: Dict[str, float]) -> None:
        """
        Update the influence matrix based on component activity and interaction outcomes
        
        Args:
            component_activity: Dictionary of component activity levels
            outcomes: Dictionary of outcome metrics
        """
        if not self.influence_matrix:
            return
        
        # Prepare interaction data
        interaction_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "active_components": component_activity
        }
        
        # Learn from this interaction
        self.influence_matrix.learn_from_interaction(interaction_data, outcomes)
    
    async def optimize_influence_matrix(self) -> Dict[str, Any]:
        """
        Optimize the influence matrix based on historical data
        
        Returns:
            Optimization results
        """
        if not self.influence_matrix:
            return {"error": "Influence matrix not initialized"}
        
        # Run optimization
        results = self.influence_matrix.optimize_weights()
        
        # Save matrix state if significant changes
        if results.get("changes", 0) > 3:
            try:
                save_path = f"nyx_influence_matrix_{self.user_id}_{self.conversation_id}.json"
                saved = self.influence_matrix.save_to_file(save_path)
                results["saved_to_file"] = saved
                results["save_path"] = save_path
            except Exception as e:
                logger.error(f"Error saving influence matrix: {e}")
                results["save_error"] = str(e)
        
        return results
    
    async def process_user_input_enhanced(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced processing of user input with comprehensive results.
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Comprehensive processing results
        """
        with trace(workflow_name="process_user_input_enhanced", group_id=self.trace_group_id):
            # Process through the brain agent
            if self.brain_agent:
                try:
                    agent_input = {
                        "role": "user",
                        "content": f"Process this user input: {user_input}"
                    }
                    
                    if context:
                        agent_input["context"] = context
                    
                    result = await Runner.run(
                        self.brain_agent,
                        agent_input,
                        context=self
                    )
                    
                    # If agent successfully processed, return the result
                    if hasattr(result, "final_output") and result.final_output:
                        return result.final_output
                except Exception as e:
                    logger.error(f"Error in brain agent processing: {str(e)}")
            
            # If agent failed or isn't available, use direct processing
            result = await self.process_input(user_input, context)
            
            # Add additional processing information
            system_stats = await self.get_system_stats()
            
            # Add influence matrix report
            influence_report = {}
            if self.influence_matrix:
                influence_report = self.influence_matrix.get_influence_report()
            
            # Return enhanced result
            return {
                "input": user_input,
                "emotional_state": result["emotional_state"],
                "memories": result["memories"],
                "memory_count": result["memory_count"],
                "has_experience": result["has_experience"],
                "experience_response": result["experience_response"],
                "memory_id": result["memory_id"],
                "response_time": result["response_time"],
                "component_activity": self.component_activity,
                "influence_report": influence_report,
                "system_stats": {
                    "memory_stats": system_stats["memory_stats"],
                    "emotional_state": system_stats["emotional_state"],
                    "performance_metrics": system_stats["performance_metrics"]
                }
            }

# For backward compatibility in case directly imported
NyxBrainInstance = NyxBrain
