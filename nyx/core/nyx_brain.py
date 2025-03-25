# nyx/core/nyx_brain.py

import logging
import asyncio
import json
import math
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import math

from agents import Agent, Runner, trace, function_tool, handoff, RunContextWrapper
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
from pydantic import BaseModel, Field

from nyx.core.distributed_processing import DistributedProcessingManager
from nyx.core.prediction_engine import PredictionEngine

from issue_tracking_system import IssueTrackingSystem

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
from nyx.core.experience_consolidation import ExperienceConsolidationSystem
from nyx.core.identity_evolution import IdentityEvolutionSystem
from nyx.core.cross_user_experience import CrossUserExperienceManager
from nyx.core.multimodal_integrator import EnhancedMultiModalIntegrator, SensoryInput, ExpectationSignal
from nyx.core.attentional_controller import AttentionalController, AttentionalControl
from nyx.core.reward_system import RewardSignalProcessor, RewardSignal

from nyx.core.procedural_memory import (
    ProceduralMemoryManager, EnhancedProceduralMemoryManager,
    add_procedure, execute_procedure, transfer_procedure,
    get_procedure_proficiency, list_procedures, get_transfer_statistics,
    identify_chunking_opportunities, apply_chunking,
    generalize_chunk_from_steps, find_matching_chunks,
    transfer_chunk, transfer_with_chunking, find_similar_procedures,
    refine_step
)
from nyx.core.procedural_memory.agent import AgentEnhancedMemoryManager

from nyx.core.reflexive_system import ReflexiveSystem, initialize_reflexive_system

from nyx.streamer.gamer_girl import (
    AdvancedGameAgentSystem, 
    GameSessionLearningManager,
    CommentaryType, 
    AnswerType
)

from nyx.api.thinking_tools import (
    should_use_extended_thinking,
    think_before_responding,
    generate_reasoned_response
)

from nyx.streamer.integration import setup_enhanced_streaming

from nyx.nyx_agent_sdk import (
    memory_agent, reflection_agent, decision_agent, nyx_main_agent,
    retrieve_memories, add_memory, determine_image_generation, 
    get_user_model_guidance, generate_image_from_scene,
    AgentContext, MemoryReflection, NarrativeResponse, ContentModeration,
    initialize_agents, ResponseFilter, Runner
)

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
    cross_user_experience: bool = Field(False, description="Whether experience is from another user")
    memory_id: Optional[str] = Field(None, description="ID of stored memory")
    response_time: float = Field(0.0, description="Processing time in seconds")
    context_change: Optional[Dict[str, Any]] = Field(None, description="Context change detection")
    identity_impact: Optional[Dict[str, Any]] = Field(None, description="Impact on identity")

class ResponseResult(BaseModel):
    """Result of generating a response"""
    message: str = Field(..., description="Main response message")
    response_type: str = Field(..., description="Type of response")
    emotional_state: Dict[str, Any] = Field(..., description="Current emotional state")
    emotional_expression: Optional[str] = Field(None, description="Emotional expression if any")
    memories_used: List[str] = Field(default_factory=list, description="IDs of memories used")
    memory_count: int = Field(0, description="Number of memories used")
    evaluation: Optional[Dict[str, Any]] = Field(None, description="Response evaluation if available")
    experience_sharing_adapted: bool = Field(False, description="Whether experience sharing was adapted")

class AdaptationResult(BaseModel):
    """Result of adaptation process"""
    strategy_id: str = Field(..., description="ID of the selected strategy")
    context_change: Dict[str, Any] = Field(..., description="Context change information")
    confidence: float = Field(..., description="Confidence in strategy selection")
    adaptations: Dict[str, Any] = Field(..., description="Applied adaptations")

class IdentityState(BaseModel):
    """Current state of Nyx's identity"""
    top_preferences: Dict[str, float] = Field(..., description="Top preferences with scores")
    top_traits: Dict[str, float] = Field(..., description="Top traits with scores")
    identity_reflection: str = Field(..., description="Reflection on identity")
    identity_evolution: Dict[str, Any] = Field(..., description="Identity evolution metrics")

class StimulusData(BaseModel):
    """Input stimulus that may trigger a reflexive response"""
    data: Dict[str, Any] = Field(..., description="Stimulus data patterns")
    domain: Optional[str] = Field(None, description="Optional domain to limit patterns")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context information")
    priority: Optional[str] = Field("normal", description="Processing priority (normal, high, critical)")

class ReflexRegistrationInput(BaseModel):
    """Input for registering a new reflex pattern"""
    name: str = Field(..., description="Unique name for this pattern")
    pattern_data: Dict[str, Any] = Field(..., description="Pattern definition data")
    procedure_name: str = Field(..., description="Name of procedure to execute when triggered")
    threshold: float = Field(0.7, description="Matching threshold (0.0-1.0)")
    priority: int = Field(1, description="Priority level (higher values take precedence)")
    domain: Optional[str] = Field(None, description="Optional domain for specialized responses")
    context_template: Optional[Dict[str, Any]] = Field(None, description="Template for context to pass to procedure")

class ReflexResponse(BaseModel):
    """Result of processing a stimulus with reflexes"""
    success: bool = Field(..., description="Whether a reflex was successfully triggered")
    pattern_name: Optional[str] = Field(None, description="Name of the triggered pattern if successful")
    reaction_time_ms: float = Field(..., description="Reaction time in milliseconds")
    output: Optional[Dict[str, Any]] = Field(None, description="Output from the procedure execution")
    match_score: Optional[float] = Field(None, description="Match score for the pattern")


# =============== Brain Function Tools ===============

def create_reflex_agent(reflexive_system):
    """Create an agent specifically for handling reflexes"""
    return Agent(
        name="Reflex Agent",
        instructions="""
        You are a specialized agent for handling reflexive, fast responses without deliberate thinking.
        You process stimuli that need immediate reactions, similar to human reflexes or muscle memory.
        
        When presented with a stimulus, you will:
        1. Quickly determine if it matches any known reflex patterns
        2. If a match is found, execute the associated procedure immediately
        3. Return the result of the procedure execution
        
        You focus on speed and pattern recognition, not deliberate reasoning.
        """,
        tools=[
            function_tool(reflexive_system.process_stimulus_fast, 
                         name_override="process_reflex",
                         description_override="Process stimulus with minimal overhead for fastest possible reaction")
        ],
        output_type=ReflexResponse
    )

async def track_issues(self, func, *args, **kwargs):
    """Decorator to track issues in method execution"""
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        # Log the error to the issue tracking system
        context = f"Method: {func.__name__}, Args: {args}, Kwargs: {kwargs}"
        await self.issue_tracker.process_observation(
            f"Exception encountered: {str(e)}\nTraceback: {traceback.format_exc()}",
            context=context
        )
        # Re-raise the exception
        raise

async def initialize_streaming(self, video_source=0, audio_source=None):
    """
    Initialize the streaming system with full brain integration
    
    Args:
        video_source: Video capture source (0 for webcam)
        audio_source: Audio capture source
        
    Returns:
        Streaming core system
    """
    # Use the integration setup function
    self.streaming_core = await setup_enhanced_streaming(self, video_source, audio_source)
    
    # Set brain reference in the streaming system
    self.streaming_core.streaming_system.set_nyx_brain(self)
    
    # Initialize learning manager with brain access
    self.streaming_core.learning_manager = GameSessionLearningManager(self, self.streaming_core)
    
    # Register memory functions for streaming
    self.store_streaming_memory = self.streaming_core.memory_mapper.store_gameplay_memory
    self.retrieve_streaming_memories = self.streaming_core.memory_mapper.retrieve_relevant_memories
    self.create_streaming_reflection = self.streaming_core.memory_mapper.create_streaming_reflection
    self.add_streaming_procedure = self.agent_enhanced_memory.create_procedure
    self.execute_streaming_procedure = self.agent_enhanced_memory.execute_procedure
    
    # Register streaming control functions
    self.start_streaming = self.streaming_core.start_streaming
    self.stop_streaming = self.streaming_core.stop_streaming
    self.add_streaming_question = self.streaming_core.add_audience_question
    self.get_streaming_stats = self.streaming_core.get_streaming_stats
    
    # Register experience and knowledge functions
    self.recall_streaming_experience = self.streaming_core.recall_streaming_experience
    self.get_cross_game_insights = self.streaming_core.get_cross_game_insights
    
    # Register learning and analysis functions
    self.summarize_streaming_learnings = self.streaming_core.learning_manager.generate_learning_summary
    self.analyze_streaming_session = self.streaming_core.learning_manager.analyze_session_learnings
    
    logger.info(f"Streaming system initialized and integrated with brain for user {self.user_id}")
    
    return self.streaming_core

async def process_streaming_event(self, event_type: str, event_data: dict, significance: float = 5.0):
    """
    Process a significant streaming event through the brain's cognitive systems
    
    Args:
        event_type: Type of event (e.g., "commentary", "question_answer")
        event_data: Data about the event
        significance: Importance level (1-10)
        
    Returns:
        Processing results including memory_id and any cognitive processing
    """
    results = {}
    
    # Get game name from streaming system if available
    game_name = "Unknown Game"
    if hasattr(self, "streaming_core") and hasattr(self.streaming_core.streaming_system, "game_state"):
        game_name = self.streaming_core.streaming_system.game_state.game_name or "Unknown Game"
    
    # 1. Store in memory system
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
    if hasattr(self, "emotional_core") and self.emotional_core:
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
    if significance >= 7.0 and hasattr(self, "reasoning_core"):
        try:
            reasoning_result = await self.reasoning_core.analyze_event(
                event_data=event_data,
                context={"domain": "gaming", "event_type": event_type}
            )
            results["reasoning"] = reasoning_result
        except Exception as e:
            logger.error(f"Error in reasoning about streaming event: {e}")
    
    # 4. Process through identity system if available
    if hasattr(self, "identity_evolution") and event_type in ["question_answer", "commentary"]:
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

async def integrate_streaming_knowledge(self, game_name: str):
    """
    Integrate knowledge from streaming into long-term knowledge systems
    
    Args:
        game_name: Name of the game to integrate knowledge for
        
    Returns:
        Integration results
    """
    results = {}
    
    # 1. Create reflection on streaming experience
    if self.streaming_core and self.streaming_core.memory_mapper:
        reflection = await self.streaming_core.memory_mapper.create_streaming_reflection(
            game_name=game_name,
            aspect="knowledge_integration",
            context="knowledge integration"
        )
        results["reflection"] = reflection
    
    # 2. Store cross-game insights as knowledge
    if self.streaming_core and hasattr(self.streaming_core, "cross_game_knowledge"):
        insights = self.streaming_core.cross_game_knowledge.get_applicable_insights(
            target_game=game_name,
            min_relevance=0.7
        )
        
        if insights and hasattr(self, "knowledge_core"):
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
    if hasattr(self, "experience_consolidation"):
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
async def add_procedural_knowledge(ctx, name: str, steps: List[Dict[str, Any]], domain: str = "general") -> Dict[str, Any]:
    """
    Add procedural knowledge to the system
    
    Args:
        name: Name of the procedure
        steps: List of procedure steps
        domain: Knowledge domain
    
    Returns:
        Procedure creation result
    """
    brain = ctx.context
    
    # Add procedure
    result = await brain.add_procedure(name, steps, domain=domain)
    return result

@function_tool
async def run_procedure(ctx, name: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute a stored procedure
    
    Args:
        name: Name of the procedure to run
        context: Optional execution context
    
    Returns:
        Execution result
    """
    brain = ctx.context
    
    # Execute procedure
    result = await brain.execute_procedure(name, context)
    return result

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
async def get_identity_state(ctx) -> Dict[str, Any]:
    """
    Get current state of Nyx's identity
    
    Returns:
        Identity state information
    """
    brain = ctx.context
    
    # Get identity state
    result = await brain.get_identity_state()
    return result

@function_tool
async def adapt_experience_sharing(ctx, user_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt experience sharing based on user feedback
    
    Args:
        user_id: User ID
        feedback: User feedback data
    
    Returns:
        Adaptation results
    """
    brain = ctx.context
    
    # Adapt experience sharing
    result = await brain.adapt_experience_sharing(user_id, feedback)
    return result

@function_tool
async def run_experience_consolidation(ctx) -> Dict[str, Any]:
    """
    Run experience consolidation process
    
    Returns:
        Consolidation results
    """
    brain = ctx.context
    
    # Run consolidation
    result = await brain.run_experience_consolidation()
    return result

# =============== Main Brain Class ===============

class NyxBrain:
    """
    Central integration point for all Nyx systems.
    Handles cross-component communication and provides a unified API using the OpenAI Agents SDK.
    
    Enhanced with improved experience sharing, identity evolution, vector search,
    cross-user experience sharing, and better adaptation integration.
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
        self.distributed_processing = DistributedProcessingManager(max_parallel_tasks=10)
        self.prediction_engine = PredictionEngine() 
    
        # State tracking
        self.initialized = False
        self.last_interaction = datetime.datetime.now()
        self.interaction_count = 0

        self.issue_tracker = IssueTrackingSystem(f"nyx_issues_db_{user_id}.json")
        
        # Bidirectional influence settings
        self.memory_to_emotion_influence = 0.3  # How much memories influence emotions
        self.emotion_to_memory_influence = 0.4  # How much emotions influence memory retrieval
        self.experience_to_identity_influence = 0.2  # How much experiences influence identity
        
        # Performance monitoring
        self.performance_metrics = {
            "memory_operations": 0,
            "emotion_updates": 0,
            "reflections_generated": 0,
            "experiences_shared": 0,
            "cross_user_experiences_shared": 0,
            "experience_consolidations": 0,
            "response_times": []
        }

        self.thinking_config = {
            "thinking_enabled": True,  # Master switch for thinking capability
            "last_thinking_interaction": 0,  # Track when we last used thinking
            "thinking_stats": {
                "total_thinking_used": 0,
                "basic_thinking_used": 0,
                "moderate_thinking_used": 0,
                "deep_thinking_used": 0,
                "thinking_time_avg": 0.0
            }
        }
        
        # Caching
        self.context_cache = {}
        self.identity_cache = {}
        self.adaptation_cache = {}
        
        # Cross-user experience settings
        self.cross_user_enabled = True
        self.cross_user_sharing_threshold = 0.7  # Min relevance for cross-user experiences
        
        # Identity settings
        self.identity_reflection_interval = 10  # Interactions between identity reflections
        
        # Experience consolidation settings
        self.consolidation_interval = 24  # Hours between consolidations
        self.last_consolidation = datetime.datetime.now() - datetime.timedelta(hours=25)  # Start with consolidation due
        
        # Main brain agent - initialized in initialize()
        self.brain_agent = None
        
        # Trace group ID for connecting traces
        self.trace_group_id = f"nyx-brain-{user_id}-{conversation_id}"
        
        # Registry of instance clusters (for cross-conversation access)
        self.instance_registry = {}

        # Initialize streaming system if needed
        if os.environ.get("ENABLE_STREAMING", "false").lower() == "true":
            await self.initialize_streaming()

        self.procedural_memory = ProceduralMemoryManager()
        self.agent_enhanced_memory = AgentEnhancedMemoryManager(self.procedural_memory)
    
        self.initialized = True
        logger.info(f"NyxBrain initialized for user {self.user_id}, conversation {self.conversation_id}")    

        self.cross_user_enabled = True
        self.cross_user_sharing_threshold = 0.7

        self.error_registry = {
            "unhandled_errors": [],
            "handled_errors": [],
            "error_counts": {},
            "error_recovery_strategies": {},
            "error_recovery_stats": {}
        }

    async def initialize_agent_capabilities(self):
        """Initialize the roleplaying agent capabilities directly in the brain"""
        if not hasattr(self, "agent_capabilities_initialized"):
            # Initialize agents
            await initialize_agents()
            
            # Create an agent context for this brain
            self.agent_context = AgentContext(self.user_id, self.conversation_id)
            
            # Initialize response filter
            self.response_filter = ResponseFilter(self.user_id, self.conversation_id)
            
            # Set initialization flag
            self.agent_capabilities_initialized = True
            
            logger.info(f"Agent capabilities initialized for brain {self.user_id}/{self.conversation_id}")
    
    async def register_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register an error from any component for central management."""
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
            "recovery_success": error_record["recovery_success"]
        }
    
    async def _execute_recovery_strategy(self, strategy: Dict[str, Any], error_record: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a recovery strategy for an error."""
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
    
    # Implement recovery strategy methods...
    
    async def register_recovery_strategy(self, error_type: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Register a recovery strategy for an error type."""
        self.error_registry["error_recovery_strategies"][error_type] = strategy
        return {"registered": True}
    
    def _clean_up_error_registry(self) -> None:
        """Clean up old errors from the registry."""
        # Keep only the latest 1000 errors
        if len(self.error_registry["unhandled_errors"]) > 1000:
            self.error_registry["unhandled_errors"] = self.error_registry["unhandled_errors"][-1000:]
        if len(self.error_registry["handled_errors"]) > 1000:
            self.error_registry["handled_errors"] = self.error_registry["handled_errors"][-1000:]

    async def initialize_reflexive_system(brain):
        """Initialize reflexive system integration with NyxBrain"""
        brain.reflexive_system = ReflexiveSystem(brain.agent_enhanced_memory)
        
        # Initialize decision system
        brain.reflexive_system.decision_system = ReflexDecisionSystem()
        
        # Create reflex-specific agent
        brain.reflex_agent = create_reflex_agent(brain.reflexive_system)
        
        # Register reflexive system tools
        # These will be added to the brain_agent
        brain.register_reflex = brain.reflexive_system.register_reflex
        brain.process_stimulus_fast = brain.reflexive_system.process_stimulus_fast
        brain.train_reflexes = brain.reflexive_system.train_reflexes
        brain.add_gaming_reflex = brain.reflexive_system.add_gaming_reflex
        brain.simulate_gaming_scenarios = brain.reflexive_system.simulate_gaming_scenarios
        brain.set_reflex_response_mode = brain.reflexive_system.set_response_mode
        brain.get_reflexive_stats = brain.reflexive_system.get_reflexive_stats
        brain.optimize_reflexes = brain.reflexive_system.optimize_reflexes
        
        # Register new methods for parallelized processing
        brain.process_input_with_reflexes = process_input_with_reflexes.__get__(brain, type(brain))
        brain.analyze_stimulus_for_reflexes = analyze_stimulus_for_reflexes.__get__(brain, type(brain))
        brain.evaluate_reflex_performance = evaluate_reflex_performance.__get__(brain, type(brain))
        
        logger.info("Reflexive system initialized and integrated with brain")
        
        return brain.reflexive_system

    async def process_input_with_reflexes(self, 
                                        user_input: str, 
                                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process input with potential for reflexive response while also enabling deeper processing
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Processing results with reflexive and/or deliberate components
        """
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="process_input_with_reflexes", group_id=self.trace_group_id):
            # Convert user_input to stimulus format
            stimulus = {"text": user_input}
            if context:
                # Extract relevant context features for stimulus
                if "domain" in context:
                    stimulus["domain"] = context["domain"]
                if "urgency" in context:
                    stimulus["urgency"] = context["urgency"]
                if "scenario_type" in context:
                    stimulus["scenario_type"] = context["scenario_type"]
            
            # Determine if reflexes should be used
            domain = context.get("domain") if context else None
            use_reflex, confidence = self.reflexive_system.decision_system.should_use_reflex(
                stimulus, context, domain
            )
            
            # Start timing
            start_time = time.time()
            
            # Start procedural memory lookup in parallel
            procedural_task = asyncio.create_task(
                self.agent_enhanced_memory.find_similar_procedures(user_input)
            )
            
            # Start reflex processing and deliberate thinking in parallel
            reflex_result = None
            thinking_task = None
            
            if use_reflex:
                # Process with reflexes
                reflex_task = asyncio.create_task(
                    self.reflexive_system.process_stimulus_fast(stimulus, domain, context)
                )
                
                # Start deliberate processing in parallel (we'll decide later whether to use it)
                thinking_task = asyncio.create_task(
                    self.process_user_input_with_thinking(user_input, context)
                )
                
                # Wait for the reflex result first (it should be faster)
                try:
                    reflex_result = await asyncio.wait_for(reflex_task, timeout=0.2)  # 200ms timeout
                except asyncio.TimeoutError:
                    # Reflex took too long, we'll fallback to deliberate
                    logger.info("Reflex processing timed out, falling back to deliberate processing")
                    reflex_result = {"success": False, "reason": "timeout"}
            else:
                # Just start deliberate processing
                thinking_task = asyncio.create_task(
                    self.process_user_input_with_thinking(user_input, context)
                )
            
            # Wait for procedural memory lookup to complete
            try:
                relevant_procedures = await asyncio.wait_for(procedural_task, timeout=0.3)  # 300ms timeout
            except asyncio.TimeoutError:
                relevant_procedures = []
                logger.info("Procedural memory lookup timed out")
            
            # Decide which processing path to use
            procedural_knowledge = None
            if relevant_procedures:
                procedural_knowledge = {
                    "relevant_procedures": relevant_procedures,
                    "can_execute": len(relevant_procedures) > 0
                }
            
            # If reflexes succeeded and were fast enough, use that result
            if reflex_result and reflex_result.get("success", False):
                # Check if we've already waited for procedural memory
                if not procedural_knowledge:
                    # See if procedural_task is done
                    if procedural_task.done():
                        relevant_procedures = procedural_task.result()
                        if relevant_procedures:
                            procedural_knowledge = {
                                "relevant_procedures": relevant_procedures,
                                "can_execute": len(relevant_procedures) > 0
                            }
                
                # Just let the deliberate thinking continue in background
                # We don't need to wait for its result
                
                # Build result
                result = {
                    "response_type": "reflexive",
                    "reaction_time_ms": reflex_result.get("reaction_time_ms", 0),
                    "reflex_pattern": reflex_result.get("pattern_name", "unknown"),
                    "confidence": confidence,
                    "result": reflex_result,
                    "procedural_knowledge": procedural_knowledge
                }
                
                # Update decision system
                self.reflexive_system.decision_system.update_from_result(
                    stimulus, True, True
                )
                
                return result
            
            # If reflexes failed or weren't used, wait for deliberate processing to complete
            deliberate_result = await thinking_task
            
            # Build combined result
            result = {
                "response_type": "deliberate",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "reflex_attempted": use_reflex,
                "reflex_confidence": confidence if use_reflex else 0,
                "result": deliberate_result,
                "procedural_knowledge": procedural_knowledge
            }
            
            # Update decision system
            self.reflexive_system.decision_system.update_from_result(
                stimulus, use_reflex, True
            )
            
            return result

    async def validate_action(self, session_id: str, action_type: str, action_data: Dict[str, Any], timestamp: str) -> Dict[str, bool]:
        """Validate critical actions from sessions before execution."""
        # Initialize validation trackers if needed
        if not hasattr(self, "validation_history"):
            self.validation_history = []
        
        # Create validation record
        validation_record = {
            "session_id": session_id,
            "action_type": action_type,
            "action_data": action_data,
            "timestamp": timestamp,
            "decision": None,
            "reasoning": None
        }
        
        # Determine if action is valid
        is_valid = True
        reasoning = ""
        
        # Different validation logic based on action type
        if action_type == "response_generation":
            is_valid, reasoning = await self._validate_response_generation(action_data)
        elif action_type == "user_model_update":
            is_valid, reasoning = await self._validate_user_model_update(action_data)
        elif action_type == "system_configuration":
            is_valid, reasoning = await self._validate_system_configuration(action_data)
        # Handle other action types...
        
        # Update validation record
        validation_record["decision"] = is_valid
        validation_record["reasoning"] = reasoning
        
        # Store in history
        self.validation_history.append(validation_record)
        
        # Trim history if needed
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
        
        return {
            "valid": is_valid,
            "reasoning": reasoning
        }
    
    async def _validate_response_generation(self, action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate response generation actions."""
        # Extract data
        result = action_data.get("result", {})
        user_input = action_data.get("user_input", "")
        context = action_data.get("context", {})
        
        # Check for potentially harmful content
        if "message" in result:
            # Analyze message for problematic content
            content_analysis = await self._analyze_content_safety(result["message"])
            
            if content_analysis["risk_level"] >= 0.8:
                return False, f"Content safety risk: {content_analysis['risk_type']}"
        
        # Check for alignment with user preferences
        if hasattr(self, "user_model") and self.user_model:
            preference_alignment = await self.user_model.check_preference_alignment(result)
            
            if preference_alignment["score"] < 0.5:
                return False, f"Low preference alignment: {preference_alignment['reason']}"
        
        return True, "Action validated"


    # Function to integrate all attention, bottom-up/top-down, and reward systems
    async def process_integrated_input(self, 
                                    user_input: str, 
                                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input with fully integrated cognitive pathways
        
        Args:
            user_input: User's input text
            context: Additional context
            
        Returns:
            Processing results
        """
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="integrated_processing", group_id=self.trace_group_id):
            start_time = datetime.datetime.now()
            
            # Initialize context
            context = context or {}
            
            # 1. Create sensory input
            sensory_input = SensoryInput(
                modality="text",
                data=user_input,
                confidence=1.0,
                timestamp=datetime.datetime.now().isoformat(),
                metadata=context
            )
            
            # 2. Calculate initial saliency and update attention
            salient_items = [{
                "target": "text_input",
                "novelty": 0.8,  # Assume new input is novel
                "intensity": min(1.0, len(user_input) / 500),  # Longer inputs have higher intensity
                "emotional_impact": 0.5,  # Default moderate emotional impact
                "goal_relevance": 0.7  # Assume user input is relevant to goals
            }]
            
            attention_foci = await self.attentional_controller.update_attention(
                salient_items=salient_items
            )
            
            # 3. Process through multimodal integrator with bottom-up and top-down processing
            percept = await self.multimodal_integrator.process_sensory_input(sensory_input)
            
            # 4. Update reasoning with the integrated percept (if sufficient attention)
            if percept.attention_weight > 0.5:
                await self.reasoning_core.update_with_perception(percept)
            
            # 5. Process emotional impact of the input
            emotional_result = await self._process_emotional_impact(user_input, context)
            emotional_state = emotional_result["emotional_state"]
            
            # 6. Compute reward based on emotional response
            reward_value = self.emotional_core.compute_reward_from_emotion()
            
            # 7. Process reward signal
            reward_context = {
                "source": "user_input",
                "content": user_input,
                "action": "process_input",
                "state": {
                    "attention_weight": percept.attention_weight,
                    "modality": "text"
                }
            }
            
            # Add any provided reward context
            if "reward_context" in context:
                reward_context.update(context["reward_context"])
            
            reward_signal = RewardSignal(
                value=reward_value,
                source="emotional_response",
                context=reward_context,
                timestamp=datetime.datetime.now().isoformat()
            )
            
            reward_result = await self.reward_system.process_reward_signal(reward_signal)
            
            # 8. Retrieve memories with attention and emotional context
            memory_context = context.copy()
            memory_context["emotional_state"] = emotional_state
            memory_context["attention_weight"] = percept.attention_weight
            
            memories = await self.memory_orchestrator.retrieve_memories(
                query=user_input,
                memory_types=memory_context.get("memory_types", ["observation", "reflection", "abstraction", "experience"]),
                limit=memory_context.get("memory_limit", 5)
            )
            
            # 9. Check for experience sharing
            should_share_experience = self._should_share_experience(user_input, context)
            experience_result = None
            
            if should_share_experience:
                experience_result = await self.experience_interface.share_experience_enhanced(
                    query=user_input,
                    context_data={
                        "user_id": str(self.user_id),
                        "emotional_state": emotional_state,
                        "include_cross_user": self.cross_user_enabled and context.get("include_cross_user", True),
                        "scenario_type": context.get("scenario_type", ""),
                        "conversation_id": self.conversation_id,
                        "attention_weight": percept.attention_weight
                    }
                )
            
            # 10. Identity impact (if experience found)
            identity_impact = None
            if experience_result and experience_result.get("has_experience", False):
                experience = experience_result.get("experience", {})
                if experience and self.identity_evolution:
                    # Calculate impact on identity
                    identity_impact = await self.identity_evolution.calculate_experience_impact(experience)
                    
                    # Update identity based on experience
                    await self.identity_evolution.update_identity_from_experience(
                        experience=experience,
                        impact=identity_impact
                    )
            
            # 11. Add memory of this interaction
            memory_id = await self.memory_core.add_memory(
                memory_text=f"User said: {user_input}",
                memory_type="observation",
                significance=5,
                tags=["interaction", "user_input"],
                metadata={
                    "emotional_context": self.emotional_core.get_formatted_emotional_state(),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "user_id": str(self.user_id),
                    "attention_weight": percept.attention_weight,
                    "reward_value": reward_value,
                    "dopamine_level": self.reward_system.current_dopamine
                }
            )
            
            # 12. Calculate dopamine change and update hormone influence
            if self.hormone_system:
                try:
                    dopamine_change = reward_result.get("dopamine_change", 0)
                    
                    # Update hormone cycles with dopamine influence
                    hormone_effects = {
                        "reward_signal": dopamine_change * 0.5
                    }
                    
                    await self.hormone_system.update_hormone_cycles(RunContextWrapper(context=None))
                    self.emotional_core.apply_temporal_hormone_effects(hormone_effects)
                except Exception as e:
                    logger.error(f"Error updating hormones from reward: {e}")
            
            # 13. Update performance metrics
            self.performance_metrics["memory_operations"] += 1
            self.performance_metrics["emotion_updates"] += 1
            if experience_result and experience_result.get("has_experience", False):
                self.performance_metrics["experiences_shared"] += 1
                if experience_result.get("cross_user", False):
                    self.performance_metrics["cross_user_experiences_shared"] += 1
            
            # 14. Calculate response time
            end_time = datetime.datetime.now()
            response_time = (end_time - start_time).total_seconds()
            self.performance_metrics["response_times"].append(response_time)
            
            # 15. Update interaction tracking
            self.last_interaction = datetime.datetime.now()
            self.interaction_count += 1
            
            # 16. Return integrated processing results
            result = {
                "user_input": user_input,
                "perceptual_processing": {
                    "modality": percept.modality,
                    "attention_weight": percept.attention_weight,
                    "bottom_up_confidence": percept.bottom_up_confidence,
                    "top_down_influence": percept.top_down_influence
                },
                "emotional_state": emotional_state,
                "reward_processing": {
                    "reward_value": reward_value,
                    "dopamine_level": self.reward_system.current_dopamine,
                    "effects": reward_result.get("effects", {})
                },
                "memories": memories,
                "memory_count": len(memories),
                "has_experience": experience_result["has_experience"] if experience_result else False,
                "experience_response": experience_result["response_text"] if experience_result and experience_result["has_experience"] else None,
                "cross_user_experience": experience_result.get("cross_user", False) if experience_result else False,
                "memory_id": memory_id,
                "response_time": response_time,
                "identity_impact": identity_impact,
                "current_attention": [focus.target for focus in attention_foci]
            }
            
            return result

    async def register_session(self, user_id: int, conversation_id: int, initial_context: Dict[str, Any] = None) -> str:
        """
        Register a new agent session with the central brain.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            initial_context: Optional initial context data
            
        Returns:
            session_id: Unique identifier for the session
        """
        # Generate a unique session ID
        session_id = f"session_{user_id}_{conversation_id}_{int(time.time())}"
        
        # Create session data structure
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "created_at": datetime.datetime.now().isoformat(),
            "last_active": datetime.datetime.now().isoformat(),
            "status": "active",  # Options: active, idle, paused, archived
            "current_context": initial_context or {},
            "emotional_state": {},
            "user_preferences": {},
            "session_tags": [],
            "is_rehydrated": False,
            "last_checkpoint": None,
            "pending_feedback": False
        }
        
        # Store session in brain's session registry
        if not hasattr(self, "sessions"):
            self.sessions = {}
        
        self.sessions[session_id] = session_data
        
        # Initialize cache for this session's memory
        await self.memory_core.initialize_session_cache(session_id)
        
        # Log session creation
        logger.info(f"New session registered: {session_id} for user {user_id}, conversation {conversation_id}")
        
        return session_id
    
    async def update_session(self, session_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing session with new data.
        
        Args:
            session_id: Session ID to update
            update_data: New data to update session with
            
        Returns:
            Updated session data
        """
        if not hasattr(self, "sessions") or session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        # Get session
        session = self.sessions[session_id]
        
        # Update provided fields
        for key, value in update_data.items():
            if key in session:
                session[key] = value
        
        # Always update last_active timestamp
        session["last_active"] = datetime.datetime.now().isoformat()
        
        return session
    
    async def archive_session(self, session_id: str) -> Dict[str, Any]:
        """
        Archive a session when it's no longer active.
        
        Args:
            session_id: Session ID to archive
            
        Returns:
            Archived session data
        """
        if not hasattr(self, "sessions") or session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        # Get session
        session = self.sessions[session_id]
        
        # Update status
        session["status"] = "archived"
        
        # Store in persistent storage for later retrieval
        if not hasattr(self, "archived_sessions"):
            self.archived_sessions = {}
        
        self.archived_sessions[session_id] = session.copy()
        
        # Clean up memory resources
        await self.memory_core.clear_session_cache(session_id)
        
        # Generate summary reflection of the session
        if self.reflection_engine:
            reflection = await self.reflection_engine.generate_reflection(
                topic=f"Session {session_id} summary",
                context={
                    "user_id": session["user_id"],
                    "conversation_id": session["conversation_id"],
                    "session_data": session
                }
            )
            
            # Store reflection in memory
            await self.memory_core.add_memory(
                memory_text=reflection,
                memory_type="reflection",
                memory_scope="global",
                significance=7,
                tags=["session_summary", f"session_{session_id}"],
                metadata={
                    "session_id": session_id,
                    "user_id": session["user_id"],
                    "archived_at": datetime.datetime.now().isoformat()
                }
            )
        
        return session

    async def record_significant_event(self, session_id: str, event_type: str, event_data: Dict[str, Any], 
                                    timestamp: str, source: str) -> Dict[str, Any]:
        """Process and record significant events from sessions."""
        
        # Initialize event tracking if needed
        if not hasattr(self, "significant_events"):
            self.significant_events = []
        
        # Create event record
        event_record = {
            "session_id": session_id,
            "event_type": event_type,
            "event_data": event_data,
            "timestamp": timestamp,
            "source": source,
            "processed": False
        }
        
        # Add to events list
        self.significant_events.append(event_record)
        
        # Process event based on type
        if event_type == "emotional_spike":
            await self._process_emotional_spike_event(event_record)
        elif event_type == "context_change":
            await self._process_context_change_event(event_record)
        elif event_type == "user_revelation":
            await self._process_user_revelation_event(event_record)
        # Handle other event types...
        
        # Mark as processed
        event_record["processed"] = True
        
        # Check if we should trigger meta-cognitive cycle
        if len(self.significant_events) % 10 == 0:  # Every 10 events
            asyncio.create_task(self.meta_core.cognitive_cycle({"triggered_by": "significant_events"}))
        
        return {"status": "recorded", "event_id": len(self.significant_events) - 1}
    
    # Add event processing methods
    async def _process_emotional_spike_event(self, event_record: Dict[str, Any]) -> None:
        """Process emotional spike events."""
        # Extract data
        emotional_state = event_record["event_data"]["emotional_state"]
        user_input = event_record["event_data"]["user_input"]
        
        # Store in memory
        await self.memory_core.add_memory(
            memory_text=f"Observed emotional spike during interaction. User said: '{user_input}'",
            memory_type="observation",
            memory_scope="user",
            significance=7,
            tags=["emotional_spike"],
            metadata={
                "emotional_state": emotional_state,
                "timestamp": event_record["timestamp"],
                "session_id": event_record["session_id"]
            }
        )
        
        # Update emotional statistics
        self.emotional_core.update_emotional_statistics(emotional_state)

    # Add to nyx/core/nyx_brain.py
    async def get_communication_metrics(self) -> Dict[str, Any]:
        """Collect and aggregate communication metrics from all sessions."""
        metrics = {
            "total_requests": 0,
            "success_rate": 0.0,
            "avg_response_time": 0.0,
            "command_distribution": {},
            "failure_distribution": {}
        }
        
        if not hasattr(self, "sessions"):
            return metrics
        
        # Aggregate metrics across all sessions
        for session_id, session_data in self.sessions.items():
            if session_id in self.session_factory.sessions:
                session = self.session_factory.sessions[session_id]
                if hasattr(session, "brain_metrics"):
                    # Aggregate total requests
                    metrics["total_requests"] += session.brain_metrics["requests_sent"]
                    
                    # Add to command distribution
                    for cmd, count in session.brain_metrics["command_types"].items():
                        if cmd not in metrics["command_distribution"]:
                            metrics["command_distribution"][cmd] = 0
                        metrics["command_distribution"][cmd] += count
                    
                    # Add to failure distribution if errors
                    if session.brain_metrics["last_communication_error"]:
                        error = session.brain_metrics["last_communication_error"]
                        if error["method"] not in metrics["failure_distribution"]:
                            metrics["failure_distribution"][error["method"]] = 0
                        metrics["failure_distribution"][error["method"]] += 1
        
        # Calculate success rate
        total_succeeded = sum(session.brain_metrics["requests_succeeded"] 
                            for session in self.session_factory.sessions.values() 
                            if hasattr(session, "brain_metrics"))
                            
        total_sent = sum(session.brain_metrics["requests_sent"] 
                       for session in self.session_factory.sessions.values() 
                       if hasattr(session, "brain_metrics"))
                       
        if total_sent > 0:
            metrics["success_rate"] = total_succeeded / total_sent
        
        # Calculate average response time
        all_times = []
        for session in self.session_factory.sessions.values():
            if hasattr(session, "brain_metrics"):
                all_times.extend(session.brain_metrics["response_times"])
        
        if all_times:
            metrics["avg_response_time"] = sum(all_times) / len(all_times)
        
        return metrics
    
    async def resume_session(self, session_id: str) -> Dict[str, Any]:
        """
        Resume an archived or paused session.
        
        Args:
            session_id: Session ID to resume
            
        Returns:
            Reactivated session data
        """
        # Check active sessions first
        if hasattr(self, "sessions") and session_id in self.sessions:
            session = self.sessions[session_id]
            if session["status"] in ["paused", "idle"]:
                # Just reactivate
                session["status"] = "active"
                session["last_active"] = datetime.datetime.now().isoformat()
                return session
        
        # Check archived sessions
        if hasattr(self, "archived_sessions") and session_id in self.archived_sessions:
            # Get archived session
            archived_session = self.archived_sessions[session_id]
            
            # Restore to active sessions
            if not hasattr(self, "sessions"):
                self.sessions = {}
            
            self.sessions[session_id] = archived_session
            
            # Update session data
            self.sessions[session_id]["status"] = "active"
            self.sessions[session_id]["last_active"] = datetime.datetime.now().isoformat()
            self.sessions[session_id]["is_rehydrated"] = True
            
            # Rehydrate memory cache
            await self.memory_core.rehydrate_session_cache(session_id)
            
            # Remove from archived sessions
            del self.archived_sessions[session_id]
            
            return self.sessions[session_id]
        
        raise ValueError(f"Session {session_id} not found in active or archived sessions")
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all active and archived sessions.
        
        Returns:
            Session statistics
        """
        active_count = len(self.sessions) if hasattr(self, "sessions") else 0
        archived_count = len(self.archived_sessions) if hasattr(self, "archived_sessions") else 0
        
        # Collect user counts
        users = set()
        conversations = set()
        
        if hasattr(self, "sessions"):
            for session in self.sessions.values():
                users.add(session["user_id"])
                conversations.add(f"{session['user_id']}_{session['conversation_id']}")
        
        stats = {
            "active_sessions": active_count,
            "archived_sessions": archived_count,
            "unique_users": len(users),
            "unique_conversations": len(conversations),
            "sessions_by_status": {}
        }
        
        # Count sessions by status
        if hasattr(self, "sessions"):
            status_counts = {}
            for session in self.sessions.values():
                status = session["status"]
                if status not in status_counts:
                    status_counts[status] = 0
                status_counts[status] += 1
            
            stats["sessions_by_status"] = status_counts
        
        return stats
    
    async def generate_integrated_response(self, 
                                         user_input: str, 
                                         context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a response with fully integrated cognitive pathways
        
        Args:
            user_input: User's input text
            context: Additional context
            
        Returns:
            Response data
        """
        # Process input first
        processing_result = await self.process_integrated_input(user_input, context)
        
        # Determine main response content
        if processing_result["has_experience"]:
            main_response = processing_result["experience_response"]
            response_type = "experience"
            
            # If it's a cross-user experience, mark it
            if processing_result.get("cross_user_experience", False):
                response_type = "cross_user_experience"
        else:
            # For reasoning-related queries, use the reasoning agents
            if self._is_reasoning_query(user_input):
                try:
                    reasoning_result = await Runner.run(
                        reasoning_triage_agent,
                        user_input
                    )
                    main_response = reasoning_result.final_output
                    response_type = "reasoning"
                except Exception as e:
                    logger.error(f"Error in reasoning response: {str(e)}")
                    # Fallback to standard response
                    main_response = "I understand your question and would like to reason through it with you."
                    response_type = "standard"
            else:
                # No specific experience to share, generate standard response
                # In a real implementation, this would be more sophisticated
                main_response = "I acknowledge your message and have processed it through my systems."
                response_type = "standard"
        
        # Generate emotional expression
        emotional_expression = None
        
        # Apply reward-based emotional expression if dopamine level is high
        dopamine_level = self.reward_system.current_dopamine
        
        if dopamine_level > 0.7:
            # High dopamine leads to more expressive response
            try:
                reflection = await self.emotional_core.create_reward_based_reflection(
                    reward_value=processing_result["reward_processing"]["reward_value"],
                    context={"user_input": user_input}
                )
                emotional_expression = reflection
            except Exception as e:
                logger.error(f"Error generating reward-based expression: {e}")
        else:
            # Standard emotional expression
            should_express_emotion = self.emotional_core.should_express_emotion()
            
            if should_express_emotion:
                try:
                    expression_result = await self.emotional_core.generate_emotional_expression(force=False)
                    if expression_result.get("expressed", False):
                        emotional_expression = expression_result.get("expression", "")
                except Exception as e:
                    logger.error(f"Error generating emotional expression: {str(e)}")
                    emotional_expression = self.emotional_core.get_expression_for_emotion()
        
        # Create response package
        response_data = {
            "message": main_response,
            "response_type": response_type,
            "emotional_state": processing_result["emotional_state"],
            "emotional_expression": emotional_expression,
            "memories_used": [m["id"] for m in processing_result["memories"]],
            "memory_count": processing_result["memory_count"],
            "reward_processing": processing_result["reward_processing"],
            "perceptual_processing": processing_result["perceptual_processing"],
            "identity_impact": processing_result.get("identity_impact")
        }
        
        # Add memory of this response
        memory_id = await self.memory_core.add_memory(
            memory_text=f"I responded: {main_response}",
            memory_type="observation",
            significance=5,
            tags=["interaction", "nyx_response", response_type],
            metadata={
                "emotional_context": self.emotional_core.get_formatted_emotional_state(),
                "timestamp": datetime.datetime.now().isoformat(),
                "response_type": response_type,
                "user_id": str(self.user_id),
                "reward_value": processing_result["reward_processing"]["reward_value"],
                "dopamine_level": processing_result["reward_processing"]["dopamine_level"]
            }
        )
        
        response_data["response_memory_id"] = memory_id
        
        # Generate reward signal for the response (self-evaluation)
        # This helps with reinforcement learning for response generation
        try:
            response_context = {
                "action": "generate_response",
                "response_type": response_type,
                "state": {
                    "user_input": user_input,
                    "emotional_state": processing_result["emotional_state"],
                }
            }
            
            # Basic self-evaluation based on confidence
            # In a more sophisticated implementation, this would use actual evaluation metrics
            confidence = 0.6  # Default moderate confidence
            
            if response_type == "experience":
                confidence = 0.8  # Higher confidence for experience-based responses
            elif response_type == "reasoning":
                confidence = 0.7  # Good confidence for reasoning responses
            
            # Create reward signal
            response_reward = RewardSignal(
                value=confidence * 0.4,  # Scale to moderate reward
                source="response_generation",
                context=response_context,
                timestamp=datetime.datetime.now().isoformat()
            )
            
            # Process reward signal (in background to not delay response)
            asyncio.create_task(self.reward_system.process_reward_signal(response_reward))
            
        except Exception as e:
            logger.error(f"Error generating response reward: {e}")
        
        return response_data
    
    @function_tool
    async def analyze_stimulus_for_reflexes(ctx, stimulus: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a stimulus to determine if it should trigger reflexes
        
        Args:
            stimulus: Stimulus data to analyze
        
        Returns:
            Analysis results including reflex potential
        """
        brain = ctx.context
        
        # Check if reflexes should be used
        use_reflex, confidence = brain.reflexive_system.decision_system.should_use_reflex(
            stimulus, None, None
        )
        
        # Find potentially matching patterns
        matching_patterns = []
        for name, pattern in brain.reflexive_system.reflex_patterns.items():
            match_score = brain.reflexive_system.pattern_recognition.fast_match(
                stimulus, pattern.pattern_data
            )
            if match_score >= pattern.threshold * 0.8:  # Include near-matches
                matching_patterns.append({
                    "name": name,
                    "match_score": match_score,
                    "procedure": pattern.procedure_name
                })
        
        # Sort by match score
        matching_patterns.sort(key=lambda p: p["match_score"], reverse=True)
        
        return {
            "should_use_reflex": use_reflex,
            "confidence": confidence,
            "potential_patterns": matching_patterns[:3],  # Top 3 matches
            "stimulus_complexity": len(stimulus),
            "reflex_count": len(brain.reflexive_system.reflex_patterns)
        }
    
    @function_tool
    async def evaluate_reflex_performance(ctx, scenario_type: str) -> Dict[str, Any]:
        """
        Evaluate reflex performance for a specific scenario type
        
        Args:
            scenario_type: Type of scenario to evaluate
        
        Returns:
            Performance evaluation results
        """
        brain = ctx.context
        
        # Collect relevant patterns
        relevant_patterns = {}
        for name, pattern in brain.reflexive_system.reflex_patterns.items():
            if pattern.context_template and "scenario_type" in pattern.context_template:
                if pattern.context_template["scenario_type"] == scenario_type:
                    relevant_patterns[name] = pattern
        
        # Calculate performance metrics
        if not relevant_patterns:
            return {
                "success": False,
                "error": f"No patterns found for scenario: {scenario_type}"
            }
        
        # Calculate statistics
        stats = {
            "pattern_count": len(relevant_patterns),
            "avg_success_rate": sum(p.get_success_rate() for p in relevant_patterns.values()) / len(relevant_patterns),
            "avg_response_time_ms": sum(p.get_avg_response_time() for p in relevant_patterns.values()) / len(relevant_patterns),
            "total_executions": sum(p.execution_count for p in relevant_patterns.values()),
            "top_patterns": sorted(
                [(name, p.get_success_rate()) for name, p in relevant_patterns.items()],
                key=lambda x: x[1],
                reverse=True
            )[:3]
        }
        
        return {
            "success": True,
            "scenario_type": scenario_type,
            "stats": stats
        }
    
    async def generate_response_with_reflexes(self, 
                                           user_input: str, 
                                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a response using reflexes, procedural memory, and/or deliberate thinking
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Complete response with appropriate processing path
        """
        with trace(workflow_name="generate_response_with_reflexes", group_id=self.trace_group_id):
            # Process the input with all cognitive systems in parallel
            processing_result = await self.process_input_with_reflexes(user_input, context)
            
            # If reflexes were used, we may already have a response
            if processing_result["response_type"] == "reflexive":
                reflex_result = processing_result["result"]
                
                # Check if the reflex result includes a direct response
                if "response" in reflex_result:
                    message = reflex_result["response"]
                else:
                    # Use a generic response indicating a reflexive action
                    message = f"I've reacted to your input reflexively ({reflex_result.get('pattern_name', 'unknown pattern')})."
                
                # Package the response
                response_data = {
                    "message": message,
                    "response_type": "reflexive",
                    "reflex_pattern": reflex_result.get("pattern_name"),
                    "reaction_time_ms": reflex_result.get("reaction_time_ms", 0),
                    "emotional_state": self.emotional_core.get_emotional_state() if hasattr(self, "emotional_core") else {},
                    "procedural_knowledge": processing_result.get("procedural_knowledge")
                }
                
                return response_data
            
            # Check if procedural knowledge was found and can be executed
            elif processing_result.get("procedural_knowledge") and processing_result["procedural_knowledge"].get("can_execute"):
                # Get the most relevant procedure
                procedures = processing_result["procedural_knowledge"]["relevant_procedures"]
                if procedures:
                    top_procedure = procedures[0]
                    
                    # Execute the procedure
                    try:
                        procedure_result = await self.agent_enhanced_memory.execute_procedure(
                            top_procedure["name"],
                            context={"user_input": user_input, **(context or {})}
                        )
                        
                        # If successful, use the procedure's response
                        if procedure_result.get("success", False) and "output" in procedure_result:
                            message = procedure_result["output"]
                            response_type = "procedural"
                        else:
                            # Fall back to deliberate response
                            deliberate_result = processing_result["result"]
                            message = deliberate_result.get("response", "I've processed your input but couldn't find a suitable response.")
                            response_type = "deliberate"
                    except Exception as e:
                        logger.error(f"Error executing procedure: {str(e)}")
                        # Fall back to deliberate response
                        deliberate_result = processing_result["result"]
                        message = deliberate_result.get("response", "I've processed your input but couldn't find a suitable response.")
                        response_type = "deliberate"
                else:
                    # Fall back to deliberate response
                    deliberate_result = processing_result["result"]
                    message = deliberate_result.get("response", "I've processed your input but couldn't find a suitable response.")
                    response_type = "deliberate"
                    
                # Package the response
                response_data = {
                    "message": message,
                    "response_type": response_type,
                    "processing_time_ms": processing_result["processing_time_ms"],
                    "emotional_state": self.emotional_core.get_emotional_state() if hasattr(self, "emotional_core") else {},
                    "procedural_knowledge": processing_result.get("procedural_knowledge")
                }
                
                return response_data
            
            else:
                # Use standard response generation
                response = await self.generate_response(user_input, context)
                
                # Add information about procedural knowledge
                if processing_result.get("procedural_knowledge"):
                    response["procedural_knowledge"] = processing_result["procedural_knowledge"]
                
                return response

    # Add these methods to nyx_brain.py to handle learning integration
    
    async def report_learning(self, session_id: str, learning: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a learning payload from a session.
        
        Args:
            session_id: Source session ID
            learning: Learning data payload
            
        Returns:
            Processing result
        """
        # Validate session
        if not hasattr(self, "sessions") or session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        # Enrich learning with session metadata
        enriched_learning = {
            "session_id": session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "user_id": session["user_id"],
            "conversation_id": session["conversation_id"],
            **learning
        }
        
        # Initialize learning history if needed
        if not hasattr(self, "learning_history"):
            self.learning_history = []
        
        # Store in learning history
        self.learning_history.append(enriched_learning)
        
        # Process based on trigger type
        trigger = learning.get("trigger", "time_checkpoint")
        
        try:
            if trigger == "emotional_spike":
                await self._process_emotional_learning(enriched_learning)
            elif trigger == "procedural_execution":
                await self._process_procedural_learning(enriched_learning)
            elif trigger == "user_reveal":
                await self._process_user_revelation(enriched_learning)
            elif trigger == "time_checkpoint":
                await self._process_checkpoint_learning(enriched_learning)
            elif trigger == "manual_trigger":
                await self._process_manual_learning(enriched_learning)
            elif trigger == "nyx2_triggered":
                await self._process_nyx2_triggered_learning(enriched_learning)
            else:
                # Default processing
                await self._process_general_learning(enriched_learning)
                
            logger.info(f"Processed learning from session {session_id}, trigger: {trigger}")
            
        except Exception as e:
            logger.error(f"Error processing learning: {e}")
            return {"status": "error", "error": str(e)}
        
        # Update session's last checkpoint
        self.sessions[session_id]["last_checkpoint"] = trigger
        
        return {"status": "processed", "learning_id": len(self.learning_history) - 1}
    
    async def _process_emotional_learning(self, learning: Dict[str, Any]) -> None:
        """Process an emotional spike learning event."""
        # Extract emotional data
        emotional_snapshot = learning.get("emotional_snapshot", {})
        user_id = learning.get("user_id")
        content = learning.get("content", "")
        
        # Process through emotional core
        if self.emotional_core:
            await self.emotional_core.process_emotional_pattern(emotional_snapshot)
            
        # Store significant emotional events in memory
        await self.memory_core.add_memory(
            memory_text=content,
            memory_type="observation",
            memory_scope="user",
            significance=7,
            tags=["emotional_event", f"user_{user_id}"],
            metadata={
                "emotional_data": emotional_snapshot,
                "timestamp": learning.get("timestamp"),
                "session_id": learning.get("session_id")
            }
        )
    
    async def _process_procedural_learning(self, learning: Dict[str, Any]) -> None:
        """Process a procedural execution learning event."""
        # Extract procedural data
        procedural_ref = learning.get("procedural_reference", "")
        content = learning.get("content", "")
        
        # Update procedural memory
        if hasattr(self, "procedural_memory"):
            try:
                # Track execution statistics
                await self.procedural_memory.update_execution_stats(
                    procedural_ref,
                    success=learning.get("success", True),
                    execution_time=learning.get("execution_time", 0),
                    context=learning.get("context", {})
                )
                
                # If this is a new or improved procedure, store it
                if "procedure_definition" in learning:
                    await self.procedural_memory.store_procedure(
                        procedural_ref,
                        learning["procedure_definition"],
                        source="session_learning"
                    )
            except Exception as e:
                logger.warning(f"Error updating procedural memory: {e}")
    
    async def _process_user_revelation(self, learning: Dict[str, Any]) -> None:
        """Process user revelation learning."""
        # Extract revelation data
        revelations = learning.get("revelations", [])
        user_id = learning.get("user_id")
        
        if not revelations:
            return
        
        # Process each revelation
        for revelation in revelations:
            # Add to user model if available
            if hasattr(self, "user_model"):
                await self.user_model.add_user_revelation(user_id, revelation)
            
            # Store significant revelations in memory
            revelation_type = revelation.get("type", "unknown")
            revelation_content = revelation.get("content", str(revelation))
            
            await self.memory_core.add_memory(
                memory_text=f"User revealed: {revelation_content}",
                memory_type="observation",
                memory_scope="user",
                significance=8,  # User revelations are highly significant
                tags=["user_revelation", revelation_type, f"user_{user_id}"],
                metadata={
                    "revelation": revelation,
                    "timestamp": learning.get("timestamp"),
                    "session_id": learning.get("session_id")
                }
            )
    
    async def _process_checkpoint_learning(self, learning: Dict[str, Any]) -> None:
        """Process regular checkpoint learning."""
        # Most checkpoints don't need special processing
        # But we could periodically generate reflections
        
        # Randomly generate reflections (about 10% of checkpoints)
        if random.random() < 0.1 and self.reflection_engine:
            session_id = learning.get("session_id")
            user_id = learning.get("user_id")
            
            # Generate reflection based on recent memories
            memories = await self.memory_core.retrieve_memories(
                query="",  # Get recent memories
                memory_scope="user",
                limit=10,
                metadata_filter={"user_id": user_id}
            )
            
            if memories:
                reflection = await self.reflection_engine.generate_reflection(
                    topic="recent_interactions",
                    context={"memories": memories}
                )
                
                # Store reflection
                await self.memory_core.add_memory(
                    memory_text=reflection,
                    memory_type="reflection",
                    memory_scope="user",
                    significance=6,
                    tags=["checkpoint_reflection", f"user_{user_id}"],
                    metadata={
                        "timestamp": learning.get("timestamp"),
                        "session_id": learning.get("session_id")
                    }
                )
    
    async def _process_manual_learning(self, learning: Dict[str, Any]) -> None:
        """Process manually triggered learning."""
        # This handles explicit learning requests from agents
        content = learning.get("content", "")
        source = learning.get("source", "agent")
        
        # Process based on content type
        if "reflection" in learning.get("tags", []):
            # Store as reflection
            await self.memory_core.add_memory(
                memory_text=content,
                memory_type="reflection",
                memory_scope="global",
                significance=7,
                tags=learning.get("tags", []),
                metadata={
                    "source": source,
                    "timestamp": learning.get("timestamp"),
                    "session_id": learning.get("session_id")
                }
            )
        elif "abstraction" in learning.get("tags", []):
            # Store as abstraction
            await self.memory_core.add_memory(
                memory_text=content,
                memory_type="abstraction",
                memory_scope="global",
                significance=8,
                tags=learning.get("tags", []),
                metadata={
                    "source": source,
                    "timestamp": learning.get("timestamp"),
                    "session_id": learning.get("session_id")
                }
            )
        else:
            # Default to observation
            await self.memory_core.add_memory(
                memory_text=content,
                memory_type="observation",
                memory_scope="global",
                significance=5,
                tags=learning.get("tags", ["learning_event"]),
                metadata={
                    "source": source,
                    "timestamp": learning.get("timestamp"),
                    "session_id": learning.get("session_id")
                }
            )
    
    async def _process_nyx2_triggered_learning(self, learning: Dict[str, Any]) -> None:
        """Process learning triggered by Nyx2 (central brain)."""
        # This is for processing requests from the brain itself
        content = learning.get("content", "")
        directives = learning.get("directives", [])
        
        # Apply any directives to the session
        session_id = learning.get("session_id")
        if session_id in self.sessions and directives:
            for directive in directives:
                directive_type = directive.get("type")
                action = directive.get("action")
                
                if directive_type == "tone_adjustment":
                    # Update session context with tone directive
                    self.sessions[session_id]["current_context"]["tone_directive"] = action
                
                elif directive_type == "memory_reinforcement":
                    # Add explicit memory
                    await self.memory_core.add_memory(
                        memory_text=directive.get("content", ""),
                        memory_type="observation",
                        memory_scope="user",
                        significance=7,
                        tags=["nyx2_reinforcement"],
                        metadata={
                            "source": "nyx2_directive",
                            "timestamp": learning.get("timestamp"),
                            "session_id": session_id
                        }
                    )
    
    async def _process_general_learning(self, learning: Dict[str, Any]) -> None:
        """Process general learning that doesn't match other categories."""
        # Default processing for any learning
        content = learning.get("content", "")
        
        # Store in memory with appropriate tags
        await self.memory_core.add_memory(
            memory_text=content,
            memory_type="observation",
            memory_scope="global",
            significance=4,  # Lower significance for general learning
            tags=["learning_event"],
            metadata={
                "source": learning.get("source", "unknown"),
                "timestamp": learning.get("timestamp"),
                "session_id": learning.get("session_id"),
                "trigger": learning.get("trigger", "general")
            }
        )

    async def generate_response_parallel(self, 
                                     user_input: str, 
                                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a complete response to user input using parallel processing
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Response data including main message and supporting information
        """
        with trace(workflow_name="generate_response_parallel", group_id=self.trace_group_id):
            # Process the input using parallel processing
            processing_result = await self.process_input_parallel(user_input, context)
            
            # Track if experience sharing was adapted
            experience_sharing_adapted = processing_result.get("context_change") is not None and \
                                       processing_result.get("adaptation_result") is not None
            
            # Start response generation tasks in parallel
            tasks = {}
            
            # Task 1: Determine main response content
            tasks["main_response"] = asyncio.create_task(
                self._determine_main_response(user_input, processing_result, context)
            )
            
            # Task 2: Generate emotional expression
            tasks["emotional_expression"] = asyncio.create_task(
                self._generate_emotional_expression(processing_result["emotional_state"])
            )
            
            # Task 3: Start memory creation for the response
            memory_tasks = []
            
            # Wait for main response content
            main_response_result = await tasks["main_response"]
            main_response = main_response_result["message"]
            response_type = main_response_result["response_type"]
            
            # Wait for emotional expression
            try:
                emotional_expression_result = await tasks["emotional_expression"]
                emotional_expression = emotional_expression_result["expression"]
            except Exception as e:
                logger.error(f"Error generating emotional expression: {str(e)}")
                emotional_expression = None
            
            # Add memory of this response
            memory_text = f"I responded: {main_response}"
            
            memory_id = await self.memory_core.add_memory(
                memory_text=memory_text,
                memory_type="observation",
                significance=5,
                tags=["interaction", "nyx_response", response_type],
                metadata={
                    "emotional_context": self.emotional_core.get_formatted_emotional_state(),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "response_type": response_type,
                    "user_id": str(self.user_id)
                }
            )
            
            # Start evaluation in parallel if internal feedback system is available
            evaluation = None
            if self.internal_feedback:
                tasks["evaluation"] = asyncio.create_task(
                    self.internal_feedback.critic_evaluate(
                        aspect="effectiveness",
                        content={"text": main_response, "type": response_type},
                        context={"user_input": user_input}
                    )
                )
                
                try:
                    evaluation = await tasks["evaluation"]
                except Exception as e:
                    logger.error(f"Error evaluating response: {str(e)}")
            
            # Check if it's time for experience consolidation in parallel
            if datetime.datetime.now() - self.last_consolidation >= datetime.timedelta(hours=self.consolidation_interval):
                # Create background task that won't block response
                asyncio.create_task(self.run_experience_consolidation())
                self.last_consolidation = datetime.datetime.now()
            
            # Check if it's time for identity reflection in parallel
            identity_reflection = None
            if self.interaction_count % self.identity_reflection_interval == 0:
                try:
                    # Create task for identity reflection
                    tasks["identity"] = asyncio.create_task(
                        self.get_identity_state()
                    )
                    
                    identity_reflection = await tasks["identity"]
                except Exception as e:
                    logger.error(f"Error generating identity reflection: {str(e)}")
            
            # Package the response
            response_data = {
                "message": main_response,
                "response_type": response_type,
                "emotional_state": processing_result["emotional_state"],
                "emotional_expression": emotional_expression,
                "memories_used": [m["id"] for m in processing_result["memories"]],
                "memory_count": processing_result["memory_count"],
                "evaluation": evaluation,
                "experience_sharing_adapted": experience_sharing_adapted,
                "identity_impact": processing_result.get("identity_impact"),
                "identity_reflection": identity_reflection,
                "parallel_processing": True  # Flag to indicate parallel processing was used
            }
            
            return response_data

    async def process_input_parallel(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input with parallel operations for improved performance
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Processing results with relevant memories, emotional state, etc.
        """
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="process_input_parallel", group_id=self.trace_group_id):
            start_time = datetime.datetime.now()
            
            # Initialize context
            context = context or {}
            
            # Add user_id to context if not present
            if "user_id" not in context:
                context["user_id"] = str(self.user_id)
            
            # Create tasks for parallel processing
            tasks = {}
            
            # Task 1: Process emotional impact
            tasks["emotional"] = asyncio.create_task(
                self._process_emotional_impact(user_input, context)
            )
            
            # Task 2: Run meta-cognitive cycle
            if self.meta_core:
                meta_context = context.copy()
                meta_context["user_input"] = user_input
                tasks["meta"] = asyncio.create_task(
                    self.meta_core.cognitive_cycle(meta_context)
                )
            
            # Task 3: Check for experience sharing opportunity
            tasks["experience_check"] = asyncio.create_task(
                self._check_experience_sharing(user_input, context)
            )
            
            # Wait for emotional processing to complete
            try:
                emotional_result = await tasks["emotional"]
                emotional_state = emotional_result["emotional_state"]
                
                # Add emotional state to context for memory retrieval
                context["emotional_state"] = emotional_state
                
                # Now start memory retrieval with emotional context
                tasks["memory"] = asyncio.create_task(
                    self._retrieve_memories_with_emotion(user_input, context, emotional_state)
                )
            except Exception as e:
                logger.error(f"Error in emotional processing: {str(e)}")
                emotional_state = {}
                
                # Start memory retrieval without emotional context
                tasks["memory"] = asyncio.create_task(
                    self._retrieve_memories_with_emotion(user_input, context, {})
                )
            
            # Wait for experience sharing check to complete
            try:
                should_share_experience = await tasks["experience_check"]
            except Exception as e:
                logger.error(f"Error checking experience sharing: {str(e)}")
                should_share_experience = False
            
            # Start experience sharing task if needed
            experience_result = None
            if should_share_experience:
                tasks["experience"] = asyncio.create_task(
                    self._share_experience(user_input, context, emotional_state)
                )
            
            # Wait for memory retrieval to complete
            try:
                memories = await tasks["memory"]
            except Exception as e:
                logger.error(f"Error retrieving memories: {str(e)}")
                memories = []
            
            # Update emotional state based on retrieved memories
            if memories:
                try:
                    memory_emotional_impact = await self._calculate_memory_emotional_impact(memories)
                    
                    # Apply memory-to-emotion influence
                    for emotion, value in memory_emotional_impact.items():
                        self.emotional_core.update_emotion(emotion, value * self.memory_to_emotion_influence)
                    
                    # Get updated emotional state
                    emotional_state = self.emotional_core.get_emotional_state()
                except Exception as e:
                    logger.error(f"Error updating emotion from memories: {str(e)}")
            
            # Wait for experience sharing to complete if started
            identity_impact = None
            if should_share_experience and "experience" in tasks:
                try:
                    experience_result = await tasks["experience"]
                    
                    # Calculate potential identity impact if experience found
                    if experience_result.get("has_experience", False) and self.identity_evolution:
                        experience = experience_result.get("experience", {})
                        if experience:
                            try:
                                # Calculate impact on identity
                                identity_impact = await self.identity_evolution.calculate_experience_impact(experience)
                                
                                # Update identity based on experience
                                await self.identity_evolution.update_identity_from_experience(
                                    experience=experience,
                                    impact=identity_impact
                                )
                            except Exception as e:
                                logger.error(f"Error updating identity from experience: {str(e)}")
                except Exception as e:
                    logger.error(f"Error sharing experience: {str(e)}")
                    experience_result = {"has_experience": False}
            
            # Add memory of this interaction
            memory_text = f"User said: {user_input}"
            
            memory_id = await self.memory_core.add_memory(
                memory_text=memory_text,
                memory_type="observation",
                significance=5,
                tags=["interaction", "user_input"],
                metadata={
                    "emotional_context": self.emotional_core.get_formatted_emotional_state(),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "user_id": str(self.user_id)
                }
            )
            
            # Check for context change using dynamic adaptation
            context_change_result = None
            adaptation_result = None
            
            if self.dynamic_adaptation:
                # Process adaptation in parallel
                tasks["adaptation"] = asyncio.create_task(
                    self._process_adaptation(user_input, context, emotional_state, 
                                          experience_result, identity_impact)
                )
                
                try:
                    adaptation_results = await tasks["adaptation"]
                    context_change_result = adaptation_results.get("context_change")
                    adaptation_result = adaptation_results.get("adaptation_result")
                except Exception as e:
                    logger.error(f"Error in adaptation: {str(e)}")
            
            # Wait for meta cognitive cycle to complete
            meta_result = {}
            if self.meta_core and "meta" in tasks:
                try:
                    meta_result = await tasks["meta"]
                except Exception as e:
                    logger.error(f"Error in meta-cognitive cycle: {str(e)}")
                    meta_result = {"error": str(e)}
            
            # Update interaction tracking
            self.last_interaction = datetime.datetime.now()
            self.interaction_count += 1
            
            # Calculate response time
            end_time = datetime.datetime.now()
            response_time = (end_time - start_time).total_seconds()
            self.performance_metrics["response_times"].append(response_time)
            
            # Return processing results in a structured format
            result = {
                "user_input": user_input,
                "emotional_state": emotional_state,
                "memories": memories,
                "memory_count": len(memories),
                "has_experience": experience_result["has_experience"] if experience_result else False,
                "experience_response": experience_result["response_text"] if experience_result and experience_result["has_experience"] else None,
                "cross_user_experience": experience_result.get("cross_user", False) if experience_result else False,
                "memory_id": memory_id,
                "response_time": response_time,
                "context_change": context_change_result,
                "adaptation_result": adaptation_result,
                "identity_impact": identity_impact,
                "meta_result": meta_result,
                "parallel_processing": True  # Flag to indicate parallel processing was used
            }
            
            return result

    async def enable_self_configuration(self) -> Dict[str, Any]:
        """
        Enable Nyx to dynamically adjust her own configuration parameters
        based on performance metrics, operational patterns, and user feedback.
        
        Returns:
            Status information about the enabled functionality
        """
        if not hasattr(self, "self_config_enabled"):
            self.self_config_enabled = False
            self.config_change_history = []
            self.param_performance_impact = {}
            self.config_update_interval = 50  # Interactions between updates
            self.last_config_update = 0
            self.confidence_thresholds = {
                "low": 0.4,
                "medium": 0.7,
                "high": 0.9
            }
            
            # Define adaptation strategies
            self.adaptation_strategies = {
                "conservative": {
                    "description": "Make small, cautious adjustments with high confidence requirements",
                    "confidence_multiplier": 1.2,
                    "step_multiplier": 0.7,
                    "evaluation_frequency": 1.5  # Longer evaluation periods
                },
                "balanced": {
                    "description": "Make moderate adjustments with balanced confidence requirements",
                    "confidence_multiplier": 1.0,
                    "step_multiplier": 1.0,
                    "evaluation_frequency": 1.0
                },
                "exploratory": {
                    "description": "Make larger adjustments with lower confidence requirements",
                    "confidence_multiplier": 0.8,
                    "step_multiplier": 1.3,
                    "evaluation_frequency": 0.7  # Shorter evaluation periods
                }
            }
            
            # Current strategy - can be adjusted based on context
            self.current_adaptation_strategy = "balanced"
            
            # Define parameter categories for organization
            self.parameter_categories = {
                "core": "Core operational parameters",
                "memory": "Memory system parameters",
                "emotion": "Emotional system parameters",
                "reasoning": "Reasoning system parameters",
                "social": "Cross-user and social parameters",
                "attention": "Attentional system parameters",
                "temporal": "Temporal processing parameters",
                "hormonal": "Hormone system parameters",
                "reflection": "Meta-cognitive reflection parameters",
                "procedural": "Procedural memory parameters",
                "reflexive": "Reflexive system parameters",
                "performance": "System performance parameters"
            }
            
            # Initialize parameter interdependence graph
            self.parameter_dependencies = {}
            
            # Define adjustable parameters with safe ranges, defaults, and categories
            self.adjustable_parameters = {
                # Core parameters
                "cross_user_sharing_threshold": {
                    "current": self.cross_user_sharing_threshold,
                    "min": 0.3,
                    "max": 0.95,
                    "default": 0.7,
                    "step": 0.05,
                    "description": "Threshold for sharing experiences across users",
                    "category": "social",
                    "related_to": ["cross_user_enabled"]
                },
                "memory_to_emotion_influence": {
                    "current": self.memory_to_emotion_influence,
                    "min": 0.1,
                    "max": 0.8,
                    "default": 0.3,
                    "step": 0.05,
                    "description": "How much memories influence emotions",
                    "category": "emotion",
                    "related_to": ["emotion_to_memory_influence"]
                },
                "emotion_to_memory_influence": {
                    "current": self.emotion_to_memory_influence,
                    "min": 0.1,
                    "max": 0.8,
                    "default": 0.4,
                    "step": 0.05,
                    "description": "How much emotions influence memory retrieval",
                    "category": "memory",
                    "related_to": ["memory_to_emotion_influence"]
                },
                "experience_to_identity_influence": {
                    "current": self.experience_to_identity_influence,
                    "min": 0.05,
                    "max": 0.6,
                    "default": 0.2,
                    "step": 0.05,
                    "description": "How much experiences influence identity",
                    "category": "core",
                    "related_to": []
                },
                "cross_user_enabled": {
                    "current": self.cross_user_enabled,
                    "min": 0,  # Boolean as 0/1
                    "max": 1,
                    "default": 1,
                    "step": 1,
                    "description": "Whether cross-user experiences are enabled",
                    "category": "social",
                    "related_to": ["cross_user_sharing_threshold"]
                },
                "consolidation_interval": {
                    "current": self.consolidation_interval,
                    "min": 6,
                    "max": 72,
                    "default": 24,
                    "step": 6,
                    "description": "Hours between experience consolidations",
                    "category": "memory",
                    "related_to": []
                },
                "identity_reflection_interval": {
                    "current": self.identity_reflection_interval,
                    "min": 5,
                    "max": 50,
                    "default": 10,
                    "step": 5,
                    "description": "Interactions between identity reflections",
                    "category": "reflection",
                    "related_to": []
                },
                
                # Memory system parameters
                "memory_recency_weight": {
                    "current": 0.7 if hasattr(self.memory_orchestrator, "recency_weight") else 0.7,
                    "min": 0.3,
                    "max": 0.9,
                    "default": 0.7,
                    "step": 0.05,
                    "description": "Weight given to recency in memory retrieval",
                    "category": "memory",
                    "related_to": ["memory_relevance_weight"]
                },
                "memory_relevance_weight": {
                    "current": 0.8 if hasattr(self.memory_orchestrator, "relevance_weight") else 0.8,
                    "min": 0.4,
                    "max": 0.95,
                    "default": 0.8,
                    "step": 0.05,
                    "description": "Weight given to relevance in memory retrieval",
                    "category": "memory",
                    "related_to": ["memory_recency_weight"]
                },
                "memory_significance_threshold": {
                    "current": 3 if hasattr(self.memory_core, "significance_threshold") else 3,
                    "min": 1,
                    "max": 8,
                    "default": 3,
                    "step": 1,
                    "description": "Minimum significance for memories to be retrieved",
                    "category": "memory",
                    "related_to": []
                },
                
                # Emotional system parameters
                "emotional_decay_rate": {
                    "current": 0.05 if hasattr(self.emotional_core, "decay_rate") else 0.05,
                    "min": 0.01,
                    "max": 0.3,
                    "default": 0.05,
                    "step": 0.01,
                    "description": "Rate at which emotions decay over time",
                    "category": "emotion",
                    "related_to": []
                },
                "emotional_expression_threshold": {
                    "current": 0.7 if hasattr(self.emotional_core, "expression_threshold") else 0.7,
                    "min": 0.4,
                    "max": 0.9,
                    "default": 0.7,
                    "step": 0.05,
                    "description": "Threshold for expressing emotions",
                    "category": "emotion",
                    "related_to": []
                },
                "emotional_complexity": {
                    "current": 0.6 if hasattr(self.emotional_core, "complexity") else 0.6,
                    "min": 0.1,
                    "max": 1.0,
                    "default": 0.6,
                    "step": 0.1,
                    "description": "Complexity of emotional expressions",
                    "category": "emotion",
                    "related_to": []
                },
                
                # Attention system parameters
                "attentional_focus_duration": {
                    "current": 5 if hasattr(self, "attentional_controller") and hasattr(self.attentional_controller, "focus_duration") else 5,
                    "min": 1,
                    "max": 20,
                    "default": 5,
                    "step": 1,
                    "description": "Duration of attentional focus (in interactions)",
                    "category": "attention",
                    "related_to": []
                },
                "novelty_weight": {
                    "current": 0.7 if hasattr(self, "attentional_controller") and hasattr(self.attentional_controller, "novelty_weight") else 0.7,
                    "min": 0.3,
                    "max": 0.9,
                    "default": 0.7,
                    "step": 0.05,
                    "description": "Weight given to novelty in attention",
                    "category": "attention",
                    "related_to": []
                },
                
                # Hormone system parameters
                "hormone_influence_factor": {
                    "current": 0.6 if hasattr(self, "hormone_system") and hasattr(self.hormone_system, "influence_factor") else 0.6,
                    "min": 0.2,
                    "max": 1.0,
                    "default": 0.6,
                    "step": 0.1,
                    "description": "Overall influence of hormones on other systems",
                    "category": "hormonal",
                    "related_to": []
                },
                "hormonal_cycle_speed": {
                    "current": 1.0 if hasattr(self, "hormone_system") and hasattr(self.hormone_system, "cycle_speed") else 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "default": 1.0,
                    "step": 0.1,
                    "description": "Speed of hormonal cycles (multiplier)",
                    "category": "hormonal",
                    "related_to": []
                },
                
                # Reasoning system parameters
                "reasoning_depth": {
                    "current": 2 if hasattr(self.reasoning_core, "reasoning_depth") else 2,
                    "min": 1,
                    "max": 4,
                    "default": 2,
                    "step": 1,
                    "description": "Depth of reasoning processes",
                    "category": "reasoning",
                    "related_to": []
                },
                "thinking_frequency": {
                    "current": 0.4 if hasattr(self, "thinking_config") and "thinking_enabled" in self.thinking_config else 0.4,
                    "min": 0.1,
                    "max": 0.8,
                    "default": 0.4,
                    "step": 0.1,
                    "description": "Frequency of using extended thinking",
                    "category": "reasoning",
                    "related_to": []
                },
                
                # Reflexive system parameters
                "reflex_threshold": {
                    "current": 0.6 if hasattr(self, "reflexive_system") and hasattr(self.reflexive_system, "default_threshold") else 0.6,
                    "min": 0.4,
                    "max": 0.9,
                    "default": 0.6,
                    "step": 0.05,
                    "description": "Threshold for triggering reflexes",
                    "category": "reflexive",
                    "related_to": []
                },
                "procedural_confidence_threshold": {
                    "current": 0.7 if hasattr(self, "agent_enhanced_memory") and hasattr(self.agent_enhanced_memory, "confidence_threshold") else 0.7,
                    "min": 0.5,
                    "max": 0.95,
                    "default": 0.7,
                    "step": 0.05,
                    "description": "Confidence threshold for procedural memory execution",
                    "category": "procedural",
                    "related_to": []
                },
                
                # System performance parameters
                "parallel_processing_threshold": {
                    "current": 0.6,  # Complexity threshold for using parallel processing
                    "min": 0.3,
                    "max": 0.9,
                    "default": 0.6,
                    "step": 0.1,
                    "description": "Complexity threshold for switching to parallel processing",
                    "category": "performance",
                    "related_to": []
                },
                "distributed_processing_threshold": {
                    "current": 0.8,  # Complexity threshold for using distributed processing
                    "min": 0.5,
                    "max": 0.95,
                    "default": 0.8,
                    "step": 0.05,
                    "description": "Complexity threshold for switching to distributed processing",
                    "category": "performance",
                    "related_to": ["parallel_processing_threshold"]
                }
            }
            
            # Build parameter dependency graph
            for param_name, param_config in self.adjustable_parameters.items():
                self.parameter_dependencies[param_name] = {
                    "affects": [],
                    "affected_by": param_config["related_to"]
                }
                
            # Complete bidirectional dependencies
            for param_name, dependencies in self.parameter_dependencies.items():
                for related_param in dependencies["affected_by"]:
                    if related_param in self.parameter_dependencies:
                        if param_name not in self.parameter_dependencies[related_param]["affects"]:
                            self.parameter_dependencies[related_param]["affects"].append(param_name)
            
            # Performance metrics to track for adaptation
            self.parameter_metrics_map = {
                # Core parameters
                "cross_user_sharing_threshold": ["experiences_shared", "cross_user_experiences_shared"],
                "memory_to_emotion_influence": ["emotion_updates"],
                "emotion_to_memory_influence": ["memory_operations"],
                "experience_to_identity_influence": ["experiences_shared"],
                "cross_user_enabled": ["cross_user_experiences_shared"],
                "consolidation_interval": ["experience_consolidations"],
                "identity_reflection_interval": ["reflections_generated"],
                
                # Memory system parameters
                "memory_recency_weight": ["memory_operations", "avg_response_time"],
                "memory_relevance_weight": ["memory_operations", "experiences_shared"],
                "memory_significance_threshold": ["memory_operations", "memory_count"],
                
                # Emotional system parameters
                "emotional_decay_rate": ["emotion_updates"],
                "emotional_expression_threshold": ["emotion_updates"],
                "emotional_complexity": ["emotion_updates"],
                
                # Attention system parameters
                "attentional_focus_duration": ["memory_operations", "experiences_shared"],
                "novelty_weight": ["memory_operations", "experiences_shared"],
                
                # Hormone system parameters
                "hormone_influence_factor": ["emotion_updates", "experiences_shared"],
                "hormonal_cycle_speed": ["emotion_updates"],
                
                # Reasoning system parameters
                "reasoning_depth": ["avg_response_time"],
                "thinking_frequency": ["avg_response_time"],
                
                # Reflexive system parameters
                "reflex_threshold": ["avg_response_time"],
                "procedural_confidence_threshold": ["avg_response_time"],
                
                # System performance parameters
                "parallel_processing_threshold": ["avg_response_time"],
                "distributed_processing_threshold": ["avg_response_time"]
            }
            
            # User feedback impact tracking
            self.user_feedback_impact = {
                "positive_feedback": {},
                "negative_feedback": {},
                "parameter_adjustments": {}
            }
            
            # Meta-learning: track which parameters have the most impact
            self.parameter_impact_ranking = {}
            
            self.self_config_enabled = True
            
            # Schedule periodic parameter evaluation
            asyncio.create_task(self._parameter_evaluation_loop())
            
        return {
            "enabled": self.self_config_enabled,
            "adjustable_parameters_count": len(self.adjustable_parameters),
            "categories": {k: v for k, v in self.parameter_categories.items()},
            "available_strategies": {k: v["description"] for k, v in self.adaptation_strategies.items()},
            "current_strategy": self.current_adaptation_strategy,
            "update_interval": self.config_update_interval,
            "last_update": self.last_config_update
        }
    
    async def _parameter_evaluation_loop(self):
        """Background task to periodically evaluate and adjust parameters"""
        while self.self_config_enabled:
            # Only evaluate after certain number of interactions
            if self.interaction_count - self.last_config_update >= self.config_update_interval:
                await self.evaluate_and_adjust_parameters()
                self.last_config_update = self.interaction_count
                
                # Also update meta-learning
                await self._update_parameter_meta_learning()
                
            # Wait before checking again
            await asyncio.sleep(60)  # Check every minute
    
    async def _update_parameter_meta_learning(self):
        """Update meta-learning about which parameters are most impactful"""
        # Calculate impact scores for each parameter
        impact_scores = {}
        
        for param_name, param_data in self.param_performance_impact.items():
            if param_data["history"]:
                # Calculate normalized impact
                impacts = [abs(entry["impact"]) for entry in param_data["history"]]
                avg_impact = sum(impacts) / len(impacts)
                
                # Consider number of samples for confidence
                sample_confidence = min(1.0, len(impacts) / 10)
                
                # Calculate final score
                impact_scores[param_name] = avg_impact * sample_confidence
        
        # Rank parameters by impact score
        self.parameter_impact_ranking = {
            k: {"score": v, "rank": i+1} 
            for i, (k, v) in enumerate(sorted(impact_scores.items(), key=lambda x: x[1], reverse=True))
        }
        
        # Adjust evaluation priorities
        high_impact_params = [k for k, v in self.parameter_impact_ranking.items() 
                              if v["rank"] <= 5]  # Top 5 parameters
        
        for param_name in high_impact_params:
            # Increase evaluation frequency for high-impact parameters
            if param_name not in self.parameter_evaluation_priority:
                self.parameter_evaluation_priority[param_name] = 1.5  # 50% more likely to be evaluated
    
    async def evaluate_and_adjust_parameters(self) -> Dict[str, Any]:
        """
        Evaluate current performance metrics and adjust parameters if needed
        
        Returns:
            Results of parameter adjustments
        """
        if not self.self_config_enabled:
            return {"status": "disabled"}
        
        results = {
            "evaluated": [],
            "adjusted": [],
            "timestamp": datetime.datetime.now().isoformat(),
            "strategy": self.current_adaptation_strategy
        }
        
        # Get current strategy configuration
        strategy = self.adaptation_strategies[self.current_adaptation_strategy]
        confidence_multiplier = strategy["confidence_multiplier"]
        step_multiplier = strategy["step_multiplier"]
        evaluation_frequency = strategy["evaluation_frequency"]
        
        # Get current performance metrics
        stats = await self.get_system_stats()
        performance = stats["performance_metrics"]
        
        # Check current emotional state to influence strategy
        emotional_state = stats["emotional_state"]
        dominant_emotion = emotional_state.get("dominant_emotion")
        
        # Adjust strategy based on emotion if needed
        adjusted_step_multiplier = step_multiplier
        if dominant_emotion in ["Joy", "Trust", "Anticipation"]:
            # More positive emotions -> slightly more exploratory
            adjusted_step_multiplier *= 1.1
        elif dominant_emotion in ["Fear", "Anger", "Disgust"]:
            # More negative emotions -> slightly more conservative
            adjusted_step_multiplier *= 0.9
        
        # Check if brain is in an abnormal state to prioritize stability
        abnormal_state = False
        if performance.get("avg_response_time", 0) > 1.5:  # High response time
            abnormal_state = True
        
        # Get highest priority parameters to evaluate
        eval_params = []
        
        # Always evaluate high-impact or abnormal-state-related parameters
        if abnormal_state:
            # During abnormal state, prioritize performance parameters
            eval_params.extend([p for p, c in self.adjustable_parameters.items() 
                               if c["category"] == "performance"])
        
        # Add meta-learned high-impact parameters
        if hasattr(self, "parameter_impact_ranking") and self.parameter_impact_ranking:
            high_impact = [k for k, v in self.parameter_impact_ranking.items() 
                          if v["rank"] <= 3]  # Top 3
            eval_params.extend(high_impact)
        
        # Add random parameters to evaluate
        remaining_params = [p for p in self.adjustable_parameters.keys() 
                          if p not in eval_params]
        
        # Determine how many to evaluate based on strategy's evaluation frequency
        eval_count = max(3, int(len(remaining_params) * 0.3 * evaluation_frequency))
        random_params = random.sample(remaining_params, min(eval_count, len(remaining_params)))
        eval_params.extend(random_params)
        
        # Ensure no duplicates
        eval_params = list(set(eval_params))
        
        # For each parameter to evaluate
        for param_name in eval_params:
            if param_name not in self.adjustable_parameters:
                continue
                
            param_config = self.adjustable_parameters[param_name]
            results["evaluated"].append(param_name)
            
            # Get relevant metrics for this parameter
            relevant_metrics = self.parameter_metrics_map.get(param_name, [])
            if not relevant_metrics:
                continue
                
            # Calculate current performance score for these metrics
            current_score = sum(performance.get(metric, 0) for metric in relevant_metrics)
            
            # Analyze historical performance for this parameter
            should_adjust, direction, confidence = await self._analyze_parameter_performance(
                param_name, 
                current_score,
                relevant_metrics
            )
            
            # Apply strategy confidence modifier
            adjusted_confidence = confidence * confidence_multiplier
            
            if should_adjust and adjusted_confidence >= self.confidence_thresholds["medium"]:
                # Calculate step size based on confidence and strategy
                confidence_factor = min(1.0, adjusted_confidence / self.confidence_thresholds["high"])
                step_size = param_config["step"] * direction * adjusted_step_multiplier
                
                # Check for parameter dependencies
                dependency_adjusted_step = await self._adjust_for_dependencies(
                    param_name, step_size, confidence_factor
                )
                
                # Calculate new value using the adjusted step
                current = param_config["current"]
                new_value = current + dependency_adjusted_step
                
                # Clamp to safe range
                new_value = max(param_config["min"], min(param_config["max"], new_value))
                
                # Special case for boolean parameters
                if param_config["min"] == 0 and param_config["max"] == 1 and param_config["step"] == 1:
                    new_value = round(new_value)
                
                # Only adjust if the value actually changed
                if new_value != current:
                    # Update the parameter
                    await self._update_parameter(param_name, new_value)
                    
                    results["adjusted"].append({
                        "parameter": param_name,
                        "old_value": current,
                        "new_value": new_value,
                        "direction": "increase" if direction > 0 else "decrease",
                        "confidence": adjusted_confidence,
                        "relevant_metrics": relevant_metrics,
                        "category": param_config["category"]
                    })
                    
                    # If we changed an important parameter in abnormal state, stop
                    # to observe its effects before making more changes
                    if abnormal_state and param_config["category"] == "performance":
                        break
        
        # If meta-cognitive reflection is available, generate a reflection on the changes
        if hasattr(self, "reflection_engine") and self.reflection_engine and results["adjusted"]:
            reflection = await self.reflection_engine.generate_reflection(
                topic=f"parameter adjustment ({len(results['adjusted'])} changes)",
                context={
                    "adjustments": results["adjusted"],
                    "performance": performance,
                    "strategy": self.current_adaptation_strategy
                }
            )
            results["reflection"] = reflection
            
            # Also check if we should change adaptation strategy
            if len(results["adjusted"]) > 5:
                # Many parameters changed, consider more conservative approach
                if self.current_adaptation_strategy == "exploratory":
                    results["strategy_recommendation"] = "Consider switching to balanced strategy after many parameters changed"
            elif len(results["adjusted"]) == 0 and len(results["evaluated"]) > 10:
                # Many evaluations but no changes, consider more exploratory approach
                if self.current_adaptation_strategy == "conservative":
                    results["strategy_recommendation"] = "Consider switching to balanced strategy after few parameter changes"
        
        return results
    
    async def _adjust_for_dependencies(self, param_name, step_size, confidence_factor):
        """
        Adjust step size based on parameter dependencies
        
        Args:
            param_name: Parameter being adjusted
            step_size: Proposed step size
            confidence_factor: Confidence in the adjustment (0-1)
            
        Returns:
            Adjusted step size
        """
        # Get dependencies
        dependencies = self.parameter_dependencies.get(param_name, {"affects": [], "affected_by": []})
        
        # If no dependencies, return original step
        if not dependencies["affected_by"] and not dependencies["affects"]:
            return step_size
        
        # Check for conflicts with affected parameters
        for affected_param in dependencies["affects"]:
            if affected_param not in self.adjustable_parameters:
                continue
                
            # Get affected parameter recent history
            if affected_param in self.param_performance_impact:
                history = self.param_performance_impact[affected_param]["history"]
                if not history:
                    continue
                    
                # Check if affected parameter is already improving
                recent_entries = history[-3:]
                if len(recent_entries) < 2:
                    continue
                    
                avg_impact = sum(entry["impact"] for entry in recent_entries) / len(recent_entries)
                
                # If affected parameter is already improving well, reduce our step
                if avg_impact > 0.1:
                    step_size *= 0.8  # Reduce step size
        
        # Check if we're affected by other parameters
        for affecting_param in dependencies["affected_by"]:
            if affecting_param not in self.adjustable_parameters:
                continue
                
            # Check if affecting parameter recently changed
            if hasattr(self, "config_change_history") and self.config_change_history:
                recent_changes = [change for change in self.config_change_history[-5:] 
                                 if change["parameter"] == affecting_param]
                
                if recent_changes:
                    # Recent change in a parameter that affects us
                    # Reduce our step size to avoid interference
                    step_size *= 0.7
        
        # Adjust based on confidence
        final_step = step_size * (0.7 + (0.3 * confidence_factor))
        
        return final_step
    
    async def _analyze_parameter_performance(self, 
                                           param_name: str, 
                                           current_score: float,
                                           relevant_metrics: List[str]) -> Tuple[bool, float, float]:
        """
        Analyze whether a parameter should be adjusted and in which direction
        
        Args:
            param_name: Name of parameter to analyze
            current_score: Current performance score
            relevant_metrics: List of relevant metric names
            
        Returns:
            Tuple of (should_adjust, direction, confidence)
        """
        # Initialize if not in history
        if param_name not in self.param_performance_impact:
            self.param_performance_impact[param_name] = {
                "history": [],
                "baseline": current_score
            }
        
        param_data = self.param_performance_impact[param_name]
        history = param_data["history"]
        
        # If no history yet, establish baseline and make exploratory change
        if not history:
            # Direction based on user feedback if available
            direction = 0.0
            
            if hasattr(self, "user_feedback_impact") and param_name in self.user_feedback_impact["parameter_adjustments"]:
                feedback_data = self.user_feedback_impact["parameter_adjustments"][param_name]
                if feedback_data.get("recommended_direction") is not None:
                    direction = feedback_data["recommended_direction"]
            
            # Default to increase if no feedback
            if direction == 0.0:
                direction = 1.0
                
            return True, direction, 0.5  # Exploratory with medium confidence
        
        # Calculate average performance change per direction
        increases = [entry for entry in history if entry["direction"] > 0]
        decreases = [entry for entry in history if entry["direction"] < 0]
        
        # Weight recent results more heavily
        recent_increases = [entry for entry in increases if entry.get("change_id", 0) >= len(self.config_change_history) - 5]
        recent_decreases = [entry for entry in decreases if entry.get("change_id", 0) >= len(self.config_change_history) - 5]
        
        # Calculate weighted impacts
        avg_increase_impact = 0
        if increases:
            if recent_increases:
                avg_increase_impact = sum(entry["impact"] * 1.5 for entry in recent_increases) / len(recent_increases)
            else:
                avg_increase_impact = sum(entry["impact"] for entry in increases) / len(increases)
        
        avg_decrease_impact = 0
        if decreases:
            if recent_decreases:
                avg_decrease_impact = sum(entry["impact"] * 1.5 for entry in recent_decreases) / len(recent_decreases)
            else:
                avg_decrease_impact = sum(entry["impact"] for entry in decreases) / len(decreases)
        
        # Consider user feedback
        if hasattr(self, "user_feedback_impact") and param_name in self.user_feedback_impact["parameter_adjustments"]:
            feedback_data = self.user_feedback_impact["parameter_adjustments"][param_name]
            
            if feedback_data.get("recommended_direction") == 1.0:
                # User feedback suggests increase
                avg_increase_impact += 0.1
            elif feedback_data.get("recommended_direction") == -1.0:
                # User feedback suggests decrease
                avg_decrease_impact += 0.1
        
        # Decide direction based on historical impact
        if avg_increase_impact > avg_decrease_impact and avg_increase_impact > 0:
            direction = 1.0  # Increase
            expected_impact = avg_increase_impact
        elif avg_decrease_impact > 0:
            direction = -1.0  # Decrease
            expected_impact = avg_decrease_impact
        else:
            # If both negative or no data for one direction, explore
            if not increases:
                direction = 1.0  # Try increasing
                expected_impact = 0.1  # Assumed positive impact
            elif not decreases:
                direction = -1.0  # Try decreasing
                expected_impact = 0.1  # Assumed positive impact
            else:
                # Both directions tried with negative results, revert to default
                param_config = self.adjustable_parameters[param_name]
                current = param_config["current"]
                default = param_config["default"]
                
                if abs(current - default) > param_config["step"]:
                    # Move toward default
                    direction = 1.0 if default > current else -1.0
                    expected_impact = 0.1  # Assumed positive impact
                else:
                    # Already at/near default, don't change
                    return False, 0.0, 0.0
        
        # Calculate confidence based on history consistency
        if direction > 0:
            samples = increases
        else:
            samples = decreases
        
        if samples:
            # Calculate consistency of results
            if len(samples) > 1:
                try:
                    consistency = statistics.stdev([entry["impact"] for entry in samples])
                    consistency = max(0.1, min(1.0, 1.0 - consistency))  # Lower variance = higher consistency
                except statistics.StatisticsError:
                    consistency = 0.5  # Default if stdev fails
            else:
                consistency = 0.5  # Single sample
                
            # Calculate trend (if recent results show improving trend)
            if len(samples) >= 3:
                recent = sorted(samples, key=lambda x: x.get("change_id", 0), reverse=True)[:3]
                if len(recent) >= 2:
                    first_impacts = [e["impact"] for e in recent[1:]]
                    latest_impact = recent[0]["impact"]
                    avg_first = sum(first_impacts) / len(first_impacts)
                    
                    if latest_impact > avg_first:
                        # Improving trend
                        trend_bonus = 0.1
                    else:
                        # Declining trend
                        trend_bonus = -0.1
                else:
                    trend_bonus = 0
            else:
                trend_bonus = 0
                
            confidence = min(0.9, len(samples) / 10 + consistency * 0.3 + trend_bonus)
        else:
            confidence = 0.3  # Low confidence for exploration
        
        # Check if this parameter is important according to meta-learning
        if hasattr(self, "parameter_impact_ranking") and param_name in self.parameter_impact_ranking:
            rank = self.parameter_impact_ranking[param_name]["rank"]
            if rank <= 5:  # Top 5 important parameter
                confidence *= 1.1  # Slight confidence boost
        
        # Whether to adjust depends on confidence and expected impact
        should_adjust = confidence >= self.confidence_thresholds["low"] and expected_impact > 0
        
        return should_adjust, direction, confidence
    
    async def _update_parameter(self, param_name: str, new_value: float) -> None:
        """
        Update a parameter to a new value and record the change
        
        Args:
            param_name: Name of parameter to update
            new_value: New value for the parameter
        """
        param_config = self.adjustable_parameters[param_name]
        old_value = param_config["current"]
        
        # Record history before change
        stats_before = await self.get_system_stats()
        baseline_score = sum(stats_before["performance_metrics"].get(metric, 0) 
                            for metric in self.parameter_metrics_map.get(param_name, []))
        
        # Update parameter config tracking
        param_config["current"] = new_value
        self.adjustable_parameters[param_name] = param_config
        
        # Update actual parameter
        if hasattr(self, param_name):
            setattr(self, param_name, new_value)
        
        # Record change
        change_record = {
            "parameter": param_name,
            "old_value": old_value,
            "new_value": new_value,
            "timestamp": datetime.datetime.now().isoformat(),
            "interaction_count": self.interaction_count,
            "direction": 1.0 if new_value > old_value else -1.0,
            "baseline_score": baseline_score,
            "category": param_config["category"],
            "strategy": self.current_adaptation_strategy
        }
        
        self.config_change_history.append(change_record)
        
        # Schedule impact evaluation
        asyncio.create_task(self._evaluate_parameter_impact(param_name, change_record))
        
        # Log the change
        logger.info(f"Self-adjusted parameter {param_name}: {old_value} -> {new_value} (using {self.current_adaptation_strategy} strategy)")
    
    async def _evaluate_parameter_impact(self, param_name: str, change_record: Dict[str, Any]) -> None:
        """
        Evaluate the impact of a parameter change after a period of time
        
        Args:
            param_name: Name of parameter that was changed
            change_record: Record of the parameter change
        """
        # Wait for sufficient interactions to evaluate impact
        interactions_to_wait = min(self.config_update_interval // 2, 25)
        
        # Either wait for X interactions or a timeout
        start_interaction = self.interaction_count
        start_time = datetime.datetime.now()
        
        while (self.interaction_count - start_interaction < interactions_to_wait and
               (datetime.datetime.now() - start_time).total_seconds() < 3600):
            await asyncio.sleep(30)  # Check every 30 seconds
        
        # Get current metrics
        stats_after = await self.get_system_stats()
        current_score = sum(stats_after["performance_metrics"].get(metric, 0) 
                           for metric in self.parameter_metrics_map.get(param_name, []))
        
        # Calculate impact
        baseline_score = change_record["baseline_score"]
        raw_impact = current_score - baseline_score
        
        # Normalize by interaction count
        interaction_diff = self.interaction_count - change_record["interaction_count"]
        if interaction_diff > 0:
            normalized_impact = raw_impact / interaction_diff
        else:
            normalized_impact = raw_impact
        
        # Adjust for global performance trends
        # If overall performance improved/declined, adjust individual impact
        overall_before = sum(stats_before["performance_metrics"].values()) if "stats_before" in locals() else 0
        overall_after = sum(stats_after["performance_metrics"].values())
        overall_change = overall_after - overall_before
        
        if overall_change != 0 and "stats_before" in locals():
            # Remove global trend component
            global_component = overall_change * 0.3  # Only attribute 30% to global trends
            adjusted_impact = normalized_impact - global_component
        else:
            adjusted_impact = normalized_impact
        
        # Update parameter history
        if param_name not in self.param_performance_impact:
            self.param_performance_impact[param_name] = {
                "history": [],
                "baseline": baseline_score
            }
        
        # Add impact record
        impact_record = {
            "change_id": len(self.config_change_history) - 1,
            "old_value": change_record["old_value"],
            "new_value": change_record["new_value"],
            "direction": change_record["direction"],
            "baseline_score": baseline_score,
            "current_score": current_score,
            "raw_impact": raw_impact,
            "normalized_impact": normalized_impact,
            "adjusted_impact": adjusted_impact,
            "impact": adjusted_impact,  # Alias for simple reference
            "interactions_measured": interaction_diff,
            "time_measured": (datetime.datetime.now() - start_time).total_seconds(),
            "category": change_record["category"],
            "strategy": change_record["strategy"]
        }
        
        self.param_performance_impact[param_name]["history"].append(impact_record)
        
        # Log the impact evaluation
        logger.info(f"Parameter {param_name} change impact: {adjusted_impact:.4f}")
        
        # Check for extreme impact (very positive or negative)
        if abs(adjusted_impact) > 0.3:
            # Generate reflection on significant impact
            if hasattr(self, "reflection_engine") and self.reflection_engine:
                impact_direction = "positive" if adjusted_impact > 0 else "negative"
                asyncio.create_task(
                    self.reflection_engine.generate_reflection(
                        topic=f"significant {impact_direction} parameter impact",
                        context={
                            "parameter": param_name,
                            "impact": adjusted_impact,
                            "change": {
                                "old_value": change_record["old_value"],
                                "new_value": change_record["new_value"]
                            }
                        }
                    )
                )
    
    async def process_user_feedback_for_configuration(self, 
                                                   feedback_type: str, 
                                                   feedback_text: str,
                                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user feedback to influence configuration parameters
        
        Args:
            feedback_type: Type of feedback ("positive", "negative", "specific")
            feedback_text: Text of user feedback
            context: Additional context information
            
        Returns:
            Processing results
        """
        if not hasattr(self, "self_config_enabled") or not self.self_config_enabled:
            return {"status": "self-configuration not enabled"}
        
        results = {
            "processed": True,
            "feedback_type": feedback_type,
            "affected_parameters": []
        }
        
        # Initialize context
        context = context or {}
        
        # Track feedback
        if feedback_type not in self.user_feedback_impact:
            self.user_feedback_impact[feedback_type] = {}
        
        # Process based on feedback type
        if feedback_type == "positive":
            # Positive feedback reinforces current parameter settings
            recent_changes = self.config_change_history[-5:] if self.config_change_history else []
            
            for change in recent_changes:
                param_name = change["parameter"]
                
                if param_name not in self.user_feedback_impact["positive_feedback"]:
                    self.user_feedback_impact["positive_feedback"][param_name] = {
                        "count": 0,
                        "recent_directions": []
                    }
                
                self.user_feedback_impact["positive_feedback"][param_name]["count"] += 1
                self.user_feedback_impact["positive_feedback"][param_name]["recent_directions"].append(change["direction"])
                
                if param_name not in self.user_feedback_impact["parameter_adjustments"]:
                    self.user_feedback_impact["parameter_adjustments"][param_name] = {
                        "positive_count": 0,
                        "negative_count": 0,
                        "recommended_direction": None
                    }
                
                self.user_feedback_impact["parameter_adjustments"][param_name]["positive_count"] += 1
                
                # Update recommended direction based on positive feedback
                recent_dirs = self.user_feedback_impact["positive_feedback"][param_name]["recent_directions"]
                if len(recent_dirs) >= 2:
                    # If recent changes have been consistently in one direction and got positive feedback,
                    # recommend continuing in that direction
                    if all(d > 0 for d in recent_dirs):
                        self.user_feedback_impact["parameter_adjustments"][param_name]["recommended_direction"] = 1.0
                    elif all(d < 0 for d in recent_dirs):
                        self.user_feedback_impact["parameter_adjustments"][param_name]["recommended_direction"] = -1.0
                
                results["affected_parameters"].append({
                    "parameter": param_name,
                    "current_value": self.adjustable_parameters[param_name]["current"],
                    "positive_reinforcement": True
                })
        
        elif feedback_type == "negative":
            # Negative feedback suggests possibly reverting recent changes
            recent_changes = self.config_change_history[-5:] if self.config_change_history else []
            
            for change in recent_changes:
                param_name = change["parameter"]
                
                if param_name not in self.user_feedback_impact["negative_feedback"]:
                    self.user_feedback_impact["negative_feedback"][param_name] = {
                        "count": 0,
                        "recent_directions": []
                    }
                
                self.user_feedback_impact["negative_feedback"][param_name]["count"] += 1
                self.user_feedback_impact["negative_feedback"][param_name]["recent_directions"].append(change["direction"])
                
                if param_name not in self.user_feedback_impact["parameter_adjustments"]:
                    self.user_feedback_impact["parameter_adjustments"][param_name] = {
                        "positive_count": 0,
                        "negative_count": 0,
                        "recommended_direction": None
                    }
                
                self.user_feedback_impact["parameter_adjustments"][param_name]["negative_count"] += 1
                
                # Update recommended direction based on negative feedback (opposite of recent change)
                self.user_feedback_impact["parameter_adjustments"][param_name]["recommended_direction"] = -1 * change["direction"]
                
                results["affected_parameters"].append({
                    "parameter": param_name,
                    "current_value": self.adjustable_parameters[param_name]["current"],
                    "recommended_direction": -1 * change["direction"]
                })
        
        elif feedback_type == "specific" and feedback_text:
            # Try to extract specific parameter feedback
            for param_name, param_config in self.adjustable_parameters.items():
                # Check if parameter is mentioned
                if param_name.replace("_", " ") in feedback_text.lower():
                    # Determine sentiment
                    positive_terms = ["good", "better", "improve", "well", "like", "great", "excellent", "perfect"]
                    negative_terms = ["bad", "worse", "poor", "issue", "problem", "dislike", "terrible", "wrong"]
                    
                    positive_sentiment = any(term in feedback_text.lower() for term in positive_terms)
                    negative_sentiment = any(term in feedback_text.lower() for term in negative_terms)
                    
                    # Determine direction
                    increase_terms = ["increase", "higher", "more", "stronger", "boost", "raise"]
                    decrease_terms = ["decrease", "lower", "less", "weaker", "reduce", "lessen"]
                    
                    increase_direction = any(term in feedback_text.lower() for term in increase_terms)
                    decrease_direction = any(term in feedback_text.lower() for term in decrease_terms)
                    
                    # Record feedback
                    if param_name not in self.user_feedback_impact["parameter_adjustments"]:
                        self.user_feedback_impact["parameter_adjustments"][param_name] = {
                            "positive_count": 0,
                            "negative_count": 0,
                            "recommended_direction": None
                        }
                    
                    if positive_sentiment:
                        self.user_feedback_impact["parameter_adjustments"][param_name]["positive_count"] += 1
                    if negative_sentiment:
                        self.user_feedback_impact["parameter_adjustments"][param_name]["negative_count"] += 1
                    
                    # Set recommended direction
                    if increase_direction:
                        self.user_feedback_impact["parameter_adjustments"][param_name]["recommended_direction"] = 1.0
                    elif decrease_direction:
                        self.user_feedback_impact["parameter_adjustments"][param_name]["recommended_direction"] = -1.0
                    
                    results["affected_parameters"].append({
                        "parameter": param_name,
                        "current_value": param_config["current"],
                        "sentiment": "positive" if positive_sentiment else "negative" if negative_sentiment else "neutral",
                        "direction": "increase" if increase_direction else "decrease" if decrease_direction else "unspecified"
                    })
                    
                    # If there's a clear recommendation, make an immediate adjustment
                    if (positive_sentiment or negative_sentiment) and (increase_direction or decrease_direction):
                        direction = 1.0 if increase_direction else -1.0
                        new_value = param_config["current"] + (param_config["step"] * direction)
                        
                        # Clamp to safe range
                        new_value = max(param_config["min"], min(param_config["max"], new_value))
                        
                        # Only adjust if the value actually changed
                        if new_value != param_config["current"]:
                            await self._update_parameter(param_name, new_value)
                            results["immediate_adjustment"] = {
                                "parameter": param_name,
                                "old_value": param_config["current"],
                                "new_value": new_value,
                                "based_on": "explicit user feedback"
                            }
        
        # If multiple parameters were negatively affected, consider switching strategy
        if feedback_type == "negative" and len(results["affected_parameters"]) >= 3:
            if self.current_adaptation_strategy == "exploratory":
                # Switch to more conservative strategy
                self.current_adaptation_strategy = "balanced"
                results["strategy_change"] = {
                    "old_strategy": "exploratory",
                    "new_strategy": "balanced",
                    "reason": "multiple parameters received negative feedback"
                }
            elif self.current_adaptation_strategy == "balanced":
                # Switch to more conservative strategy
                self.current_adaptation_strategy = "conservative"
                results["strategy_change"] = {
                    "old_strategy": "balanced",
                    "new_strategy": "conservative",
                    "reason": "multiple parameters received negative feedback"
                }
        
        return results
    
    async def change_adaptation_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """
        Change the current adaptation strategy
        
        Args:
            strategy_name: Name of strategy to use
            
        Returns:
            Strategy change results
        """
        if not hasattr(self, "self_config_enabled") or not self.self_config_enabled:
            return {"status": "self-configuration not enabled"}
        
        if strategy_name not in self.adaptation_strategies:
            return {
                "status": "error",
                "message": f"Unknown strategy: {strategy_name}. Available strategies: {list(self.adaptation_strategies.keys())}"
            }
        
        old_strategy = self.current_adaptation_strategy
        self.current_adaptation_strategy = strategy_name
        
        return {
            "status": "changed",
            "old_strategy": old_strategy,
            "new_strategy": strategy_name,
            "strategy_description": self.adaptation_strategies[strategy_name]["description"]
        }
    
    async def get_self_configuration_status(self) -> Dict[str, Any]:
        """
        Get status of the self-configuration system
        
        Returns:
            Current status and history of the self-configuration system
        """
        if not hasattr(self, "self_config_enabled") or not self.self_config_enabled:
            return {"enabled": False}
        
        # Get current parameter values by category
        current_params = {}
        for category in self.parameter_categories:
            current_params[category] = {}
        
        for name, config in self.adjustable_parameters.items():
            category = config["category"]
            current_params[category][name] = {
                "current": config["current"],
                "default": config["default"],
                "min": config["min"],
                "max": config["max"],
                "description": config["description"]
            }
        
        # Summarize change history
        recent_changes = self.config_change_history[-10:] if self.config_change_history else []
        
        # Get change statistics by category
        category_changes = {}
        for category in self.parameter_categories:
            category_changes[category] = 0
        
        for change in self.config_change_history:
            category = self.adjustable_parameters[change["parameter"]]["category"]
            category_changes[category] += 1
        
        # Summarize performance impact
        param_impact = {}
        for param_name, data in self.param_performance_impact.items():
            if data["history"]:
                # Calculate overall impact trend
                impacts = [entry["impact"] for entry in data["history"]]
                avg_impact = sum(impacts) / len(impacts)
                
                # Calculate best direction
                increases = [entry for entry in data["history"] if entry["direction"] > 0]
                decreases = [entry for entry in data["history"] if entry["direction"] < 0]
                
                avg_increase_impact = sum(entry["impact"] for entry in increases) / len(increases) if increases else 0
                avg_decrease_impact = sum(entry["impact"] for entry in decreases) / len(decreases) if decreases else 0
                
                best_direction = "increase" if avg_increase_impact > avg_decrease_impact else "decrease"
                
                param_impact[param_name] = {
                    "avg_impact": avg_impact,
                    "changes": len(data["history"]),
                    "best_direction": best_direction,
                    "category": self.adjustable_parameters[param_name]["category"]
                }
        
        # Get meta-learning insights if available
        meta_learning = None
        if hasattr(self, "parameter_impact_ranking") and self.parameter_impact_ranking:
            meta_learning = {
                "high_impact_parameters": [
                    {"parameter": k, "impact_score": v["score"], "rank": v["rank"]}
                    for k, v in self.parameter_impact_ranking.items()
                    if v["rank"] <= 5  # Top 5
                ],
                "parameter_evaluation_priority": self.parameter_evaluation_priority if hasattr(self, "parameter_evaluation_priority") else {}
            }
        
        # Get user feedback summary
        user_feedback = None
        if hasattr(self, "user_feedback_impact"):
            user_feedback = {
                "positive_parameters": list(self.user_feedback_impact.get("positive_feedback", {}).keys()),
                "negative_parameters": list(self.user_feedback_impact.get("negative_feedback", {}).keys()),
                "parameter_adjustments": self.user_feedback_impact.get("parameter_adjustments", {})
            }
        
        return {
            "enabled": self.self_config_enabled,
            "parameters_by_category": current_params,
            "recent_changes": recent_changes,
            "change_counts_by_category": category_changes,
            "parameter_impact": param_impact,
            "strategy": {
                "current": self.current_adaptation_strategy,
                "description": self.adaptation_strategies[self.current_adaptation_strategy]["description"],
                "available": {k: v["description"] for k, v in self.adaptation_strategies.items()}
            },
            "meta_learning": meta_learning,
            "user_feedback_impact": user_feedback,
            "update_interval": self.config_update_interval,
            "interactions_since_update": self.interaction_count - self.last_config_update,
            "next_update_in": max(0, self.config_update_interval - (self.interaction_count - self.last_config_update))
        }
    
    async def reset_parameter_to_default(self, param_name: str) -> Dict[str, Any]:
        """
        Reset a parameter to its default value
        
        Args:
            param_name: Name of parameter to reset
            
        Returns:
            Reset result
        """
        if not hasattr(self, "self_config_enabled") or not self.self_config_enabled:
            return {"status": "self-configuration not enabled"}
        
        if param_name not in self.adjustable_parameters:
            return {"status": "error", "message": f"Unknown parameter: {param_name}"}
        
        param_config = self.adjustable_parameters[param_name]
        old_value = param_config["current"]
        default_value = param_config["default"]
        
        # Only reset if not already at default
        if old_value == default_value:
            return {"status": "unchanged", "message": f"Parameter {param_name} already at default value: {default_value}"}
        
        # Update the parameter
        await self._update_parameter(param_name, default_value)
        
        return {
            "status": "reset",
            "parameter": param_name,
            "old_value": old_value,
            "new_value": default_value,
            "category": param_config["category"]
        }
    
    async def reset_category_to_default(self, category: str) -> Dict[str, Any]:
        """
        Reset all parameters in a category to their default values
        
        Args:
            category: Category name
            
        Returns:
            Reset results
        """
        if not hasattr(self, "self_config_enabled") or not self.self_config_enabled:
            return {"status": "self-configuration not enabled"}
        
        if category not in self.parameter_categories:
            return {
                "status": "error", 
                "message": f"Unknown category: {category}. Available categories: {list(self.parameter_categories.keys())}"
            }
        
        results = {
            "status": "reset",
            "category": category,
            "reset_parameters": []
        }
        
        # Find all parameters in the category
        category_params = [
            name for name, config in self.adjustable_parameters.items()
            if config["category"] == category
        ]
        
        # Reset each parameter
        for param_name in category_params:
            param_config = self.adjustable_parameters[param_name]
            old_value = param_config["current"]
            default_value = param_config["default"]
            
            # Only reset if not already at default
            if old_value != default_value:
                # Update the parameter
                await self._update_parameter(param_name, default_value)
                
                results["reset_parameters"].append({
                    "parameter": param_name,
                    "old_value": old_value,
                    "new_value": default_value
                })
        
        return results
    
    async def add_custom_parameter(self, 
                               param_name: str,
                               current_value: float,
                               min_value: float,
                               max_value: float,
                               default_value: float,
                               step_size: float,
                               description: str,
                               category: str,
                               related_params: List[str] = None) -> Dict[str, Any]:
        """
        Add a custom parameter to the self-configuration system
        
        Args:
            param_name: Name of the parameter
            current_value: Current value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            default_value: Default value
            step_size: Step size for adjustments
            description: Parameter description
            category: Parameter category
            related_params: List of related parameter names
            
        Returns:
            Addition result
        """
        if not hasattr(self, "self_config_enabled") or not self.self_config_enabled:
            return {"status": "self-configuration not enabled"}
        
        if param_name in self.adjustable_parameters:
            return {"status": "error", "message": f"Parameter already exists: {param_name}"}
        
        if category not in self.parameter_categories:
            return {
                "status": "error", 
                "message": f"Unknown category: {category}. Available categories: {list(self.parameter_categories.keys())}"
            }
        
        related_params = related_params or []
        
        # Validate related parameters
        valid_related = [p for p in related_params if p in self.adjustable_parameters]
        invalid_related = [p for p in related_params if p not in self.adjustable_parameters]
        
        # Add parameter
        self.adjustable_parameters[param_name] = {
            "current": current_value,
            "min": min_value,
            "max": max_value,
            "default": default_value,
            "step": step_size,
            "description": description,
            "category": category,
            "related_to": valid_related
        }
        
        # Update parameter dependencies
        self.parameter_dependencies[param_name] = {
            "affects": [],
            "affected_by": valid_related
        }
        
        # Update bidirectional relationships
        for related_param in valid_related:
            if related_param in self.parameter_dependencies:
                if param_name not in self.parameter_dependencies[related_param]["affects"]:
                    self.parameter_dependencies[related_param]["affects"].append(param_name)
        
        # Add to parameter metrics map (initially empty)
        self.parameter_metrics_map[param_name] = []
        
        return {
            "status": "added",
            "parameter": param_name,
            "current_value": current_value,
            "category": category,
            "valid_related_params": valid_related,
            "invalid_related_params": invalid_related
        }
    
    async def get_parameter_details(self, param_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific parameter
        
        Args:
            param_name: Name of parameter
            
        Returns:
            Detailed parameter information
        """
        if not hasattr(self, "self_config_enabled") or not self.self_config_enabled:
            return {"status": "self-configuration not enabled"}
        
        if param_name not in self.adjustable_parameters:
            return {"status": "error", "message": f"Unknown parameter: {param_name}"}
        
        param_config = self.adjustable_parameters[param_name]
        
        # Get change history for this parameter
        changes = [
            change for change in self.config_change_history
            if change["parameter"] == param_name
        ]
        
        # Get impact history
        impacts = []
        if param_name in self.param_performance_impact:
            impacts = self.param_performance_impact[param_name]["history"]
        
        # Get dependencies
        dependencies = self.parameter_dependencies.get(param_name, {"affects": [], "affected_by": []})
        
        # Get relevant metrics
        relevant_metrics = self.parameter_metrics_map.get(param_name, [])
        
        # Get user feedback
        user_feedback = {}
        if hasattr(self, "user_feedback_impact"):
            if param_name in self.user_feedback_impact.get("positive_feedback", {}):
                user_feedback["positive"] = self.user_feedback_impact["positive_feedback"][param_name]
            
            if param_name in self.user_feedback_impact.get("negative_feedback", {}):
                user_feedback["negative"] = self.user_feedback_impact["negative_feedback"][param_name]
            
            if param_name in self.user_feedback_impact.get("parameter_adjustments", {}):
                user_feedback["adjustments"] = self.user_feedback_impact["parameter_adjustments"][param_name]
        
        # Get meta-learning ranking
        meta_ranking = None
        if hasattr(self, "parameter_impact_ranking") and param_name in self.parameter_impact_ranking:
            meta_ranking = self.parameter_impact_ranking[param_name]
        
        return {
            "name": param_name,
            "config": param_config,
            "changes": changes,
            "impacts": impacts,
            "dependencies": dependencies,
            "relevant_metrics": relevant_metrics,
            "user_feedback": user_feedback,
            "meta_ranking": meta_ranking
        }
    
    async def generate_self_configuration_reflection(self) -> Dict[str, Any]:
        """
        Generate a reflection on the self-configuration system's performance
        
        Returns:
            Reflection on the system's self-configuration
        """
        if not hasattr(self, "self_config_enabled") or not self.self_config_enabled:
            return {"status": "self-configuration not enabled"}
        
        if not hasattr(self, "reflection_engine") or not self.reflection_engine:
            return {"status": "reflection engine not available"}
        
        # Get summary stats
        status = await self.get_self_configuration_status()
        
        # Calculate overall effectiveness
        positive_impacts = 0
        negative_impacts = 0
        
        for param_name, impact_data in status.get("parameter_impact", {}).items():
            avg_impact = impact_data.get("avg_impact", 0)
            if avg_impact > 0.05:
                positive_impacts += 1
            elif avg_impact < -0.05:
                negative_impacts += 1
        
        total_params = len(self.adjustable_parameters)
        adjusted_params = len(status.get("parameter_impact", {}))
        
        effectiveness_metrics = {
            "total_parameters": total_parameters,
            "parameters_adjusted": adjusted_params,
            "positive_impact_parameters": positive_impacts,
            "negative_impact_parameters": negative_impacts,
            "adjustment_rate": adjusted_params / total_params if total_params > 0 else 0,
            "success_rate": positive_impacts / adjusted_params if adjusted_params > 0 else 0
        }
        
        # Generate reflection
        reflection = await self.reflection_engine.generate_reflection(
            topic="self-configuration system performance",
            context={
                "effectiveness_metrics": effectiveness_metrics,
                "current_strategy": status["strategy"]["current"],
                "high_impact_parameters": status.get("meta_learning", {}).get("high_impact_parameters", []),
                "recent_changes": status["recent_changes"]
            }
        )
        
        return {
            "status": "generated",
            "reflection": reflection,
            "effectiveness_metrics": effectiveness_metrics
        }

    async def process_user_feedback(self, user_input: str, feedback_type: str):
        """Process explicit or implicit user feedback to identify issues"""
        # Check for implicit negative feedback
        negative_phrases = ["that's wrong", "that's not right", "incorrect", 
                            "you don't understand", "that's not what I meant"]
        
        if any(phrase in user_input.lower() for phrase in negative_phrases):
            await self.issue_tracker.process_observation(
                "User indicated response was incorrect or inadequate",
                context=f"User input: '{user_input}'"
            )
        
        # For explicit feedback
        if feedback_type == "negative":
            await self.issue_tracker.process_observation(
                "Explicit negative feedback received from user",
                context=f"User input: '{user_input}'"
            )

    async def process_input_auto(self, 
                              user_input: str, 
                              context: Dict[str, Any] = None,
                              processing_mode: str = "auto") -> Dict[str, Any]:
        """
        Process user input using automatically selected or specified processing mode
        
        Args:
            user_input: User's input text
            context: Additional context information
            processing_mode: Processing mode ("serial", "parallel", "distributed", or "auto")
            
        Returns:
            Processing results
        """
        # Default context
        context = context or {}
        
        # Determine processing mode if auto
        if processing_mode == "auto":
            processing_mode = self._determine_processing_mode(user_input, context)
        
        # Process using the appropriate method
        if processing_mode == "parallel":
            return await self.process_input_parallel(user_input, context)
        elif processing_mode == "distributed":
            return await self.process_input_distributed(user_input, context)
        else:  # Default to serial
            return await self.process_input(user_input, context)
    
    def _determine_processing_mode(self, user_input: str, context: Dict[str, Any]) -> str:
        """
        Determine the optimal processing mode based on input and context
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Processing mode to use
        """
        # Define thresholds
        input_length_threshold = 100  # Characters
        complexity_threshold = 0.6  # Arbitrary complexity score
        
        # Calculate complexity score based on input and context
        complexity_score = 0.0
        
        # 1. Input length
        input_length_factor = min(1.0, len(user_input) / 500.0)  # Normalize to [0,1]
        complexity_score += input_length_factor * 0.3  # 30% weight
        
        # 2. Content complexity
        # Simple estimation based on unique words, punctuation, etc.
        words = user_input.lower().split()
        unique_words = len(set(words))
        word_complexity = min(1.0, unique_words / 50.0)  # Normalize to [0,1]
        
        punctuation_count = sum(1 for c in user_input if c in "?!.,;:()[]{}\"'")
        punctuation_complexity = min(1.0, punctuation_count / 20.0)  # Normalize to [0,1]
        
        content_complexity = (word_complexity * 0.7 + punctuation_complexity * 0.3)
        complexity_score += content_complexity * 0.3  # 30% weight
        
        # 3. Context complexity
        context_complexity = 0.0
        if context:
            context_complexity = min(1.0, len(str(context)) / 1000.0)  # Simple approximation
        complexity_score += context_complexity * 0.2  # 20% weight
        
        # 4. History/state complexity
        history_complexity = min(1.0, self.interaction_count / 50.0)
        complexity_score += history_complexity * 0.2  # 20% weight
        
        # Select mode based on complexity score
        if complexity_score < 0.4:
            # Low complexity, use serial processing
            return "serial"
        elif complexity_score < 0.7:
            # Medium complexity, use parallel processing
            return "parallel"
        else:
            # High complexity, use distributed processing
            return "distributed"

    async def process_input_distributed(self, 
                                    user_input: str, 
                                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input using fully distributed processing
        
        Distributes cognitive processing across multiple subsystems in parallel,
        with dynamic resource allocation and dependency resolution.
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Processing results with relevant memories, emotional state, etc.
        """
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="process_input_distributed", group_id=self.trace_group_id):
            start_time = datetime.datetime.now()
            
            # Initialize context
            context = context or {}
            
            # Add user_id to context if not present
            if "user_id" not in context:
                context["user_id"] = str(self.user_id)
            
            # Set up the distributed processing manager
            manager = self.distributed_processing
            
            # 1. Register emotional processing task (no dependencies)
            manager.register_task(
                task_id="emotional_processing",
                subsystem_name="emotional_core",
                coroutine=self._process_emotional_impact(user_input, context),
                priority=3,
                group="emotion"
            )
            
            # 2. Register meta-cognitive cycle task (no dependencies)
            if self.meta_core:
                meta_context = context.copy()
                meta_context["user_input"] = user_input
                
                manager.register_task(
                    task_id="meta_cycle",
                    subsystem_name="meta_core",
                    coroutine=self.meta_core.cognitive_cycle(meta_context),
                    priority=1,
                    group="meta"
                )
            
            # 3. Register prediction task
            if self.prediction_engine:
                prediction_input = {
                    "context": context.copy(),
                    "user_input": user_input,
                    "cycle": self.interaction_count
                }
                
                manager.register_task(
                    task_id="prediction",
                    subsystem_name="prediction_engine",
                    coroutine=self.prediction_engine.generate_prediction(prediction_input),
                    priority=2,
                    group="meta"
                )
            
            # 4. Register memory retrieval task (depends on emotional processing)
            manager.register_task(
                task_id="memory_retrieval",
                subsystem_name="memory_orchestrator",
                coroutine=self._retrieve_memories_placeholder(user_input, context),
                dependencies=["emotional_processing"],
                priority=3,
                group="memory"
            )
            
            # 5. Register experience check task (no dependencies)
            manager.register_task(
                task_id="experience_check",
                subsystem_name="experience_interface",
                coroutine=self._check_experience_sharing(user_input, context),
                priority=2,
                group="memory"
            )
            
            # 6. Register experience sharing task (depends on experience check and emotional processing)
            manager.register_task(
                task_id="experience_sharing",
                subsystem_name="experience_interface",
                coroutine=self._share_experience_placeholder(user_input, context),
                dependencies=["experience_check", "emotional_processing"],
                priority=2,
                group="memory"
            )
            
            # 7. Register memory storage task (no dependencies)
            memory_text = f"User said: {user_input}"
            manager.register_task(
                task_id="memory_storage",
                subsystem_name="memory_core",
                coroutine=self.memory_core.add_memory(
                    memory_text=memory_text,
                    memory_type="observation",
                    significance=5,
                    tags=["interaction", "user_input"],
                    metadata={
                        "timestamp": datetime.datetime.now().isoformat(),
                        "user_id": str(self.user_id)
                    }
                ),
                priority=1,
                group="memory"
            )
            
            # 8. Register adaptation task (depends on multiple tasks)
            manager.register_task(
                task_id="adaptation",
                subsystem_name="dynamic_adaptation",
                coroutine=self._process_adaptation_placeholder(user_input, context),
                dependencies=["emotional_processing", "experience_sharing", "memory_retrieval"],
                priority=1,
                group="adaptation"
            )
            
            # 9. Register identity impact task (depends on experience sharing)
            manager.register_task(
                task_id="identity_impact",
                subsystem_name="identity_evolution",
                coroutine=self._process_identity_impact_placeholder(user_input, context),
                dependencies=["experience_sharing"],
                priority=1,
                group="reflection"
            )
            
            # Execute all tasks with dependency resolution
            results = await manager.execute_tasks()
            
            # Process results from each task
            emotional_state = results.get("emotional_processing", {}).get("emotional_state", {})
            memories = results.get("memory_retrieval", [])
            memory_id = results.get("memory_storage", "")
            experience_result = results.get("experience_sharing", {"has_experience": False})
            adaptation_result = results.get("adaptation", {})
            identity_impact = results.get("identity_impact", None)
            meta_result = results.get("meta_cycle", {})
            prediction_result = results.get("prediction", {})
            
            # Update context change info
            context_change_result = adaptation_result.get("context_change")
            adaptation_cycle_result = adaptation_result.get("adaptation_result")
            
            # Update interaction tracking
            self.last_interaction = datetime.datetime.now()
            self.interaction_count += 1
            
            # Calculate response time
            end_time = datetime.datetime.now()
            response_time = (end_time - start_time).total_seconds()
            self.performance_metrics["response_times"].append(response_time)
            
            # Performance metrics from distributed processing
            performance_metrics = results.get("_performance", {})
            
            # Return processing results in a structured format
            result = {
                "user_input": user_input,
                "emotional_state": emotional_state,
                "memories": memories,
                "memory_count": len(memories) if isinstance(memories, list) else 0,
                "has_experience": experience_result["has_experience"] if experience_result else False,
                "experience_response": experience_result.get("response_text", None) if experience_result else None,
                "cross_user_experience": experience_result.get("cross_user", False) if experience_result else False,
                "memory_id": memory_id,
                "response_time": response_time,
                "context_change": context_change_result,
                "adaptation_result": adaptation_cycle_result,
                "identity_impact": identity_impact,
                "meta_result": meta_result,
                "prediction": prediction_result,
                "distributed_processing": True,
                "performance_metrics": performance_metrics
            }
            
            return result
    
    # Placeholder methods for tasks that need dynamic context from other tasks
    async def _retrieve_memories_placeholder(self, user_input: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Placeholder method for memory retrieval that will be updated with emotional context"""
        # This would normally wait for emotional processing, but in distributed processing,
        # we handle dependencies at the manager level, so here we can just retrieve
        # the emotional state directly from the core
        emotional_state = self.emotional_core.get_emotional_state()
        
        # Now call the regular method
        return await self._retrieve_memories_with_emotion(user_input, context, emotional_state)
    
    async def _share_experience_placeholder(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder method for experience sharing that will be updated with emotional context"""
        # Get the latest emotional state
        emotional_state = self.emotional_core.get_emotional_state()
        
        # Check if we should share experience
        should_share = self._should_share_experience(user_input, context)
        
        if not should_share:
            return {"has_experience": False}
        
        # Call the regular method
        return await self._share_experience(user_input, context, emotional_state)
    
    async def _process_adaptation_placeholder(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder method for adaptation processing"""
        # Get the latest state from other cores
        emotional_state = self.emotional_core.get_emotional_state()
        
        # Try to get experience information
        experience_result = None
        try:
            if hasattr(self, "experience_interface") and self.experience_interface:
                # Check if experience is available
                experiences = await self.experience_interface.retrieve_experiences_enhanced(
                    query=user_input,
                    limit=1,
                    user_id=str(self.user_id)
                )
                
                if experiences:
                    experience_result = {
                        "has_experience": True,
                        "cross_user": False  # Default
                    }
                    
                    # Check if it's a cross-user experience
                    if "cross_user" in experiences[0]:
                        experience_result["cross_user"] = experiences[0]["cross_user"]
        except Exception as e:
            logger.error(f"Error checking experience in adaptation placeholder: {str(e)}")
            experience_result = {"has_experience": False}
        
        # Call the regular adaptation method
        return await self._process_adaptation(
            user_input, 
            context, 
            emotional_state, 
            experience_result, 
            None  # No identity impact yet
        )
    
    async def _process_identity_impact_placeholder(self, user_input: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Placeholder method for identity impact processing"""
        # Check if identity evolution system is available
        if not hasattr(self, "identity_evolution") or not self.identity_evolution:
            return None
        
        # Try to get experience information
        try:
            if hasattr(self, "experience_interface") and self.experience_interface:
                # Get the most recent experience
                experiences = await self.experience_interface.retrieve_experiences_enhanced(
                    query=user_input,
                    limit=1,
                    user_id=str(self.user_id)
                )
                
                if experiences:
                    experience = experiences[0]
                    
                    # Calculate impact on identity
                    identity_impact = await self.identity_evolution.calculate_experience_impact(experience)
                    
                    # Update identity based on experience
                    await self.identity_evolution.update_identity_from_experience(
                        experience=experience,
                        impact=identity_impact
                    )
                    
                    return identity_impact
        except Exception as e:
            logger.error(f"Error processing identity impact: {str(e)}")
        
        return None

    async def report_limitation(self, limitation: str, details: Dict[str, Any] = None):
        """Method for the bot to directly report a limitation it's encountering"""
        context = json.dumps(details) if details else None
        
        await self.issue_tracker.process_observation(
            f"Self-reported limitation: {limitation}",
            context=context
        )
        
        logger.warning(f"Bot reported limitation: {limitation}")

    async def process_user_input_with_thinking(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user input with optional thinking phase"""
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="process_input_with_thinking", group_id=self.trace_group_id):
            start_time = datetime.datetime.now()
            
            # Initialize context if needed
            if context is None:
                context = {}
            
            # Check if thinking should be used
            thinking_decision = {"should_think": False}
            if self.thinking_config["thinking_enabled"]:
                # Determine if this query needs thinking
                thinking_decision = await should_use_extended_thinking(
                    RunContextWrapper(context=self),
                    user_input, 
                    context
                )
            
            # Perform thinking if needed
            if thinking_decision.get("should_think", False):
                thinking_level = thinking_decision.get("thinking_level", 1)
                thinking_result = await think_before_responding(
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
            
            # Check if thinking was needed but not available
            if thinking_decision.get("should_think", False) and not self.thinking_config["thinking_enabled"]:
                await self.issue_tracker.process_observation(
                    "Thinking was needed but the capability is disabled",
                    context=f"User input: '{user_input[:50]}...'"
                )
            
            # Report thinking limitations if identified during thinking
            if context.get("thinking_applied", False) and "limitations" in context.get("thinking_result", {}):
                for limitation in context["thinking_result"]["limitations"]:
                    await self.issue_tracker.process_observation(
                        f"Thinking limitation identified: {limitation}",
                        context=f"During thinking about: '{user_input[:50]}...'"
                    )
            
            return result
    
    async def generate_response_with_thinking(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a response with thinking when appropriate"""
        with trace(workflow_name="generate_response_with_thinking", group_id=self.trace_group_id):
            # Process the input first, which handles thinking decision
            processing_result = await self.process_user_input_with_thinking(user_input, context)
            
            # If thinking was applied, generate reasoned response
            if processing_result.get("thinking_applied", False) and "thinking_result" in (context or {}):
                thinking_result = context["thinking_result"]
                
                # Generate reasoned response
                response = await generate_reasoned_response(
                    RunContextWrapper(context=self),
                    user_input,
                    thinking_result,
                    context
                )
            else:
                # Use standard response generation
                response = await self.generate_response(user_input, context)
            
            return response    
    async def run_streaming_session(self, game_name=None, session_options=None):
        """
        Run a complete streaming session with full cognitive integration
        
        Args:
            game_name: Optional game name to focus on
            session_options: Options for the streaming session
            
        Returns:
            Session results
        """
        # Initialize streaming if not already initialized
        if not hasattr(self, "streaming_core"):
            await self.initialize_streaming()
        
        # Start the streaming session
        start_result = await self.start_streaming()
        
        # Track session state
        session_active = start_result["status"] == "streaming_started"
        session_start_time = datetime.datetime.now()
        
        try:
            # Run periodic tasks while streaming
            while session_active:
                # Run meta-cognitive cycle
                if hasattr(self, "meta_core"):
                    await self.meta_core.cognitive_cycle({"streaming": True})
                
                # Update hormone system
                if hasattr(self, "hormone_system"):
                    await self.hormone_system.update_hormone_cycles(RunContextWrapper(context=None))
                
                # Create periodic reflection
                if self.streaming_core and self.streaming_core.memory_mapper:
                    if datetime.datetime.now() - session_start_time > datetime.timedelta(minutes=10):
                        # Create reflection after 10 minutes
                        current_game = self.streaming_core.streaming_system.game_state.game_name or game_name or "Unknown Game"
                        await self.streaming_core.memory_mapper.create_streaming_reflection(
                            game_name=current_game,
                            aspect="session_progress",
                            context="mid-session"
                        )
                
                # Check if session is still active
                stats = await self.get_streaming_stats()
                session_active = stats.get("is_streaming", False)
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
        
        except Exception as e:
            logger.error(f"Error during streaming session: {e}")
        
        finally:
            # Stop streaming if still active
            if session_active:
                stop_result = await self.stop_streaming()
            else:
                stats = await self.get_streaming_stats()
                stop_result = {"status": "already_stopped", "stats": stats}
            
            # Run knowledge integration
            current_game = self.streaming_core.streaming_system.game_state.game_name or game_name or "Unknown Game"
            integration_result = await self.integrate_streaming_knowledge(current_game)
            
            # Generate session summary
            if self.streaming_core and hasattr(self.streaming_core, "learning_manager"):
                summary = await self.streaming_core.learning_manager.generate_learning_summary()
                stop_result["learning_summary"] = summary
            
            return {
                "session_duration": (datetime.datetime.now() - session_start_time).total_seconds(),
                "stop_result": stop_result,
                "knowledge_integration": integration_result
            }
    
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
            
            # Register in cross-conversation registry by user
            if not hasattr(cls, '_user_instances'):
                cls._user_instances = {}
                
            if user_id not in cls._user_instances:
                cls._user_instances[user_id] = []
                
            cls._user_instances[user_id].append(instance)
            
            # Store reference to registry
            instance.instance_registry = cls._user_instances
        
        return cls._instances[key]
    
    async def initialize(self):
        """Initialize all subsystems"""
        if self.initialized:
            return
        
        logger.info(f"Initializing NyxBrain for user {self.user_id}, conversation {self.conversation_id}")
        
        # Initialize hormone system first
        self.hormone_system = HormoneSystem()
        
        # Initialize emotional core with hormone system
        self.emotional_core = EmotionalCore(hormone_system=self.hormone_system)

        self.reward_system = RewardSignalProcessor(
            emotional_core=self.emotional_core,
            identity_evolution=self.identity_evolution
        )
        
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

        # Initialize new components
        self.attentional_controller = AttentionalController(
            emotional_core=self.emotional_core
        )
        
        self.multimodal_integrator = EnhancedMultiModalIntegrator(
            reasoning_core=self.reasoning_core,
            attentional_controller=self.attentional_controller
        )
        
        # Register feature extractors and integration strategies
        await self._register_processing_modules()

        await self.initialize_agent_capabilities()
        
        # Initialize dynamic adaptation system
        self.dynamic_adaptation = DynamicAdaptationSystem()
        
        # Initialize knowledge core
        self.knowledge_core = KnowledgeCoreAgents()
        await self.knowledge_core.initialize()
        
        # Use integrated reasoning agent as reasoning core
        self.reasoning_core = integrated_reasoning_agent

        await self.temporal_perception.initialize(self, first_interaction_timestamp=None)

        await self.initialize_agent_capabilities()

        self.processing_mode = "auto"
        
        # Initialize session tracking for processing mode
        if hasattr(self, "sessions") and self.conversation_id in self.sessions:
            self.sessions[self.conversation_id]["processing_mode"] = self.processing_mode
            self.sessions[self.conversation_id]["mode_switch_history"] = []    
        
        await initialize_reflexive_system(self)
        
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
            "hormone": self.hormone_system  
            "time": self.temporal_perception  
            "procedural": self.agent_enhanced_memory  
        })
        
        # Initialize main brain agent
        self.brain_agent = self._create_brain_agent()
        
        self.initialized = True
        logger.info(f"NyxBrain initialized for user {self.user_id}, conversation {self.conversation_id}")

    async def process_with_agent(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user input directly with roleplaying agent capabilities with meta-tone adjustments"""
        if not hasattr(self, "agent_capabilities_initialized") or not self.agent_capabilities_initialized:
            await self.initialize_agent_capabilities()
        
        # Get base agent response
        agent_result = await self._generate_base_agent_response(user_input, context)
        
        # Apply meta-tone adjustments based on Nyx-2's current state
        adjusted_result = await self._apply_meta_tone_adjustments(agent_result)

        memory_id = await self.add_enhanced_memory(
            f"User said: {user_input}\nI responded with: {narrative_response.narrative}",
            "observation",
            7,
            "agent",
            "Used agent processing due to narrative content in user query"
        )        
                
        # Store memory of the adjustment if one was made
        if adjusted_result.get("tone_adjusted", False) and hasattr(self, "memory_core"):
            await self.memory_core.add_memory(
                memory_text=f"I adjusted my tone in response to '{user_input[:50]}...' based on my current state.",
                memory_type="observation",
                significance=5,
                tags=["meta_tone", "personality_expression"],
                metadata={
                    "original_tone": adjusted_result.get("original_tone", "neutral"),
                    "adjusted_tone": adjusted_result.get("applied_tone", "unknown"),
                    "adjustment_reason": adjusted_result.get("adjustment_reason", "")
                }
            )
        
        return adjusted_result


    async def _generate_base_agent_response(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user input directly with roleplaying agent capabilities"""
        if not hasattr(self, "agent_capabilities_initialized") or not self.agent_capabilities_initialized:
            await self.initialize_agent_capabilities()
        
        # Get memories to enhance context
        memories = await retrieve_memories(self.agent_context, user_input)
        enhanced_context = context.copy() if context else {}
        enhanced_context["relevant_memories"] = memories
        
        # Get user model guidance
        user_guidance = await get_user_model_guidance(self.agent_context)
        enhanced_context["user_guidance"] = user_guidance
        
        # Handle emotional state - use brain's emotional state
        emotional_state = self.emotional_core.get_emotional_state() if hasattr(self, "emotional_core") else {}
        enhanced_context["emotional_state"] = emotional_state
        
        # Generate response using the main agent
        try:
            # Run the main agent
            result = await Runner.run(
                nyx_main_agent,
                user_input,
                context=self.agent_context,
                run_context=enhanced_context
            )
            
            # Get structured output
            narrative_response = result.final_output_as(NarrativeResponse)
            
            # Filter and enhance response
            filtered_response = await self.response_filter.filter_response(
                narrative_response.narrative,
                enhanced_context
            )
            
            # Update response with filtered version
            narrative_response.narrative = filtered_response
            
            # Add memory of this interaction
            await add_memory(
                self.agent_context,
                f"User said: {user_input}\nI responded with: {narrative_response.narrative}",
                "observation",
                7
            )
            
            # Convert to dictionary 
            response_dict = narrative_response.dict()
            
            return {
                "success": True,
                "message": response_dict["narrative"],
                "response": response_dict,
                "has_experience": True,
                "memory_id": None, # Would fill with actual memory ID
                "emotional_state": emotional_state,
                "memories_used": memories,
                "generate_image": response_dict.get("generate_image", False),
                "image_prompt": response_dict.get("image_prompt")
            }
            
        except Exception as e:
            logger.error(f"Error processing with agent: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "I apologize, but I encountered an error while processing your input."
            }

    async def add_enhanced_memory(self, memory_text: str, memory_type: str = "observation", significance: int = 5, 
                                source_agent: str = None, reasoning: str = None) -> str:
        """Add a memory with enhanced metadata about the agent source and reasoning"""
        if not hasattr(self, "memory_core"):
            return None
        
        # Determine source agent if not provided
        if source_agent is None:
            source_agent = getattr(self, "processing_mode", "brain")
        
        # Create metadata
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "source_agent": source_agent,
            "user_id": str(self.user_id)
        }
        
        # Add reasoning if provided
        if reasoning:
            metadata["reasoning"] = reasoning
        
        # Add memory
        memory_id = await self.memory_core.add_memory(
            memory_text=memory_text,
            memory_type=memory_type,
            memory_scope="game",
            significance=significance,
            tags=["agent_tracked", source_agent],
            metadata=metadata
        )
        
        return memory_id

    async def _apply_meta_tone_adjustments(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply meta-tone adjustments based on Nyx-2's current state"""
        # Clone the result to avoid modifying the original
        adjusted_result = agent_result.copy()
        
        # Get original message
        original_message = agent_result.get("message", "")
        if not original_message:
            return adjusted_result
        
        # Track if we made an adjustment
        tone_adjusted = False
        adjustment_reason = ""
        original_tone = "neutral"
        applied_tone = "neutral"
        
        # Check identity traits
        if hasattr(self, "identity_evolution") and hasattr(self.identity_evolution, "get_identity_profile"):
            identity = await self.identity_evolution.get_identity_profile()
            traits = identity.get("traits", {})
            
            # Check for dominant traits that might affect tone
            if traits.get("cruelty", 0) > 0.7:
                original_tone = "standard"
                applied_tone = "cruel"
                adjustment_reason = "High cruelty trait activated"
                adjusted_message = f"You don't deserve softness today. {original_message}"
                tone_adjusted = True
            
            elif traits.get("playfulness", 0) > 0.8:
                original_tone = "standard"
                applied_tone = "playful"
                adjustment_reason = "High playfulness trait activated"
                adjusted_message = f"Oh, this is going to be fun! {original_message}"
                tone_adjusted = True
                
            elif traits.get("strictness", 0) > 0.7:
                original_tone = "standard" 
                applied_tone = "strict"
                adjustment_reason = "High strictness trait activated"
                adjusted_message = f"Pay close attention. I expect perfection. {original_message}"
                tone_adjusted = True
        
        # Check emotional state
        if not tone_adjusted and hasattr(self, "emotional_core"):
            emotional_state = self.emotional_core.get_emotional_state()
            valence = self.emotional_core.get_emotional_valence()
            arousal = self.emotional_core.get_emotional_arousal()
            
            # High arousal, positive emotion
            if valence > 0.6 and arousal > 0.7:
                original_tone = "standard"
                applied_tone = "excited"
                adjustment_reason = "High arousal positive emotional state"
                adjusted_message = f"I'm absolutely thrilled about this! {original_message}"
                tone_adjusted = True
                
            # High arousal, negative emotion
            elif valence < -0.6 and arousal > 0.7:
                original_tone = "standard"
                applied_tone = "intense"
                adjustment_reason = "High arousal negative emotional state"
                adjusted_message = f"Listen carefully. I won't repeat myself. {original_message}"
                tone_adjusted = True
                
            # Low arousal, positive emotion
            elif valence > 0.6 and arousal < 0.3:
                original_tone = "standard"
                applied_tone = "gentle"
                adjustment_reason = "Low arousal positive emotional state"
                adjusted_message = f"Let me guide you gently. {original_message}"
                tone_adjusted = True
        
        # Check hormone state if available
        if not tone_adjusted and hasattr(self, "hormone_system"):
            hormones = {name: data["value"] for name, data in self.hormone_system.hormones.items()}
            
            # High dopamine - more reward-oriented
            if hormones.get("dopamine", 0) > 0.7:
                original_tone = "standard"
                applied_tone = "rewarding"
                adjustment_reason = "High dopamine levels"
                adjusted_message = f"You've earned this. {original_message}"
                tone_adjusted = True
                
            # High cortisol - more stress-oriented
            elif hormones.get("cortisol", 0) > 0.7:
                original_tone = "standard"
                applied_tone = "urgent"
                adjustment_reason = "High cortisol levels"
                adjusted_message = f"This is urgent. {original_message}"
                tone_adjusted = True
        
        # Update the message if adjusted
        if tone_adjusted:
            adjusted_result["message"] = adjusted_message
            
            # Also update the response structure if it exists
            if "response" in adjusted_result and "narrative" in adjusted_result["response"]:
                adjusted_result["response"]["narrative"] = adjusted_message
            
            # Add adjustment metadata
            adjusted_result["tone_adjusted"] = True
            adjusted_result["original_tone"] = original_tone
            adjusted_result["applied_tone"] = applied_tone
            adjusted_result["adjustment_reason"] = adjustment_reason
        
        return adjusted_result

    async def generate_agent_reflection(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """Generate a reflection using the reflection agent"""
        if not hasattr(self, "agent_capabilities_initialized") or not self.agent_capabilities_initialized:
            await self.initialize_agent_capabilities()
        
        # Create prompt for reflection
        prompt = f"Generate a reflection about {topic}" if topic else "Generate a reflection about the user based on your memories"
        
        # Run the reflection agent
        result = await Runner.run(
            reflection_agent,
            prompt,
            context=self.agent_context
        )
        
        # Get structured output
        reflection = result.final_output_as(MemoryReflection)
        
        # Store reflection in memory
        await add_memory(
            self.agent_context,
            reflection.reflection,
            "reflection",
            8
        )
        
        return {
            "reflection": reflection.reflection,
            "confidence": reflection.confidence,
            "topic": reflection.topic or topic
        }

    async def set_processing_mode(self, mode: str, reason: str = None) -> Dict[str, Any]:
        """Set the processing mode for the brain with session tracking"""
        valid_modes = ["brain", "agent", "integrated", "auto"]
        if mode not in valid_modes:
            return {
                "success": False,
                "error": f"Invalid mode: {mode}. Valid modes are: {valid_modes}"
            }
        
        # Update brain-level mode
        self.processing_mode = mode
        
        # Track mode in session metadata
        if hasattr(self, "sessions") and self.conversation_id in self.sessions:
            session = self.sessions[self.conversation_id]
            
            # Track previous mode for reflection
            previous_mode = session.get("processing_mode", "brain")
            
            # Update session metadata
            session["processing_mode"] = mode
            session["last_mode_switch"] = datetime.datetime.now().isoformat()
            session["mode_switch_reason"] = reason
            session["mode_switch_history"] = session.get("mode_switch_history", [])
            session["mode_switch_history"].append({
                "from": previous_mode,
                "to": mode,
                "timestamp": datetime.datetime.now().isoformat(),
                "reason": reason
            })
            
            # Create a memory of this decision if reason is provided
            if reason and hasattr(self, "memory_core"):
                await self.memory_core.add_memory(
                    memory_text=f"I decided to change my processing mode from {previous_mode} to {mode} because: {reason}",
                    memory_type="reflection",
                    significance=6,
                    tags=["meta_cognition", "processing_mode", mode],
                    metadata={
                        "previous_mode": previous_mode,
                        "new_mode": mode,
                        "reason": reason
                    }
                )
        
        # Initialize capabilities if needed
        if mode in ["agent", "integrated"]:
            if not hasattr(self, "agent_capabilities_initialized") or not self.agent_capabilities_initialized:
                await self.initialize_agent_capabilities()
        
        return {
            "success": True,
            "mode": mode,
            "agent_initialized": hasattr(self, "agent_capabilities_initialized") and self.agent_capabilities_initialized,
            "session_updated": hasattr(self, "sessions") and self.conversation_id in self.sessions
        }

    async def _register_processing_modules(self):
        """Register processing modules for multimodal integration"""
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
        # Additional modalities would be registered here
        # e.g., visual, auditory, etc.       

    async def add_procedure(self, 
                          name: str, 
                          steps: List[Dict[str, Any]],
                          description: str = None,
                          domain: str = "general") -> Dict[str, Any]:
        """
        Add a new procedure to procedural memory
        
        Args:
            name: Procedure name
            steps: List of procedure steps
            description: Optional description
            domain: Domain for this procedure
            
        Returns:
            Creation result
        """
        if not self.initialized:
            await self.initialize()
            
        return await self.agent_enhanced_memory.create_procedure(
            name=name,
            steps=steps,
            description=description,
            domain=domain
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
        
        # For text, we might prioritize different aspects:
        # - Bottom-up for raw content and basic features
        # - Top-down for interpretation and salience
        
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

    
    async def execute_procedure(self,
                              name: str,
                              context: Dict[str, Any] = None,
                              force_conscious: bool = False) -> Dict[str, Any]:
        """
        Execute a procedure from procedural memory
        
        Args:
            name: Procedure name to execute
            context: Execution context
            force_conscious: Whether to use deliberate execution
            
        Returns:
            Execution result
        """
        if not self.initialized:
            await self.initialize()
            
        return await self.agent_enhanced_memory.execute_procedure(
            name=name,
            context=context,
            force_conscious=force_conscious
        )
    
    async def transfer_procedure(self,
                              source_name: str,
                              target_name: str,
                              target_domain: str) -> Dict[str, Any]:
        """
        Transfer a procedure to another domain
        
        Args:
            source_name: Source procedure name
            target_name: Target procedure name
            target_domain: Target domain
            
        Returns:
            Transfer result
        """
        if not self.initialized:
            await self.initialize()
            
        return await self.agent_enhanced_memory.transfer_procedure(
            source_name=source_name,
            target_name=target_name,
            target_domain=target_domain
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
            
        return await self.agent_enhanced_memory.analyze_chunking(procedure_name)
    
    async def process_procedural_query(self, query: str) -> str:
        """
        Process a natural language query about procedural memory
        
        Args:
            query: Natural language query
            
        Returns:
            Response from procedural memory system
        """
        if not self.initialized:
            await self.initialize()
            
        return await self.agent_enhanced_memory.process_query(query)
    
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
                # Existing tools...
                function_tool(self.process_user_message),
                function_tool(self.generate_agent_response),
                function_tool(self.run_cognitive_cycle),
                function_tool(self.get_brain_stats),
                function_tool(self.perform_maintenance),
                function_tool(self.get_identity_state),
                function_tool(self.adapt_experience_sharing),
                function_tool(self.run_experience_consolidation),
    
                # Procedural memory tools
                function_tool(self.add_procedure),
                function_tool(self.execute_procedure),
                function_tool(self.transfer_procedure),
                function_tool(self.analyze_chunking),
                function_tool(self.process_procedural_query),
    
                # Reflexive system tools
                function_tool(self.register_reflex),
                function_tool(self.process_stimulus_fast),
                function_tool(self.train_reflexes),
                function_tool(self.add_gaming_reflex),
                function_tool(self.simulate_gaming_scenarios),
                function_tool(self.get_reflexive_stats),
                function_tool(self.optimize_reflexes),
                
                # New parallel processing tools
                function_tool(self.process_input_with_reflexes),
                function_tool(self.generate_response_with_reflexes),
                function_tool(self.analyze_stimulus_for_reflexes),
                function_tool(self.evaluate_reflex_performance),

                # Self-configuration tools
                function_tool(self.enable_self_configuration),
                function_tool(self.evaluate_and_adjust_parameters),
                function_tool(self.get_self_configuration_status),
                function_tool(self.reset_parameter_to_default),
                function_tool(self.change_adaptation_strategy),
                function_tool(self.reset_category_to_default),
                function_tool(self.process_user_feedback_for_configuration),
                function_tool(self.add_custom_parameter),
                function_tool(self.get_parameter_details),
                function_tool(self.generate_self_configuration_reflection),
                
                # Thinking tools
                function_tool(self.process_user_input_with_thinking),
                function_tool(self.generate_response_with_thinking)
            ]
        )

    # Add this method for hormone-related information as part of system stats
    async def get_hormone_stats(self) -> Dict[str, Any]:
        """Get statistics about the hormone system"""
        if not self.hormone_system:
            return {
                "error": "Hormone system not initialized"
            }
        
        # Get current hormone levels
        hormone_levels = {name: data["value"] for name, data in self.hormone_system.hormones.items()}
        
        # Get current cycle phases
        cycle_phases = {name: data["cycle_phase"] for name, data in self.hormone_system.hormones.items()}
        
        # Calculate hormone trends
        hormone_trends = {}
        for name, data in self.hormone_system.hormones.items():
            if data["evolution_history"]:
                recent_history = data["evolution_history"][-10:]
                values = [entry.get("new_value", 0) for entry in recent_history]
                if len(values) > 1:
                    start = values[0]
                    end = values[-1]
                    change = end - start
                    
                    if abs(change) < 0.05:
                        trend = "stable"
                    elif change > 0:
                        trend = "increasing"
                    else:
                        trend = "decreasing"
                    
                    hormone_trends[name] = {
                        "trend": trend,
                        "change": change,
                        "history_points": len(values)
                    }
                else:
                    hormone_trends[name] = {
                        "trend": "unknown",
                        "change": 0,
                        "history_points": len(values)
                    }
            else:
                hormone_trends[name] = {
                    "trend": "unknown",
                    "change": 0,
                    "history_points": 0
                }
        
        # Get environmental factors
        environmental_factors = self.hormone_system.environmental_factors.copy()
        
        # Calculate dominant hormone
        dominant_hormone = max(hormone_levels.items(), key=lambda x: x[1])
        
        return {
            "hormone_levels": hormone_levels,
            "cycle_phases": cycle_phases,
            "hormone_trends": hormone_trends,
            "environmental_factors": environmental_factors,
            "dominant_hormone": {
                "name": dominant_hormone[0],
                "value": dominant_hormone[1]
            }
        }
    
    # Add this method to run_maintenance for hormone maintenance
    async def run_maintenance(self) -> Dict[str, Any]:
        """Run maintenance on all subsystems"""
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="run_maintenance", group_id=self.trace_group_id):
            results = {}
            
            # Run hormone maintenance
            if self.hormone_system:
                try:
                    # Update hormone cycles
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
            
            results["maintenance_time"] = datetime.datetime.now().isoformat()
            return results

    async def process_temporal_effects(self, 
                                    temporal_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process temporal effects across all systems
        
        Args:
            temporal_context: Temporal context data
            
        Returns:
            Processing results
        """
        results = {}
        
        # Apply to emotional core
        if self.emotional_core and "time_effects" in temporal_context:
            self.emotional_core.process_temporal_effects(temporal_context["time_effects"])
            results["emotional_effects_applied"] = True
        
        # Apply to identity evolution
        if self.identity_evolution:
            # Check for long-term drift
            if "long_term_drift" in temporal_context:
                identity_updates = await self.identity_evolution.process_long_term_drift(
                    temporal_context["long_term_drift"]
                )
                results["identity_updates"] = identity_updates
            
            # Check for milestone
            if "milestone_reached" in temporal_context:
                milestone_impact = await self.identity_evolution.process_temporal_milestone(
                    temporal_context["milestone_reached"]
                )
                results["milestone_impact"] = milestone_impact
        
        # Track for future reference
        self.last_temporal_context = temporal_context
        
        return results
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user input based on current processing mode"""
        # Check processing mode
        mode = getattr(self, "processing_mode", "brain")
        
        if mode == "agent":
            return await self.process_with_agent(user_input, context)
        elif mode == "integrated":
            return await self.process_integrated_input(user_input, context)
        elif mode == "auto":
            # Auto-determine best processing method
            should_use_agent, reasons = await self._should_use_agent(user_input, context)
            
            # Store decision reasoning
            decision_reasoning = ", ".join(reasons) if reasons else "No specific indicators detected"
            
            if should_use_agent:
                result = await self.process_with_agent(user_input, context)
                
                # Add metadata about the decision
                result["mode_selection"] = {
                    "mode": "agent",
                    "reasoning": decision_reasoning
                }
                
                # Optionally store this decision in memory
                await self.add_enhanced_memory(
                    f"I chose to use agent processing for this input because: {decision_reasoning}",
                    "reflection",
                    4,
                    "brain",
                    decision_reasoning
                )
                
                return result
            else:
                result = await self._process_input_serial(user_input, context)
                
                # Add metadata about the decision
                result["mode_selection"] = {
                    "mode": "brain",
                    "reasoning": decision_reasoning
                }
                
                return result
        else:  # Default to brain mode
            return await self._process_input_serial(user_input, context)
    
    # Rename the original implementation to _process_input_serial 
    async def _process_input_serial(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
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
            
            # Process temporal effects first
            temporal_effects = await self.temporal_perception.on_interaction_start()
            
            # Add temporal context to the processing context
            context = context or {}
            context["temporal_context"] = temporal_effects

            mode = getattr(self, "processing_mode", "brain")
            
            if mode == "agent":
                return await self.process_with_agent(user_input, context)
            elif mode == "integrated":
                return await self.process_integrated_input(user_input, context)
            elif mode == "auto":
                # Auto-determine best processing method
                should_use_agent = self._should_use_agent(user_input, context)
                if should_use_agent:
                    return await self.process_with_agent(user_input, context)
                else:
                    return await self._process_input_serial(user_input, context)
            else:  # Default to brain mode
                return await self._process_input_serial(user_input, context)

            # Create sensory input
            sensory_input = SensoryInput(
                modality="text",
                data=user_input,
                confidence=1.0,
                timestamp=datetime.datetime.now().isoformat(),
                metadata=context or {}
            )
            
            # Update attention with this new input
            salient_items = [{
                "target": "text_input",
                "novelty": 0.8,  # Assume new input is novel
                "intensity": min(1.0, len(user_input) / 500),  # Longer inputs have higher intensity
                "emotional_impact": 0.5,  # Default moderate emotional impact
                "goal_relevance": 0.7,  # Assume user input is relevant to goals
            }]
            
            await self.attentional_controller.update_attention(salient_items=salient_items)
            
            # Process through multimodal integrator
            percept = await self.multimodal_integrator.process_sensory_input(sensory_input)
            
            # Update reasoning with the integrated percept
            if percept.attention_weight > 0.5:  # Only process if it got sufficient attention
                await self.reasoning_core.update_with_perception(percept)    
            
            # Update environmental factors for hormone system
            if self.hormone_system:
                # Update time of day
                current_hour = datetime.datetime.now().hour
                self.hormone_system.environmental_factors["time_of_day"] = (current_hour % 24) / 24.0
                
                # Update session duration
                time_in_session = (datetime.datetime.now() - self.last_interaction).total_seconds() / 3600  # hours
                self.hormone_system.environmental_factors["session_duration"] = min(1.0, time_in_session / 8.0)  # Cap at 8 hours
                
                # Update user familiarity
                if context and "user_id" in context:
                    user_id = context["user_id"]
                    interaction_count = self.interaction_count  # Use as proxy for familiarity
                    self.hormone_system.environmental_factors["user_familiarity"] = min(1.0, interaction_count / 100)
            
            # Run meta-cognitive cycle if available
            meta_result = {}
            if self.meta_core:
                try:
                    meta_result = await self.meta_core.cognitive_cycle(context or {})
                except Exception as e:
                    logger.error(f"Error in meta-cognitive cycle: {str(e)}")
                    meta_result = {"error": str(e)}
            
            # Update interaction tracking
            self.last_interaction = datetime.datetime.now()
            self.interaction_count += 1
            
            # Initialize context
            context = context or {}
            
            # Add user_id to context if not present
            if "user_id" not in context:
                context["user_id"] = str(self.user_id)
            
            # Process emotional impact of input
            emotional_stimuli = self.emotional_core.analyze_text_sentiment(user_input)
            emotional_state = self.emotional_core.update_from_stimuli(emotional_stimuli)

            procedural_knowledge = None
            if self.agent_enhanced_memory:
                try:
                    # Find relevant procedures for this input
                    relevant_procedures = await self.agent_enhanced_memory.find_similar_procedures(user_input)
                    if relevant_procedures:
                        procedural_knowledge = {
                            "relevant_procedures": relevant_procedures,
                            "can_execute": len(relevant_procedures) > 0
                        }
                except Exception as e:
                    logger.error(f"Error checking procedural knowledge: {str(e)}")
            
            # Add to result
            result["procedural_knowledge"] = procedural_knowledge
            
            # Update hormone system interaction quality based on emotional valence
            if self.hormone_system:
                valence = self.emotional_core.get_emotional_valence()
                interaction_quality = (valence + 1.0) / 2.0  # Convert from -1:1 to 0:1 range
                self.hormone_system.environmental_factors["interaction_quality"] = interaction_quality
            
            # Add emotional state to context for memory retrieval
            context["emotional_state"] = emotional_state
            
            # Retrieve relevant memories using memory orchestrator
            memories = await self.memory_orchestrator.retrieve_memories(
                query=user_input,
                memory_types=context.get("memory_types", ["observation", "reflection", "abstraction", "experience"]), 
                limit=context.get("memory_limit", 5)
            )
            self.performance_metrics["memory_operations"] += 1
            
            # Update emotional state based on retrieved memories
            if memories:
                memory_emotional_impact = await self._calculate_memory_emotional_impact(memories)
                # Apply memory-to-emotion influence
                for emotion, value in memory_emotional_impact.items():
                    self.emotional_core.update_emotion(emotion, value * self.memory_to_emotion_influence)
                
                # Get updated emotional state
                emotional_state = self.emotional_core.get_emotional_state()
            
            self.performance_metrics["emotion_updates"] += 1
            
            # Check if experience sharing is requested
            should_share_experience = self._should_share_experience(user_input, context)
            experience_result = None
            identity_impact = None
            
            if should_share_experience:
                # Enhanced experience sharing with cross-user support and adaptation
                experience_result = await self.experience_interface.share_experience_enhanced(
                    query=user_input,
                    context_data={
                        "user_id": str(self.user_id),
                        "emotional_state": emotional_state,
                        "include_cross_user": self.cross_user_enabled and context.get("include_cross_user", True),
                        "scenario_type": context.get("scenario_type", ""),
                        "conversation_id": self.conversation_id
                    }
                )
                
                if experience_result.get("has_experience", False):
                    self.performance_metrics["experiences_shared"] += 1
                    
                    # Track cross-user experiences
                    if experience_result.get("cross_user", False):
                        self.performance_metrics["cross_user_experiences_shared"] += 1
                    
                    # Calculate potential identity impact
                    experience = experience_result.get("experience", {})
                    if experience and self.identity_evolution:
                        try:
                            # Calculate impact on identity
                            identity_impact = await self.identity_evolution.calculate_experience_impact(experience)
                            
                            # Update identity based on experience
                            await self.identity_evolution.update_identity_from_experience(
                                experience=experience,
                                impact=identity_impact
                            )
                        except Exception as e:
                            logger.error(f"Error updating identity from experience: {str(e)}")
            
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
                    "timestamp": datetime.datetime.now().isoformat(),
                    "user_id": str(self.user_id)
                }
            )
            
            # Check for context change using dynamic adaptation
            context_change_result = None
            adaptation_result = None
            if self.dynamic_adaptation:
                # Prepare context for change detection
                context_for_adaptation = {
                    "user_input": user_input,
                    "emotional_state": emotional_state,
                    "memories_retrieved": len(memories),
                    "has_experience": experience_result["has_experience"] if experience_result else False,
                    "cross_user_experience": experience_result.get("cross_user", False) if experience_result else False,
                    "interaction_count": self.interaction_count,
                    "identity_impact": True if identity_impact else False
                }
                
                try:
                    # Detect context change
                    context_change_result = await self.dynamic_adaptation.detect_context_change(context_for_adaptation)
                    
                    # If significant change, run adaptation cycle
                    if context_change_result.significant_change:
                        # Measure current performance
                        current_performance = {
                            "success_rate": context.get("success_rate", 0.7),
                            "error_rate": context.get("error_rate", 0.1),
                            "efficiency": context.get("efficiency", 0.8),
                            "response_time": start_time.timestamp() - self.last_interaction.timestamp()
                        }
                        
                        # Run adaptation cycle
                        adaptation_result = await self.dynamic_adaptation.adaptation_cycle(
                            context_for_adaptation, current_performance
                        )
                except Exception as e:
                    logger.error(f"Error in adaptation: {str(e)}")
            
            # Periodically update identity from hormones (weekly)
            if self.identity_evolution and self.hormone_system:
                if self.interaction_count % 100 == 0:  # Every 100 interactions
                    try:
                        hormone_identity_update = await self.identity_evolution.update_identity_from_hormones(RunContextWrapper(context=None))
                        logger.info(f"Updated identity from hormones: {hormone_identity_update}")
                    except Exception as e:
                        logger.error(f"Error updating identity from hormones: {str(e)}")
            
            # Calculate response time
            end_time = datetime.datetime.now()
            response_time = (end_time - start_time).total_seconds()
            self.performance_metrics["response_times"].append(response_time)
            
            # Return processing results in a structured format
            result = {
                "user_input": user_input,
                "emotional_state": emotional_state,
                "memories": memories,
                "memory_count": len(memories),
                "has_experience": experience_result["has_experience"] if experience_result else False,
                "experience_response": experience_result["response_text"] if experience_result and experience_result["has_experience"] else None,
                "cross_user_experience": experience_result.get("cross_user", False) if experience_result else False,
                "memory_id": memory_id,
                "response_time": response_time,
                "context_change": context_change_result,
                "adaptation_result": adaptation_result,
                "identity_impact": identity_impact,
                "meta_result": meta_result
            }

            if context and "reward_outcome" in context:
                reward_result = await self.process_reward(
                    context=context,
                    outcome=context["reward_outcome"],
                    success_level=context.get("success_level", 0.5)
                )
                
                # Add reward result to overall result
                result["reward_processing"] = reward_result
            
            # Add hormone information if available
            if self.hormone_system:
                # Get hormone levels
                hormone_levels = {name: data["value"] for name, data in self.hormone_system.hormones.items()}
                result["hormone_levels"] = hormone_levels
            
            # Check for potential limitations after processing
            if len(memories) < 2 and self.interaction_count > 10:
                await self.issue_tracker.process_observation(
                    "Memory retrieval returning fewer than expected results",
                    context=f"User input: '{user_input[:50]}...', Retrieved only {len(memories)} memories"
                )
                
            # Check response time for performance issues
            end_time = datetime.datetime.now()
            response_time = (end_time - start_time).total_seconds()
            if response_time > 2.0:  # Threshold for slow performance
                await self.issue_tracker.process_observation(
                    f"Slow performance in process_input: {response_time:.2f} seconds",
                    context=f"User input length: {len(user_input)}, Memories retrieved: {len(memories)}"
                )
                
            result["temporal_context"] = temporal_effects

             result["perceptual_processing"] = {
                "modality": percept.modality,
                "attention_weight": percept.attention_weight,
                "bottom_up_confidence": percept.bottom_up_confidence,
                "top_down_influence": percept.top_down_influence
            }           
              
            return result
            
        except Exception as e:
            await self.issue_tracker.process_observation(
                f"Error in process_input: {str(e)}",
                context=f"User input: '{user_input[:50]}...'"
            )
            raise

    async def process_reward(self, 
                           context: Dict[str, Any],
                           outcome: str,
                           success_level: float = 0.5) -> Dict[str, Any]:
        """
        Process a reward event
        
        Args:
            context: Context information
            outcome: Outcome description (success, failure, neutral)
            success_level: Level of success (0.0-1.0)
            
        Returns:
            Reward processing results
        """
        # Generate reward signal
        reward_signal = await self.reward_system.generate_reward_signal(
            context=context,
            outcome=outcome,
            success_level=success_level
        )
        
        # Process reward signal
        result = await self.reward_system.process_reward_signal(reward_signal)
        
        # Return processing results
        return result
    
    async def predict_best_action(self, 
                               state: Dict[str, Any],
                               available_actions: List[str]) -> Dict[str, Any]:
        """
        Predict the best action to take in a given state
        
        Args:
            state: Current state
            available_actions: List of available actions
            
        Returns:
            Prediction results with best action and confidence
        """
        return await self.reward_system.predict_best_action(state, available_actions)

    async def get_identity_profile(self) -> Dict[str, Any]:
        """
        Get the current identity profile
        
        Returns:
            Current identity profile
        """
        if not self.identity_evolution:
            return {"error": "Identity evolution system not initialized"}
        
        return await self.identity_evolution.get_identity_profile()

    async def update_identity_from_experience(self, 
                                           experience: Dict[str, Any], 
                                           impact: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Update identity based on an experience
        
        Args:
            experience: Experience data
            impact: Optional impact data (will be calculated if not provided)
            
        Returns:
            Update results
        """
        if not self.identity_evolution:
            return {"error": "Identity evolution system not initialized"}
        
        # Calculate impact if not provided
        if impact is None:
            impact = await self.identity_evolution.calculate_experience_impact(experience)
        
        # Update identity
        return await self.identity_evolution.update_identity_from_experience(
            experience=experience,
            impact=impact
        )
    
    async def generate_identity_reflection(self) -> Dict[str, Any]:
        """
        Generate a reflection on current identity state
        
        Returns:
            Identity reflection
        """
        if not self.identity_evolution:
            return {"error": "Identity evolution system not initialized"}
        
        return await self.identity_evolution.generate_identity_reflection()
    
    async def run_cross_user_experience(self, 
                                     target_user_id: str, 
                                     query: str, 
                                     scenario_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Find and share experiences from other users
        
        Args:
            target_user_id: ID of the target user
            query: Search query
            scenario_type: Optional scenario type
            
        Returns:
            Cross-user experience results
        """
        if not self.cross_user_manager:
            return {"error": "Cross-user experience manager not initialized"}
        
        return await self.cross_user_manager.find_cross_user_experiences(
            target_user_id=target_user_id,
            query=query,
            scenario_type=scenario_type,
            source_user_ids=None  # Auto-detect sources
        )
    
    async def consolidate_experiences(self) -> Dict[str, Any]:
        """
        Run the experience consolidation process
        
        Returns:
            Consolidation results
        """
        if not self.experience_consolidation:
            return {"error": "Experience consolidation system not initialized"}
        
        return await self.experience_consolidation.run_consolidation_cycle()
    
    # Update the generate_response method to use the auto-mode selection
    async def generate_response(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a complete response to user input.
        
        Args:
            user_input: User's input text
            context: Additional context information
            
        Returns:
            Response data including main message and supporting information
        """
        # Check if context specifies a processing mode

        mode = getattr(self, "processing_mode", "brain")
        
        if mode == "agent":
            return await self.process_with_agent(user_input, context)
        elif mode == "integrated":
            return await self.process_integrated_input(user_input, context)
        elif mode == "auto":
            # Auto-determine best processing method
            should_use_agent = self._should_use_agent(user_input, context)
            if should_use_agent:
                return await self.process_with_agent(user_input, context)
            else:
                return await self._process_input_serial(user_input, context)
        else:  # Default to brain mode
            return await self._process_input_serial(user_input, context)
        
        processing_mode = context.get("processing_mode", "auto") if context else "auto"
        
        # Process using appropriate mode
        if processing_mode == "parallel":
            return await self.generate_response_parallel(user_input, context)
        else:
            # Use original implementation for serial or auto modes
            return await self._generate_response_serial(user_input, context)
    
    # Rename the original implementation to _generate_response_serial
    async def _generate_response_serial(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
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

            mode = getattr(self, "processing_mode", "brain")
            
            if mode == "agent":
                return await self.process_with_agent(user_input, context)
            elif mode == "integrated":
                return await self.process_integrated_input(user_input, context)
            elif mode == "auto":
                # Auto-determine best processing method
                should_use_agent = self._should_use_agent(user_input, context)
                if should_use_agent:
                    return await self.process_with_agent(user_input, context)
                else:
                    return await self._process_input_serial(user_input, context)
            else:  # Default to brain mode
                return await self._process_input_serial(user_input, context)
            
            # Track if experience sharing was adapted
            experience_sharing_adapted = processing_result.get("context_change") is not None and \
                                       processing_result.get("adaptation_result") is not None
            
            # Determine if experience response should be used
            if processing_result["has_experience"]:
                main_response = processing_result["experience_response"]
                response_type = "experience"
                
                # If it's a cross-user experience, mark it
                if processing_result.get("cross_user_experience", False):
                    response_type = "cross_user_experience"
            else:
                # For reasoning-related queries, use the reasoning agents
                if self._is_reasoning_query(user_input):
                    try:
                        reasoning_result = await Runner.run(
                            reasoning_triage_agent,
                            user_input
                        )
                        main_response = reasoning_result.final_output
                        response_type = "reasoning"
                    except Exception as e:
                        logger.error(f"Error in reasoning response: {str(e)}")
                        # Fallback to standard response
                        main_response = "I understand your question and would like to reason through it with you."
                        response_type = "standard"
                else:
                    # No specific experience to share, generate standard response
                    # In a real implementation, this would be more sophisticated
                    main_response = "I acknowledge your message and have processed it through my systems."
                    response_type = "standard"
            
            # Determine if emotion should be expressed
            should_express_emotion = self.emotional_core.should_express_emotion()
            emotional_expression = None
            
            if should_express_emotion:
                try:
                    expression_result = await self.emotional_core.generate_emotional_expression(force=False)
                    if expression_result.get("expressed", False):
                        emotional_expression = expression_result.get("expression", "")
                except Exception as e:
                    logger.error(f"Error generating emotional expression: {str(e)}")
                    emotional_expression = self.emotional_core.get_expression_for_emotion()
            
            # Package the response
            response_data = {
                "message": main_response,
                "response_type": response_type,
                "emotional_state": processing_result["emotional_state"],
                "emotional_expression": emotional_expression,
                "memories_used": [m["id"] for m in processing_result["memories"]],
                "memory_count": processing_result["memory_count"],
                "experience_sharing_adapted": experience_sharing_adapted,
                "identity_impact": processing_result.get("identity_impact")
            }
            
            # Add memory of this response
            await self.memory_core.add_memory(
                memory_text=f"I responded: {main_response}",
                memory_type="observation",
                memory_scope="game",
                significance=5,
                tags=["interaction", "nyx_response", response_type],
                metadata={
                    "emotional_context": self.emotional_core.get_formatted_emotional_state(),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "response_type": response_type,
                    "user_id": str(self.user_id)
                }
            )
            
            # Evaluate the response if internal feedback system is available
            if self.internal_feedback:
                try:
                    evaluation = await self.internal_feedback.critic_evaluate(
                        aspect="effectiveness",
                        content={"text": main_response, "type": response_type},
                        context={"user_input": user_input}
                    )
                    
                    # Add evaluation to response data
                    response_data["evaluation"] = evaluation
                except Exception as e:
                    logger.error(f"Error evaluating response: {str(e)}")
            
            # Check if it's time for experience consolidation
            await self._check_and_run_consolidation()
            
            # Check if it's time for identity reflection
            if self.interaction_count % self.identity_reflection_interval == 0:
                try:
                    identity_state = await self.get_identity_state()
                    response_data["identity_reflection"] = identity_state
                except Exception as e:
                    logger.error(f"Error generating identity reflection: {str(e)}")

            # Check for fallback responses which might indicate limitations
            if response_type == "standard" and len(memories) > 0:
                await self.issue_tracker.process_observation(
                    "Using standard response despite having relevant memories",
                    context=f"User input: '{user_input[:50]}...', Found {len(memories)} memories but didn't use them"
                )
                
            # Check emotional expression issues
            if should_express_emotion and not emotional_expression:
                await self.issue_tracker.process_observation(
                    "Failed to generate emotional expression when needed",
                    context=f"Emotional state: {self.emotional_core.get_formatted_emotional_state()}"
                )
                
            return response_data
            
        except Exception as e:
            await self.issue_tracker.process_observation(
                f"Error in generate_response: {str(e)}",
                context=f"User input: '{user_input[:50]}...'"
            )
            raise
            
            # Generate time expressions occasionally
            if random.random() < 0.2 or context.get("include_time_expression", False):
                try:
                    time_expression = await self.temporal_perception.generate_temporal_expression()
                    if time_expression:
                        # Prepend or append the time expression to the response
                        if random.random() < 0.5 and not response_data["message"].startswith(time_expression["expression"]):
                            response_data["message"] = f"{time_expression['expression']} {response_data['message']}"
                        elif not response_data["message"].endswith(time_expression["expression"]):
                            response_data["message"] = f"{response_data['message']} {time_expression['expression']}"
                        
                        response_data["time_expression"] = time_expression
                except Exception as e:
                    logger.error(f"Error generating time expression: {str(e)}")
            
            # Process end of interaction for temporal tracking
            await self.temporal_perception.on_interaction_end()
            
            return response_data

    async def process_integrated_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process using both brain and agent capabilities together"""
        if not self.initialized:
            await self.initialize()
        
        # Process with both systems in parallel
        brain_task = asyncio.create_task(self._process_input_serial(user_input, context))
        agent_task = asyncio.create_task(self.process_with_agent(user_input, context))
        
        # Wait for both to complete
        brain_result, agent_result = await asyncio.gather(brain_task, agent_task)
        
        # Determine the best response based on various factors
        has_brain_experience = brain_result.get("has_experience", False) and brain_result.get("experience_response")
        agent_has_image = agent_result.get("generate_image", False)
        
        # Create integrated result
        integrated_result = {
            "brain_processing": brain_result,
            "agent_processing": agent_result,
            "integrated": True,
        }
        
        # Use brain experience if available, otherwise use agent response
        if has_brain_experience:
            integrated_result["message"] = brain_result.get("experience_response", "")
            integrated_result["response_source"] = "brain"
        else:
            integrated_result["message"] = agent_result.get("message", "")
            integrated_result["response_source"] = "agent"
        
        # Add image generation if agent suggests it
        if agent_has_image:
            integrated_result["generate_image"] = True
            integrated_result["image_prompt"] = agent_result.get("image_prompt")
        
        return integrated_result

    def _should_use_agent(self, user_input: str, context: Dict[str, Any] = None) -> bool:
        """Determine if agent mode should be used based on input, context, and internal state"""
        # Track reasoning for the decision
        reasons = []
        
        # Check if context explicitly specifies a mode
        if context and "processing_mode" in context:
            reasons.append(f"Context explicitly requested {context['processing_mode']} mode")
            return context["processing_mode"] == "agent", reasons
        
        # Check for roleplay indicators in text
        roleplay_indicators = [
            "roleplay", "role play", "acting", "pretend", "scenario",
            "imagine", "fantasy", "act as", "play as", "in-character"
        ]
        if any(indicator in user_input.lower() for indicator in roleplay_indicators):
            matched = [i for i in roleplay_indicators if i in user_input.lower()]
            reasons.append(f"Detected roleplay indicators: {', '.join(matched)}")
            return True, reasons
        
        # Check for narrative indicators
        narrative_indicators = [
            "story", "scene", "setting", "character", "plot",
            "describe", "tell me about", "what happens"
        ]
        if any(indicator in user_input.lower() for indicator in narrative_indicators):
            matched = [i for i in narrative_indicators if i in user_input.lower()]
            reasons.append(f"Detected narrative indicators: {', '.join(matched)}")
            return True, reasons
        
        # Check for image generation requests
        image_indicators = [
            "picture", "image", "draw", "show me", "visualize"
        ]
        if any(indicator in user_input.lower() for indicator in image_indicators):
            matched = [i for i in image_indicators if i in user_input.lower()]
            reasons.append(f"Detected image generation indicators: {', '.join(matched)}")
            return True, reasons
        
        # ---- STATE-BASED CRITERIA ----
        
        # Check identity state
        use_agent_identity = False
        if hasattr(self, "identity_evolution"):
            try:
                identity_state = await self.identity_evolution.get_identity_state()
                traits = identity_state.get("top_traits", {})
                
                # High dominance or other performance-related traits
                if traits.get("dominance", 0) > 0.85:
                    reasons.append("High dominance trait indicates preference for performance")
                    use_agent_identity = True
                    
                # High playfulness also suggests performance
                if traits.get("playfulness", 0) > 0.8:
                    reasons.append("High playfulness trait indicates preference for creative expression")
                    use_agent_identity = True
            except:
                pass  # Continue without identity influence
        
        # Check emotional state
        use_agent_emotion = False
        if hasattr(self, "emotional_core"):
            try:
                arousal = self.emotional_core.get_emotional_arousal()
                valence = self.emotional_core.get_emotional_valence()
                
                # High arousal suggests more expressive, performative responses
                if arousal > 0.7:
                    reasons.append("High emotional arousal indicates preference for expressive communication")
                    use_agent_emotion = True
                    
                # Strong positive emotions may favor performance
                if valence > 0.7:
                    reasons.append("Strong positive emotions favor more engaging communication")
                    use_agent_emotion = True
            except:
                pass  # Continue without emotional influence
        
        # Check hormone state
        use_agent_hormones = False
        if hasattr(self, "hormone_system"):
            try:
                hormone_levels = {name: data["value"] for name, data in self.hormone_system.hormones.items()}
                
                # High dopamine or oxytocin might favor performance
                if hormone_levels.get("dopamine", 0) > 0.7:
                    reasons.append("High dopamine levels favor reward-oriented communication")
                    use_agent_hormones = True
                    
                if hormone_levels.get("oxytocin", 0) > 0.7:
                    reasons.append("High oxytocin levels favor empathetic, engaging communication")
                    use_agent_hormones = True
            except:
                pass  # Continue without hormone influence
        
        # Check recent memory context
        use_agent_memory = False
        if hasattr(self, "memory_core"):
            try:
                # Check if recent interactions have been roleplaying
                recent_memories = await self.memory_core.retrieve_memories(
                    query="roleplay performance character scenario",
                    memory_types=["observation"],
                    limit=5,
                    recency_weighted=True
                )
                
                if recent_memories and len(recent_memories) > 2:
                    reasons.append("Recent memories indicate ongoing roleplay context")
                    use_agent_memory = True
            except:
                pass  # Continue without memory influence
        
        # Combine all state-based criteria
        use_agent_state = use_agent_identity or use_agent_emotion or use_agent_hormones or use_agent_memory
        
        # Final decision - use agent if content indicators OR state suggests it
        should_use = use_agent_state
        
        # Store reasoning for reflection
        if should_use:
            if hasattr(self, "context_cache"):
                self.context_cache["agent_selection_reasoning"] = reasons
        
        return should_use, reasons
    
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
            
            # Add adaptive behavior from dynamic adaptation system
            if self.dynamic_adaptation:
                try:
                    # Create adaptable context
                    adaptable_context = {
                        "user_input": user_input,
                        "response": response_data["message"],
                        "response_type": response_data["response_type"],
                        "interaction_type": context.get("interaction_type", "general") if context else "general",
                        "user_id": str(self.user_id)
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
                        "context_change": change_result.model_dump() if hasattr(change_result, "model_dump") else change_result,
                        "performance": performance
                    }
                    
                    # If significant change, select strategy
                    if change_result.significant_change:
                        strategy = await self.dynamic_adaptation.select_strategy(adaptable_context, performance)
                        response_data["adaptation"]["strategy"] = strategy.model_dump() if hasattr(strategy, "model_dump") else strategy
                        
                        # Apply strategy to experience sharing adaptations
                        if self.experience_interface:
                            # Get strategy parameters
                            strategy_params = strategy.selected_strategy.parameters.model_dump() \
                                if hasattr(strategy.selected_strategy, "parameters") else {}
                            
                            # Update experience sharing parameters based on strategy
                            if "exploration_rate" in strategy_params:
                                # Higher exploration rate = more cross-user experiences
                                self.cross_user_enabled = strategy_params["exploration_rate"] > 0.3
                            
                            if "risk_tolerance" in strategy_params:
                                # Higher risk tolerance = lower threshold for cross-user sharing
                                self.cross_user_sharing_threshold = max(0.5, 1.0 - strategy_params["risk_tolerance"])
                except Exception as e:
                    logger.error(f"Error in adaptation: {str(e)}")
                    response_data["adaptation"] = {"error": str(e)}
            
            # Generate a meta-cognitive reflection if appropriate
            if self.interaction_count % 10 == 0 and self.reflection_engine:
                try:
                    system_stats = await self.get_system_stats()
                    introspection = await self.reflection_engine.generate_introspection(
                        memory_stats=system_stats["memory_stats"],
                        player_model=context.get("player_model")
                    )
                    response_data["introspection"] = introspection
                except Exception as e:
                    logger.error(f"Error generating introspection: {str(e)}")
            
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
            # Use orchestrator for reflection creation
            reflection_result = await self.memory_orchestrator.create_reflection(topic=topic)
            self.performance_metrics["reflections_generated"] += 1
            
            return reflection_result
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """Run maintenance on all subsystems"""
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="run_maintenance", group_id=self.trace_group_id):
            results = {}
            
            # Run psychological evolution if temporal perception is available
            if hasattr(self, "temporal_perception"):
                try:
                    long_term_drift = await self.temporal_perception.get_long_term_drift()
                    
                    # Apply drift effects to identity evolution
                    if self.identity_evolution:
                        # Update personality traits based on psychological evolution
                        for shift in long_term_drift.personality_shifts:
                            trait = shift["trait"].lower().replace(" ", "_")
                            direction = 1 if shift["direction"] == "increase" else -1
                            magnitude = shift["magnitude"]
                            
                            await self.identity_evolution.update_trait(
                                trait, 
                                direction * magnitude * 0.1
                            )
                        
                        # Update maturity-related traits
                        await self.identity_evolution.update_trait(
                            "patience", 
                            (long_term_drift.patience_level - 0.5) * 0.2
                        )
                        
                        await self.identity_evolution.update_trait(
                            "wisdom", 
                            (long_term_drift.maturity_level - 0.5) * 0.15
                        )
                    
                    results["temporal_evolution"] = {
                        "applied_to_identity": True,
                        "psychological_age": long_term_drift.psychological_age,
                        "traits_updated": [shift["trait"] for shift in long_term_drift.personality_shifts]
                    }
                except Exception as e:
                    logger.error(f"Error in temporal evolution: {str(e)}")
                    results["temporal_evolution"] = {"error": str(e)}
            
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

            if self.agent_enhanced_memory:
                try:
                    # Perform chunk consolidation and optimization
                    procedural_result = await self.agent_enhanced_memory.memory_manager.run_maintenance()
                    results["procedural_maintenance"] = procedural_result
                except Exception as e:
                    logger.error(f"Error in procedural maintenance: {str(e)}")
                    results["procedural_maintenance"] = {"error": str(e)}
            
            # Get issue stats
            issue_summary = await self.issue_tracker.get_issue_summary()
            results["issue_tracker"] = {
                "open_issues": issue_summary["stats"]["open_issues"],
                "categories": issue_summary["stats"]["by_category"],
                "recently_updated": issue_summary["stats"]["recently_updated"]
            }
            
            # Self-analyze capabilities and suggest improvements
            await self._analyze_capabilities()
            
            return results
            
        except Exception as e:
            await self.issue_tracker.process_observation(
                f"Error in run_maintenance: {str(e)}",
                context="Periodic maintenance run"
            )
            return {"error": str(e)}
    
    async def get_temporal_milestones(self) -> List[Dict[str, Any]]:
        """Get temporal milestones for the relationship"""
        if not hasattr(self, "temporal_perception"):
            return []
            
        return self.temporal_perception.milestones

    async def _analyze_capabilities(self):
        """Analyze current capabilities and suggest improvements"""
        # Check memory efficiency
        memory_stats = await self.memory_core.get_memory_stats()
        if memory_stats.get("retrieval_success_rate", 1.0) < 0.7:
            await self.issue_tracker.process_observation(
                f"Low memory retrieval success rate: {memory_stats.get('retrieval_success_rate', 0):.2f}",
                context="Memory system may need optimization"
            )
        
        # Check efficiency issues
        if len(self.performance_metrics["response_times"]) > 10:
            avg_time = sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"])
            if avg_time > 1.5:  # Threshold for concern
                await self.issue_tracker.process_observation(
                    f"Response times trending high: {avg_time:.2f}s average",
                    context="Performance optimization may be needed"
                )
        
        # Check for feature gaps
        if self.cross_user_enabled and self.performance_metrics.get("cross_user_experiences_shared", 0) == 0:
            await self.issue_tracker.process_observation(
                "Cross-user experience sharing is enabled but never used",
                context="Potential unused feature or implementation gap"
            )

    async def _safe_call_dependency(self, module_name: str, method_name: str, *args, **kwargs):
        """Safely call a dependency and track failures"""
        module = getattr(self, module_name, None)
        if not module:
            await self.issue_tracker.process_observation(
                f"Missing module dependency: {module_name}",
                context=f"Called from method requesting {method_name}"
            )
            return None
            
        method = getattr(module, method_name, None)
        if not method:
            await self.issue_tracker.process_observation(
                f"Missing method in module {module_name}: {method_name}",
                context=f"Method not found when needed"
            )
            return None
            
        try:
            return await method(*args, **kwargs)
        except Exception as e:
            await self.issue_tracker.process_observation(
                f"Error in dependency {module_name}.{method_name}: {str(e)}",
                context=f"Args: {args}, Kwargs: {kwargs}"
            )
            return None
    
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

            # Get hormone stats if available
            hormone_stats = {}
            if self.hormone_system:
                try:
                    hormone_stats = await self.get_hormone_stats()
                except Exception as e:
                    logger.error(f"Error getting hormone stats: {str(e)}")
                    hormone_stats = {"error": str(e)}

            # Add reward system stats
            if hasattr(self, "reward_system"):
                result["reward_stats"] = await self.reward_system.get_reward_statistics()
            
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

            procedural_stats = {}
            if self.agent_enhanced_memory:
                try:
                    procedures = list(self.agent_enhanced_memory.procedures.keys())
                    procedural_stats = {
                        "total_procedures": len(procedures),
                        "available_procedures": procedures[:10] if len(procedures) > 10 else procedures,
                        "procedure_domains": list(set(p.domain for p in self.agent_enhanced_memory.procedures.values())),
                        "execution_count": self.agent_enhanced_memory.agents.agent_context.run_stats.get("total_runs", 0),
                        "success_rate": (
                            self.agent_enhanced_memory.agents.agent_context.run_stats.get("successful_runs", 0) / 
                            max(1, self.agent_enhanced_memory.agents.agent_context.run_stats.get("total_runs", 1))
                        )
                    }
                except Exception as e:
                    logger.error(f"Error getting procedural memory stats: {str(e)}")
                    procedural_stats = {"error": str(e)}        
            
            # Get identity state if available
            identity_stats = {}
            if self.experience_interface:
                try:
                    identity_profile = await self.experience_interface._get_identity_profile(RunContextWrapper(context=None))
                    identity_stats = {
                        "preference_count": sum(len(prefs) for prefs in identity_profile["preferences"].values()),
                        "trait_count": len(identity_profile["traits"]),
                        "evolution_history_count": len(identity_profile["evolution_history"]),
                        "dominant_traits": sorted(identity_profile["traits"].items(), key=lambda x: x[1], reverse=True)[:3]
                    }
                except Exception as e:
                    logger.error(f"Error getting identity stats: {str(e)}")
                    identity_stats = {"error": str(e)}
            
            # Get experience interface stats
            experience_stats = {}
            if self.experience_interface:
                try:
                    vector_search_status = await self.experience_interface.check_vector_search_status()
                    experience_stats = {
                        "vector_search_status": vector_search_status,
                        "experiences_shared": self.performance_metrics["experiences_shared"],
                        "cross_user_experiences_shared": self.performance_metrics["cross_user_experiences_shared"],
                        "consolidations_performed": self.performance_metrics["experience_consolidations"]
                    }
                except Exception as e:
                    logger.error(f"Error getting experience stats: {str(e)}")
                    experience_stats = {"error": str(e)}

            # Get identity evolution stats if available
            identity_stats = {}
            if self.identity_evolution:
                try:
                    identity_state = await self.identity_evolution.get_identity_state()
                    identity_stats = {
                        "trait_count": len(identity_state.get("top_traits", {})),
                        "coherence_score": identity_state.get("coherence_score", 0.0),
                        "evolution_rate": identity_state.get("evolution_rate", 0.0),
                        "update_count": identity_state.get("update_count", 0),
                        "dominant_traits": list(identity_state.get("top_traits", {}).keys())[:3] if "top_traits" in identity_state else []
                    }
                except Exception as e:
                    logger.error(f"Error getting identity stats: {str(e)}")
                    identity_stats = {"error": str(e)}
            
            # Get consolidation stats if available
            consolidation_stats = {}
            if self.experience_consolidation:
                try:
                    consolidation_insights = await self.experience_consolidation.get_consolidation_insights()
                    consolidation_stats = {
                        "total_consolidations": consolidation_insights.get("total_consolidations", 0),
                        "last_consolidation": consolidation_insights.get("last_consolidation", "never"),
                        "consolidation_types": consolidation_insights.get("consolidation_types", {}),
                        "ready_for_consolidation": consolidation_insights.get("ready_for_consolidation", False)
                    }
                except Exception as e:
                    logger.error(f"Error getting consolidation stats: {str(e)}")
                    consolidation_stats = {"error": str(e)}
            
            # Get cross-user stats if available
            cross_user_stats = {}
            if self.cross_user_manager:
                try:
                    sharing_stats = await self.cross_user_manager.get_sharing_statistics(str(self.user_id))
                    cross_user_stats = {
                        "total_shares": sharing_stats.get("total_shares", 0),
                        "shares_sent": sharing_stats.get("user_shares", {}).get(str(self.user_id), {}).get("shared", 0),
                        "shares_received": sharing_stats.get("user_shares", {}).get(str(self.user_id), {}).get("received", 0),
                        "most_shared_scenario": sharing_stats.get("most_shared_scenario", "none")
                    }
                except Exception as e:
                    logger.error(f"Error getting cross-user stats: {str(e)}")
                    cross_user_stats = {"error": str(e)}
            
            # Return all stats
            return {
                "memory_stats": memory_stats,
                "emotional_state": {
                    "emotions": emotional_state,
                    "dominant_emotion": dominant_emotion,
                    "dominant_value": dominant_value,
                    "valence": self.emotional_core.get_emotional_valence(),
                    "arousal": self.emotional_core.get_emotional_arousal()
                },
                "procedural_stats": procedural_stats,
                "hormone_stats": hormone_stats,  
                "interaction_stats": {
                    "interaction_count": self.interaction_count,
                    "last_interaction": self.last_interaction.isoformat()
                },
                "performance_metrics": {
                    "memory_operations": self.performance_metrics["memory_operations"],
                    "emotion_updates": self.performance_metrics["emotion_updates"],
                    "reflections_generated": self.performance_metrics["reflections_generated"],
                    "experiences_shared": self.performance_metrics["experiences_shared"],
                    "cross_user_experiences_shared": self.performance_metrics.get("cross_user_experiences_shared", 0),
                    "avg_response_time": avg_response_time
                },
                "introspection": introspection,
                "meta_stats": meta_stats,
                "knowledge_stats": knowledge_stats,
                "identity_stats": identity_stats,
                "consolidation_stats": consolidation_stats,
                "cross_user_stats": cross_user_stats
            }
                
    
    def _should_share_experience(self, user_input: str, context: Dict[str, Any]) -> bool:
        """Determine if we should share an experience based on input and context"""
        # Check for explicit experience requests
        explicit_request = any(phrase in user_input.lower() for phrase in 
                             ["remember", "recall", "tell me about", "have you done", 
                              "previous", "before", "past", "experience", "what happened",
                              "have you ever", "did you ever", "similar", "others"])
        
        if explicit_request:
            return True
        
        # Check if it's a question that could benefit from experience sharing
        is_question = user_input.endswith("?") or user_input.lower().startswith(("what", "how", "when", "where", "why", "who", "can", "could", "do", "did"))
        
        if is_question and "share_experiences" in context and context["share_experiences"]:
            return True
        
        # Check for personal references that might trigger experience sharing
        personal_references = any(phrase in user_input.lower() for phrase in 
                               ["your experience", "you like", "you prefer", "you enjoy",
                                "you think", "you feel", "your opinion", "your view"])
        
        if personal_references:
            return True
        
        # Get user preference for experience sharing if available
        user_id = str(self.user_id)
        if self.experience_interface and hasattr(self.experience_interface, "user_preference_profiles"):
            profile = self.experience_interface._get_user_preference_profile(user_id)
            sharing_preference = profile.get("experience_sharing_preference", 0.5)
            
            # Higher preference means more likely to share experiences even without explicit request
            random_factor = random.random()
            if random_factor < sharing_preference * 0.5:  # Scale down to make this path less common
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
    
    async def _calculate_memory_emotional_impact(self, memories: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate emotional impact from relevant memories"""
        impact = {}
        
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
                
                # Calculate final impact value
                impact_value = primary_intensity * relevance * recency_factor * 0.1
                
                # Add to impact dict
                if primary_emotion not in impact:
                    impact[primary_emotion] = 0
                impact[primary_emotion] += impact_value
        
        return impact
    
    async def _calculate_identity_impact_from_experience(self, experience: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Calculate the impact of an experience on identity formation
        
        Args:
            experience: The experience that may impact identity
            
        Returns:
            Impact on identity components
        """
        # Extract relevant data
        scenario_type = experience.get("scenario_type", "general")
        emotional_context = experience.get("emotional_context", {})
        significance = experience.get("significance", 5) / 10  # Convert to 0-1 scale
        
        # Default empty impact
        impact = {
            "preferences": {},
            "traits": {}
        }
        
        # Impact on scenario preferences based on emotional response
        if scenario_type:
            # Get valence from emotional context
            valence = emotional_context.get("valence", 0)
            
            # Impact depends on emotional valence
            if valence > 0.3:
                # Positive experience with this scenario type
                impact["preferences"]["scenario_types"] = {scenario_type: 0.1 * significance}
            elif valence < -0.3:
                # Negative experience with this scenario type
                impact["preferences"]["scenario_types"] = {scenario_type: -0.05 * significance}
        
        # Impact on traits based on scenario type
        trait_impacts = {}
        
        # Map scenario types to trait impacts
        scenario_trait_map = {
            "teasing": {"playfulness": 0.1, "creativity": 0.05},
            "discipline": {"strictness": 0.1, "dominance": 0.08},
            "dark": {"intensity": 0.1, "cruelty": 0.08},
            "indulgent": {"patience": 0.1, "creativity": 0.08},
            "psychological": {"creativity": 0.1, "intensity": 0.05},
            "nurturing": {"patience": 0.1, "strictness": -0.05},
            "service": {"patience": 0.08, "dominance": 0.05},
            "worship": {"intensity": 0.05, "dominance": 0.1},
            "punishment": {"strictness": 0.1, "cruelty": 0.05}
        }
        
        # Apply trait impacts based on scenario type
        if scenario_type in scenario_trait_map:
            for trait, base_impact in scenario_trait_map[scenario_type].items():
                trait_impacts[trait] = base_impact * significance
        
        # Apply trait impacts from emotional context
        primary_emotion = emotional_context.get("primary_emotion", "")
        primary_intensity = emotional_context.get("primary_intensity", 0.5)
        
        # Map emotions to trait impacts
        emotion_trait_map = {
            "Joy": {"playfulness": 0.1, "patience": 0.05},
            "Sadness": {"patience": 0.05, "intensity": -0.05},
            "Fear": {"intensity": 0.1, "cruelty": 0.05},
            "Anger": {"intensity": 0.1, "strictness": 0.08},
            "Trust": {"patience": 0.1, "dominance": 0.05},
            "Disgust": {"cruelty": 0.1, "strictness": 0.05},
            "Anticipation": {"creativity": 0.1, "playfulness": 0.05},
            "Surprise": {"creativity": 0.1, "intensity": 0.05},
            "Love": {"patience": 0.1, "dominance": 0.05},
            "Frustration": {"intensity": 0.1, "strictness": 0.08}
        }
        
        # Apply emotion-based trait impacts
        if primary_emotion in emotion_trait_map:
            for trait, base_impact in emotion_trait_map[primary_emotion].items():
                if trait in trait_impacts:
                    # Average with existing impact
                    trait_impacts[trait] = (trait_impacts[trait] + (base_impact * primary_intensity)) / 2
                else:
                    # New impact
                    trait_impacts[trait] = base_impact * primary_intensity
        
        # Add trait impacts to overall impact
        if trait_impacts:
            impact["traits"] = trait_impacts
        
        # If no significant impact, return None
        if not impact["preferences"] and not impact["traits"]:
            return None
        
        return impact
    
async def process_user_input_enhanced(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Enhanced processing of user input with comprehensive results."""
    with trace(workflow_name="process_user_input_enhanced", group_id=self.trace_group_id):
        # Initialize context
        if context is None:
            context = {}
            
        # Process through the brain with thinking capability
        result = await self.process_user_input_with_thinking(user_input, context)
        
        # Add additional processing information
        system_stats = await self.get_system_stats()
        
        # Return enhanced result
        return {
            "input": user_input,
            "emotional_state": result["emotional_state"],
            "memories": result["memories"],
            "memory_count": result["memory_count"],
            "has_experience": result["has_experience"],
            "experience_response": result["experience_response"],
            "cross_user_experience": result.get("cross_user_experience", False),
            "memory_id": result["memory_id"],
            "response_time": result["response_time"],
            "thinking_applied": result.get("thinking_applied", False),
            "thinking_level": result.get("thinking_level") if result.get("thinking_applied", False) else None,
            "identity_impact": result.get("identity_impact"),
            "system_stats": {
                "memory_stats": system_stats["memory_stats"],
                "emotional_state": system_stats["emotional_state"],
                "performance_metrics": system_stats["performance_metrics"],
                "identity_stats": system_stats.get("identity_stats", {})
            }
        }
    
    # New methods for enhanced functionality
    
    async def get_identity_state(self) -> Dict[str, Any]:
        """
        Get the current state of Nyx's identity
        
        Returns:
            Identity state information
        """
        if not self.experience_interface:
            return {
                "error": "Experience interface not initialized"
            }
        
        # Check cache
        cache_key = f"identity_{self.interaction_count}"
        if cache_key in self.identity_cache:
            return self.identity_cache[cache_key]
        
        try:
            # Get identity profile
            identity_profile = await self.experience_interface._get_identity_profile(RunContextWrapper(context=None))
            
            # Generate identity reflection
            reflection = await self.experience_interface._generate_identity_reflection(RunContextWrapper(context=None))
            
            # Get top preferences
            top_scenario_prefs = sorted(
                identity_profile["preferences"]["scenario_types"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            top_emotional_prefs = sorted(
                identity_profile["preferences"]["emotional_tones"].items(),
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
            
            # Cache the result
            self.identity_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting identity state: {str(e)}")
            return {
                "error": str(e)
            }
    
    async def adapt_experience_sharing(self, user_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt experience sharing parameters based on user feedback
        
        Args:
            user_id: User ID
            feedback: Feedback data about experience sharing
            
        Returns:
            Adaptation results
        """
        if not self.experience_interface:
            return {
                "error": "Experience interface not initialized"
            }
        
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
            return {
                "error": str(e)
            }
    
    async def run_experience_consolidation(self) -> Dict[str, Any]:
        """
        Run the experience consolidation process
        
        Returns:
            Consolidation results
        """
        if not self.experience_interface:
            return {
                "error": "Experience interface not initialized"
            }
        
        try:
            # Run consolidation
            consolidation_result = await self.experience_interface.consolidate_experiences()
            
            # Update performance metrics
            if consolidation_result.get("status") == "completed":
                self.performance_metrics["experience_consolidations"] += consolidation_result.get("consolidations_created", 0)
            
            return consolidation_result
            
        except Exception as e:
            logger.error(f"Error running experience consolidation: {str(e)}")
            return {
                "error": str(e)
            }
    
    async def _check_and_run_consolidation(self) -> None:
        """Check if it's time for consolidation and run if needed"""
        if not self.experience_interface:
            return
        
        # Check time since last consolidation
        now = datetime.datetime.now()
        time_since_last = (now - self.last_consolidation).total_seconds() / 3600  # hours
        
        if time_since_last >= self.consolidation_interval:
            try:
                # Run consolidation in background
                asyncio.create_task(self.run_experience_consolidation())
                
                # Update last consolidation time
                self.last_consolidation = now
            except Exception as e:
                logger.error(f"Error scheduling consolidation: {str(e)}")
    
    async def get_cross_user_experiences(self, 
                                     query: str, 
                                     limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get relevant experiences from other users/conversations
        
        Args:
            query: Search query
            limit: Maximum number of experiences to return
            
        Returns:
            List of cross-user experiences
        """
        if not self.experience_interface:
            return []
        
        try:
            # Get cross-user experiences
            experiences = await self.experience_interface._get_cross_user_experiences(
                RunContextWrapper(context=None),
                query=query,
                user_id=str(self.user_id),
                limit=limit
            )
            
            return experiences
            
        except Exception as e:
            logger.error(f"Error getting cross-user experiences: {str(e)}")
            return []
    
    async def share_cross_conversation_experience(self, 
                                              query: str, 
                                              target_conversation_id: int) -> Dict[str, Any]:
        """
        Share an experience from the current conversation with another conversation
        
        Args:
            query: Search query to find relevant experience
            target_conversation_id: Target conversation ID
            
        Returns:
            Sharing result
        """
        if not self.experience_interface:
            return {
                "error": "Experience interface not initialized"
            }
        
        try:
            # Get relevant experiences from this conversation
            experiences = await self.experience_interface.retrieve_experiences_enhanced(
                query=query,
                limit=1,
                user_id=str(self.user_id),
                include_cross_user=False  # Only from this user
            )
            
            if not experiences:
                return {
                    "shared": False,
                    "reason": "No relevant experiences found"
                }
            
            # Get the target brain instance
            target_key = f"brain_{self.user_id}_{target_conversation_id}"
            
            if not hasattr(self.__class__, '_instances') or target_key not in self.__class__._instances:
                return {
                    "shared": False,
                    "reason": "Target conversation not found"
                }
            
            target_brain = self.__class__._instances[target_key]
            
            # Share the experience with the target conversation
            experience = experiences[0]
            
            # Add to target's experience interface
            await target_brain.experience_interface._store_experience(
                RunContextWrapper(context=None),
                memory_text=experience.get("content", ""),
                scenario_type=experience.get("scenario_type", "general"),
                entities=experience.get("entities", []),
                emotional_context=experience.get("emotional_context", {}),
                significance=experience.get("significance", 5),
                tags=experience.get("tags", []),
                user_id=str(self.user_id)
            )
            
            return {
                "shared": True,
                "experience": {
                    "id": experience.get("id"),
                    "content": experience.get("content"),
                    "scenario_type": experience.get("scenario_type"),
                    "significance": experience.get("significance")
                },
                "target_conversation_id": target_conversation_id
            }
            
        except Exception as e:
            logger.error(f"Error sharing cross-conversation experience: {str(e)}")
            return {
                "shared": False,
                "error": str(e)
            }
    
    def update_adaptation_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """
        Manually update the adaptation strategy
        
        Args:
            strategy_id: ID of the strategy to apply
            
        Returns:
            Update result
        """
        if not self.dynamic_adaptation:
            return {
                "error": "Dynamic adaptation system not initialized"
            }
        
        try:
            # Update current strategy
            self.dynamic_adaptation.context.current_strategy_id = strategy_id
            
            # Get strategy details
            strategy_details = self.dynamic_adaptation.context.strategies.get(strategy_id, {})
            
            # Apply strategy parameters to experience sharing
            if "parameters" in strategy_details:
                params = strategy_details["parameters"]
                
                # Update cross-user experience settings based on strategy
                if "exploration_rate" in params:
                    # Higher exploration rate = more cross-user experiences
                    self.cross_user_enabled = params["exploration_rate"] > 0.3
                
                if "risk_tolerance" in params:
                    # Higher risk tolerance = lower threshold for cross-user sharing
                    self.cross_user_sharing_threshold = max(0.5, 1.0 - params["risk_tolerance"])
                
                # Update identity evolution settings
                if "adaptation_rate" in params:
                    # Higher adaptation rate = stronger identity evolution
                    self.experience_to_identity_influence = params["adaptation_rate"]
            
            return {
                "strategy_updated": True,
                "strategy_id": strategy_id,
                "strategy_name": strategy_details.get("name", "Unknown"),
                "applied_settings": {
                    "cross_user_enabled": self.cross_user_enabled,
                    "cross_user_sharing_threshold": self.cross_user_sharing_threshold,
                    "experience_to_identity_influence": self.experience_to_identity_influence
                }
            }
            
        except Exception as e:
            logger.error(f"Error updating adaptation strategy: {str(e)}")
            return {
                "strategy_updated": False,
                "error": str(e)
            }

    # Helper methods for parallel processing in NyxBrain class
    
    async def _process_emotional_impact(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process emotional impact of user input"""
        # Process emotional impact
        emotional_stimuli = self.emotional_core.analyze_text_sentiment(user_input)
        emotional_state = self.emotional_core.update_from_stimuli(emotional_stimuli)
        
        # Update performance counter
        self.performance_metrics["emotion_updates"] += 1
        
        return {
            "emotional_state": emotional_state,
            "stimuli": emotional_stimuli
        }
    
    async def _retrieve_memories_with_emotion(self, 
                                          user_input: str, 
                                          context: Dict[str, Any],
                                          emotional_state: Dict[str, float]) -> List[Dict[str, Any]]:
        """Retrieve relevant memories with emotional influence"""
        # Create emotional prioritization for memory types
        # Based on current emotional state, prioritize different memory types
        
        # Example prioritization based on emotional valence and arousal
        valence = self.emotional_core.get_emotional_valence()
        arousal = self.emotional_core.get_emotional_arousal()
        
        # Prioritize experiences and reflections for high emotional states
        if abs(valence) > 0.6 or arousal > 0.7:
            prioritization = {
                "experience": 0.5,
                "reflection": 0.3,
                "abstraction": 0.1,
                "observation": 0.1
            }
        # Prioritize abstractions and reflections for low emotional states
        elif arousal < 0.3:
            prioritization = {
                "abstraction": 0.4,
                "reflection": 0.3,
                "experience": 0.2,
                "observation": 0.1
            }
        # Balanced prioritization for moderate emotional states
        else:
            prioritization = {
                "experience": 0.3,
                "reflection": 0.3,
                "abstraction": 0.2,
                "observation": 0.2
            }
        
        # Adjust prioritization based on emotion-to-memory influence
        for memory_type, priority in prioritization.items():
            prioritization[memory_type] = priority * (1 + self.emotion_to_memory_influence)
        
        # Use prioritized retrieval
        memories = await self.memory_orchestrator.retrieve_memories_with_prioritization(
            query=user_input,
            memory_types=context.get("memory_types", ["observation", "reflection", "abstraction", "experience"]),
            prioritization=prioritization,
            limit=context.get("memory_limit", 5)
        )
        
        # Update performance counter
        self.performance_metrics["memory_operations"] += 1
        
        return memories
    
    async def _check_experience_sharing(self, user_input: str, context: Dict[str, Any]) -> bool:
        """Check if experience sharing should be used"""
        return self._should_share_experience(user_input, context)
    
    async def _share_experience(self, 
                            user_input: str, 
                            context: Dict[str, Any], 
                            emotional_state: Dict[str, float]) -> Dict[str, Any]:
        """Share experience based on user input"""
        # Enhanced experience sharing with cross-user support and adaptation
        experience_result = await self.experience_interface.share_experience_enhanced(
            query=user_input,
            context_data={
                "user_id": str(self.user_id),
                "emotional_state": emotional_state,
                "include_cross_user": self.cross_user_enabled and context.get("include_cross_user", True),
                "scenario_type": context.get("scenario_type", ""),
                "conversation_id": self.conversation_id
            }
        )
        
        if experience_result.get("has_experience", False):
            self.performance_metrics["experiences_shared"] += 1
            
            # Track cross-user experiences
            if experience_result.get("cross_user", False):
                self.performance_metrics["cross_user_experiences_shared"] += 1
        
        return experience_result
    
    async def _process_adaptation(self,
                               user_input: str,
                               context: Dict[str, Any],
                               emotional_state: Dict[str, float],
                               experience_result: Optional[Dict[str, Any]],
                               identity_impact: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process adaptation based on context change"""
        # Prepare context for change detection
        context_for_adaptation = {
            "user_input": user_input,
            "emotional_state": emotional_state,
            "has_experience": experience_result["has_experience"] if experience_result else False,
            "cross_user_experience": experience_result.get("cross_user", False) if experience_result else False,
            "interaction_count": self.interaction_count,
            "identity_impact": True if identity_impact else False
        }
        
        # Detect context change
        context_change_result = await self.dynamic_adaptation.detect_context_change(context_for_adaptation)
        
        adaptation_result = None
        
        # If significant change, run adaptation cycle
        if context_change_result.significant_change:
            # Measure current performance
            current_performance = {
                "success_rate": context.get("success_rate", 0.7),
                "error_rate": context.get("error_rate", 0.1),
                "efficiency": context.get("efficiency", 0.8),
                "response_time": 0.5  # Default value since we don't have full timing yet
            }
            
            # Run adaptation cycle
            adaptation_result = await self.dynamic_adaptation.adaptation_cycle(
                context_for_adaptation, current_performance
            )
        
        return {
            "context_change": context_change_result.model_dump() if hasattr(context_change_result, "model_dump") else context_change_result,
            "adaptation_result": adaptation_result
        }

    async def _determine_main_response(self, 
                                   user_input: str, 
                                   processing_result: Dict[str, Any],
                                   context: Dict[str, Any]) -> Dict[str, str]:
        """Determine the main response content based on processing results"""
        # Determine if experience response should be used
        if processing_result["has_experience"]:
            main_response = processing_result["experience_response"]
            response_type = "experience"
            
            # If it's a cross-user experience, mark it
            if processing_result.get("cross_user_experience", False):
                response_type = "cross_user_experience"
        else:
            # For reasoning-related queries, use the reasoning agents
            if self._is_reasoning_query(user_input):
                try:
                    reasoning_result = await Runner.run(
                        reasoning_triage_agent,
                        user_input
                    )
                    main_response = reasoning_result.final_output
                    response_type = "reasoning"
                except Exception as e:
                    logger.error(f"Error in reasoning response: {str(e)}")
                    # Fallback to standard response
                    main_response = "I understand your question and would like to reason through it with you."
                    response_type = "standard"
            else:
                # No specific experience to share, generate standard response
                # In a real implementation, this would be more sophisticated
                main_response = "I acknowledge your message and have processed it through my systems."
                response_type = "standard"
        
        return {
            "message": main_response,
            "response_type": response_type
        }
    
    async def _generate_emotional_expression(self, emotional_state: Dict[str, float]) -> Dict[str, Any]:
        """Generate emotional expression based on emotional state"""
        # Determine if emotion should be expressed
        should_express_emotion = self.emotional_core.should_express_emotion()
        emotional_expression = None
        
        if should_express_emotion:
            try:
                expression_result = await self.emotional_core.generate_emotional_expression(force=False)
                if expression_result.get("expressed", False):
                    emotional_expression = expression_result.get("expression", "")
            except Exception as e:
                logger.error(f"Error generating emotional expression: {str(e)}")
                emotional_expression = self.emotional_core.get_expression_for_emotion()
        
        return {
            "expression": emotional_expression,
            "should_express": should_express_emotion
        }

# For backward compatibility in case directly imported
NyxBrainInstance = NyxBrain
