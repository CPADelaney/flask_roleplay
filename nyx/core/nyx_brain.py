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
        
        
        # Initialize dynamic adaptation system
        self.dynamic_adaptation = DynamicAdaptationSystem()
        
        # Initialize knowledge core
        self.knowledge_core = KnowledgeCoreAgents()
        await self.knowledge_core.initialize()
        
        # Use integrated reasoning agent as reasoning core
        self.reasoning_core = integrated_reasoning_agent

        await self.temporal_perception.initialize(self, first_interaction_timestamp=None)
        
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
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None):
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
              
            return result
            
        except Exception as e:
            await self.issue_tracker.process_observation(
                f"Error in process_input: {str(e)}",
                context=f"User input: '{user_input[:50]}...'"
            )
            raise

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
