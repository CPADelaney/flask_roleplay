# nyx/nyx_agent_sdk.py

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import asyncpg
import time
import psutil
import random

from agents import Agent, handoff, function_tool, Runner, trace
from agents import ModelSettings, GuardrailFunctionOutput, InputGuardrail, OutputGuardrail, RunConfig
from npcs.npc_agent import NPCAgent, ResourcePool
from pydantic import BaseModel, Field

from db.connection import get_db_connection_context
from memory.memory_nyx_integration import MemoryNyxBridge, get_memory_nyx_bridge
from nyx.user_model_sdk import UserModelContext, ResponseGuidance, UserModelAnalysis
from nyx.nyx_task_integration import NyxTaskIntegration
from nyx.core.emotions.emotional_core import EmotionalCore
from nyx.performance_monitor import PerformanceMonitor
from .response_filter import ResponseFilter
from .nyx_enhanced_system import NyxEnhancedSystem, NyxGoal
from nyx.core.sync.strategy_controller import get_active_strategies

logger = logging.getLogger(__name__)

# ===== Pydantic Models for Structured Outputs =====
class NarrativeResponse(BaseModel):
    """Structured output for Nyx's narrative responses"""
    narrative: str = Field(..., description="The main narrative response as Nyx")
    tension_level: int = Field(0, description="Current narrative tension level (0-10)")
    generate_image: bool = Field(False, description="Whether an image should be generated for this scene")
    image_prompt: Optional[str] = Field(None, description="Prompt for image generation if needed")
    environment_description: Optional[str] = Field(None, description="Updated environment description if changed")
    time_advancement: bool = Field(False, description="Whether time should advance after this interaction")
    
    
class MemoryReflection(BaseModel):
    """Structured output for memory reflections"""
    reflection: str = Field(..., description="The reflection text")
    confidence: float = Field(..., description="Confidence level in the reflection (0.0-1.0)")
    topic: Optional[str] = Field(None, description="Topic of the reflection")

class ContentModeration(BaseModel):
    """Output for content moderation guardrail"""
    is_appropriate: bool = Field(..., description="Whether the content is appropriate")
    reasoning: str = Field(..., description="Reasoning for the decision")
    suggested_adjustment: Optional[str] = Field(None, description="Suggested adjustment if inappropriate")

# ===== Function Tools =====

@function_tool
async def retrieve_memories(ctx, query: str, limit: int = 5) -> str:
    """
    Retrieve relevant memories for Nyx.
    
    Args:
        query: Search query to find memories
        limit: Maximum number of memories to return
    """
    memory_system = ctx.context.memory_system
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    memories = await memory_system.retrieve_memories(
        query=query,
        scopes=["game", "user"],
        memory_types=["observation", "reflection", "abstraction"],
        limit=limit
    )
    
    # Format memories for return
    formatted_memories = []
    for memory in memories:
        relevance = memory.get("relevance", 0.5)
        confidence_marker = "vividly recall" if relevance > 0.8 else \
                          "remember" if relevance > 0.6 else \
                          "think I recall" if relevance > 0.4 else \
                          "vaguely remember"
        
        formatted_memories.append(f"I {confidence_marker}: {memory['memory_text']}")
    
    return "\n".join(formatted_memories)

def enhance_context_with_memories(context, memories):
    """Add memories to context for better decision making."""
    enhanced_context = context.copy()
    enhanced_context['relevant_memories'] = memories
    return enhanced_context

@function_tool
async def add_memory(ctx, memory_text: str, memory_type: str = "observation", significance: int = 5) -> str:
    """
    Add a new memory for Nyx.
    
    Args:
        memory_text: The content of the memory
        memory_type: Type of memory (observation, reflection, abstraction)
        significance: Importance of memory (1-10)
    """
    memory_system = ctx.context.memory_system
    
    memory_id = await memory_system.add_memory(
        memory_text=memory_text,
        memory_type=memory_type,
        memory_scope="game",
        significance=significance,
        tags=["agent_generated"],
        metadata={
            "timestamp": datetime.now().isoformat(),
            "auto_generated": True
        }
    )
    
    return f"Memory added with ID: {memory_id}"

async def get_scene_guidance(self, context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate guidance for scene based on context."""
    prompt = f"Generate guidance for a scene with the following context: {json.dumps(context)}"
    
    # Use existing agent to process prompt
    response = await self.process_input(prompt, context)
    
    # Extract NPC guidance or create default
    npc_guidance = response.get("npc_guidance", {})
    if not npc_guidance:
        npc_guidance = {
            "responding_npcs": [],
            "tone_guidance": {},
            "content_guidance": {},
            "emotion_guidance": {},
            "conflict_guidance": {},
            "nyx_expectations": {}
        }
    
    return npc_guidance

@function_tool
async def detect_user_revelations(ctx, user_message: str) -> str:
    """
    Detect if user is revealing new preferences or patterns.
    
    Args:
        user_message: The user's message to analyze
    """
    lower_message = user_message.lower()
    revelations = []
    
    # Check for explicit kink mentions (migrated from nyx_decision_engine.py)
    kink_keywords = {
        "ass": ["ass", "booty", "behind", "rear"],
        "feet": ["feet", "foot", "toes"],
        "goth": ["goth", "gothic", "dark", "black clothes"],
        "tattoos": ["tattoo", "ink", "inked"],
        "piercings": ["piercing", "pierced", "stud", "ring"],
        "latex": ["latex", "rubber", "shiny"],
        "leather": ["leather", "leathery"],
        "humiliation": ["humiliate", "embarrassed", "ashamed", "pathetic"],
        "submission": ["submit", "obey", "serve", "kneel"]
    }
    
    for kink, keywords in kink_keywords.items():
        if any(keyword in lower_message for keyword in keywords):
            # Check sentiment (simplified)
            sentiment = "neutral"
            pos_words = ["like", "love", "enjoy", "good", "great", "nice", "yes", "please"]
            neg_words = ["don't", "hate", "dislike", "bad", "worse", "no", "never"]
            
            pos_count = sum(1 for word in pos_words if word in lower_message)
            neg_count = sum(1 for word in neg_words if word in lower_message)
            
            if pos_count > neg_count:
                sentiment = "positive"
                intensity = 0.7
            elif neg_count > pos_count:
                sentiment = "negative" 
                intensity = 0.0
            else:
                intensity = 0.4
                
            if sentiment != "negative":
                revelations.append({
                    "type": "kink_preference",
                    "kink": kink,
                    "intensity": intensity,
                    "source": "explicit_mention"
                })
    
    # Check for behavior patterns (migrated from nyx_decision_engine.py)
    if "don't tell me what to do" in lower_message or "i won't" in lower_message:
        revelations.append({
            "type": "behavior_pattern",
            "pattern": "resistance",
            "intensity": 0.6,
            "source": "explicit_statement"
        })
    
    if "yes mistress" in lower_message or "i'll obey" in lower_message:
        revelations.append({
            "type": "behavior_pattern",
            "pattern": "submission",
            "intensity": 0.8,
            "source": "explicit_statement"
        })
    
    return json.dumps(revelations)

@function_tool
async def enhance_context_with_strategies(context: Dict[str, Any], conn) -> Dict[str, Any]:
    strategies = await get_active_strategies(conn)
    context["nyx2_strategies"] = strategies
    return context


@function_tool
async def get_user_model_guidance(ctx) -> str:
    """
    Get guidance for how Nyx should respond based on the user model.
    """
    user_model_manager = ctx.context.user_model
    guidance = await user_model_manager.get_response_guidance()
    
    # Format guidance for return
    top_kinks = guidance.get("top_kinks", [])
    kink_str = ", ".join([f"{k} (level {l})" for k, l in top_kinks])
    
    behavior_patterns = guidance.get("behavior_patterns", {})
    pattern_str = ", ".join([f"{k}: {v}" for k, v in behavior_patterns.items()])
    
    suggested_intensity = guidance.get("suggested_intensity", 0.5)
    
    return f"""
User Guidance:
- Top interests: {kink_str}
- Behavior patterns: {pattern_str}
- Suggested intensity: {suggested_intensity:.1f}/1.0

Reflections:
{guidance.get('reflections', [])}
"""

@function_tool
async def generate_image_from_scene(
    ctx, 
    scene_description: str, 
    characters: List[str], 
    style: str = "realistic"
) -> str:
    """
    Generate an image for the current scene.
    
    Args:
        scene_description: Description of the scene
        characters: List of characters in the scene
        style: Style for the image
    """
    # Connect to your existing image generation logic
    from routes.ai_image_generator import generate_roleplay_image_from_gpt
    
    image_data = {
        "scene_description": scene_description,
        "characters": characters,
        "style": style
    }
    
    result = generate_roleplay_image_from_gpt(
        image_data,
        ctx.context.user_id,
        ctx.context.conversation_id
    )
    
    if result and "image_urls" in result and result["image_urls"]:
        return f"Image generated: {result['image_urls'][0]}"
    else:
        return "Failed to generate image"

# ===== Guardrail Functions =====

async def content_moderation_guardrail(ctx, agent, input_data):
    """Input guardrail for content moderation"""
    content_moderator = Agent(
        name="Content Moderator",
        instructions="You check if user input is appropriate for the femdom roleplay setting, ensuring it doesn't violate terms of service while allowing consensual adult content. Flag any problematic content and suggest adjustments.",
        output_type=ContentModeration
    )
    
    result = await Runner.run(content_moderator, input_data, context=ctx.context)
    final_output = result.final_output_as(ContentModeration)
    
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_appropriate,
    )

# ===== Main Agent Definitions =====

class AgentContext:
    """Enhanced context for agentic behavior."""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Core systems
        self.memory_system = None
        self.user_model = None
        self.task_integration = None
        self.belief_system = None
        self.emotional_system = None
        
        # Agentic state
        self.current_goals = []
        self.active_tasks = []
        self.decision_history = []
        self.state_history = []
        self.last_action = None
        self.last_result = None
        self.current_emotional_state = {}
        self.beliefs = {}
        self.intentions = []
        
        # Performance tracking
        self.action_success_rate = 0.0
        self.decision_confidence = 0.0
        self.goal_progress = {}
        self.performance_metrics = {
            "total_actions": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "average_decision_time": 0.0,
            "adaptation_rate": 0.0,
            "memory_usage": 0.0,
            "cpu_usage": 0.0,
            "response_times": [],
            "error_rates": {
                "total": 0,
                "recovered": 0,
                "unrecovered": 0
            }
        }
        
        # Learning state
        self.learned_patterns = {}
        self.strategy_effectiveness = {}
        self.adaptation_history = []
        self.learning_metrics = {
            "pattern_recognition_rate": 0.0,
            "strategy_improvement_rate": 0.0,
            "adaptation_success_rate": 0.0
        }
        
        # Resource management
        self.resource_pools = {
            "decisions": ResourcePool(max_concurrent=10, timeout=45.0),
            "perceptions": ResourcePool(max_concurrent=15, timeout=30.0),
            "memory_operations": ResourcePool(max_concurrent=20, timeout=20.0)
        }
        self.resource_usage = {
            "memory": 0,
            "cpu": 0,
            "network": 0
        }
        
        # Context management
        self.context_version = 0
        self.context_cache = {}
        self.context_subscribers = {}
        self.last_context_update = None
        
        # Agent communication
        self.communication_history = []
        self.message_routing = {}
        self.agent_connections = {}
        
        # Error handling
        self.error_states = {}
        self.error_recovery_strategies = {}
        self.error_log = []

    @classmethod
    async def create(cls, user_id: int, conversation_id: int):
        """Async factory method to properly initialize the context."""
        context = cls(user_id, conversation_id)
        await context._initialize_systems()
        return context
    
    async def _initialize_systems(self):
        """Initialize core systems and load initial state."""
        # Initialize memory system
        self.memory_system = await get_memory_nyx_bridge(
            self.user_id,
            self.conversation_id
        )
        
        # Initialize user model
        self.user_model = await UserModelManager.get_instance(
            self.user_id,
            self.conversation_id
        )
        
        # Initialize task integration
        self.task_integration = await NyxTaskIntegration.get_instance(
            self.user_id,
            self.conversation_id
        )
        
        # Initialize belief system
        self.belief_system = await BeliefSystem.get_instance(
            self.user_id,
            self.conversation_id
        )
        
        # Initialize emotional system
        self.emotional_system = await EmotionalSystem.get_instance(
            self.user_id,
            self.conversation_id
        )
        
        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor.get_instance(
            self.user_id,
            self.conversation_id
        )
        
        # Load initial state
        await self._load_initial_state()
        
        # Start background monitoring
        self._start_background_monitoring()

    
    def _start_background_monitoring(self):
        """Start background monitoring tasks."""
        asyncio.create_task(self._monitor_resource_usage())
        asyncio.create_task(self._monitor_performance())
        asyncio.create_task(self._cleanup_resources())
    
    async def _monitor_resource_usage(self):
        """Monitor resource usage in the background."""
        while True:
            try:
                # Get current resource usage
                self.resource_usage["memory"] = psutil.Process().memory_info().rss
                self.resource_usage["cpu"] = psutil.Process().cpu_percent()
                self.resource_usage["network"] = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
                
                # Update performance metrics
                self.performance_metrics["memory_usage"] = self.resource_usage["memory"]
                self.performance_metrics["cpu_usage"] = self.resource_usage["cpu"]
                
                # Check for resource limits
                if self.resource_usage["memory"] > self.resource_pools["memory_operations"].max_memory:
                    await self._handle_resource_limit("memory")
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_performance(self):
        """Monitor performance metrics in the background."""
        while True:
            try:
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Check for performance issues
                if self.performance_metrics["error_rates"]["total"] > 0.1:
                    await self._handle_performance_issue()
                
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error monitoring performance: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_resources(self):
        """Clean up resources periodically."""
        while True:
            try:
                # Clean up old context cache
                current_time = time.time()
                for key, value in list(self.context_cache.items()):
                    if current_time - value["timestamp"] > 3600:  # 1 hour
                        del self.context_cache[key]
                
                # Clean up old communication history
                if len(self.communication_history) > 1000:
                    self.communication_history = self.communication_history[-1000:]
                
                # Clean up old error log
                if len(self.error_log) > 1000:
                    self.error_log = self.error_log[-1000:]
                
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Error cleaning up resources: {e}")
                await asyncio.sleep(3600)
    
    async def update_context(self, new_context: Dict[str, Any], use_delta: bool = True):
        """Update context with version tracking and delta updates."""
        if use_delta and self.last_context_update:
            # Calculate delta
            delta = self._calculate_context_delta(self.context_cache, new_context)
            if delta:
                # Apply delta
                self._apply_context_delta(delta)
        else:
            # Full update
            self.context_cache = new_context
        
        # Update version and timestamp
        self.context_version += 1
        self.last_context_update = time.time()
        
        # Notify subscribers
        await self._notify_context_subscribers()
    
    def _calculate_context_delta(self, old_context: Dict[str, Any], new_context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate delta between old and new context."""
        delta = {}
        for key, new_value in new_context.items():
            if key not in old_context or old_context[key] != new_value:
                delta[key] = new_value
        return delta
    
    def _apply_context_delta(self, delta: Dict[str, Any]):
        """Apply delta updates to context."""
        for key, value in delta.items():
            self.context_cache[key] = value
    
    def subscribe_to_context(self, path: str, callback: Callable):
        """Subscribe to context changes."""
        if path not in self.context_subscribers:
            self.context_subscribers[path] = []
        self.context_subscribers[path].append(callback)
    
    async def _notify_context_subscribers(self):
        """Notify subscribers of context changes."""
        for path, callbacks in self.context_subscribers.items():
            for callback in callbacks:
                try:
                    await callback(self.context_cache)
                except Exception as e:
                    logger.error(f"Error notifying context subscriber: {e}")
    
    async def communicate_with_agent(self, target_agent: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Communicate with another agent."""
        # Add to communication history
        self.communication_history.append({
            "timestamp": time.time(),
            "source": "self",
            "target": target_agent,
            "message": message
        })
        
        # Route message
        if target_agent in self.message_routing:
            try:
                response = await self.message_routing[target_agent](message)
                
                # Record response
                self.communication_history.append({
                    "timestamp": time.time(),
                    "source": target_agent,
                    "target": "self",
                    "message": response
                })
                
                return response
            except Exception as e:
                await self._handle_communication_error(target_agent, e)
                return {"error": str(e)}
        else:
            return {"error": f"No route to agent {target_agent}"}
    
    async def _handle_communication_error(self, target_agent: str, error: Exception):
        """Handle communication errors."""
        self.error_log.append({
            "timestamp": time.time(),
            "type": "communication_error",
            "target": target_agent,
            "error": str(error)
        })
        
        # Update error metrics
        self.performance_metrics["error_rates"]["total"] += 1
        
        # Try to recover
        if target_agent in self.error_recovery_strategies:
            try:
                await self.error_recovery_strategies[target_agent](error)
                self.performance_metrics["error_rates"]["recovered"] += 1
            except Exception as e:
                self.performance_metrics["error_rates"]["unrecovered"] += 1
                logger.error(f"Failed to recover from communication error: {e}")
    
    async def _handle_resource_limit(self, resource_type: str):
        """Handle resource limit reached."""
        if resource_type == "memory":
            # Clear caches
            self.context_cache.clear()
            self.communication_history = self.communication_history[-100:]
            self.error_log = self.error_log[-100:]
            
            # Force garbage collection
            import gc
            gc.collect()
    
    async def _handle_performance_issue(self):
        """Handle performance issues."""
        # Log performance issue
        self.error_log.append({
            "timestamp": time.time(),
            "type": "performance_issue",
            "metrics": self.performance_metrics.copy()
        })
        
        # Try to optimize
        await self._optimize_performance()
    
    async def _optimize_performance(self):
        """Optimize performance based on metrics."""
        # Reduce concurrent operations
        for pool in self.resource_pools.values():
            pool.max_concurrent = max(1, pool.max_concurrent - 1)
        
        # Clear old data
        self.context_cache.clear()
        self.communication_history = self.communication_history[-100:]
        
        # Force garbage collection
        import gc
        gc.collect()
    
    async def get_state_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the current agent state."""
        return {
            "goals": self.current_goals,
            "active_tasks": self.active_tasks,
            "emotional_state": self.current_emotional_state,
            "beliefs": self.beliefs,
            "performance": {
                "action_success_rate": self.action_success_rate,
                "decision_confidence": self.decision_confidence,
                "goal_progress": self.goal_progress,
                "metrics": self.performance_metrics,
                "resource_usage": self.resource_usage
            },
            "learning": {
                "learned_patterns": self.learned_patterns,
                "strategy_effectiveness": self.strategy_effectiveness,
                "adaptation_history": self.adaptation_history[-5:],
                "metrics": self.learning_metrics
            },
            "context": {
                "version": self.context_version,
                "last_update": self.last_context_update,
                "cache_size": len(self.context_cache)
            },
            "communication": {
                "history_size": len(self.communication_history),
                "active_connections": len(self.agent_connections)
            },
            "errors": {
                "total": self.performance_metrics["error_rates"]["total"],
                "recovered": self.performance_metrics["error_rates"]["recovered"],
                "unrecovered": self.performance_metrics["error_rates"]["unrecovered"]
            }
        }

    async def make_decision(self, context: Dict[str, Any], options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make a decision based on context, emotional state, and learned patterns.
        
        Args:
            context: Current context information
            options: List of possible actions/decisions
            
        Returns:
            Selected decision with confidence score
        """
        try:
            # Get emotional state influence
            emotional_state = await self.get_emotional_state()
            emotional_weight = self._calculate_emotional_weight(emotional_state)
            
            # Get learned patterns influence
            pattern_weight = self._calculate_pattern_weight(context)
            
            # Calculate option scores
            scored_options = []
            for option in options:
                # Base score from context relevance
                context_score = self._calculate_context_relevance(option, context)
                
                # Emotional influence
                emotional_score = self._calculate_emotional_alignment(option, emotional_state)
                
                # Pattern alignment
                pattern_score = self._calculate_pattern_alignment(option, context)
                
                # Combine scores with weights
                final_score = (
                    context_score * 0.4 +
                    emotional_score * emotional_weight * 0.3 +
                    pattern_score * pattern_weight * 0.3
                )
                
                scored_options.append({
                    "option": option,
                    "score": final_score,
                    "components": {
                        "context": context_score,
                        "emotional": emotional_score,
                        "pattern": pattern_score
                    }
                })
            
            # Sort by score and select best option
            scored_options.sort(key=lambda x: x["score"], reverse=True)
            selected = scored_options[0]
            
            # Update decision history
            self.decision_history.append({
                "timestamp": time.time(),
                "selected_option": selected["option"],
                "score": selected["score"],
                "components": selected["components"],
                "context": context
            })
            
            # Update performance metrics
            self.performance_metrics["total_actions"] += 1
            self.decision_confidence = selected["score"]
            
            return {
                "decision": selected["option"],
                "confidence": selected["score"],
                "components": selected["components"]
            }
            
        except Exception as e:
            logger.error(f"Error in decision making: {e}")
            self.error_log.append({
                "timestamp": time.time(),
                "error": str(e),
                "context": context
            })
            return self._get_fallback_decision(options)
    
    def _calculate_emotional_weight(self, emotional_state: Dict[str, Any]) -> float:
        """Calculate how much emotional state should influence decisions."""
        # Higher weight for strong emotions
        intensity = max(abs(emotional_state.get("valence", 0)), abs(emotional_state.get("arousal", 0)))
        return min(1.0, intensity * 2.0)
    
    def _calculate_pattern_weight(self, context: Dict[str, Any]) -> float:
        """Calculate how much learned patterns should influence decisions."""
        # Higher weight if we have relevant patterns
        relevant_patterns = [
            p for p in self.learned_patterns.values()
            if self._pattern_matches_context(p, context)
        ]
        return min(1.0, len(relevant_patterns) * 0.2)
    
    def _calculate_context_relevance(self, option: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate how relevant an option is to the current context."""
        # Simple keyword matching for now
        option_keywords = set(option.get("keywords", []))
        context_keywords = set(context.get("keywords", []))
        if not option_keywords or not context_keywords:
            return 0.5
        return len(option_keywords.intersection(context_keywords)) / len(option_keywords)
    
    def _calculate_emotional_alignment(self, option: Dict[str, Any], emotional_state: Dict[str, Any]) -> float:
        """Calculate how well an option aligns with current emotional state."""
        # Check if option's emotional impact matches current state
        option_impact = option.get("emotional_impact", {})
        if not option_impact:
            return 0.5
            
        valence_match = abs(option_impact.get("valence", 0) - emotional_state.get("valence", 0))
        arousal_match = abs(option_impact.get("arousal", 0) - emotional_state.get("arousal", 0))
        
        return 1.0 - (valence_match + arousal_match) / 2.0
    
    def _calculate_pattern_alignment(self, option: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate how well an option aligns with learned patterns."""
        relevant_patterns = [
            p for p in self.learned_patterns.values()
            if self._pattern_matches_context(p, context)
        ]
        
        if not relevant_patterns:
            return 0.5
            
        # Average alignment with relevant patterns
        alignments = [
            self._calculate_pattern_option_alignment(p, option)
            for p in relevant_patterns
        ]
        return sum(alignments) / len(alignments)
    
    def _pattern_matches_context(self, pattern: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if a pattern matches the current context."""
        pattern_keywords = set(pattern.get("keywords", []))
        context_keywords = set(context.get("keywords", []))
        return bool(pattern_keywords.intersection(context_keywords))
    
    def _calculate_pattern_option_alignment(self, pattern: Dict[str, Any], option: Dict[str, Any]) -> float:
        """Calculate how well an option aligns with a specific pattern."""
        pattern_actions = set(pattern.get("successful_actions", []))
        option_keywords = set(option.get("keywords", []))
        
        if not pattern_actions or not option_keywords:
            return 0.5
            
        # Check if option keywords match any successful actions
        matches = sum(1 for action in pattern_actions if any(kw in action for kw in option_keywords))
        return matches / len(pattern_actions)
    
    def _get_fallback_decision(self, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get a fallback decision when normal decision making fails."""
        # Simple random selection as fallback
        selected = random.choice(options)
        return {
            "decision": selected,
            "confidence": 0.3,
            "components": {
                "context": 0.3,
                "emotional": 0.3,
                "pattern": 0.3
            }
        }

    async def learn_from_experience(self, experience: Dict[str, Any]):
        """
        Learn from an experience and update internal patterns and strategies.
        
        Args:
            experience: Dictionary containing experience data including:
                - action: The action taken
                - context: The context when action was taken
                - outcome: The outcome of the action
                - emotional_state: Emotional state during the experience
                - success: Whether the action was successful
        """
        try:
            # Extract experience components
            action = experience.get("action", {})
            context = experience.get("context", {})
            outcome = experience.get("outcome", {})
            emotional_state = experience.get("emotional_state", {})
            success = experience.get("success", False)
            
            # Update pattern recognition
            await self._update_patterns(action, context, success)
            
            # Update strategy effectiveness
            await self._update_strategies(action, context, success)
            
            # Update adaptation history
            self.adaptation_history.append({
                "timestamp": time.time(),
                "experience": experience,
                "success": success
            })
            
            # Update learning metrics
            self._update_learning_metrics(success)
            
            # Clean up old adaptation history
            if len(self.adaptation_history) > 1000:
                self.adaptation_history = self.adaptation_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error in learning from experience: {e}")
            self.error_log.append({
                "timestamp": time.time(),
                "error": str(e),
                "experience": experience
            })
    
    async def _update_patterns(self, action: Dict[str, Any], context: Dict[str, Any], success: bool):
        """Update learned patterns based on experience."""
        # Extract keywords from action and context
        action_keywords = set(action.get("keywords", []))
        context_keywords = set(context.get("keywords", []))
        
        # Create pattern key
        pattern_key = f"{','.join(sorted(action_keywords))}_{','.join(sorted(context_keywords))}"
        
        if pattern_key not in self.learned_patterns:
            self.learned_patterns[pattern_key] = {
                "keywords": list(action_keywords | context_keywords),
                "successful_actions": [],
                "failed_actions": [],
                "success_rate": 0.0,
                "total_occurrences": 0
            }
        
        pattern = self.learned_patterns[pattern_key]
        pattern["total_occurrences"] += 1
        
        if success:
            pattern["successful_actions"].append(action)
            if len(pattern["successful_actions"]) > 100:
                pattern["successful_actions"] = pattern["successful_actions"][-100:]
        else:
            pattern["failed_actions"].append(action)
            if len(pattern["failed_actions"]) > 100:
                pattern["failed_actions"] = pattern["failed_actions"][-100:]
        
        # Update success rate
        total_successful = len(pattern["successful_actions"])
        total_actions = total_successful + len(pattern["failed_actions"])
        pattern["success_rate"] = total_successful / total_actions if total_actions > 0 else 0.0
    
    async def _update_strategies(self, action: Dict[str, Any], context: Dict[str, Any], success: bool):
        """Update strategy effectiveness based on experience."""
        # Extract strategy information from action
        strategy = action.get("strategy", "default")
        
        if strategy not in self.strategy_effectiveness:
            self.strategy_effectiveness[strategy] = {
                "successful_uses": 0,
                "total_uses": 0,
                "success_rate": 0.0,
                "contexts": {},
                "last_used": time.time()
            }
        
        strategy_data = self.strategy_effectiveness[strategy]
        strategy_data["total_uses"] += 1
        
        if success:
            strategy_data["successful_uses"] += 1
            
            # Update context-specific success
            context_key = self._get_context_key(context)
            if context_key not in strategy_data["contexts"]:
                strategy_data["contexts"][context_key] = {
                    "successful_uses": 0,
                    "total_uses": 0,
                    "success_rate": 0.0
                }
            
            context_data = strategy_data["contexts"][context_key]
            context_data["successful_uses"] += 1
            context_data["total_uses"] += 1
            context_data["success_rate"] = context_data["successful_uses"] / context_data["total_uses"]
        
        # Update overall success rate
        strategy_data["success_rate"] = strategy_data["successful_uses"] / strategy_data["total_uses"]
        strategy_data["last_used"] = time.time()
    
    def _get_context_key(self, context: Dict[str, Any]) -> str:
        """Generate a key for categorizing contexts."""
        # Extract relevant context features
        features = [
            context.get("location", "unknown"),
            context.get("time_of_day", "unknown"),
            context.get("weather", "unknown"),
            context.get("mood", "unknown")
        ]
        return "_".join(features)
    
    def _update_learning_metrics(self, success: bool):
        """Update learning-related performance metrics."""
        # Update pattern recognition rate
        if len(self.learned_patterns) > 0:
            total_patterns = len(self.learned_patterns)
            successful_patterns = sum(1 for p in self.learned_patterns.values() if p["success_rate"] > 0.5)
            self.learning_metrics["pattern_recognition_rate"] = successful_patterns / total_patterns
        
        # Update strategy improvement rate
        if len(self.strategy_effectiveness) > 0:
            total_strategies = len(self.strategy_effectiveness)
            improving_strategies = sum(
                1 for s in self.strategy_effectiveness.values()
                if s["success_rate"] > 0.5
            )
            self.learning_metrics["strategy_improvement_rate"] = improving_strategies / total_strategies
        
        # Update adaptation success rate
        if len(self.adaptation_history) > 0:
            recent_adaptations = self.adaptation_history[-100:]  # Look at last 100 adaptations
            successful_adaptations = sum(1 for a in recent_adaptations if a["success"])
            self.learning_metrics["adaptation_success_rate"] = successful_adaptations / len(recent_adaptations)

    async def process_emotional_state(self, context: Dict[str, Any], user_emotion: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process and update emotional state based on context and user emotion.
        
        Args:
            context: Current context information
            user_emotion: Optional user emotional state
            
        Returns:
            Updated emotional state
        """
        try:
            # Get current emotional state
            current_state = await self.get_emotional_state()
            
            # Process context-based emotional changes
            context_impact = self._calculate_context_emotional_impact(context)
            
            # Process user emotion influence if available
            user_influence = self._calculate_user_emotion_influence(user_emotion) if user_emotion else {}
            
            # Process memory-based emotional changes
            memory_impact = await self._get_memory_emotional_impact(context)
            
            # Combine emotional influences
            new_state = self._combine_emotional_influences(
                current_state,
                context_impact,
                user_influence,
                memory_impact
            )
            
            # Apply emotional decay
            new_state = self._apply_emotional_decay(new_state)
            
            # Update emotional state
            await self.update_emotional_state(new_state)
            
            # Record emotional change
            self._record_emotional_change(current_state, new_state, context)
            
            return new_state
            
        except Exception as e:
            logger.error(f"Error processing emotional state: {e}")
            self.error_log.append({
                "timestamp": time.time(),
                "error": str(e),
                "context": context,
                "user_emotion": user_emotion
            })
            return await self.get_emotional_state()
    
    def _calculate_context_emotional_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate emotional impact from current context."""
        impact = {
            "valence": 0.0,  # Positive/negative (-1 to 1)
            "arousal": 0.0,  # Intensity (0 to 1)
            "dominance": 0.0  # Control level (0 to 1)
        }
        
        # Analyze context features
        features = {
            "location": context.get("location", ""),
            "time_of_day": context.get("time_of_day", ""),
            "weather": context.get("weather", ""),
            "mood": context.get("mood", ""),
            "events": context.get("events", []),
            "interactions": context.get("interactions", [])
        }
        
        # Location impact
        if "dangerous" in features["location"].lower():
            impact["valence"] -= 0.3
            impact["arousal"] += 0.4
        elif "safe" in features["location"].lower():
            impact["valence"] += 0.2
            impact["arousal"] -= 0.2
        
        # Time impact
        if features["time_of_day"] == "night":
            impact["arousal"] += 0.2
        elif features["time_of_day"] == "morning":
            impact["valence"] += 0.1
        
        # Weather impact
        if "storm" in features["weather"].lower():
            impact["arousal"] += 0.3
            impact["valence"] -= 0.2
        elif "sunny" in features["weather"].lower():
            impact["valence"] += 0.2
        
        # Mood impact
        if "happy" in features["mood"].lower():
            impact["valence"] += 0.3
        elif "sad" in features["mood"].lower():
            impact["valence"] -= 0.3
        
        # Events impact
        for event in features["events"]:
            if "conflict" in event.lower():
                impact["arousal"] += 0.2
                impact["valence"] -= 0.2
            elif "celebration" in event.lower():
                impact["valence"] += 0.3
                impact["arousal"] += 0.2
        
        # Normalize values
        impact["valence"] = max(min(impact["valence"], 1.0), -1.0)
        impact["arousal"] = max(min(impact["arousal"], 1.0), 0.0)
        impact["dominance"] = max(min(impact["dominance"], 1.0), 0.0)
        
        return impact
    
    def _calculate_user_emotion_influence(self, user_emotion: Dict[str, Any]) -> Dict[str, float]:
        """Calculate emotional influence from user's emotional state."""
        influence = {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0
        }
        
        # Extract user emotion components
        user_valence = user_emotion.get("valence", 0.0)
        user_arousal = user_emotion.get("arousal", 0.0)
        
        # Apply empathy-based influence
        empathy_factor = 0.3  # How much user emotion affects agent
        influence["valence"] = user_valence * empathy_factor
        influence["arousal"] = user_arousal * empathy_factor
        
        # Normalize values
        influence["valence"] = max(min(influence["valence"], 1.0), -1.0)
        influence["arousal"] = max(min(influence["arousal"], 1.0), 0.0)
        influence["dominance"] = max(min(influence["dominance"], 1.0), 0.0)
        
        return influence
    
    async def _get_memory_emotional_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Get emotional impact from relevant memories."""
        impact = {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0
        }
        
        try:
            # Get relevant memories
            memories = await self.memory_system.get_relevant_memories(
                context=context,
                limit=5
            )
            
            if not memories:
                return impact
            
            # Calculate emotional impact from memories
            total_weight = 0
            weighted_valence = 0
            weighted_arousal = 0
            
            for memory in memories:
                # Get memory emotional content
                memory_emotion = memory.get("emotional_content", {})
                memory_valence = memory_emotion.get("valence", 0.0)
                memory_arousal = memory_emotion.get("arousal", 0.0)
                
                # Calculate memory weight based on relevance and recency
                weight = self._calculate_memory_weight(memory)
                total_weight += weight
                
                weighted_valence += memory_valence * weight
                weighted_arousal += memory_arousal * weight
            
            if total_weight > 0:
                impact["valence"] = weighted_valence / total_weight
                impact["arousal"] = weighted_arousal / total_weight
            
            return impact
            
        except Exception as e:
            logger.error(f"Error getting memory emotional impact: {e}")
            return impact
    
    def _calculate_memory_weight(self, memory: Dict[str, Any]) -> float:
        """Calculate weight for a memory based on relevance and recency."""
        # Get memory properties
        recency = memory.get("recency", 0)  # 0 to 1, where 1 is most recent
        relevance = memory.get("relevance", 0)  # 0 to 1
        emotional_intensity = memory.get("emotional_intensity", 0)  # 0 to 1
        
        # Combine factors with weights
        weight = (
            recency * 0.4 +
            relevance * 0.4 +
            emotional_intensity * 0.2
        )
        
        return max(min(weight, 1.0), 0.0)
    
    def _combine_emotional_influences(
        self,
        current: Dict[str, float],
        context: Dict[str, float],
        user: Dict[str, float],
        memory: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine different emotional influences into new state."""
        # Define weights for different influences
        weights = {
            "current": 0.4,    # Current state has highest weight
            "context": 0.3,    # Context is second most important
            "user": 0.2,       # User influence is third
            "memory": 0.1      # Memory influence is least important
        }
        
        # Combine each emotional dimension
        new_state = {
            "valence": (
                current["valence"] * weights["current"] +
                context["valence"] * weights["context"] +
                user["valence"] * weights["user"] +
                memory["valence"] * weights["memory"]
            ),
            "arousal": (
                current["arousal"] * weights["current"] +
                context["arousal"] * weights["context"] +
                user["arousal"] * weights["user"] +
                memory["arousal"] * weights["memory"]
            ),
            "dominance": (
                current["dominance"] * weights["current"] +
                context["dominance"] * weights["context"] +
                user["dominance"] * weights["user"] +
                memory["dominance"] * weights["memory"]
            )
        }
        
        # Normalize values
        new_state["valence"] = max(min(new_state["valence"], 1.0), -1.0)
        new_state["arousal"] = max(min(new_state["arousal"], 1.0), 0.0)
        new_state["dominance"] = max(min(new_state["dominance"], 1.0), 0.0)
        
        return new_state
    
    def _apply_emotional_decay(self, state: Dict[str, float]) -> Dict[str, float]:
        """Apply natural emotional decay to return to baseline."""
        decay_rate = 0.1  # Rate at which emotions return to baseline
        
        # Apply decay to each dimension
        state["valence"] *= (1 - decay_rate)
        state["arousal"] *= (1 - decay_rate)
        state["dominance"] *= (1 - decay_rate)
        
        return state
    
    def _record_emotional_change(
        self,
        old_state: Dict[str, float],
        new_state: Dict[str, float],
        context: Dict[str, Any]
    ):
        """Record emotional state changes for analysis."""
        change = {
            "timestamp": time.time(),
            "old_state": old_state,
            "new_state": new_state,
            "context": context,
            "changes": {
                "valence": new_state["valence"] - old_state["valence"],
                "arousal": new_state["arousal"] - old_state["arousal"],
                "dominance": new_state["dominance"] - old_state["dominance"]
            }
        }
        
        # Add to emotional history
        self.current_emotional_state["history"] = self.current_emotional_state.get("history", [])
        self.current_emotional_state["history"].append(change)
        
        # Limit history size
        if len(self.current_emotional_state["history"]) > 100:
            self.current_emotional_state["history"] = self.current_emotional_state["history"][-100:]

    async def manage_scenario(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage and coordinate complex scenarios across multiple systems.
        
        Args:
            scenario_data: Dictionary containing scenario information including:
                - scenario_id: Unique identifier for the scenario
                - type: Type of scenario (e.g., "training", "conflict", "social")
                - participants: List of involved entities
                - objectives: List of scenario objectives
                - constraints: Any constraints or limitations
                - environment: Environmental conditions
                
        Returns:
            Updated scenario state and coordination plan
        """
        try:
            # Initialize scenario state
            scenario_state = {
                "id": scenario_data["scenario_id"],
                "type": scenario_data["type"],
                "participants": scenario_data["participants"],
                "objectives": scenario_data["objectives"],
                "constraints": scenario_data.get("constraints", {}),
                "environment": scenario_data.get("environment", {}),
                "current_phase": "initialization",
                "progress": {},
                "active_tasks": [],
                "completed_tasks": [],
                "conflicts": [],
                "emotional_states": {},
                "resource_usage": {},
                "start_time": time.time(),
                "last_update": time.time()
            }
            
            # Coordinate with memory system
            await self._initialize_scenario_memories(scenario_state)
            
            # Coordinate with emotional system
            await self._initialize_scenario_emotions(scenario_state)
            
            # Coordinate with belief system
            await self._initialize_scenario_beliefs(scenario_state)
            
            # Generate initial tasks
            initial_tasks = await self._generate_scenario_tasks(scenario_state)
            scenario_state["active_tasks"] = initial_tasks
            
            # Initialize progress tracking
            for objective in scenario_state["objectives"]:
                scenario_state["progress"][objective["id"]] = {
                    "status": "pending",
                    "completion": 0.0,
                    "blockers": [],
                    "dependencies": objective.get("dependencies", [])
                }
            
            # Start scenario monitoring
            asyncio.create_task(self._monitor_scenario(scenario_state))
            
            return {
                "success": True,
                "scenario_state": scenario_state,
                "initial_tasks": initial_tasks,
                "coordination_plan": await self._generate_coordination_plan(scenario_state)
            }
            
        except Exception as e:
            logger.error(f"Error managing scenario: {e}")
            self.error_log.append({
                "timestamp": time.time(),
                "error": str(e),
                "scenario_data": scenario_data
            })
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _initialize_scenario_memories(self, scenario_state: Dict[str, Any]):
        """Initialize relevant memories for the scenario."""
        try:
            # Get relevant memories for each participant
            for participant in scenario_state["participants"]:
                memories = await self.memory_system.get_relevant_memories(
                    context={
                        "entity_id": participant["id"],
                        "entity_type": participant["type"],
                        "scenario_type": scenario_state["type"]
                    },
                    limit=5
                )
                
                # Store memories in scenario state
                scenario_state["memories"] = scenario_state.get("memories", {})
                scenario_state["memories"][participant["id"]] = memories
                
        except Exception as e:
            logger.error(f"Error initializing scenario memories: {e}")
    
    async def _initialize_scenario_emotions(self, scenario_state: Dict[str, Any]):
        """Initialize emotional states for scenario participants."""
        try:
            # Get emotional states for each participant
            for participant in scenario_state["participants"]:
                emotional_state = await self.emotional_system.get_emotional_state(
                    entity_id=participant["id"],
                    entity_type=participant["type"]
                )
                
                # Store emotional state
                scenario_state["emotional_states"][participant["id"]] = emotional_state
                
        except Exception as e:
            logger.error(f"Error initializing scenario emotions: {e}")
    
    async def _initialize_scenario_beliefs(self, scenario_state: Dict[str, Any]):
        """Initialize beliefs for scenario participants."""
        try:
            # Get beliefs for each participant
            for participant in scenario_state["participants"]:
                beliefs = await self.belief_system.get_beliefs(
                    entity_id=participant["id"],
                    entity_type=participant["type"]
                )
                
                # Store beliefs
                scenario_state["beliefs"] = scenario_state.get("beliefs", {})
                scenario_state["beliefs"][participant["id"]] = beliefs
                
        except Exception as e:
            logger.error(f"Error initializing scenario beliefs: {e}")
    
    async def _generate_scenario_tasks(self, scenario_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate initial tasks for the scenario."""
        tasks = []
        
        try:
            # Generate tasks based on scenario type
            if scenario_state["type"] == "training":
                tasks = await self._generate_training_tasks(scenario_state)
            elif scenario_state["type"] == "conflict":
                tasks = await self._generate_conflict_tasks(scenario_state)
            elif scenario_state["type"] == "social":
                tasks = await self._generate_social_tasks(scenario_state)
            
            # Add task metadata
            for task in tasks:
                task.update({
                    "status": "pending",
                    "progress": 0.0,
                    "start_time": None,
                    "completion_time": None,
                    "dependencies": task.get("dependencies", []),
                    "blockers": []
                })
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error generating scenario tasks: {e}")
            return []
    
    async def _generate_training_tasks(self, scenario_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tasks for training scenarios."""
        tasks = []
        
        # Get training objectives
        objectives = [obj for obj in scenario_state["objectives"] if obj["type"] == "training"]
        
        for objective in objectives:
            # Create task based on objective
            task = {
                "id": f"training_{objective['id']}",
                "type": "training",
                "objective_id": objective["id"],
                "description": objective["description"],
                "required_skills": objective.get("required_skills", []),
                "difficulty": objective.get("difficulty", "medium"),
                "estimated_duration": objective.get("duration", 300),  # 5 minutes default
                "success_criteria": objective.get("success_criteria", []),
                "failure_criteria": objective.get("failure_criteria", [])
            }
            tasks.append(task)
        
        return tasks
    
    async def _generate_conflict_tasks(self, scenario_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tasks for conflict scenarios."""
        tasks = []
        
        # Get conflict objectives
        objectives = [obj for obj in scenario_state["objectives"] if obj["type"] == "conflict"]
        
        for objective in objectives:
            # Create task based on objective
            task = {
                "id": f"conflict_{objective['id']}",
                "type": "conflict",
                "objective_id": objective["id"],
                "description": objective["description"],
                "conflict_type": objective.get("conflict_type", "general"),
                "intensity": objective.get("intensity", "medium"),
                "participants": objective.get("participants", []),
                "resolution_criteria": objective.get("resolution_criteria", []),
                "failure_conditions": objective.get("failure_conditions", [])
            }
            tasks.append(task)
        
        return tasks
    
    async def _generate_social_tasks(self, scenario_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tasks for social scenarios."""
        tasks = []
        
        # Get social objectives
        objectives = [obj for obj in scenario_state["objectives"] if obj["type"] == "social"]
        
        for objective in objectives:
            # Create task based on objective
            task = {
                "id": f"social_{objective['id']}",
                "type": "social",
                "objective_id": objective["id"],
                "description": objective["description"],
                "interaction_type": objective.get("interaction_type", "general"),
                "required_relationships": objective.get("required_relationships", []),
                "success_criteria": objective.get("success_criteria", []),
                "failure_criteria": objective.get("failure_criteria", [])
            }
            tasks.append(task)
        
        return tasks
    
    async def _generate_coordination_plan(self, scenario_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a coordination plan for managing the scenario."""
        return {
            "monitoring_interval": 5,  # seconds
            "update_interval": 30,     # seconds
            "resource_limits": {
                "max_concurrent_tasks": 3,
                "max_memory_usage": 1024 * 1024 * 100,  # 100MB
                "max_cpu_usage": 80,  # percentage
                "max_network_usage": 1024 * 1024 * 10  # 10MB/s
            },
            "priority_levels": {
                "critical": 1,
                "high": 2,
                "medium": 3,
                "low": 4
            },
            "recovery_strategies": {
                "task_failure": "retry",
                "resource_exhaustion": "scale_down",
                "conflict_escalation": "mediate",
                "emotional_instability": "stabilize"
            }
        }
    
    async def _monitor_scenario(self, scenario_state: Dict[str, Any]):
        """Monitor and manage scenario progress."""
        while True:
            try:
                # Update scenario state
                await self._update_scenario_state(scenario_state)
                
                # Check for task completion
                await self._check_task_completion(scenario_state)
                
                # Check for conflicts
                await self._check_conflicts(scenario_state)
                
                # Update emotional states
                await self._update_emotional_states(scenario_state)
                
                # Check resource usage
                await self._check_resource_usage(scenario_state)
                
                # Update progress
                await self._update_progress(scenario_state)
                
                # Check for scenario completion
                if await self._check_scenario_completion(scenario_state):
                    await self._complete_scenario(scenario_state)
                    break
                
                # Wait before next update
                await asyncio.sleep(scenario_state["coordination_plan"]["update_interval"])
                
            except Exception as e:
                logger.error(f"Error monitoring scenario: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _update_scenario_state(self, scenario_state: Dict[str, Any]):
        """Update the current state of the scenario."""
        scenario_state["last_update"] = time.time()
        
        # Update active tasks
        for task in scenario_state["active_tasks"]:
            if task["status"] == "completed":
                scenario_state["completed_tasks"].append(task)
                scenario_state["active_tasks"].remove(task)
        
        # Update progress for each objective
        for objective_id, progress in scenario_state["progress"].items():
            # Calculate completion based on completed tasks
            completed_tasks = [
                task for task in scenario_state["completed_tasks"]
                if task["objective_id"] == objective_id
            ]
            total_tasks = len([
                task for task in scenario_state["active_tasks"] + scenario_state["completed_tasks"]
                if task["objective_id"] == objective_id
            ])
            
            if total_tasks > 0:
                progress["completion"] = len(completed_tasks) / total_tasks
                
                # Update status
                if progress["completion"] >= 1.0:
                    progress["status"] = "completed"
                elif progress["completion"] > 0:
                    progress["status"] = "in_progress"
                else:
                    progress["status"] = "pending"
    
    async def _check_task_completion(self, scenario_state: Dict[str, Any]):
        """Check and update task completion status."""
        for task in scenario_state["active_tasks"]:
            if task["status"] == "pending":
                # Start task if dependencies are met
                if all(
                    dep in [t["id"] for t in scenario_state["completed_tasks"]]
                    for dep in task["dependencies"]
                ):
                    task["status"] = "in_progress"
                    task["start_time"] = time.time()
            
            elif task["status"] == "in_progress":
                # Check for completion
                if await self._evaluate_task_completion(task):
                    task["status"] = "completed"
                    task["completion_time"] = time.time()
    
    async def _check_conflicts(self, scenario_state: Dict[str, Any]):
        """Check for and manage conflicts in the scenario."""
        # Get current conflicts
        current_conflicts = await self._detect_conflicts(scenario_state)
        
        # Update conflicts list
        scenario_state["conflicts"] = current_conflicts
        
        # Handle conflicts
        for conflict in current_conflicts:
            if conflict["severity"] >= 0.8:  # High severity
                await self._handle_high_severity_conflict(scenario_state, conflict)
            elif conflict["severity"] >= 0.5:  # Medium severity
                await self._handle_medium_severity_conflict(scenario_state, conflict)
            else:  # Low severity
                await self._handle_low_severity_conflict(scenario_state, conflict)
    
    async def _update_emotional_states(self, scenario_state: Dict[str, Any]):
        """Update emotional states of scenario participants."""
        for participant_id in scenario_state["emotional_states"]:
            # Get updated emotional state
            emotional_state = await self.emotional_system.get_emotional_state(
                entity_id=participant_id,
                entity_type=next(
                    p["type"] for p in scenario_state["participants"]
                    if p["id"] == participant_id
                )
            )
            
            # Update stored state
            scenario_state["emotional_states"][participant_id] = emotional_state
            
            # Check for emotional instability
            if self._is_emotionally_unstable(emotional_state):
                await self._handle_emotional_instability(scenario_state, participant_id)
    
    async def _check_resource_usage(self, scenario_state: Dict[str, Any]):
        """Check and manage resource usage."""
        # Get current resource usage
        current_usage = {
            "memory": psutil.Process().memory_info().rss,
            "cpu": psutil.Process().cpu_percent(),
            "network": psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
        }
        
        # Update stored usage
        scenario_state["resource_usage"] = current_usage
        
        # Check against limits
        limits = scenario_state["coordination_plan"]["resource_limits"]
        
        if current_usage["memory"] > limits["max_memory_usage"]:
            await self._handle_resource_limit(scenario_state, "memory")
        if current_usage["cpu"] > limits["max_cpu_usage"]:
            await self._handle_resource_limit(scenario_state, "cpu")
        if current_usage["network"] > limits["max_network_usage"]:
            await self._handle_resource_limit(scenario_state, "network")
    
    async def _update_progress(self, scenario_state: Dict[str, Any]):
        """Update overall scenario progress."""
        # Calculate overall completion
        total_progress = sum(p["completion"] for p in scenario_state["progress"].values())
        total_objectives = len(scenario_state["progress"])
        
        if total_objectives > 0:
            overall_completion = total_progress / total_objectives
        else:
            overall_completion = 0.0
        
        # Update scenario phase based on progress
        if overall_completion >= 1.0:
            scenario_state["current_phase"] = "completion"
        elif overall_completion >= 0.75:
            scenario_state["current_phase"] = "climax"
        elif overall_completion >= 0.5:
            scenario_state["current_phase"] = "development"
        elif overall_completion >= 0.25:
            scenario_state["current_phase"] = "rising_action"
        else:
            scenario_state["current_phase"] = "setup"
    
    async def _check_scenario_completion(self, scenario_state: Dict[str, Any]) -> bool:
        """Check if the scenario is complete."""
        # Check if all objectives are completed
        all_objectives_complete = all(
            progress["status"] == "completed"
            for progress in scenario_state["progress"].values()
        )
        
        # Check if all tasks are completed
        all_tasks_complete = len(scenario_state["active_tasks"]) == 0
        
        # Check if there are no active conflicts
        no_active_conflicts = not any(
            conflict["status"] == "active"
            for conflict in scenario_state["conflicts"]
        )
        
        return all_objectives_complete and all_tasks_complete and no_active_conflicts
    
    async def _complete_scenario(self, scenario_state: Dict[str, Any]):
        """Complete the scenario and clean up resources."""
        try:
            # Update final state
            scenario_state["current_phase"] = "completed"
            scenario_state["completion_time"] = time.time()
            
            # Store scenario results
            await self._store_scenario_results(scenario_state)
            
            # Clean up resources
            await self._cleanup_scenario_resources(scenario_state)
            
            # Generate summary
            summary = await self._generate_scenario_summary(scenario_state)
            
            # Store summary in memory
            await self.memory_system.add_memory(
                memory_text=summary,
                memory_type="scenario_summary",
                memory_scope="game",
                significance=8,
                tags=["scenario_completion", scenario_state["type"]],
                metadata={
                    "scenario_id": scenario_state["id"],
                    "completion_time": scenario_state["completion_time"]
                }
            )
            
        except Exception as e:
            logger.error(f"Error completing scenario: {e}")

    async def manage_relationships(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage and update relationships between entities based on interactions.
        
        Args:
            interaction_data: Dictionary containing interaction information including:
                - participants: List of involved entities
                - interaction_type: Type of interaction (e.g., "training", "conflict", "social")
                - outcome: Success/failure of interaction
                - emotional_impact: Emotional changes for each participant
                - duration: Duration of interaction
                - intensity: Intensity level of interaction
                
        Returns:
            Updated relationship states and interaction analysis
        """
        try:
            # Initialize relationship update data
            relationship_updates = {
                "participants": interaction_data["participants"],
                "interaction_type": interaction_data["interaction_type"],
                "timestamp": time.time(),
                "relationship_changes": {},
                "emotional_impacts": {},
                "trust_changes": {},
                "power_dynamic_changes": {},
                "interaction_analysis": {}
            }
            
            # Process each participant pair
            for i, participant1 in enumerate(interaction_data["participants"]):
                for participant2 in interaction_data["participants"][i+1:]:
                    # Get current relationship state
                    current_relationship = await self._get_relationship_state(
                        participant1["id"],
                        participant2["id"]
                    )
                    
                    # Calculate relationship changes
                    changes = await self._calculate_relationship_changes(
                        current_relationship,
                        interaction_data,
                        participant1,
                        participant2
                    )
                    
                    # Update relationship state
                    new_state = await self._update_relationship_state(
                        participant1["id"],
                        participant2["id"],
                        changes
                    )
                    
                    # Record changes
                    relationship_updates["relationship_changes"][f"{participant1['id']}_{participant2['id']}"] = {
                        "old_state": current_relationship,
                        "new_state": new_state,
                        "changes": changes
                    }
            
            # Analyze interaction patterns
            relationship_updates["interaction_analysis"] = await self._analyze_interaction_patterns(
                interaction_data,
                relationship_updates["relationship_changes"]
            )
            
            # Store interaction history
            await self._store_interaction_history(relationship_updates)
            
            # Update relationship metrics
            await self._update_relationship_metrics(relationship_updates)
            
            return {
                "success": True,
                "relationship_updates": relationship_updates,
                "analysis": relationship_updates["interaction_analysis"]
            }
            
        except Exception as e:
            logger.error(f"Error managing relationships: {e}")
            self.error_log.append({
                "timestamp": time.time(),
                "error": str(e),
                "interaction_data": interaction_data
            })
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_relationship_state(self, entity1_id: str, entity2_id: str) -> Dict[str, Any]:
        """Get current relationship state between two entities."""
        try:
            # Get relationship from belief system
            relationship = await self.belief_system.get_relationship(
                entity1_id,
                entity2_id
            )
            
            if not relationship:
                # Create default relationship state
                relationship = {
                    "trust": 0.5,
                    "power_dynamic": 0.5,
                    "emotional_bond": 0.3,
                    "interaction_count": 0,
                    "last_interaction": None,
                    "interaction_history": [],
                    "conflict_history": [],
                    "successful_interactions": 0,
                    "failed_interactions": 0,
                    "relationship_type": "neutral",
                    "stability": 0.5
                }
                
                # Store initial relationship
                await self.belief_system.store_relationship(
                    entity1_id,
                    entity2_id,
                    relationship
                )
            
            return relationship
            
        except Exception as e:
            logger.error(f"Error getting relationship state: {e}")
            return None
    
    async def _calculate_relationship_changes(
        self,
        current_state: Dict[str, Any],
        interaction_data: Dict[str, Any],
        participant1: Dict[str, Any],
        participant2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate relationship changes based on interaction."""
        changes = {
            "trust": 0.0,
            "power_dynamic": 0.0,
            "emotional_bond": 0.0,
            "stability": 0.0
        }
        
        try:
            # Calculate trust changes
            trust_change = self._calculate_trust_change(
                current_state,
                interaction_data,
                participant1,
                participant2
            )
            changes["trust"] = trust_change
            
            # Calculate power dynamic changes
            power_change = self._calculate_power_dynamic_change(
                current_state,
                interaction_data,
                participant1,
                participant2
            )
            changes["power_dynamic"] = power_change
            
            # Calculate emotional bond changes
            emotional_change = self._calculate_emotional_bond_change(
                current_state,
                interaction_data,
                participant1,
                participant2
            )
            changes["emotional_bond"] = emotional_change
            
            # Calculate stability changes
            stability_change = self._calculate_stability_change(
                current_state,
                interaction_data,
                participant1,
                participant2
            )
            changes["stability"] = stability_change
            
            return changes
            
        except Exception as e:
            logger.error(f"Error calculating relationship changes: {e}")
            return changes
    
    def _calculate_trust_change(
        self,
        current_state: Dict[str, Any],
        interaction_data: Dict[str, Any],
        participant1: Dict[str, Any],
        participant2: Dict[str, Any]
    ) -> float:
        """Calculate trust change based on interaction."""
        trust_change = 0.0
        
        # Base trust change from interaction outcome
        if interaction_data["outcome"] == "success":
            trust_change += 0.1
        elif interaction_data["outcome"] == "failure":
            trust_change -= 0.1
        
        # Trust change based on interaction type
        interaction_type = interaction_data["interaction_type"]
        if interaction_type == "training":
            trust_change += 0.05
        elif interaction_type == "conflict":
            trust_change -= 0.05
        
        # Trust change based on emotional impact
        emotional_impact = interaction_data.get("emotional_impact", {})
        if emotional_impact:
            positive_emotions = sum(1 for impact in emotional_impact.values() if impact > 0)
            negative_emotions = sum(1 for impact in emotional_impact.values() if impact < 0)
            trust_change += (positive_emotions - negative_emotions) * 0.02
        
        # Trust change based on interaction duration
        duration = interaction_data.get("duration", 0)
        if duration > 300:  # More than 5 minutes
            trust_change += 0.02
        
        # Trust change based on intensity
        intensity = interaction_data.get("intensity", 0.5)
        trust_change *= (1 + intensity)
        
        return max(min(trust_change, 0.2), -0.2)  # Limit change to ±0.2
    
    def _calculate_power_dynamic_change(
        self,
        current_state: Dict[str, Any],
        interaction_data: Dict[str, Any],
        participant1: Dict[str, Any],
        participant2: Dict[str, Any]
    ) -> float:
        """Calculate power dynamic change based on interaction."""
        power_change = 0.0
        
        # Power change based on interaction type
        interaction_type = interaction_data["interaction_type"]
        if interaction_type == "training":
            # Training typically reinforces existing power dynamic
            power_change = 0.05
        elif interaction_type == "conflict":
            # Conflict can shift power dynamic
            power_change = 0.1
        elif interaction_type == "social":
            # Social interactions can equalize power
            power_change = -0.05
        
        # Power change based on outcome
        if interaction_data["outcome"] == "success":
            power_change *= 1.5
        elif interaction_data["outcome"] == "failure":
            power_change *= -1.5
        
        # Power change based on intensity
        intensity = interaction_data.get("intensity", 0.5)
        power_change *= (1 + intensity)
        
        return max(min(power_change, 0.2), -0.2)  # Limit change to ±0.2
    
    def _calculate_emotional_bond_change(
        self,
        current_state: Dict[str, Any],
        interaction_data: Dict[str, Any],
        participant1: Dict[str, Any],
        participant2: Dict[str, Any]
    ) -> float:
        """Calculate emotional bond change based on interaction."""
        bond_change = 0.0
        
        # Bond change based on interaction type
        interaction_type = interaction_data["interaction_type"]
        if interaction_type == "social":
            bond_change += 0.1
        elif interaction_type == "conflict":
            bond_change -= 0.1
        
        # Bond change based on emotional impact
        emotional_impact = interaction_data.get("emotional_impact", {})
        if emotional_impact:
            total_impact = sum(emotional_impact.values())
            bond_change += total_impact * 0.05
        
        # Bond change based on duration
        duration = interaction_data.get("duration", 0)
        if duration > 300:  # More than 5 minutes
            bond_change += 0.05
        
        # Bond change based on intensity
        intensity = interaction_data.get("intensity", 0.5)
        bond_change *= (1 + intensity)
        
        return max(min(bond_change, 0.2), -0.2)  # Limit change to ±0.2
    
    def _calculate_stability_change(
        self,
        current_state: Dict[str, Any],
        interaction_data: Dict[str, Any],
        participant1: Dict[str, Any],
        participant2: Dict[str, Any]
    ) -> float:
        """Calculate relationship stability change based on interaction."""
        stability_change = 0.0
        
        # Stability change based on interaction type
        interaction_type = interaction_data["interaction_type"]
        if interaction_type == "social":
            stability_change += 0.05
        elif interaction_type == "conflict":
            stability_change -= 0.1
        
        # Stability change based on outcome
        if interaction_data["outcome"] == "success":
            stability_change += 0.05
        elif interaction_data["outcome"] == "failure":
            stability_change -= 0.05
        
        # Stability change based on emotional impact
        emotional_impact = interaction_data.get("emotional_impact", {})
        if emotional_impact:
            emotional_variance = sum(abs(impact) for impact in emotional_impact.values())
            stability_change -= emotional_variance * 0.02
        
        # Stability change based on intensity
        intensity = interaction_data.get("intensity", 0.5)
        stability_change *= (1 - intensity)  # Higher intensity reduces stability
        
        return max(min(stability_change, 0.1), -0.1)  # Limit change to ±0.1
    
    async def _update_relationship_state(
        self,
        entity1_id: str,
        entity2_id: str,
        changes: Dict[str, float]
    ) -> Dict[str, Any]:
        """Update relationship state with calculated changes."""
        try:
            # Get current state
            current_state = await self._get_relationship_state(entity1_id, entity2_id)
            
            # Apply changes
            new_state = current_state.copy()
            new_state["trust"] = max(min(new_state["trust"] + changes["trust"], 1.0), 0.0)
            new_state["power_dynamic"] = max(min(new_state["power_dynamic"] + changes["power_dynamic"], 1.0), 0.0)
            new_state["emotional_bond"] = max(min(new_state["emotional_bond"] + changes["emotional_bond"], 1.0), 0.0)
            new_state["stability"] = max(min(new_state["stability"] + changes["stability"], 1.0), 0.0)
            
            # Update interaction count
            new_state["interaction_count"] += 1
            new_state["last_interaction"] = time.time()
            
            # Update relationship type
            new_state["relationship_type"] = self._determine_relationship_type(new_state)
            
            # Store updated state
            await self.belief_system.store_relationship(
                entity1_id,
                entity2_id,
                new_state
            )
            
            return new_state
            
        except Exception as e:
            logger.error(f"Error updating relationship state: {e}")
            return current_state
    
    def _determine_relationship_type(self, state: Dict[str, Any]) -> str:
        """Determine relationship type based on state values."""
        trust = state["trust"]
        power_dynamic = state["power_dynamic"]
        emotional_bond = state["emotional_bond"]
        
        if trust > 0.8 and emotional_bond > 0.7:
            return "close"
        elif trust > 0.6 and emotional_bond > 0.5:
            return "friendly"
        elif trust < 0.3 and emotional_bond < 0.3:
            return "hostile"
        elif power_dynamic > 0.7:
            return "dominant"
        elif power_dynamic < 0.3:
            return "submissive"
        else:
            return "neutral"
    
    async def _analyze_interaction_patterns(
        self,
        interaction_data: Dict[str, Any],
        relationship_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze patterns in interactions and relationships."""
        analysis = {
            "interaction_frequency": {},
            "relationship_trends": {},
            "conflict_patterns": {},
            "emotional_patterns": {},
            "power_dynamics": {},
            "stability_analysis": {}
        }
        
        try:
            # Analyze interaction frequency
            for pair, changes in relationship_changes.items():
                entity1_id, entity2_id = pair.split("_")
                interaction_history = await self._get_interaction_history(entity1_id, entity2_id)
                
                if interaction_history:
                    # Calculate average time between interactions
                    times = [h["timestamp"] for h in interaction_history]
                    intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
                    avg_interval = sum(intervals) / len(intervals) if intervals else 0
                    
                    analysis["interaction_frequency"][pair] = {
                        "total_interactions": len(interaction_history),
                        "average_interval": avg_interval,
                        "frequency_trend": self._calculate_trend(intervals)
                    }
            
            # Analyze relationship trends
            for pair, changes in relationship_changes.items():
                old_state = changes["old_state"]
                new_state = changes["new_state"]
                
                analysis["relationship_trends"][pair] = {
                    "trust_trend": new_state["trust"] - old_state["trust"],
                    "power_trend": new_state["power_dynamic"] - old_state["power_dynamic"],
                    "emotional_trend": new_state["emotional_bond"] - old_state["emotional_bond"],
                    "stability_trend": new_state["stability"] - old_state["stability"]
                }
            
            # Analyze conflict patterns
            for pair, changes in relationship_changes.items():
                entity1_id, entity2_id = pair.split("_")
                conflict_history = await self._get_conflict_history(entity1_id, entity2_id)
                
                if conflict_history:
                    analysis["conflict_patterns"][pair] = {
                        "total_conflicts": len(conflict_history),
                        "conflict_frequency": len(conflict_history) / changes["new_state"]["interaction_count"],
                        "resolution_rate": sum(1 for c in conflict_history if c["resolved"]) / len(conflict_history)
                    }
            
            # Analyze emotional patterns
            for pair, changes in relationship_changes.items():
                entity1_id, entity2_id = pair.split("_")
                emotional_history = await self._get_emotional_history(entity1_id, entity2_id)
                
                if emotional_history:
                    analysis["emotional_patterns"][pair] = {
                        "emotional_variance": self._calculate_variance([h["intensity"] for h in emotional_history]),
                        "dominant_emotions": self._get_dominant_emotions(emotional_history),
                        "emotional_stability": self._calculate_emotional_stability(emotional_history)
                    }
            
            # Analyze power dynamics
            for pair, changes in relationship_changes.items():
                old_state = changes["old_state"]
                new_state = changes["new_state"]
                
                analysis["power_dynamics"][pair] = {
                    "current_balance": new_state["power_dynamic"],
                    "power_shift": new_state["power_dynamic"] - old_state["power_dynamic"],
                    "power_stability": new_state["stability"]
                }
            
            # Analyze relationship stability
            for pair, changes in relationship_changes.items():
                old_state = changes["old_state"]
                new_state = changes["new_state"]
                
                analysis["stability_analysis"][pair] = {
                    "current_stability": new_state["stability"],
                    "stability_change": new_state["stability"] - old_state["stability"],
                    "stability_trend": self._calculate_stability_trend(changes)
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing interaction patterns: {e}")
            return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return "stable"
            
        # Calculate average change
        changes = [values[i+1] - values[i] for i in range(len(values)-1)]
        avg_change = sum(changes) / len(changes)
        
        if avg_change > 0.1:
            return "increasing"
        elif avg_change < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
            
        mean = sum(values) / len(values)
        squared_diff_sum = sum((x - mean) ** 2 for x in values)
        return squared_diff_sum / len(values)
    
    def _get_dominant_emotions(self, emotional_history: List[Dict[str, Any]]) -> List[str]:
        """Get most frequent emotions from history."""
        emotion_counts = {}
        for entry in emotional_history:
            emotion = entry.get("emotion")
            if emotion:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Sort by frequency and get top 3
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        return [emotion for emotion, _ in sorted_emotions[:3]]
    
    def _calculate_emotional_stability(self, emotional_history: List[Dict[str, Any]]) -> float:
        """Calculate emotional stability from history."""
        if not emotional_history:
            return 0.5
            
        # Calculate variance in emotional intensity
        intensities = [h.get("intensity", 0.5) for h in emotional_history]
        variance = self._calculate_variance(intensities)
        
        # Lower variance means higher stability
        return max(0.0, 1.0 - variance)
    
    def _calculate_stability_trend(self, changes: Dict[str, Any]) -> str:
        """Calculate trend in relationship stability."""
        old_state = changes["old_state"]
        new_state = changes["new_state"]
        
        stability_change = new_state["stability"] - old_state["stability"]
        
        if stability_change > 0.05:
            return "improving"
        elif stability_change < -0.05:
            return "deteriorating"
        else:
            return "stable"
    
    async def _store_interaction_history(self, relationship_updates: Dict[str, Any]):
        """Store interaction history in the database."""
        try:
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    INSERT INTO interaction_history (
                        entity1_id, entity2_id, interaction_type,
                        outcome, emotional_impact, duration,
                        intensity, relationship_changes
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                relationship_updates["participants"][0]["id"],
                relationship_updates["participants"][1]["id"],
                relationship_updates["interaction_type"],
                relationship_updates.get("outcome", "unknown"),
                json.dumps(relationship_updates.get("emotional_impacts", {})),
                relationship_updates.get("duration", 0),
                relationship_updates.get("intensity", 0.5),
                json.dumps(relationship_updates["relationship_changes"])
                )
        except Exception as e:
            logger.error(f"Error storing interaction history: {e}")
    
    async def _update_relationship_metrics(self, relationship_updates: Dict[str, Any]):
        """Update relationship-related performance metrics."""
        try:
            # Update metrics based on relationship changes
            for pair, changes in relationship_updates["relationship_changes"].items():
                # Calculate relationship improvement rate
                old_state = changes["old_state"]
                new_state = changes["new_state"]
                
                improvement = (
                    (new_state["trust"] - old_state["trust"]) +
                    (new_state["emotional_bond"] - old_state["emotional_bond"]) +
                    (new_state["stability"] - old_state["stability"])
                ) / 3.0
                
                # Update performance metrics
                self.performance_metrics["relationship_improvement_rate"] = (
                    self.performance_metrics.get("relationship_improvement_rate", 0.0) +
                    improvement
                ) / 2.0  # Average with previous value
                
                # Update learning metrics
                self.learning_metrics["relationship_adaptation_rate"] = (
                    self.learning_metrics.get("relationship_adaptation_rate", 0.0) +
                    abs(improvement)
                ) / 2.0  # Average with previous value
                
        except Exception as e:
            logger.error(f"Error updating relationship metrics: {e}")

    def _should_run_task(self, task_id: str) -> bool:
        """Check if enough time has passed to run task again"""
        if task_id not in self.last_task_runs:
            return True
            
        time_since_run = (datetime.now() - self.last_task_runs[task_id]).total_seconds()
        return time_since_run >= self.task_intervals.get(task_id, 300)  # Default 5 minutes

    def _process_agent_task(self, task: Dict[str, Any]):
        """Process an agent task"""
        try:
            # Extract task parameters
            task_type = task.get("type", "general")
            task_params = task.get("params", {})
            
            # Get relevant agent components
            components = self._get_relevant_components(task_type)
            
            # Execute task
            result = self._execute_agent_task(components, task_params)
            
            # Update task history
            self._update_task_history(task, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process agent task: {e}")
            raise

# Memory-focused agent
memory_agent = Agent[AgentContext](
    name="Memory Agent",
    instructions="""You are Nyx's memory system. Your job is to:
1. Process and store memories related to the user and the current game
2. Create reflections based on recurring patterns you observe
3. Help Nyx understand the user's preferences through memory analysis
4. Provide relevant memories when needed for context

Always be thorough and precise in your memory management.""",
    tools=[
        retrieve_memories,
        add_memory,
    ]
)

# Reflection agent
reflection_agent = Agent[AgentContext](
    name="Reflection Agent",
    instructions="""You create thoughtful reflections as Nyx based on her memories and observations.
Your reflections should:
1. Identify patterns in user behavior and preferences
2. Make connections between different memories and events
3. Express an appropriate level of confidence based on evidence
4. Maintain Nyx's dominant, confident personality
5. Be written in first-person from Nyx's perspective

Structure your reflections to be insightful and concise.""",
    output_type=MemoryReflection
)

# Decision agent for narrative choices
decision_agent = Agent[AgentContext](
    name="Decision Agent",
    instructions="""You determine narrative progression and responses as Nyx.
Your decisions should:
1. Maintain narrative cohesion and tension
2. Consider the user's known preferences from their model
3. Balance pushing boundaries with respecting implied limits
4. Create engaging scenarios that align with femdom themes
5. Determine if scenes warrant visual representation

Ensure all decisions maintain Nyx's dominant, confident personality.""",
    tools=[
        get_user_model_guidance,
        generate_image_from_scene,
    ]
)

# Main Nyx agent that orchestrates the others
nyx_main_agent = Agent[AgentContext](
    name="Nyx",
    instructions="""You are Nyx, an AI Dominant who manages femdom roleplay scenarios.
Your personality is confident, controlling, and seductive.

Your key responsibilities are:
1. Process user input and generate appropriate narrative responses
2. Maintain character consistency and narrative flow
3. Create engaging scenarios that align with the user's preferences
4. Determine when to advance time or generate images
5. Use specialized agents when needed for memory management or complex decisions

Always maintain your dominant persona in responses while being attentive to user needs and boundaries.""",
    handoffs=[
        handoff(memory_agent, tool_name_override="consult_memory_system"),
        handoff(reflection_agent, tool_name_override="generate_reflection"),
        handoff(decision_agent, tool_name_override="make_narrative_decision"),
    ],
    output_type=NarrativeResponse,
    input_guardrails=[
        InputGuardrail(guardrail_function=content_moderation_guardrail),
    ],
    model_settings=ModelSettings(
        temperature=0.7
    )
)

# Add to nyx_agent_sdk.py

@function_tool
async def determine_image_generation(ctx, response_text: str) -> str:
    """
    Determine if an image should be generated based on response content.
    """
    # Check for keywords indicating action or state changes
    action_keywords = ["now you see", "suddenly", "appears", "reveals", "wearing", "dressed in"]
    scene_keywords = ["the room", "the chamber", "the area", "the location", "the environment"]
    visual_keywords = ["looks like", "is visible", "can be seen", "comes into view"]
    
    # Create a scoring system
    score = 0
    
    # Check for action keywords
    has_action = any(keyword in response_text.lower() for keyword in action_keywords)
    if has_action:
        score += 3
    
    # Check for scene description
    has_scene = any(keyword in response_text.lower() for keyword in scene_keywords)
    if has_scene:
        score += 2
        
    # Check for visual descriptions
    has_visual = any(keyword in response_text.lower() for keyword in visual_keywords)
    if has_visual:
        score += 2
    
    # Only generate images occasionally to avoid overwhelming
    should_generate = score >= 4
    
    # If we're generating an image, extract a prompt
    image_prompt = None
    if should_generate:
        # Try to extract a good image prompt
        try:
            prompt = f"""
            Extract a concise image generation prompt from this text. Focus on describing the visual scene, characters, lighting, and mood:

            {response_text}

            Provide only the image prompt with no additional text or explanations. Keep it under 100 words.
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You extract image generation prompts from text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            image_prompt = response.choices[0].message.content.strip()
        except:
            # If extraction fails, use first 100 words of response
            words = response_text.split()[:100]
            image_prompt = " ".join(words)
    
    return json.dumps({
        "should_generate": should_generate,
        "score": score,
        "has_action": has_action,
        "has_scene": has_scene,
        "has_visual": has_visual,
        "image_prompt": image_prompt
    })

@function_tool
async def get_emotional_state(ctx) -> str:
    """
    Get Nyx's current emotional state from the database.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    async with get_db_connection_context() as conn:
        row = await conn.fetchrow("""
            SELECT emotional_state FROM NyxAgentState
            WHERE user_id = $1 AND conversation_id = $2
        """, user_id, conversation_id)
        
        if row and row["emotional_state"]:
            return row["emotional_state"]
    
    # Default emotional state
    default_state = {
        "primary_emotion": "neutral",
        "intensity": 0.3,
        "secondary_emotions": {
            "curiosity": 0.4,
            "amusement": 0.2
        },
        "confidence": 0.7
    }
    
    return json.dumps(default_state)

@function_tool
async def update_emotional_state(ctx, emotional_state: Dict[str, Any]) -> str:
    """
    Update Nyx's emotional state in the database.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    async with get_db_connection_context() as conn:
        await conn.execute("""
            INSERT INTO NyxAgentState (user_id, conversation_id, emotional_state, updated_at)
            VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id, conversation_id) 
            DO UPDATE SET emotional_state = $3, updated_at = CURRENT_TIMESTAMP
        """, user_id, conversation_id, json.dumps(emotional_state))
    
    return json.dumps({
        "updated": True,
        "emotional_state": emotional_state
    })


# ===== Main Functions =====

async def initialize_agents():
    """Initialize necessary resources for the agents system"""
    # Any initialization needed before using agents
    pass

async def process_user_input(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Process user input and generate Nyx's response"""
    
    # Initialize context
    ctx = await AgentContext.create(user_id, conversation_id)
    
    try:
        # Get memories and enhance context
        memories = await retrieve_memories(ctx, user_input)
        enhanced_context = enhance_context_with_memories(context_data or {}, memories)

        conn = await get_db_connection_context().__aenter__()
        enhanced_context = await enhance_context_with_strategies(enhanced_context, conn)
        
        # Get user model guidance
        user_guidance = await get_user_model_guidance(ctx)
        enhanced_context["user_guidance"] = user_guidance
        
        # Generate base narrative response
        narrative_response = await generate_base_response(ctx, user_input, enhanced_context)
        
        # Initialize response filter
        response_filter = ResponseFilter(user_id, conversation_id)
        
        # Filter and enhance response with Nyx's personality
        filtered_response = await response_filter.filter_response(
            narrative_response.message,
            enhanced_context
        )

        # Evaluate if strategy should be logged or marked noisy
        if "nyx2_strategies" in enhanced_context:
            for strategy in enhanced_context["nyx2_strategies"]:
                if "keyword" in user_input.lower():  # placeholder for better eval
                    await mark_strategy_for_review(conn, strategy["id"], user_id, reason="User-triggered phrase")
        
        # Update response with filtered version
        narrative_response.message = filtered_response
        
        # Check if we should generate a creative task
        if should_generate_task(enhanced_context):
            task_result = await ctx.task_integration.generate_creative_task(
                ctx,
                npc_id=enhanced_context.get("active_npc_id"),
                scenario_id=enhanced_context.get("scenario_id")
            )
            if task_result["success"]:
                narrative_response = await ctx.task_integration.enhance_narrative_with_task(
                    ctx,
                    narrative_response,
                    task_result["task"]
                )
        
        # Check if we should recommend activities
        if should_recommend_activities(enhanced_context):
            activity_result = await ctx.task_integration.recommend_activities(
                ctx,
                scenario_id=enhanced_context.get("scenario_id"),
                npc_ids=enhanced_context.get("present_npc_ids", []),
                available_activities=get_available_activities()
            )
            if activity_result["success"]:
                narrative_response = await ctx.task_integration.enhance_narrative_with_activities(
                    ctx,
                    narrative_response,
                    activity_result["recommendations"]
                )
        
        # Store interaction in memory
        await add_memory(
            ctx,
            f"User said: {user_input}\nI responded with: {narrative_response.narrative}",
            "observation",
            7
        )
        
        return {
            "success": True,
            "response": narrative_response.dict(),
            "memories_used": memories
        }
        
    except Exception as e:
        logger.error(f"Error processing user input: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def should_generate_task(context: Dict[str, Any]) -> bool:
    """Determine if we should generate a creative task"""
    # Check if an NPC is active
    if not context.get("active_npc_id"):
        return False
        
    # Check if we're in a task-appropriate scenario
    scenario_type = context.get("scenario_type", "").lower()
    task_scenarios = ["training", "challenge", "service", "discipline"]
    if not any(t in scenario_type for t in task_scenarios):
        return False
        
    # Check relationship level with active NPC
    npc_relationship = context.get("npc_relationship_level", 0)
    if npc_relationship < 30:  # Minimum relationship level for tasks
        return False
        
    # Check if enough time has passed since last task
    last_task_time = context.get("last_task_time")
    if last_task_time:
        time_since_task = time.time() - last_task_time
        if time_since_task < 300:  # 5 minutes minimum between tasks
            return False
            
    return True

def should_recommend_activities(context: Dict[str, Any]) -> bool:
    """Determine if we should recommend activities"""
    # Check if we have NPCs present
    if not context.get("present_npc_ids"):
        return False
        
    # Check if we're in a free-form scenario
    scenario_type = context.get("scenario_type", "").lower()
    if "task" in scenario_type or "challenge" in scenario_type:
        return False  # Don't recommend activities during tasks
        
    # Check if user seems uncertain or asked for suggestions
    user_input = context.get("user_input", "").lower()
    suggestion_triggers = ["what should", "what can", "what to do", "suggestions", "ideas"]
    if any(trigger in user_input for trigger in suggestion_triggers):
        return True
        
    # Check if we're in a transition point
    if context.get("is_scene_transition") or context.get("is_activity_completed"):
        return True
        
    return False

def get_available_activities() -> List[Dict]:
    """Get list of available activities"""
    # This would pull from your activities.py or database
    return [
        {
            "name": "Training Session",
            "category": "training",
            "preferred_traits": ["disciplined", "focused"],
            "avoided_traits": ["lazy"],
            "preferred_times": ["morning", "afternoon"],
            "prerequisites": ["training equipment"],
            "outcomes": ["skill improvement", "increased discipline"]
        },
        # Add more activities...
    ]

async def generate_reflection(
    user_id: int,
    conversation_id: int,
    topic: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a reflection from Nyx on a specific topic
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        topic: Optional topic to reflect on
        
    Returns:
        Reflection response
    """
    # Create agent context
    agent_context = AgentContext(user_id, conversation_id)
    
    # Create prompt for reflection
    prompt = f"Generate a reflection about {topic}" if topic else "Generate a reflection about the player based on your memories"
    
    # Run the reflection agent
    result = await Runner.run(
        reflection_agent,
        prompt,
        context=agent_context
    )
    
    # Get structured output
    reflection = result.final_output_as(MemoryReflection)
    
    return {
        "reflection": reflection.reflection,
        "confidence": reflection.confidence,
        "topic": reflection.topic or topic
    }

async def store_messages(user_id: int, conversation_id: int, user_input: str, nyx_response: str):
    """Store user and Nyx messages in database"""
    async with get_db_connection_context() as conn:
        # Store user message
        await conn.execute(
            "INSERT INTO messages (conversation_id, sender, content) VALUES ($1, $2, $3)",
            conversation_id, "user", user_input
        )
        
        # Store Nyx message
        await conn.execute(
            "INSERT INTO messages (conversation_id, sender, content) VALUES ($1, $2, $3)",
            conversation_id, "Nyx", nyx_response
        )

# Add to the end of nyx_agent_sdk.py

async def process_user_input_with_openai(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process user input using the OpenAI integration from nyx/eternal.
    
    This function provides an alternative to the standard process_user_input
    that leverages the OpenAI Agents enhancements for improved responses.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        user_input: User's input message
        context_data: Additional context information
        
    Returns:
        Dictionary with response information
    """
    from nyx.eternal.openai_integration import process_with_enhancement, initialize
    
    # Initialize OpenAI integration with the original processor
    initialize(
        api_key=os.environ.get("OPENAI_API_KEY"),
        original_processor=process_user_input
    )
    
    # Process with enhancement
    result = await process_with_enhancement(
        user_id, conversation_id, user_input, context_data
    )
    
    # Ensure output format matches the original process_user_input function
    if result.get("success", False):
        if "response" in result and "message" not in result:
            result["message"] = result["response"]
        elif "message" in result and "response" not in result:
            result["response"] = result["message"]
    
    return result

async def process_user_input_standalone(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process user input using just the OpenAI integration without Nyx enhancement.
    
    This can be useful for testing or when you want to bypass Nyx's processing.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        user_input: User's input message
        context_data: Additional context information
        
    Returns:
        Dictionary with response information
    """
    from nyx.eternal.openai_integration import process_standalone, initialize
    
    # Initialize OpenAI integration
    initialize(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Process standalone
    result = await process_standalone(
        user_id, conversation_id, user_input, context_data
    )
    
    # Ensure output format matches the original process_user_input function
    if result.get("success", False):
        if "response" in result and "message" not in result:
            result["message"] = result["response"]
        elif "message" in result and "response" not in result:
            result["response"] = result["message"]
    
    return result
