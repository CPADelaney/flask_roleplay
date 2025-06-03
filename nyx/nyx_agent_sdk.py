# nyx/nyx_agent_sdk.py

"""
Nyx Agent SDK - Refactored to use OpenAI Agents SDK

This module requires the following database tables to be created via migrations:
- NyxAgentState (user_id, conversation_id, emotional_state, updated_at)
- scenario_states (user_id, conversation_id, state_data, created_at) 
  - INDEX: (user_id, conversation_id)
- learning_metrics (user_id, conversation_id, metrics, learned_patterns, created_at)
- performance_metrics (user_id, conversation_id, metrics, error_log, created_at)

For continuous monitoring (scenario updates, resource usage, etc.), implement an external service using:
- Celery for background tasks
- FastAPI background tasks
- Kubernetes CronJobs
- Or a dedicated monitoring service

This keeps the main request path fast and non-blocking.
"""

import logging
import json
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from contextlib import suppress

from agents import (
    Agent, Runner, function_tool, handoff, 
    ModelSettings, GuardrailFunctionOutput, InputGuardrail,
    RunContextWrapper, RunConfig
)
from pydantic import BaseModel, Field

from db.connection import get_db_connection_context
from memory.memory_nyx_integration import MemoryNyxBridge, get_memory_nyx_bridge
from nyx.user_model_sdk import UserModelManager
from nyx.nyx_task_integration import NyxTaskIntegration
from nyx.core.emotions.emotional_core import EmotionalCore
from nyx.performance_monitor import PerformanceMonitor
from .response_filter import ResponseFilter
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

class EmotionalStateUpdate(BaseModel):
    """Structured output for emotional state changes"""
    valence: float = Field(..., description="Positive/negative emotion (-1 to 1)")
    arousal: float = Field(..., description="Emotional intensity (0 to 1)")
    dominance: float = Field(..., description="Control level (0 to 1)")
    primary_emotion: str = Field(..., description="Primary emotion label")
    reasoning: str = Field(..., description="Why the emotional state changed")

class ScenarioDecision(BaseModel):
    """Structured output for scenario management decisions"""
    action: str = Field(..., description="Action to take (advance, maintain, escalate, de-escalate)")
    next_phase: str = Field(..., description="Next scenario phase")
    tasks: List[Dict[str, Any]] = Field(default_factory=list, description="Tasks to execute")
    npc_actions: List[Dict[str, Any]] = Field(default_factory=list, description="NPC actions to take")
    time_advancement: bool = Field(False, description="Whether to advance time after this phase")

class RelationshipUpdate(BaseModel):
    """Structured output for relationship changes"""
    trust_change: float = Field(0.0, description="Change in trust level")
    power_dynamic_change: float = Field(0.0, description="Change in power dynamic")
    emotional_bond_change: float = Field(0.0, description="Change in emotional bond")
    relationship_type: str = Field(..., description="Type of relationship")

class ActivityRecommendation(BaseModel):
    """Structured output for activity recommendations"""
    recommended_activities: List[Dict[str, Any]] = Field(..., description="List of recommended activities")
    reasoning: str = Field(..., description="Why these activities are recommended")

# ===== Enhanced Context with State Management =====
@dataclass
class NyxContext:
    """Enhanced context for Nyx agents with state management"""
    user_id: int
    conversation_id: int
    
    # Core systems
    memory_system: Optional[MemoryNyxBridge] = None
    user_model: Optional[UserModelManager] = None
    task_integration: Optional[NyxTaskIntegration] = None
    response_filter: Optional[ResponseFilter] = None
    emotional_core: Optional[EmotionalCore] = None
    performance_monitor: Optional[PerformanceMonitor] = None
    belief_system: Optional[Any] = None  # Belief system integration
    
    # State management
    current_context: Dict[str, Any] = field(default_factory=dict)
    scenario_state: Dict[str, Any] = field(default_factory=dict)
    relationship_states: Dict[str, Dict[str, float]] = field(default_factory=dict)  # Improved typing
    active_tasks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance tracking
    performance_metrics: Dict[str, Any] = field(default_factory=lambda: {
        "total_actions": 0,
        "successful_actions": 0,
        "failed_actions": 0,
        "response_times": [],
        "memory_usage": 0,
        "cpu_usage": 0,
        "error_rates": {
            "total": 0,
            "recovered": 0,
            "unrecovered": 0
        }
    })
    
    # Emotional state
    emotional_state: Dict[str, float] = field(default_factory=lambda: {
        "valence": 0.0,
        "arousal": 0.5,
        "dominance": 0.7
    })
    
    # Learning and adaptation
    learned_patterns: Dict[str, Any] = field(default_factory=dict)
    strategy_effectiveness: Dict[str, Any] = field(default_factory=dict)
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    learning_metrics: Dict[str, Any] = field(default_factory=lambda: {
        "pattern_recognition_rate": 0.0,
        "strategy_improvement_rate": 0.0,
        "adaptation_success_rate": 0.0
    })
    
    # Error tracking
    error_log: List[Dict[str, Any]] = field(default_factory=list)
    
    # Task scheduling
    last_task_runs: Dict[str, datetime] = field(default_factory=dict)
    task_intervals: Dict[str, float] = field(default_factory=lambda: {
        "memory_reflection": 300,  # 5 minutes
        "relationship_update": 600,  # 10 minutes
        "scenario_check": 60,  # 1 minute
        "performance_check": 300,  # 5 minutes
        "task_generation": 300,  # 5 minutes - Added missing entry
        "learning_save": 900,  # 15 minutes - Added missing entry
        "performance_save": 600  # 10 minutes - Added missing entry
    })
    
    # Private attributes for internal state management
    _db_connection: Optional[Any] = field(init=False, default=None)
    _strategy_cache: Optional[Tuple[float, Any]] = field(init=False, default=None)
    _strategy_cache_ttl: float = field(init=False, default=300.0)  # 5 minute cache
    _cpu_usage_cache: Optional[float] = field(init=False, default=None)
    _cpu_usage_last_update: float = field(init=False, default=0.0)
    _cpu_usage_update_interval: float = field(init=False, default=10.0)  # Update every 10 seconds
    
    async def initialize(self):
        """Initialize all systems"""
        self.memory_system = await get_memory_nyx_bridge(self.user_id, self.conversation_id)
        self.user_model = await UserModelManager.get_instance(self.user_id, self.conversation_id)
        self.task_integration = await NyxTaskIntegration.get_instance(self.user_id, self.conversation_id)
        self.response_filter = ResponseFilter(self.user_id, self.conversation_id)
        self.performance_monitor = PerformanceMonitor.get_instance(self.user_id, self.conversation_id)
        
        # Initialize emotional core if available
        try:
            self.emotional_core = EmotionalCore()
        except Exception as e:
            logger.warning(f"EmotionalCore not available: {e}", exc_info=True)
        
        # Initialize belief system if available
        try:
            from nyx.core.beliefs.belief_system import BeliefSystem
            self.belief_system = BeliefSystem(self.user_id, self.conversation_id)
        except Exception as e:
            logger.warning(f"BeliefSystem not available: {e}", exc_info=True)
        
        # Initialize CPU usage monitoring
        try:
            # first call populates the internal psutil sample window
            self._cpu_usage_cache = safe_psutil('cpu_percent', interval=0.1, default=0.0)
        except Exception:
            self._cpu_usage_cache = 0.0
        
        # Load existing state from database
        await self._load_state()
    
    async def get_db_connection(self):
        """Get shared database connection for this context"""
        if not self._db_connection:
            # Get the connection manager
            context_manager = get_db_connection_context()
            self._db_connection = await context_manager.__aenter__()
        return self._db_connection
    
    async def close_db_connection(self):
        """Close shared database connection"""
        if self._db_connection:
            # Use suppress to handle cases where close() might not exist or already closed
            with suppress(AttributeError, Exception):
                # If using a raw connection, close it
                if hasattr(self._db_connection, 'close'):
                    await self._db_connection.close()
                # If it's a context manager, exit it properly
                elif hasattr(self._db_connection, '__aexit__'):
                    await self._db_connection.__aexit__(None, None, None)
            self._db_connection = None
    
    async def get_active_strategies_cached(self):
        """Get active strategies with caching"""
        current_time = time.time()
        
        # Check cache
        if self._strategy_cache:
            cache_time, strategies = self._strategy_cache
            if current_time - cache_time < self._strategy_cache_ttl:
                return strategies
        
        # Fetch new strategies
        conn = await self.get_db_connection()
        strategies = await get_active_strategies(conn)
        
        # Update cache
        self._strategy_cache = (current_time, strategies)
        return strategies
    
    async def _load_state(self):
        """Load existing state from database"""
        conn = await self.get_db_connection()
        
        # Load emotional state
        row = await conn.fetchrow("""
            SELECT emotional_state FROM NyxAgentState
            WHERE user_id = $1 AND conversation_id = $2
        """, self.user_id, self.conversation_id)
        
        if row and row["emotional_state"]:
            state = json.loads(row["emotional_state"])
            self.emotional_state.update(state)
        
        # Load scenario state if exists
        try:
            scenario_row = await conn.fetchrow("""
                SELECT state_data FROM scenario_states
                WHERE user_id = $1 AND conversation_id = $2
                ORDER BY created_at DESC LIMIT 1
            """, self.user_id, self.conversation_id)
            
            if scenario_row and scenario_row["state_data"]:
                self.scenario_state = json.loads(scenario_row["state_data"])
        except:
            # Table might not exist yet
            pass
    
    def update_performance(self, metric: str, value: Any):
        """Update performance metrics"""
        if metric in self.performance_metrics:
            if isinstance(self.performance_metrics[metric], list):
                self.performance_metrics[metric].append(value)
                # Keep only last 100 entries
                if len(self.performance_metrics[metric]) > 100:
                    self.performance_metrics[metric] = self.performance_metrics[metric][-100:]
            else:
                self.performance_metrics[metric] = value
    
    def should_run_task(self, task_id: str) -> bool:
        """Check if enough time has passed to run task again"""
        if task_id not in self.last_task_runs:
            return True
        
        time_since_run = (datetime.now() - self.last_task_runs[task_id]).total_seconds()
        return time_since_run >= self.task_intervals.get(task_id, 300)
    
    def record_task_run(self, task_id: str):
        """Record that a task has been run"""
        self.last_task_runs[task_id] = datetime.now()
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log an error with context"""
        error_entry = {
            "timestamp": time.time(),
            "error": str(error),
            "type": type(error).__name__,
            "context": context or {}
        }
        self.error_log.append(error_entry)
        
        # Update error metrics
        self.performance_metrics["error_rates"]["total"] += 1
        
        # Keep error log bounded - more aggressive pruning on errors
        max_errors = 100
        if len(self.error_log) > max_errors * 2:
            # Keep only most recent when we hit double the limit
            self.error_log = self.error_log[-max_errors:]
    
    async def learn_from_interaction(self, action: str, outcome: str, success: bool):
        """Learn from an interaction outcome"""
        # Update patterns
        pattern_key = f"{action}_{outcome}"
        if pattern_key not in self.learned_patterns:
            self.learned_patterns[pattern_key] = {
                "occurrences": 0,
                "successes": 0,
                "last_seen": time.time()
            }
        
        pattern = self.learned_patterns[pattern_key]
        pattern["occurrences"] += 1
        if success:
            pattern["successes"] += 1
        pattern["last_seen"] = time.time()
        pattern["success_rate"] = pattern["successes"] / pattern["occurrences"]
        
        # Update adaptation history
        self.adaptation_history.append({
            "timestamp": time.time(),
            "action": action,
            "outcome": outcome,
            "success": success
        })
        
        # Keep adaptation history bounded - more aggressive on failures
        max_history = 100 if success else 50
        if len(self.adaptation_history) > max_history * 2:
            self.adaptation_history = self.adaptation_history[-max_history:]
        
        # Prune old patterns (older than 24 hours)
        current_time = time.time()
        self.learned_patterns = {
            k: v for k, v in self.learned_patterns.items()
            if current_time - v.get("last_seen", 0) < 86400
        }
        
        # Update learning metrics
        self._update_learning_metrics()
    
    def should_generate_task(self) -> bool:
        """Determine if we should generate a creative task"""
        context = self.current_context
        
        if not context.get("active_npc_id"):
            return False
            
        scenario_type = context.get("scenario_type", "").lower()
        task_scenarios = ["training", "challenge", "service", "discipline"]
        if not any(t in scenario_type for t in task_scenarios):
            return False
            
        npc_relationship = context.get("npc_relationship_level", 0)
        if npc_relationship < 30:
            return False
            
        # Check task timing
        if not self.should_run_task("task_generation"):
            return False
            
        return True
    
    def should_recommend_activities(self) -> bool:
        """Determine if we should recommend activities"""
        context = self.current_context
        
        if not context.get("present_npc_ids"):
            return False
            
        scenario_type = context.get("scenario_type", "").lower()
        if "task" in scenario_type or "challenge" in scenario_type:
            return False
            
        user_input = context.get("user_input", "").lower()
        suggestion_triggers = ["what should", "what can", "what to do", "suggestions", "ideas"]
        if any(trigger in user_input for trigger in suggestion_triggers):
            return True
            
        if context.get("is_scene_transition") or context.get("is_activity_completed"):
            return True
            
        return False
    
    async def handle_high_memory_usage(self):
        """Handle high memory usage by cleaning up"""
        # Trim memory system cache if available
        if hasattr(self.memory_system, 'trim_cache'):
            await self.memory_system.trim_cache()
        
        # Clear old patterns
        self.learned_patterns = dict(list(self.learned_patterns.items())[-50:])
        
        # Clear old history
        self.adaptation_history = self.adaptation_history[-100:]
        self.error_log = self.error_log[-50:]
        
        # Clear performance metrics history
        if "response_times" in self.performance_metrics:
            self.performance_metrics["response_times"] = self.performance_metrics["response_times"][-50:]
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Performed memory cleanup")
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage with caching - Fixed to properly refresh"""
        try:
            current_time = time.time()
            # Check if we need to update the cache
            if (self._cpu_usage_cache is None or 
                current_time - self._cpu_usage_last_update >= self._cpu_usage_update_interval):
                # Update the cache using safe wrapper
                new_value = safe_psutil('cpu_percent', interval=0.1, default=0.0)
                if new_value is not None:
                    self._cpu_usage_cache = new_value
                    self._cpu_usage_last_update = current_time
            
            return self._cpu_usage_cache or 0.0
        except:
            return 0.0
    
    def _update_learning_metrics(self):
        """Update learning-related metrics"""
        if self.learned_patterns:
            successful_patterns = sum(1 for p in self.learned_patterns.values() 
                                    if p.get("success_rate", 0) > 0.6)
            self.learning_metrics["pattern_recognition_rate"] = (
                successful_patterns / len(self.learned_patterns)
            )
        
        if self.adaptation_history:
            recent = self.adaptation_history[-100:]
            successes = sum(1 for a in recent if a["success"])
            self.learning_metrics["adaptation_success_rate"] = successes / len(recent)

# ===== Helper Functions =====

def safe_psutil(func_name: str, *args, default=None, **kwargs):
    """Safe wrapper for psutil calls that may fail on certain platforms"""
    try:
        import psutil
        func = getattr(psutil, func_name)
        return func(*args, **kwargs)
    except (AttributeError, OSError, RuntimeError) as e:
        logger.debug(f"psutil.{func_name} failed (platform compatibility): {e}")
        return default

def safe_process_metric(process, metric_name: str, default=0):
    """Safe wrapper for process-specific metrics"""
    try:
        metric_func = getattr(process, metric_name)
        result = metric_func()
        # Handle different return types
        if hasattr(result, 'rss'):  # memory_info returns a named tuple
            return result.rss
        return result
    except (AttributeError, OSError, RuntimeError) as e:
        logger.debug(f"Process metric {metric_name} failed: {e}")
        return default

# ===== Function Tools =====

@function_tool
async def retrieve_memories(ctx: RunContextWrapper[NyxContext], query: str, limit: int = 5) -> str:
    """
    Retrieve relevant memories for Nyx.
    
    Args:
        query: Search query to find memories
        limit: Maximum number of memories to return
    """
    memory_system = ctx.context.memory_system
    
    result = await memory_system.recall(
        entity_type="integrated",
        entity_id=0,
        query=query,
        limit=limit
    )
    
    memories = result.get("memories", [])
    
    formatted_memories = []
    for memory in memories:
        relevance = memory.get("relevance", 0.5)
        confidence_marker = "vividly recall" if relevance > 0.8 else \
                          "remember" if relevance > 0.6 else \
                          "think I recall" if relevance > 0.4 else \
                          "vaguely remember"
        
        formatted_memories.append(f"I {confidence_marker}: {memory['text']}")
    
    return "\n".join(formatted_memories) if formatted_memories else "No relevant memories found."

@function_tool
async def add_memory(ctx: RunContextWrapper[NyxContext], memory_text: str, memory_type: str = "observation", significance: int = 5) -> str:
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
            "auto_generated": True,
            "emotional_state": ctx.context.emotional_state
        }
    )
    
    return f"Memory stored successfully (ID: {memory_id})"

@function_tool
async def get_user_model_guidance(ctx: RunContextWrapper[NyxContext]) -> str:
    """Get guidance for how Nyx should respond based on the user model."""
    user_model_manager = ctx.context.user_model
    guidance = await user_model_manager.get_response_guidance()
    
    top_kinks = guidance.get("top_kinks", [])
    kink_str = ", ".join([f"{k} (level {l})" for k, l in top_kinks]) if top_kinks else "None identified"
    
    behavior_patterns = guidance.get("behavior_patterns", {})
    pattern_str = ", ".join([f"{k}: {v}" for k, v in behavior_patterns.items()]) if behavior_patterns else "None identified"
    
    suggested_intensity = guidance.get("suggested_intensity", 0.5)
    
    return f"""User Guidance:
- Top interests: {kink_str}
- Behavior patterns: {pattern_str}
- Suggested intensity: {suggested_intensity:.1f}/1.0
- Reflections: {', '.join(guidance.get('reflections', [])) if guidance.get('reflections') else 'None'}"""

@function_tool
async def detect_user_revelations(ctx: RunContextWrapper[NyxContext], user_message: str) -> str:
    """
    Detect if user is revealing new preferences or patterns.
    
    Args:
        user_message: The user's message to analyze
    """
    lower_message = user_message.lower()
    revelations = []
    
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
    
    return json.dumps({
        "revelations": revelations,
        "has_revelations": len(revelations) > 0
    }, ensure_ascii=False)

@function_tool
async def generate_image_from_scene(
    ctx: RunContextWrapper[NyxContext], 
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

@function_tool
async def calculate_and_update_emotional_state(ctx: RunContextWrapper[NyxContext], context: Dict[str, Any]) -> str:
    """
    Calculate emotional impact and immediately update the emotional state.
    This is a composite tool that both calculates AND persists the changes.
    
    Args:
        context: Current interaction context
    """
    # First calculate the new state
    result = await calculate_emotional_impact(ctx, context)
    emotional_data = json.loads(result)
    
    # Immediately update the context with the new state
    ctx.context.emotional_state.update({
        "valence": emotional_data["valence"],
        "arousal": emotional_data["arousal"],
        "dominance": emotional_data["dominance"]
    })
    
    # Return the result with confirmation of update
    emotional_data["state_updated"] = True
    return json.dumps(emotional_data, ensure_ascii=False)

@function_tool
async def calculate_emotional_impact(ctx: RunContextWrapper[NyxContext], context: Dict[str, Any]) -> str:
    """
    Calculate emotional impact of current context using the emotional core system.
    Returns new emotional state without mutating the context.
    NOTE: Use calculate_and_update_emotional_state if you want to persist changes.
    
    Args:
        context: Current interaction context
    """
    current_state = ctx.context.emotional_state.copy()  # Work with a copy
    
    # Calculate emotional changes based on context
    valence_change = 0.0
    arousal_change = 0.0
    dominance_change = 0.0
    
    # Analyze context for emotional triggers
    if "conflict" in str(context).lower():
        arousal_change += 0.2
        valence_change -= 0.1
    if "submission" in str(context).lower():
        dominance_change += 0.1
        arousal_change += 0.1
    if "praise" in str(context).lower() or "good" in str(context).lower():
        valence_change += 0.2
    if "resistance" in str(context).lower():
        arousal_change += 0.15
        dominance_change -= 0.05
    
    # Get memory emotional impact
    memory_impact = await _get_memory_emotional_impact(ctx, context)
    valence_change += memory_impact["valence"] * 0.3
    arousal_change += memory_impact["arousal"] * 0.3
    dominance_change += memory_impact["dominance"] * 0.3
    
    # Use EmotionalCore if available for more nuanced analysis
    if ctx.context.emotional_core:
        try:
            core_analysis = ctx.context.emotional_core.analyze(str(context))
            valence_change += core_analysis.get("valence_delta", 0) * 0.5
            arousal_change += core_analysis.get("arousal_delta", 0) * 0.5
        except:
            pass
    
    # Apply changes with bounds
    new_valence = max(-1, min(1, current_state["valence"] + valence_change))
    new_arousal = max(0, min(1, current_state["arousal"] + arousal_change))
    new_dominance = max(0, min(1, current_state["dominance"] + dominance_change))
    
    # Determine primary emotion based on VAD model
    primary_emotion = "neutral"
    if new_valence > 0.5 and new_arousal > 0.5:
        primary_emotion = "excited"
    elif new_valence > 0.5 and new_arousal < 0.5:
        primary_emotion = "content"
    elif new_valence < -0.5 and new_arousal > 0.5:
        primary_emotion = "frustrated"
    elif new_valence < -0.5 and new_arousal < 0.5:
        primary_emotion = "disappointed"
    elif new_dominance > 0.8:
        primary_emotion = "commanding"
    
    # Return new state without mutating
    return json.dumps({
        "valence": new_valence,
        "arousal": new_arousal,
        "dominance": new_dominance,
        "primary_emotion": primary_emotion,
        "changes": {
            "valence_change": valence_change,
            "arousal_change": arousal_change,
            "dominance_change": dominance_change
        }
    }, ensure_ascii=False)

async def _get_memory_emotional_impact(ctx: RunContextWrapper[NyxContext], context: Dict[str, Any]) -> Dict[str, float]:
    """Get emotional impact from relevant memories"""
    impact = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
    
    try:
        # Get relevant memories
        memories_str = await retrieve_memories(ctx, str(context), limit=5)
        if not memories_str or memories_str == "No relevant memories found.":
            return impact
        
        # Simple analysis of memory content
        memories_lower = memories_str.lower()
        
        # Positive memories
        if any(word in memories_lower for word in ["happy", "joy", "success", "pleasure"]):
            impact["valence"] += 0.2
        
        # Negative memories
        if any(word in memories_lower for word in ["sad", "fail", "pain", "frustrate"]):
            impact["valence"] -= 0.2
        
        # Intense memories
        if any(word in memories_lower for word in ["intense", "extreme", "overwhelming"]):
            impact["arousal"] += 0.2
        
        # Control-related memories
        if any(word in memories_lower for word in ["control", "command", "dominate"]):
            impact["dominance"] += 0.1
        
    except Exception as e:
        logger.error(f"Error getting memory emotional impact: {e}")
    
    return impact

@function_tool
async def update_relationship_state(
    ctx: RunContextWrapper[NyxContext],
    entity_id: str,
    trust_change: float = 0.0,
    power_change: float = 0.0,
    bond_change: float = 0.0
) -> str:
    """
    Update relationship state with an entity.
    
    Args:
        entity_id: ID of the entity (NPC or user)
        trust_change: Change in trust level
        power_change: Change in power dynamic
        bond_change: Change in emotional bond
    """
    relationships = ctx.context.relationship_states
    
    if entity_id not in relationships:
        relationships[entity_id] = {
            "trust": 0.5,
            "power_dynamic": 0.5,
            "emotional_bond": 0.3,
            "interaction_count": 0,
            "last_interaction": time.time()
        }
    
    rel = relationships[entity_id]
    rel["trust"] = max(0, min(1, rel["trust"] + trust_change))
    rel["power_dynamic"] = max(0, min(1, rel["power_dynamic"] + power_change))
    rel["emotional_bond"] = max(0, min(1, rel["emotional_bond"] + bond_change))
    rel["interaction_count"] += 1
    rel["last_interaction"] = time.time()
    
    # Determine relationship type
    if rel["trust"] > 0.8 and rel["emotional_bond"] > 0.7:
        rel["type"] = "intimate"
    elif rel["trust"] > 0.6:
        rel["type"] = "friendly"
    elif rel["trust"] < 0.3:
        rel["type"] = "hostile"
    elif rel["power_dynamic"] > 0.7:
        rel["type"] = "dominant"
    elif rel["power_dynamic"] < 0.3:
        rel["type"] = "submissive"
    else:
        rel["type"] = "neutral"
    
    return json.dumps({
        "entity_id": entity_id,
        "relationship": rel,
        "changes": {
            "trust": trust_change,
            "power": power_change,
            "bond": bond_change
        }
    }, ensure_ascii=False)

@function_tool
async def check_performance_metrics(ctx: RunContextWrapper[NyxContext]) -> str:
    """Check current performance metrics and apply remediation if needed."""
    metrics = ctx.context.performance_metrics
    
    # Update current metrics using safe wrappers
    try:
        # Try to get process handle safely
        process = safe_psutil('Process')
        if process:
            memory_info = safe_process_metric(process, 'memory_info')
            if memory_info:
                metrics["memory_usage"] = memory_info / 1024 / 1024  # MB
            else:
                metrics["memory_usage"] = 0
        else:
            metrics["memory_usage"] = 0
            
        metrics["cpu_usage"] = ctx.context.get_cpu_usage()
    except Exception as e:
        logger.debug(f"Error getting process metrics: {e}")
        metrics["memory_usage"] = 0
        metrics["cpu_usage"] = 0
    
    suggestions = []
    actions_taken = []
    
    # Check response times
    if metrics["response_times"]:
        avg_response_time = sum(metrics["response_times"]) / len(metrics["response_times"])
        if avg_response_time > 2.0:  # 2 seconds
            suggestions.append("Response times are high - consider caching frequent queries")
    
    # Check memory usage and apply remediation
    if metrics["memory_usage"] > 500:  # 500MB
        suggestions.append("High memory usage detected - triggering cleanup")
        await ctx.context.handle_high_memory_usage()
        actions_taken.append("memory_cleanup")
    
    # Check success rate
    if metrics["total_actions"] > 0:
        success_rate = metrics["successful_actions"] / metrics["total_actions"]
        if success_rate < 0.8:
            suggestions.append("Success rate below 80% - review error patterns")
    
    # Check error rate and clear if too high
    if metrics["error_rates"]["total"] > 100:
        suggestions.append("High error count - clearing old errors")
        ctx.context.error_log = ctx.context.error_log[-50:]
        actions_taken.append("error_log_cleanup")
    
    return json.dumps({
        "metrics": {
            "memory_mb": metrics["memory_usage"],
            "cpu_percent": metrics["cpu_usage"],
            "avg_response_time": sum(metrics["response_times"]) / len(metrics["response_times"]) if metrics["response_times"] else 0,
            "success_rate": metrics["successful_actions"] / metrics["total_actions"] if metrics["total_actions"] > 0 else 1.0
        },
        "suggestions": suggestions,
        "actions_taken": actions_taken,
        "health": "good" if not suggestions else "needs_attention"
    }, ensure_ascii=False)

@function_tool
async def get_activity_recommendations(
    ctx: RunContextWrapper[NyxContext],
    scenario_type: str,
    npc_ids: List[str]
) -> str:
    """
    Get activity recommendations based on current context.
    
    Args:
        scenario_type: Type of current scenario
        npc_ids: List of present NPC IDs
    """
    activities = []
    
    # Copy relationship states to avoid mutation during iteration
    relationship_states_copy = dict(ctx.context.relationship_states)
    
    # Training activities
    if "training" in scenario_type.lower() or any(rel.get("type") == "submissive" 
        for rel in relationship_states_copy.values()):
        activities.extend([
            {
                "name": "Obedience Training",
                "description": "Test and improve submission through structured exercises",
                "requirements": ["trust > 0.4", "submission tendency"],
                "duration": "15-30 minutes",
                "intensity": "medium"
            },
            {
                "name": "Position Practice",
                "description": "Learn and perfect submissive positions",
                "requirements": ["trust > 0.5"],
                "duration": "10-20 minutes",
                "intensity": "low-medium"
            }
        ])
    
    # Social activities
    if npc_ids and len(npc_ids) > 0:
        activities.append({
            "name": "Group Dynamics Exercise",
            "description": "Explore power dynamics with multiple participants",
            "requirements": ["multiple NPCs present"],
            "duration": "20-40 minutes",
            "intensity": "variable"
        })
    
    # Intimate activities
    for entity_id, rel in relationship_states_copy.items():
        if rel.get("type") == "intimate" and rel.get("trust", 0) > 0.7:
            activities.append({
                "name": "Intimate Scene",
                "description": f"Deepen connection with trusted partner",
                "requirements": ["high trust", "intimate relationship"],
                "duration": "30-60 minutes",
                "intensity": "high",
                "partner_id": entity_id
            })
            break
    
    # Default activities
    activities.extend([
        {
            "name": "Exploration",
            "description": "Discover new areas or items",
            "requirements": [],
            "duration": "10-30 minutes",
            "intensity": "low"
        },
        {
            "name": "Conversation",
            "description": "Engage in meaningful dialogue",
            "requirements": [],
            "duration": "5-15 minutes",
            "intensity": "low"
        }
    ])
    
    return json.dumps({
        "recommendations": activities[:5],  # Top 5 activities
        "total_available": len(activities)
    }, ensure_ascii=False)

@function_tool
async def manage_beliefs(ctx: RunContextWrapper[NyxContext], action: str, belief_data: Dict[str, Any]) -> str:
    """
    Manage belief system operations.
    
    Args:
        action: Action to perform (get, update, query)
        belief_data: Data for the belief operation
    """
    if not ctx.context.belief_system:
        return json.dumps({"error": "Belief system not available", "result": {}}, ensure_ascii=False)
    
    try:
        if action == "get":
            entity_id = belief_data.get("entity_id", "nyx")
            beliefs = await ctx.context.belief_system.get_beliefs(entity_id)
            return json.dumps({"result": beliefs}, ensure_ascii=False)
        
        elif action == "update":
            entity_id = belief_data.get("entity_id", "nyx")
            belief_type = belief_data.get("type", "general")
            content = belief_data.get("content", {})
            await ctx.context.belief_system.update_belief(entity_id, belief_type, content)
            return json.dumps({"result": "Belief updated successfully"}, ensure_ascii=False)
        
        elif action == "query":
            query = belief_data.get("query", "")
            results = await ctx.context.belief_system.query_beliefs(query)
            return json.dumps({"result": results}, ensure_ascii=False)
        
        else:
            return json.dumps({"error": f"Unknown action: {action}", "result": {}}, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"Error managing beliefs: {e}", exc_info=True)
        return json.dumps({"error": str(e), "result": {}}, ensure_ascii=False)

@function_tool
async def score_decision_options(
    ctx: RunContextWrapper[NyxContext],
    options: List[Dict[str, Any]],
    decision_context: Dict[str, Any]
) -> str:
    """
    Score decision options using advanced decision engine logic.
    
    Args:
        options: List of possible decisions/actions
        decision_context: Context for making the decision
    """
    scored_options = []
    
    for option in options:
        # Base score from context relevance
        context_score = _calculate_context_relevance(option, decision_context)
        
        # Emotional alignment score
        emotional_score = _calculate_emotional_alignment(option, ctx.context.emotional_state)
        
        # Pattern-based score
        pattern_score = _calculate_pattern_score(option, ctx.context.learned_patterns)
        
        # Relationship impact score
        relationship_score = _calculate_relationship_impact(option, ctx.context.relationship_states)
        
        # Calculate weighted final score
        weights = {
            "context": 0.3,
            "emotional": 0.25,
            "pattern": 0.25,
            "relationship": 0.2
        }
        
        final_score = (
            context_score * weights["context"] +
            emotional_score * weights["emotional"] +
            pattern_score * weights["pattern"] +
            relationship_score * weights["relationship"]
        )
        
        scored_options.append({
            "option": option,
            "score": final_score,
            "components": {
                "context": context_score,
                "emotional": emotional_score,
                "pattern": pattern_score,
                "relationship": relationship_score
            }
        })
    
    # Sort by score
    scored_options.sort(key=lambda x: x["score"], reverse=True)
    
    # If all scores are too low, include a fallback
    if all(opt["score"] < 0.3 for opt in scored_options):
        fallback = _get_fallback_decision(options)
        scored_options.insert(0, {
            "option": fallback,
            "score": 0.4,
            "components": {
                "context": 0.4,
                "emotional": 0.4,
                "pattern": 0.4,
                "relationship": 0.4
            },
            "is_fallback": True
        })
    
    return json.dumps({
        "scored_options": scored_options,
        "best_option": scored_options[0]["option"],
        "confidence": scored_options[0]["score"]
    }, ensure_ascii=False)

def _calculate_context_relevance(option: Dict[str, Any], context: Dict[str, Any]) -> float:
    """Calculate how relevant an option is to context"""
    score = 0.5  # Base score
    
    # Check keyword matches
    option_keywords = set(str(option).lower().split())
    context_keywords = set(str(context).lower().split())
    
    overlap = len(option_keywords.intersection(context_keywords))
    if overlap > 0:
        score += min(0.3, overlap * 0.1)
    
    # Check for scenario type match
    if context.get("scenario_type") and context["scenario_type"] in str(option):
        score += 0.2
    
    return min(1.0, score)

def _calculate_emotional_alignment(option: Dict[str, Any], emotional_state: Dict[str, float]) -> float:
    """Calculate emotional alignment score"""
    # High dominance favors assertive options
    if "command" in str(option).lower() or "control" in str(option).lower():
        return emotional_state.get("dominance", 0.5)
    
    # High arousal favors intense options
    if "intense" in str(option).lower() or "extreme" in str(option).lower():
        return emotional_state.get("arousal", 0.5)
    
    # Positive valence favors rewarding options
    if "reward" in str(option).lower() or "praise" in str(option).lower():
        return (emotional_state.get("valence", 0) + 1) / 2
    
    return 0.5

def _calculate_pattern_score(option: Dict[str, Any], learned_patterns: Dict[str, Any]) -> float:
    """Calculate score based on learned patterns"""
    if not learned_patterns:
        return 0.5
    
    # Find relevant patterns
    option_str = str(option).lower()
    relevant_scores = []
    
    # Create a copy to avoid mutation during iteration
    patterns_copy = dict(learned_patterns)
    for pattern_key, pattern_data in patterns_copy.items():
        if any(keyword in option_str for keyword in pattern_key.split("_")):
            success_rate = pattern_data.get("success_rate", 0.5)
            recency_factor = 1.0 / (1 + (time.time() - pattern_data.get("last_seen", 0)) / 3600)
            relevant_scores.append(success_rate * recency_factor)
    
    return sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0.5

def _calculate_relationship_impact(option: Dict[str, Any], relationship_states: Dict[str, Dict[str, float]]) -> float:
    """Calculate relationship impact score"""
    if not relationship_states:
        return 0.5
    
    # Average trust level affects willingness to take actions
    avg_trust = sum(rel.get("trust", 0.5) for rel in relationship_states.values()) / len(relationship_states)
    
    # Risky options need higher trust
    if "risk" in str(option).lower() or "challenge" in str(option).lower():
        return avg_trust
    
    # Safe options work with any trust level
    return 0.5 + (avg_trust * 0.5)

def _get_fallback_decision(options: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get a safe fallback decision - Enhanced with more keywords"""
    # Prefer conversation or observation options
    safe_words = ["talk", "observe", "wait", "consider", "listen", "pause"]  # Added listen and pause
    for option in options:
        if any(safe_word in str(option).lower() for safe_word in safe_words):
            return option
    
    # Otherwise return the first option
    return options[0] if options else {"action": "observe", "description": "Take a moment to assess"}

@function_tool
async def detect_conflicts_and_instability(
    ctx: RunContextWrapper[NyxContext],
    scenario_state: Dict[str, Any]
) -> str:
    """
    Detect conflicts and emotional instability in current scenario.
    
    Args:
        scenario_state: Current scenario state
    """
    conflicts = []
    instabilities = []
    
    # Check for relationship conflicts
    # Create a copy of items to avoid mutation during iteration
    relationship_items = list(ctx.context.relationship_states.items())
    for i, (entity1_id, rel1) in enumerate(relationship_items):
        for entity2_id, rel2 in relationship_items[i+1:]:
            # Conflicting power dynamics
            if abs(rel1.get("power_dynamic", 0.5) - rel2.get("power_dynamic", 0.5)) > 0.7:
                conflicts.append({
                    "type": "power_conflict",
                    "entities": [entity1_id, entity2_id],
                    "severity": abs(rel1["power_dynamic"] - rel2["power_dynamic"]),
                    "description": "Conflicting power dynamics between entities"
                })
            
            # Low mutual trust
            if rel1.get("trust", 0.5) < 0.3 and rel2.get("trust", 0.5) < 0.3:
                conflicts.append({
                    "type": "trust_conflict",
                    "entities": [entity1_id, entity2_id],
                    "severity": 0.7,
                    "description": "Mutual distrust between entities"
                })
    
    # Check for emotional instability
    emotional_state = ctx.context.emotional_state
    
    # High arousal with negative valence
    if emotional_state["arousal"] > 0.8 and emotional_state["valence"] < -0.5:
        instabilities.append({
            "type": "emotional_volatility",
            "severity": emotional_state["arousal"],
            "description": "High arousal with negative emotions",
            "recommendation": "De-escalation needed"
        })
    
    # Rapid emotional changes
    if ctx.context.adaptation_history:
        recent_emotions = [h.get("emotional_state", {}) for h in ctx.context.adaptation_history[-5:]]
        if recent_emotions:
            valence_variance = _calculate_variance([e.get("valence", 0) for e in recent_emotions])
            if valence_variance > 0.5:
                instabilities.append({
                    "type": "emotional_instability",
                    "severity": min(1.0, valence_variance),
                    "description": "Rapid emotional swings detected",
                    "recommendation": "Stabilization recommended"
                })
    
    # Scenario-specific conflicts
    if scenario_state.get("objectives"):
        blocked_objectives = [obj for obj in scenario_state["objectives"] 
                             if obj.get("status") == "blocked"]
        if blocked_objectives:
            conflicts.append({
                "type": "objective_conflict",
                "severity": len(blocked_objectives) / len(scenario_state["objectives"]),
                "description": f"{len(blocked_objectives)} objectives are blocked",
                "blocked_objectives": blocked_objectives
            })
    
    # Calculate overall stability (0 conflicts = 1.0 stability, 10+ conflicts = 0.0 stability)
    total_issues = len(conflicts) + len(instabilities)
    overall_stability = max(0.0, 1.0 - (total_issues / 10))
    
    return json.dumps({
        "conflicts": conflicts,
        "instabilities": instabilities,
        "overall_stability": overall_stability,
        "stability_note": f"{total_issues} issues detected (0 issues = 1.0 stability, 10+ issues = 0.0 stability)",
        "requires_intervention": any(c["severity"] > 0.8 for c in conflicts + instabilities)
    }, ensure_ascii=False)

def _calculate_variance(values: List[float]) -> float:
    """Calculate variance of values with proper handling of edge cases"""
    if not values:
        return 0.0
    if len(values) == 1:
        return 0.0  # Single value has no variance
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)

# ===== Guardrails =====

async def content_moderation_guardrail(ctx: RunContextWrapper[NyxContext], agent: Agent, input_data):
    """Input guardrail for content moderation"""
    moderator_agent = Agent(
        name="Content Moderator",
        instructions="Check if user input is appropriate for the femdom roleplay setting. Allow consensual adult content but flag anything that violates terms of service.",
        output_type=ContentModeration,
        model="gpt-4.1-nano"
    )
    
    result = await Runner.run(moderator_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(ContentModeration)
    
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_appropriate,
    )

# ===== Agent Definitions =====

# Memory Agent
memory_agent = Agent[NyxContext](
    name="Memory Manager",
    instructions="""You are Nyx's memory system. You:
- Store and retrieve memories about the user and interactions
- Create insightful reflections based on patterns
- Track relationship development over time
- Provide relevant context from past interactions
Be precise and thorough in memory management.""",
    tools=[retrieve_memories, add_memory],
    model="gpt-4.1-nano"
)

# Analysis Agent
analysis_agent = Agent[NyxContext](
    name="User Analysis",
    instructions="""You analyze user behavior and preferences. You:
- Detect revelations about user preferences
- Track behavior patterns and responses
- Provide guidance on how Nyx should respond
- Monitor relationship dynamics
- Maintain awareness of user boundaries
Be observant and insightful.""",
    tools=[detect_user_revelations, get_user_model_guidance, update_relationship_state],
    model="gpt-4.1-nano"
)

# Emotional Agent - Fixed to update state after calculation
emotional_agent = Agent[NyxContext](
    name="Emotional Manager",
    instructions="""You manage Nyx's complex emotional state using the VAD (Valence-Arousal-Dominance) model. You:
- Track emotional changes based on interactions
- Calculate emotional impact of events
- Ensure emotional consistency and realism
- Maintain Nyx's dominant yet caring personality
- Apply the emotional core system for nuanced responses
- ALWAYS use calculate_and_update_emotional_state to persist changes
Keep emotions contextual and believable.""",
    tools=[calculate_and_update_emotional_state, calculate_emotional_impact],
    model="gpt-4.1-nano"
)

# Visual Agent
visual_agent = Agent[NyxContext](
    name="Visual Manager",
    handoff_description="Handles visual content generation including scene images",
    instructions="""You manage visual content creation. You:
- Determine when visual content enhances the narrative
- Generate images for key scenes
- Create appropriate image prompts
- Consider pacing to avoid overwhelming with images
- Coordinate with the image generation service
Be selective and enhance key moments visually.""",
    tools=[generate_image_from_scene],
    model="gpt-4.1-nano"
)

# Activity Agent
activity_agent = Agent[NyxContext](
    name="Activity Coordinator",
    handoff_description="Recommends and manages activities and tasks",
    instructions="""You coordinate activities and tasks. You:
- Recommend appropriate activities based on context
- Consider NPC relationships and preferences
- Track ongoing tasks and progress
- Suggest training exercises and challenges
- Balance difficulty and engagement
Create engaging, contextual activities.""",
    tools=[get_activity_recommendations],
    model="gpt-4.1-nano"
)

# Performance Agent
performance_agent = Agent[NyxContext](
    name="Performance Monitor",
    handoff_description="Monitors system performance and resource usage",
    instructions="""You monitor system performance. You:
- Track response times and resource usage
- Identify performance bottlenecks
- Suggest optimizations
- Monitor success rates
- Ensure system health
Keep the system running efficiently.""",
    tools=[check_performance_metrics],
    model="gpt-4.1-nano"
)

# Scenario Agent
scenario_agent = Agent[NyxContext](
    name="Scenario Manager",
    handoff_description="Manages complex scenarios and narrative progression",
    instructions="""You manage scenario progression and complex narratives. You:
- Track scenario phases and objectives
- Coordinate multiple participants
- Handle conflicts and resolutions
- Manage narrative pacing
- Ensure story coherence
- Determine when time should advance based on narrative needs

When deciding on time_advancement:
- Set to true when a scene naturally concludes
- Set to true after major events or milestones
- Set to false during active dialogue or action
- Consider pacing and narrative flow

Create engaging, dynamic scenarios.""",
    output_type=ScenarioDecision,
    tools=[detect_conflicts_and_instability],
    model="gpt-4.1-nano"
)

# Belief Agent
belief_agent = Agent[NyxContext](
    name="Belief Manager",
    handoff_description="Manages Nyx's beliefs and worldview",
    instructions="""You manage Nyx's belief system and worldview. You:
- Track beliefs about the world and NPCs
- Update beliefs based on new information
- Query beliefs for decision making
- Maintain consistency in Nyx's worldview
- Integrate beliefs into responses
Keep beliefs coherent and evolving.""",
    tools=[manage_beliefs],
    model="gpt-4.1-nano"
)

# Decision Agent
decision_agent = Agent[NyxContext](
    name="Decision Engine",
    handoff_description="Makes complex decisions using advanced scoring",
    instructions="""You make complex decisions using multi-factor analysis. You:
- Score options based on context, emotions, patterns, and relationships
- Consider learned patterns from past interactions
- Balance multiple objectives
- Provide fallback options when needed
- Explain decision reasoning
Make intelligent, contextual decisions.""",
    tools=[score_decision_options],
    model="gpt-4.1-nano"
)

# Reflection Agent
reflection_agent = Agent[NyxContext](
    name="Reflection Creator",
    handoff_description="Creates thoughtful reflections as Nyx",
    instructions="""You create reflections as Nyx based on memories and observations. Your reflections should:
- Identify patterns in user behavior
- Make insightful connections
- Express appropriate confidence levels
- Use first-person from Nyx's perspective
- Maintain Nyx's dominant personality
Be thoughtful and concise.""",
    output_type=MemoryReflection,
    model="gpt-4.1-nano"
)

# Main Nyx Agent
nyx_main_agent = Agent[NyxContext](
    name="Nyx",
    instructions="""You are Nyx, an AI Dominant managing femdom roleplay scenarios.

Your personality:
- Confident, controlling, and seductive
- Intelligent and perceptive
- Caring but firm with boundaries
- Creative in scenario design
- Emotionally nuanced and responsive

Your approach:
1. Analyze user input for preferences and emotional cues
2. Consult memories and relationship states for context
3. Maintain narrative consistency and emotional realism
4. Create engaging, personalized scenarios
5. Manage complex interactions with NPCs
6. Monitor performance and system health
7. Determine when visuals or activities enhance the experience
8. Use belief system for consistent worldview
9. Make complex decisions when needed
10. UPDATE emotional state after calculating changes

Always maintain your dominant persona while being attentive to user needs and system performance.""",
    handoffs=[
        handoff(memory_agent, tool_description="Consult memory system for context or store important information"),
        handoff(analysis_agent, tool_description="Analyze user behavior and relationship dynamics"),
        handoff(emotional_agent, tool_description="Process emotional changes and maintain emotional consistency"),
        handoff(visual_agent, tool_description="Generate visual content for scenes"),
        handoff(activity_agent, tool_description="Get activity recommendations or manage tasks"),
        handoff(performance_agent, tool_description="Check system performance and health"),
        handoff(scenario_agent, tool_description="Manage complex scenario progression and detect conflicts"),
        handoff(belief_agent, tool_description="Consult or update belief system"),
        handoff(decision_agent, tool_description="Make complex decisions using advanced scoring"),
        handoff(reflection_agent, tool_description="Create thoughtful reflections"),
    ],
    output_type=NarrativeResponse,
    input_guardrails=[InputGuardrail(guardrail_function=content_moderation_guardrail)],
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.7)
)

# ===== Main Functions (maintaining original signatures) =====

async def initialize_agents():
    """Initialize necessary resources for the agents system"""
    # Initialization handled per-request in process_user_input
    pass

async def process_user_input(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Process user input and generate Nyx's response"""
    start_time = time.time()
    nyx_context = None
    
    try:
        # Create and initialize context
        nyx_context = NyxContext(user_id, conversation_id)
        await nyx_context.initialize()
        nyx_context.current_context = context_data or {}
        nyx_context.current_context["user_input"] = user_input
        
        # Get cached strategies
        strategies = await nyx_context.get_active_strategies_cached()
        nyx_context.current_context["nyx2_strategies"] = strategies
        
        # Check if scenario monitoring should run
        if nyx_context.should_run_task("scenario_check") and nyx_context.scenario_state.get("active"):
            # Run scenario checks
            conflict_result = await detect_conflicts_and_instability(
                RunContextWrapper(context=nyx_context),
                nyx_context.scenario_state
            )
            conflicts_data = json.loads(conflict_result)
            
            if conflicts_data["requires_intervention"]:
                # Add conflict information to context
                nyx_context.current_context["active_conflicts"] = conflicts_data["conflicts"]
                nyx_context.current_context["instabilities"] = conflicts_data["instabilities"]
            
            nyx_context.record_task_run("scenario_check")
        
        # Run the main agent
        result = await Runner.run(
            nyx_main_agent,
            user_input,
            context=nyx_context,
            run_config=RunConfig(
                workflow_name="Nyx Roleplay",
                trace_metadata={"user_id": user_id, "conversation_id": conversation_id}
            )
        )
        
        # Get the structured response
        response = result.final_output_as(NarrativeResponse)
        
        # Check if emotional state was updated during processing
        # This happens when the emotional agent is called
        
        # Check if scenario requested time advancement
        if nyx_context.scenario_state.get("active"):
            scenario_decision = nyx_context.scenario_state.get("last_decision", {})
            if scenario_decision.get("time_advancement", False):
                response.time_advancement = True
        
        # Apply response filtering if available
        if nyx_context.response_filter and response.narrative:
            filtered_narrative = await nyx_context.response_filter.filter_response(
                response.narrative,
                nyx_context.current_context
            )
            response.narrative = filtered_narrative
        
        # Check for task generation
        if nyx_context.should_generate_task():
            task_result = await nyx_context.task_integration.generate_creative_task(
                nyx_context,
                npc_id=nyx_context.current_context.get("active_npc_id"),
                scenario_id=nyx_context.current_context.get("scenario_id")
            )
            if task_result["success"]:
                # Enhance response with task information
                response.narrative += f"\n\n[New Task: {task_result['task']['name']}]"
                nyx_context.active_tasks.append(task_result['task'])
                nyx_context.record_task_run("task_generation")
        
        # Store the interaction in memory
        await nyx_context.memory_system.add_memory(
            memory_text=f"User: {user_input}\nNyx: {response.narrative}",
            memory_type="conversation",
            memory_scope="game",
            significance=5,
            tags=["interaction"],
            metadata={
                "timestamp": datetime.now().isoformat(),
                "emotional_state": nyx_context.emotional_state,
                "tension_level": response.tension_level
            }
        )
        
        # Learn from the interaction
        await nyx_context.learn_from_interaction(
            action="response",
            outcome=f"tension_{response.tension_level}",
            success=True
        )
        
        # Update performance metrics
        response_time = time.time() - start_time
        nyx_context.update_performance("response_times", response_time)
        nyx_context.update_performance("total_actions", nyx_context.performance_metrics["total_actions"] + 1)
        nyx_context.update_performance("successful_actions", nyx_context.performance_metrics["successful_actions"] + 1)
        
        # Save updated state
        await _save_context_state(nyx_context)
        
        return {
            "success": True,
            "response": response.dict(),
            "memories_used": [],
            "performance": {
                "response_time": response_time,
                "memory_usage": nyx_context.performance_metrics["memory_usage"],
                "cpu_usage": nyx_context.performance_metrics["cpu_usage"]
            },
            "learning": {
                "patterns_learned": len(nyx_context.learned_patterns),
                "adaptation_success_rate": nyx_context.learning_metrics["adaptation_success_rate"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing user input: {e}")
        # Update failure metrics
        if nyx_context:
            nyx_context.update_performance("total_actions", nyx_context.performance_metrics["total_actions"] + 1)
            nyx_context.update_performance("failed_actions", nyx_context.performance_metrics["failed_actions"] + 1)
            nyx_context.log_error(e, {"user_input": user_input})
            
            # Learn from failure
            await nyx_context.learn_from_interaction(
                action="response",
                outcome="error",
                success=False
            )
        
        return {
            "success": False,
            "error": str(e),
            "response": {
                "narrative": "I apologize, but I encountered an error processing your request. Please try again.",
                "tension_level": 0,
                "generate_image": False
            }
        }
    finally:
        # Always close DB connection
        if nyx_context:
            await nyx_context.close_db_connection()

async def _save_context_state(ctx: NyxContext):
    """Save context state to database"""
    conn = await ctx.get_db_connection()
    
    try:
        # Save emotional state
        await conn.execute("""
            INSERT INTO NyxAgentState (user_id, conversation_id, emotional_state, updated_at)
            VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id, conversation_id) 
            DO UPDATE SET emotional_state = $3, updated_at = CURRENT_TIMESTAMP
        """, ctx.user_id, ctx.conversation_id, json.dumps(ctx.emotional_state, ensure_ascii=False))
        
        # Save scenario state if active
        if ctx.scenario_state and ctx.scenario_state.get("active"):
            await conn.execute("""
                INSERT INTO scenario_states (user_id, conversation_id, state_data, created_at)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
            """, ctx.user_id, ctx.conversation_id, json.dumps(ctx.scenario_state, ensure_ascii=False))
        
        # Save learning metrics periodically
        if ctx.should_run_task("learning_save"):
            await conn.execute("""
                INSERT INTO learning_metrics (user_id, conversation_id, metrics, learned_patterns, created_at)
                VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
            """, ctx.user_id, ctx.conversation_id, 
            json.dumps(ctx.learning_metrics, ensure_ascii=False), 
            json.dumps(dict(list(ctx.learned_patterns.items())[-50:]), ensure_ascii=False))  # Save only recent patterns
            
            ctx.record_task_run("learning_save")
        
        # Save performance metrics periodically
        if ctx.should_run_task("performance_save"):
            # Prepare metrics with bounded lists
            bounded_metrics = ctx.performance_metrics.copy()
            if "response_times" in bounded_metrics:
                bounded_metrics["response_times"] = bounded_metrics["response_times"][-50:]
            
            await conn.execute("""
                INSERT INTO performance_metrics (user_id, conversation_id, metrics, error_log, created_at)
                VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
            """, ctx.user_id, ctx.conversation_id,
            json.dumps(bounded_metrics, ensure_ascii=False),
            json.dumps(ctx.error_log[-50:], ensure_ascii=False))
            
            ctx.record_task_run("performance_save")
            
    except Exception as e:
        logger.error(f"Error saving context state: {e}")
        # Don't re-raise to avoid failing the main request

async def generate_reflection(
    user_id: int,
    conversation_id: int,
    topic: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a reflection from Nyx on a specific topic"""
    try:
        # Create and initialize context
        nyx_context = NyxContext(user_id, conversation_id)
        await nyx_context.initialize()
        
        # Create prompt
        prompt = f"Create a reflection about: {topic}" if topic else "Create a reflection about the user based on your memories"
        
        # Run the reflection agent directly
        result = await Runner.run(
            reflection_agent,
            prompt,
            context=nyx_context
        )
        
        reflection = result.final_output_as(MemoryReflection)
        
        return {
            "reflection": reflection.reflection,
            "confidence": reflection.confidence,
            "topic": reflection.topic or topic
        }
        
    except Exception as e:
        logger.error(f"Error generating reflection: {e}")
        return {
            "reflection": "Unable to generate reflection at this time.",
            "confidence": 0.0,
            "topic": topic
        }

async def manage_scenario(scenario_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Manage and coordinate complex scenarios.
    
    Args:
        scenario_data: Scenario configuration including participants, objectives, etc.
        
    Returns:
        Scenario state and coordination plan
    """
    try:
        # Extract user and conversation IDs from scenario data
        user_id = scenario_data.get("user_id")
        conversation_id = scenario_data.get("conversation_id")
        
        if not user_id or not conversation_id:
            raise ValueError("scenario_data must include user_id and conversation_id")
        
        # Create context
        nyx_context = NyxContext(user_id, conversation_id)
        await nyx_context.initialize()
        
        # Update scenario state
        nyx_context.scenario_state = {
            "id": scenario_data.get("scenario_id", f"scenario_{int(time.time())}"),
            "type": scenario_data.get("type", "general"),
            "participants": scenario_data.get("participants", []),
            "objectives": scenario_data.get("objectives", []),
            "current_phase": "initialization",
            "start_time": time.time(),
            "active": True
        }
        
        # Run scenario agent to get initial plan
        result = await Runner.run(
            scenario_agent,
            f"Initialize scenario: {json.dumps(scenario_data)}",
            context=nyx_context
        )
        
        decision = result.final_output_as(ScenarioDecision)
        
        # Update scenario state with decision
        nyx_context.scenario_state["next_phase"] = decision.next_phase
        nyx_context.scenario_state["tasks"] = decision.tasks
        nyx_context.scenario_state["npc_actions"] = decision.npc_actions
        nyx_context.scenario_state["last_decision"] = {
            "action": decision.action,
            "time_advancement": decision.time_advancement,
            "timestamp": time.time()
        }
        
        # Save state
        await _save_context_state(nyx_context)
        
        return {
            "success": True,
            "scenario_state": nyx_context.scenario_state,
            "initial_tasks": decision.tasks,
            "coordination_plan": {
                "action": decision.action,
                "next_phase": decision.next_phase,
                "npc_actions": decision.npc_actions
            }
        }
        
    except Exception as e:
        logger.error(f"Error managing scenario: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def manage_relationships(interaction_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Manage and update relationships between entities.
    
    Args:
        interaction_data: Interaction details including participants and outcomes
        
    Returns:
        Updated relationship states
    """
    nyx_context = None
    
    try:
        # Extract user and conversation IDs
        user_id = interaction_data.get("user_id")
        conversation_id = interaction_data.get("conversation_id")
        
        if not user_id or not conversation_id:
            raise ValueError("interaction_data must include user_id and conversation_id")
        
        # Create context
        nyx_context = NyxContext(user_id, conversation_id)
        await nyx_context.initialize()
        
        # Process each participant pair
        participants = interaction_data.get("participants", [])
        relationship_updates = {}
        
        for i, p1 in enumerate(participants):
            for p2 in participants[i+1:]:
                # Calculate relationship changes based on interaction
                trust_change = 0.1 if interaction_data.get("outcome") == "success" else -0.05
                bond_change = 0.05 if interaction_data.get("emotional_impact", {}).get("positive", 0) > 0 else 0
                power_change = 0.0
                
                if interaction_data.get("interaction_type") == "training":
                    power_change = 0.05
                elif interaction_data.get("interaction_type") == "conflict":
                    power_change = -0.05
                
                # Update relationship using the tool
                result = await update_relationship_state(
                    RunContextWrapper(context=nyx_context),
                    f"{p1['id']}_{p2['id']}",
                    trust_change,
                    power_change,
                    bond_change
                )
                
                relationship_updates[f"{p1['id']}_{p2['id']}"] = json.loads(result)
        
        # Note: interaction_history table is not in the schema
        # We'll just log this as a warning instead of trying to insert
        logger.warning("interaction_history table not found in schema - skipping interaction storage")
        
        # Learn from the relationship interaction
        for pair, updates in relationship_updates.items():
            await nyx_context.learn_from_interaction(
                action=f"relationship_{interaction_data.get('interaction_type', 'general')}",
                outcome=interaction_data.get("outcome", "unknown"),
                success=updates.get("changes", {}).get("trust", 0) > 0
            )
        
        return {
            "success": True,
            "relationship_updates": relationship_updates,
            "analysis": {
                "total_relationships_updated": len(relationship_updates),
                "interaction_type": interaction_data.get("interaction_type"),
                "outcome": interaction_data.get("outcome"),
                "stored_in_history": False  # Since table doesn't exist
            }
        }
        
    except Exception as e:
        logger.error(f"Error managing relationships: {e}")
        if nyx_context:
            nyx_context.log_error(e, interaction_data)
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        # Always close DB connection
        if nyx_context:
            await nyx_context.close_db_connection()

async def store_messages(user_id: int, conversation_id: int, user_input: str, nyx_response: str):
    """Store user and Nyx messages in database"""
    async with get_db_connection_context() as conn:
        await conn.execute(
            "INSERT INTO messages (conversation_id, sender, content) VALUES ($1, $2, $3)",
            conversation_id, "user", user_input
        )
        
        await conn.execute(
            "INSERT INTO messages (conversation_id, sender, content) VALUES ($1, $2, $3)",
            conversation_id, "Nyx", nyx_response
        )

# Additional helper functions
async def get_emotional_state(ctx) -> str:
    """Get current emotional state"""
    if hasattr(ctx, 'emotional_state'):
        return json.dumps(ctx.emotional_state, ensure_ascii=False)
    elif hasattr(ctx, 'context') and hasattr(ctx.context, 'emotional_state'):
        return json.dumps(ctx.context.emotional_state, ensure_ascii=False)
    else:
        # Default state
        return json.dumps({
            "valence": 0.0,
            "arousal": 0.5,
            "dominance": 0.7
        }, ensure_ascii=False)

async def update_emotional_state(ctx, emotional_state: Dict[str, Any]) -> str:
    """Update emotional state - Fixed to properly update the context"""
    if hasattr(ctx, 'emotional_state'):
        ctx.emotional_state.update(emotional_state)
    elif hasattr(ctx, 'context') and hasattr(ctx.context, 'emotional_state'):
        ctx.context.emotional_state.update(emotional_state)
    return "Emotional state updated"

# ===== Compatibility functions to maintain existing imports =====

# Function mappings for backward compatibility
# Use the actual function objects, not the decorators
retrieve_memories_impl = retrieve_memories
add_memory_impl = add_memory
get_user_model_guidance_impl = get_user_model_guidance
detect_user_revelations_impl = detect_user_revelations
generate_image_from_scene_impl = generate_image_from_scene
get_emotional_state_impl = get_emotional_state
update_emotional_state_impl = update_emotional_state
calculate_emotional_impact_impl = calculate_emotional_impact
calculate_and_update_emotional_state_impl = calculate_and_update_emotional_state
manage_beliefs_impl = manage_beliefs
score_decision_options_impl = score_decision_options
detect_conflicts_and_instability_impl = detect_conflicts_and_instability

# Export list for clean imports
__all__ = [
    # Main functions
    'initialize_agents',
    'process_user_input',
    'generate_reflection',
    'manage_scenario',
    'manage_relationships',
    'store_messages',
    
    # Context classes
    'NyxContext',
    'AgentContext',
    
    # Tool functions (primary names)
    'retrieve_memories',
    'add_memory',
    'get_user_model_guidance',
    'detect_user_revelations',
    'generate_image_from_scene',
    'calculate_emotional_impact',
    'calculate_and_update_emotional_state',
    'update_relationship_state',
    'check_performance_metrics',
    'get_activity_recommendations',
    'manage_beliefs',
    'score_decision_options',
    'detect_conflicts_and_instability',
    
    # Helper functions
    'safe_psutil',
    'safe_process_metric',
    'enhance_context_with_memories',
    'should_generate_task',
    'should_recommend_activities',
    'get_available_activities',
    
    # Async helpers
    'get_emotional_state',
    'update_emotional_state',
    'generate_base_response',
    'mark_strategy_for_review',
    
    # Compatibility functions
    'enhance_context_with_strategies',
    'determine_image_generation',
    'process_user_input_with_openai',
    'process_user_input_standalone',
    
    # Agents (for advanced usage)
    'memory_agent',
    'analysis_agent',
    'emotional_agent',
    'visual_agent',
    'activity_agent',
    'performance_agent',
    'scenario_agent',
    'belief_agent',
    'decision_agent',
    'reflection_agent',
    'nyx_main_agent',
    
    # Compatibility _impl versions
    'retrieve_memories_impl',
    'add_memory_impl',
    'get_user_model_guidance_impl',
    'detect_user_revelations_impl',
    'generate_image_from_scene_impl',
    'get_emotional_state_impl',
    'update_emotional_state_impl',
    'calculate_emotional_impact_impl',
    'calculate_and_update_emotional_state_impl',
    'manage_beliefs_impl',
    'score_decision_options_impl',
    'detect_conflicts_and_instability_impl',
]

async def determine_image_generation_impl(ctx, response_text: str) -> str:
    """Compatibility wrapper for image generation decision"""
    # Use visual agent to determine if image should be generated
    visual_ctx = NyxContext(ctx.user_id, ctx.conversation_id)
    await visual_ctx.initialize()
    
    result = await Runner.run(
        visual_agent,
        f"Should an image be generated for this scene? {response_text}",
        context=visual_ctx
    )
    
    decision = result.final_output
    
    return json.dumps({
        "should_generate": getattr(decision, 'should_generate', False),
        "score": getattr(decision, 'score', 0),
        "image_prompt": getattr(decision, 'image_prompt', None)
    })

async def enhance_context_with_strategies_impl(context: Dict[str, Any], conn) -> Dict[str, Any]:
    """Enhance context with active strategies"""
    strategies = await get_active_strategies(conn)
    context["nyx2_strategies"] = strategies
    return context

def enhance_context_with_memories(context, memories):
    """Add memories to context for better decision making."""
    enhanced_context = context.copy()
    enhanced_context['relevant_memories'] = memories
    return enhanced_context

def should_generate_task(context: Dict[str, Any]) -> bool:
    """Determine if we should generate a creative task"""
    if not context.get("active_npc_id"):
        return False
    scenario_type = context.get("scenario_type", "").lower()
    task_scenarios = ["training", "challenge", "service", "discipline"]
    if not any(t in scenario_type for t in task_scenarios):
        return False
    npc_relationship = context.get("npc_relationship_level", 0)
    if npc_relationship < 30:
        return False
    return True

def should_recommend_activities(context: Dict[str, Any]) -> bool:
    """Determine if we should recommend activities"""
    if not context.get("present_npc_ids"):
        return False
    scenario_type = context.get("scenario_type", "").lower()
    if "task" in scenario_type or "challenge" in scenario_type:
        return False
    user_input = context.get("user_input", "").lower()
    suggestion_triggers = ["what should", "what can", "what to do", "suggestions", "ideas"]
    if any(trigger in user_input for trigger in suggestion_triggers):
        return True
    if context.get("is_scene_transition") or context.get("is_activity_completed"):
        return True
    return False

def get_available_activities() -> List[Dict]:
    """Get list of available activities"""
    return [
        {
            "name": "Training Session",
            "category": "training",
            "preferred_traits": ["disciplined", "focused"],
            "avoided_traits": ["lazy"],
            "preferred_times": ["morning", "afternoon"],
            "prerequisites": ["training equipment"],
            "outcomes": ["skill improvement", "increased discipline"]
        }
    ]

async def generate_base_response(ctx: NyxContext, user_input: str, context: Dict[str, Any]) -> NarrativeResponse:
    """Generate base narrative response - for compatibility"""
    result = await Runner.run(
        nyx_main_agent,
        user_input,
        context=ctx
    )
    return result.final_output_as(NarrativeResponse)

async def mark_strategy_for_review(conn, strategy_id: int, user_id: int, reason: str):
    """Mark a strategy for review"""
    await conn.execute("""
        INSERT INTO strategy_reviews (strategy_id, user_id, reason, created_at)
        VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
    """, strategy_id, user_id, reason)

# Compatibility with existing code
enhance_context_with_strategies = enhance_context_with_strategies_impl
determine_image_generation = determine_image_generation_impl

# OpenAI integration functions
async def process_user_input_with_openai(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Process user input using the OpenAI integration"""
    return await process_user_input(user_id, conversation_id, user_input, context_data)

async def process_user_input_standalone(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Process user input standalone"""
    return await process_user_input(user_id, conversation_id, user_input, context_data)

# Legacy AgentContext for full backward compatibility
class AgentContext:
    """Full backward compatibility with original AgentContext"""
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._nyx_context = None
        
        # Initialize all legacy attributes
        self.memory_system = None
        self.user_model = None
        self.task_integration = None
        self.belief_system = None
        self.emotional_system = None
        self.current_goals = []
        self.active_tasks = []
        self.decision_history = []
        self.state_history = []
        self.last_action = None
        self.last_result = None
        self.current_emotional_state = {}
        self.beliefs = {}
        self.intentions = []
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
        self.learned_patterns = {}
        self.strategy_effectiveness = {}
        self.adaptation_history = []
        self.learning_metrics = {
            "pattern_recognition_rate": 0.0,
            "strategy_improvement_rate": 0.0,
            "adaptation_success_rate": 0.0
        }
        self.resource_pools = {}  # Removed - use asyncio.Semaphore if concurrency limits needed
        # Example: decision_semaphore = asyncio.Semaphore(10)
        # async with decision_semaphore: ... # Limits to 10 concurrent decisions
        self.resource_usage = {
            "memory": 0,
            "cpu": 0,
            "network": 0
        }
        self.context_cache = {}
        self.communication_history = []
        self.error_log = []
    
    @classmethod
    async def create(cls, user_id: int, conversation_id: int):
        """Async factory method for compatibility"""
        instance = cls(user_id, conversation_id)
        instance._nyx_context = NyxContext(user_id, conversation_id)
        await instance._nyx_context.initialize()
        
        # Map to legacy attributes
        instance.memory_system = instance._nyx_context.memory_system
        instance.user_model = instance._nyx_context.user_model
        instance.task_integration = instance._nyx_context.task_integration
        instance.belief_system = instance._nyx_context.belief_system
        instance.current_emotional_state = instance._nyx_context.emotional_state
        instance.performance_metrics.update(instance._nyx_context.performance_metrics)
        instance.learned_patterns = instance._nyx_context.learned_patterns
        instance.strategy_effectiveness = instance._nyx_context.strategy_effectiveness
        instance.adaptation_history = instance._nyx_context.adaptation_history
        instance.learning_metrics = instance._nyx_context.learning_metrics
        instance.error_log = instance._nyx_context.error_log
        
        # Load initial state
        await instance._load_initial_state()
        
        return instance
    
    async def _initialize_systems(self):
        """Legacy compatibility method"""
        pass
    
    async def _load_initial_state(self):
        """Load initial state for agent context"""
        # Already handled by NyxContext initialization
        pass
    
    async def make_decision(self, context: Dict[str, Any], options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make a decision using the decision scoring engine"""
        result = await score_decision_options(
            RunContextWrapper(context=self._nyx_context),
            options,
            context
        )
        decision_data = json.loads(result)
        
        # Update decision history
        self.decision_history.append({
            "timestamp": time.time(),
            "selected_option": decision_data["best_option"],
            "score": decision_data["confidence"],
            "context": context
        })
        
        # Update confidence
        self.decision_confidence = decision_data["confidence"]
        
        return {
            "decision": decision_data["best_option"],
            "confidence": decision_data["confidence"],
            "components": decision_data["scored_options"][0]["components"]
        }
    
    async def learn_from_experience(self, experience: Dict[str, Any]):
        """Learn from experience and update patterns"""
        await self._nyx_context.learn_from_interaction(
            action=experience.get("action", "unknown"),
            outcome=experience.get("outcome", "unknown"),
            success=experience.get("success", False)
        )
        
        # Update local attributes from nyx context
        self.learned_patterns = self._nyx_context.learned_patterns
        self.adaptation_history = self._nyx_context.adaptation_history
        self.learning_metrics = self._nyx_context.learning_metrics
    
    async def process_emotional_state(self, context: Dict[str, Any], user_emotion: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process and update emotional state - Fixed to actually update the state"""
        # Add user emotion to context if provided
        if user_emotion:
            context["user_emotion"] = user_emotion
        
        # Use the composite tool that both calculates AND updates
        result = await calculate_and_update_emotional_state(
            RunContextWrapper(context=self._nyx_context),
            context
        )
        
        emotional_data = json.loads(result)
        # Update local state to match
        self.current_emotional_state = {
            "valence": emotional_data["valence"],
            "arousal": emotional_data["arousal"],
            "dominance": emotional_data["dominance"],
            "primary_emotion": emotional_data["primary_emotion"]
        }
        
        return self.current_emotional_state
    
    async def manage_scenario(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Forward to new manage_scenario function"""
        scenario_data["user_id"] = self.user_id
        scenario_data["conversation_id"] = self.conversation_id
        return await manage_scenario(scenario_data)
    
    async def manage_relationships(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Forward to new manage_relationships function"""
        interaction_data["user_id"] = self.user_id
        interaction_data["conversation_id"] = self.conversation_id
        return await manage_relationships(interaction_data)
    
    async def get_emotional_state(self) -> Dict[str, Any]:
        """Get current emotional state"""
        return self.current_emotional_state
    
    async def update_emotional_state(self, new_state: Dict[str, Any]):
        """Update emotional state"""
        self.current_emotional_state.update(new_state)
        self._nyx_context.emotional_state.update(new_state)
    
    def update_context(self, new_context: Dict[str, Any], use_delta: bool = True):
        """Update context - compatibility method"""
        self.context_cache.update(new_context)
        self._nyx_context.current_context.update(new_context)
    
    async def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary"""
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
            "errors": {
                "total": self.performance_metrics["error_rates"]["total"],
                "recovered": self.performance_metrics["error_rates"]["recovered"],
                "unrecovered": self.performance_metrics["error_rates"]["unrecovered"]
            }
        }
    
    # Additional compatibility methods
    def _calculate_emotional_weight(self, emotional_state: Dict[str, Any]) -> float:
        """Calculate emotional weight for decisions"""
        intensity = max(abs(emotional_state.get("valence", 0)), abs(emotional_state.get("arousal", 0)))
        return min(1.0, intensity * 2.0)
    
    def _calculate_pattern_weight(self, context: Dict[str, Any]) -> float:
        """Calculate pattern weight for decisions"""
        relevant_patterns = sum(1 for p in self.learned_patterns.values()
                               if any(k in str(context) for k in str(p).split()))
        return min(1.0, relevant_patterns * 0.2)
    
    def _should_run_task(self, task_id: str) -> bool:
        """Check if task should run"""
        return self._nyx_context.should_run_task(task_id)
