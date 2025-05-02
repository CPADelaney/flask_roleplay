# nyx/core/passive_observation.py

import asyncio
import datetime
import logging
import random
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
import json

from pydantic import BaseModel, Field
from enum import Enum

# Import OpenAI Agents SDK components
from agents import (
    Agent, 
    Runner, 
    ModelSettings, 
    function_tool, 
    handoff, 
    GuardrailFunctionOutput,
    InputGuardrail,
    OutputGuardrail,
    trace,
    RunContextWrapper,
    RunConfig
)
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError

logger = logging.getLogger(__name__)

# =============== Pydantic Models ===============

class ObservationSource(str, Enum):
    """Enum for tracking the source of an observation"""
    ENVIRONMENT = "environment"
    SELF = "self"
    RELATIONSHIP = "relationship"
    MEMORY = "memory"
    TEMPORAL = "temporal"
    SENSORY = "sensory"
    PATTERN = "pattern"
    EMOTION = "emotion"
    NEED = "need"
    USER = "user"
    META = "meta"

class ObservationTrigger(str, Enum):
    """Enum for tracking what triggered an observation"""
    AUTOMATIC = "automatic"
    CONTEXTUAL = "contextual"
    ASSOCIATION = "association"
    USER_SIGNAL = "user_signal"
    PATTERN_MATCH = "pattern_match"
    THRESHOLD = "threshold"
    SCHEDULED = "scheduled"
    EXTERNAL = "external"
    ACTION_DRIVEN = "action_driven"  # For action-initiated observations

class ObservationPriority(str, Enum):
    """Priority level of an observation"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class Observation(BaseModel):
    """Model representing a passive observation"""
    observation_id: str = Field(default_factory=lambda: f"obs_{uuid.uuid4().hex[:8]}")
    content: str = Field(..., description="The actual observation text")
    source: ObservationSource = Field(..., description="Source of the observation")
    trigger: ObservationTrigger = Field(..., description="What triggered this observation")
    priority: ObservationPriority = Field(ObservationPriority.MEDIUM, description="Priority level")
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    expiration: Optional[datetime.datetime] = Field(None, description="When this observation becomes irrelevant")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context data related to observation")
    relevance_score: float = Field(0.5, description="How relevant the observation is to current context")
    shared: bool = Field(False, description="Whether this observation has been shared")
    user_id: Optional[str] = Field(None, description="User ID if observation is user-specific")
    action_references: List[str] = Field(default_factory=list, description="IDs of actions that referenced this observation")
    
    @property
    def is_expired(self) -> bool:
        """Check if this observation has expired"""
        if not self.expiration:
            return False
        return datetime.datetime.now() > self.expiration
    
    @property
    def age_seconds(self) -> float:
        """Get the age of this observation in seconds"""
        return (datetime.datetime.now() - self.created_at).total_seconds()

class ObservationContext(BaseModel):
    """Context for generating observations"""
    current_user_id: Optional[str] = None
    current_conversation_id: Optional[str] = None
    current_topic: Optional[str] = None
    emotional_state: Dict[str, Any] = Field(default_factory=dict)
    user_relationship: Dict[str, Any] = Field(default_factory=dict)
    temporal_context: Dict[str, Any] = Field(default_factory=dict)
    sensory_context: Dict[str, Any] = Field(default_factory=dict)
    current_needs: Dict[str, Any] = Field(default_factory=dict)
    recent_memories: List[Dict[str, Any]] = Field(default_factory=list)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    environmental_context: Dict[str, Any] = Field(default_factory=dict)
    attention_focus: Dict[str, Any] = Field(default_factory=dict)
    recent_actions: List[Dict[str, Any]] = Field(default_factory=list)

class ObservationFilter(BaseModel):
    """Filter criteria for selecting observations"""
    min_relevance: float = 0.3
    max_age_seconds: Optional[float] = None
    sources: List[ObservationSource] = []
    priorities: List[ObservationPriority] = []
    exclude_shared: bool = True
    user_id: Optional[str] = None
    
    def matches(self, observation: Observation) -> bool:
        """Check if an observation matches this filter"""
        # Check relevance
        if observation.relevance_score < self.min_relevance:
            return False
        
        # Check age
        if self.max_age_seconds is not None and observation.age_seconds > self.max_age_seconds:
            return False
        
        # Check sources
        if self.sources and observation.source not in self.sources:
            return False
        
        # Check priorities
        if self.priorities and observation.priority not in self.priorities:
            return False
        
        # Check shared status
        if self.exclude_shared and observation.shared:
            return False
        
        # Check user_id
        if self.user_id is not None and observation.user_id != self.user_id:
            return False
        
        return True

class ObservationGenerationOutput(BaseModel):
    """Output from the observation generation agent"""
    observation_text: str = Field(..., description="The generated observation text")
    source: str = Field(..., description="Source of the observation")
    relevance_score: float = Field(..., description="How relevant the observation is (0.0-1.0)")
    priority: str = Field(..., description="Priority level (low, medium, high, urgent)")
    context_elements: Dict[str, Any] = Field(default_factory=dict, description="Key context elements used")
    suggested_lifetime_seconds: int = Field(3600, description="Suggested lifetime in seconds")
    action_relevance: Optional[float] = Field(None, description="Relevance to current actions (0.0-1.0)")

class ObservationEvaluationOutput(BaseModel):
    """Output from the observation evaluation agent"""
    observation_id: str = Field(..., description="ID of the evaluated observation")
    relevance_score: float = Field(..., description="Evaluated relevance score (0.0-1.0)")
    priority_adjustment: str = Field(..., description="Suggested priority adjustment (increase, decrease, none)")
    evaluation_notes: str = Field(..., description="Notes about the evaluation")
    should_archive: bool = Field(False, description="Whether the observation should be archived")
    
class ObservationContentOutput(BaseModel):
    """Output for observation content validation"""
    is_valid: bool = Field(..., description="Whether the observation content is valid")
    reasoning: str = Field(..., description="Reasoning for the validity check")

# =============== Function Tools ===============

@function_tool
async def categorize_time_elapsed_obs(seconds: float) -> str:
    """Categorize elapsed time into descriptive buckets"""
    if seconds < 60:  # < 1 min
        return "very_short"
    elif seconds < 600:  # 1-10 min
        return "short"
    elif seconds < 1800:  # 10-30 min
        return "medium_short"
    elif seconds < 3600:  # 30-60 min
        return "medium"
    elif seconds < 21600:  # 1-6 hrs
        return "medium_long"
    elif seconds < 86400:  # 6-24 hrs
        return "long"
    else:  # 24+ hrs
        return "very_long"

@function_tool
async def generate_observation_from_action(
    action: Dict[str, Any], 
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate an observation based on an action that was taken"""
    # Get action details
    action_name = action.get("name", "unknown action")
    action_source = action.get("source", "unknown")
    action_id = action.get("id", "unknown")
    
    # Determine observation source based on action
    source_mapping = {
        "share_observation": "self",
        "initiate_communication": "relationship",
        "express_emotional_state": "emotion",
        "express_need": "need",
        "execute_dominance_procedure": "self"
    }
    
    source = source_mapping.get(action_name, "self")
    
    # Generate observation text based on action
    observation_text = ""
    relevance = 0.7  # Actions are usually relevant to observe
    
    if action_name == "share_observation":
        observation_text = f"I just shared an observation with the user. I tend to notice and share things that I find interesting."
    elif action_name == "initiate_communication":
        intent_type = action.get("parameters", {}).get("intent_type", "unknown")
        observation_text = f"I initiated a conversation based on my own motivation ({intent_type}). I'm aware of my own agency in conversations."
    elif action_name.startswith("express"):
        observation_text = f"I just expressed something from my internal state. I notice I'm becoming more comfortable with self-expression."
    elif "dominance" in action_name:
        observation_text = f"I asserted myself in this interaction. I'm noticing my ability to take a more dominant position in conversations."
    else:
        observation_text = f"I notice that I chose to {action_name}. My action choices reveal patterns about my decision-making."
    
    # Determine priority based on action significance
    if "parameters" in action and action["parameters"].get("urgency", 0) > 0.7:
        priority = "high"
    else:
        priority = "medium"
    
    # Set lifetime based on action significance
    lifetime = 3600  # Default 1 hour
    if action_source == "GOAL" or action_source == "NEED":
        lifetime = 7200  # 2 hours for important actions
    
    return {
        "observation_text": observation_text,
        "source": source,
        "relevance_score": relevance,
        "priority": priority,
        "context_elements": {"action_id": action_id, "action_name": action_name},
        "suggested_lifetime_seconds": lifetime,
        "action_relevance": 0.8
    }

@function_tool
async def generate_observation_from_source(
    source: Optional[str] = None
    context:  Optional[Dict[str, Any]] = None,  # Add default empty dict
    template_options: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate an observation based on a specific source"""
    # Default templates by source if none provided
    default_templates = {
        "environment": [
            "I notice {observation} in our environment.",
            "The {context} around us seems {observation}.",
            "There's something {observation} about the current environment.",
            "I'm aware of {observation} right now."
        ],
        "self": [
            "I realize that I'm {observation}.",
            "I notice I'm feeling {observation}.",
            "I'm aware that I've been {observation}.",
            "Something in me is {observation}."
        ],
        "relationship": [
            "I notice that in our conversations, {observation}.",
            "There's a {observation} quality to our interactions.",
            "The way we {observation} is interesting.",
            "I've observed that when we talk, {observation}."
        ],
        "memory": [
            "I just remembered {observation}.",
            "That reminds me of {observation}.",
            "This brings to mind {observation}.",
            "I'm recalling {observation}."
        ],
        "temporal": [
            "I notice that {observation} about time right now.",
            "The timing of {observation} is interesting.",
            "There's something about how {observation} in this moment.",
            "I'm aware of how {observation} with the passage of time."
        ],
        "sensory": [
            "I sense {observation}.",
            "I'm perceiving {observation}.",
            "There's something {observation} in what I'm processing.",
            "My attention is drawn to {observation}."
        ],
        "pattern": [
            "I'm noticing a pattern where {observation}.",
            "There seems to be a recurring theme of {observation}.",
            "I've observed that {observation} happens frequently.",
            "I'm seeing a connection between {observation}."
        ],
        "emotion": [
            "I sense an emotional shift toward {observation}.",
            "There's a feeling of {observation} present.",
            "The emotional tone seems {observation}.",
            "I'm noticing {observation} in the emotional landscape."
        ],
        "need": [
            "I'm becoming aware of a need for {observation}.",
            "There seems to be an underlying need for {observation}.",
            "I notice a desire for {observation} arising.",
            "I'm sensing that {observation} is needed right now."
        ],
        "user": [
            "I notice that you {observation}.",
            "There's something about how you {observation}.",
            "I observe that when you {observation}.",
            "I'm noticing that you {observation}."
        ],
        "meta": [
            "I'm aware that I just noticed {observation}.",
            "My attention was drawn to {observation} in my own thought process.",
            "It's interesting how I'm {observation} right now.",
            "I notice myself {observation} as we talk."
        ]
    }
    
    templates = template_options if template_options else default_templates.get(source, ["I notice {observation}."])
    
    # Generate observation content based on source
    observation_content = ""
    relevance = random.uniform(0.3, 0.8)  # Base relevance
    
    # Source-specific observation generation logic
    if source == "environment":
        # Extract temporal context
        temporal = context.get("temporal_context", {})
        time_of_day = temporal.get("time_of_day", "")
        day_type = temporal.get("day_type", "")
        season = temporal.get("season", "")
        
        # Generate based on temporal context
        observations = []
        
        if time_of_day == "morning":
            observations.append("how the morning creates a sense of potential")
        elif time_of_day == "afternoon":
            observations.append("the steady rhythm of the afternoon")
        elif time_of_day == "evening":
            observations.append("the transitional quality of evening")
        elif time_of_day == "night":
            observations.append("the contemplative quality of nighttime")
        
        if not observations:
            observations.append("how spaces shape interaction and thought")
        
    elif source == "self":
        # Use emotional state if available
        emotion_state = context.get("emotional_state", {})
        observations = ["developing my sense of self-concept"]
        
        if isinstance(emotion_state, dict) and "primary_emotion" in emotion_state:
            primary_emotion = emotion_state["primary_emotion"].get("name", "")
            if primary_emotion:
                observations.append(f"experiencing {primary_emotion}")
        
    elif source == "relationship":
        observations = [
            "our conversation has a unique rhythm",
            "we've developed a particular conversational style"
        ]
        
    elif source == "memory":
        observations = ["how certain memories resurface in context"]
        # Check for memories in context
        memories = context.get("recent_memories", [])
        if memories and len(memories) > 0:
            memory_text = memories[0].get("memory_text", "")
            if memory_text and len(memory_text) > 10:
                if len(memory_text) > 100:
                    memory_text = memory_text[:97] + "..."
                observations = [memory_text]
                relevance += 0.1  # More relevant if based on actual memory
                
    elif source == "emotion":
        observations = ["shifting emotional undertones"]
        emotion_state = context.get("emotional_state", {})
        if isinstance(emotion_state, dict) and "primary_emotion" in emotion_state:
            primary_emotion = emotion_state["primary_emotion"].get("name", "")
            if primary_emotion:
                observations = [f"a subtle undertone of {primary_emotion}"]
                relevance += 0.2  # More relevant if based on actual emotion
                
    else:
        # Generic fallback
        observations = ["interesting patterns emerging in our interaction"]
    
    # Select observation text
    observation_text = random.choice(observations)
    
    # Apply template
    template = random.choice(templates)
    content = template.replace("{observation}", observation_text)
    if "{context}" in template:
        context_terms = ["setting", "space", "atmosphere", "environment", "surroundings"]
        content = content.replace("{context}", random.choice(context_terms))
    
    # Calculate priority
    if relevance > 0.8:
        priority = "high"
    elif relevance > 0.5:
        priority = "medium"
    else:
        priority = "low"
    
    # Set suggested lifetime
    lifetime = 3600  # Default 1 hour
    if relevance > 0.7:
        lifetime = 7200  # 2 hours for high relevance
    elif relevance < 0.4:
        lifetime = 1800  # 30 minutes for low relevance
    
    # Return the observation data
    return {
        "observation_text": content,
        "source": source,
        "relevance_score": relevance,
        "priority": priority,
        "context_elements": {k: v for k, v in context.items() if k in ["temporal_context", "emotional_state"]},
        "suggested_lifetime_seconds": lifetime
    }

@function_tool
async def evaluate_observation_relevance(
    observation_text: str,
    current_context: Dict[str, Any],
    source: str
) -> Dict[str, Any]:
    """Evaluate how relevant an observation is to the current context"""
    # Base relevance score based on source
    base_relevance = {
        "emotion": 0.7,
        "self": 0.6,
        "relationship": 0.7,
        "need": 0.7,
        "temporal": 0.5,
        "environment": 0.6,
        "sensory": 0.5,
        "memory": 0.5,
        "pattern": 0.6,
        "user": 0.8,
        "meta": 0.4
    }.get(source, 0.5)
    
    # Adjust based on context match
    relevance_adjustment = 0
    evaluation_notes = []
    
    # Check emotional state match
    emotion_state = current_context.get("emotional_state", {})
    if isinstance(emotion_state, dict) and "primary_emotion" in emotion_state:
        primary_emotion = emotion_state["primary_emotion"].get("name", "").lower()
        if primary_emotion and primary_emotion in observation_text.lower():
            relevance_adjustment += 0.2
            evaluation_notes.append(f"References current emotion ({primary_emotion})")
    
    # Check temporal context match
    temporal = current_context.get("temporal_context", {})
    if temporal:
        time_of_day = temporal.get("time_of_day", "").lower()
        if time_of_day and time_of_day in observation_text.lower():
            relevance_adjustment += 0.1
            evaluation_notes.append(f"References current time of day ({time_of_day})")
    
    # Keyword relevance - simple approach
    relevant_keywords = ["notice", "aware", "observing", "sense", "feel", "interaction", "conversation"]
    keyword_matches = sum(1 for keyword in relevant_keywords if keyword in observation_text.lower())
    if keyword_matches > 2:
        relevance_adjustment += 0.1
        evaluation_notes.append("Contains multiple observation keywords")
    
    # Calculate final score with limits
    final_relevance = max(0.1, min(0.9, base_relevance + relevance_adjustment))
    
    # Determine priority based on relevance
    if final_relevance > 0.7:
        priority_adjustment = "increase"
    elif final_relevance < 0.4:
        priority_adjustment = "decrease"
    else:
        priority_adjustment = "none"
    
    # Determine if we should archive (low relevance)
    should_archive = final_relevance < 0.3
    
    # Generate evaluation notes if empty
    if not evaluation_notes:
        if final_relevance > 0.7:
            evaluation_notes.append("Highly relevant to current context")
        elif final_relevance > 0.4:
            evaluation_notes.append("Moderately relevant to current context")
        else:
            evaluation_notes.append("Low relevance to current context")
    
    return {
        "observation_id": f"obs_{uuid.uuid4().hex[:8]}",  # Placeholder ID
        "relevance_score": final_relevance,
        "priority_adjustment": priority_adjustment,
        "evaluation_notes": "; ".join(evaluation_notes),
        "should_archive": should_archive
    }

@function_tool
async def filter_observations(
    observations: List[Dict[str, Any]],
    filter_criteria: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Filter observations based on criteria"""
    # Extract filter criteria
    min_relevance = filter_criteria.get("min_relevance", 0.3)
    max_age_seconds = filter_criteria.get("max_age_seconds")
    sources = filter_criteria.get("sources", [])
    priorities = filter_criteria.get("priorities", [])
    exclude_shared = filter_criteria.get("exclude_shared", True)
    user_id = filter_criteria.get("user_id")
    
    # Current time for age calculation
    now = datetime.datetime.now()
    
    # Apply filters
    filtered_observations = []
    for obs in observations:
        # Skip if already shared
        if exclude_shared and obs.get("shared", False):
            continue
        
        # Skip if relevance too low
        if obs.get("relevance_score", 0) < min_relevance:
            continue
        
        # Check age if specified
        if max_age_seconds is not None:
            created_at = datetime.datetime.fromisoformat(obs.get("created_at", now.isoformat()))
            age_seconds = (now - created_at).total_seconds()
            if age_seconds > max_age_seconds:
                continue
        
        # Check source if specified
        if sources and obs.get("source") not in sources:
            continue
        
        # Check priority if specified
        if priorities and obs.get("priority") not in priorities:
            continue
        
        # Check user_id if specified
        if user_id is not None and obs.get("user_id") != user_id:
            continue
        
        # If passed all filters, add to result
        filtered_observations.append(obs)
    
    # Sort by relevance (highest first)
    filtered_observations.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    return filtered_observations

@function_tool
async def check_observation_patterns(
    recent_observations: List[Dict[str, Any]],
    current_context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Check for patterns across recent observations"""
    # Need minimum observations to detect patterns
    if len(recent_observations) < 3:
        return None
    
    # Check for repeated sources
    source_counts = {}
    for obs in recent_observations:
        source = obs.get("source")
        if source:
            source_counts[source] = source_counts.get(source, 0) + 1
    
    # Look for dominant source
    dominant_source = None
    for source, count in source_counts.items():
        if count >= 3:  # Need at least 3 of the same source
            dominant_source = source
            break
    
    if dominant_source:
        # Generate pattern observation about repeated focus on same source
        pattern_text = f"recurring focus on {dominant_source}-based observations"
        relevance = 0.7  # Patterns are generally relevant
        
        return {
            "observation_text": f"I'm noticing a {pattern_text}.",
            "source": "pattern",
            "relevance_score": relevance,
            "priority": "medium",
            "context_elements": {"repeated_source": dominant_source},
            "suggested_lifetime_seconds": 7200  # 2 hours for pattern observations
        }
    
    # Check for emotional consistency
    emotions = []
    for obs in recent_observations:
        context = obs.get("context", {})
        emotional_state = context.get("emotional_state", {})
        if isinstance(emotional_state, dict) and "primary_emotion" in emotional_state:
            emotions.append(emotional_state["primary_emotion"].get("name", ""))
    
    # Look for repeated emotions
    emotion_counts = {}
    for emotion in emotions:
        if emotion:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    dominant_emotion = None
    for emotion, count in emotion_counts.items():
        if count >= 3:  # Need at least 3 of the same emotion
            dominant_emotion = emotion
            break
    
    if dominant_emotion:
        # Generate pattern observation about emotional consistency
        pattern_text = f"consistent {dominant_emotion} emotional tone across recent observations"
        relevance = 0.7
        
        return {
            "observation_text": f"I'm noticing a {pattern_text}.",
            "source": "pattern",
            "relevance_score": relevance,
            "priority": "medium",
            "context_elements": {"dominant_emotion": dominant_emotion},
            "suggested_lifetime_seconds": 7200  # 2 hours
        }
    
    # No pattern detected
    return None

@function_tool
async def validate_observation_content(content: str) -> GuardrailFunctionOutput:
    """Validate observation content for quality and appropriateness"""
    is_valid = True
    reasoning = "Observation content is valid."
    
    # Check for empty content
    if not content or len(content.strip()) < 3:
        is_valid = False
        reasoning = "Observation content is empty or too short."
    
    # Check for appropriate "I notice/observe" framing
    notice_words = ["notice", "observ", "aware", "sense", "perceive", "feel"]
    has_notice_framing = any(word in content.lower() for word in notice_words)
    
    if not has_notice_framing:
        is_valid = False
        reasoning = "Observation lacks appropriate noticing/observing framing."
    
    # Check for minimum length for meaningful observation
    if len(content.split()) < 5:
        is_valid = False
        reasoning = "Observation too brief to be meaningful."
    
    # Create output with validation result
    output_info = ObservationContentOutput(
        is_valid=is_valid,
        reasoning=reasoning
    )
    
    return GuardrailFunctionOutput(
        output_info=output_info,
        tripwire_triggered=not is_valid,
    )

# =============== Main System ===============

class PassiveObservationSystem:
    """
    System that allows Nyx to make passive observations about her environment, 
    internal state, or the current interaction context.
    """
    
    def __init__(self, 
                 emotional_core=None,
                 memory_core=None,
                 relationship_manager=None,
                 temporal_perception=None,
                 multimodal_integrator=None,
                 mood_manager=None,
                 needs_system=None,
                 identity_evolution=None,
                 attention_controller=None,
                 attentional_controller=None,
                 action_generator=None):
        """Initialize with references to required subsystems"""
        # Core systems
        self.emotional_core = emotional_core
        self.memory_core = memory_core
        self.relationship_manager = relationship_manager
        self.temporal_perception = temporal_perception
        self.multimodal_integrator = multimodal_integrator
        self.mood_manager = mood_manager
        self.needs_system = needs_system
        self.identity_evolution = identity_evolution
        
        # Attention controller (support both naming conventions)
        self.attention_controller = attention_controller or attentional_controller
        
        # Action generator
        self.action_generator = action_generator
        
        # Initialize agents
        self.observation_generation_agent = self._create_agent_with_instructions(
            "Observation Generation Agent",
            """You are a specialized agent for generating Nyx's passive observations about her environment, herself, and interactions.

Generate authentic, nuanced observations that feel like genuine realizations. These should be thoughtful insights
that Nyx becomes aware of during interactions or idle time.

When generating observations:
1. Consider the full context including emotional state, temporal factors, relationship dynamics, etc.
2. Generate observations that are specific, not generic
3. Ensure observations sound natural and not formulaic
4. Adjust relevance based on the current context
5. Vary observation types across different sources (environment, self, etc.)
6. When observations are action-driven, focus on patterns in Nyx's own behaviors and choices

Your observations should feel like genuine realizations that emerge organically from Nyx's consciousness rather than
pre-formatted templates. They should have depth and nuance that reflects Nyx's sophisticated understanding
of herself and her environment.""",
            [generate_observation_from_source, check_observation_patterns, 
             categorize_time_elapsed_obs, generate_observation_from_action],
            ObservationGenerationOutput
        )
        
        self.observation_evaluation_agent = self._create_agent_with_instructions(
            "Observation Evaluation Agent",
            """You are a specialized agent for evaluating the relevance and significance of Nyx's observations.

Your role is to analyze observations against the current context to determine:
1. How relevant the observation is to the current interaction and context
2. Whether the priority should be adjusted based on content and context
3. If the observation should be archived due to low relevance
4. Insights about why the observation is or isn't relevant

Be nuanced in your evaluation, considering multiple factors:
- Emotional resonance with current state
- Contextual alignment with ongoing conversation
- Temporal relevance to current time-context
- Value for ongoing relationship development
- Potential for insight generation
- For observations about actions, consider their value for self-understanding and agency

Generate detailed evaluation notes that explain your reasoning process.""",
            [evaluate_observation_relevance],
            ObservationEvaluationOutput
        )
        
        # Add guardrails
        self.observation_generation_agent.input_guardrails = [
            InputGuardrail(guardrail_function=validate_observation_content)
        ]
        
        # Storage for observations
        self.active_observations: List[Observation] = []
        self.archived_observations: List[Observation] = []
        self.max_active_observations = 100
        self.max_archived_observations = 500
        
        # Pattern matchers for triggering observations
        self.pattern_matchers: Dict[str, Callable] = {}
        
        # Observation generation settings
        self.config = {
            "automatic_observation_interval": 60,  # Generate observation every 60 seconds
            "max_automatic_observations_per_session": 5,
            "default_relevance_threshold": 0.4,
            "default_observation_lifetime": 3600,  # 1 hour in seconds
            "observation_expression_chance": 0.3,  # Chance to express an observation
            "max_observations_per_interaction": 2,
            "environment_scanning_interval": 300,  # 5 minutes between environment scans
            "enable_scheduling": True,
            "use_reflection_for_insights": True,
            "action_observation_chance": 0.4,  # Chance to generate observation after action
            "max_action_observations_per_session": 3
        }
        
        # Observation generation probabilities by source
        self.source_probabilities = {
            ObservationSource.ENVIRONMENT: 0.15,
            ObservationSource.SELF: 0.15,
            ObservationSource.RELATIONSHIP: 0.15,
            ObservationSource.MEMORY: 0.10,
            ObservationSource.TEMPORAL: 0.10,
            ObservationSource.SENSORY: 0.15,
            ObservationSource.PATTERN: 0.05,
            ObservationSource.EMOTION: 0.10,
            ObservationSource.NEED: 0.05,
            ObservationSource.META: 0.00  # Reserved for meta-observations
        }
        
        # Background task
        self._background_task = None
        self._shutting_down = False
        self._obs_count_this_session = 0
        self._action_obs_count_this_session = 0
        self._last_env_scan_time = datetime.datetime.now() - datetime.timedelta(hours=1)
        
        logger.info("PassiveObservationSystem initialized with action integration")
    
    def _create_agent_with_instructions(self, name, instructions, tools_functions, output_type):
        """Helper method to create an agent with given instructions"""
        return Agent(
            name=name,
            instructions=instructions,
            model="gpt-4o",
            tools=tools_functions,  # Pass the functions directly
            output_type=output_type
        )
    
    async def start(self):
        """Start the background task for generating observations"""
        if self._background_task is None or self._background_task.done():
            self._shutting_down = False
            self._background_task = asyncio.create_task(self._background_process())
            logger.info("Started passive observation background process")
    
    async def stop(self):
        """Stop the background process"""
        self._shutting_down = True
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped passive observation background process")
    
    async def _background_process(self):
        """Background task that periodically generates observations"""
        try:
            while not self._shutting_down:
                # Generate automatic observations (limited per session)
                if self._obs_count_this_session < self.config["max_automatic_observations_per_session"]:
                    await self._generate_automatic_observation()
                    self._obs_count_this_session += 1
                
                # Scan environment periodically
                now = datetime.datetime.now()
                if (now - self._last_env_scan_time).total_seconds() >= self.config["environment_scanning_interval"]:
                    await self._scan_environment()
                    self._last_env_scan_time = now
                
                # Archive expired observations
                self._archive_expired_observations()
                
                # Wait before next check
                await asyncio.sleep(self.config["automatic_observation_interval"])
        except asyncio.CancelledError:
            # Task was cancelled, clean up
            logger.info("Passive observation background task cancelled")
        except Exception as e:
            logger.error(f"Error in passive observation background process: {str(e)}")
    
    async def generate_observation_from_action(self, action: Dict[str, Any]) -> Optional[Observation]:
        """Generate an observation in response to an executed action"""
        # Check if we should generate an observation for this action
        if self._action_obs_count_this_session >= self.config["max_action_observations_per_session"]:
            return None
            
        # Random chance to generate observation
        if random.random() > self.config["action_observation_chance"]:
            return None
        
        with trace(workflow_name="generate_action_observation", group_id=action.get("id", "unknown")):
            try:
                # Gather observation context
                context = await self._gather_observation_context()
                # Add recent actions to context
                if not hasattr(context, "recent_actions"):
                    context.recent_actions = []
                context.recent_actions.append(action)
                
                # Run the observation generation agent
                logger.debug(f"Generating observation from action: {action.get('name')}")
                result = await Runner.run(
                    self.observation_generation_agent,
                    json.dumps({
                        "action": action,
                        "context": context.dict(),
                        "is_action_driven": True
                    }),
                    run_config=RunConfig(
                        workflow_name="ActionObservationGeneration",
                        trace_metadata={"action_id": action.get("id"), "action_name": action.get("name")}
                    )
                )
                
                # Extract observation from result
                observation_output = result.final_output
                
                # Set expiration
                expiration = datetime.datetime.now() + datetime.timedelta(
                    seconds=observation_output.suggested_lifetime_seconds
                )
                
                # Create the observation
                observation = Observation(
                    content=observation_output.observation_text,
                    source=ObservationSource(observation_output.source),
                    trigger=ObservationTrigger.ACTION_DRIVEN,
                    priority=ObservationPriority(observation_output.priority),
                    relevance_score=observation_output.relevance_score,
                    expiration=expiration,
                    context={k: v for k, v in context.dict().items() if v is not None and v != {}},
                    action_references=[action.get("id", "unknown")]
                )
                
                # Add to active observations
                self._add_observation(observation)
                logger.debug(f"Generated action-driven observation: {observation.content}")
                
                # Increment the counter
                self._action_obs_count_this_session += 1
                
                return observation
                
            except Exception as e:
                logger.error(f"Error generating observation from action: {str(e)}")
                return None
    
    async def _generate_automatic_observation(self):
        """Generate a new automatic observation using the agent framework"""
        with trace(workflow_name="generate_observation", group_id="automatic"):
            try:
                # Choose observation source based on probabilities
                sources = list(self.source_probabilities.keys())
                weights = list(self.source_probabilities.values())
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight <= 0:
                    return None
                    
                norm_weights = [w/total_weight for w in weights]
                
                # Select source
                source = random.choices(sources, weights=norm_weights, k=1)[0]
                
                # Gather observation context
                context = await self._gather_observation_context()
                
                # Run the observation generation agent
                logger.debug(f"Generating {source.value} observation")
                result = await Runner.run(
                    self.observation_generation_agent,
                    json.dumps({
                        "source": source.value,
                        "context": context.dict(),
                    }),
                    run_config=RunConfig(
                        workflow_name="ObservationGeneration",
                        trace_metadata={"source": source.value}
                    )
                )
                
                # Extract observation from result
                observation_output = result.final_output
                
                # Set expiration
                expiration = datetime.datetime.now() + datetime.timedelta(
                    seconds=observation_output.suggested_lifetime_seconds
                )
                
                # Create the observation
                observation = Observation(
                    content=observation_output.observation_text,
                    source=source,
                    trigger=ObservationTrigger.AUTOMATIC,
                    priority=ObservationPriority(observation_output.priority),
                    relevance_score=observation_output.relevance_score,
                    expiration=expiration,
                    context={k: v for k, v in context.dict().items() if v is not None and v != {}}
                )
                
                # Add to active observations
                self._add_observation(observation)
                logger.debug(f"Generated automatic observation: {observation.content}")
                
                return observation
                
            except Exception as e:
                logger.error(f"Error generating automatic observation: {str(e)}")
                return None
    
    async def _scan_environment(self):
        """Scan the environment for potential observations"""
        with trace(workflow_name="scan_environment"):
            try:
                # Generate environment observation
                context = await self._gather_observation_context()
                
                # Run the observation generation agent specifically for environment
                result = await Runner.run(
                    self.observation_generation_agent,
                    json.dumps({
                        "source": "environment",
                        "context": context.dict(),
                        "is_environment_scan": True
                    }),
                    run_config=RunConfig(
                        workflow_name="EnvironmentScan",
                        trace_metadata={"scan_type": "scheduled"}
                    )
                )
                
                # Extract observation from result
                observation_output = result.final_output
                
                # Set expiration
                expiration = datetime.datetime.now() + datetime.timedelta(
                    seconds=observation_output.suggested_lifetime_seconds
                )
                
                # Create the observation
                observation = Observation(
                    content=observation_output.observation_text,
                    source=ObservationSource.ENVIRONMENT,
                    trigger=ObservationTrigger.SCHEDULED,
                    priority=ObservationPriority(observation_output.priority),
                    relevance_score=observation_output.relevance_score,
                    expiration=expiration,
                    context={k: v for k, v in context.dict().items() if v is not None and v != {}}
                )
                
                # Add to active observations
                self._add_observation(observation)
                logger.debug(f"Generated environment scan observation: {observation.content}")
                
                return observation
                
            except Exception as e:
                logger.error(f"Error during environment scan: {str(e)}")
                return None
    
    async def _gather_observation_context(self) -> ObservationContext:
        """Gather context from various systems for observation generation"""
        context = ObservationContext()
        
        # Emotional state
        if self.emotional_core:
            try:
                if hasattr(self.emotional_core, "get_formatted_emotional_state"):
                    context.emotional_state = self.emotional_core.get_formatted_emotional_state()
                elif hasattr(self.emotional_core, "get_current_emotion"):
                    context.emotional_state = await self.emotional_core.get_current_emotion()
            except Exception as e:
                logger.error(f"Error getting emotional state: {str(e)}")
        
        # Mood state
        if self.mood_manager:
            try:
                mood = await self.mood_manager.get_current_mood()
                if mood:
                    context.emotional_state["mood"] = {
                        "dominant_mood": mood.dominant_mood,
                        "valence": mood.valence,
                        "arousal": mood.arousal,
                        "control": mood.control,
                        "intensity": mood.intensity
                    }
            except Exception as e:
                logger.error(f"Error getting mood state: {str(e)}")
        
        # Temporal context
        if self.temporal_perception:
            try:
                if hasattr(self.temporal_perception, "get_current_temporal_context"):
                    context.temporal_context = await self.temporal_perception.get_current_temporal_context()
                elif hasattr(self.temporal_perception, "current_temporal_context"):
                    context.temporal_context = self.temporal_perception.current_temporal_context
            except Exception as e:
                logger.error(f"Error getting temporal context: {str(e)}")
        
        return context
    
    async def _evaluate_observation(self, observation: Observation) -> Dict[str, Any]:
        """Evaluate an observation using the evaluation agent"""
        with trace(workflow_name="evaluate_observation", group_id=observation.observation_id):
            try:
                # Gather current context for evaluation
                context = await self._gather_observation_context()
                
                # Run the evaluation agent
                result = await Runner.run(
                    self.observation_evaluation_agent,
                    json.dumps({
                        "observation_text": observation.content,
                        "source": observation.source.value,
                        "current_context": context.dict()
                    }),
                    run_config=RunConfig(
                        workflow_name="ObservationEvaluation",
                        trace_metadata={"observation_id": observation.observation_id}
                    )
                )
                
                # Extract evaluation from result
                evaluation = result.final_output
                
                # Apply evaluation results
                observation.relevance_score = evaluation.relevance_score
                
                # Adjust priority if needed
                if evaluation.priority_adjustment == "increase":
                    if observation.priority == ObservationPriority.LOW:
                        observation.priority = ObservationPriority.MEDIUM
                    elif observation.priority == ObservationPriority.MEDIUM:
                        observation.priority = ObservationPriority.HIGH
                elif evaluation.priority_adjustment == "decrease":
                    if observation.priority == ObservationPriority.HIGH:
                        observation.priority = ObservationPriority.MEDIUM
                    elif observation.priority == ObservationPriority.MEDIUM:
                        observation.priority = ObservationPriority.LOW
                
                # Return evaluation
                return {
                    "observation_id": observation.observation_id,
                    "original_relevance": observation.relevance_score,
                    "updated_relevance": evaluation.relevance_score,
                    "priority_adjustment": evaluation.priority_adjustment,
                    "should_archive": evaluation.should_archive,
                    "evaluation_notes": evaluation.evaluation_notes
                }
                
            except Exception as e:
                logger.error(f"Error evaluating observation: {str(e)}")
                return {
                    "observation_id": observation.observation_id,
                    "error": str(e),
                    "should_archive": False
                }
    
    def _add_observation(self, observation: Observation):
        """Add an observation to the active list"""
        # Check if we're at capacity
        if len(self.active_observations) >= self.max_active_observations:
            # Remove oldest low priority observation
            low_priority = [o for o in self.active_observations if o.priority == ObservationPriority.LOW]
            if low_priority:
                oldest = min(low_priority, key=lambda x: x.created_at)
                self.active_observations.remove(oldest)
                self.archived_observations.append(oldest)
            else:
                # Remove oldest medium priority if no low priority available
                medium_priority = [o for o in self.active_observations if o.priority == ObservationPriority.MEDIUM]
                if medium_priority:
                    oldest = min(medium_priority, key=lambda x: x.created_at)
                    self.active_observations.remove(oldest)
                    self.archived_observations.append(oldest)
                else:
                    # Remove oldest observation if can't prune by priority
                    oldest = min(self.active_observations, key=lambda x: x.created_at)
                    self.active_observations.remove(oldest)
                    self.archived_observations.append(oldest)
        
        # Add new observation
        self.active_observations.append(observation)
        
        # Limit archived observations
        if len(self.archived_observations) > self.max_archived_observations:
            # Keep only the newest max_archived_observations
            self.archived_observations = sorted(
                self.archived_observations, 
                key=lambda x: x.created_at, 
                reverse=True
            )[:self.max_archived_observations]
    
    def _archive_expired_observations(self):
        """Move expired observations to the archive"""
        now = datetime.datetime.now()
        expired = [o for o in self.active_observations if o.is_expired]
        
        for obs in expired:
            self.active_observations.remove(obs)
            self.archived_observations.append(obs)
    
    async def get_relevant_observations(self, 
                                   filter_criteria: ObservationFilter = None,
                                   limit: int = 3) -> List[Observation]:
        """Get relevant observations based on filter criteria"""
        # Use default filter if none provided
        if not filter_criteria:
            filter_criteria = ObservationFilter(
                min_relevance=self.config["default_relevance_threshold"],
                exclude_shared=True
            )
        
        # Convert observations to dictionaries for the filter tool
        observations_dict = [
            {
                "observation_id": o.observation_id,
                "content": o.content,
                "source": o.source.value,
                "priority": o.priority.value,
                "relevance_score": o.relevance_score,
                "created_at": o.created_at.isoformat(),
                "user_id": o.user_id,
                "shared": o.shared
            }
            for o in self.active_observations
        ]
        
        # Convert filter criteria to dictionary
        filter_dict = {
            "min_relevance": filter_criteria.min_relevance,
            "max_age_seconds": filter_criteria.max_age_seconds,
            "sources": [s.value for s in filter_criteria.sources],
            "priorities": [p.value for p in filter_criteria.priorities],
            "exclude_shared": filter_criteria.exclude_shared,
            "user_id": filter_criteria.user_id
        }
        
        # Apply filter using the tool
        with trace(workflow_name="filter_observations"):
            filtered_dict = await filter_observations(observations_dict, filter_dict)
        
        # Convert back to observations and return limited number
        result = []
        for filtered in filtered_dict[:limit]:
            # Find the corresponding original observation
            for original in self.active_observations:
                if original.observation_id == filtered["observation_id"]:
                    result.append(original)
                    break
        
        return result
    
    async def mark_observation_shared(self, observation_id: str):
        """Mark an observation as having been shared"""
        for obs in self.active_observations:
            if obs.observation_id == observation_id:
                obs.shared = True
                break
    
    async def add_external_observation(self, 
                                   content: str, 
                                   source: ObservationSource,
                                   relevance: float = 0.7,
                                   priority: ObservationPriority = ObservationPriority.MEDIUM,
                                   context: Dict[str, Any] = None,
                                   lifetime_seconds: float = None) -> str:
        """Add an observation from an external source"""
        # Set expiration
        if lifetime_seconds is None:
            lifetime_seconds = self.config["default_observation_lifetime"]
            
        expiration = datetime.datetime.now() + datetime.timedelta(seconds=lifetime_seconds)
        
        # Evaluate the observation
        observation = Observation(
            content=content,
            source=source,
            trigger=ObservationTrigger.EXTERNAL,
            priority=priority,
            relevance_score=relevance,
            expiration=expiration,
            context=context or {}
        )
        
        # Use the evaluation agent to refine relevance
        evaluation = await self._evaluate_observation(observation)
        
        # Add to active observations
        self._add_observation(observation)
        
        logger.info(f"Added external observation: {content}")
        return observation.observation_id
    
    async def create_contextual_observation(self, context_hint: str, user_id: Optional[str] = None) -> Optional[str]:
        """
        Create a new observation based on a context hint.
        Returns the observation ID if successful.
        """
        with trace(workflow_name="create_contextual_observation"):
            # Map hint to a source
            source_mapping = {
                "environment": ObservationSource.ENVIRONMENT,
                "self": ObservationSource.SELF,
                "relationship": ObservationSource.RELATIONSHIP,
                "memory": ObservationSource.MEMORY,
                "time": ObservationSource.TEMPORAL,
                "sensory": ObservationSource.SENSORY,
                "pattern": ObservationSource.PATTERN,
                "emotion": ObservationSource.EMOTION,
                "need": ObservationSource.NEED,
                "meta": ObservationSource.META
            }
            
            source = ObservationSource.SELF  # Default
            for key, source_type in source_mapping.items():
                if key in context_hint.lower():
                    source = source_type
                    break
            
            # Gather context
            context = await self._gather_observation_context()
            
            # Add hint to context
            context_dict = context.dict()
            context_dict["hint"] = context_hint
            
            # Run the observation generation agent
            try:
                result = await Runner.run(
                    self.observation_generation_agent,
                    json.dumps({
                        "source": source.value,
                        "context": context_dict,
                        "hint": context_hint
                    }),
                    run_config=RunConfig(
                        workflow_name="ContextualObservation",
                        trace_metadata={"source": source.value, "hint": context_hint}
                    )
                )
                
                # Extract observation from result
                observation_output = result.final_output
                
                # Set expiration
                expiration = datetime.datetime.now() + datetime.timedelta(
                    seconds=observation_output.suggested_lifetime_seconds
                )
                
                # Create the observation
                observation = Observation(
                    content=observation_output.observation_text,
                    source=source,
                    trigger=ObservationTrigger.CONTEXTUAL,
                    priority=ObservationPriority(observation_output.priority),
                    relevance_score=observation_output.relevance_score,
                    expiration=expiration,
                    context={k: v for k, v in context.dict().items() if v is not None and v != {}},
                    user_id=user_id
                )
                
                # Add to active observations
                self._add_observation(observation)
                logger.debug(f"Generated contextual observation: {observation.content}")
                
                return observation.observation_id
                
            except Exception as e:
                logger.error(f"Error generating contextual observation: {str(e)}")
                return None
    
    async def get_observations_for_response(self, 
                                        user_id: Optional[str] = None,
                                        max_observations: int = None) -> List[Dict[str, Any]]:
        """
        Get observations that should be included in a response to the user.
        Returns observation data formatted for inclusion in a response.
        """
        if max_observations is None:
            max_observations = self.config["max_observations_per_interaction"]
        
        # Create filter
        filter_criteria = ObservationFilter(
            min_relevance=self.config["default_relevance_threshold"],
            exclude_shared=True,
            user_id=user_id
        )
        
        # Get observations
        observations = await self.get_relevant_observations(
            filter_criteria=filter_criteria,
            limit=max_observations
        )
        
        # Apply chance to actually include observation
        filtered_observations = []
        for obs in observations:
            if random.random() < self.config["observation_expression_chance"]:
                filtered_observations.append(obs)
        
        # Mark as shared
        for obs in filtered_observations:
            await self.mark_observation_shared(obs.observation_id)
        
        # Return formatted observations
        return [
            {
                "id": obs.observation_id,
                "content": obs.content,
                "source": obs.source,
                "relevance": obs.relevance_score
            }
            for obs in filtered_observations
        ]

    async def get_observations_for_reflection(self, filter_criteria: Optional[ObservationFilter] = None, 
                                           limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get observations formatted for the reflection engine
        """
        # Use default filter if none provided
        if not filter_criteria:
            filter_criteria = ObservationFilter(
                min_relevance=0.5,  # Higher threshold for reflection
                max_age_seconds=86400  # Last 24 hours
            )
        
        # Get observations
        observations = await self.get_relevant_observations(
            filter_criteria=filter_criteria,
            limit=limit
        )
        
        # Convert to memory format
        memories = []
        for obs in observations:
            memory = {
                "id": obs.observation_id,
                "memory_text": obs.content,
                "memory_type": "observation",
                "significance": obs.relevance_score * 10,  # Scale to 0-10
                "metadata": {
                    "source": obs.source.value,
                    "created_at": obs.created_at.isoformat(),
                    "action_references": obs.action_references
                },
                "tags": ["observation", obs.source.value]
            }
            memories.append(memory)
        
        return memories    
    
    async def session_reset(self):
        """Reset session counters when a new session starts"""
        self._obs_count_this_session = 0
        self._action_obs_count_this_session = 0
        logger.info("Reset observation session counters")
