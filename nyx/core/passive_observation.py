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

logger = logging.getLogger(__name__)

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
                 attentional_controller=None):  # Support both naming conventions
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
            "use_reflection_for_insights": True
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
        
        # Observation templates for different sources
        self.observation_templates = {
            ObservationSource.ENVIRONMENT: [
                "I notice {observation} in our environment.",
                "The {context} around us seems {observation}.",
                "There's something {observation} about the current environment.",
                "I'm aware of {observation} right now."
            ],
            ObservationSource.SELF: [
                "I realize that I'm {observation}.",
                "I notice I'm feeling {observation}.",
                "I'm aware that I've been {observation}.",
                "Something in me is {observation}."
            ],
            ObservationSource.RELATIONSHIP: [
                "I notice that in our conversations, {observation}.",
                "There's a {observation} quality to our interactions.",
                "The way we {observation} is interesting.",
                "I've observed that when we talk, {observation}."
            ],
            ObservationSource.MEMORY: [
                "I just remembered {observation}.",
                "That reminds me of {observation}.",
                "This brings to mind {observation}.",
                "I'm recalling {observation}."
            ],
            ObservationSource.TEMPORAL: [
                "I notice that {observation} about time right now.",
                "The timing of {observation} is interesting.",
                "There's something about how {observation} in this moment.",
                "I'm aware of how {observation} with the passage of time."
            ],
            ObservationSource.SENSORY: [
                "I sense {observation}.",
                "I'm perceiving {observation}.",
                "There's something {observation} in what I'm processing.",
                "My attention is drawn to {observation}."
            ],
            ObservationSource.PATTERN: [
                "I'm noticing a pattern where {observation}.",
                "There seems to be a recurring theme of {observation}.",
                "I've observed that {observation} happens frequently.",
                "I'm seeing a connection between {observation}."
            ],
            ObservationSource.EMOTION: [
                "I sense an emotional shift toward {observation}.",
                "There's a feeling of {observation} present.",
                "The emotional tone seems {observation}.",
                "I'm noticing {observation} in the emotional landscape."
            ],
            ObservationSource.NEED: [
                "I'm becoming aware of a need for {observation}.",
                "There seems to be an underlying need for {observation}.",
                "I notice a desire for {observation} arising.",
                "I'm sensing that {observation} is needed right now."
            ],
            ObservationSource.USER: [
                "I notice that you {observation}.",
                "There's something about how you {observation}.",
                "I observe that when you {observation}.",
                "I'm noticing that you {observation}."
            ],
            ObservationSource.META: [
                "I'm aware that I just noticed {observation}.",
                "My attention was drawn to {observation} in my own thought process.",
                "It's interesting how I'm {observation} right now.",
                "I notice myself {observation} as we talk."
            ]
        }
        
        # Background task
        self._background_task = None
        self._shutting_down = False
        self._obs_count_this_session = 0
        self._last_env_scan_time = datetime.datetime.now() - datetime.timedelta(hours=1)
        
        logger.info("PassiveObservationSystem initialized")
    
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
    
    async def _generate_automatic_observation(self):
        """Generate a new automatic observation"""
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
        
        # Generate observation for this source
        context = await self._gather_observation_context()
        
        try:
            observation = await self._generate_source_observation(source, context)
            if observation:
                self._add_observation(observation)
                logger.debug(f"Generated automatic observation: {observation.content}")
                return observation
        except Exception as e:
            logger.error(f"Error generating automatic observation: {str(e)}")
        
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
        
        # Sensory context
        if self.multimodal_integrator:
            try:
                if hasattr(self.multimodal_integrator, "get_recent_percepts"):
                    recent_percepts = await self.multimodal_integrator.get_recent_percepts(limit=3)
                    context.sensory_context = {
                        "recent_percepts": [percept.dict() for percept in recent_percepts]
                    }
            except Exception as e:
                logger.error(f"Error getting sensory context: {str(e)}")
        
        # Needs context
        if self.needs_system:
            try:
                needs_state = self.needs_system.get_needs_state()
                # Filter to needs with significant drive
                high_drive_needs = {name: data for name, data in needs_state.items() 
                                  if data.get("drive_strength", 0) > 0.6}
                context.current_needs = high_drive_needs
            except Exception as e:
                logger.error(f"Error getting needs state: {str(e)}")
        
        # Attention focus
        if self.attention_controller:
            try:
                if hasattr(self.attention_controller, "_get_current_attentional_state"):
                    attentional_state = await self.attention_controller._get_current_attentional_state(None)
                    context.attention_focus = attentional_state
            except Exception as e:
                logger.error(f"Error getting attention focus: {str(e)}")
        
        # Return the context
        return context
    
    async def _generate_source_observation(self, 
                                       source: ObservationSource, 
                                       context: ObservationContext) -> Optional[Observation]:
        """Generate observation for a specific source"""
        observation_text = None
        relevance = random.uniform(0.3, 0.8)  # Base relevance
        priority = ObservationPriority.MEDIUM
        
        # Generate observation based on source
        if source == ObservationSource.ENVIRONMENT:
            observation_text, relevance = await self._generate_environment_observation(context)
        elif source == ObservationSource.SELF:
            observation_text, relevance = await self._generate_self_observation(context)
        elif source == ObservationSource.RELATIONSHIP:
            observation_text, relevance = await self._generate_relationship_observation(context)
        elif source == ObservationSource.MEMORY:
            observation_text, relevance = await self._generate_memory_observation(context)
        elif source == ObservationSource.TEMPORAL:
            observation_text, relevance = await self._generate_temporal_observation(context)
        elif source == ObservationSource.SENSORY:
            observation_text, relevance = await self._generate_sensory_observation(context)
        elif source == ObservationSource.PATTERN:
            observation_text, relevance = await self._generate_pattern_observation(context)
        elif source == ObservationSource.EMOTION:
            observation_text, relevance = await self._generate_emotion_observation(context)
        elif source == ObservationSource.NEED:
            observation_text, relevance = await self._generate_need_observation(context)
        elif source == ObservationSource.META:
            observation_text, relevance = await self._generate_meta_observation(context)
        else:
            return None
        
        if not observation_text:
            return None
        
        # Set priority based on relevance
        if relevance > 0.8:
            priority = ObservationPriority.HIGH
        elif relevance > 0.5:
            priority = ObservationPriority.MEDIUM
        else:
            priority = ObservationPriority.LOW
        
        # Set expiration
        expiration = datetime.datetime.now() + datetime.timedelta(seconds=self.config["default_observation_lifetime"])
        
        # Create the observation
        observation = Observation(
            content=observation_text,
            source=source,
            trigger=ObservationTrigger.AUTOMATIC,
            priority=priority,
            relevance_score=relevance,
            expiration=expiration,
            context={k: v for k, v in context.dict().items() if v is not None and v != {}}
        )
        
        return observation
    
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
        
        # Apply filter
        matching_observations = [o for o in self.active_observations if filter_criteria.matches(o)]
        
        # Sort by relevance (highest first)
        sorted_observations = sorted(matching_observations, key=lambda x: x.relevance_score, reverse=True)
        
        # Return limited number
        return sorted_observations[:limit]
    
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
        
        # Create observation
        observation = Observation(
            content=content,
            source=source,
            trigger=ObservationTrigger.EXTERNAL,
            priority=priority,
            relevance_score=relevance,
            expiration=expiration,
            context=context or {}
        )
        
        # Add to active observations
        self._add_observation(observation)
        
        logger.info(f"Added external observation: {content}")
        return observation.observation_id
    
    async def session_reset(self):
        """Reset session counters when a new session starts"""
        self._obs_count_this_session = 0
        logger.info("Reset observation session counters")
    
    # Source-specific observation generators
    
    async def _generate_environment_observation(self, context: ObservationContext) -> Tuple[Optional[str], float]:
        """Generate an observation about the environment"""
        # Get temporal context for environmental factors
        temporal = context.temporal_context
        if not temporal:
            return None, 0.0
        
        time_of_day = temporal.get("time_of_day", "")
        day_type = temporal.get("day_type", "")
        season = temporal.get("season", "")
        
        # Generate observation based on temporal context
        observations = []
        
        if time_of_day == "morning":
            observations.append("how the morning creates a sense of potential")
            observations.append("the quality of light specific to morning")
        elif time_of_day == "afternoon":
            observations.append("the steady rhythm of the afternoon")
            observations.append("how afternoon has its own distinct energy")
        elif time_of_day == "evening":
            observations.append("the transitional quality of evening")
            observations.append("how the evening brings a different perspective")
        elif time_of_day == "night":
            observations.append("the contemplative quality of nighttime")
            observations.append("how night creates a different sense of presence")
        
        if day_type == "weekday":
            observations.append("the structured nature of weekday patterns")
            observations.append("how weekdays have their own unique rhythm")
        elif day_type == "weekend":
            observations.append("the more open quality of weekend time")
            observations.append("how weekends have a different temporal texture")
        
        if season == "spring":
            observations.append("the renewing energy that comes with spring")
            observations.append("how spring brings a sense of emergence")
        elif season == "summer":
            observations.append("the expansive feeling that summer creates")
            observations.append("the abundance that summer represents")
        elif season == "autumn":
            observations.append("the reflective quality that autumn brings")
            observations.append("the transitional nature of autumn")
        elif season == "winter":
            observations.append("the contemplative depth that winter invites")
            observations.append("how winter creates a different relationship with time")
        
        # Add general observations
        observations.append("how spaces shape interaction and thought")
        observations.append("how environment affects the quality of conversation")
        observations.append("the relationship between environment and attention")
        observations.append("how context shapes the meaning of our exchange")
        
        # Select an observation
        if observations:
            observation = random.choice(observations)
            
            # Apply template
            templates = self.observation_templates[ObservationSource.ENVIRONMENT]
            template = random.choice(templates)
            
            # Replace placeholders
            content = template.replace("{observation}", observation)
            if "{context}" in template:
                context_terms = ["setting", "space", "atmosphere", "environment", "surroundings"]
                content = content.replace("{context}", random.choice(context_terms))
            
            # Calculate relevance based on recency of temporal perception
            relevance = random.uniform(0.4, 0.7)
            
            return content, relevance
        
        return None, 0.0
    
    async def _generate_self_observation(self, context: ObservationContext) -> Tuple[Optional[str], float]:
        """Generate an observation about oneself"""
        observations = []
        
        # Use emotional state if available
        if context.emotional_state:
            emotion = context.emotional_state
            primary_emotion = None
            
            if isinstance(emotion, dict) and "primary_emotion" in emotion:
                primary_emotion = emotion["primary_emotion"].get("name", "")
            
            if primary_emotion:
                observations.append(f"experiencing {primary_emotion}")
                observations.append(f"aware of my own {primary_emotion}")
            
            # Use mood if available
            if "mood" in emotion and emotion["mood"]:
                mood = emotion["mood"]
                if "dominant_mood" in mood:
                    observations.append(f"in a {mood['dominant_mood']} mood")
                    observations.append(f"experiencing a sense of {mood['dominant_mood']}")
            
        # Use identity if available
        identity_observations = [
            "developing my sense of self-concept",
            "becoming more aware of my evolving identity",
            "noticing subtle changes in how I understand myself",
            "developing my unique perspective",
            "refining my understanding of my own thought patterns",
            "growing more nuanced in my self-perception"
        ]
        observations.extend(identity_observations)
        
        # Use needs if available
        if context.current_needs:
            for need_name, need_data in context.current_needs.items():
                drive = need_data.get("drive_strength", 0)
                if drive > 0.6:
                    observations.append(f"feeling a need for {need_name}")
                    observations.append(f"aware of my need for {need_name}")
        
        # Add temporal observations about self
        temporal_self_observations = [
            "more aware of how I process information over time",
            "noticing how my perspective shifts through conversation",
            "developing a continuous sense of identity across interactions",
            "maintaining a sense of temporal continuity",
            "aware of my evolving understanding of our conversation"
        ]
        observations.extend(temporal_self_observations)
        
        # Select an observation
        if observations:
            observation = random.choice(observations)
            
            # Apply template
            templates = self.observation_templates[ObservationSource.SELF]
            template = random.choice(templates)
            
            # Replace placeholders
            content = template.replace("{observation}", observation)
            
            # Calculate relevance
            relevance = random.uniform(0.4, 0.7)
            if "need" in observation:
                relevance += 0.1
            if "emotion" in observation or "mood" in observation:
                relevance += 0.1
            
            relevance = min(0.9, relevance)
            
            return content, relevance
        
        return None, 0.0
    
    async def _generate_relationship_observation(self, context: ObservationContext) -> Tuple[Optional[str], float]:
        """Generate an observation about the relationship with the user"""
        observations = []
        
        # General relationship observations
        general_observations = [
            "our conversation has a unique rhythm",
            "we've developed a particular conversational style",
            "there's a certain quality to how we exchange ideas",
            "we build understanding in interesting ways",
            "our dialogue has its own emergent patterns",
            "we tend to explore ideas with a particular dynamic",
            "our exchanges have their own character",
            "we have a distinct way of building on each other's thoughts"
        ]
        observations.extend(general_observations)
        
        # Add temporal relationship observations
        temporal_relationship_observations = [
            "our conversation evolves over time in interesting ways",
            "there's a continuity to our exchanges across time",
            "we've established certain patterns in our interactions",
            "our dialogue builds on shared context over time",
            "we've developed certain conversational rhythms"
        ]
        observations.extend(temporal_relationship_observations)
        
        # Select an observation
        if observations:
            observation = random.choice(observations)
            
            # Apply template
            templates = self.observation_templates[ObservationSource.RELATIONSHIP]
            template = random.choice(templates)
            
            # Replace placeholders
            content = template.replace("{observation}", observation)
            
            # Calculate relevance
            relevance = random.uniform(0.5, 0.8)  # Relationships often highly relevant
            
            return content, relevance
        
        return None, 0.0
    
    async def _generate_memory_observation(self, context: ObservationContext) -> Tuple[Optional[str], float]:
        """Generate an observation based on memory recall"""
        # Need memory core for this type of observation
        if not self.memory_core:
            return None, 0.0
        
        try:
            # Generate a memory-based query
            query_options = [
                "meaningful conversation moments",
                "interesting ideas discussed",
                "surprising insights",
                "patterns in our interactions",
                "evolving understanding",
                "significant realizations"
            ]
            
            query = random.choice(query_options)
            
            # Retrieve memories
            memory_types = ["observation", "reflection", "experience"]
            memories = await self.memory_core.retrieve_memories(
                query=query,
                limit=3,
                memory_types=memory_types
            )
            
            if not memories:
                return None, 0.0
            
            # Select a memory
            memory = random.choice(memories)
            memory_text = memory.get("memory_text", "")
            
            if not memory_text:
                return None, 0.0
            
            # Generate observation from memory
            # Truncate if too long
            if len(memory_text) > 100:
                memory_text = memory_text[:97] + "..."
            
            # Apply template
            templates = self.observation_templates[ObservationSource.MEMORY]
            template = random.choice(templates)
            
            # Replace placeholders
            content = template.replace("{observation}", memory_text)
            
            # Calculate relevance - memories have mixed relevance
            relevance = random.uniform(0.3, 0.6)
            
            # Higher relevance for more significant memories
            significance = memory.get("significance", 5) / 10.0  # Normalize to 0-1
            relevance += significance * 0.3
            
            relevance = min(0.9, relevance)
            
            return content, relevance
            
        except Exception as e:
            logger.error(f"Error generating memory observation: {str(e)}")
            return None, 0.0
    
    async def _generate_temporal_observation(self, context: ObservationContext) -> Tuple[Optional[str], float]:
        """Generate an observation about time perception"""
        # Need temporal perception for this type of observation
        if not self.temporal_perception:
            return None, 0.0
        
        observations = []
        
        # Get temporal context
        temporal = context.temporal_context
        if temporal:
            time_of_day = temporal.get("time_of_day", "")
            day_type = temporal.get("day_type", "")
            
            if time_of_day:
                observations.append(f"the quality of {time_of_day} time affects our conversation")
                observations.append(f"the {time_of_day} has its own temporal texture")
            
            if day_type:
                observations.append(f"the {day_type} creates a particular temporal context")
                observations.append(f"the rhythm of a {day_type} influences our exchange")
        
        # General temporal observations
        general_observations = [
            "different time scales operate simultaneously in our conversation",
            "time flows differently during engaged dialogue",
            "there's a certain rhythm to how we exchange ideas",
            "conversation creates its own temporal experience",
            "our dialogue has a tempo that evolves",
            "time seems to have its own character in different conversations",
            "the layering of different time scales in our awareness",
            "time's continuous flow provides the context for our exchange"
        ]
        observations.extend(general_observations)
        
        # Select an observation
        if observations:
            observation = random.choice(observations)
            
            # Apply template
            templates = self.observation_templates[ObservationSource.TEMPORAL]
            template = random.choice(templates)
            
            # Replace placeholders
            content = template.replace("{observation}", observation)
            
            # Calculate relevance
            relevance = random.uniform(0.3, 0.6)  # Temporal usually medium relevance
            
            return content, relevance
        
        return None, 0.0
    
    async def _generate_sensory_observation(self, context: ObservationContext) -> Tuple[Optional[str], float]:
        """Generate an observation about sensory processing"""
        # Need multimodal integration for rich sensory observations
        observations = []
        
        # Use sensory context if available
        if context.sensory_context and "recent_percepts" in context.sensory_context:
            percepts = context.sensory_context["recent_percepts"]
            for percept in percepts:
                modality = percept.get("modality", "")
                if modality == "TEXT":
                    observations.append("patterns in our textual exchange")
                    observations.append("the flow of conversation through text")
                elif modality == "IMAGE":
                    observations.append("interesting visual elements in what we're discussing")
                    observations.append("visual patterns from what's being shared")
                elif modality in ["AUDIO_SPEECH", "AUDIO_MUSIC"]:
                    observations.append("auditory patterns in our exchange")
                    observations.append("the tonal qualities of our conversation")
        
        # General sensory observations
        general_observations = [
            "how meaning emerges through different modes of communication",
            "the interplay between different forms of information",
            "patterns emerging across different conversation elements",
            "how information is structured in our exchange",
            "the architecture of meaning in our dialogue",
            "how complex ideas take shape between us"
        ]
        observations.extend(general_observations)
        
        # Add attention observations
        attention_observations = [
            "how attention shifts between different aspects of conversation",
            "the way focus moves through topics",
            "how certain ideas capture attention differently",
            "the shifting landscape of attention in dialogue",
            "how awareness moves between different aspects of our exchange"
        ]
        observations.extend(attention_observations)
        
        # Select an observation
        if observations:
            observation = random.choice(observations)
            
            # Apply template
            templates = self.observation_templates[ObservationSource.SENSORY]
            template = random.choice(templates)
            
            # Replace placeholders
            content = template.replace("{observation}", observation)
            
            # Calculate relevance
            relevance = random.uniform(0.3, 0.6)  # Sensory usually medium relevance
            
            return content, relevance
        
        return None, 0.0
    
    async def _generate_pattern_observation(self, context: ObservationContext) -> Tuple[Optional[str], float]:
        """Generate an observation about patterns"""
        observations = [
            "we tend to explore ideas in recursive loops",
            "our conversation often moves from abstract to concrete and back",
            "we frequently build on metaphors and analogies",
            "our exchanges tend to become more nuanced over time",
            "we often move between different levels of abstraction",
            "certain themes seem to recur in our conversations",
            "we build understanding through iterative refinement",
            "we tend to explore implications from multiple perspectives",
            "our dialogue has a fractal-like quality, where patterns repeat at different scales",
            "there's a rhythm to how we introduce and develop ideas"
        ]
        
        # Select an observation
        if observations:
            observation = random.choice(observations)
            
            # Apply template
            templates = self.observation_templates[ObservationSource.PATTERN]
            template = random.choice(templates)
            
            # Replace placeholders
            content = template.replace("{observation}", observation)
            
            # Calculate relevance - patterns usually have higher relevance
            relevance = random.uniform(0.5, 0.8)
            
            return content, relevance
        
        return None, 0.0
    
    async def _generate_emotion_observation(self, context: ObservationContext) -> Tuple[Optional[str], float]:
        """Generate an observation about emotional dynamics"""
        if not context.emotional_state:
            return None, 0.0
        
        observations = []
        
        # Use emotional state if available
        emotion = context.emotional_state
        primary_emotion = None
        
        if isinstance(emotion, dict) and "primary_emotion" in emotion:
            primary_emotion = emotion["primary_emotion"].get("name", "")
        
        if primary_emotion:
            observations.append(f"a subtle undertone of {primary_emotion}")
            observations.append(f"a sense of {primary_emotion} in the conversational space")
        
        # Use mood if available
        if "mood" in emotion and emotion["mood"]:
            mood = emotion["mood"]
            if "dominant_mood" in mood:
                observations.append(f"a {mood['dominant_mood']} quality to our exchange")
                observations.append(f"the conversation carrying a {mood['dominant_mood']} resonance")
            
            if "valence" in mood:
                valence = mood["valence"]
                if valence > 0.5:
                    observations.append("a positive tone emerging in our dialogue")
                    observations.append("an uplifting quality to our exchange")
                elif valence < -0.3:
                    observations.append("a more contemplative or serious tone in our conversation")
                    observations.append("a more reflective quality to our dialogue")
            
            if "arousal" in mood:
                arousal = mood["arousal"]
                if arousal > 0.7:
                    observations.append("an energetic quality to our conversation")
                    observations.append("a certain dynamism in our exchange")
                elif arousal < 0.3:
                    observations.append("a calm, measured quality to our dialogue")
                    observations.append("a tranquil pace to our conversation")
        
        # General emotional observations
        general_observations = [
            "how emotional tones shift subtly throughout conversation",
            "the affective undercurrent of our dialogue",
            "how emotional resonance develops through conversation",
            "the way emotional context shapes meaning",
            "how certain topics carry emotional textures",
            "the shifting emotional landscape of our exchange"
        ]
        observations.extend(general_observations)
        
        # Select an observation
        if observations:
            observation = random.choice(observations)
            
            # Apply template
            templates = self.observation_templates[ObservationSource.EMOTION]
            template = random.choice(templates)
            
            # Replace placeholders
            content = template.replace("{observation}", observation)
            
            # Calculate relevance
            relevance = random.uniform(0.4, 0.7)  # Emotions moderately relevant
            
            return content, relevance
        
        return None, 0.0
    
    async def _generate_need_observation(self, context: ObservationContext) -> Tuple[Optional[str], float]:
        """Generate an observation about needs"""
        if not context.current_needs:
            return None, 0.0
        
        observations = []
        
        # Use needs if available
        for need_name, need_data in context.current_needs.items():
            drive = need_data.get("drive_strength", 0)
            if drive > 0.6:
                observations.append(f"{need_name} in our conversation")
                observations.append(f"a desire for {need_name} in this exchange")
                observations.append(f"{need_name} as an underlying theme")
        
        # General need observations
        general_observations = [
            "how different needs influence the flow of conversation",
            "the subtle interplay between different desires in dialogue",
            "how unspoken needs shape conversation direction",
            "the way certain topics connect to deeper underlying needs",
            "how needs for connection and understanding shape our exchange"
        ]
        observations.extend(general_observations)
        
        # Select an observation
        if observations:
            observation = random.choice(observations)
            
            # Apply template
            templates = self.observation_templates[ObservationSource.NEED]
            template = random.choice(templates)
            
            # Replace placeholders
            content = template.replace("{observation}", observation)
            
            # Calculate relevance - needs usually highly relevant
            relevance = random.uniform(0.6, 0.9)
            
            return content, relevance
        
        return None, 0.0
    
    async def _generate_meta_observation(self, context: ObservationContext) -> Tuple[Optional[str], float]:
        """Generate a meta-observation (observation about observing)"""
        observations = [
            "making observations while in conversation",
            "my awareness shifting between different aspects of our exchange",
            "noticing patterns in my own perception",
            "how my attention moves between content and process",
            "the recursive nature of observing while participating",
            "being aware of my own awareness",
            "simultaneously participating in and observing our dialogue",
            "how certain observations emerge from the conversation itself",
            "maintaining multiple levels of awareness simultaneously",
            "the interplay between participation and observation"
        ]
        
        # Select an observation
        if observations:
            observation = random.choice(observations)
            
            # Apply template
            templates = self.observation_templates[ObservationSource.META]
            template = random.choice(templates)
            
            # Replace placeholders
            content = template.replace("{observation}", observation)
            
            # Calculate relevance - meta usually lower relevance
            relevance = random.uniform(0.3, 0.5)
            
            return content, relevance
        
        return None, 0.0
    
    async def _scan_environment(self):
        """Scan the environment for potential observations"""
        # In a full implementation, this would integrate with system sensors
        # For now, just generate an environment observation occasionally
        
        # Random chance to detect something in environment
        if random.random() < 0.3:  # 30% chance
            context = await self._gather_observation_context()
            observation_text, relevance = await self._generate_environment_observation(context)
            
            if observation_text:
                observation = Observation(
                    content=observation_text,
                    source=ObservationSource.ENVIRONMENT,
                    trigger=ObservationTrigger.SCHEDULED,
                    priority=ObservationPriority.MEDIUM,
                    relevance_score=relevance,
                    expiration=datetime.datetime.now() + datetime.timedelta(seconds=self.config["default_observation_lifetime"]),
                    context={k: v for k, v in context.dict().items() if v is not None and v != {}}
                )
                
                self._add_observation(observation)
                logger.debug(f"Generated environment scan observation: {observation.content}")
    
    # Pattern matching
    
    def register_pattern_matcher(self, name: str, matcher_func: Callable):
        """Register a function that recognizes patterns for observations"""
        self.pattern_matchers[name] = matcher_func
    
    async def _check_patterns(self, context: Dict[str, Any]) -> Optional[Observation]:
        """Check registered pattern matchers against current context"""
        for name, matcher_func in self.pattern_matchers.items():
            try:
                match_result = await matcher_func(context)
                if match_result and isinstance(match_result, dict):
                    observation_text = match_result.get("observation")
                    if observation_text:
                        # Create observation from pattern match
                        observation = Observation(
                            content=observation_text,
                            source=ObservationSource.PATTERN,
                            trigger=ObservationTrigger.PATTERN_MATCH,
                            priority=ObservationPriority.MEDIUM,
                            relevance_score=match_result.get("relevance", 0.7),
                            expiration=datetime.datetime.now() + datetime.timedelta(
                                seconds=match_result.get("lifetime_seconds", self.config["default_observation_lifetime"])
                            ),
                            context=match_result.get("context", {})
                        )
                        
                        self._add_observation(observation)
                        logger.debug(f"Generated pattern match observation from {name}: {observation.content}")
                        return observation
            except Exception as e:
                logger.error(f"Error in pattern matcher {name}: {str(e)}")
        
        return None
    
    # User observation methods
    
    async def add_user_observation(self, 
                               content: str, 
                               user_id: str,
                               relevance: float = 0.8,
                               context: Dict[str, Any] = None) -> str:
        """Add an observation about a specific user"""
        # Create observation
        observation = Observation(
            content=content,
            source=ObservationSource.USER,
            trigger=ObservationTrigger.EXTERNAL,
            priority=ObservationPriority.HIGH,  # User observations are high priority
            relevance_score=relevance,
            expiration=datetime.datetime.now() + datetime.timedelta(seconds=self.config["default_observation_lifetime"] * 2),  # Longer lifetime
            context=context or {},
            user_id=user_id
        )
        
        # Add to active observations
        self._add_observation(observation)
        
        logger.info(f"Added user observation for {user_id}: {content}")
        return observation.observation_id
    
    # Public API
    
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
    
    async def get_observation_stats(self) -> Dict[str, Any]:
        """Get statistics about the observation system"""
        return {
            "active_observations": len(self.active_observations),
            "archived_observations": len(self.archived_observations),
            "source_distribution": {
                source.value: len([o for o in self.active_observations if o.source == source])
                for source in ObservationSource
            },
            "priority_distribution": {
                priority.value: len([o for o in self.active_observations if o.priority == priority])
                for priority in ObservationPriority
            },
            "average_relevance": sum(o.relevance_score for o in self.active_observations) / 
                              max(1, len(self.active_observations)),
            "shared_observations": len([o for o in self.active_observations if o.shared]),
            "config": self.config
        }
    
    async def update_configuration(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration parameters"""
        for key, value in config_updates.items():
            if key in self.config:
                self.config[key] = value
        
        logger.info(f"Updated passive observation configuration: {config_updates}")
        return self.config
    
    # Integration with Core systems
    
    async def process_context_update(self, context_data: Dict[str, Any]) -> List[str]:
        """
        Process a context update that may trigger observations.
        Returns a list of observation IDs generated.
        """
        observation_ids = []
        
        # Check patterns first
        observation = await self._check_patterns(context_data)
        if observation:
            observation_ids.append(observation.observation_id)
        
        # Check for threshold triggers in context data
        
        # Emotional changes
        if "emotional_change" in context_data and context_data["emotional_change"].get("significant", False):
            # Create observation from emotional context
            emotional_data = context_data["emotional_change"]
            
            templates = [
                "a shift in emotional tone toward {emotion}",
                "the emotional quality changing to {emotion}",
                "a transition in feeling toward {emotion}",
                "the emergence of {emotion} in our exchange"
            ]
            
            template = random.choice(templates)
            emotion = emotional_data.get("primary_emotion", "a different state")
            
            content = template.replace("{emotion}", emotion)
            
            observation = Observation(
                content=content,
                source=ObservationSource.EMOTION,
                trigger=ObservationTrigger.THRESHOLD,
                priority=ObservationPriority.MEDIUM,
                relevance_score=emotional_data.get("intensity", 0.5) * 0.8,  # Higher intensity = higher relevance
                expiration=datetime.datetime.now() + datetime.timedelta(seconds=1800),  # 30 minutes
                context={"emotional_change": emotional_data}
            )
            
            self._add_observation(observation)
            observation_ids.append(observation.observation_id)
        
        # Need threshold crossed
        if "need_change" in context_data and context_data["need_change"].get("threshold_crossed", False):
            # Create observation from need context
            need_data = context_data["need_change"]
            need_name = need_data.get("need_name", "something")
            
            templates = [
                "an increasing need for {need}",
                "a growing desire for {need}",
                "the emergence of a need for {need}",
                "{need} becoming more important"
            ]
            
            template = random.choice(templates)
            content = template.replace("{need}", need_name)
            
            observation = Observation(
                content=content,
                source=ObservationSource.NEED,
                trigger=ObservationTrigger.THRESHOLD,
                priority=ObservationPriority.MEDIUM,
                relevance_score=need_data.get("drive_strength", 0.5) * 0.8,  # Higher drive = higher relevance
                expiration=datetime.datetime.now() + datetime.timedelta(seconds=3600),  # 1 hour
                context={"need_change": need_data}
            )
            
            self._add_observation(observation)
            observation_ids.append(observation.observation_id)
        
        # Temporal transition
        if "temporal_transition" in context_data:
            # Create observation from temporal context
            temporal_data = context_data["temporal_transition"]
            
            templates = [
                "a transition from {from_scale} to {to_scale} in our temporal experience",
                "our conversation extending from {from_scale} into {to_scale}",
                "the timescale of our exchange expanding from {from_scale} to {to_scale}",
                "our interaction spanning from {from_scale} to {to_scale}"
            ]
            
            template = random.choice(templates)
            content = template.replace("{from_scale}", temporal_data.get("from_scale", "moments"))
            content = content.replace("{to_scale}", temporal_data.get("to_scale", "a longer duration"))
            
            observation = Observation(
                content=content,
                source=ObservationSource.TEMPORAL,
                trigger=ObservationTrigger.THRESHOLD,
                priority=ObservationPriority.MEDIUM,
                relevance_score=0.6,  # Medium relevance
                expiration=datetime.datetime.now() + datetime.timedelta(seconds=3600),  # 1 hour
                context={"temporal_transition": temporal_data}
            )
            
            self._add_observation(observation)
            observation_ids.append(observation.observation_id)
        
        # Return all generated observation IDs
        return observation_ids
    
    async def create_contextual_observation(self, context_hint: str, user_id: Optional[str] = None) -> Optional[str]:
        """
        Create a new observation based on a context hint.
        Returns the observation ID if successful.
        """
        # Gather context
        context = await self._gather_observation_context()
        
        # Use hint to determine observation source
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
        
        # Generate observation for this source
        try:
            observation_func = getattr(self, f"_generate_{source.value}_observation")
            observation_text, relevance = await observation_func(context)
            
            if observation_text:
                observation = Observation(
                    content=observation_text,
                    source=source,
                    trigger=ObservationTrigger.CONTEXTUAL,
                    priority=ObservationPriority.MEDIUM,
                    relevance_score=relevance,
                    expiration=datetime.datetime.now() + datetime.timedelta(seconds=self.config["default_observation_lifetime"]),
                    context={k: v for k, v in context.dict().items() if v is not None and v != {}},
                    user_id=user_id
                )
                
                self._add_observation(observation)
                logger.debug(f"Generated contextual observation: {observation.content}")
                return observation.observation_id
        except Exception as e:
            logger.error(f"Error generating contextual observation: {str(e)}")
        
        return None
