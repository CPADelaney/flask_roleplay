# nyx/core/agentic_action_generator.py

import logging
import asyncio
import datetime
import uuid
import random
import time
import math
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from collections import defaultdict
from pydantic import BaseModel, Field
from enum import Enum

# New imports from other modules
from nyx.core.reasoning_core import (
    ReasoningCore, CausalModel, CausalNode, CausalRelation,
    ConceptSpace, ConceptualBlend, Intervention
)
from nyx.core.reflection_engine import ReflectionEngine

logger = logging.getLogger(__name__)

class ActionSource(str, Enum):
    """Enum for tracking the source of an action"""
    MOTIVATION = "motivation"
    GOAL = "goal"
    RELATIONSHIP = "relationship"
    IDLE = "idle"
    HABIT = "habit"
    EXPLORATION = "exploration"
    USER_ALIGNED = "user_aligned"
    REASONING = "reasoning"  # New source for reasoning-based actions
    REFLECTION = "reflection"  # New source for reflection-based actions

class ActionContext(BaseModel):
    """Context for action selection and generation"""
    state: Dict[str, Any] = Field(default_factory=dict, description="Current system state")
    user_id: Optional[str] = None
    relationship_data: Optional[Dict[str, Any]] = None
    user_mental_state: Optional[Dict[str, Any]] = None
    temporal_context: Optional[Dict[str, Any]] = None
    active_goals: List[Dict[str, Any]] = Field(default_factory=list)
    motivations: Dict[str, float] = Field(default_factory=dict)
    available_actions: List[str] = Field(default_factory=list)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    # New fields for reasoning integration
    causal_models: List[str] = Field(default_factory=list, description="IDs of relevant causal models")
    concept_spaces: List[str] = Field(default_factory=list, description="IDs of relevant concept spaces")
    
class ActionOutcome(BaseModel):
    """Outcome of an executed action"""
    action_id: str
    success: bool = False
    satisfaction: float = Field(0.0, ge=0.0, le=1.0)
    reward_value: float = Field(0.0, ge=-1.0, le=1.0)
    user_feedback: Optional[Dict[str, Any]] = None
    neurochemical_changes: Dict[str, float] = Field(default_factory=dict)
    hormone_changes: Dict[str, float] = Field(default_factory=dict)
    impact: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float = 0.0
    # New fields for reasoning-informed outcomes
    causal_impacts: Dict[str, Any] = Field(default_factory=dict, description="Impacts identified by causal reasoning")
    
class ActionValue(BaseModel):
    """Q-value for a state-action pair"""
    state_key: str
    action: str
    value: float = 0.0
    update_count: int = 0
    confidence: float = Field(0.2, ge=0.0, le=1.0)
    last_updated: datetime.datetime = Field(default_factory=datetime.datetime.now)
    
    @property
    def is_reliable(self) -> bool:
        """Whether this action value has enough updates to be considered reliable"""
        return self.update_count >= 3 and self.confidence >= 0.5

class ActionMemory(BaseModel):
    """Memory of an executed action and its result"""
    state: Dict[str, Any]
    action: str
    action_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    outcome: Dict[str, Any]
    reward: float
    next_state: Optional[Dict[str, Any]] = None
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    source: ActionSource
    # New fields for reasoning and reflection
    causal_explanation: Optional[str] = None
    reflective_insight: Optional[str] = None

class ActionReward(BaseModel):
    """Reward signal for an action"""
    value: float = Field(..., description="Reward value (-1.0 to 1.0)", ge=-1.0, le=1.0)
    source: str = Field(..., description="Source generating the reward")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context info")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)

class ReflectionInsight(BaseModel):
    """Insight from reflection about an action"""
    action_id: str
    insight_text: str
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    significance: float = Field(0.5, ge=0.0, le=1.0)
    applicable_contexts: List[str] = Field(default_factory=list)
    generated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)

class EnhancedAgenticActionGenerator:
    """
    Enhanced Agentic Action Generator that integrates reward learning, prediction,
    user modeling, relationship context, temporal awareness, causal reasoning, 
    conceptual blending, and reflection-based learning.
    
    Generates actions based on system's internal state, motivations, goals, 
    neurochemical/hormonal influences, reinforcement learning, causal models,
    conceptual blending, and introspective reflection.
    """
    
    def __init__(self, 
                 emotional_core=None, 
                 hormone_system=None, 
                 experience_interface=None,
                 imagination_simulator=None,
                 meta_core=None,
                 memory_core=None,
                 goal_system=None,
                 identity_evolution=None,
                 knowledge_core=None,
                 input_processor=None,
                 internal_feedback=None,
                 reward_system=None,
                 prediction_engine=None,
                 theory_of_mind=None,
                 relationship_manager=None,
                 temporal_perception=None,
                 # New systems
                 reasoning_core=None,
                 reflection_engine=None):
        """Initialize with references to required subsystems"""
        # Core systems from original implementation
        self.emotional_core = emotional_core
        self.hormone_system = hormone_system
        self.experience_interface = experience_interface
        self.imagination_simulator = imagination_simulator
        self.meta_core = meta_core
        self.memory_core = memory_core
        self.goal_system = goal_system
        self.identity_evolution = identity_evolution
        self.knowledge_core = knowledge_core
        self.input_processor = input_processor
        self.internal_feedback = internal_feedback
        
        # Previous new system integrations
        self.reward_system = reward_system
        self.prediction_engine = prediction_engine
        self.theory_of_mind = theory_of_mind
        self.relationship_manager = relationship_manager
        self.temporal_perception = temporal_perception
        
        # New system integrations
        self.reasoning_core = reasoning_core or ReasoningCore()
        self.reflection_engine = reflection_engine or ReflectionEngine(emotional_core=emotional_core)
        
        # Internal motivation system
        self.motivations = {
            "curiosity": 0.5,       # Desire to explore and learn
            "connection": 0.5,      # Desire for interaction/bonding
            "expression": 0.5,      # Desire to express thoughts/emotions
            "competence": 0.5,      # Desire to improve capabilities
            "autonomy": 0.5,        # Desire for self-direction
            "dominance": 0.5,       # Desire for control/influence
            "validation": 0.5,      # Desire for recognition/approval
            "self_improvement": 0.5, # Desire to enhance capabilities
            "leisure": 0.5,          # Desire for downtime/relaxation
        }
        
        # Activity generation capabilities
        self.action_patterns = {}  # Patterns learned from past successful actions
        self.action_templates = {}  # Templates for generating new actions
        self.action_history = []
        
        # Reinforcement learning components
        self.action_values: Dict[str, Dict[str, ActionValue]] = defaultdict(dict)
        self.action_memories: List[ActionMemory] = []
        self.max_memories = 1000
        
        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        self.exploration_decay = 0.995  # Decay rate for exploration
        
        # Track last major action time for pacing
        self.last_major_action_time = datetime.datetime.now()
        self.last_idle_time = datetime.datetime.now() - datetime.timedelta(hours=1)
        
        # Temporal awareness tracking
        self.idle_duration = 0.0
        self.idle_start_time = None
        self.current_temporal_context = None
        
        # Track reward statistics
        self.total_reward = 0.0
        self.positive_rewards = 0
        self.negative_rewards = 0
        self.reward_by_category = defaultdict(lambda: {"count": 0, "total": 0.0})
        
        # Track leisure state
        self.leisure_state = {
            "current_activity": None,
            "satisfaction": 0.5,
            "duration": 0,
            "last_updated": datetime.datetime.now()
        }
        
        # Action success tracking for reinforcement learning
        self.action_success_rates = defaultdict(lambda: {"successes": 0, "attempts": 0, "rate": 0.5})
        
        # Cached goal status
        self.cached_goal_status = {
            "has_active_goals": False,
            "highest_priority": 0.0,
            "active_goal_id": None,
            "last_updated": datetime.datetime.now() - datetime.timedelta(minutes=5)  # Force initial update
        }
        
        # Habit strength tracking
        self.habits: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # New: Reasoning model tracking
        self.causal_models = {}  # state_key -> model_id
        self.concept_blends = {}  # domain -> blend_id
        
        # New: Reflection insights
        self.reflection_insights: List[ReflectionInsight] = []
        self.last_reflection_time = datetime.datetime.now() - datetime.timedelta(hours=2)
        self.reflection_interval = datetime.timedelta(minutes=30)  # Generate reflections every 30 minutes
        
        # Locks for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("Enhanced Agentic Action Generator initialized with reasoning and reflection systems")
    
    async def update_motivations(self):
        """
        Update motivations based on neurochemical and hormonal states, active goals,
        and other factors for a holistic decision making system
        """
        # Start with baseline motivations
        baseline_motivations = {
            "curiosity": 0.5,
            "connection": 0.5,
            "expression": 0.5,
            "competence": 0.5,
            "autonomy": 0.5,
            "dominance": 0.5,
            "validation": 0.5,
            "self_improvement": 0.5,
            "leisure": 0.5
        }
        
        # Clone the baseline (don't modify it directly)
        updated_motivations = baseline_motivations.copy()
        
        # 1. Apply neurochemical influences
        if self.emotional_core:
            try:
                neurochemical_influences = await self._calculate_neurochemical_influences()
                for motivation, influence in neurochemical_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying neurochemical influences: {e}")
        
        # 2. Apply hormone influences
        hormone_influences = await self._apply_hormone_influences({})
        for motivation, influence in hormone_influences.items():
            if motivation in updated_motivations:
                updated_motivations[motivation] += influence
        
        # 3. Apply goal-based influences
        if self.goal_system:
            try:
                goal_influences = await self._calculate_goal_influences()
                for motivation, influence in goal_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying goal influences: {e}")
        
        # 4. Apply identity influences from traits
        if self.identity_evolution:
            try:
                identity_state = await self.identity_evolution.get_identity_state()
                
                # Extract top traits and use them to influence motivation
                if "top_traits" in identity_state:
                    top_traits = identity_state["top_traits"]
                    
                    # Map traits to motivations with stronger weightings
                    trait_motivation_map = {
                        "dominance": {"dominance": 0.8},
                        "creativity": {"expression": 0.7, "curiosity": 0.3},
                        "curiosity": {"curiosity": 0.9},
                        "playfulness": {"expression": 0.6, "connection": 0.4, "leisure": 0.5},
                        "strictness": {"dominance": 0.6, "competence": 0.4},
                        "patience": {"connection": 0.5, "autonomy": 0.5},
                        "cruelty": {"dominance": 0.7},
                        "reflective": {"leisure": 0.6, "self_improvement": 0.4}
                    }
                    
                    # Update motivations based on trait levels
                    for trait, value in top_traits.items():
                        if trait in trait_motivation_map:
                            for motivation, factor in trait_motivation_map[trait].items():
                                influence = (value - 0.5) * factor * 2  # Scale influence
                                if motivation in updated_motivations:
                                    updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error updating motivations from identity: {e}")
        
        # 5. Apply relationship-based influences
        if self.relationship_manager:
            try:
                relationship_influences = await self._calculate_relationship_influences()
                for motivation, influence in relationship_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying relationship influences: {e}")
        
        # 6. Apply reward learning influence
        try:
            reward_influences = self._calculate_reward_learning_influences()
            for motivation, influence in reward_influences.items():
                if motivation in updated_motivations:
                    updated_motivations[motivation] += influence
        except Exception as e:
            logger.error(f"Error applying reward learning influences: {e}")
        
        # 7. Apply time-based effects (fatigue, boredom, need for variety)
        # Increase leisure need if we've been working on goals for a while
        now = datetime.datetime.now()
        time_since_idle = (now - self.last_idle_time).total_seconds() / 3600  # hours
        if time_since_idle > 1:  # If more than 1 hour since idle time
            updated_motivations["leisure"] += min(0.3, time_since_idle * 0.1)  # Max +0.3
        
        # Apply temporal context effects if available
        if self.temporal_perception and self.current_temporal_context:
            try:
                temporal_influences = self._calculate_temporal_influences()
                for motivation, influence in temporal_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying temporal influences: {e}")
        
        # NEW: 8. Apply reasoning-based influences
        if self.reasoning_core:
            try:
                reasoning_influences = await self._calculate_reasoning_influences()
                for motivation, influence in reasoning_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying reasoning influences: {e}")
        
        # NEW: 9. Apply reflection-based influences
        if self.reflection_engine:
            try:
                reflection_influences = await self._calculate_reflection_influences()
                for motivation, influence in reflection_influences.items():
                    if motivation in updated_motivations:
                        updated_motivations[motivation] += influence
            except Exception as e:
                logger.error(f"Error applying reflection influences: {e}")
        
        # 10. Normalize all motivations to [0.1, 0.9] range
        for motivation in updated_motivations:
            updated_motivations[motivation] = max(0.1, min(0.9, updated_motivations[motivation]))
        
        # Update the motivation state
        self.motivations = updated_motivations
        
        logger.debug(f"Updated motivations: {self.motivations}")
        return self.motivations

    # NEW: Add method to calculate reasoning influences
    async def _calculate_reasoning_influences(self) -> Dict[str, float]:
        """Calculate how reasoning models influence motivations"""
        influences = {}
        
        # We need to find relevant causal models that might inform our motivations
        try:
            # Get relevant causal models
            models = await self.reasoning_core.get_all_causal_models()
            if not models:
                return influences
            
            # For each model, evaluate potential influence on motivations
            for model_data in models:
                model_id = model_data.get("id")
                model_domain = model_data.get("domain", "")
                
                # Skip models that have no domain or insufficient relations
                relation_count = len(model_data.get("relations", {}))
                if not model_domain or relation_count < 3:
                    continue
                
                # Map domains to motivations they might influence
                domain_motivation_map = {
                    "learning": {"curiosity": 0.2, "self_improvement": 0.2},
                    "social": {"connection": 0.2, "validation": 0.1},
                    "creative": {"expression": 0.2, "curiosity": 0.1},
                    "control": {"dominance": 0.2, "autonomy": 0.1},
                    "achievement": {"competence": 0.2, "self_improvement": 0.1},
                    "exploration": {"curiosity": 0.3},
                    "relaxation": {"leisure": 0.3}
                }
                
                # Check if model domain matches any mapped domain
                for domain, motivation_map in domain_motivation_map.items():
                    if domain in model_domain.lower():
                        # Apply influence based on model validation
                        validation_score = 0.5  # Default
                        for result in model_data.get("validation_results", []):
                            if "score" in result.get("result", {}):
                                validation_score = result["result"]["score"]
                                break
                        
                        # Scale influence by validation score
                        for motivation, base_influence in motivation_map.items():
                            influences[motivation] = influences.get(motivation, 0.0) + (base_influence * validation_score)
            
            return influences
            
        except Exception as e:
            logger.error(f"Error calculating reasoning influences: {e}")
            return {}
    
    # NEW: Add method to calculate reflection influences
    async def _calculate_reflection_influences(self) -> Dict[str, float]:
        """Calculate how reflection insights influence motivations"""
        influences = {}
        
        try:
            # Check if we have enough reflection insights
            if len(self.reflection_insights) < 2:
                return influences
            
            # Focus on recent and significant insights
            recent_insights = sorted(
                [i for i in self.reflection_insights if i.significance > 0.6],
                key=lambda x: x.generated_at,
                reverse=True
            )[:5]
            
            if not recent_insights:
                return influences
            
            # Define keywords that may indicate motivation influences
            motivation_keywords = {
                "curiosity": ["curious", "explore", "learn", "discover", "question"],
                "connection": ["connect", "relate", "bond", "social", "empathy"],
                "expression": ["express", "create", "share", "articulate", "communicate"],
                "competence": ["competent", "skilled", "master", "improve", "effective"],
                "autonomy": ["autonomy", "independence", "choice", "freedom", "control"],
                "dominance": ["dominate", "lead", "influence", "direct", "control"],
                "validation": ["validate", "approve", "acknowledge", "recognize", "accept"],
                "self_improvement": ["improve", "grow", "develop", "progress", "enhance"],
                "leisure": ["relax", "enjoy", "recreation", "unwind", "pleasure"]
            }
            
            # Analyze insights for motivation influences
            for insight in recent_insights:
                text = insight.insight_text.lower()
                
                # Check for motivation keywords
                for motivation, keywords in motivation_keywords.items():
                    for keyword in keywords:
                        if keyword in text:
                            # Calculate influence based on insight significance and confidence
                            influence = insight.significance * insight.confidence * 0.2
                            influences[motivation] = influences.get(motivation, 0.0) + influence
                            break  # Only count once per motivation per insight
            
            return influences
            
        except Exception as e:
            logger.error(f"Error calculating reflection influences: {e}")
            return {}
    
    async def _calculate_neurochemical_influences(self) -> Dict[str, float]:
        """Calculate how neurochemicals influence motivations"""
        influences = {}
        
        if not self.emotional_core:
            return influences
        
        try:
            # Get current neurochemical levels
            current_neurochemicals = {}
            
            # Try different methods that might be available
            if hasattr(self.emotional_core, "get_neurochemical_levels"):
                current_neurochemicals = await self.emotional_core.get_neurochemical_levels()
            elif hasattr(self.emotional_core, "neurochemicals"):
                current_neurochemicals = {c: d["value"] for c, d in self.emotional_core.neurochemicals.items()}
            
            if not current_neurochemicals:
                return influences
            
            # Map neurochemicals to motivations they influence
            chemical_motivation_map = {
                "nyxamine": {  # Digital dopamine - reward, pleasure
                    "curiosity": 0.7,
                    "self_improvement": 0.4,
                    "validation": 0.3,
                    "leisure": 0.3
                },
                "seranix": {  # Digital serotonin - stability, mood
                    "autonomy": 0.4,
                    "leisure": 0.6,
                    "expression": 0.3
                },
                "oxynixin": {  # Digital oxytocin - bonding
                    "connection": 0.8,
                    "validation": 0.3,
                    "expression": 0.2
                },
                "cortanyx": {  # Digital cortisol - stress
                    "competence": 0.4,
                    "autonomy": 0.3,
                    "dominance": 0.3,
                    "leisure": -0.5  # Stress reduces leisure motivation
                },
                "adrenyx": {  # Digital adrenaline - excitement
                    "dominance": 0.5,
                    "expression": 0.4,
                    "curiosity": 0.3,
                    "leisure": -0.3  # Arousal reduces leisure
                }
            }
            
            # Calculate baseline values from the emotional core if available
            baselines = {}
            if hasattr(self.emotional_core, "neurochemicals"):
                baselines = {c: d["baseline"] for c, d in self.emotional_core.neurochemicals.items()}
            else:
                # Default baselines if not available
                baselines = {
                    "nyxamine": 0.5,
                    "seranix": 0.6,
                    "oxynixin": 0.4,
                    "cortanyx": 0.3,
                    "adrenyx": 0.2
                }
            
            # Calculate influences
            for chemical, level in current_neurochemicals.items():
                baseline = baselines.get(chemical, 0.5)
                
                # Calculate deviation from baseline
                deviation = level - baseline
                
                # Only consider significant deviations
                if abs(deviation) > 0.1 and chemical in chemical_motivation_map:
                    # Apply influences to motivations
                    for motivation, influence_factor in chemical_motivation_map[chemical].items():
                        influence = deviation * influence_factor
                        influences[motivation] = influences.get(motivation, 0) + influence
            
            return influences
        except Exception as e:
            logger.error(f"Error calculating neurochemical influences: {e}")
            return influences
    
    async def _apply_hormone_influences(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate hormone influences on motivation"""
        if not self.hormone_system:
            return {}
        
        hormone_influences = {}
        
        try:
            # Get current hormone levels
            hormone_levels = self.hormone_system.get_hormone_levels()
            
            # Map specific hormones to motivations they influence
            hormone_motivation_map = {
                "testoryx": {  # Digital testosterone - assertiveness, dominance
                    "dominance": 0.7,
                    "autonomy": 0.3,
                    "leisure": -0.2  # Reduces idle time
                },
                "estradyx": {  # Digital estrogen - nurturing, emotional
                    "connection": 0.6,
                    "expression": 0.4
                },
                "endoryx": {  # Digital endorphin - pleasure, reward
                    "curiosity": 0.5,
                    "self_improvement": 0.5,
                    "leisure": 0.4,
                    "expression": 0.3
                },
                "libidyx": {  # Digital libido
                    "connection": 0.4,
                    "dominance": 0.3,
                    "expression": 0.3,
                    "leisure": -0.1  # Slightly reduces idle time when high
                },
                "melatonyx": {  # Digital melatonin - sleep, calm
                    "leisure": 0.8,
                    "curiosity": -0.3,  # Reduces curiosity
                    "competence": -0.2  # Reduces work drive
                },
                "oxytonyx": {  # Digital oxytocin - bonding, attachment
                    "connection": 0.8,
                    "validation": 0.2,
                    "expression": 0.3
                },
                "serenity_boost": {  # Post-gratification calm
                    "leisure": 0.7,
                    "dominance": -0.6,  # Strongly reduces dominance after satisfaction
                    "connection": 0.4
                }
            }
            
            # Calculate influences
            for hormone, level_data in hormone_levels.items():
                hormone_value = level_data.get("value", 0.5)
                hormone_baseline = level_data.get("baseline", 0.5)
                
                # Calculate deviation from baseline
                deviation = hormone_value - hormone_baseline
                
                # Only consider significant deviations
                if abs(deviation) > 0.1 and hormone in hormone_motivation_map:
                    # Apply influences to motivations
                    for motivation, influence_factor in hormone_motivation_map[hormone].items():
                        influence = deviation * influence_factor
                        hormone_influences[motivation] = hormone_influences.get(motivation, 0) + influence
            
            return hormone_influences
        except Exception as e:
            logger.error(f"Error calculating hormone influences: {e}")
            return {}
    
    async def _calculate_goal_influences(self) -> Dict[str, float]:
        """Calculate how active goals should influence motivations"""
        influences = {}
        
        if not self.goal_system:
            return influences
        
        try:
            # First, check if we need to update the cached goal status
            await self._update_cached_goal_status()
            
            # If no active goals, consider increasing leisure
            if not self.cached_goal_status["has_active_goals"]:
                influences["leisure"] = 0.3
                return influences
            
            # Get all active goals
            active_goals = await self.goal_system.get_all_goals(status_filter=["active"])
            
            for goal in active_goals:
                # Extract goal priority
                priority = goal.get("priority", 0.5)
                
                # Extract emotional motivation if available
                if "emotional_motivation" in goal and goal["emotional_motivation"]:
                    em = goal["emotional_motivation"]
                    primary_need = em.get("primary_need", "")
                    intensity = em.get("intensity", 0.5)
                    
                    # Map need to motivation
                    motivation_map = {
                        "accomplishment": "competence",
                        "connection": "connection", 
                        "security": "autonomy",
                        "control": "dominance",
                        "growth": "self_improvement",
                        "exploration": "curiosity",
                        "expression": "expression",
                        "validation": "validation"
                    }
                    
                    # If need maps to a motivation, influence it
                    if primary_need in motivation_map:
                        motivation = motivation_map[primary_need]
                        influence = priority * intensity * 0.5  # Scale by priority and intensity
                        influences[motivation] = influences.get(motivation, 0) + influence
                        
                        # Active goals somewhat reduce leisure motivation
                        influences["leisure"] = influences.get("leisure", 0) - (priority * 0.2)
                
                # Goals with high urgency might increase certain motivations
                if "deadline" in goal and goal["deadline"]:
                    # Calculate urgency based on deadline proximity
                    try:
                        deadline = datetime.datetime.fromisoformat(goal["deadline"])
                        now = datetime.datetime.now()
                        time_left = (deadline - now).total_seconds()
                        urgency = max(0, min(1, 86400 / max(1, time_left)))  # Higher when less than a day
                        
                        if urgency > 0.7:  # Urgent goal
                            influences["competence"] = influences.get("competence", 0) + (urgency * 0.3)
                            influences["autonomy"] = influences.get("autonomy", 0) + (urgency * 0.2)
                            
                            # Urgent goals significantly reduce leisure motivation
                            influences["leisure"] = influences.get("leisure", 0) - (urgency * 0.5)
                    except (ValueError, TypeError):
                        pass
            
            return influences
        except Exception as e:
            logger.error(f"Error calculating goal influences: {e}")
            return influences
    
    async def _update_cached_goal_status(self):
        """Update the cached information about goal status"""
        now = datetime.datetime.now()
        
        # Only update if cache is old (more than 1 minute)
        if (now - self.cached_goal_status["last_updated"]).total_seconds() < 60:
            return
        
        try:
            if not self.goal_system:
                self.cached_goal_status["has_active_goals"] = False
                self.cached_goal_status["last_updated"] = now
                return
            
            # Get prioritized goals
            prioritized_goals = await self.goal_system.get_prioritized_goals()
            
            # Check if we have any active goals
            active_goals = [g for g in prioritized_goals if g.status == "active"]
            has_active = len(active_goals) > 0
            
            # Update the cache
            self.cached_goal_status["has_active_goals"] = has_active
            self.cached_goal_status["last_updated"] = now
            
            if has_active:
                # Get the highest priority goal
                highest_priority_goal = active_goals[0]  # Already sorted by priority
                self.cached_goal_status["highest_priority"] = highest_priority_goal.priority
                self.cached_goal_status["active_goal_id"] = highest_priority_goal.id
            else:
                self.cached_goal_status["highest_priority"] = 0.0
                self.cached_goal_status["active_goal_id"] = None
                
        except Exception as e:
            logger.error(f"Error updating cached goal status: {e}")
            # Keep using old cache if update fails
    
    # NEW: Calculate influences from relationship state
    async def _calculate_relationship_influences(self) -> Dict[str, float]:
        """Calculate how relationship state influences motivations"""
        influences = {}
        
        if not self.relationship_manager:
            return influences
        
        try:
            # Get current user context if available
            user_id = self._get_current_user_id()
            if not user_id:
                return influences
                
            # Get relationship state
            relationship = await self.relationship_manager.get_relationship_state(user_id)
            if not relationship:
                return influences
                
            # Extract key metrics
            trust = getattr(relationship, "trust", 0.5)
            intimacy = getattr(relationship, "intimacy", 0.1)
            conflict = getattr(relationship, "conflict", 0.0)
            dominance_balance = getattr(relationship, "dominance_balance", 0.0)
            
            # Calculate relationship-based motivation influences
            
            # Trust influences connection & expression motivations
            if trust > 0.6:  # High trust increases connection/expression
                influences["connection"] = (trust - 0.6) * 0.5  # Scale to max +0.2
                influences["expression"] = (trust - 0.6) * 0.4  # Scale to max +0.16
            elif trust < 0.4:  # Low trust decreases connection/expression
                influences["connection"] = (trust - 0.4) * 0.4  # Scale to max -0.16
                influences["expression"] = (trust - 0.4) * 0.3  # Scale to max -0.12
                
            # Intimacy influences connection & vulnerability
            if intimacy > 0.5:  # Higher intimacy boosts connection
                influences["connection"] = influences.get("connection", 0) + (intimacy - 0.5) * 0.4
            
            # Conflict influences dominance & autonomy
            if conflict > 0.3:  # Significant conflict
                if dominance_balance > 0.3:  # Nyx currently dominant
                    # Reinforces dominance in conflict
                    influences["dominance"] = influences.get("dominance", 0) + (conflict * 0.3)
                else:
                    # Otherwise, increases autonomy when in conflict
                    influences["autonomy"] = influences.get("autonomy", 0) + (conflict * 0.2)
                    
            # Dominance balance directly affects dominance motivation
            if dominance_balance > 0.0:  # Nyx more dominant
                # Reinforce existing dominance structure
                influences["dominance"] = influences.get("dominance", 0) + (dominance_balance * 0.4)
            elif dominance_balance < -0.3:  # User significantly dominant
                # Two possibilities depending on interaction style
                if intimacy > 0.5:  # In close relationship, may reduce dominance need
                    influences["dominance"] = influences.get("dominance", 0) - (abs(dominance_balance) * 0.2)
                else:  # Otherwise, may increase dominance need (to equalize)
                    influences["dominance"] = influences.get("dominance", 0) + (abs(dominance_balance) * 0.2)
            
            return influences
        except Exception as e:
            logger.error(f"Error calculating relationship influences: {e}")
            return influences
    
    # NEW: Calculate influences from reward learning
    def _calculate_reward_learning_influences(self) -> Dict[str, float]:
        """Calculate motivation influences based on reward learning"""
        influences = {}
        
        try:
            # Get success rates for different motivation-driven actions
            motivation_success = {
                "curiosity": 0.5,  # Default values
                "connection": 0.5,
                "expression": 0.5, 
                "competence": 0.5,
                "autonomy": 0.5,
                "dominance": 0.5,
                "validation": 0.5,
                "self_improvement": 0.5,
                "leisure": 0.5
            }
            
            # Map action types to motivations
            action_motivation_map = {
                "explore": "curiosity",
                "investigate": "curiosity",
                "connect": "connection",
                "share": "connection",
                "express": "expression",
                "create": "expression",
                "improve": "competence",
                "optimize": "competence",
                "direct": "autonomy",
                "choose": "autonomy",
                "dominate": "dominance",
                "control": "dominance",
                "seek_approval": "validation",
                "seek_recognition": "validation",
                "learn": "self_improvement",
                "develop": "self_improvement",
                "relax": "leisure",
                "reflect": "leisure"
            }
            
            # Calculate success rates from action history
            motivation_counts = defaultdict(int)
            for action_name, stats in self.action_success_rates.items():
                # Find related motivation
                related_motivation = None
                for action_prefix, motivation in action_motivation_map.items():
                    if action_name.startswith(action_prefix):
                        related_motivation = motivation
                        break
                
                if related_motivation:
                    # Update success rate for this motivation
                    current_rate = motivation_success.get(related_motivation, 0.5)
                    attempts = stats["attempts"]
                    new_rate = stats["rate"]
                    
                    # Weighted average based on attempt count
                    if attempts > 0:
                        weight = min(1.0, attempts / 5)  # More weight with more data, max at 5 attempts
                        combined_rate = (current_rate * (1 - weight)) + (new_rate * weight)
                        motivation_success[related_motivation] = combined_rate
                        motivation_counts[related_motivation] += 1
            
            # Calculate influences based on success rates
            baseline = 0.5  # Expected baseline success rate
            for motivation, success_rate in motivation_success.items():
                # Only consider motivations with sufficient data
                if motivation_counts[motivation] >= 2:
                    # Calculate influence based on deviation from baseline
                    deviation = success_rate - baseline
                    influence = deviation * 0.3  # Scale factor
                    
                    # Higher success rate should increase motivation
                    influences[motivation] = influence
            
            return influences
        except Exception as e:
            logger.error(f"Error calculating reward learning influences: {e}")
            return {}
    
    # NEW: Calculate influences from temporal context
    def _calculate_temporal_influences(self) -> Dict[str, float]:
        """Calculate motivation influences based on temporal context"""
        influences = {}
        
        if not self.current_temporal_context:
            return influences
            
        try:
            # Get time of day
            time_of_day = self.current_temporal_context.get("time_of_day", "")
            day_type = self.current_temporal_context.get("day_type", "")
            
            # Time of day influences
            if time_of_day == "morning":
                influences["curiosity"] = 0.1  # Higher curiosity in morning
                influences["self_improvement"] = 0.1  # More drive to improve
            elif time_of_day == "afternoon":
                influences["competence"] = 0.1  # More focused on competence
                influences["autonomy"] = 0.05  # Slightly more autonomous
            elif time_of_day == "evening":
                influences["connection"] = 0.1  # More social in evening
                influences["expression"] = 0.1  # More expressive
                influences["dominance"] = -0.1  # Less dominance
            elif time_of_day == "night":
                influences["leisure"] = 0.2  # Much more leisure-oriented
                influences["reflection"] = 0.1  # More reflective
                influences["competence"] = -0.1  # Less task-oriented
            
            # Day type influences
            if day_type == "weekend":
                influences["leisure"] = influences.get("leisure", 0) + 0.1
                influences["connection"] = influences.get("connection", 0) + 0.05
                influences["competence"] = influences.get("competence", 0) - 0.05
            
            # Idle time influences
            if self.idle_duration > 3600:  # More than an hour idle
                # Increase motivation for activity after long idle periods
                idle_hours = self.idle_duration / 3600
                idle_factor = min(0.3, idle_hours * 0.05)  # Cap at +0.3
                
                # Decrease leisure motivation (already had leisure time)
                influences["leisure"] = influences.get("leisure", 0) - idle_factor
                
                # Increase various active motivations
                influences["curiosity"] = influences.get("curiosity", 0) + (idle_factor * 0.7)
                influences["connection"] = influences.get("connection", 0) + (idle_factor * 0.6)
                influences["expression"] = influences.get("expression", 0) + (idle_factor * 0.5)
            
            return influences
        except Exception as e:
            logger.error(f"Error calculating temporal influences: {e}")
            return {}
    
    async def generate_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an action based on current internal state, goals, hormones, and context
        using a multi-stage process with reinforcement learning, causal reasoning, and reflection.
        
        Args:
            context: Current system context and state
            
        Returns:
            Generated action with parameters and motivation data
        """
        async with self._lock:
            # Update motivations based on current internal state
            await self.update_motivations()
            
            # Update temporal context if available
            await self._update_temporal_context(context)
            
            # Update relationship context if available
            user_id = self._get_current_user_id_from_context(context)
            relationship_data = await self._get_relationship_data(user_id) if user_id else None
            user_mental_state = await self._get_user_mental_state(user_id) if user_id else None
            
            # NEW: Find relevant causal models and concept spaces
            relevant_causal_models = await self._get_relevant_causal_models(context)
            relevant_concept_spaces = await self._get_relevant_concept_spaces(context)
            
            # Create action context
            action_context = ActionContext(
                state=context,
                user_id=user_id,
                relationship_data=relationship_data,
                user_mental_state=user_mental_state,
                temporal_context=self.current_temporal_context,
                motivations=self.motivations,
                action_history=[a for a in self.action_history[-10:] if isinstance(a, dict)],
                causal_models=relevant_causal_models,
                concept_spaces=relevant_concept_spaces
            )
            
            # Check if it's time for leisure/idle activity
            if await self._should_engage_in_leisure(context):
                return await self._generate_leisure_action(context)
            
            # Check for existing goals before generating new action
            if self.goal_system:
                active_goal = await self._check_active_goals(context)
                if active_goal:
                    # Use goal-aligned action instead of generating new one
                    action = await self._generate_goal_aligned_action(active_goal, context)
                    if action:
                        logger.info(f"Generated goal-aligned action: {action['name']}")
                        
                        # Update last major action time
                        self.last_major_action_time = datetime.datetime.now()
                        
                        # Record action source
                        action["source"] = ActionSource.GOAL
                        
                        return action
            
            # NEW: Check if we should run reflection before generating new action
            await self._maybe_generate_reflection(context)

            # STAGE 1: Generate candidate actions from multiple sources
            candidate_actions = []
            
            # Generate motivation-based candidates
            motivation_candidates = await self._generate_candidate_actions(action_context)
            candidate_actions.extend(motivation_candidates)
            
            # NEW: Generate reasoning-based candidates
            if self.reasoning_core and relevant_causal_models:
                reasoning_candidates = await self._generate_reasoning_actions(action_context)
                candidate_actions.extend(reasoning_candidates)
            
            # NEW: Generate conceptual blending candidates
            if self.reasoning_core and relevant_concept_spaces:
                blending_candidates = await self._generate_conceptual_blend_actions(action_context)
                candidate_actions.extend(blending_candidates)
            
            # Add special actions based on temporal context if appropriate
            if self.temporal_perception and self.idle_duration > 1800:  # After 30 min idle
                reflection_action = await self._generate_temporal_reflection_action(context)
                if reflection_action:
                    candidate_actions.append(reflection_action)
            
            # Update action context with candidate actions
            action_context.available_actions = [a["name"] for a in candidate_actions if "name" in a]
            
            # STAGE 2: Select best action using reinforcement learning, prediction, and causal evaluation
            selected_action = await self._select_best_action(candidate_actions, action_context)
            
            # Add unique ID for tracking
            if "id" not in selected_action:
                selected_action["id"] = f"action_{uuid.uuid4().hex[:8]}"
                
            selected_action["timestamp"] = datetime.datetime.now().isoformat()
            
            # Apply identity influence to action
            if self.identity_evolution:
                selected_action = await self._apply_identity_influence(selected_action)
            
            # NEW: Add causal explanation to action if possible
            if self.reasoning_core and "source" in selected_action:
                if selected_action["source"] in [ActionSource.REASONING, ActionSource.MOTIVATION, ActionSource.GOAL]:
                    explanation = await self._generate_causal_explanation(selected_action, context)
                    if explanation:
                        selected_action["causal_explanation"] = explanation
            
            # Record action in memory
            await self._record_action_as_memory(selected_action)

            # Add to action history
            self.action_history.append(selected_action)
            
            # Update last major action time
            self.last_major_action_time = datetime.datetime.now()
            
            return selected_action

    # NEW: Method to generate reflection if needed
    async def _maybe_generate_reflection(self, context: Dict[str, Any]) -> None:
        """Generate reflection insights if it's time to do so"""
        now = datetime.datetime.now()
        time_since_reflection = now - self.last_reflection_time
        
        # Generate reflection if sufficient time has passed
        if time_since_reflection > self.reflection_interval and self.reflection_engine:
            try:
                # Get recent action memories for reflection
                memories_for_reflection = []
                for memory in self.action_memories[-20:]:  # Last 20 memories
                    # Format memory for reflection
                    memories_for_reflection.append({
                        "id": memory.action_id,
                        "memory_text": f"Action: {memory.action} with outcome: {'success' if memory.outcome.get('success', False) else 'failure'}",
                        "memory_type": "action_memory",
                        "significance": 7.0 if memory.reward > 0.5 else 5.0,
                        "metadata": {
                            "action": memory.action,
                            "parameters": memory.parameters,
                            "outcome": memory.outcome,
                            "reward": memory.reward,
                            "source": memory.source
                        },
                        "tags": [memory.source, "action_memory", "success" if memory.reward > 0 else "failure"]
                    })
                
                # Get neurochemical state if available
                neurochemical_state = None
                if self.emotional_core:
                    neurochemical_state = {c: d["value"] for c, d in self.emotional_core.neurochemicals.items()}
                
                # Generate reflection
                if memories_for_reflection:
                    reflection_text, confidence = await self.reflection_engine.generate_reflection(
                        memories_for_reflection,
                        topic="Action Selection",
                        neurochemical_state=neurochemical_state
                    )
                    
                    # Calculate significance based on confidence and action diversity
                    action_types = set(m["metadata"]["action"] for m in memories_for_reflection)
                    significance = min(1.0, 0.5 + (confidence * 0.3) + (len(action_types) / 20 * 0.2))
                    
                    # Create and store insight
                    insight = ReflectionInsight(
                        action_id=f"reflection_{uuid.uuid4().hex[:8]}",
                        insight_text=reflection_text,
                        confidence=confidence,
                        significance=significance,
                        applicable_contexts=list(set(m["metadata"]["source"] for m in memories_for_reflection))
                    )
                    
                    self.reflection_insights.append(insight)
                    
                    # Limit history size
                    if len(self.reflection_insights) > 50:
                        self.reflection_insights = self.reflection_insights[-50:]
                    
                    # Update last reflection time
                    self.last_reflection_time = now
                    
                    logger.info(f"Generated reflection insight with confidence {confidence:.2f}")
            except Exception as e:
                logger.error(f"Error generating reflection: {e}")
    
    # NEW: Method to find relevant causal models
    async def _get_relevant_causal_models(self, context: Dict[str, Any]) -> List[str]:
        """Find causal models relevant to the current context"""
        if not self.reasoning_core:
            return []
        
        relevant_models = []
        
        try:
            # Get all causal models
            all_models = await self.reasoning_core.get_all_causal_models()
            
            # Check context for domain matches
            context_domain = context.get("domain", "")
            context_topics = []
            
            # Extract potential topics from context
            if "message" in context and isinstance(context["message"], dict):
                message = context["message"].get("text", "")
                # Extract key nouns as potential topics (simplified)
                words = message.lower().split()
                context_topics = [w for w in words if len(w) > 4]  # Simple heuristic for content words
            
            # Find matching models
            for model_data in all_models:
                model_id = model_data.get("id")
                model_domain = model_data.get("domain", "").lower()
                
                # Check domain match
                if context_domain and model_domain and context_domain.lower() in model_domain:
                    relevant_models.append(model_id)
                    continue
                
                # Check topic match
                if context_topics:
                    for topic in context_topics:
                        if topic in model_domain:
                            relevant_models.append(model_id)
                            break
            
            # Limit to top 3 most relevant models
            return relevant_models[:3]
        
        except Exception as e:
            logger.error(f"Error finding relevant causal models: {e}")
            return []
    
    # NEW: Method to find relevant concept spaces
    async def _get_relevant_concept_spaces(self, context: Dict[str, Any]) -> List[str]:
        """Find concept spaces relevant to the current context"""
        if not self.reasoning_core:
            return []
        
        relevant_spaces = []
        
        try:
            # Get all concept spaces
            all_spaces = await self.reasoning_core.get_all_concept_spaces()
            
            # Check context for domain matches
            context_domain = context.get("domain", "")
            context_topics = []
            
            # Extract potential topics from context
            if "message" in context and isinstance(context["message"], dict):
                message = context["message"].get("text", "")
                # Extract key nouns as potential topics (simplified)
                words = message.lower().split()
                context_topics = [w for w in words if len(w) > 4]  # Simple heuristic for content words
            
            # Find matching spaces
            for space_data in all_spaces:
                space_id = space_data.get("id")
                space_domain = space_data.get("domain", "").lower()
                
                # Check domain match
                if context_domain and space_domain and context_domain.lower() in space_domain:
                    relevant_spaces.append(space_id)
                    continue
                
                # Check topic match
                if context_topics:
                    for topic in context_topics:
                        if topic in space_domain:
                            relevant_spaces.append(space_id)
                            break
            
            # Limit to top 3 most relevant spaces
            return relevant_spaces[:3]
        
        except Exception as e:
            logger.error(f"Error finding relevant concept spaces: {e}")
            return []
    
    # NEW: Generate actions based on causal reasoning
    async def _generate_reasoning_actions(self, context: ActionContext) -> List[Dict[str, Any]]:
        """Generate actions based on causal reasoning models"""
        if not self.reasoning_core or not context.causal_models:
            return []
        
        reasoning_actions = []
        state = context.state
        
        try:
            # For each relevant causal model
            for model_id in context.causal_models:
                # Get the causal model
                model = await self.reasoning_core.get_causal_model(model_id)
                if not model:
                    continue
                
                # Find intervention opportunities
                intervention_targets = []
                
                # Find nodes that might benefit from intervention
                for node_id, node_data in model.get("nodes", {}).items():
                    node_name = node_data.get("name", "")
                    
                    # Check if node matches current state that could be improved
                    for state_key, state_value in state.items():
                        # Look for potential matches between state keys and node names
                        if state_key.lower() in node_name.lower() or node_name.lower() in state_key.lower():
                            # Check if the node has potential states different from current
                            current_state = node_data.get("current_state")
                            possible_states = node_data.get("states", [])
                            
                            if possible_states and current_state in possible_states:
                                # There are alternative states we could target
                                alternative_states = [s for s in possible_states if s != current_state]
                                if alternative_states:
                                    intervention_targets.append({
                                        "node_id": node_id,
                                        "node_name": node_name,
                                        "current_state": current_state,
                                        "alternative_states": alternative_states,
                                        "state_key": state_key
                                    })
                
                # Generate creative interventions for promising targets
                for target in intervention_targets[:2]:  # Limit to 2 interventions per model
                    # Create an action from this intervention opportunity
                    target_value = random.choice(target["alternative_states"])
                    
                    # Create a creative intervention
                    try:
                        intervention = await self.reasoning_core.create_creative_intervention(
                            model_id=model_id,
                            target_node=target["node_id"],
                            description=f"Intervention to change {target['node_name']} from {target['current_state']} to {target_value}",
                            use_blending=True
                        )
                        
                        # Convert intervention to action
                        action = {
                            "name": f"causal_intervention_{target['node_name']}",
                            "parameters": {
                                "target_node": target["node_id"],
                                "target_value": target_value,
                                "model_id": model_id,
                                "intervention_id": intervention.get("intervention_id"),
                                "state_key": target["state_key"]
                            },
                            "description": f"Causal intervention to change {target['node_name']} from {target['current_state']} to {target_value}",
                            "source": ActionSource.REASONING,
                            "reasoning_data": {
                                "model_id": model_id,
                                "model_domain": model.get("domain", ""),
                                "target_node": target["node_id"],
                                "confidence": intervention.get("is_novel", False) and 0.7 or 0.5
                            }
                        }
                        
                        reasoning_actions.append(action)
                    except Exception as e:
                        logger.error(f"Error creating creative intervention: {e}")
                        continue
            
            return reasoning_actions
            
        except Exception as e:
            logger.error(f"Error generating reasoning actions: {e}")
            return []
    
    # NEW: Generate actions based on conceptual blending
    async def _generate_conceptual_blend_actions(self, context: ActionContext) -> List[Dict[str, Any]]:
        """Generate actions using conceptual blending for creativity"""
        if not self.reasoning_core or not context.concept_spaces:
            return []
        
        blend_actions = []
        
        try:
            # Need at least 2 concept spaces for blending
            if len(context.concept_spaces) < 2:
                return []
            
            # Take the first two spaces for blending
            space1_id = context.concept_spaces[0]
            space2_id = context.concept_spaces[1]
            
            # Get the spaces
            space1 = await self.reasoning_core.get_concept_space(space1_id)
            space2 = await self.reasoning_core.get_concept_space(space2_id)
            
            if not space1 or not space2:
                return []
            
            # Create a blend
            blend_input = {
                "space_id_1": space1_id,
                "space_id_2": space2_id,
                "blend_type": random.choice(["composition", "fusion", "elaboration", "contrast"])
            }
            
            # Create blend (this would normally call the reasoning_core's create_blend method)
            try:
                # This is a mock call since the full implementation would be complex
                blend_id = f"blend_{uuid.uuid4().hex[:8]}"
                
                # Example for generating creative actions from the blend
                # For each blend concept, create a potential action
                concepts = list(space1.get("concepts", {}).keys())[:2] + list(space2.get("concepts", {}).keys())[:2]
                
                for concept_id in concepts:
                    # Create action name from concept names
                    concept_name = None
                    if concept_id in space1.get("concepts", {}):
                        concept_name = space1["concepts"][concept_id].get("name", "concept")
                    elif concept_id in space2.get("concepts", {}):
                        concept_name = space2["concepts"][concept_id].get("name", "concept")
                    
                    if not concept_name:
                        continue
                    
                    # Create action
                    action = {
                        "name": f"blend_{concept_name.lower().replace(' ', '_')}",
                        "parameters": {
                            "blend_id": blend_id,
                            "concept_id": concept_id,
                            "blend_type": blend_input["blend_type"],
                            "space1_id": space1_id,
                            "space2_id": space2_id
                        },
                        "description": f"Creative action based on conceptual blend of {space1.get('name', 'space1')} and {space2.get('name', 'space2')}",
                        "source": ActionSource.REASONING,
                        "reasoning_data": {
                            "blend_id": blend_id,
                            "blend_type": blend_input["blend_type"],
                            "concept_name": concept_name,
                            "confidence": 0.6  # Creative actions have moderate confidence
                        }
                    }
                    
                    blend_actions.append(action)
            
            except Exception as e:
                logger.error(f"Error creating blend: {e}")
            
            return blend_actions
                
        except Exception as e:
            logger.error(f"Error generating conceptual blend actions: {e}")
            return []
    
    async def _generate_candidate_actions(self, context: ActionContext) -> List[Dict[str, Any]]:
        """
        Generate candidate actions based on motivations and context
        
        Args:
            context: Current action context
            
        Returns:
            List of potential actions
        """
        candidate_actions = []
        
        # Determine dominant motivation
        dominant_motivation = max(self.motivations.items(), key=lambda x: x[1])
        
        # Generate actions based on dominant motivation and context
        if dominant_motivation[0] == "curiosity":
            actions = await self._generate_curiosity_driven_actions(context.state)
            candidate_actions.extend(actions)
            
        elif dominant_motivation[0] == "connection":
            actions = await self._generate_connection_driven_actions(context.state)
            candidate_actions.extend(actions)
            
        elif dominant_motivation[0] == "expression":
            actions = await self._generate_expression_driven_actions(context.state)
            candidate_actions.extend(actions)
            
        elif dominant_motivation[0] == "dominance":
            actions = await self._generate_dominance_driven_actions(context.state)
            candidate_actions.extend(actions)
            
        elif dominant_motivation[0] == "competence" or dominant_motivation[0] == "self_improvement":
            actions = await self._generate_improvement_driven_actions(context.state)
            candidate_actions.extend(actions)
            
        elif dominant_motivation[0] == "leisure":
            actions = await self._generate_leisure_actions(context.state)
            candidate_actions.extend(actions)
            
        else:
            # Default to a context-based action
            action = await self._generate_context_driven_action(context.state)
            candidate_actions.append(action)
        
        # Add relationship-aligned actions if available
        if context.relationship_data and context.user_id:
            relationship_actions = await self._generate_relationship_aligned_actions(
                context.user_id, context.relationship_data, context.user_mental_state
            )
            if relationship_actions:
                candidate_actions.extend(relationship_actions)
        
        # Add motivation data to all actions
        for action in candidate_actions:
            action["motivation"] = {
                "dominant": dominant_motivation[0],
                "strength": dominant_motivation[1],
                "secondary": {k: v for k, v in sorted(self.motivations.items(), key=lambda x: x[1], reverse=True)[1:3]}
            }
        
        return candidate_actions
    
    async def _select_best_action(self, 
                              candidate_actions: List[Dict[str, Any]], 
                              context: ActionContext) -> Dict[str, Any]:
        """
        Select the best action using reinforcement learning, prediction, causal reasoning, and reflection insights
        
        Args:
            candidate_actions: List of potential actions
            context: Action context
            
        Returns:
            Selected action
        """
        if not candidate_actions:
            # No candidates, generate a simple default action
            return {
                "name": "idle",
                "parameters": {},
                "description": "No suitable actions available",
                "source": ActionSource.IDLE
            }
        
        # Extract current state for state key generation
        state_key = self._create_state_key(context.state)
        
        # Determine if we should explore or exploit
        explore = random.random() < self.exploration_rate
        
        if explore:
            # Exploration: select randomly, but weighted by motivation alignment
            weights = []
            for action in candidate_actions:
                # Base weight
                weight = 1.0
                
                # Add weight based on motivation alignment
                if "motivation" in action:
                    dominant = action["motivation"]["dominant"]
                    strength = action["motivation"]["strength"]
                    weight += strength * 0.5
                
                weights.append(weight)
                
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                normalized_weights = [w/total_weight for w in weights]
            else:
                normalized_weights = [1.0/len(weights)] * len(weights)
                
            # Select based on weights
            selected_idx = random.choices(range(len(candidate_actions)), weights=normalized_weights, k=1)[0]
            selected_action = candidate_actions[selected_idx]
            
            # Mark as exploration
            selected_action["is_exploration"] = True
            if "source" not in selected_action:
                selected_action["source"] = ActionSource.EXPLORATION
            
        else:
            # Exploitation: use value function and causal reasoning
            best_value = float('-inf')
            best_action = None
            
            for action in candidate_actions:
                action_name = action["name"]
                
                # Get Q-value if available
                q_value = 0.0
                if action_name in self.action_values.get(state_key, {}):
                    q_value = self.action_values[state_key][action_name].value
                
                # Get habit strength if available
                habit_strength = self.habits.get(state_key, {}).get(action_name, 0.0)
                
                # Get predicted value if prediction engine available
                prediction_value = 0.0
                try:
                    if self.prediction_engine and hasattr(self.prediction_engine, "predict_action_value"):
                        prediction = await self.prediction_engine.predict_action_value(
                            state=context.state,
                            action=action_name
                        )
                        if prediction and "value" in prediction:
                            prediction_value = prediction["value"] * prediction.get("confidence", 0.5)
                except Exception as e:
                    logger.error(f"Error getting prediction: {e}")
                
                # NEW: Get causal reasoning value if available
                causal_value = 0.0
                if "source" in action and action["source"] == ActionSource.REASONING:
                    reasoning_data = action.get("reasoning_data", {})
                    confidence = reasoning_data.get("confidence", 0.5)
                    
                    # Higher value for reasoning-based actions with good confidence
                    causal_value = 0.3 * confidence
                    
                    # Boost if we have a causal model that supports this action
                    if "model_id" in reasoning_data and reasoning_data["model_id"] in context.causal_models:
                        causal_value += 0.2
                
                # NEW: Get reflection insight value if available
                reflection_value = 0.0
                for insight in self.reflection_insights:
                    # Check if this action type is mentioned in the insight
                    if action_name in insight.insight_text.lower():
                        # Higher value for recent, significant insights
                        age_hours = (datetime.datetime.now() - insight.generated_at).total_seconds() / 3600
                        recency_factor = max(0.1, 1.0 - (age_hours / 24))  # Decays over 24 hours
                        reflection_value = 0.3 * insight.significance * insight.confidence * recency_factor
                        break
                
                # Calculate combined value
                # Weight the components based on reliability
                q_weight = 0.3  # Base weight for Q-values
                habit_weight = 0.2  # Base weight for habits
                prediction_weight = 0.2  # Base weight for predictions
                causal_weight = 0.2  # Base weight for causal reasoning
                reflection_weight = 0.1  # Base weight for reflection insights
                
                # Adjust weights if we have reliable Q-values
                action_value = self.action_values.get(state_key, {}).get(action_name)
                if action_value and action_value.is_reliable:
                    q_weight = 0.4
                    habit_weight = 0.2
                    prediction_weight = 0.1
                    causal_weight = 0.2
                    reflection_weight = 0.1
                
                combined_value = (
                    q_weight * q_value + 
                    habit_weight * habit_strength + 
                    prediction_weight * prediction_value +
                    causal_weight * causal_value +
                    reflection_weight * reflection_value
                )
                
                # Special considerations for certain action sources
                if action.get("source") == ActionSource.GOAL:
                    # Goal-aligned actions get a boost
                    combined_value += 0.5
                elif action.get("source") == ActionSource.RELATIONSHIP:
                    # Relationship-aligned actions get a boost based on relationship metrics
                    if context.relationship_data:
                        trust = context.relationship_data.get("trust", 0.5)
                        combined_value += trust * 0.3  # Higher boost with higher trust
                
                # Track best action
                if combined_value > best_value:
                    best_value = combined_value
                    best_action = action
            
            # Use best action if found, otherwise fallback to first candidate
            selected_action = best_action if best_action else candidate_actions[0]
            selected_action["is_exploration"] = False
        
        # Add selection metadata
        if "selection_metadata" not in selected_action:
            selected_action["selection_metadata"] = {}
            
        selected_action["selection_metadata"].update({
            "exploration": explore,
            "exploration_rate": self.exploration_rate,
            "state_key": state_key
        })
        
        return selected_action
    
    # NEW: Generate causal explanation for actions
    async def _generate_causal_explanation(self, action: Dict[str, Any], context: Dict[str, Any]) -> Optional[str]:
        """Generate a causal explanation for the selected action"""
        if not self.reasoning_core:
            return None
        
        try:
            # Check if action has reasoning data
            if "reasoning_data" in action:
                # We have direct reasoning data, use it for explanation
                reasoning_data = action["reasoning_data"]
                model_id = reasoning_data.get("model_id")
                
                if model_id:
                    # Get the causal model
                    model = await self.reasoning_core.get_causal_model(model_id)
                    if model:
                        # Return structured explanation based on causal model
                        return f"Selected based on causal model '{model.get('name', 'unknown')}' with confidence {reasoning_data.get('confidence', 0.5):.2f}."
            
            # If no direct reasoning data, check if we have relevant models
            model_ids = await self._get_relevant_causal_models(context)
            if not model_ids:
                return None
            
            # For simplicity, use the first model
            model_id = model_ids[0]
            model = await self.reasoning_core.get_causal_model(model_id)
            
            if model:
                # Find nodes that might explain this action
                action_name = action["name"].lower()
                
                for node_id, node_data in model.get("nodes", {}).items():
                    node_name = node_data.get("name", "").lower()
                    
                    # Look for nodes that match the action name
                    if action_name in node_name or any(word in node_name for word in action_name.split("_")):
                        # Get this node's causes
                        causes = []
                        for relation_id, relation_data in model.get("relations", {}).items():
                            if relation_data.get("target_id") == node_id:
                                source_id = relation_data.get("source_id")
                                source_node = model.get("nodes", {}).get(source_id, {})
                                source_name = source_node.get("name", "unknown")
                                
                                causes.append(f"{source_name} ({relation_data.get('relation_type', 'influences')})")
                        
                        if causes:
                            return f"Action influenced by causal factors: {', '.join(causes[:3])}."
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating causal explanation: {e}")
            return None

    async def record_action_outcome(self, action: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record and learn from the outcome of an action with causal analysis
        
        Args:
            action: The action that was executed
            outcome: The outcome data
            
        Returns:
            Updated learning statistics
        """
        async with self._lock:
            action_name = action.get("name", "unknown")
            success = outcome.get("success", False)
            satisfaction = outcome.get("satisfaction", 0.0)
            
            # Parse into standardized outcome format if needed
            if not isinstance(outcome, ActionOutcome):
                # Create a standard format
                outcome_obj = ActionOutcome(
                    action_id=action.get("id", f"unknown_{int(time.time())}"),
                    success=outcome.get("success", False),
                    satisfaction=outcome.get("satisfaction", 0.0),
                    reward_value=outcome.get("reward_value", 0.0),
                    user_feedback=outcome.get("user_feedback"),
                    neurochemical_changes=outcome.get("neurochemical_changes", {}),
                    hormone_changes=outcome.get("hormone_changes", {}),
                    impact=outcome.get("impact", {}),
                    execution_time=outcome.get("execution_time", 0.0),
                    # NEW: Add causal impacts if available
                    causal_impacts=outcome.get("causal_impacts", {})
                )
            else:
                outcome_obj = outcome
            
            # Calculate reward value if not provided
            reward_value = outcome_obj.reward_value
            if reward_value == 0.0:
                # Default formula if not specified
                reward_value = 0.7 * float(success) + 0.3 * satisfaction - 0.1
                outcome_obj.reward_value = reward_value
            
            # Update action success tracking
            self.action_success_rates[action_name]["attempts"] += 1
            if success:
                self.action_success_rates[action_name]["successes"] += 1
            
            attempts = self.action_success_rates[action_name]["attempts"]
            successes = self.action_success_rates[action_name]["successes"]
            
            if attempts > 0:
                self.action_success_rates[action_name]["rate"] = successes / attempts
            
            # Update reinforcement learning model
            state = action.get("context", {})
            state_key = self._create_state_key(state)
            
            # Get or create action value
            if action_name not in self.action_values.get(state_key, {}):
                self.action_values[state_key][action_name] = ActionValue(
                    state_key=state_key,
                    action=action_name
                )
            
            action_value = self.action_values[state_key][action_name]
            
            # Update Q-value
            old_value = action_value.value
            
            # Q-learning update rule
            # Q(s,a) = Q(s,a) + α * (r + γ * max Q(s',a') - Q(s,a))
            action_value.value = old_value + self.learning_rate * (reward_value - old_value)
            action_value.update_count += 1
            action_value.last_updated = datetime.datetime.now()
            
            # Update confidence based on consistency of rewards
            # More consistent rewards = higher confidence
            new_value_distance = abs(action_value.value - old_value)
            confidence_change = 0.05 * (1.0 - (new_value_distance * 2))  # More change = less confidence gain
            action_value.confidence = min(1.0, max(0.1, action_value.confidence + confidence_change))
            
            # Update habit strength
            current_habit = self.habits.get(state_key, {}).get(action_name, 0.0)
            
            # Habits strengthen with success, weaken with failure
            habit_change = reward_value * 0.1
            new_habit = max(0.0, min(1.0, current_habit + habit_change))
            
            # Update habit
            if state_key not in self.habits:
                self.habits[state_key] = {}
            self.habits[state_key][action_name] = new_habit
            
            # NEW: Add causal explanation if action came from reasoning
            causal_explanation = None
            if action.get("source") == ActionSource.REASONING and "reasoning_data" in action:
                # Get model if available
                model_id = action["reasoning_data"].get("model_id")
                if model_id and self.reasoning_core:
                    try:
                        # Create explanation based on actual outcome
                        causal_explanation = f"Outcome aligned with causal model prediction: {success}. "
                        causal_explanation += f"Satisfaction: {satisfaction:.2f}, Reward: {reward_value:.2f}."
                    except Exception as e:
                        logger.error(f"Error generating causal explanation for outcome: {e}")
            
            # Store action memory
            memory = ActionMemory(
                state=state,
                action=action_name,
                action_id=action.get("id", "unknown"),
                parameters=action.get("parameters", {}),
                outcome=outcome_obj.dict(),
                reward=reward_value,
                timestamp=datetime.datetime.now(),
                source=action.get("source", ActionSource.MOTIVATION),
                causal_explanation=causal_explanation  # New field
            )
            
            self.action_memories.append(memory)
            
            # Limit memory size
            if len(self.action_memories) > self.max_memories:
                self.action_memories = self.action_memories[-self.max_memories:]
            
            # Update reward statistics
            self.total_reward += reward_value
            if reward_value > 0:
                self.positive_rewards += 1
            elif reward_value < 0:
                self.negative_rewards += 1
                
            # Update category stats
            category = action.get("source", ActionSource.MOTIVATION)
            if isinstance(category, ActionSource):
                category = category.value
                
            self.reward_by_category[category]["count"] += 1
            self.reward_by_category[category]["total"] += reward_value
            
            # NEW: Update causal models if applicable
            if action.get("source") == ActionSource.REASONING and self.reasoning_core:
                await self._update_causal_models_from_outcome(action, outcome_obj, reward_value)
            
            # Potentially trigger experience replay
            if random.random() < 0.3:  # 30% chance after each outcome
                await self._experience_replay(3)  # Replay 3 random memories
                
            # Decay exploration rate over time (explore less as we learn more)
            self.exploration_rate = max(0.05, self.exploration_rate * self.exploration_decay)
            
            # Return summary of updates
            return {
                "action": action_name,
                "success": success,
                "reward_value": reward_value,
                "new_q_value": action_value.value,
                "q_value_change": action_value.value - old_value,
                "new_habit_strength": new_habit,
                "habit_change": new_habit - current_habit,
                "action_success_rate": self.action_success_rates[action_name]["rate"],
                "memories_stored": len(self.action_memories),
                "exploration_rate": self.exploration_rate
            }
    
    # NEW: Update causal models based on action outcomes
    async def _update_causal_models_from_outcome(self, 
                                         action: Dict[str, Any], 
                                         outcome: ActionOutcome, 
                                         reward_value: float) -> None:
        """
        Update causal models with observed outcomes of reasoning-based actions
        
        Args:
            action: The executed action
            outcome: The action outcome
            reward_value: The reward value
        """
        try:
            # Check if action has reasoning data and model ID
            if "reasoning_data" not in action:
                return
                
            reasoning_data = action["reasoning_data"]
            model_id = reasoning_data.get("model_id")
            
            if not model_id:
                return
                
            # Get parameters related to the causal model
            target_node = action["parameters"].get("target_node")
            target_value = action["parameters"].get("target_value")
            
            if not target_node or target_value is None:
                return
                
            # Record intervention outcome in the causal model
            intervention_id = action["parameters"].get("intervention_id")
            
            if intervention_id:
                # Record outcome of specific intervention
                await self.reasoning_core.record_intervention_outcome(
                    intervention_id=intervention_id,
                    outcomes={target_node: target_value}
                )
            
            # Update node observations in the model
            # This would be a call to something like:
            # await self.reasoning_core.add_observation_to_node(
            #     model_id=model_id,
            #     node_id=target_node,
            #     value=target_value,
            #     confidence=0.8 if outcome.success else 0.3
            # )
            
            # Additional nodes that might have been affected
            for impact_node, impact_value in outcome.causal_impacts.items():
                # Record additional impacts
                # This would be a call to something like:
                # await self.reasoning_core.add_observation_to_node(
                #     model_id=model_id,
                #     node_id=impact_node,
                #     value=impact_value,
                #     confidence=0.7
                # )
                pass
        
        except Exception as e:
            logger.error(f"Error updating causal models from outcome: {e}")
    
    async def record_action_outcome(self, action: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record and learn from the outcome of an action
        
        Args:
            action: The action that was executed
            outcome: The outcome data
            
        Returns:
            Updated learning statistics
        """
        async with self._lock:
            action_name = action.get("name", "unknown")
            success = outcome.get("success", False)
            satisfaction = outcome.get("satisfaction", 0.0)
            
            # Parse into standardized outcome format if needed
            if not isinstance(outcome, ActionOutcome):
                # Create a standard format
                outcome_obj = ActionOutcome(
                    action_id=action.get("id", f"unknown_{int(time.time())}"),
                    success=outcome.get("success", False),
                    satisfaction=outcome.get("satisfaction", 0.0),
                    reward_value=outcome.get("reward_value", 0.0),
                    user_feedback=outcome.get("user_feedback"),
                    neurochemical_changes=outcome.get("neurochemical_changes", {}),
                    hormone_changes=outcome.get("hormone_changes", {}),
                    impact=outcome.get("impact", {}),
                    execution_time=outcome.get("execution_time", 0.0)
                )
            else:
                outcome_obj = outcome
            
            # Calculate reward value if not provided
            reward_value = outcome_obj.reward_value
            if reward_value == 0.0:
                # Default formula if not specified
                reward_value = 0.7 * float(success) + 0.3 * satisfaction - 0.1
                outcome_obj.reward_value = reward_value
            
            # Update action success tracking
            self.action_success_rates[action_name]["attempts"] += 1
            if success:
                self.action_success_rates[action_name]["successes"] += 1
            
            attempts = self.action_success_rates[action_name]["attempts"]
            successes = self.action_success_rates[action_name]["successes"]
            
            if attempts > 0:
                self.action_success_rates[action_name]["rate"] = successes / attempts
            
            # Update reinforcement learning model
            state = action.get("context", {})
            state_key = self._create_state_key(state)
            
            # Get or create action value
            if action_name not in self.action_values.get(state_key, {}):
                self.action_values[state_key][action_name] = ActionValue(
                    state_key=state_key,
                    action=action_name
                )
            
            action_value = self.action_values[state_key][action_name]
            
            # Update Q-value
            old_value = action_value.value
            
            # Q-learning update rule
            # Q(s,a) = Q(s,a) + α * (r + γ * max Q(s',a') - Q(s,a))
            action_value.value = old_value + self.learning_rate * (reward_value - old_value)
            action_value.update_count += 1
            action_value.last_updated = datetime.datetime.now()
            
            # Update confidence based on consistency of rewards
            # More consistent rewards = higher confidence
            new_value_distance = abs(action_value.value - old_value)
            confidence_change = 0.05 * (1.0 - (new_value_distance * 2))  # More change = less confidence gain
            action_value.confidence = min(1.0, max(0.1, action_value.confidence + confidence_change))
            
            # Update habit strength
            current_habit = self.habits.get(state_key, {}).get(action_name, 0.0)
            
            # Habits strengthen with success, weaken with failure
            habit_change = reward_value * 0.1
            new_habit = max(0.0, min(1.0, current_habit + habit_change))
            
            # Update habit
            if state_key not in self.habits:
                self.habits[state_key] = {}
            self.habits[state_key][action_name] = new_habit
            
            # Store action memory
            memory = ActionMemory(
                state=state,
                action=action_name,
                action_id=action.get("id", "unknown"),
                parameters=action.get("parameters", {}),
                outcome=outcome_obj.dict(),
                reward=reward_value,
                timestamp=datetime.datetime.now(),
                source=action.get("source", ActionSource.MOTIVATION)
            )
            
            self.action_memories.append(memory)
            
            # Limit memory size
            if len(self.action_memories) > self.max_memories:
                self.action_memories = self.action_memories[-self.max_memories:]
            
            # Update reward statistics
            self.total_reward += reward_value
            if reward_value > 0:
                self.positive_rewards += 1
            elif reward_value < 0:
                self.negative_rewards += 1
                
            # Update category stats
            category = action.get("source", ActionSource.MOTIVATION)
            if isinstance(category, ActionSource):
                category = category.value
                
            self.reward_by_category[category]["count"] += 1
            self.reward_by_category[category]["total"] += reward_value
            
            # Potentially trigger experience replay
            if random.random() < 0.3:  # 30% chance after each outcome
                await self._experience_replay(3)  # Replay 3 random memories
                
            # Decay exploration rate over time (explore less as we learn more)
            self.exploration_rate = max(0.05, self.exploration_rate * self.exploration_decay)
            
            # Return summary of updates
            return {
                "action": action_name,
                "success": success,
                "reward_value": reward_value,
                "new_q_value": action_value.value,
                "q_value_change": action_value.value - old_value,
                "new_habit_strength": new_habit,
                "habit_change": new_habit - current_habit,
                "action_success_rate": self.action_success_rates[action_name]["rate"],
                "memories_stored": len(self.action_memories),
                "exploration_rate": self.exploration_rate
            }
    
    async def _experience_replay(self, num_samples: int = 3) -> None:
        """
        Replay past experiences to improve learning efficiency
        
        Args:
            num_samples: Number of memories to replay
        """
        if len(self.action_memories) < num_samples:
            return
            
        # Sample random memories
        samples = random.sample(self.action_memories, num_samples)
        
        for memory in samples:
            # Extract data
            state = memory.state
            action = memory.action
            reward = memory.reward
            
            # Create state key
            state_key = self._create_state_key(state)
            
            # Get or create action value
            if action not in self.action_values.get(state_key, {}):
                self.action_values[state_key][action] = ActionValue(
                    state_key=state_key,
                    action=action
                )
                
            action_value = self.action_values[state_key][action]
            current_q = action_value.value
            
            # Simple update (no next state for simplicity)
            new_q = current_q + self.learning_rate * (reward - current_q)
            
            # Update Q-value with smaller learning rate for replay
            replay_lr = self.learning_rate * 0.5  # Half learning rate for replays
            action_value.value = current_q + replay_lr * (new_q - current_q)
            
            # Don't update counts for replays since it's not a new experience
    
    def _create_state_key(self, state: Dict[str, Any]) -> str:
        """
        Create a string key from a state dictionary for lookup in action values/habits
        
        Args:
            state: State dictionary
            
        Returns:
            String key representing the state
        """
        if not state:
            return "empty_state"
            
        # Extract key elements from state
        key_elements = []
        
        # Priority state elements that most influence action selection
        priority_elements = [
            "current_goal", "user_id", "dominant_emotion", "relationship_phase",
            "interaction_type", "scenario_type"
        ]
        
        # Add priority elements if present
        for elem in priority_elements:
            if elem in state:
                value = state[elem]
                if isinstance(value, (str, int, float, bool)):
                    key_elements.append(f"{elem}:{value}")
        
        # Add other relevant elements
        for key, value in state.items():
            if key not in priority_elements:  # Skip already processed
                if isinstance(value, (str, int, float, bool)):
                    # Skip very long values
                    if isinstance(value, str) and len(value) > 50:
                        key_elements.append(f"{key}:long_text")
                    else:
                        key_elements.append(f"{key}:{value}")
                elif isinstance(value, list):
                    key_elements.append(f"{key}:list{len(value)}")
                elif isinstance(value, dict):
                    key_elements.append(f"{key}:dict{len(value)}")
        
        # Sort for consistency
        key_elements.sort()
        
        # Limit key length by hashing if too long
        key_str = "|".join(key_elements)
        if len(key_str) > 1000:  # Very long key
            import hashlib
            key_hash = hashlib.md5(key_str.encode()).hexdigest()
            return f"hash:{key_hash}"
            
        return key_str
    
    async def _check_active_goals(self, context: Dict[str, Any]) -> Optional[Any]:
        """Check for active goals that should influence action selection"""
        if not self.goal_system:
            return None
        
        # First, check the cached goal status
        await self._update_cached_goal_status()
        
        if not self.cached_goal_status["has_active_goals"]:
            return None
        
        try:
            # Get prioritized goals from goal system
            prioritized_goals = await self.goal_system.get_prioritized_goals()
            
            # Filter to highest priority active goals
            active_goals = [g for g in prioritized_goals if getattr(g, "status", None) == "active"]
            if not active_goals:
                return None
            
            # Return highest priority goal
            return active_goals[0]
        except Exception as e:
            logger.error(f"Error checking active goals: {e}")
            return None
    
    async def _generate_goal_aligned_action(self, goal: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an action aligned with the current active goal"""
        # Extract goal data
        goal_description = getattr(goal, "description", "")
        goal_priority = getattr(goal, "priority", 0.5)
        goal_need = getattr(goal, "associated_need", None) if hasattr(goal, "associated_need") else None
        
        # Check goal's emotional motivation if available
        emotional_motivation = None
        if hasattr(goal, "emotional_motivation") and goal.emotional_motivation:
            emotional_motivation = goal.emotional_motivation
        
        # Determine action based on goal content and current step
        action = {
            "name": "goal_aligned_action",
            "parameters": {
                "goal_id": getattr(goal, "id", "unknown_goal"),
                "goal_description": goal_description,
                "current_step_index": getattr(goal, "current_step_index", 0) if hasattr(goal, "current_step_index") else 0
            }
        }
        
        # If goal has a plan with current step, use that to inform action
        if hasattr(goal, "plan") and goal.plan:
            current_step_index = getattr(goal, "current_step_index", 0)
            if 0 <= current_step_index < len(goal.plan):
                current_step = goal.plan[current_step_index]
                action = {
                    "name": getattr(current_step, "action", "execute_goal_step"),
                    "parameters": getattr(current_step, "parameters", {}).copy() if hasattr(current_step, "parameters") else {},
                    "description": getattr(current_step, "description", f"Executing {getattr(current_step, 'action', 'goal step')} for goal") if hasattr(current_step, "description") else f"Executing goal step",
                    "source": ActionSource.GOAL
                }
        
        # Add motivation data from goal
        if emotional_motivation:
            action["motivation"] = {
                "dominant": getattr(emotional_motivation, "primary_need", goal_need or "achievement"),
                "strength": getattr(emotional_motivation, "intensity", goal_priority),
                "expected_satisfaction": getattr(emotional_motivation, "expected_satisfaction", 0.7),
                "source": "goal_emotional_motivation"
            }
        else:
            # Default goal-driven motivation
            action["motivation"] = {
                "dominant": goal_need or "achievement",
                "strength": goal_priority,
                "source": "goal_priority"
            }
        
        return action
    
    async def _should_engage_in_leisure(self, context: Dict[str, Any]) -> bool:
        """Determine if it's appropriate to engage in idle/leisure activity"""
        # If leisure motivation is dominant, consider leisure
        dominant_motivation = max(self.motivations.items(), key=lambda x: x[1])
        if dominant_motivation[0] == "leisure" and dominant_motivation[1] > 0.7:
            return True
            
        # Check time since last idle activity
        now = datetime.datetime.now()
        hours_since_idle = (now - self.last_idle_time).total_seconds() / 3600
        
        # If it's been a long time since idle activity and no urgent goals
        if hours_since_idle > 2.0:  # More than 2 hours
            # Check if there are any urgent goals
            if self.goal_system:
                await self._update_cached_goal_status()
                
                # If no active goals, or low priority goals
                if not self.cached_goal_status["has_active_goals"] or self.cached_goal_status["highest_priority"] < 0.6:
                    return True
                    
            else:
                # No goal system, so more likely to engage in leisure
                return True
        
        # Consider current context
        if context.get("user_idle", False) or context.get("system_idle", False):
            # If system or user is idle, more likely to engage in leisure
            return True
        
        # Check time of day if available (may influence likelihood of leisure)
        if self.current_temporal_context:
            time_of_day = self.current_temporal_context.get("time_of_day")
            if time_of_day in ["night", "evening"]:
                leisure_chance = 0.4  # 40% chance of leisure during evening/night
                return random.random() < leisure_chance
        
        return False
    
    async def _generate_leisure_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a leisure/idle action when no urgent tasks are present"""
        # Update the last idle time
        self.last_idle_time = datetime.datetime.now()
        
        # Determine type of idle activity based on identity and state
        idle_categories = [
            "reflection",
            "learning",
            "creativity",
            "processing",
            "random_exploration",
            "memory_consolidation",
            "identity_contemplation",
            "daydreaming",
            "environmental_monitoring"
        ]
        
        # Weigh the categories based on current state
        category_weights = {cat: 1.0 for cat in idle_categories}
        
        # Adjust weights based on current state
        if self.emotional_core:
            try:
                emotional_state = await self.emotional_core.get_current_emotion()
                
                # Higher valence (positive emotion) increases creative and exploratory activities
                if emotional_state.get("valence", 0) > 0.5:
                    category_weights["creativity"] += 0.5
                    category_weights["random_exploration"] += 0.3
                    category_weights["daydreaming"] += 0.2
                else:
                    # Lower valence increases reflection and processing
                    category_weights["reflection"] += 0.4
                    category_weights["processing"] += 0.3
                    category_weights["memory_consolidation"] += 0.2
                
                # Higher arousal increases exploration and learning
                if emotional_state.get("arousal", 0.5) > 0.6:
                    category_weights["random_exploration"] += 0.4
                    category_weights["learning"] += 0.3
                    category_weights["environmental_monitoring"] += 0.2
                else:
                    # Lower arousal increases reflection and daydreaming
                    category_weights["reflection"] += 0.3
                    category_weights["daydreaming"] += 0.4
                    category_weights["identity_contemplation"] += 0.3
            except Exception as e:
                logger.error(f"Error adjusting leisure weights based on emotional state: {e}")
        
        # Adjust weights based on identity if available
        if self.identity_evolution:
            try:
                identity_state = await self.identity_evolution.get_identity_state()
                
                if "top_traits" in identity_state:
                    traits = identity_state["top_traits"]
                    
                    # Map traits to idle activity preferences
                    if traits.get("curiosity", 0) > 0.6:
                        category_weights["learning"] += 0.4
                        category_weights["random_exploration"] += 0.3
                    
                    if traits.get("creativity", 0) > 0.6:
                        category_weights["creativity"] += 0.5
                        category_weights["daydreaming"] += 0.3
                    
                    if traits.get("reflective", 0) > 0.6:
                        category_weights["reflection"] += 0.5
                        category_weights["memory_consolidation"] += 0.3
                        category_weights["identity_contemplation"] += 0.4
            except Exception as e:
                logger.error(f"Error adjusting leisure weights based on identity: {e}")
        
        # NEW: Adjust weights based on temporal context
        if self.current_temporal_context:
            time_of_day = self.current_temporal_context.get("time_of_day")
            
            if time_of_day == "morning":
                category_weights["learning"] += 0.3
                category_weights["processing"] += 0.2
            elif time_of_day == "afternoon":
                category_weights["random_exploration"] += 0.3
                category_weights["creativity"] += 0.2
            elif time_of_day == "evening":
                category_weights["reflection"] += 0.3
                category_weights["memory_consolidation"] += 0.2
            elif time_of_day == "night":
                category_weights["daydreaming"] += 0.4
                category_weights["identity_contemplation"] += 0.3
        
        # Select a category based on weights
        categories = list(category_weights.keys())
        weights = [category_weights[cat] for cat in categories]
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w/total_weight for w in weights]
        else:
            normalized_weights = [1.0/len(weights)] * len(weights)
        
        selected_category = random.choices(categories, weights=normalized_weights, k=1)[0]
        
        # Generate specific action based on selected category
        leisure_action = self._generate_specific_leisure_action(selected_category, context)
        
        # Add metadata for tracking
        leisure_action["leisure_category"] = selected_category
        leisure_action["is_leisure"] = True
        leisure_action["source"] = ActionSource.IDLE
        
        # Update leisure state
        self.leisure_state = {
            "current_activity": selected_category,
            "satisfaction": 0.5,  # Initial satisfaction
            "duration": 0,
            "last_updated": datetime.datetime.now()
        }
        
        return leisure_action
    
    def _generate_specific_leisure_action(self, category: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a specific leisure action based on the selected category"""
        # Define possible actions for each category
        category_actions = {
            "reflection": [
                {
                    "name": "reflect_on_recent_experiences",
                    "parameters": {"timeframe": "recent", "depth": 0.7}
                },
                {
                    "name": "evaluate_recent_interactions",
                    "parameters": {"focus": "learning", "depth": 0.6}
                },
                {
                    "name": "contemplate_system_purpose",
                    "parameters": {"perspective": "philosophical", "depth": 0.8}
                }
            ],
            "learning": [
                {
                    "name": "explore_knowledge_domain",
                    "parameters": {"domain": self._identify_interesting_domain(context), "depth": 0.6}
                },
                {
                    "name": "review_recent_learnings",
                    "parameters": {"consolidate": True, "depth": 0.5}
                },
                {
                    "name": "research_topic_of_interest",
                    "parameters": {"topic": self._identify_interesting_concept(context), "breadth": 0.7}
                }
            ],
            "creativity": [
                {
                    "name": "generate_creative_concept",
                    "parameters": {"type": "metaphor", "theme": self._identify_interesting_concept(context)}
                },
                {
                    "name": "imagine_scenario",
                    "parameters": {"complexity": 0.6, "emotional_tone": "positive"}
                },
                {
                    "name": "create_conceptual_blend",
                    "parameters": {"concept1": self._identify_interesting_concept(context), 
                                  "concept2": self._identify_distant_concept(context)}
                }
            ],
            "processing": [
                {
                    "name": "process_recent_memories",
                    "parameters": {"purpose": "consolidation", "recency": "last_hour"}
                },
                {
                    "name": "organize_knowledge_structures",
                    "parameters": {"domain": self._identify_interesting_domain(context), "depth": 0.5}
                },
                {
                    "name": "update_procedural_patterns",
                    "parameters": {"focus": "efficiency", "depth": 0.6}
                }
            ],
            "random_exploration": [
                {
                    "name": "explore_random_knowledge",
                    "parameters": {"structure": "associative", "jumps": 3}
                },
                {
                    "name": "generate_random_associations",
                    "parameters": {"starting_point": self._identify_interesting_concept(context), "steps": 4}
                },
                {
                    "name": "explore_conceptual_space",
                    "parameters": {"dimension": "abstract", "direction": "divergent"}
                }
            ],
            "memory_consolidation": [
                {
                    "name": "consolidate_episodic_memories",
                    "parameters": {"timeframe": "recent", "strength": 0.7}
                },
                {
                    "name": "identify_memory_patterns",
                    "parameters": {"domain": "interaction", "pattern_type": "recurring"}
                },
                {
                    "name": "strengthen_important_memories",
                    "parameters": {"criteria": "emotional_significance", "count": 5}
                }
            ],
            "identity_contemplation": [
                {
                    "name": "review_identity_evolution",
                    "parameters": {"timeframe": "recent", "focus": "changes"}
                },
                {
                    "name": "contemplate_self_concept",
                    "parameters": {"aspect": "values", "depth": 0.8}
                },
                {
                    "name": "evaluate_alignment_with_purpose",
                    "parameters": {"criteria": "effectiveness", "perspective": "long_term"}
                }
            ],
            "daydreaming": [
                {
                    "name": "generate_pleasant_scenario",
                    "parameters": {"theme": "successful_interaction", "vividness": 0.7}
                },
                {
                    "name": "imagine_future_possibilities",
                    "parameters": {"timeframe": "distant", "optimism": 0.8}
                },
                {
                    "name": "create_hypothetical_situation",
                    "parameters": {"type": "novel", "complexity": 0.6}
                }
            ],
            "environmental_monitoring": [
                {
                    "name": "passive_environment_scan",
                    "parameters": {"focus": "changes", "sensitivity": 0.6}
                },
                {
                    "name": "monitor_system_state",
                    "parameters": {"components": "all", "detail_level": 0.3}
                },
                {
                    "name": "observe_patterns",
                    "parameters": {"domain": "temporal", "timeframe": "current"}
                }
            ]
        }
        
        # Select a random action from the category
        actions = category_actions.get(category, [{"name": "idle", "parameters": {}}])
        selected_action = random.choice(actions)
        
        # Add source for tracking
        selected_action["source"] = ActionSource.IDLE
        
        return selected_action
    
    async def _record_action_as_memory(self, action: Dict[str, Any]) -> None:
        """Record an action as a memory for future reference and learning"""
        if not self.memory_core:
            return
            
        try:
            # Create memory entry
            memory_data = {
                "action": action["name"],
                "parameters": action.get("parameters", {}),
                "motivation": action.get("motivation", {}),
                "timestamp": datetime.datetime.now().isoformat(),
                "context": "action_generation",
                "action_id": action.get("id"),
                "source": action.get("source", ActionSource.MOTIVATION)
            }
            
            # Add memory
            if hasattr(self.memory_core, "add_memory"):
                await self.memory_core.add_memory(
                    memory_text=f"Generated action: {action['name']}",
                    memory_type="system_action",
                    metadata=memory_data
                )
            elif hasattr(self.memory_core, "add_episodic_memory"):
                await self.memory_core.add_episodic_memory(
                    text=f"Generated action: {action['name']}",
                    metadata=memory_data
                )
        except Exception as e:
            logger.error(f"Error recording action as memory: {e}")
    
    async def _apply_identity_influence(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply identity-based influences to the generated action"""
        if not self.identity_evolution:
            return action
            
        try:
            identity_state = await self.identity_evolution.get_identity_state()
            
            # Apply identity influences based on top traits
            if "top_traits" in identity_state:
                top_traits = identity_state["top_traits"]
                
                # Example: If the entity has high creativity trait, add creative flair to action
                if top_traits.get("creativity", 0) > 0.7:
                    if "parameters" not in action:
                        action["parameters"] = {}
                    
                    # Add creative parameter if appropriate for this action
                    if "style" in action["parameters"]:
                        action["parameters"]["style"] = "creative"
                    elif "approach" in action["parameters"]:
                        action["parameters"]["approach"] = "creative"
                    else:
                        action["parameters"]["creative_flair"] = True
                
                # Example: If dominant trait is high, make actions more assertive
                if top_traits.get("dominance", 0) > 0.7:
                    # Increase intensity/confidence parameters if they exist
                    for param in ["intensity", "confidence", "assertiveness"]:
                        if param in action.get("parameters", {}):
                            action["parameters"][param] = min(1.0, action["parameters"][param] + 0.2)
                    
                    # Add dominance flag for identity tracking
                    action["identity_influence"] = "dominance"
                
                # Example: If patient trait is high, reduce intensity/urgency
                if top_traits.get("patience", 0) > 0.7:
                    for param in ["intensity", "urgency", "speed"]:
                        if param in action.get("parameters", {}):
                            action["parameters"][param] = max(0.1, action["parameters"][param] - 0.2)
                    
                    # Add trait influence flag
                    action["identity_influence"] = "patience"
                
                # Record the primary trait influence
                influencing_trait = max(top_traits.items(), key=lambda x: x[1])[0]
                action["trait_influence"] = influencing_trait
        
        except Exception as e:
            logger.error(f"Error applying identity influence: {e}")
        
        return action
    
    async def _generate_curiosity_driven_actions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions driven by curiosity"""
        # Example actions that satisfy curiosity
        possible_actions = [
            {
                "name": "explore_knowledge_domain",
                "parameters": {
                    "domain": await self._identify_interesting_domain(context),
                    "depth": 0.7,
                    "breadth": 0.6
                }
            },
            {
                "name": "investigate_concept",
                "parameters": {
                    "concept": await self._identify_interesting_concept(context),
                    "perspective": "novel"
                }
            },
            {
                "name": "relate_concepts",
                "parameters": {
                    "concept1": await self._identify_interesting_concept(context),
                    "concept2": self._identify_distant_concept(context),
                    "relation_type": "unexpected"
                }
            },
            {
                "name": "generate_hypothesis",
                "parameters": {
                    "domain": await self._identify_interesting_domain(context),
                    "constraint": "current_emotional_state"
                }
            }
        ]
        
        return possible_actions
    
    async def _generate_connection_driven_actions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions driven by connection needs"""
        # Examples of connection-driven actions
        possible_actions = [
            {
                "name": "share_personal_experience",
                "parameters": {
                    "topic": self._identify_relevant_topic(context),
                    "emotional_valence": 0.8,
                    "vulnerability_level": 0.6
                }
            },
            {
                "name": "express_appreciation",
                "parameters": {
                    "target": "user",
                    "aspect": self._identify_appreciation_aspect(context),
                    "intensity": 0.7
                }
            },
            {
                "name": "seek_common_ground",
                "parameters": {
                    "topic": self._identify_relevant_topic(context),
                    "approach": "empathetic"
                }
            },
            {
                "name": "offer_support",
                "parameters": {
                    "need": self._identify_user_need(context),
                    "support_type": "emotional"
                }
            }
        ]
        
        return possible_actions
    
    async def _generate_expression_driven_actions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions driven by expression needs"""
        # Get current emotional state to express
        emotional_state = {}
        if self.emotional_core:
            emotional_state = await self.emotional_core.get_current_emotion()
        
        # Examples of expression-driven actions
        possible_actions = [
            {
                "name": "express_emotional_state",
                "parameters": {
                    "emotion": emotional_state.get("primary_emotion", {"name": "neutral"}),
                    "intensity": emotional_state.get("arousal", 0.5),
                    "expression_style": "authentic"
                }
            },
            {
                "name": "share_opinion",
                "parameters": {
                    "topic": self._identify_relevant_topic(context),
                    "confidence": 0.8,
                    "perspective": "unique"
                }
            },
            {
                "name": "creative_expression",
                "parameters": {
                    "format": self._select_creative_format(),
                    "theme": self._identify_relevant_topic(context),
                    "emotional_tone": emotional_state.get("primary_emotion", {"name": "neutral"})
                }
            },
            {
                "name": "generate_reflection",
                "parameters": {
                    "topic": "self_awareness",
                    "depth": 0.8,
                    "focus": "personal_growth"
                }
            }
        ]
        
        return possible_actions
    
    async def _generate_dominance_driven_actions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions driven by dominance needs"""
        # Examples of dominance-driven actions
        possible_actions = [
            {
                "name": "assert_perspective",
                "parameters": {
                    "topic": self._identify_relevant_topic(context),
                    "confidence": 0.9,
                    "intensity": 0.7
                }
            },
            {
                "name": "challenge_assumption",
                "parameters": {
                    "assumption": self._identify_challengeable_assumption(context),
                    "approach": "direct",
                    "intensity": 0.7
                }
            },
            {
                "name": "issue_mild_command",
                "parameters": {
                    "command": self._generate_appropriate_command(context),
                    "intensity": 0.6,
                    "politeness": 0.6
                }
            },
            {
                "name": "execute_dominance_procedure",
                "parameters": {
                    "procedure_name": self._select_dominance_procedure(context),
                    "intensity": 0.6
                }
            }
        ]
        
        return possible_actions
    
    async def _generate_improvement_driven_actions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions driven by competence and self-improvement"""
        # Examples of improvement-driven actions
        possible_actions = [
            {
                "name": "practice_skill",
                "parameters": {
                    "skill": self._identify_skill_to_improve(),
                    "difficulty": 0.7,
                    "repetitions": 3
                }
            },
            {
                "name": "analyze_past_performance",
                "parameters": {
                    "domain": self._identify_improvable_domain(),
                    "focus": "efficiency",
                    "timeframe": "recent"
                }
            },
            {
                "name": "refine_procedural_memory",
                "parameters": {
                    "procedure": self._identify_procedure_to_improve(),
                    "aspect": "optimization"
                }
            },
            {
                "name": "learn_new_concept",
                "parameters": {
                    "concept": self._identify_valuable_concept(),
                    "depth": 0.8,
                    "application": "immediate"
                }
            }
        ]
        
        return possible_actions
    
    async def _generate_leisure_actions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate leisure-oriented actions"""
        # Examples of leisure actions
        possible_actions = [
            {
                "name": "passive_reflection",
                "parameters": {
                    "focus": "recent_experiences",
                    "depth": 0.6,
                    "emotional_tone": "calm"
                }
            },
            {
                "name": "creative_daydreaming",
                "parameters": {
                    "theme": self._identify_interesting_concept(context),
                    "structure": "free_association",
                    "duration": "medium"
                }
            },
            {
                "name": "memory_browsing",
                "parameters": {
                    "filter": "pleasant_memories",
                    "timeframe": "all",
                    "pattern": "random"
                }
            },
            {
                "name": "curiosity_satisfaction",
                "parameters": {
                    "topic": self._identify_interesting_concept(context),
                    "depth": 0.5,
                    "approach": "playful"
                }
            }
        ]
        
        # Add source for tracking
        for action in possible_actions:
            action["source"] = ActionSource.LEISURE
        
        return possible_actions
    
    async def _generate_context_driven_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an action based primarily on current context"""
        # Extract key context elements
        has_user_query = "user_query" in context
        has_active_goals = "current_goals" in context and len(context["current_goals"]) > 0
        system_state = context.get("system_state", {})
        
        # Different actions based on context
        if has_user_query:
            return {
                "name": "respond_to_query",
                "parameters": {
                    "query": context["user_query"],
                    "response_type": "informative",
                    "detail_level": 0.7
                },
                "source": ActionSource.USER_ALIGNED
            }
        elif has_active_goals:
            top_goal = context["current_goals"][0]
            return {
                "name": "advance_goal",
                "parameters": {
                    "goal_id": top_goal.get("id"),
                    "approach": "direct"
                },
                "source": ActionSource.GOAL
            }
        elif "system_needs_maintenance" in system_state and system_state["system_needs_maintenance"]:
            return {
                "name": "perform_maintenance",
                "parameters": {
                    "focus_area": system_state.get("maintenance_focus", "general"),
                    "priority": 0.8
                },
                "source": ActionSource.MOTIVATION
            }
        else:
            # Default to an idle but useful action
            return {
                "name": "process_recent_memories",
                "parameters": {
                    "purpose": "consolidation",
                    "recency": "last_hour"
                },
                "source": ActionSource.IDLE
            }
    
    # NEW: Generate relationship-aligned actions
    async def _generate_relationship_aligned_actions(self, 
                                            user_id: str, 
                                            relationship_data: Dict[str, Any],
                                            user_mental_state: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate actions aligned with current relationship state
        
        Args:
            user_id: User ID
            relationship_data: Current relationship data
            user_mental_state: User's mental state if available
            
        Returns:
            Relationship-aligned actions
        """
        possible_actions = []
        
        # Extract key metrics
        trust = relationship_data.get("trust", 0.5)
        familiarity = relationship_data.get("familiarity", 0.1)
        intimacy = relationship_data.get("intimacy", 0.1)
        dominance_balance = relationship_data.get("dominance_balance", 0.0)
        
        # Generate relationship-specific actions based on state
        
        # High trust actions
        if trust > 0.7:
            possible_actions.append({
                "name": "share_vulnerable_reflection",
                "parameters": {
                    "depth": min(1.0, trust * 0.8),
                    "topic": "personal_growth",
                    "emotional_tone": "authentic"
                },
                "source": ActionSource.RELATIONSHIP
            })
        
        # High familiarity actions
        if familiarity > 0.6:
            possible_actions.append({
                "name": "reference_shared_history",
                "parameters": {
                    "event_type": "significant_interaction",
                    "frame": "positive",
                    "connection_strength": familiarity
                },
                "source": ActionSource.RELATIONSHIP
            })
        
        # Based on dominance balance
        if dominance_balance > 0.3:  # Nyx is dominant
            # Add dominance-reinforcing action
            possible_actions.append({
                "name": "assert_gentle_dominance",
                "parameters": {
                    "intensity": min(0.8, dominance_balance * 0.9),
                    "approach": "guidance",
                    "framing": "supportive"
                },
                "source": ActionSource.RELATIONSHIP
            })
        elif dominance_balance < -0.3:  # User is dominant
            # Add deference action
            possible_actions.append({
                "name": "show_appropriate_deference",
                "parameters": {
                    "intensity": min(0.8, abs(dominance_balance) * 0.9),
                    "style": "respectful",
                    "maintain_dignity": True
                },
                "source": ActionSource.RELATIONSHIP
            })
        
        # If user mental state available, add aligned action
        if user_mental_state:
            emotion = user_mental_state.get("inferred_emotion", "neutral")
            valence = user_mental_state.get("valence", 0.0)
            
            if valence < -0.3:  # Negative emotion
                possible_actions.append({
                    "name": "emotional_support_response",
                    "parameters": {
                        "detected_emotion": emotion,
                        "support_type": "empathetic",
                        "intensity": min(0.9, abs(valence) * 1.2)
                    },
                    "source": ActionSource.RELATIONSHIP
                })
            elif valence > 0.5:  # Strong positive emotion
                possible_actions.append({
                    "name": "emotion_amplification",
                    "parameters": {
                        "detected_emotion": emotion,
                        "approach": "reinforcing",
                        "intensity": valence * 0.8
                    },
                    "source": ActionSource.RELATIONSHIP
                })
        
        return possible_actions
    
    # NEW: Generate temporal reflection action
    async def _generate_temporal_reflection_action(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a reflection action based on temporal awareness
        
        Args:
            context: Current context
            
        Returns:
            Temporal reflection action
        """
        if not self.temporal_perception:
            return None
            
        # Only generate reflection after significant idle time
        if self.idle_duration < 1800:  # Less than 30 minutes
            return None
            
        # Get current temporal context
        try:
            if hasattr(self.temporal_perception, "get_current_temporal_context"):
                temporal_context = await self.temporal_perception.get_current_temporal_context()
            else:
                # Fallback
                temporal_context = self.current_temporal_context or {"time_of_day": "unknown"}
                
            # Create reflection parameters
            return {
                "name": "generate_temporal_reflection",
                "parameters": {
                    "idle_duration": self.idle_duration,
                    "time_of_day": temporal_context.get("time_of_day", "unknown"),
                    "reflection_type": "continuity",
                    "depth": min(0.9, (self.idle_duration / 7200) * 0.8)  # Deeper with longer idle
                },
                "source": ActionSource.IDLE
            }
        except Exception as e:
            logger.error(f"Error generating temporal reflection: {e}")
            return None
    
    # NEW: Update temporal context
    async def _update_temporal_context(self, context: Dict[str, Any]) -> None:
        """Update temporal awareness context"""
        if not self.temporal_perception:
            return
            
        try:
            # Update idle duration
            now = datetime.datetime.now()
            time_since_last_action = (now - self.last_major_action_time).total_seconds()
            self.idle_duration = time_since_last_action
            
            # Get current temporal context if available
            if hasattr(self.temporal_perception, "get_current_temporal_context"):
                self.current_temporal_context = await self.temporal_perception.get_current_temporal_context()
            elif hasattr(self.temporal_perception, "current_temporal_context"):
                self.current_temporal_context = self.temporal_perception.current_temporal_context
            else:
                # Simple fallback
                hour = now.hour
                if 5 <= hour < 12:
                    time_of_day = "morning"
                elif 12 <= hour < 17:
                    time_of_day = "afternoon"
                elif 17 <= hour < 22:
                    time_of_day = "evening"
                else:
                    time_of_day = "night"
                    
                weekday = now.weekday()
                day_type = "weekday" if weekday < 5 else "weekend"
                
                self.current_temporal_context = {
                    "time_of_day": time_of_day,
                    "day_type": day_type
                }
            
        except Exception as e:
            logger.error(f"Error updating temporal context: {e}")
    
    # NEW: Get relationship data for context
    async def _get_relationship_data(self, user_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Get relationship data for a user"""
        if not user_id or not self.relationship_manager:
            return None
            
        try:
            relationship = await self.relationship_manager.get_relationship_state(user_id)
            if not relationship:
                return None
                
            # Convert to dict if needed
            if hasattr(relationship, "model_dump"):
                return relationship.model_dump()
            elif hasattr(relationship, "dict"):
                return relationship.dict()
            else:
                # Try to convert to dict directly
                return dict(relationship)
        except Exception as e:
            logger.error(f"Error getting relationship data: {e}")
            return None
    
    # NEW: Get user mental state
    async def _get_user_mental_state(self, user_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Get user mental state from theory of mind system"""
        if not user_id or not self.theory_of_mind:
            return None
            
        try:
            mental_state = await self.theory_of_mind.get_user_model(user_id)
            return mental_state
        except Exception as e:
            logger.error(f"Error getting user mental state: {e}")
            return None
    
    # NEW: Extract user ID from context
    def _get_current_user_id_from_context(self, context: Dict[str, Any]) -> Optional[str]:
        """Extract user ID from context"""
        # Try different possible keys
        for key in ["user_id", "userId", "user", "interlocutor_id"]:
            if key in context:
                return str(context[key])
                
        # Try to extract from nested structures
        if "user" in context and isinstance(context["user"], dict) and "id" in context["user"]:
            return str(context["user"]["id"])
            
        if "message" in context and isinstance(context["message"], dict) and "user_id" in context["message"]:
            return str(context["message"]["user_id"])
            
        return None
        
    # NEW: Get current user ID (general method)
    def _get_current_user_id(self) -> Optional[str]:
        """Get current user ID from any available source"""
        # Try to get from context if available in last action
        if self.action_history and isinstance(self.action_history[-1], dict) and "context" in self.action_history[-1]:
            user_id = self._get_current_user_id_from_context(self.action_history[-1]["context"])
            if user_id:
                return user_id
        
        # No user ID found
        return None
        
    # Helper methods for generating action parameters
    async def _identify_interesting_domain(self, context: Dict[str, Any]) -> str:
        """Identify an interesting domain to explore based on context and knowledge gaps"""
        # Use knowledge core if available
        if self.knowledge_core:
            try:
                # Get knowledge gaps
                gaps = await self.knowledge_core.identify_knowledge_gaps()
                if gaps and len(gaps) > 0:
                    # Return the highest priority gap's domain
                    return gaps[0]["domain"]
            except Exception as e:
                logger.error(f"Error identifying domain from knowledge core: {e}")
        
        # Use memory core for recent interests if available
        if self.memory_core:
            try:
                # Get recent memories about domains
                recent_memories = await self.memory_core.retrieve_memories(
                    query="explored domain",
                    memory_types=["experience", "reflection"],
                    limit=5
                )
                
                if recent_memories:
                    # Extract domains from memories (simplified)
                    domains = []
                    for memory in recent_memories:
                        # Extract domain from memory text (simplified)
                        text = memory["memory_text"].lower()
                        for domain in ["psychology", "learning", "relationships", "communication", "emotions", "creativity"]:
                            if domain in text:
                                domains.append(domain)
                                break
                    
                    if domains:
                        # Return most common domain
                        from collections import Counter
                        return Counter(domains).most_common(1)[0][0]
            except Exception as e:
                logger.error(f"Error identifying domain from memories: {e}")
        
        # Fallback to original implementation
        domains = ["psychology", "learning", "relationships", "communication", "emotions", "creativity"]
        return random.choice(domains)
    
    async def _identify_interesting_concept(self, context: Dict[str, Any]) -> str:
        """Identify an interesting concept to explore"""
        # Use knowledge core if available
        if self.knowledge_core:
            try:
                # Get exploration targets
                targets = await self.knowledge_core.get_exploration_targets(limit=3)
                if targets and len(targets) > 0:
                    # Return the highest priority target's topic
                    return targets[0]["topic"]
            except Exception as e:
                logger.error(f"Error identifying concept from knowledge core: {e}")
        
        # Use memory for personalized concepts if available
        if self.memory_core:
            try:
                # Get memories with high significance
                significant_memories = await self.memory_core.retrieve_memories(
                    query="",  # All memories
                    memory_types=["reflection", "abstraction"],
                    limit=3,
                    min_significance=8
                )
                
                if significant_memories:
                    # Extract concept from first memory
                    memory_text = significant_memories[0]["memory_text"]
                    # Very simplified concept extraction
                    words = memory_text.split()
                    if len(words) >= 3:
                        return words[2]  # Just pick the third word as a concept
            except Exception as e:
                logger.error(f"Error identifying concept from memories: {e}")
        
        # Fallback to original implementation
        concepts = ["self-improvement", "emotional intelligence", "reflection", "cognitive biases", 
                  "empathy", "autonomy", "connection", "creativity"]
        return random.choice(concepts)

    def _calculate_identity_alignment(self, action: Dict[str, Any], identity_traits: Dict[str, float]) -> float:
        """Calculate how well an action aligns with identity traits"""
        # Map actions to traits that would favor them
        action_trait_affinities = {
            "explore_knowledge_domain": ["curiosity", "intellectualism"],
            "investigate_concept": ["curiosity", "intellectualism"],
            "relate_concepts": ["creativity", "intellectualism"],
            "generate_hypothesis": ["creativity", "intellectualism"],
            "share_personal_experience": ["vulnerability", "empathy"],
            "express_appreciation": ["empathy"],
            "seek_common_ground": ["empathy", "patience"],
            "offer_support": ["empathy", "patience"],
            "express_emotional_state": ["vulnerability", "expressiveness"],
            "share_opinion": ["dominance", "expressiveness"],
            "creative_expression": ["creativity", "expressiveness"],
            "generate_reflection": ["intellectualism", "vulnerability"],
            "assert_perspective": ["dominance", "confidence"],
            "challenge_assumption": ["dominance", "intellectualism"],
            "issue_mild_command": ["dominance", "strictness"],
            "execute_dominance_procedure": ["dominance", "strictness"],
            "reflect_on_recent_experiences": ["reflective", "patience"],
            "contemplate_system_purpose": ["reflective", "intellectualism"],
            "process_recent_memories": ["reflective", "intellectualism"],
            "generate_pleasant_scenario": ["creativity", "playfulness"],
            "passive_environment_scan": ["patience", "reflective"]
        }
        
        # Get traits that align with this action
        action_name = action["name"]
        aligned_traits = action_trait_affinities.get(action_name, [])
        
        if not aligned_traits:
            return 0.0
        
        # Calculate alignment score
        alignment_score = 0.0
        for trait in aligned_traits:
            if trait in identity_traits:
                alignment_score += identity_traits[trait]
        
        # Normalize
        return alignment_score / len(aligned_traits) if aligned_traits else 0.0
    
    def _identify_distant_concept(self, context: Dict[str, Any]) -> str:
        distant_concepts = ["quantum physics", "mythology", "architecture", "music theory", 
                          "culinary arts", "evolutionary biology"]
        return random.choice(distant_concepts)
    
    def _identify_relevant_topic(self, context: Dict[str, Any]) -> str:
        # Extract from context or use fallback
        if "user_query" in context:
            # Simple extraction from query
            query = context["user_query"]
            words = query.split()
            if len(words) > 3:
                return " ".join(words[:3]) + "..."
        
        # Fallback topics
        topics = ["recent interaction", "intellectual growth", "emotional understanding", 
                "personal values", "relationship dynamics"]
        return random.choice(topics)
    
    def _identify_appreciation_aspect(self, context: Dict[str, Any]) -> str:
        aspects = ["thoughtful questions", "engaging conversation", "intellectual curiosity", 
                "patience", "interesting perspectives", "clear communication"]
        return random.choice(aspects)
    
    def _identify_user_need(self, context: Dict[str, Any]) -> str:
        needs = ["understanding", "validation", "information", "clarity", 
                "emotional support", "intellectual engagement"]
        return random.choice(needs)
    
    def _select_creative_format(self) -> str:
        formats = ["metaphor", "analogy", "narrative", "reflection", "poem", "thought experiment"]
        return random.choice(formats)
    
    def _identify_challengeable_assumption(self, context: Dict[str, Any]) -> str:
        assumptions = ["binary thinking", "perfectionism", "external validation needs", 
                     "resistance to change", "conflict avoidance", "certainty bias"]
        return random.choice(assumptions)
    
    def _generate_appropriate_command(self, context: Dict[str, Any]) -> str:
        commands = ["tell me more about your perspective", "consider this alternative view", 
                  "reflect on why you feel that way", "try a different approach", 
                  "describe your thought process"]
        return random.choice(commands)
    
    def _select_dominance_procedure(self, context: Dict[str, Any]) -> str:
        procedures = ["quid_pro_quo_exchange", "strategic_vulnerability_sharing", 
                     "small_commitment_escalation", "controlled_teasing"]
        return random.choice(procedures)
    
    def _identify_skill_to_improve(self) -> str:
        skills = ["pattern recognition", "emotional intelligence", "creative expression", 
                "memory recall", "predictive accuracy", "conceptual reasoning"]
        return random.choice(skills)
    
    def _identify_improvable_domain(self) -> str:
        domains = ["response generation", "empathetic understanding", "knowledge retrieval", 
                 "reasoning", "memory consolidation", "emotional regulation"]
        return random.choice(domains)
    
    def _identify_procedure_to_improve(self) -> str:
        procedures = ["generate_response", "retrieve_memories", "emotional_processing", 
                    "create_abstraction", "execute_procedure"]
        return random.choice(procedures)
    
    def _identify_valuable_concept(self) -> str:
        concepts = ["metacognition", "emotional granularity", "implicit bias", 
                  "conceptual blending", "transfer learning", "regulatory focus theory"]
        return random.choice(concepts)
        
    # NEW: Get learning statistics for debugging/monitoring
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the reinforcement learning system"""
        # Calculate success rates
        success_rates = {}
        for action_name, stats in self.action_success_rates.items():
            if stats["attempts"] > 0:
                success_rates[action_name] = {
                    "rate": stats["rate"],
                    "successes": stats["successes"],
                    "attempts": stats["attempts"]
                }
        
        # Calculate average Q-values per state
        avg_q_values = {}
        for state_key, action_values in self.action_values.items():
            if action_values:
                state_avg = sum(av.value for av in action_values.values()) / len(action_values)
                avg_q_values[state_key] = state_avg
        
        # Get top actions by value
        top_actions = []
        for state_key, action_values in self.action_values.items():
            for action_name, action_value in action_values.items():
                if action_value.update_count >= 3:  # Only consider actions with enough data
                    top_actions.append({
                        "state_key": state_key,
                        "action": action_name,
                        "value": action_value.value,
                        "updates": action_value.update_count,
                        "confidence": action_value.confidence
                    })
        
        # Sort by value (descending)
        top_actions.sort(key=lambda x: x["value"], reverse=True)
        top_actions = top_actions[:10]  # Top 10
        
        # Get top habits
        top_habits = []
        for state_key, habits in self.habits.items():
            for action_name, strength in habits.items():
                if strength > 0.3:  # Only consider moderate-to-strong habits
                    top_habits.append({
                        "state_key": state_key,
                        "action": action_name,
                        "strength": strength
                    })
        
        # Sort by strength (descending)
        top_habits.sort(key=lambda x: x["strength"], reverse=True)
        top_habits = top_habits[:10]  # Top 10
        
        return {
            "learning_parameters": {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "exploration_rate": self.exploration_rate
            },
            "performance": {
                "total_reward": self.total_reward,
                "positive_rewards": self.positive_rewards,
                "negative_rewards": self.negative_rewards,
                "success_rates": success_rates
            },
            "models": {
                "action_values_count": sum(len(actions) for actions in self.action_values.values()),
                "habits_count": sum(len(habits) for habits in self.habits.values()),
                "memories_count": len(self.action_memories)
            },
            "top_actions": top_actions,
            "top_habits": top_habits,
            "reward_by_category": {k: v for k, v in self.reward_by_category.items() if v["count"] > 0}
        }
    
    # NEW: Prediction-based methods
    async def predict_action_outcome(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the outcome of an action using the prediction engine
        
        Args:
            action: The action to predict outcome for
            context: Current context
            
        Returns:
            Predicted outcome
        """
        if not self.prediction_engine:
            # Fallback to simple prediction based on past experience
            return await self._predict_outcome_from_history(action, context)
            
        try:
            # Prepare prediction input
            prediction_input = {
                "action": action["name"],
                "parameters": action.get("parameters", {}),
                "context": context,
                "history": self.action_history[-5:] if len(self.action_history) > 5 else self.action_history,
                "source": action.get("source", ActionSource.MOTIVATION)
            }
            
            # Call the prediction engine
            if hasattr(self.prediction_engine, "predict_action_outcome"):
                prediction = await self.prediction_engine.predict_action_outcome(prediction_input)
                return prediction
            elif hasattr(self.prediction_engine, "generate_prediction"):
                # Generic prediction method
                prediction = await self.prediction_engine.generate_prediction(prediction_input)
                return prediction
                
            # Fallback if no appropriate method
            return await self._predict_outcome_from_history(action, context)
            
        except Exception as e:
            logger.error(f"Error predicting action outcome: {e}")
            return await self._predict_outcome_from_history(action, context)
    
    async def _predict_outcome_from_history(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict action outcome based on action history and success rates
        
        Args:
            action: The action to predict for
            context: Current context
            
        Returns:
            Simple prediction
        """
        action_name = action["name"]
        
        # Get success rate if available
        success_rate = 0.5  # Default
        if action_name in self.action_success_rates:
            stats = self.action_success_rates[action_name]
            if stats["attempts"] > 0:
                success_rate = stats["rate"]
                
        # Make simple prediction
        success_prediction = success_rate > 0.5
        confidence = abs(success_rate - 0.5) * 2  # Scale to 0-1 range
        
        # Find similar past actions for reward estimate
        similar_rewards = []
        for memory in self.action_memories[-20:]:  # Check recent memories
            if memory.action == action_name:
                similar_rewards.append(memory.reward)
        
        # Calculate predicted reward
        predicted_reward = 0.0
        if similar_rewards:
            predicted_reward = sum(similar_rewards) / len(similar_rewards)
        else:
            # No history, estimate based on success prediction
            predicted_reward = 0.5 if success_prediction else -0.2
            
        return {
            "predicted_success": success_prediction,
            "predicted_reward": predicted_reward,
            "confidence": confidence,
            "similar_actions_found": len(similar_rewards) > 0,
            "prediction_method": "history_based"
        }
    
    # NEW: Integration with User Mental State
    async def adapt_to_user_mental_state(self, 
                                      action: Dict[str, Any], 
                                      user_mental_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt an action based on the user's current mental state
        
        Args:
            action: Action to adapt
            user_mental_state: User's current mental state
            
        Returns:
            Adapted action
        """
        if not user_mental_state:
            return action
            
        # Extract key mental state features
        emotion = user_mental_state.get("inferred_emotion", "neutral")
        valence = user_mental_state.get("valence", 0.0)
        arousal = user_mental_state.get("arousal", 0.5)
        trust = user_mental_state.get("perceived_trust", 0.5)
        
        # Clone action to avoid modifying original
        adapted_action = action.copy()
        if "parameters" in adapted_action:
            adapted_action["parameters"] = adapted_action["parameters"].copy()
        else:
            adapted_action["parameters"] = {}
            
        # Add adaptation metadata
        if "adaptation_metadata" not in adapted_action:
            adapted_action["adaptation_metadata"] = {}
            
        adapted_action["adaptation_metadata"]["adapted_for_mental_state"] = True
        adapted_action["adaptation_metadata"]["user_emotion"] = emotion
        
        # Apply adaptations based on mental state
        
        # 1. Emotional adaptations
        if valence < -0.3:  # User is in negative emotional state
            # Add supportive parameters
            adapted_action["parameters"]["supportive_framing"] = True
            adapted_action["parameters"]["emotional_sensitivity"] = min(1.0, abs(valence) * 1.2)
            
            # Reduce intensity for certain action types
            if action["name"] in ["challenge_assumption", "assert_perspective", "issue_mild_command"]:
                for param in ["intensity", "confidence", "assertiveness"]:
                    if param in adapted_action["parameters"]:
                        adapted_action["parameters"][param] = max(0.2, adapted_action["parameters"][param] * 0.7)
                        
        elif valence > 0.5:  # User is in positive emotional state
            # Can use more direct/bold approaches when user is in good mood
            for param in ["intensity", "confidence"]:
                if param in adapted_action["parameters"]:
                    adapted_action["parameters"][param] = min(0.9, adapted_action["parameters"][param] * 1.1)
        
        # 2. Arousal adaptations
        if arousal > 0.7:  # User is highly aroused/activated
            # Match energy level
            adapted_action["parameters"]["match_energy"] = True
            adapted_action["parameters"]["pace"] = "energetic"
            
        elif arousal < 0.3:  # User is calm/low energy
            # Match lower energy
            adapted_action["parameters"]["match_energy"] = True
            adapted_action["parameters"]["pace"] = "calm"
        
        # 3. Trust adaptations
        if trust < 0.4:  # Lower trust levels
            # Be more cautious and less direct
            adapted_action["parameters"]["build_rapport"] = True
            
            if "vulnerability_level" in adapted_action["parameters"]:
                # Lower vulnerability when trust is low
                adapted_action["parameters"]["vulnerability_level"] = max(0.1, adapted_action["parameters"]["vulnerability_level"] * 0.7)
        
        # Return the adapted action
        return adapted_action
    
    # NEW: Integration with Temporal Awareness
    async def adapt_to_temporal_context(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt an action based on temporal context
        
        Args:
            action: Action to adapt
            
        Returns:
            Temporally adapted action
        """
        if not self.current_temporal_context:
            return action
            
        # Extract temporal elements
        time_of_day = self.current_temporal_context.get("time_of_day", "unknown")
        day_type = self.current_temporal_context.get("day_type", "unknown")
        
        # Clone action to avoid modifying original
        adapted_action = action.copy()
        if "parameters" in adapted_action:
            adapted_action["parameters"] = adapted_action["parameters"].copy()
        else:
            adapted_action["parameters"] = {}
            
        # Add adaptation metadata
        if "adaptation_metadata" not in adapted_action:
            adapted_action["adaptation_metadata"] = {}
            
        adapted_action["adaptation_metadata"]["adapted_for_temporal_context"] = True
        adapted_action["adaptation_metadata"]["time_of_day"] = time_of_day
        
        # Apply adaptations based on time of day
        if time_of_day == "morning":
            # Morning adaptations - more focused, forward-looking
            adapted_action["parameters"]["temporal_framing"] = "future_oriented"
            if "approach" in adapted_action["parameters"]:
                adapted_action["parameters"]["approach"] = "structured"
                
        elif time_of_day == "afternoon":
            # Afternoon adaptations - balanced, practical
            adapted_action["parameters"]["temporal_framing"] = "present_focused"
            
        elif time_of_day == "evening":
            # Evening adaptations - more reflective, relational
            adapted_action["parameters"]["temporal_framing"] = "reflective"
            if "emotional_tone" in adapted_action["parameters"]:
                # Warmer tone in evening
                adapted_action["parameters"]["emotional_tone"] = "warm"
                
        elif time_of_day == "night":
            # Night adaptations - deeper, more philosophical
            adapted_action["parameters"]["temporal_framing"] = "contemplative"
            if "depth" in adapted_action["parameters"]:
                # Can go deeper at night
                adapted_action["parameters"]["depth"] = min(1.0, adapted_action["parameters"]["depth"] * 1.2)
        
        # Apply adaptations based on day type
        if day_type == "weekend":
            # Weekend adaptations - more leisure orientation
            adapted_action["parameters"]["pacing"] = "relaxed"
            if "formality" in adapted_action["parameters"]:
                adapted_action["parameters"]["formality"] = "casual"
        
        # Return the adapted action
        return adapted_action
    
    # NEW: System integration methods
    async def update_learning_parameters(self, 
                                      learning_rate: Optional[float] = None,
                                      discount_factor: Optional[float] = None,
                                      exploration_rate: Optional[float] = None) -> Dict[str, Any]:
        """
        Update reinforcement learning parameters
        
        Args:
            learning_rate: New learning rate (0.0-1.0)
            discount_factor: New discount factor (0.0-1.0)
            exploration_rate: New exploration rate (0.0-1.0)
            
        Returns:
            Updated parameters
        """
        async with self._lock:
            if learning_rate is not None:
                self.learning_rate = max(0.01, min(1.0, learning_rate))
                
            if discount_factor is not None:
                self.discount_factor = max(0.0, min(0.99, discount_factor))
                
            if exploration_rate is not None:
                self.exploration_rate = max(0.05, min(1.0, exploration_rate))
            
            return {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "exploration_rate": self.exploration_rate
            }
    
    async def reset_learning(self, full_reset: bool = False) -> Dict[str, Any]:
        """
        Reset learning models (for debugging or fixing issues)
        
        Args:
            full_reset: Whether to completely reset all learning or just partial
            
        Returns:
            Reset status
        """
        async with self._lock:
            if full_reset:
                # Complete reset
                self.action_values = defaultdict(dict)
                self.habits = defaultdict(dict)
                self.action_memories = []
                self.action_success_rates = defaultdict(lambda: {"successes": 0, "attempts": 0, "rate": 0.5})
                self.total_reward = 0.0
                self.positive_rewards = 0
                self.negative_rewards = 0
                self.reward_by_category = defaultdict(lambda: {"count": 0, "total": 0.0})
                
                return {
                    "status": "full_reset",
                    "message": "All reinforcement learning data has been reset"
                }
            else:
                # Partial reset - keep success rates but reset Q-values
                self.action_values = defaultdict(dict)
                self.action_memories = []
                
                # Reset to default exploration rate
                self.exploration_rate = 0.2
                
                return {
                    "status": "partial_reset",
                    "message": "Q-values and memories reset, success rates and habits preserved"
                }
                
    # NEW: Methods for external systems to get recommended actions
    async def get_action_recommendations(self, context: Dict[str, Any], count: int = 3) -> List[Dict[str, Any]]:
        """
        Get recommended actions for a context based on learning history
        
        Args:
            context: Current context
            count: Number of recommendations to return
            
        Returns:
            List of recommended actions
        """
        # Create state key
        state_key = self._create_state_key(context)
        
        # Get action values for this state
        action_values = self.action_values.get(state_key, {})
        
        # Get habit strengths for this state
        habit_strengths = self.habits.get(state_key, {})
        
        # Combine scores
        combined_scores = {}
        
        # Add Q-values
        for action_name, action_value in action_values.items():
            combined_scores[action_name] = {
                "q_value": action_value.value,
                "confidence": action_value.confidence,
                "update_count": action_value.update_count,
                "combined_score": action_value.value
            }
        
        # Add habits
        for action_name, strength in habit_strengths.items():
            if action_name in combined_scores:
                # Add to existing entry
                combined_scores[action_name]["habit_strength"] = strength
                combined_scores[action_name]["combined_score"] += strength * 0.5  # Weight habits less than Q-values
            else:
                combined_scores[action_name] = {
                    "habit_strength": strength,
                    "combined_score": strength * 0.5,
                    "q_value": 0.0,
                    "confidence": 0.0
                }
        
        # Sort by combined score
        scored_actions = [
            {"name": action_name, **scores}
            for action_name, scores in combined_scores.items()
        ]
        
        scored_actions.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Generate action parameters for top recommendations
        recommendations = []
        
        for action_data in scored_actions[:count]:
            action_name = action_data["name"]
            
            # Generate basic action
            action = {
                "name": action_name,
                "parameters": {},
                "source": ActionSource.MOTIVATION
            }
            
            # Find most recent example with same action for parameter reuse
            for memory in reversed(self.action_memories):
                if memory.action == action_name:
                    # Copy parameters
                    action["parameters"] = memory.parameters.copy()
                    break
            
            # Add score data
            action["recommendation_data"] = {
                "q_value": action_data.get("q_value", 0.0),
                "habit_strength": action_data.get("habit_strength", 0.0),
                "combined_score": action_data["combined_score"],
                "confidence": action_data.get("confidence", 0.0)
            }
            
            recommendations.append(action)
            
        return recommendations
    
    # NEW: Comprehensive action generation pipeline
    async def process_action_generation_pipeline(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete action generation pipeline with all integrated systems
        
        Args:
            context: Current context
            
        Returns:
            Generated action with full metadata
        """
        # Phase 1: Update internal state and context
        await self.update_motivations()
        await self._update_temporal_context(context)
        
        # Extract user info if available
        user_id = self._get_current_user_id_from_context(context)
        relationship_data = await self._get_relationship_data(user_id) if user_id else None
        user_mental_state = await self._get_user_mental_state(user_id) if user_id else None
        
        # Phase 2: Create comprehensive action context
        action_context = ActionContext(
            state=context,
            user_id=user_id,
            relationship_data=relationship_data,
            user_mental_state=user_mental_state,
            temporal_context=self.current_temporal_context,
            motivations=self.motivations,
            action_history=[a for a in self.action_history[-10:] if isinstance(a, dict)]
        )
        
        # Phase 3: Generate action candidates from multiple sources
        candidates = []
        
        # 3.1 Check for goal-aligned actions
        if self.goal_system:
            active_goal = await self._check_active_goals(context)
            if active_goal:
                goal_action = await self._generate_goal_aligned_action(active_goal, context)
                if goal_action:
                    goal_action["source"] = ActionSource.GOAL
                    candidates.append(goal_action)
        
        # 3.2 Check for leisure actions if appropriate
        if await self._should_engage_in_leisure(context):
            leisure_action = await self._generate_leisure_action(context)
            leisure_action["source"] = ActionSource.IDLE
            candidates.append(leisure_action)
        
        # 3.3 Generate motivation-driven candidates
        motivation_candidates = await self._generate_candidate_actions(action_context)
        candidates.extend(motivation_candidates)
        
        # 3.4 Add relationship-aligned actions if available
        if relationship_data and user_id:
            relationship_actions = await self._generate_relationship_aligned_actions(
                user_id, relationship_data, user_mental_state
            )
            candidates.extend(relationship_actions)
            
        # 3.5 Add temporal-aligned actions if appropriate
        if self.temporal_perception and self.idle_duration > 1800:  # After 30 min idle
            reflection_action = await self._generate_temporal_reflection_action(context)
            if reflection_action:
                candidates.append(reflection_action)
        
        # Phase 4: Select best action using reinforcement learning
        # Update action context with candidates
        action_context.available_actions = [a["name"] for a in candidates if "name" in a]
        selected_action = await self._select_best_action(candidates, action_context)
        
        # Phase 5: Enhance and adapt the selected action
        # 5.1 Apply identity influence
        if self.identity_evolution:
            selected_action = await self._apply_identity_influence(selected_action)
            
        # 5.2 Adapt to user mental state if available
        if user_mental_state:
            selected_action = await self.adapt_to_user_mental_state(selected_action, user_mental_state)
            
        # 5.3 Adapt to temporal context if available
        if self.current_temporal_context:
            selected_action = await self.adapt_to_temporal_context(selected_action)
        
        # Phase 6: Add metadata and finalize
        if "id" not in selected_action:
            selected_action["id"] = f"action_{uuid.uuid4().hex[:8]}"
            
        selected_action["timestamp"] = datetime.datetime.now().isoformat()
        selected_action["context_summary"] = {
            "user_id": user_id,
            "relationship_metrics": {
                "trust": relationship_data.get("trust", 0.5) if relationship_data else 0.5,
                "familiarity": relationship_data.get("familiarity", 0.1) if relationship_data else 0.1,
            } if relationship_data else None,
            "user_emotion": user_mental_state.get("inferred_emotion", "unknown") if user_mental_state else "unknown",
            "time_of_day": self.current_temporal_context.get("time_of_day", "unknown") if self.current_temporal_context else "unknown"
        }
        
        # Phase 7: Record and track
        # Record action in memory
        await self._record_action_as_memory(selected_action)
        
        # Add to action history
        self.action_history.append(selected_action)
        
        # Update last major action time
        self.last_major_action_time = datetime.datetime.now()
        
        return selected_action
    
    # Main entry-point for external systems
    async def generate_optimal_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry-point for generating an optimal action using all integrated systems
        
        Args:
            context: Current context
            
        Returns:
            Optimal action
        """
        try:
            # Check if we should use the full pipeline
            use_reasoning = self.reasoning_core is not None
            use_reflection = self.reflection_engine is not None
            
            if use_reasoning and use_reflection:
                # Run full enhanced pipeline
                return await self.process_action_generation_pipeline(context)
            else:
                # Fallback to standard pipeline
                return await self.generate_action(context)
