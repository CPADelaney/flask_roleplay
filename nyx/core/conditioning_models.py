# nyx/core/conditioning_models.py

import datetime
import logging
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
import os

logger = logging.getLogger(__name__)

# ==================== Data Models ====================

class ConditionedAssociation(BaseModel):
    """Represents a conditioned association between stimuli and responses"""
    stimulus: str = Field(..., description="The triggering stimulus")
    response: str = Field(..., description="The conditioned response")
    association_strength: float = Field(0.0, description="Strength of the association (0.0-1.0)")
    formation_date: str = Field(..., description="When this association was formed")
    last_reinforced: str = Field(..., description="When this association was last reinforced")
    reinforcement_count: int = Field(0, description="Number of times this association has been reinforced")
    valence: float = Field(0.0, description="Emotional valence of this association (-1.0 to 1.0)")
    context_keys: List[str] = Field(default_factory=list, description="Contextual keys where this association applies")
    decay_rate: float = Field(0.05, description="Rate at which this association decays if not reinforced")

# ==================== Configuration Models ====================

class ConditioningParameters(BaseModel):
    """Configuration parameters for the conditioning system"""
    association_learning_rate: float = Field(0.1, description="How quickly new associations form")
    extinction_rate: float = Field(0.05, description="How quickly associations weaken without reinforcement")
    generalization_factor: float = Field(0.3, description="How much conditioning generalizes to similar stimuli")
    weak_association_threshold: float = Field(0.3, description="Threshold for weak associations")
    moderate_association_threshold: float = Field(0.6, description="Threshold for moderate associations")
    strong_association_threshold: float = Field(0.8, description="Threshold for strong associations")
    maintenance_interval_hours: int = Field(24, description="Hours between maintenance runs")
    consolidation_interval_days: int = Field(7, description="Days between consolidation runs")
    extinction_threshold: float = Field(0.05, description="Threshold for removing weak associations")
    reinforcement_threshold: float = Field(0.3, description="Threshold for reinforcing core traits")
    max_trait_imbalance: float = Field(0.3, description="Maximum allowed trait imbalance")
    correction_strength: float = Field(0.3, description="Strength of balance corrections")
    reward_scaling_factor: float = Field(0.5, description="How strongly rewards affect conditioning")
    negative_punishment_factor: float = Field(0.8, description="Scaling factor for negative punishments")
    pattern_match_confidence: float = Field(0.7, description="Confidence threshold for pattern matching")
    response_modification_strength: float = Field(0.5, description="How strongly conditioning affects responses")

class PreferenceDetail(BaseModel):
    type: str = Field(..., description="Type of preference (e.g., like, dislike, want, avoid)")
    value: float = Field(..., description="Strength/direction of the preference")

class EmotionTriggerDetail(BaseModel):
    emotion: str = Field(..., description="The emotion to be triggered")
    intensity: float = Field(0.5, description="The intensity of the triggered emotion (0.0-1.0)")
    valence: Optional[float] = Field(None, description="Optional valence override for the emotion trigger")

class PersonalityProfile(BaseModel):
    """Personality profile configuration"""
    traits: Dict[str, float] = Field(default_factory=dict, description="Personality traits and their strengths (0.0-1.0)")
    preferences: Dict[str, PreferenceDetail] = Field(default_factory=dict, description="Preferences for various stimuli")
    emotion_triggers: Dict[str, EmotionTriggerDetail] = Field(default_factory=dict, description="Emotion triggers")
    behaviors: Dict[str, List[str]] = Field(default_factory=dict, description="Behaviors and associated traits")

# ==================== Explicit Models for Strict Schema Compliance ====================

# Models for BehaviorEvaluationOutput
class RelevantAssociation(BaseModel):
    """Model for relevant associations in behavior evaluation"""
    key: str = Field(..., description="Association key")
    stimulus: str = Field(..., description="Stimulus")
    response: str = Field(..., description="Response")
    strength: float = Field(..., description="Association strength")
    valence: float = Field(..., description="Valence")
    context_match: bool = Field(False, description="Whether context matches")

# Models for MaintenanceSummaryOutput
class MaintenanceTask(BaseModel):
    """Model for maintenance tasks performed"""
    task_type: str = Field(..., description="Type of maintenance task")
    entity_type: str = Field(..., description="Type of entity maintained")
    entity_id: str = Field(..., description="ID of entity")
    action_taken: str = Field(..., description="Action that was taken")
    result: str = Field(..., description="Result of the action")
    timestamp: str = Field(..., description="When the task was performed")

# Models for BalanceAnalysisOutput
class TraitImbalanceInfo(BaseModel):
    """Model for trait imbalance information"""
    trait: Optional[str] = Field(None, description="Single trait name if applicable")
    traits: Optional[List[str]] = Field(None, description="Multiple traits if applicable")
    imbalance_type: str = Field(..., description="Type of imbalance")
    severity: float = Field(..., description="Severity of imbalance (0.0-1.0)")
    current_value: Optional[float] = Field(None, description="Current trait value")
    recommended_value: Optional[float] = Field(None, description="Recommended trait value")
    difference: Optional[float] = Field(None, description="Difference between traits")

class TraitRecommendation(BaseModel):
    """Model for trait adjustment recommendations"""
    trait: str = Field(..., description="Trait to adjust")
    current_value: float = Field(..., description="Current value")
    recommended_adjustment: float = Field(..., description="Recommended adjustment")
    reason: str = Field(..., description="Reason for recommendation")
    priority: float = Field(..., description="Priority of adjustment (0.0-1.0)")

class BehaviorRecommendation(BaseModel):
    """Model for behavior conditioning recommendations"""
    behavior: str = Field(..., description="Behavior to condition")
    conditioning_type: str = Field(..., description="Type of conditioning to apply")
    intensity: float = Field(..., description="Recommended intensity")
    reason: str = Field(..., description="Reason for recommendation")

# Models for AssociationConsolidationOutput
class ConsolidationAction(BaseModel):
    """Model for consolidation actions taken"""
    action_type: str = Field(..., description="Type of consolidation action")
    source_keys: List[str] = Field(..., description="Source association keys")
    target_key: Optional[str] = Field(None, description="Target association key if merged")
    strength_before: float = Field(..., description="Strength before consolidation")
    strength_after: float = Field(..., description="Strength after consolidation")
    reason: str = Field(..., description="Reason for consolidation")

# ==================== Output Schema Models ====================

class ClassicalConditioningOutput(BaseModel):
    association_key: str = Field(..., description="Key for the association")
    type: str = Field(..., description="Type of association (new_association or reinforcement)")
    association_strength: float = Field(..., description="Strength of the association")
    reinforcement_count: int = Field(..., description="Number of reinforcements")
    valence: float = Field(..., description="Emotional valence of the association")
    explanation: str = Field(..., description="Explanation of the conditioning process")

class OperantConditioningOutput(BaseModel):
    association_key: str = Field(..., description="Key for the association")
    type: str = Field(..., description="Type of association (new_association or update)")
    behavior: str = Field(..., description="The behavior being conditioned")
    consequence_type: str = Field(..., description="Type of consequence")
    association_strength: float = Field(..., description="Strength of the association")
    is_reinforcement: bool = Field(..., description="Whether this is reinforcement or punishment")
    is_positive: bool = Field(..., description="Whether this is positive or negative")
    explanation: str = Field(..., description="Explanation of the conditioning process")

# ─── at the top of nyx/core/conditioning_models.py ───────────────────────────
from pydantic import BaseModel, Field, ConfigDict   # ← add ConfigDict import
# -----------------------------------------------------------------------------


# ─── replace the old BehaviorEvaluationOutput definition with this one ───────
class BehaviorEvaluationOutput(BaseModel):
    """
    Output of Behavior-Evaluation agent (Pydantic v2-ready).

    • Any extra keys the LLM decides to emit are allowed – they’ll simply be
      included in the model instance instead of raising a validation error.
    """

    # Pydantic v2 style config
    model_config = ConfigDict(extra='allow')

    behavior: str = Field(default="", description="The behavior being evaluated")
    expected_valence: float = Field(
        default=0.0,
        description="Expected outcome valence (-1.0 … 1.0)",
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence in the evaluation (0.0 … 1.0)",
    )
    recommendation: str = Field(
        default="neutral",
        description="Recommendation (approach | avoid | neutral)",
    )
    explanation: str = Field(
        default="",
        description="Explanation of the recommendation",
    )
    relevant_associations: Optional[List[RelevantAssociation]] = Field(
        default=None,
        description="Associations considered in the evaluation",
    )
# ─────────────────────────────────────────────────────────────────────────────


class TraitConditioningOutput(BaseModel):
    trait: str = Field(default="", description="The personality trait being conditioned")
    target_value: float = Field(default=0.0, description="Target trait value")
    actual_value: float = Field(default=0.0, description="Achieved trait value after conditioning")
    conditioned_behaviors: List[str] = Field(default_factory=list, description="Behaviors conditioned for this trait")
    identity_impact: str = Field(default="", description="Description of impact on identity")
    conditioning_strategy: str = Field(default="", description="Strategy used for conditioning")

class MaintenanceRecommendation(BaseModel):
    recommendation_type: str = Field(..., description="Type of recommendation")
    entity_type: str = Field(..., description="Type of entity (association, trait, etc.)")
    entity_id: str = Field(..., description="ID of entity")
    action: str = Field(..., description="Recommended action")
    reasoning: str = Field(..., description="Reasoning for recommendation")
    priority: float = Field(..., description="Priority level (0.0-1.0)")

class BalanceAnalysisOutput(BaseModel):
    is_balanced: bool = Field(default=False, description="Whether personality is balanced")
    imbalances: List[TraitImbalanceInfo] = Field(default_factory=list, description="Detected imbalances")
    trait_recommendations: List[TraitRecommendation] = Field(default_factory=list, description="Trait recommendations")
    behavior_recommendations: List[BehaviorRecommendation] = Field(default_factory=list, description="Behavior recommendations")
    balance_score: float = Field(default=0.0, description="Overall balance score (0.0-1.0)")
    analysis: str = Field(default="", description="Analysis of personality balance")

class AssociationConsolidationOutput(BaseModel):
    consolidations: List[ConsolidationAction] = Field(default_factory=list, description="Consolidations performed")
    removed_keys: List[str] = Field(default_factory=list, description="Association keys removed")
    strengthened_keys: List[str] = Field(default_factory=list, description="Association keys strengthened")
    efficiency_gain: float = Field(default=0.0, description="Efficiency gain from consolidation (0.0-1.0)")
    reasoning: str = Field(default="", description="Reasoning for consolidations")

class MaintenanceSummaryOutput(BaseModel):
    tasks_performed: List[MaintenanceTask] = Field(default_factory=list, description="Tasks performed during the maintenance run")
    time_taken_seconds: float = Field(default=0.0, description="Total time taken for the maintenance run in seconds")
    associations_modified: int = Field(default=0, description="Number of conditioning associations modified")
    traits_adjusted: int = Field(default=0, description="Number of personality traits adjusted")
    extinction_count: int = Field(default=0, description="Number of associations removed due to extinction")
    improvements: List[str] = Field(default_factory=list, description="List of key improvements made during the run")
    next_maintenance_recommendation: str = Field(default="", description="Recommendation for the next maintenance run")

# ==================== Context Classes ====================

class ConditioningContext:
    """Shared context for conditioning system"""
    def __init__(self, reward_system=None, emotional_core=None, memory_core=None, somatosensory_system=None):
        # External systems
        self.reward_system = reward_system
        self.emotional_core = emotional_core
        self.memory_core = memory_core
        self.somatosensory_system = somatosensory_system
        
        # Association stores
        self.classical_associations: Dict[str, ConditionedAssociation] = {}
        self.operant_associations: Dict[str, ConditionedAssociation] = {}
        
        # Parameters
        self.parameters = ConditioningParameters()
        
        # Counters
        self.total_associations = 0
        self.total_reinforcements = 0
        self.successful_associations = 0
        
        # Identity store (simple dict for now)
        self.identity_traits_store: Dict[str, float] = {}

class MaintenanceContext:
    """Context for maintenance operations"""
    def __init__(self, conditioning_system, reward_system=None):
        self.conditioning_system = conditioning_system
        self.reward_system = reward_system
        
        # Get parameters from conditioning system
        params = conditioning_system.context.parameters
        self.maintenance_interval_hours = params.maintenance_interval_hours
        self.extinction_threshold = params.extinction_threshold
        self.reinforcement_threshold = params.reinforcement_threshold
        self.consolidation_interval_days = params.consolidation_interval_days
        
        # Maintenance stats
        self.last_maintenance_time = None
        self.maintenance_history = []
        self.max_history_entries = 30
        self.maintenance_task = None
        self.trace_group_id = f"maintenance_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

class ConfigurationContext:
    """Context for configuration management"""
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.params_file = os.path.join(config_dir, "conditioning_params.json")
        self.personality_file = os.path.join(config_dir, "personality_profile.json")
        self.parameters: Optional[ConditioningParameters] = None
        self.personality_profile: Optional[PersonalityProfile] = None
        self.trace_group_id = f"conditioning_config_{os.path.basename(config_dir)}"
