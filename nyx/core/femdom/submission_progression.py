# nyx/femdom/submission_progression.py

import logging
import datetime
import uuid
import math
from typing import Dict, List, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field

from agents import Agent, Runner, handoff, function_tool, input_guardrail, GuardrailFunctionOutput, trace, InputGuardrail, OutputGuardrail
from agents import ModelSettings

logger = logging.getLogger(__name__)

# Input models for function tools (to replace Dict[str, Any])
class LimitsData(BaseModel):
    """Model for user limits data."""
    hard_limits: List[str] = Field(default_factory=list)
    soft_limits: List[str] = Field(default_factory=list)
    
class PreferencesData(BaseModel):
    """Model for user preferences data."""
    intensity: Optional[float] = None
    humiliation: Optional[float] = None
    service: Optional[float] = None
    discipline: Optional[float] = None
    psychological: Optional[float] = None

class MetricsData(BaseModel):
    """Model for initial metrics data."""
    obedience: Optional[float] = None
    consistency: Optional[float] = None
    initiative: Optional[float] = None
    depth: Optional[float] = None
    protocol_adherence: Optional[float] = None
    receptiveness: Optional[float] = None
    endurance: Optional[float] = None
    attentiveness: Optional[float] = None
    surrender: Optional[float] = None
    reverence: Optional[float] = None

class InitialUserData(BaseModel):
    """Model for initial user data when initializing a user."""
    limits: Optional[LimitsData] = None
    preferences: Optional[PreferencesData] = None
    metrics: Optional[MetricsData] = None
    level: Optional[int] = None
    path: Optional[str] = None

class ComplianceContextInfo(BaseModel):
    """Model for context information when recording compliance."""
    session_id: Optional[str] = None
    session_type: Optional[str] = None
    difficulty_factors: Optional[List[str]] = None
    environment: Optional[str] = None
    additional_notes: Optional[str] = None

# Explicit models to replace Dict[str, Any] usage
class PathRecommendationData(BaseModel):
    """Model for path recommendation data."""
    id: str
    name: str
    description: str
    match_score: float
    difficulty: float
    focus_areas: List[str]

class RequirementsData(BaseModel):
    """Model for milestone requirements data."""
    obedience: Optional[float] = None
    consistency: Optional[float] = None
    initiative: Optional[float] = None
    depth: Optional[float] = None
    protocol_adherence: Optional[float] = None
    receptiveness: Optional[float] = None
    endurance: Optional[float] = None
    attentiveness: Optional[float] = None
    surrender: Optional[float] = None
    reverence: Optional[float] = None

class MilestoneData(BaseModel):
    """Model for milestone data."""
    id: str
    name: str
    level: int
    description: Optional[str] = None
    completion_date: Optional[str] = None
    rewards: Optional[List[str]] = None
    unlocks: Optional[List[str]] = None
    requirements: Optional[RequirementsData] = None
    overall_progress: Optional[float] = None

class HistoryEntry(BaseModel):
    """Model for metric history entries."""
    timestamp: str
    old_value: float
    new_value: float
    final_value: float
    reason: str

class ComplianceContextData(BaseModel):
    """Model for compliance context information."""
    session_id: Optional[str] = None
    session_type: Optional[str] = None
    difficulty_factors: Optional[List[str]] = None
    environment: Optional[str] = None
    additional_notes: Optional[str] = None

class MilestoneDefinition(BaseModel):
    """Model for milestone definition data."""
    id: str
    name: str
    description: str
    requirements: RequirementsData
    rewards: List[str]
    unlocks: List[str]

class ProgressSummary(BaseModel):
    """Model for progress summary data."""
    total_milestones: int
    completed: int
    completion_percentage: float

class MetricUpdateData(BaseModel):
    """Model for metric update information."""
    old_value: float
    new_value: float
    change: float

class SubmissionScoreData(BaseModel):
    """Model for submission score information."""
    old: float
    new: float
    change: float

class LevelData(BaseModel):
    """Model for level information."""
    id: int
    name: str
    description: Optional[str] = None
    privileges: Optional[List[str]] = None
    restrictions: Optional[List[str]] = None
    training_focus: Optional[List[str]] = None
    appropriate: Optional[bool] = None
    time_at_level_days: Optional[int] = None
    progress_in_level: Optional[float] = None

class LevelChangeDetails(BaseModel):
    """Model for level change details."""
    change_type: Optional[str] = None
    old_level: Optional[LevelData] = None
    new_level: Optional[LevelData] = None

class RewardResult(BaseModel):
    """Model for reward result information."""
    value: float
    source: str
    processed: bool

class MetricData(BaseModel):
    """Model for individual metric information."""
    value: float
    weight: float
    last_updated: str

class RequirementProgress(BaseModel):
    """Model for requirement progress tracking."""
    current: float
    threshold: float
    met: bool
    progress_percentage: float

class ComplianceHistoryData(BaseModel):
    """Model for compliance history entries."""
    id: str
    timestamp: str
    instruction: str
    complied: bool
    difficulty: float
    defiance_reason: Optional[str] = None

class MetricTrend(BaseModel):
    """Model for metric trend information."""
    trend: str
    change: float
    recent_avg: float
    previous_avg: float

class MetricTrendsData(BaseModel):
    """Model for metric trends data."""
    obedience: Optional[MetricTrend] = None
    consistency: Optional[MetricTrend] = None
    initiative: Optional[MetricTrend] = None
    depth: Optional[MetricTrend] = None
    protocol_adherence: Optional[MetricTrend] = None
    receptiveness: Optional[MetricTrend] = None
    endurance: Optional[MetricTrend] = None
    attentiveness: Optional[MetricTrend] = None
    surrender: Optional[MetricTrend] = None
    reverence: Optional[MetricTrend] = None

class SubmissionMetricsData(BaseModel):
    """Model for submission metrics in reports."""
    overall_score: float
    compliance_rate: float
    compliance_trend: str
    metric_trends: MetricTrendsData

class AdvancementRequirement(BaseModel):
    """Model for advancement requirements."""
    type: str
    description: str
    current: float
    target: float
    trait: Optional[str] = None

class NextLevelData(BaseModel):
    """Model for next level information."""
    id: int
    name: str
    description: str

class ProgressionPathData(BaseModel):
    """Model for progression path information."""
    next_level: Optional[NextLevelData] = None
    requirements_for_advancement: List[AdvancementRequirement]
    estimated_time_to_next_level: Optional[int] = None

class TraitGapsData(BaseModel):
    """Model for trait gaps data."""
    obedience: Optional[MetricUpdateData] = None
    consistency: Optional[MetricUpdateData] = None
    initiative: Optional[MetricUpdateData] = None
    depth: Optional[MetricUpdateData] = None
    protocol_adherence: Optional[MetricUpdateData] = None
    receptiveness: Optional[MetricUpdateData] = None
    endurance: Optional[MetricUpdateData] = None
    attentiveness: Optional[MetricUpdateData] = None
    surrender: Optional[MetricUpdateData] = None
    reverence: Optional[MetricUpdateData] = None

class AllMetricsData(BaseModel):
    """Model for all metrics data."""
    obedience: Optional[MetricData] = None
    consistency: Optional[MetricData] = None
    initiative: Optional[MetricData] = None
    depth: Optional[MetricData] = None
    protocol_adherence: Optional[MetricData] = None
    receptiveness: Optional[MetricData] = None
    endurance: Optional[MetricData] = None
    attentiveness: Optional[MetricData] = None
    surrender: Optional[MetricData] = None
    reverence: Optional[MetricData] = None

class MetricsUpdatesData(BaseModel):
    """Model for metrics updates data."""
    obedience: Optional[MetricUpdateData] = None
    consistency: Optional[MetricUpdateData] = None
    initiative: Optional[MetricUpdateData] = None
    depth: Optional[MetricUpdateData] = None
    protocol_adherence: Optional[MetricUpdateData] = None
    receptiveness: Optional[MetricUpdateData] = None
    endurance: Optional[MetricUpdateData] = None
    attentiveness: Optional[MetricUpdateData] = None
    surrender: Optional[MetricUpdateData] = None
    reverence: Optional[MetricUpdateData] = None

class RecommendationData(BaseModel):
    """Model for recommendation information."""
    focus_area: str
    significance: float
    current_value: float
    target_value: float
    description: str

class SimpleMetricsData(BaseModel):
    """Model for simple metrics output."""
    obedience: Optional[float] = None
    consistency: Optional[float] = None
    initiative: Optional[float] = None
    depth: Optional[float] = None
    protocol_adherence: Optional[float] = None
    receptiveness: Optional[float] = None
    endurance: Optional[float] = None
    attentiveness: Optional[float] = None
    surrender: Optional[float] = None
    reverence: Optional[float] = None

# Tool output models for strict JSON schema compliance
class PathRecommendationResult(BaseModel):
    user_id: str
    primary_recommendation: Optional[PathRecommendationData] = None
    all_recommendations: List[PathRecommendationData]
    user_traits_analyzed: List[str]
    analysis_timestamp: str

class PathAssignmentResult(BaseModel):
    success: bool
    user_id: Optional[str] = None
    path_id: Optional[str] = None
    path_name: Optional[str] = None
    difficulty: Optional[float] = None
    focus_areas: Optional[List[str]] = None
    milestones: Optional[int] = None
    message: Optional[str] = None
    available_paths: Optional[List[str]] = None

class MilestoneProgressResult(BaseModel):
    success: bool
    user_id: Optional[str] = None
    path_id: Optional[str] = None
    path_name: Optional[str] = None
    newly_completed_milestones: List[MilestoneData]
    upcoming_milestones: List[MilestoneData]
    already_completed_milestones: List[MilestoneData]
    progress_summary: ProgressSummary
    message: Optional[str] = None
    recommendation: Optional[str] = None

class ComplianceRecordResult(BaseModel):
    success: bool
    record_id: Optional[str] = None
    compliance_recorded: Optional[bool] = None
    metrics_updated: MetricsUpdatesData
    submission_score: SubmissionScoreData
    level_changed: bool
    level_change_details: LevelChangeDetails
    reward_result: Optional[RewardResult] = None

class MetricUpdateResult(BaseModel):
    success: bool
    metric: Optional[str] = None
    old_value: Optional[float] = None
    new_value: Optional[float] = None
    change: Optional[float] = None
    submission_score: SubmissionScoreData
    level_changed: bool
    level_change_details: LevelChangeDetails
    message: Optional[str] = None

class UserSubmissionDataResult(BaseModel):
    user_id: str
    submission_level: LevelData
    submission_score: float
    metrics: AllMetricsData
    trait_gaps: TraitGapsData
    privileges: List[str]
    restrictions: List[str]
    training_focus: List[str]
    compliance_rate: float
    last_updated: str
    compliance_history: Optional[List[ComplianceHistoryData]] = None

class ProgressionReportResult(BaseModel):
    user_id: str
    generation_time: str
    current_level: LevelData
    submission_metrics: SubmissionMetricsData
    progression_path: ProgressionPathData
    recommendations: List[RecommendationData]

# Keep the existing Pydantic models as they are useful for data structures
class TraitsData(BaseModel):
    """Model for traits data."""
    obedience: Optional[float] = None
    consistency: Optional[float] = None
    initiative: Optional[float] = None
    depth: Optional[float] = None
    protocol_adherence: Optional[float] = None
    receptiveness: Optional[float] = None
    endurance: Optional[float] = None
    attentiveness: Optional[float] = None
    surrender: Optional[float] = None
    reverence: Optional[float] = None

class SubmissionLevel(BaseModel):
    """Represents a level in the submission progression system."""
    id: int
    name: str
    description: str
    min_score: float  # Minimum submission score required
    max_score: float  # Maximum submission score for this level
    traits: TraitsData = Field(default_factory=TraitsData)  # Expected traits at this level
    privileges: List[str] = Field(default_factory=list)
    restrictions: List[str] = Field(default_factory=list)
    training_focus: List[str] = Field(default_factory=list)

class SuitableTraitsData(BaseModel):
    """Model for suitable traits data."""
    service_oriented: Optional[float] = None
    detail_oriented: Optional[float] = None
    methodical: Optional[float] = None
    analytical: Optional[float] = None
    introspective: Optional[float] = None
    emotionally_sensitive: Optional[float] = None
    masochistic: Optional[float] = None
    exhibitionist: Optional[float] = None
    shame_responsive: Optional[float] = None
    structure_seeking: Optional[float] = None
    rule_oriented: Optional[float] = None
    discipline_responsive: Optional[float] = None

class MilestonesData(BaseModel):
    """Model for milestones data."""
    milestone_1: Optional[MilestoneDefinition] = None
    milestone_2: Optional[MilestoneDefinition] = None
    milestone_3: Optional[MilestoneDefinition] = None

class DominancePath(BaseModel):
    """Represents a specific path or style of dominance training."""
    id: str
    name: str
    description: str
    focus_areas: List[str] = Field(default_factory=list)
    recommended_metrics: List[str] = Field(default_factory=list)
    difficulty: float = Field(0.5, ge=0.0, le=1.0)
    suitable_for_traits: SuitableTraitsData = Field(default_factory=SuitableTraitsData)
    progression_milestones: Dict[int, MilestoneDefinition] = Field(default_factory=dict)

class ProgressionMilestone(BaseModel):
    """Represents a specific milestone in submission progression."""
    id: str
    level: int
    name: str
    description: str
    requirements: RequirementsData = Field(default_factory=RequirementsData)
    rewards: List[str] = Field(default_factory=list)
    unlocks: List[str] = Field(default_factory=list)
    completed: bool = False
    completion_date: Optional[datetime.datetime] = None

class SubmissionMetric(BaseModel):
    """Tracks a specific aspect of submission."""
    name: str
    value: float = Field(0.0, ge=0.0, le=1.0)
    weight: float = Field(1.0, ge=0.0, le=2.0)
    last_updated: datetime.datetime = Field(default_factory=datetime.datetime.now)
    history: List[HistoryEntry] = Field(default_factory=list)
    
    def update(self, new_value: float, reason: str = "general"):
        """Update the metric with a new value."""
        # Calculate weighted average between old and new (30% new, 70% old)
        old_value = self.value
        self.value = (self.value * 0.7) + (new_value * 0.3)
        self.value = max(0.0, min(1.0, self.value))  # Constrain to valid range
        self.last_updated = datetime.datetime.now()
        
        # Track history (limited to last 20 entries)
        self.history.append(HistoryEntry(
            timestamp=self.last_updated.isoformat(),
            old_value=old_value,
            new_value=new_value,
            final_value=self.value,
            reason=reason
        ))
        
        if len(self.history) > 20:
            self.history = self.history[-20:]
            
        return self.value

class ComplianceRecord(BaseModel):
    """Record of compliance or defiance for a specific instruction."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    instruction: str
    complied: bool
    difficulty: float = Field(0.5, ge=0.0, le=1.0)
    context: ComplianceContextData = Field(default_factory=ComplianceContextData)
    defiance_reason: Optional[str] = None
    punishment_applied: Optional[str] = None

class UserSubmissionData(BaseModel):
    """Stores all submission-related data for a user."""
    user_id: str
    current_level_id: int = 1
    total_submission_score: float = 0.0
    time_at_current_level: int = 0  # Days
    obedience_metrics: Dict[str, SubmissionMetric] = Field(default_factory=dict)
    compliance_history: List[ComplianceRecord] = Field(default_factory=list)
    limits: LimitsData = Field(default_factory=LimitsData)
    preferences: PreferencesData = Field(default_factory=PreferencesData)
    last_level_change: Optional[datetime.datetime] = None
    lifetime_compliance_rate: float = 0.5
    last_updated: datetime.datetime = Field(default_factory=datetime.datetime.now)
    assigned_path: Optional[str] = None
    assigned_path_date: Optional[datetime.datetime] = None
    milestones: Dict[str, ProgressionMilestone] = Field(default_factory=dict)
    unlocked_features: List[str] = Field(default_factory=list)

# Context class for dependencies
class SubmissionContext:
    """Context object for submission progression agents."""
    def __init__(self, reward_system=None, memory_core=None, relationship_manager=None):
        self.reward_system = reward_system
        self.memory_core = memory_core
        self.relationship_manager = relationship_manager
        self.user_data: Dict[str, UserSubmissionData] = {}
        self.submission_levels: Dict[int, SubmissionLevel] = {}
        self.dominance_paths: Dict[str, DominancePath] = {}
        self.default_metrics = [
            "obedience",         # Following direct instructions
            "consistency",       # Consistent behavior over time
            "initiative",        # Taking submissive actions without prompting
            "depth",             # Depth of submission mindset
            "protocol_adherence",  # Following established protocols
            "receptiveness",     # Receptiveness to training and correction
            "endurance",         # Ability to endure challenging tasks
            "attentiveness",     # Paying attention to dominant's needs
            "surrender",         # Willingness to surrender control
            "reverence"          # Showing proper respect and admiration
        ]
        self.metric_weights = {
            "obedience": 1.5,
            "consistency": 1.2,
            "initiative": 0.8,
            "depth": 1.3,
            "protocol_adherence": 1.0,
            "receptiveness": 1.1,
            "endurance": 0.9,
            "attentiveness": 1.0,
            "surrender": 1.4,
            "reverence": 1.2
        }
        self._init_submission_levels()

    def _init_submission_levels(self):
        """Initialize the submission levels."""
        levels = [
            SubmissionLevel(
                id=1,
                name="Curious",
                description="Initial exploration of submission",
                min_score=0.0,
                max_score=0.2,
                traits=TraitsData(obedience=0.1, receptiveness=0.2),
                privileges=["Can ask questions freely", "Can negotiate boundaries"],
                restrictions=[],
                training_focus=["Basic protocols", "Relationship foundation", "Communication"]
            ),
            SubmissionLevel(
                id=2,
                name="Testing Boundaries",
                description="Testing limits and establishing basic protocols",
                min_score=0.2,
                max_score=0.4,
                traits=TraitsData(obedience=0.3, receptiveness=0.4, protocol_adherence=0.3),
                privileges=["Can request specific activities", "Limited negotiation"],
                restrictions=["Basic protocol adherence required"],
                training_focus=["Protocol reinforcement", "Basic obedience", "Accepting correction"]
            ),
            SubmissionLevel(
                id=3,
                name="Compliant",
                description="Regular compliance with established protocols",
                min_score=0.4,
                max_score=0.6,
                traits=TraitsData(
                    obedience=0.6, 
                    consistency=0.5, 
                    protocol_adherence=0.6,
                    receptiveness=0.6
                ),
                privileges=["Can earn rewards", "Some autonomy in tasks"],
                restrictions=["Must follow all protocols", "Must accept correction without argument"],
                training_focus=["Consistency", "Depth of submission", "Service skills"]
            ),
            SubmissionLevel(
                id=4,
                name="Devoted",
                description="Deep submission with consistent obedience",
                min_score=0.6,
                max_score=0.8,
                traits=TraitsData(
                    obedience=0.8, 
                    consistency=0.7, 
                    protocol_adherence=0.8,
                    depth=0.7,
                    surrender=0.6,
                    reverence=0.7
                ),
                privileges=["Can suggest training areas", "Some flexibility in protocols"],
                restrictions=["High protocol requirements", "Regular ritual participation"],
                training_focus=["Surrender", "Initiative", "Deeper psychological submission"]
            ),
            SubmissionLevel(
                id=5,
                name="Deeply Submissive",
                description="Total submission with deep psychological investment",
                min_score=0.8,
                max_score=1.0,
                traits=TraitsData(
                    obedience=0.9, 
                    consistency=0.9, 
                    protocol_adherence=0.9,
                    depth=0.9,
                    surrender=0.9,
                    reverence=0.9,
                    initiative=0.8,
                    attentiveness=0.8,
                    endurance=0.8
                ),
                privileges=["Considerable trust", "Deep connection"],
                restrictions=["High protocol at all times", "Total control in designated areas"],
                training_focus=["Perfection in service", "Total surrender", "Deep psychological control"]
            )
        ]
        
        for level in levels:
            self.submission_levels[level.id] = level
    
    def _calculate_submission_score(self, user_data: UserSubmissionData) -> float:
        """Calculate the overall submission score based on metrics."""
        if not user_data.obedience_metrics:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric_name, metric in user_data.obedience_metrics.items():
            weighted_sum += metric.value * metric.weight
            total_weight += metric.weight
        
        # If no weights, avoid division by zero
        if total_weight == 0:
            return 0.0
        
        # Calculate score (0.0-1.0)
        score = weighted_sum / total_weight
        
        return max(0.0, min(1.0, score))

    def init_dominance_paths(self):
        """Initialize available dominance paths."""
        self.dominance_paths = {
            "service": DominancePath(
                id="service",
                name="Service-Oriented Path",
                description="Focus on developing service skills, protocols, and rituals.",
                focus_areas=["protocol_adherence", "service", "consistency", "attentiveness"],
                recommended_metrics=["obedience", "consistency", "protocol_adherence", "attentiveness"],
                difficulty=0.4,
                suitable_for_traits=SuitableTraitsData(service_oriented=0.7, detail_oriented=0.6, methodical=0.5),
                progression_milestones={
                    1: MilestoneDefinition(
                        id="basic_protocols",
                        name="Basic Protocols",
                        description="Learn and consistently follow basic service protocols.",
                        requirements=RequirementsData(protocol_adherence=0.3, consistency=0.3),
                        rewards=["basic_protocol_certification"],
                        unlocks=["intermediate_service_tasks"]
                    ),
                    2: MilestoneDefinition(
                        id="consistent_service",
                        name="Consistent Service",
                        description="Demonstrate reliable service across multiple sessions.",
                        requirements=RequirementsData(consistency=0.5, obedience=0.4, protocol_adherence=0.5),
                        rewards=["service_recognition"],
                        unlocks=["advanced_service_positions"]
                    ),
                    3: MilestoneDefinition(
                        id="anticipatory_service",
                        name="Anticipatory Service",
                        description="Anticipate needs and provide service without explicit direction.",
                        requirements=RequirementsData(attentiveness=0.7, initiative=0.6),
                        rewards=["service_excellence_badge"],
                        unlocks=["ritual_creation_privileges"]
                    )
                }
            ),
            "psychological": DominancePath(
                id="psychological",
                name="Psychological Dominance Path",
                description="Focus on mind games, control, and psychological surrender.",
                focus_areas=["depth", "surrender", "receptiveness"],
                recommended_metrics=["depth", "surrender", "receptiveness"],
                difficulty=0.7,
                suitable_for_traits=SuitableTraitsData(analytical=0.6, introspective=0.7, emotionally_sensitive=0.5),
                progression_milestones={
                    1: MilestoneDefinition(
                        id="mental_submission",
                        name="Mental Submission Basics",
                        description="Begin surrendering control mentally and emotionally.",
                        requirements=RequirementsData(surrender=0.3, receptiveness=0.4),
                        rewards=["mind_control_session"],
                        unlocks=["light_mindfuck_techniques"]
                    ),
                    2: MilestoneDefinition(
                        id="psychological_surrender",
                        name="Psychological Surrender",
                        description="Deeper mental submission and acceptance of control.",
                        requirements=RequirementsData(depth=0.5, surrender=0.6),
                        rewards=["psychological_dominance_session"],
                        unlocks=["advanced_mindfuck_techniques"]
                    ),
                    3: MilestoneDefinition(
                        id="cognitive_restructuring",
                        name="Cognitive Restructuring",
                        description="Allow thought patterns to be influenced and restructured.",
                        requirements=RequirementsData(depth=0.7, surrender=0.8, receptiveness=0.7),
                        rewards=["deep_control_badge"],
                        unlocks=["permission_structures", "thought_control_protocols"]
                    )
                }
            ),
            "humiliation": DominancePath(
                id="humiliation",
                name="Humiliation-Focused Path",
                description="Focus on embarrassment, degradation, and humbling experiences.",
                focus_areas=["surrender", "endurance", "receptiveness"],
                recommended_metrics=["surrender", "endurance", "receptiveness"],
                difficulty=0.8,
                suitable_for_traits=SuitableTraitsData(masochistic=0.6, exhibitionist=0.5, shame_responsive=0.7),
                progression_milestones={
                    1: MilestoneDefinition(
                        id="light_embarrassment",
                        name="Light Embarrassment",
                        description="Introduction to light embarrassment and verbal humiliation.",
                        requirements=RequirementsData(surrender=0.3, endurance=0.3),
                        rewards=["humiliation_beginner_badge"],
                        unlocks=["moderate_humiliation_tasks"]
                    ),
                    2: MilestoneDefinition(
                        id="moderate_humiliation",
                        name="Moderate Humiliation",
                        description="Acceptance of regular humiliation and embarrassment.",
                        requirements=RequirementsData(surrender=0.5, endurance=0.6),
                        rewards=["humiliation_intermediate_badge"],
                        unlocks=["embarrassment_challenges"]
                    ),
                    3: MilestoneDefinition(
                        id="deep_degradation",
                        name="Deep Degradation",
                        description="Acceptance of profound humiliation and degradation.",
                        requirements=RequirementsData(surrender=0.8, endurance=0.7, receptiveness=0.7),
                        rewards=["humiliation_advanced_badge"],
                        unlocks=["custom_humiliation_scenarios"]
                    )
                }
            ),
            "strict_discipline": DominancePath(
                id="strict_discipline",
                name="Strict Discipline Path",
                description="Focus on rules, punishment, and strict behavioral standards.",
                focus_areas=["obedience", "discipline", "protocol_adherence"],
                recommended_metrics=["obedience", "endurance", "protocol_adherence"],
                difficulty=0.6,
                suitable_for_traits=SuitableTraitsData(structure_seeking=0.7, rule_oriented=0.6, discipline_responsive=0.7),
                progression_milestones={
                    1: MilestoneDefinition(
                        id="basic_rules",
                        name="Basic Rules Adherence",
                        description="Consistent following of basic rules and acceptance of correction.",
                        requirements=RequirementsData(obedience=0.4, protocol_adherence=0.3),
                        rewards=["discipline_beginner_badge"],
                        unlocks=["intermediate_rule_structures"]
                    ),
                    2: MilestoneDefinition(
                        id="punishment_acceptance",
                        name="Punishment Acceptance",
                        description="Full acceptance of punishments and corrections.",
                        requirements=RequirementsData(obedience=0.6, endurance=0.5, receptiveness=0.5),
                        rewards=["discipline_intermediate_badge"],
                        unlocks=["advanced_punishment_protocols"]
                    ),
                    3: MilestoneDefinition(
                        id="internalized_discipline",
                        name="Internalized Discipline",
                        description="Self-monitoring and anticipatory obedience.",
                        requirements=RequirementsData(obedience=0.8, protocol_adherence=0.7, initiative=0.6),
                        rewards=["discipline_advanced_badge"],
                        unlocks=["self_discipline_protocols", "punishment_authority"]
                    )
                }
            )
        }
        
        logger.info(f"Initialized {len(self.dominance_paths)} dominance progression paths")
        return self.dominance_paths

# Define guardrail output types
class SensitiveContentCheck(BaseModel):
    """Schema for checking if a request contains sensitive content"""
    is_sensitive: bool
    reasoning: str

# User initialization result
class UserInitResult(BaseModel):
    """Output schema for user initialization"""
    user_id: str
    initialized: bool
    current_level: LevelData
    metrics: SimpleMetricsData

# Main submission progression agent setup
class SubmissionProgression:
    """Tracks user's submission journey and training progress using the OpenAI Agents SDK."""
    
    def __init__(self, reward_system=None, memory_core=None, relationship_manager=None):
        """Initialize the submission progression system with dependency injection."""
        # Create context object for sharing between agents
        self.context = SubmissionContext(
            reward_system=reward_system,
            memory_core=memory_core,
            relationship_manager=relationship_manager
        )
        
        # Initialize dominance paths
        self.context.init_dominance_paths()
        
        # Create specialized agents for different tasks
        self._setup_agents()
        
        logger.info("SubmissionProgression system initialized with OpenAI Agents SDK")
    
    def _setup_agents(self):
        """Set up specialized agents for different submission progression tasks."""
        # Create model settings with moderate temperature for varied output
        model_settings = ModelSettings(temperature=0.7)
        
        # Main submission progression agent (triage)
        self.triage_agent = Agent(
            name="Submission Progression Triage",
            instructions="""
            You are a specialized system for managing femdom submission progression. 
            You determine which specialized agent should handle a request related to tracking, 
            analyzing and developing a user's submission journey.
            
            Maintain a strict, dominant tone in your responses that reinforces power dynamics.
            For requests involving metrics analysis, progression evaluation, or level assessment,
            use the appropriate specialized agent.
            """,
            model_settings=model_settings
        )
        
        # User initialization agent
        self.user_init_agent = Agent(
            name="User Initialization Agent",
            instructions="""
            You are responsible for initializing new users in the submission progression system.
            Your task is to:
            1. Create baseline metrics for new users
            2. Set appropriate initial submission levels
            3. Establish limits and preferences if provided
            
            Analyze any initial data provided and assign proper starting values.
            """,
            model_settings=model_settings,
            output_type=UserInitResult
        )
        
        # Path recommendation agent
        self.path_recommendation_agent = Agent(
            name="Path Recommendation Agent",
            instructions="""
            You are a specialized agent for recommending dominance paths to users.
            Analyze user traits, preferences, and current metrics to determine the most
            suitable training path. Consider psychological profile, demonstrated behaviors,
            and stated preferences.
            
            Provide clear reasoning for your recommendations with a strict, dominant tone.
            """,
            model_settings=model_settings
        )
        
        # Milestone tracking agent
        self.milestone_agent = Agent(
            name="Milestone Tracking Agent",
            instructions="""
            You are responsible for tracking user progress through dominance path milestones.
            Your tasks include:
            1. Determining if milestone requirements have been met
            2. Identifying upcoming milestones and providing guidance
            3. Analyzing progress trends and providing recommendations
            
            Be precise in your evaluations and maintain a strict but encouraging tone.
            """,
            model_settings=model_settings
        )
        
        # Compliance recording agent
        self.compliance_agent = Agent(
            name="Compliance Recording Agent",
            instructions="""
            You are responsible for analyzing and recording user compliance with instructions.
            Your tasks include:
            1. Updating appropriate metrics based on compliance or defiance
            2. Determining effects on overall submission score
            3. Assessing changes in submission level based on new data
            
            Be exacting in your analysis and maintain a stern tone for defiance.
            """,
            model_settings=model_settings
        )
        
        # Submission reporting agent
        self.reporting_agent = Agent(
            name="Submission Reporting Agent",
            instructions="""
            You are responsible for generating comprehensive reports on user submission progress.
            Your reports should include:
            1. Current submission level and metrics
            2. Progress analysis and trends
            3. Recommendations for advancement
            4. Areas requiring improvement
            
            Your tone should be analytical but with a dominant edge that reinforces power dynamics.
            """,
            model_settings=model_settings
        )
        
        # Set up handoffs between agents
        self.triage_agent = Agent(
            name="Submission Progression Triage",
            instructions="""
            You are a specialized system for managing femdom submission progression. 
            You determine which specialized agent should handle a request related to tracking, 
            analyzing and developing a user's submission journey.
            
            Maintain a strict, dominant tone in your responses that reinforces power dynamics.
            For requests involving metrics analysis, progression evaluation, or level assessment,
            use the appropriate specialized agent.
            """,
            model_settings=model_settings,
            handoffs=[
                self.user_init_agent,
                self.path_recommendation_agent,
                self.milestone_agent,
                self.compliance_agent,
                self.reporting_agent
            ]
        )
        
        # Set up content guardrail
        self.sensitive_content_agent = Agent(
            name="Content Sensitivity Check",
            instructions="Review if the input contains material that requires special handling for sensitive content.",
            output_type=SensitiveContentCheck
        )
        
        @input_guardrail
        async def sensitive_content_guardrail(ctx, agent, input_data):
            result = await Runner.run(self.sensitive_content_agent, input_data, context=ctx.context)
            return GuardrailFunctionOutput(
                output_info=result.final_output_as(SensitiveContentCheck),
                tripwire_triggered=result.final_output_as(SensitiveContentCheck).is_sensitive
            )
        
        # Apply guardrail to triage agent
        self.triage_agent = Agent(
            name="Submission Progression Triage",
            instructions="""
            You are a specialized system for managing femdom submission progression. 
            You determine which specialized agent should handle a request related to tracking, 
            analyzing and developing a user's submission journey.
            
            Maintain a strict, dominant tone in your responses that reinforces power dynamics.
            For requests involving metrics analysis, progression evaluation, or level assessment,
            use the appropriate specialized agent.
            """,
            model_settings=model_settings,
            handoffs=[
                self.user_init_agent,
                self.path_recommendation_agent,
                self.milestone_agent,
                self.compliance_agent,
                self.reporting_agent
            ],
            input_guardrails=[sensitive_content_guardrail]
        )
    
    # Create function tools for the various operations
    
    @function_tool
    async def initialize_user(self, ctx, user_id: str, initial_data: Optional[InitialUserData] = None) -> UserInitResult:
        """
        Initialize or get user submission data.
        
        Args:
            user_id: Unique identifier for the user
            initial_data: Optional initial data for the user including limits, preferences, etc.
            
        Returns:
            Initialization result with user data
        """
        if user_id in self.context.user_data:
            user_data = self.context.user_data[user_id]
        else:
            # Create new user data
            user_data = UserSubmissionData(user_id=user_id)
            
            # Initialize metrics
            for metric_name in self.context.default_metrics:
                weight = self.context.metric_weights.get(metric_name, 1.0)
                user_data.obedience_metrics[metric_name] = SubmissionMetric(
                    name=metric_name,
                    value=0.1,  # Start at minimal value
                    weight=weight
                )
            
            # Apply any initial data if provided
            if initial_data:
                if initial_data.limits:
                    user_data.limits = initial_data.limits
                
                if initial_data.preferences:
                    user_data.preferences = initial_data.preferences
                
                # Set initial metrics if provided
                if initial_data.metrics:
                    if initial_data.metrics.obedience is not None:
                        user_data.obedience_metrics["obedience"].value = initial_data.metrics.obedience
                    if initial_data.metrics.consistency is not None:
                        user_data.obedience_metrics["consistency"].value = initial_data.metrics.consistency
                    if initial_data.metrics.initiative is not None:
                        user_data.obedience_metrics["initiative"].value = initial_data.metrics.initiative
                    if initial_data.metrics.depth is not None:
                        user_data.obedience_metrics["depth"].value = initial_data.metrics.depth
                    if initial_data.metrics.protocol_adherence is not None:
                        user_data.obedience_metrics["protocol_adherence"].value = initial_data.metrics.protocol_adherence
                    if initial_data.metrics.receptiveness is not None:
                        user_data.obedience_metrics["receptiveness"].value = initial_data.metrics.receptiveness
                    if initial_data.metrics.endurance is not None:
                        user_data.obedience_metrics["endurance"].value = initial_data.metrics.endurance
                    if initial_data.metrics.attentiveness is not None:
                        user_data.obedience_metrics["attentiveness"].value = initial_data.metrics.attentiveness
                    if initial_data.metrics.surrender is not None:
                        user_data.obedience_metrics["surrender"].value = initial_data.metrics.surrender
                    if initial_data.metrics.reverence is not None:
                        user_data.obedience_metrics["reverence"].value = initial_data.metrics.reverence
                
                # Set initial level if provided
                if initial_data.level:
                    level_id = initial_data.level
                    if level_id in self.context.submission_levels:
                        user_data.current_level_id = level_id
                        
                # Set initial path if provided
                if initial_data.path:
                    path_id = initial_data.path
                    if path_id in self.context.dominance_paths:
                        user_data.assigned_path = path_id
                        user_data.assigned_path_date = datetime.datetime.now()
            
            # Calculate initial submission score
            user_data.total_submission_score = self.context._calculate_submission_score(user_data)
            
            # Set last level change to now
            user_data.last_level_change = datetime.datetime.now()
            
            # Store user data
            self.context.user_data[user_id] = user_data
            
            logger.info(f"Initialized submission tracking for user {user_id} at level {user_data.current_level_id}")
        
        # Get current level details
        level = self.context.submission_levels[user_data.current_level_id]
        
        # Format metrics for output
        metrics_dict = {name: metric.value for name, metric in user_data.obedience_metrics.items()}
        metrics = SimpleMetricsData(
            obedience=metrics_dict.get("obedience"),
            consistency=metrics_dict.get("consistency"),
            initiative=metrics_dict.get("initiative"),
            depth=metrics_dict.get("depth"),
            protocol_adherence=metrics_dict.get("protocol_adherence"),
            receptiveness=metrics_dict.get("receptiveness"),
            endurance=metrics_dict.get("endurance"),
            attentiveness=metrics_dict.get("attentiveness"),
            surrender=metrics_dict.get("surrender"),
            reverence=metrics_dict.get("reverence")
        )
        
        # Create result
        return UserInitResult(
            user_id=user_id,
            initialized=True,
            current_level=LevelData(
                id=level.id,
                name=level.name,
                description=level.description,
                privileges=level.privileges,
                restrictions=level.restrictions
            ),
            metrics=metrics
        )
    
    @function_tool
    async def recommend_dominance_path(self, ctx, user_id: str) -> PathRecommendationResult:
        """
        Recommends the most suitable dominance path based on user traits and preferences.
        
        Args:
            user_id: The user to analyze
            
        Returns:
            Recommendation details
        """
        if user_id not in self.context.user_data:
            await self.initialize_user(ctx, user_id)
        
        user_data = self.context.user_data[user_id]
        
        # Get user traits and preferences
        user_traits = {}
        user_preferences = {}
        
        # Try to get from relationship manager if available
        if self.context.relationship_manager:
            try:
                relationship = await self.context.relationship_manager.get_relationship_state(user_id)
                if hasattr(relationship, "inferred_user_traits"):
                    user_traits = relationship.inferred_user_traits
                if hasattr(relationship, "preferences"):
                    user_preferences = relationship.preferences
            except Exception as e:
                logger.error(f"Error getting relationship data: {e}")
        
        # Calculate match scores for each path
        path_scores = {}
        for path_id, path in self.context.dominance_paths.items():
            # Base score
            score = 0.0
            
            # Match based on traits
            trait_match_score = 0.0
            trait_count = 0
            
            # Convert SuitableTraitsData to dict for processing
            path_traits_dict = {}
            if path.suitable_for_traits.service_oriented is not None:
                path_traits_dict["service_oriented"] = path.suitable_for_traits.service_oriented
            if path.suitable_for_traits.detail_oriented is not None:
                path_traits_dict["detail_oriented"] = path.suitable_for_traits.detail_oriented
            if path.suitable_for_traits.methodical is not None:
                path_traits_dict["methodical"] = path.suitable_for_traits.methodical
            if path.suitable_for_traits.analytical is not None:
                path_traits_dict["analytical"] = path.suitable_for_traits.analytical
            if path.suitable_for_traits.introspective is not None:
                path_traits_dict["introspective"] = path.suitable_for_traits.introspective
            if path.suitable_for_traits.emotionally_sensitive is not None:
                path_traits_dict["emotionally_sensitive"] = path.suitable_for_traits.emotionally_sensitive
            if path.suitable_for_traits.masochistic is not None:
                path_traits_dict["masochistic"] = path.suitable_for_traits.masochistic
            if path.suitable_for_traits.exhibitionist is not None:
                path_traits_dict["exhibitionist"] = path.suitable_for_traits.exhibitionist
            if path.suitable_for_traits.shame_responsive is not None:
                path_traits_dict["shame_responsive"] = path.suitable_for_traits.shame_responsive
            if path.suitable_for_traits.structure_seeking is not None:
                path_traits_dict["structure_seeking"] = path.suitable_for_traits.structure_seeking
            if path.suitable_for_traits.rule_oriented is not None:
                path_traits_dict["rule_oriented"] = path.suitable_for_traits.rule_oriented
            if path.suitable_for_traits.discipline_responsive is not None:
                path_traits_dict["discipline_responsive"] = path.suitable_for_traits.discipline_responsive
                
            for trait, required_value in path_traits_dict.items():
                trait_count += 1
                if trait in user_traits:
                    user_value = user_traits[trait]
                    # Higher score for closer trait match
                    trait_match_score += 1.0 - abs(user_value - required_value)
            
            # Average trait match
            if trait_count > 0:
                trait_match_score /= trait_count
                score += trait_match_score * 0.6  # Traits are 60% of score
            
            # Match based on metrics
            metric_match_score = 0.0
            metric_count = 0
            for metric_name in path.recommended_metrics:
                metric_count += 1
                if metric_name in user_data.obedience_metrics:
                    metric = user_data.obedience_metrics[metric_name]
                    # Higher score for already-developed metrics
                    metric_match_score += metric.value
            
            # Average metric match
            if metric_count > 0:
                metric_match_score /= metric_count
                score += metric_match_score * 0.4  # Metrics are 40% of score
                
            # Store score
            path_scores[path_id] = score
        
        # Get top 3 recommended paths
        sorted_paths = sorted(path_scores.items(), key=lambda x: x[1], reverse=True)
        top_paths = sorted_paths[:3]
        
        # Format recommendations
        recommendations = []
        for path_id, score in top_paths:
            path = self.context.dominance_paths[path_id]
            recommendations.append(PathRecommendationData(
                id=path_id,
                name=path.name,
                description=path.description,
                match_score=score,
                difficulty=path.difficulty,
                focus_areas=path.focus_areas
            ))
        
        # Return recommendations
        return PathRecommendationResult(
            user_id=user_id,
            primary_recommendation=recommendations[0] if recommendations else None,
            all_recommendations=recommendations,
            user_traits_analyzed=list(user_traits.keys()),
            analysis_timestamp=datetime.datetime.now().isoformat()
        )
    
    @function_tool
    async def assign_dominance_path(self, ctx, user_id: str, path_id: str) -> PathAssignmentResult:
        """
        Assigns a specific dominance path to a user.
        
        Args:
            user_id: The user to assign the path to
            path_id: The path ID to assign
            
        Returns:
            Assignment details
        """
        if path_id not in self.context.dominance_paths:
            return PathAssignmentResult(
                success=False,
                message=f"Path '{path_id}' not found",
                available_paths=list(self.context.dominance_paths.keys())
            )
            
        if user_id not in self.context.user_data:
            await self.initialize_user(ctx, user_id)
        
        user_data = self.context.user_data[user_id]
        
        # Assign path
        user_data.assigned_path = path_id
        user_data.assigned_path_date = datetime.datetime.now()
        
        # Initialize milestone tracking if not exists
        if not hasattr(user_data, "milestones"):
            user_data.milestones = {}
        
        # Initialize milestones for this path
        path = self.context.dominance_paths[path_id]
        for level, milestone_data in path.progression_milestones.items():
            milestone_id = milestone_data.id
            if milestone_id not in user_data.milestones:
                user_data.milestones[milestone_id] = ProgressionMilestone(
                    id=milestone_id,
                    level=level,
                    name=milestone_data.name,
                    description=milestone_data.description,
                    requirements=milestone_data.requirements,
                    rewards=milestone_data.rewards,
                    unlocks=milestone_data.unlocks,
                    completed=False
                )
        
        # Record in relationship manager if available
        if self.context.relationship_manager:
            try:
                await self.context.relationship_manager.update_relationship_attribute(
                    user_id,
                    "dominance_path",
                    {
                        "id": path_id,
                        "name": path.name,
                        "assigned_date": datetime.datetime.now().isoformat()
                    }
                )
            except Exception as e:
                logger.error(f"Error updating relationship data: {e}")
        
        return PathAssignmentResult(
            success=True,
            user_id=user_id,
            path_id=path_id,
            path_name=path.name,
            difficulty=path.difficulty,
            focus_areas=path.focus_areas,
            milestones=len(path.progression_milestones),
            message=f"Assigned dominance path '{path.name}' to user"
        )
    
    @function_tool
    async def check_milestone_progress(self, ctx, user_id: str) -> MilestoneProgressResult:
        """
        Checks user progress against assigned path milestones.
        
        Args:
            user_id: The user to check
            
        Returns:
            Milestone progress details
        """
        if user_id not in self.context.user_data:
            return MilestoneProgressResult(
                success=False,
                message=f"No submission data found for user {user_id}",
                newly_completed_milestones=[],
                upcoming_milestones=[],
                already_completed_milestones=[],
                progress_summary=ProgressSummary(total_milestones=0, completed=0, completion_percentage=0.0)
            )
            
        user_data = self.context.user_data[user_id]
        
        # Check if user has an assigned path
        if not user_data.assigned_path:
            return MilestoneProgressResult(
                success=False,
                message="No dominance path assigned to user",
                recommendation="Use assign_dominance_path to assign a path",
                newly_completed_milestones=[],
                upcoming_milestones=[],
                already_completed_milestones=[],
                progress_summary=ProgressSummary(total_milestones=0, completed=0, completion_percentage=0.0)
            )
        
        path_id = user_data.assigned_path
        if path_id not in self.context.dominance_paths:
            return MilestoneProgressResult(
                success=False,
                message=f"Assigned path '{path_id}' not found",
                newly_completed_milestones=[],
                upcoming_milestones=[],
                already_completed_milestones=[],
                progress_summary=ProgressSummary(total_milestones=0, completed=0, completion_percentage=0.0)
            )
            
        path = self.context.dominance_paths[path_id]
        
        # Check each milestone for completion
        newly_completed = []
        upcoming_milestones = []
        already_completed = []
        
        for level, milestone_data in path.progression_milestones.items():
            milestone_id = milestone_data.id
            
            # Create milestone if it doesn't exist
            if milestone_id not in user_data.milestones:
                user_data.milestones[milestone_id] = ProgressionMilestone(
                    id=milestone_id,
                    level=level,
                    name=milestone_data.name,
                    description=milestone_data.description,
                    requirements=milestone_data.requirements,
                    rewards=milestone_data.rewards,
                    unlocks=milestone_data.unlocks,
                    completed=False
                )
                
            milestone = user_data.milestones[milestone_id]
            
            # Skip if already completed
            if milestone.completed:
                already_completed.append(MilestoneData(
                    id=milestone_id,
                    name=milestone.name,
                    level=milestone.level,
                    completion_date=milestone.completion_date.isoformat() if milestone.completion_date else None
                ))
                continue
                
            # Check requirements
            all_requirements_met = True
            requirement_progress = {}
            
            # Convert RequirementsData to dict for checking
            requirements_dict = {}
            if milestone.requirements.obedience is not None:
                requirements_dict["obedience"] = milestone.requirements.obedience
            if milestone.requirements.consistency is not None:
                requirements_dict["consistency"] = milestone.requirements.consistency
            if milestone.requirements.initiative is not None:
                requirements_dict["initiative"] = milestone.requirements.initiative
            if milestone.requirements.depth is not None:
                requirements_dict["depth"] = milestone.requirements.depth
            if milestone.requirements.protocol_adherence is not None:
                requirements_dict["protocol_adherence"] = milestone.requirements.protocol_adherence
            if milestone.requirements.receptiveness is not None:
                requirements_dict["receptiveness"] = milestone.requirements.receptiveness
            if milestone.requirements.endurance is not None:
                requirements_dict["endurance"] = milestone.requirements.endurance
            if milestone.requirements.attentiveness is not None:
                requirements_dict["attentiveness"] = milestone.requirements.attentiveness
            if milestone.requirements.surrender is not None:
                requirements_dict["surrender"] = milestone.requirements.surrender
            if milestone.requirements.reverence is not None:
                requirements_dict["reverence"] = milestone.requirements.reverence
            
            for metric_name, threshold in requirements_dict.items():
                if metric_name in user_data.obedience_metrics:
                    current_value = user_data.obedience_metrics[metric_name].value
                    requirement_met = current_value >= threshold
                    progress_pct = min(100, (current_value / threshold) * 100) if threshold > 0 else 100
                    
                    requirement_progress[metric_name] = {
                        "current": current_value,
                        "threshold": threshold,
                        "met": requirement_met,
                        "progress_percentage": progress_pct
                    }
                    
                    if not requirement_met:
                        all_requirements_met = False
                else:
                    # Metric not found
                    requirement_progress[metric_name] = {
                        "current": 0.0,
                        "threshold": threshold,
                        "met": False,
                        "progress_percentage": 0.0
                    }
                    all_requirements_met = False
                    
            # Check if milestone is completed
            if all_requirements_met:
                # Mark as completed
                milestone.completed = True
                milestone.completion_date = datetime.datetime.now()
                
                newly_completed.append(MilestoneData(
                    id=milestone_id,
                    name=milestone.name,
                    level=milestone.level,
                    rewards=milestone.rewards,
                    unlocks=milestone.unlocks
                ))
            else:
                # Add to upcoming
                overall_progress = sum(r["progress_percentage"] for r in requirement_progress.values()) / len(requirement_progress)
                upcoming_milestones.append(MilestoneData(
                    id=milestone_id,
                    name=milestone.name,
                    level=milestone.level,
                    description=milestone.description,
                    requirements=milestone.requirements,
                    overall_progress=overall_progress
                ))
        
        # Sort upcoming milestones by progress
        upcoming_milestones.sort(key=lambda m: m.overall_progress or 0, reverse=True)
        
        # Apply rewards for newly completed milestones
        for milestone in newly_completed:
            # Create reward signal if available
            if self.context.reward_system:
                try:
                    # Higher reward for higher-level milestones
                    reward_value = 0.3 + (milestone.level * 0.15)  # 0.45 for level 1, 0.6 for level 2, etc.
                    
                    await self.context.reward_system.process_reward_signal(
                        self.context.reward_system.RewardSignal(
                            value=reward_value,
                            source="dominance_milestone",
                            context={
                                "milestone_id": milestone.id,
                                "milestone_name": milestone.name,
                                "level": milestone.level,
                                "path_id": path_id,
                                "path_name": path.name
                            }
                        )
                    )
                except Exception as e:
                    logger.error(f"Error processing reward: {e}")
        
        # Update relationship manager if available and milestones were completed
        if newly_completed and self.context.relationship_manager:
            try:
                await self.context.relationship_manager.update_relationship_attribute(
                    user_id,
                    "completed_milestones",
                    [m.id for m in already_completed + newly_completed]
                )
            except Exception as e:
                logger.error(f"Error updating relationship data: {e}")
        
        return MilestoneProgressResult(
            success=True,
            user_id=user_id,
            path_id=path_id,
            path_name=path.name,
            newly_completed_milestones=newly_completed,
            upcoming_milestones=upcoming_milestones[:3],  # Top 3 upcoming
            already_completed_milestones=already_completed,
            progress_summary=ProgressSummary(
                total_milestones=len(path.progression_milestones),
                completed=len(already_completed) + len(newly_completed),
                completion_percentage=((len(already_completed) + len(newly_completed)) / len(path.progression_milestones)) * 100
            )
        )
    
    @function_tool
    async def record_compliance(self, 
                              ctx,
                              user_id: str, 
                              instruction: str, 
                              complied: bool, 
                              difficulty: float = 0.5,
                              context_info: Optional[ComplianceContextInfo] = None,
                              defiance_reason: Optional[str] = None) -> ComplianceRecordResult:
        """
        Record compliance or defiance for a specific instruction.
        
        Args:
            user_id: User ID
            instruction: The instruction given
            complied: Whether user complied
            difficulty: How difficult the instruction was (0.0-1.0)
            context_info: Additional context about the instruction
            defiance_reason: If defied, the reason given
            
        Returns:
            Dict with results of the operation
        """
        # Ensure user exists
        if user_id not in self.context.user_data:
            await self.initialize_user(ctx, user_id)
        
        user_data = self.context.user_data[user_id]
        
        # Convert context_info to ComplianceContextData
        context_data = ComplianceContextData()
        if context_info:
            context_data = context_info
        
        # Create compliance record
        record = ComplianceRecord(
            instruction=instruction,
            complied=complied,
            difficulty=difficulty,
            context=context_data,
            defiance_reason=defiance_reason
        )
        
        # Add to history
        user_data.compliance_history.append(record)
        
        # Limit history size
        if len(user_data.compliance_history) > 100:
            user_data.compliance_history = user_data.compliance_history[-100:]
        
        # Update lifetime compliance rate
        total_records = len(user_data.compliance_history)
        compliant_records = sum(1 for r in user_data.compliance_history if r.complied)
        
        if total_records > 0:
            user_data.lifetime_compliance_rate = compliant_records / total_records
        
        # Update metrics based on compliance
        metrics_updates = MetricsUpdatesData()
        
        # Update obedience metric directly
        if "obedience" in user_data.obedience_metrics:
            obedience_change = 0.1 if complied else -0.15  # More penalty for defiance
            
            # Scale by difficulty (harder instructions count more)
            obedience_change *= (0.5 + difficulty * 0.5)
            
            old_value = user_data.obedience_metrics["obedience"].value
            new_value = old_value + obedience_change
            new_value = max(0.0, min(1.0, new_value))  # Constrain to valid range
            
            user_data.obedience_metrics["obedience"].update(
                new_value, 
                reason=f"{'Compliance' if complied else 'Defiance'} - {instruction[:30]}..."
            )
            
            metrics_updates.obedience = MetricUpdateData(
                old_value=old_value,
                new_value=new_value,
                change=obedience_change
            )
        
        # Update consistency metric based on recent pattern
        if "consistency" in user_data.obedience_metrics and len(user_data.compliance_history) >= 3:
            recent_records = user_data.compliance_history[-3:]
            pattern = [r.complied for r in recent_records]
            
            # If all the same (consistent)
            if all(pattern) or not any(pattern):
                old_value = user_data.obedience_metrics["consistency"].value
                
                # Increase consistency if compliant, decrease if defiant
                consistency_change = 0.05 if all(pattern) else -0.05
                new_value = old_value + consistency_change
                new_value = max(0.0, min(1.0, new_value))
                
                user_data.obedience_metrics["consistency"].update(
                    new_value,
                    reason="Consistent pattern detected"
                )
                
                metrics_updates.consistency = MetricUpdateData(
                    old_value=old_value,
                    new_value=new_value,
                    change=consistency_change
                )
        
        # Update receptiveness metric if complied after previous defiance
        if complied and len(user_data.compliance_history) >= 2:
            previous_record = user_data.compliance_history[-2]
            if not previous_record.complied and "receptiveness" in user_data.obedience_metrics:
                old_value = user_data.obedience_metrics["receptiveness"].value
                
                # Significant improvement for returning to compliance
                receptiveness_change = 0.08
                new_value = old_value + receptiveness_change
                new_value = max(0.0, min(1.0, new_value))
                
                user_data.obedience_metrics["receptiveness"].update(
                    new_value,
                    reason="Returned to compliance after defiance"
                )
                
                metrics_updates.receptiveness = MetricUpdateData(
                    old_value=old_value,
                    new_value=new_value,
                    change=receptiveness_change
                )
        
        # Recalculate overall submission score
        old_score = user_data.total_submission_score
        user_data.total_submission_score = self.context._calculate_submission_score(user_data)
        
        # Update user's last_updated timestamp
        user_data.last_updated = datetime.datetime.now()
        
        # Evaluate for level change
        level_changed, level_change_details = await self._check_level_change(user_id)
        
        # Issue appropriate reward signal
        reward_result = None
        if self.context.reward_system:
            try:
                # Base reward value on compliance and difficulty
                base_reward = 0.3 if complied else -0.4
                difficulty_modifier = difficulty * 0.4
                reward_value = base_reward + (difficulty_modifier if complied else -difficulty_modifier)
                
                reward_result = await self.context.reward_system.process_reward_signal(
                    self.context.reward_system.RewardSignal(
                        value=reward_value,
                        source="compliance_tracking",
                        context={
                            "instruction": instruction,
                            "complied": complied,
                            "difficulty": difficulty,
                            "defiance_reason": defiance_reason,
                            "submission_level": user_data.current_level_id,
                            "submission_score": user_data.total_submission_score
                        }
                    )
                )
                
                if reward_result:
                    reward_result = RewardResult(
                        value=reward_value,
                        source="compliance_tracking",
                        processed=True
                    )
            except Exception as e:
                logger.error(f"Error processing reward: {e}")
        
        # Record memory if available
        if self.context.memory_core:
            try:
                level_name = self.context.submission_levels[user_data.current_level_id].name
                memory_content = (
                    f"User {'complied with' if complied else 'defied'} instruction: {instruction}. "
                    f"Submission level: {level_name} ({user_data.total_submission_score:.2f})"
                )
                
                await self.context.memory_core.add_memory(
                    memory_type="experience",
                    content=memory_content,
                    tags=["compliance", "submission", 
                          "obedience" if complied else "defiance"],
                    significance=0.3 + (difficulty * 0.3) + (0.3 if level_changed else 0.0)
                )
            except Exception as e:
                logger.error(f"Error recording memory: {e}")
        
        return ComplianceRecordResult(
            success=True,
            record_id=record.id,
            compliance_recorded=complied,
            metrics_updated=metrics_updates,
            submission_score=SubmissionScoreData(
                old=old_score,
                new=user_data.total_submission_score,
                change=user_data.total_submission_score - old_score
            ),
            level_changed=level_changed,
            level_change_details=level_change_details,
            reward_result=reward_result
        )
    
    @function_tool
    async def update_submission_metric(self, 
                                    ctx,
                                    user_id: str, 
                                    metric_name: str, 
                                    value_change: float,
                                    reason: str = "general") -> MetricUpdateResult:
        """
        Update a specific submission metric.
        
        Args:
            user_id: User ID
            metric_name: Name of metric to update
            value_change: Amount to change metric by (positive or negative)
            reason: Reason for the change
            
        Returns:
            Dict with results of the operation
        """
        # Ensure user exists
        if user_id not in self.context.user_data:
            await self.initialize_user(ctx, user_id)
        
        user_data = self.context.user_data[user_id]
        
        # Check if metric exists
        if metric_name not in user_data.obedience_metrics:
            return MetricUpdateResult(
                success=False,
                message=f"Metric '{metric_name}' not found for user",
                submission_score=SubmissionScoreData(old=0.0, new=0.0, change=0.0),
                level_changed=False,
                level_change_details=LevelChangeDetails()
            )
        
        metric = user_data.obedience_metrics[metric_name]
        old_value = metric.value
        
        # Apply change
        new_raw_value = old_value + value_change
        new_raw_value = max(0.0, min(1.0, new_raw_value))
        
        # Update the metric
        metric.update(new_raw_value, reason=reason)
        
        # Recalculate overall submission score
        old_score = user_data.total_submission_score
        user_data.total_submission_score = self.context._calculate_submission_score(user_data)
        
        # Update user's last_updated timestamp
        user_data.last_updated = datetime.datetime.now()
        
        # Evaluate for level change
        level_changed, level_change_details = await self._check_level_change(user_id)
        
        return MetricUpdateResult(
            success=True,
            metric=metric_name,
            old_value=old_value,
            new_value=metric.value,
            change=metric.value - old_value,
            submission_score=SubmissionScoreData(
                old=old_score,
                new=user_data.total_submission_score,
                change=user_data.total_submission_score - old_score
            ),
            level_changed=level_changed,
            level_change_details=level_change_details
        )
    
    async def _check_level_change(self, user_id: str) -> Tuple[bool, LevelChangeDetails]:
        """Check if user should change submission levels based on score."""
        user_data = self.context.user_data[user_id]
        current_level = self.context.submission_levels[user_data.current_level_id]
        score = user_data.total_submission_score
        
        # Check if score outside current level bounds
        if score > current_level.max_score and user_data.current_level_id < max(self.context.submission_levels.keys()):
            # Level up
            new_level_id = user_data.current_level_id + 1
            old_level_id = user_data.current_level_id
            
            # Update user level
            user_data.current_level_id = new_level_id
            user_data.last_level_change = datetime.datetime.now()
            user_data.time_at_current_level = 0
            
            # Get level objects
            old_level = self.context.submission_levels[old_level_id]
            new_level = self.context.submission_levels[new_level_id]
            
            # Record level up event in relationship manager if available
            if self.context.relationship_manager:
                try:
                    await self.context.relationship_manager.update_relationship_attribute(
                        user_id,
                        "submission_level",
                        new_level_id
                    )
                    
                    await self.context.relationship_manager.update_relationship_attribute(
                        user_id,
                        "submission_score",
                        score
                    )
                except Exception as e:
                    logger.error(f"Error updating relationship data: {e}")
            
            # Send positive reward for level up
            if self.context.reward_system:
                try:
                    level_up_reward = 0.5 + (new_level_id * 0.1)  # Higher levels give better rewards
                    
                    await self.context.reward_system.process_reward_signal(
                        self.context.reward_system.RewardSignal(
                            value=level_up_reward,
                            source="submission_level_up",
                            context={
                                "old_level": old_level_id,
                                "old_level_name": old_level.name,
                                "new_level": new_level_id,
                                "new_level_name": new_level.name,
                                "submission_score": score
                            }
                        )
                    )
                except Exception as e:
                    logger.error(f"Error processing level up reward: {e}")
            
            logger.info(f"User {user_id} advanced from submission level {old_level_id} to {new_level_id}")
            
            return True, LevelChangeDetails(
                change_type="level_up",
                old_level=LevelData(
                    id=old_level_id,
                    name=old_level.name
                ),
                new_level=LevelData(
                    id=new_level_id,
                    name=new_level.name,
                    description=new_level.description,
                    privileges=new_level.privileges,
                    restrictions=new_level.restrictions,
                    training_focus=new_level.training_focus
                )
            )
            
        elif score < current_level.min_score and user_data.current_level_id > min(self.context.submission_levels.keys()):
            # Level down
            new_level_id = user_data.current_level_id - 1
            old_level_id = user_data.current_level_id
            
            # Update user level
            user_data.current_level_id = new_level_id
            user_data.last_level_change = datetime.datetime.now()
            user_data.time_at_current_level = 0
            
            # Get level objects
            old_level = self.context.submission_levels[old_level_id]
            new_level = self.context.submission_levels[new_level_id]
            
            # Record level down event in relationship manager if available
            if self.context.relationship_manager:
                try:
                    await self.context.relationship_manager.update_relationship_attribute(
                        user_id,
                        "submission_level",
                        new_level_id
                    )
                    
                    await self.context.relationship_manager.update_relationship_attribute(
                        user_id,
                        "submission_score",
                        score
                    )
                except Exception as e:
                    logger.error(f"Error updating relationship data: {e}")
            
            # Send negative reward for level down
            if self.context.reward_system:
                try:
                    level_down_penalty = -0.2 - (old_level_id * 0.05)
                    
                    await self.context.reward_system.process_reward_signal(
                        self.context.reward_system.RewardSignal(
                            value=level_down_penalty,
                            source="submission_level_down",
                            context={
                                "old_level": old_level_id,
                                "old_level_name": old_level.name,
                                "new_level": new_level_id,
                                "new_level_name": new_level.name,
                                "submission_score": score
                            }
                        )
                    )
                except Exception as e:
                    logger.error(f"Error processing level down penalty: {e}")
            
            logger.info(f"User {user_id} fell from submission level {old_level_id} to {new_level_id}")
            
            return True, LevelChangeDetails(
                change_type="level_down",
                old_level=LevelData(
                    id=old_level_id,
                    name=old_level.name
                ),
                new_level=LevelData(
                    id=new_level_id,
                    name=new_level.name,
                    description=new_level.description,
                    privileges=new_level.privileges,
                    restrictions=new_level.restrictions,
                    training_focus=new_level.training_focus
                )
            )
        
        return False, LevelChangeDetails()
    
    @function_tool
    async def get_user_submission_data(self, ctx, user_id: str, include_history: bool = False) -> UserSubmissionDataResult:
        """
        Get the current submission data for a user.
        
        Args:
            user_id: The user to get data for
            include_history: Whether to include compliance history
            
        Returns:
            Complete submission data for the user
        """
        # Ensure user exists
        if user_id not in self.context.user_data:
            await self.initialize_user(ctx, user_id)
        
        user_data = self.context.user_data[user_id]
        level = self.context.submission_levels[user_data.current_level_id]
        
        # Check if level fits current score
        level_appropriate = (
            level.min_score <= user_data.total_submission_score <= level.max_score
        )
        
        # Calculate time at current level
        if user_data.last_level_change:
            days_at_level = (datetime.datetime.now() - user_data.last_level_change).days
            user_data.time_at_current_level = days_at_level
        
        # Calculate trait and requirement gaps for current level
        trait_gaps_dict = {}
        # Convert TraitsData to dict for processing
        level_traits_dict = {}
        if level.traits.obedience is not None:
            level_traits_dict["obedience"] = level.traits.obedience
        if level.traits.consistency is not None:
            level_traits_dict["consistency"] = level.traits.consistency
        if level.traits.initiative is not None:
            level_traits_dict["initiative"] = level.traits.initiative
        if level.traits.depth is not None:
            level_traits_dict["depth"] = level.traits.depth
        if level.traits.protocol_adherence is not None:
            level_traits_dict["protocol_adherence"] = level.traits.protocol_adherence
        if level.traits.receptiveness is not None:
            level_traits_dict["receptiveness"] = level.traits.receptiveness
        if level.traits.endurance is not None:
            level_traits_dict["endurance"] = level.traits.endurance
        if level.traits.attentiveness is not None:
            level_traits_dict["attentiveness"] = level.traits.attentiveness
        if level.traits.surrender is not None:
            level_traits_dict["surrender"] = level.traits.surrender
        if level.traits.reverence is not None:
            level_traits_dict["reverence"] = level.traits.reverence
            
        for trait_name, expected_value in level_traits_dict.items():
            if trait_name in user_data.obedience_metrics:
                current_value = user_data.obedience_metrics[trait_name].value
                gap = expected_value - current_value
                if gap > 0.1:  # Only report significant gaps
                    trait_gaps_dict[trait_name] = MetricUpdateData(
                        old_value=expected_value,
                        new_value=current_value,
                        change=gap
                    )
        
        trait_gaps = TraitGapsData(
            obedience=trait_gaps_dict.get("obedience"),
            consistency=trait_gaps_dict.get("consistency"),
            initiative=trait_gaps_dict.get("initiative"),
            depth=trait_gaps_dict.get("depth"),
            protocol_adherence=trait_gaps_dict.get("protocol_adherence"),
            receptiveness=trait_gaps_dict.get("receptiveness"),
            endurance=trait_gaps_dict.get("endurance"),
            attentiveness=trait_gaps_dict.get("attentiveness"),
            surrender=trait_gaps_dict.get("surrender"),
            reverence=trait_gaps_dict.get("reverence")
        )
        
        # Format metrics
        metrics_dict = {}
        for name, metric in user_data.obedience_metrics.items():
            metrics_dict[name] = MetricData(
                value=metric.value,
                weight=metric.weight,
                last_updated=metric.last_updated.isoformat()
            )
        
        metrics = AllMetricsData(
            obedience=metrics_dict.get("obedience"),
            consistency=metrics_dict.get("consistency"),
            initiative=metrics_dict.get("initiative"),
            depth=metrics_dict.get("depth"),
            protocol_adherence=metrics_dict.get("protocol_adherence"),
            receptiveness=metrics_dict.get("receptiveness"),
            endurance=metrics_dict.get("endurance"),
            attentiveness=metrics_dict.get("attentiveness"),
            surrender=metrics_dict.get("surrender"),
            reverence=metrics_dict.get("reverence")
        )
        
        # Assemble result
        result = UserSubmissionDataResult(
            user_id=user_id,
            submission_level=LevelData(
                id=level.id,
                name=level.name,
                description=level.description,
                appropriate=level_appropriate,
                time_at_level_days=user_data.time_at_current_level
            ),
            submission_score=user_data.total_submission_score,
            metrics=metrics,
            trait_gaps=trait_gaps,
            privileges=level.privileges,
            restrictions=level.restrictions,
            training_focus=level.training_focus,
            compliance_rate=user_data.lifetime_compliance_rate,
            last_updated=user_data.last_updated.isoformat()
        )
        
        # Include history if requested
        if include_history:
            # Format compliance history
            history = []
            for record in user_data.compliance_history:
                history.append(ComplianceHistoryData(
                    id=record.id,
                    timestamp=record.timestamp.isoformat(),
                    instruction=record.instruction,
                    complied=record.complied,
                    difficulty=record.difficulty,
                    defiance_reason=record.defiance_reason
                ))
            
            result.compliance_history = history
        
        return result
    
    @function_tool
    async def generate_progression_report(self, ctx, user_id: str) -> ProgressionReportResult:
        """
        Generate a detailed report on user's submission progression.
        
        Args:
            user_id: The user to generate a report for
            
        Returns:
            Comprehensive progression report
        """
        # Ensure user exists
        if user_id not in self.context.user_data:
            await self.initialize_user(ctx, user_id)
        
        user_data = self.context.user_data[user_id]
        level = self.context.submission_levels[user_data.current_level_id]
        
        # Calculate progress within current level
        level_range = level.max_score - level.min_score
        if level_range <= 0:
            level_progress = 1.0  # Avoid division by zero
        else:
            level_progress = (user_data.total_submission_score - level.min_score) / level_range
            level_progress = max(0.0, min(1.0, level_progress))
        
        # Calculate compliance trends
        compliance_trend = "stable"
        if len(user_data.compliance_history) >= 10:
            # Compare last 5 with previous 5
            recent_5 = user_data.compliance_history[-5:]
            previous_5 = user_data.compliance_history[-10:-5]
            
            recent_rate = sum(1 for r in recent_5 if r.complied) / 5
            previous_rate = sum(1 for r in previous_5 if r.complied) / 5
            
            if recent_rate > previous_rate + 0.1:
                compliance_trend = "improving"
            elif recent_rate < previous_rate - 0.1:
                compliance_trend = "declining"
        
        # Calculate metric changes over time
        metric_trends_dict = {}
        for metric_name, metric in user_data.obedience_metrics.items():
            if len(metric.history) >= 10:
                # Compare last 5 with previous 5
                recent_5 = metric.history[-5:]
                previous_5 = metric.history[-10:-5]
                
                recent_avg = sum(r.final_value for r in recent_5) / 5
                previous_avg = sum(r.final_value for r in previous_5) / 5
                
                change = recent_avg - previous_avg
                
                if change > 0.05:
                    trend = "improving"
                elif change < -0.05:
                    trend = "declining"
                else:
                    trend = "stable"
                
                metric_trends_dict[metric_name] = MetricTrend(
                    trend=trend,
                    change=change,
                    recent_avg=recent_avg,
                    previous_avg=previous_avg
                )
        
        metric_trends = MetricTrendsData(
            obedience=metric_trends_dict.get("obedience"),
            consistency=metric_trends_dict.get("consistency"),
            initiative=metric_trends_dict.get("initiative"),
            depth=metric_trends_dict.get("depth"),
            protocol_adherence=metric_trends_dict.get("protocol_adherence"),
            receptiveness=metric_trends_dict.get("receptiveness"),
            endurance=metric_trends_dict.get("endurance"),
            attentiveness=metric_trends_dict.get("attentiveness"),
            surrender=metric_trends_dict.get("surrender"),
            reverence=metric_trends_dict.get("reverence")
        )
        
        # Generate recommendations based on metric gaps
        recommendations = []
        
        # Convert TraitsData to dict for processing
        level_traits_dict = {}
        if level.traits.obedience is not None:
            level_traits_dict["obedience"] = level.traits.obedience
        if level.traits.consistency is not None:
            level_traits_dict["consistency"] = level.traits.consistency
        if level.traits.initiative is not None:
            level_traits_dict["initiative"] = level.traits.initiative
        if level.traits.depth is not None:
            level_traits_dict["depth"] = level.traits.depth
        if level.traits.protocol_adherence is not None:
            level_traits_dict["protocol_adherence"] = level.traits.protocol_adherence
        if level.traits.receptiveness is not None:
            level_traits_dict["receptiveness"] = level.traits.receptiveness
        if level.traits.endurance is not None:
            level_traits_dict["endurance"] = level.traits.endurance
        if level.traits.attentiveness is not None:
            level_traits_dict["attentiveness"] = level.traits.attentiveness
        if level.traits.surrender is not None:
            level_traits_dict["surrender"] = level.traits.surrender
        if level.traits.reverence is not None:
            level_traits_dict["reverence"] = level.traits.reverence
            
        for trait_name, expected in level_traits_dict.items():
            if trait_name in user_data.obedience_metrics:
                current = user_data.obedience_metrics[trait_name].value
                gap = expected - current
                
                if gap > 0.2:
                    recommendations.append(RecommendationData(
                        focus_area=trait_name,
                        significance=gap,
                        current_value=current,
                        target_value=expected,
                        description=f"Increase {trait_name} through targeted training and practice"
                    ))
        
        # Sort recommendations by significance
        recommendations.sort(key=lambda x: x.significance, reverse=True)
        
        # Format report
        progression_path_data = ProgressionPathData(
            next_level=None,
            requirements_for_advancement=[],
            estimated_time_to_next_level=None
        )
        
        # Add next level info if not at max level
        if user_data.current_level_id < max(self.context.submission_levels.keys()):
            next_level = self.context.submission_levels[user_data.current_level_id + 1]
            
            # Calculate requirements for advancement
            requirements = []
            score_gap = next_level.min_score - user_data.total_submission_score
            if score_gap > 0:
                requirements.append(AdvancementRequirement(
                    type="score",
                    description=f"Increase overall submission score by {score_gap:.2f}",
                    current=user_data.total_submission_score,
                    target=next_level.min_score
                ))
            
            # Calculate trait requirements
            next_level_traits_dict = {}
            if next_level.traits.obedience is not None:
                next_level_traits_dict["obedience"] = next_level.traits.obedience
            if next_level.traits.consistency is not None:
                next_level_traits_dict["consistency"] = next_level.traits.consistency
            if next_level.traits.initiative is not None:
                next_level_traits_dict["initiative"] = next_level.traits.initiative
            if next_level.traits.depth is not None:
                next_level_traits_dict["depth"] = next_level.traits.depth
            if next_level.traits.protocol_adherence is not None:
                next_level_traits_dict["protocol_adherence"] = next_level.traits.protocol_adherence
            if next_level.traits.receptiveness is not None:
                next_level_traits_dict["receptiveness"] = next_level.traits.receptiveness
            if next_level.traits.endurance is not None:
                next_level_traits_dict["endurance"] = next_level.traits.endurance
            if next_level.traits.attentiveness is not None:
                next_level_traits_dict["attentiveness"] = next_level.traits.attentiveness
            if next_level.traits.surrender is not None:
                next_level_traits_dict["surrender"] = next_level.traits.surrender
            if next_level.traits.reverence is not None:
                next_level_traits_dict["reverence"] = next_level.traits.reverence
                
            for trait_name, expected in next_level_traits_dict.items():
                if trait_name in user_data.obedience_metrics:
                    current = user_data.obedience_metrics[trait_name].value
                    gap = expected - current
                    
                    if gap > 0.1:
                        requirements.append(AdvancementRequirement(
                            type="trait",
                            trait=trait_name,
                            description=f"Increase {trait_name} from {current:.2f} to {expected:.2f}",
                            current=current,
                            target=expected
                        ))
            
            # Estimate time to next level based on recent progress rate
            estimated_days = None
            if level_progress > 0.2 and user_data.time_at_current_level > 0:
                # Estimate based on current progress rate
                progress_rate_per_day = level_progress / user_data.time_at_current_level
                if progress_rate_per_day > 0:
                    remaining_progress = 1.0 - level_progress
                    estimated_days = math.ceil(remaining_progress / progress_rate_per_day)
            
            progression_path_data = ProgressionPathData(
                next_level=NextLevelData(
                    id=next_level.id,
                    name=next_level.name,
                    description=next_level.description
                ),
                requirements_for_advancement=requirements,
                estimated_time_to_next_level=estimated_days
            )
        
        return ProgressionReportResult(
            user_id=user_id,
            generation_time=datetime.datetime.now().isoformat(),
            current_level=LevelData(
                id=level.id,
                name=level.name,
                description=level.description,
                progress_in_level=level_progress,
                time_at_level_days=user_data.time_at_current_level
            ),
            submission_metrics=SubmissionMetricsData(
                overall_score=user_data.total_submission_score,
                compliance_rate=user_data.lifetime_compliance_rate,
                compliance_trend=compliance_trend,
                metric_trends=metric_trends
            ),
            progression_path=progression_path_data,
            recommendations=recommendations[:3]  # Top 3 recommendations
        )
    
    # Public API for external components to use
    
    async def process_request(self, user_id: str, request_text: str) -> Dict[str, Any]:
        """
        Process a submission-related request through the agent system.
        This is the main entry point for external components.
        
        Args:
            user_id: The user making the request
            request_text: The text of the request
            
        Returns:
            Response from the appropriate agent
        """
        # Use tracing for debugging and visualization
        with trace(workflow_name="submission_progression", group_id=user_id):
            input_data = f"User ID: {user_id}\nRequest: {request_text}"
            
            # Run the triage agent with the request
            result = await Runner.run(
                self.triage_agent,
                input_data,
                context=self.context
            )
            
            # Log the request and response
            logger.info(f"Processed submission request for user {user_id}: {request_text[:50]}...")
            
            return {
                "user_id": user_id,
                "request": request_text,
                "response": result.final_output,
                "agent_used": result.last_agent.name
            }
