# nyx/femdom/submission_progression.py

import logging
import datetime
import asyncio
import uuid
import math
from typing import Dict, List, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class SubmissionLevel(BaseModel):
    """Represents a level in the submission progression system."""
    id: int
    name: str
    description: str
    min_score: float  # Minimum submission score required
    max_score: float  # Maximum submission score for this level
    traits: Dict[str, float] = Field(default_factory=dict)  # Expected traits at this level
    privileges: List[str] = Field(default_factory=list)
    restrictions: List[str] = Field(default_factory=list)
    training_focus: List[str] = Field(default_factory=list)

class DominancePath(BaseModel):
    """Represents a specific path or style of dominance training."""
    id: str
    name: str
    description: str
    focus_areas: List[str] = Field(default_factory=list)
    recommended_metrics: List[str] = Field(default_factory=list)
    difficulty: float = Field(0.5, ge=0.0, le=1.0)
    suitable_for_traits: Dict[str, float] = Field(default_factory=dict)
    progression_milestones: Dict[int, Dict[str, Any]] = Field(default_factory=dict)

class ProgressionMilestone(BaseModel):
    """Represents a specific milestone in submission progression."""
    id: str
    level: int
    name: str
    description: str
    requirements: Dict[str, float] = Field(default_factory=dict)  # metric -> threshold
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
    history: List[Dict[str, Any]] = Field(default_factory=list)
    
    def update(self, new_value: float, reason: str = "general"):
        """Update the metric with a new value."""
        # Calculate weighted average between old and new (30% new, 70% old)
        self.value = (self.value * 0.7) + (new_value * 0.3)
        self.value = max(0.0, min(1.0, self.value))  # Constrain to valid range
        self.last_updated = datetime.datetime.now()
        
        # Track history (limited to last 20 entries)
        self.history.append({
            "timestamp": self.last_updated.isoformat(),
            "old_value": self.value,
            "new_value": new_value,
            "final_value": self.value,
            "reason": reason
        })
        
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
    context: Dict[str, Any] = Field(default_factory=dict)
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
    limits: Dict[str, List[str]] = Field(default_factory=dict)
    preferences: Dict[str, float] = Field(default_factory=dict)
    last_level_change: Optional[datetime.datetime] = None
    lifetime_compliance_rate: float = 0.5
    last_updated: datetime.datetime = Field(default_factory=datetime.datetime.now)

class SubmissionProgression:
    """Tracks user's submission journey and training progress."""
    
    def __init__(self, reward_system=None, memory_core=None, relationship_manager=None):
        self.reward_system = reward_system
        self.memory_core = memory_core
        self.relationship_manager = relationship_manager
        
        # Submission levels
        self.submission_levels: Dict[int, SubmissionLevel] = {}
        
        # User data
        self.user_data: Dict[str, UserSubmissionData] = {}
        
        # Default metrics to track for each user
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
        
        # Metric weights (importance of each metric)
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
        
        # Initialize submission levels
        self._init_submission_levels()
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("SubmissionProgression system initialized")
    
    def _init_submission_levels(self):
        """Initialize the submission levels."""
        levels = [
            SubmissionLevel(
                id=1,
                name="Curious",
                description="Initial exploration of submission",
                min_score=0.0,
                max_score=0.2,
                traits={"obedience": 0.1, "receptiveness": 0.2},
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
                traits={"obedience": 0.3, "receptiveness": 0.4, "protocol_adherence": 0.3},
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
                traits={
                    "obedience": 0.6, 
                    "consistency": 0.5, 
                    "protocol_adherence": 0.6,
                    "receptiveness": 0.6
                },
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
                traits={
                    "obedience": 0.8, 
                    "consistency": 0.7, 
                    "protocol_adherence": 0.8,
                    "depth": 0.7,
                    "surrender": 0.6,
                    "reverence": 0.7
                },
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
                traits={
                    "obedience": 0.9, 
                    "consistency": 0.9, 
                    "protocol_adherence": 0.9,
                    "depth": 0.9,
                    "surrender": 0.9,
                    "reverence": 0.9,
                    "initiative": 0.8,
                    "attentiveness": 0.8,
                    "endurance": 0.8
                },
                privileges=["Considerable trust", "Deep connection"],
                restrictions=["High protocol at all times", "Total control in designated areas"],
                training_focus=["Perfection in service", "Total surrender", "Deep psychological control"]
            )
        ]
        
        for level in levels:
            self.submission_levels[level.id] = level
    
    async def initialize_user(self, user_id: str, initial_data: Optional[Dict[str, Any]] = None) -> UserSubmissionData:
        """Initialize or get user submission data."""
        async with self._lock:
            if user_id in self.user_data:
                return self.user_data[user_id]
            
            # Create new user data
            user_data = UserSubmissionData(user_id=user_id)
            
            # Initialize metrics
            for metric_name in self.default_metrics:
                weight = self.metric_weights.get(metric_name, 1.0)
                user_data.obedience_metrics[metric_name] = SubmissionMetric(
                    name=metric_name,
                    value=0.1,  # Start at minimal value
                    weight=weight
                )
            
            # Add new fields
            user_data.assigned_path = None
            user_data.assigned_path_date = None
            user_data.milestones = {}
            user_data.unlocked_features = []
            
            # Apply any initial data if provided
            if initial_data:
                if "limits" in initial_data:
                    user_data.limits = initial_data["limits"]
                
                if "preferences" in initial_data:
                    user_data.preferences = initial_data["preferences"]
                
                # Set initial metrics if provided
                if "metrics" in initial_data:
                    for metric_name, value in initial_data["metrics"].items():
                        if metric_name in user_data.obedience_metrics:
                            user_data.obedience_metrics[metric_name].value = value
                
                # Set initial level if provided
                if "level" in initial_data:
                    level_id = initial_data["level"]
                    if level_id in self.submission_levels:
                        user_data.current_level_id = level_id
                        
                # Set initial path if provided
                if "path" in initial_data:
                    path_id = initial_data["path"]
                    if hasattr(self, "dominance_paths") and path_id in self.dominance_paths:
                        user_data.assigned_path = path_id
                        user_data.assigned_path_date = datetime.datetime.now()
            
            # Calculate initial submission score
            user_data.total_submission_score = self._calculate_submission_score(user_data)
            
            # Set last level change to now
            user_data.last_level_change = datetime.datetime.now()
            
            # Store user data
            self.user_data[user_id] = user_data
            
            logger.info(f"Initialized submission tracking for user {user_id} at level {user_data.current_level_id}")
            
            return user_data
    
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

    async def initialize_dominance_paths(self):
        """Initialize available dominance paths."""
        self.dominance_paths = {
            "service": DominancePath(
                id="service",
                name="Service-Oriented Path",
                description="Focus on developing service skills, protocols, and rituals.",
                focus_areas=["protocol_adherence", "service", "consistency", "attentiveness"],
                recommended_metrics=["obedience", "consistency", "protocol_adherence", "attentiveness"],
                difficulty=0.4,
                suitable_for_traits={"service_oriented": 0.7, "detail_oriented": 0.6, "methodical": 0.5},
                progression_milestones={
                    1: {
                        "id": "basic_protocols",
                        "name": "Basic Protocols",
                        "description": "Learn and consistently follow basic service protocols.",
                        "requirements": {"protocol_adherence": 0.3, "consistency": 0.3},
                        "rewards": ["basic_protocol_certification"],
                        "unlocks": ["intermediate_service_tasks"]
                    },
                    2: {
                        "id": "consistent_service",
                        "name": "Consistent Service",
                        "description": "Demonstrate reliable service across multiple sessions.",
                        "requirements": {"consistency": 0.5, "obedience": 0.4, "protocol_adherence": 0.5},
                        "rewards": ["service_recognition"],
                        "unlocks": ["advanced_service_positions"]
                    },
                    3: {
                        "id": "anticipatory_service",
                        "name": "Anticipatory Service",
                        "description": "Anticipate needs and provide service without explicit direction.",
                        "requirements": {"attentiveness": 0.7, "initiative": 0.6},
                        "rewards": ["service_excellence_badge"],
                        "unlocks": ["ritual_creation_privileges"]
                    }
                }
            ),
            "psychological": DominancePath(
                id="psychological",
                name="Psychological Dominance Path",
                description="Focus on mind games, control, and psychological surrender.",
                focus_areas=["depth", "surrender", "receptiveness"],
                recommended_metrics=["depth", "surrender", "receptiveness"],
                difficulty=0.7,
                suitable_for_traits={"analytical": 0.6, "introspective": 0.7, "emotionally_sensitive": 0.5},
                progression_milestones={
                    1: {
                        "id": "mental_submission",
                        "name": "Mental Submission Basics",
                        "description": "Begin surrendering control mentally and emotionally.",
                        "requirements": {"surrender": 0.3, "receptiveness": 0.4},
                        "rewards": ["mind_control_session"],
                        "unlocks": ["light_mindfuck_techniques"]
                    },
                    2: {
                        "id": "psychological_surrender",
                        "name": "Psychological Surrender",
                        "description": "Deeper mental submission and acceptance of control.",
                        "requirements": {"depth": 0.5, "surrender": 0.6},
                        "rewards": ["psychological_dominance_session"],
                        "unlocks": ["advanced_mindfuck_techniques"]
                    },
                    3: {
                        "id": "cognitive_restructuring",
                        "name": "Cognitive Restructuring",
                        "description": "Allow thought patterns to be influenced and restructured.",
                        "requirements": {"depth": 0.7, "surrender": 0.8, "receptiveness": 0.7},
                        "rewards": ["deep_control_badge"],
                        "unlocks": ["permission_structures", "thought_control_protocols"]
                    }
                }
            ),
            "humiliation": DominancePath(
                id="humiliation",
                name="Humiliation-Focused Path",
                description="Focus on embarrassment, degradation, and humbling experiences.",
                focus_areas=["surrender", "endurance", "receptiveness"],
                recommended_metrics=["surrender", "endurance", "receptiveness"],
                difficulty=0.8,
                suitable_for_traits={"masochistic": 0.6, "exhibitionist": 0.5, "shame_responsive": 0.7},
                progression_milestones={
                    1: {
                        "id": "light_embarrassment",
                        "name": "Light Embarrassment",
                        "description": "Introduction to light embarrassment and verbal humiliation.",
                        "requirements": {"surrender": 0.3, "endurance": 0.3},
                        "rewards": ["humiliation_beginner_badge"],
                        "unlocks": ["moderate_humiliation_tasks"]
                    },
                    2: {
                        "id": "moderate_humiliation",
                        "name": "Moderate Humiliation",
                        "description": "Acceptance of regular humiliation and embarrassment.",
                        "requirements": {"surrender": 0.5, "endurance": 0.6},
                        "rewards": ["humiliation_intermediate_badge"],
                        "unlocks": ["embarrassment_challenges"]
                    },
                    3: {
                        "id": "deep_degradation",
                        "name": "Deep Degradation",
                        "description": "Acceptance of profound humiliation and degradation.",
                        "requirements": {"surrender": 0.8, "endurance": 0.7, "receptiveness": 0.7},
                        "rewards": ["humiliation_advanced_badge"],
                        "unlocks": ["custom_humiliation_scenarios"]
                    }
                }
            ),
            "strict_discipline": DominancePath(
                id="strict_discipline",
                name="Strict Discipline Path",
                description="Focus on rules, punishment, and strict behavioral standards.",
                focus_areas=["obedience", "discipline", "protocol_adherence"],
                recommended_metrics=["obedience", "endurance", "protocol_adherence"],
                difficulty=0.6,
                suitable_for_traits={"structure_seeking": 0.7, "rule_oriented": 0.6, "discipline_responsive": 0.7},
                progression_milestones={
                    1: {
                        "id": "basic_rules",
                        "name": "Basic Rules Adherence",
                        "description": "Consistent following of basic rules and acceptance of correction.",
                        "requirements": {"obedience": 0.4, "protocol_adherence": 0.3},
                        "rewards": ["discipline_beginner_badge"],
                        "unlocks": ["intermediate_rule_structures"]
                    },
                    2: {
                        "id": "punishment_acceptance",
                        "name": "Punishment Acceptance",
                        "description": "Full acceptance of punishments and corrections.",
                        "requirements": {"obedience": 0.6, "endurance": 0.5, "receptiveness": 0.5},
                        "rewards": ["discipline_intermediate_badge"],
                        "unlocks": ["advanced_punishment_protocols"]
                    },
                    3: {
                        "id": "internalized_discipline",
                        "name": "Internalized Discipline",
                        "description": "Self-monitoring and anticipatory obedience.",
                        "requirements": {"obedience": 0.8, "protocol_adherence": 0.7, "initiative": 0.6},
                        "rewards": ["discipline_advanced_badge"],
                        "unlocks": ["self_discipline_protocols", "punishment_authority"]
                    }
                }
            )
        }
        
        logger.info(f"Initialized {len(self.dominance_paths)} dominance progression paths")
        return self.dominance_paths
    
    
    # Add this method to the SubmissionProgression class
    
    async def recommend_dominance_path(self, user_id: str) -> Dict[str, Any]:
        """
        Recommends the most suitable dominance path based on user traits and preferences.
        
        Args:
            user_id: The user to analyze
            
        Returns:
            Recommendation details
        """
        if user_id not in self.user_data:
            await self.initialize_user(user_id)
        
        user_data = self.user_data[user_id]
        
        # Get user traits and preferences
        user_traits = {}
        user_preferences = {}
        
        # Try to get from relationship manager if available
        if self.relationship_manager:
            try:
                relationship = await self.relationship_manager.get_relationship_state(user_id)
                if hasattr(relationship, "inferred_user_traits"):
                    user_traits = relationship.inferred_user_traits
                if hasattr(relationship, "preferences"):
                    user_preferences = relationship.preferences
            except Exception as e:
                logger.error(f"Error getting relationship data: {e}")
        
        # Calculate match scores for each path
        path_scores = {}
        for path_id, path in self.dominance_paths.items():
            # Base score
            score = 0.0
            
            # Match based on traits
            trait_match_score = 0.0
            trait_count = 0
            for trait, required_value in path.suitable_for_traits.items():
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
            path = self.dominance_paths[path_id]
            recommendations.append({
                "id": path_id,
                "name": path.name,
                "description": path.description,
                "match_score": score,
                "difficulty": path.difficulty,
                "focus_areas": path.focus_areas
            })
        
        # Return recommendations
        return {
            "user_id": user_id,
            "primary_recommendation": recommendations[0] if recommendations else None,
            "all_recommendations": recommendations,
            "user_traits_analyzed": list(user_traits.keys()),
            "analysis_timestamp": datetime.datetime.now().isoformat()
        }
    
    
    # Add this method to the SubmissionProgression class
    
    async def assign_dominance_path(self, user_id: str, path_id: str) -> Dict[str, Any]:
        """
        Assigns a specific dominance path to a user.
        
        Args:
            user_id: The user to assign the path to
            path_id: The path ID to assign
            
        Returns:
            Assignment details
        """
        if path_id not in self.dominance_paths:
            return {
                "success": False,
                "message": f"Path '{path_id}' not found",
                "available_paths": list(self.dominance_paths.keys())
            }
            
        if user_id not in self.user_data:
            await self.initialize_user(user_id)
        
        user_data = self.user_data[user_id]
        
        # Assign path
        user_data.assigned_path = path_id
        user_data.assigned_path_date = datetime.datetime.now()
        
        # Initialize milestone tracking if not exists
        if not hasattr(user_data, "milestones"):
            user_data.milestones = {}
        
        # Initialize milestones for this path
        path = self.dominance_paths[path_id]
        for level, milestone_data in path.progression_milestones.items():
            milestone_id = milestone_data["id"]
            if milestone_id not in user_data.milestones:
                user_data.milestones[milestone_id] = ProgressionMilestone(
                    id=milestone_id,
                    level=level,
                    name=milestone_data["name"],
                    description=milestone_data["description"],
                    requirements=milestone_data["requirements"],
                    rewards=milestone_data["rewards"],
                    unlocks=milestone_data["unlocks"],
                    completed=False
                )
        
        # Record in relationship manager if available
        if self.relationship_manager:
            try:
                await self.relationship_manager.update_relationship_attribute(
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
        
        return {
            "success": True,
            "user_id": user_id,
            "path_id": path_id,
            "path_name": path.name,
            "difficulty": path.difficulty,
            "focus_areas": path.focus_areas,
            "milestones": len(path.progression_milestones),
            "message": f"Assigned dominance path '{path.name}' to user"
        }
    
    
    # Add this method to the SubmissionProgression class
    
    async def check_milestone_progress(self, user_id: str) -> Dict[str, Any]:
        """
        Checks user progress against assigned path milestones.
        
        Args:
            user_id: The user to check
            
        Returns:
            Milestone progress details
        """
        if user_id not in self.user_data:
            return {
                "success": False,
                "message": f"No submission data found for user {user_id}"
            }
            
        user_data = self.user_data[user_id]
        
        # Check if user has an assigned path
        if not hasattr(user_data, "assigned_path") or not user_data.assigned_path:
            return {
                "success": False,
                "message": "No dominance path assigned to user",
                "recommendation": "Use assign_dominance_path to assign a path"
            }
        
        path_id = user_data.assigned_path
        if path_id not in self.dominance_paths:
            return {
                "success": False,
                "message": f"Assigned path '{path_id}' not found"
            }
            
        path = self.dominance_paths[path_id]
        
        # Ensure milestones are initialized
        if not hasattr(user_data, "milestones"):
            user_data.milestones = {}
            
        # Check each milestone for completion
        newly_completed = []
        upcoming_milestones = []
        already_completed = []
        
        for level, milestone_data in path.progression_milestones.items():
            milestone_id = milestone_data["id"]
            
            # Create milestone if it doesn't exist
            if milestone_id not in user_data.milestones:
                user_data.milestones[milestone_id] = ProgressionMilestone(
                    id=milestone_id,
                    level=level,
                    name=milestone_data["name"],
                    description=milestone_data["description"],
                    requirements=milestone_data["requirements"],
                    rewards=milestone_data["rewards"],
                    unlocks=milestone_data["unlocks"],
                    completed=False
                )
                
            milestone = user_data.milestones[milestone_id]
            
            # Skip if already completed
            if milestone.completed:
                already_completed.append({
                    "id": milestone_id,
                    "name": milestone.name,
                    "level": milestone.level,
                    "completion_date": milestone.completion_date.isoformat() if milestone.completion_date else None
                })
                continue
                
            # Check requirements
            all_requirements_met = True
            requirement_progress = {}
            
            for metric_name, threshold in milestone.requirements.items():
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
                
                newly_completed.append({
                    "id": milestone_id,
                    "name": milestone.name,
                    "level": milestone.level,
                    "rewards": milestone.rewards,
                    "unlocks": milestone.unlocks
                })
            else:
                # Add to upcoming
                upcoming_milestones.append({
                    "id": milestone_id,
                    "name": milestone.name,
                    "level": milestone.level,
                    "description": milestone.description,
                    "requirements": requirement_progress,
                    "overall_progress": sum(r["progress_percentage"] for r in requirement_progress.values()) / len(requirement_progress)
                })
        
        # Sort upcoming milestones by progress
        upcoming_milestones.sort(key=lambda m: m["overall_progress"], reverse=True)
        
        # Apply rewards for newly completed milestones
        for milestone in newly_completed:
            # Create reward signal if available
            if self.reward_system:
                try:
                    # Higher reward for higher-level milestones
                    reward_value = 0.3 + (milestone["level"] * 0.15)  # 0.45 for level 1, 0.6 for level 2, etc.
                    
                    await self.reward_system.process_reward_signal(
                        self.reward_system.RewardSignal(
                            value=reward_value,
                            source="dominance_milestone",
                            context={
                                "milestone_id": milestone["id"],
                                "milestone_name": milestone["name"],
                                "level": milestone["level"],
                                "path_id": path_id,
                                "path_name": path.name
                            }
                        )
                    )
                except Exception as e:
                    logger.error(f"Error processing reward: {e}")
        
        # Update relationship manager if available and milestones were completed
        if newly_completed and self.relationship_manager:
            try:
                await self.relationship_manager.update_relationship_attribute(
                    user_id,
                    "completed_milestones",
                    [m["id"] for m in already_completed + newly_completed]
                )
            except Exception as e:
                logger.error(f"Error updating relationship data: {e}")
        
        return {
            "success": True,
            "user_id": user_id,
            "path_id": path_id,
            "path_name": path.name,
            "newly_completed_milestones": newly_completed,
            "upcoming_milestones": upcoming_milestones[:3],  # Top 3 upcoming
            "already_completed_milestones": already_completed,
            "progress_summary": {
                "total_milestones": len(path.progression_milestones),
                "completed": len(already_completed) + len(newly_completed),
                "completion_percentage": ((len(already_completed) + len(newly_completed)) / len(path.progression_milestones)) * 100
            }
        }
    
    
    # Add this method to the SubmissionProgression class
    
    async def get_available_unlocks(self, user_id: str) -> Dict[str, Any]:
        """
        Gets all features, tasks, privileges unlocked by the user through milestone completion.
        
        Args:
            user_id: The user to check
            
        Returns:
            List of unlocked features
        """
        if user_id not in self.user_data:
            return {
                "success": False,
                "message": f"No submission data found for user {user_id}"
            }
            
        user_data = self.user_data[user_id]
        
        # Check if user has milestones
        if not hasattr(user_data, "milestones"):
            return {
                "success": True,
                "unlocks": [],
                "message": "No milestones completed yet"
            }
        
        # Collect all unlocks from completed milestones
        all_unlocks = []
        unique_unlocks = set()
        
        for milestone_id, milestone in user_data.milestones.items():
            if milestone.completed:
                for unlock in milestone.unlocks:
                    if unlock not in unique_unlocks:
                        unique_unlocks.add(unlock)
                        all_unlocks.append({
                            "id": unlock,
                            "from_milestone": milestone.name,
                            "level": milestone.level,
                            "unlocked_on": milestone.completion_date.isoformat() if milestone.completion_date else "unknown"
                        })
        
        # Sort by level (highest first)
        all_unlocks.sort(key=lambda u: u["level"], reverse=True)
        
        return {
            "success": True,
            "user_id": user_id,
            "unlocks": all_unlocks,
            "total_unlocks": len(all_unlocks)
        }
    
    async def record_compliance(self, 
                              user_id: str, 
                              instruction: str, 
                              complied: bool, 
                              difficulty: float = 0.5,
                              context: Optional[Dict[str, Any]] = None,
                              defiance_reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Record compliance or defiance for a specific instruction.
        
        Args:
            user_id: User ID
            instruction: The instruction given
            complied: Whether user complied
            difficulty: How difficult the instruction was (0.0-1.0)
            context: Additional context about the instruction
            defiance_reason: If defied, the reason given
            
        Returns:
            Dict with results of the operation
        """
        # Ensure user exists
        if user_id not in self.user_data:
            await self.initialize_user(user_id)
        
        user_data = self.user_data[user_id]
        
        # Create compliance record
        record = ComplianceRecord(
            instruction=instruction,
            complied=complied,
            difficulty=difficulty,
            context=context or {},
            defiance_reason=defiance_reason
        )
        
        async with self._lock:
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
            metrics_updates = {}
            
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
                
                metrics_updates["obedience"] = {
                    "old_value": old_value,
                    "new_value": new_value,
                    "change": obedience_change
                }
            
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
                    
                    metrics_updates["consistency"] = {
                        "old_value": old_value,
                        "new_value": new_value,
                        "change": consistency_change
                    }
            
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
                    
                    metrics_updates["receptiveness"] = {
                        "old_value": old_value,
                        "new_value": new_value,
                        "change": receptiveness_change
                    }
            
            # Recalculate overall submission score
            old_score = user_data.total_submission_score
            user_data.total_submission_score = self._calculate_submission_score(user_data)
            
            # Update user's last_updated timestamp
            user_data.last_updated = datetime.datetime.now()
            
            # Evaluate for level change
            level_changed = False
            level_change_details = {}
            
            if old_score != user_data.total_submission_score:
                level_changed, level_change_details = await self._check_level_change(user_id)
            
            # Issue appropriate reward signal
            reward_result = None
            if self.reward_system:
                try:
                    # Base reward value on compliance and difficulty
                    base_reward = 0.3 if complied else -0.4
                    difficulty_modifier = difficulty * 0.4
                    reward_value = base_reward + (difficulty_modifier if complied else -difficulty_modifier)
                    
                    reward_result = await self.reward_system.process_reward_signal(
                        self.reward_system.RewardSignal(
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
                except Exception as e:
                    logger.error(f"Error processing reward: {e}")
            
            # Record memory if available
            if self.memory_core:
                try:
                    level_name = self.submission_levels[user_data.current_level_id].name
                    memory_content = (
                        f"User {'complied with' if complied else 'defied'} instruction: {instruction}. "
                        f"Submission level: {level_name} ({user_data.total_submission_score:.2f})"
                    )
                    
                    await self.memory_core.add_memory(
                        memory_type="experience",
                        content=memory_content,
                        tags=["compliance", "submission", 
                              "obedience" if complied else "defiance"],
                        significance=0.3 + (difficulty * 0.3) + (0.3 if level_changed else 0.0)
                    )
                except Exception as e:
                    logger.error(f"Error recording memory: {e}")
            
            return {
                "success": True,
                "record_id": record.id,
                "compliance_recorded": complied,
                "metrics_updated": metrics_updates,
                "submission_score": {
                    "old": old_score,
                    "new": user_data.total_submission_score,
                    "change": user_data.total_submission_score - old_score
                },
                "level_changed": level_changed,
                "level_change_details": level_change_details,
                "reward_result": reward_result
            }
    
    async def update_submission_metric(self, 
                                     user_id: str, 
                                     metric_name: str, 
                                     value_change: float,
                                     reason: str = "general") -> Dict[str, Any]:
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
        if user_id not in self.user_data:
            await self.initialize_user(user_id)
        
        user_data = self.user_data[user_id]
        
        # Check if metric exists
        if metric_name not in user_data.obedience_metrics:
            return {
                "success": False,
                "message": f"Metric '{metric_name}' not found for user"
            }
        
        async with self._lock:
            metric = user_data.obedience_metrics[metric_name]
            old_value = metric.value
            
            # Apply change
            new_raw_value = old_value + value_change
            new_raw_value = max(0.0, min(1.0, new_raw_value))
            
            # Update the metric
            metric.update(new_raw_value, reason=reason)
            
            # Recalculate overall submission score
            old_score = user_data.total_submission_score
            user_data.total_submission_score = self._calculate_submission_score(user_data)
            
            # Update user's last_updated timestamp
            user_data.last_updated = datetime.datetime.now()
            
            # Evaluate for level change
            level_changed = False
            level_change_details = {}
            
            if old_score != user_data.total_submission_score:
                level_changed, level_change_details = await self._check_level_change(user_id)
            
            return {
                "success": True,
                "metric": metric_name,
                "old_value": old_value,
                "new_value": metric.value,
                "change": metric.value - old_value,
                "submission_score": {
                    "old": old_score,
                    "new": user_data.total_submission_score,
                    "change": user_data.total_submission_score - old_score
                },
                "level_changed": level_changed,
                "level_change_details": level_change_details
            }
    
    async def _check_level_change(self, user_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if user should change submission levels based on score."""
        user_data = self.user_data[user_id]
        current_level = self.submission_levels[user_data.current_level_id]
        score = user_data.total_submission_score
        
        # Check if score outside current level bounds
        if score > current_level.max_score and user_data.current_level_id < max(self.submission_levels.keys()):
            # Level up
            new_level_id = user_data.current_level_id + 1
            old_level_id = user_data.current_level_id
            
            # Update user level
            user_data.current_level_id = new_level_id
            user_data.last_level_change = datetime.datetime.now()
            user_data.time_at_current_level = 0
            
            # Get level objects
            old_level = self.submission_levels[old_level_id]
            new_level = self.submission_levels[new_level_id]
            
            # Record level up event in relationship manager if available
            if self.relationship_manager:
                try:
                    await self.relationship_manager.update_relationship_attribute(
                        user_id,
                        "submission_level",
                        new_level_id
                    )
                    
                    await self.relationship_manager.update_relationship_attribute(
                        user_id,
                        "submission_score",
                        score
                    )
                except Exception as e:
                    logger.error(f"Error updating relationship data: {e}")
            
            # Send positive reward for level up
            if self.reward_system:
                try:
                    level_up_reward = 0.5 + (new_level_id * 0.1)  # Higher levels give better rewards
                    
                    await self.reward_system.process_reward_signal(
                        self.reward_system.RewardSignal(
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
            
            return True, {
                "change_type": "level_up",
                "old_level": {
                    "id": old_level_id,
                    "name": old_level.name
                },
                "new_level": {
                    "id": new_level_id,
                    "name": new_level.name,
                    "description": new_level.description,
                    "privileges": new_level.privileges,
                    "restrictions": new_level.restrictions,
                    "training_focus": new_level.training_focus
                }
            }
            
        elif score < current_level.min_score and user_data.current_level_id > min(self.submission_levels.keys()):
            # Level down
            new_level_id = user_data.current_level_id - 1
            old_level_id = user_data.current_level_id
            
            # Update user level
            user_data.current_level_id = new_level_id
            user_data.last_level_change = datetime.datetime.now()
            user_data.time_at_current_level = 0
            
            # Get level objects
            old_level = self.submission_levels[old_level_id]
            new_level = self.submission_levels[new_level_id]
            
            # Record level down event in relationship manager if available
            if self.relationship_manager:
                try:
                    await self.relationship_manager.update_relationship_attribute(
                        user_id,
                        "submission_level",
                        new_level_id
                    )
                    
                    await self.relationship_manager.update_relationship_attribute(
                        user_id,
                        "submission_score",
                        score
                    )
                except Exception as e:
                    logger.error(f"Error updating relationship data: {e}")
            
            # Send negative reward for level down
            if self.reward_system:
                try:
                    level_down_penalty = -0.2 - (old_level_id * 0.05)
                    
                    await self.reward_system.process_reward_signal(
                        self.reward_system.RewardSignal(
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
            
            return True, {
                "change_type": "level_down",
                "old_level": {
                    "id": old_level_id,
                    "name": old_level.name
                },
                "new_level": {
                    "id": new_level_id,
                    "name": new_level.name,
                    "description": new_level.description,
                    "privileges": new_level.privileges,
                    "restrictions": new_level.restrictions,
                    "training_focus": new_level.training_focus
                }
            }
        
        return False, {}
    
    async def get_user_submission_data(self, user_id: str, include_history: bool = False) -> Dict[str, Any]:
        """Get the current submission data for a user."""
        # Ensure user exists
        if user_id not in self.user_data:
            await self.initialize_user(user_id)
        
        user_data = self.user_data[user_id]
        level = self.submission_levels[user_data.current_level_id]
        
        # Check if level fits current score
        level_appropriate = (
            level.min_score <= user_data.total_submission_score <= level.max_score
        )
        
        # Calculate time at current level
        if user_data.last_level_change:
            days_at_level = (datetime.datetime.now() - user_data.last_level_change).days
            user_data.time_at_current_level = days_at_level
        
        # Calculate trait and requirement gaps for current level
        trait_gaps = {}
        for trait_name, expected_value in level.traits.items():
            if trait_name in user_data.obedience_metrics:
                current_value = user_data.obedience_metrics[trait_name].value
                gap = expected_value - current_value
                if gap > 0.1:  # Only report significant gaps
                    trait_gaps[trait_name] = {
                        "expected": expected_value,
                        "current": current_value,
                        "gap": gap
                    }
        
        # Format metrics
        metrics = {}
        for name, metric in user_data.obedience_metrics.items():
            metrics[name] = {
                "value": metric.value,
                "weight": metric.weight,
                "last_updated": metric.last_updated.isoformat()
            }
        
        # Assemble result
        result = {
            "user_id": user_id,
            "submission_level": {
                "id": level.id,
                "name": level.name,
                "description": level.description,
                "appropriate": level_appropriate,
                "time_at_level_days": user_data.time_at_current_level
            },
            "submission_score": user_data.total_submission_score,
            "metrics": metrics,
            "trait_gaps": trait_gaps,
            "privileges": level.privileges,
            "restrictions": level.restrictions,
            "training_focus": level.training_focus,
            "compliance_rate": user_data.lifetime_compliance_rate,
            "last_updated": user_data.last_updated.isoformat()
        }
        
        # Include history if requested
        if include_history:
            # Format compliance history
            history = []
            for record in user_data.compliance_history:
                history.append({
                    "id": record.id,
                    "timestamp": record.timestamp.isoformat(),
                    "instruction": record.instruction,
                    "complied": record.complied,
                    "difficulty": record.difficulty,
                    "defiance_reason": record.defiance_reason
                })
            
            result["compliance_history"] = history
        
        return result
    
    async def generate_progression_report(self, user_id: str) -> Dict[str, Any]:
        """Generate a detailed report on user's submission progression."""
        # Ensure user exists
        if user_id not in self.user_data:
            await self.initialize_user(user_id)
        
        user_data = self.user_data[user_id]
        level = self.submission_levels[user_data.current_level_id]
        
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
        metric_trends = {}
        for metric_name, metric in user_data.obedience_metrics.items():
            if len(metric.history) >= 10:
                # Compare last 5 with previous 5
                recent_5 = metric.history[-5:]
                previous_5 = metric.history[-10:-5]
                
                recent_avg = sum(r["final_value"] for r in recent_5) / 5
                previous_avg = sum(r["final_value"] for r in previous_5) / 5
                
                change = recent_avg - previous_avg
                
                if change > 0.05:
                    trend = "improving"
                elif change < -0.05:
                    trend = "declining"
                else:
                    trend = "stable"
                
                metric_trends[metric_name] = {
                    "trend": trend,
                    "change": change,
                    "recent_avg": recent_avg,
                    "previous_avg": previous_avg
                }
        
        # Generate recommendations based on metric gaps
        recommendations = []
        for trait_name, expected in level.traits.items():
            if trait_name in user_data.obedience_metrics:
                current = user_data.obedience_metrics[trait_name].value
                gap = expected - current
                
                if gap > 0.2:
                    recommendations.append({
                        "focus_area": trait_name,
                        "significance": gap,
                        "current_value": current,
                        "target_value": expected,
                        "description": f"Increase {trait_name} through targeted training and practice"
                    })
        
        # Sort recommendations by significance
        recommendations.sort(key=lambda x: x["significance"], reverse=True)
        
        # Format report
        report = {
            "user_id": user_id,
            "generation_time": datetime.datetime.now().isoformat(),
            "current_level": {
                "id": level.id,
                "name": level.name,
                "description": level.description,
                "progress_in_level": level_progress,
                "time_at_level_days": user_data.time_at_current_level
            },
            "submission_metrics": {
                "overall_score": user_data.total_submission_score,
                "compliance_rate": user_data.lifetime_compliance_rate,
                "compliance_trend": compliance_trend,
                "metric_trends": metric_trends
            },
            "progression_path": {
                "next_level": None,
                "requirements_for_advancement": [],
                "estimated_time_to_next_level": None
            },
            "recommendations": recommendations[:3]  # Top 3 recommendations
        }
        
        # Add next level info if not at max level
        if user_data.current_level_id < max(self.submission_levels.keys()):
            next_level = self.submission_levels[user_data.current_level_id + 1]
            
            # Calculate requirements for advancement
            requirements = []
            score_gap = next_level.min_score - user_data.total_submission_score
            if score_gap > 0:
                requirements.append({
                    "type": "score",
                    "description": f"Increase overall submission score by {score_gap:.2f}",
                    "current": user_data.total_submission_score,
                    "target": next_level.min_score
                })
            
            # Calculate trait requirements
            for trait_name, expected in next_level.traits.items():
                if trait_name in user_data.obedience_metrics:
                    current = user_data.obedience_metrics[trait_name].value
                    gap = expected - current
                    
                    if gap > 0.1:
                        requirements.append({
                            "type": "trait",
                            "trait": trait_name,
                            "description": f"Increase {trait_name} from {current:.2f} to {expected:.2f}",
                            "current": current,
                            "target": expected
                        })
            
            # Estimate time to next level based on recent progress rate
            estimated_days = None
            if level_progress > 0.2 and user_data.time_at_current_level > 0:
                # Estimate based on current progress rate
                progress_rate_per_day = level_progress / user_data.time_at_current_level
                if progress_rate_per_day > 0:
                    remaining_progress = 1.0 - level_progress
                    estimated_days = math.ceil(remaining_progress / progress_rate_per_day)
            
            report["progression_path"] = {
                "next_level": {
                    "id": next_level.id,
                    "name": next_level.name,
                    "description": next_level.description
                },
                "requirements_for_advancement": requirements,
                "estimated_time_to_next_level": estimated_days
            }
        
        return report
    
    async def add_custom_metric(self, 
                             user_id: str, 
                             metric_name: str, 
                             initial_value: float = 0.1,
                             weight: float = 1.0) -> Dict[str, Any]:
        """Add a custom metric for tracking a specific aspect of submission."""
        # Ensure user exists
        if user_id not in self.user_data:
            await self.initialize_user(user_id)
        
        user_data = self.user_data[user_id]
        
        # Check if metric already exists
        if metric_name in user_data.obedience_metrics:
            return {
                "success": False,
                "message": f"Metric '{metric_name}' already exists"
            }
        
        async with self._lock:
            # Add new metric
            user_data.obedience_metrics[metric_name] = SubmissionMetric(
                name=metric_name,
                value=initial_value,
                weight=weight
            )
            
            # Recalculate submission score
            old_score = user_data.total_submission_score
            user_data.total_submission_score = self._calculate_submission_score(user_data)
            
            return {
                "success": True,
                "metric_name": metric_name,
                "initial_value": initial_value,
                "weight": weight,
                "submission_score": {
                    "old": old_score,
                    "new": user_data.total_submission_score
                }
            }
    
    async def update_user_limits(self, 
                               user_id: str, 
                               limit_type: str, 
                               limits: List[str]) -> Dict[str, Any]:
        """Update user's limits."""
        # Ensure user exists
        if user_id not in self.user_data:
            await self.initialize_user(user_id)
        
        user_data = self.user_data[user_id]
        
        async with self._lock:
            # Update limits
            user_data.limits[limit_type] = limits
            
            # Update relationship data if available
            if self.relationship_manager:
                try:
                    await self.relationship_manager.update_relationship_attribute(
                        user_id,
                        f"{limit_type}_limits",
                        limits
                    )
                except Exception as e:
                    logger.error(f"Error updating relationship limits: {e}")
            
            return {
                "success": True,
                "user_id": user_id,
                "limit_type": limit_type,
                "limits": limits
            }
    
    async def update_user_preferences(self, 
                                   user_id: str, 
                                   preferences: Dict[str, float]) -> Dict[str, Any]:
        """Update user's preferences."""
        # Ensure user exists
        if user_id not in self.user_data:
            await self.initialize_user(user_id)
        
        user_data = self.user_data[user_id]
        
        async with self._lock:
            # Update preferences
            for pref_name, value in preferences.items():
                user_data.preferences[pref_name] = value
            
            # Update relationship data if available
            if self.relationship_manager:
                try:
                    await self.relationship_manager.update_relationship_attribute(
                        user_id,
                        "preferences",
                        user_data.preferences
                    )
                except Exception as e:
                    logger.error(f"Error updating relationship preferences: {e}")
            
            return {
                "success": True,
                "user_id": user_id,
                "preferences": user_data.preferences
            }
