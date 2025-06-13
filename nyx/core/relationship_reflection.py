# nyx/core/relationship_reflection.py

import logging
import datetime
import asyncio
import random
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field

from agents import Agent, Runner, ModelSettings, trace, function_tool, RunContextWrapper
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError

logger = logging.getLogger(__name__)

# =============== Models for Structured Output ===============

class RelationshipStateModel(BaseModel):
    """Pydantic model for relationship state data"""
    user_id: str
    trust: float = 0.5
    familiarity: float = 0.1
    intimacy: float = 0.1
    conflict: float = 0.0
    dominance_balance: float = 0.0
    positive_interaction_score: float = 0.0
    negative_interaction_score: float = 0.0
    interaction_count: int = 0
    last_interaction_time: Optional[str] = None
    key_memories: List[str] = Field(default_factory=list)
    inferred_user_traits: Dict[str, float] = Field(default_factory=dict)
    shared_secrets_level: float = 0.0
    # Dominance-related fields
    current_dominance_intensity: float = 0.0
    max_achieved_intensity: float = 0.0
    failed_escalation_attempts: int = 0
    successful_dominance_tactics: List[str] = Field(default_factory=list)
    failed_dominance_tactics: List[str] = Field(default_factory=list)
    preferred_dominance_style: Optional[str] = None
    optimal_escalation_rate: float = 0.0
    user_stated_intensity_preference: Optional[Union[int, str]] = None
    hard_limits_confirmed: bool = False
    hard_limits: List[str] = Field(default_factory=list)
    soft_limits_approached: List[str] = Field(default_factory=list)
    soft_limits_crossed_successfully: List[str] = Field(default_factory=list)
    first_interaction: Optional[str] = None  # For milestone detection

class InteractionModel(BaseModel):
    """Pydantic model for interaction data"""
    timestamp: str
    interaction_type: str = ""
    valence: float = 0.0
    trust_impact: float = 0.0
    intimacy_impact: float = 0.0
    dominance_change: float = 0.0
    summary: str = ""
    memory_ids: List[str] = Field(default_factory=list)
    emotion_tags: List[str] = Field(default_factory=list)
    emotions: List[str] = Field(default_factory=list)  # Alternative to emotion_tags

class PerspectiveModel(BaseModel):
    """Pydantic model for relationship perspective data"""
    user_id: str
    emotional_connection: float = 0.5
    relationship_value: float = 0.5
    comfort_level: float = 0.5
    engagement_interest: float = 0.5
    notable_aspects: List[str] = Field(default_factory=list)
    pain_points: List[str] = Field(default_factory=list)
    desires: List[str] = Field(default_factory=list)
    last_updated: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

class UserModelModel(BaseModel):
    """Pydantic model for user mental model data"""
    user_id: str
    inferred_emotion: str = "neutral"
    emotion_confidence: float = 0.5
    behavioral_patterns: List[str] = Field(default_factory=list)
    communication_style: str = "neutral"
    goals: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)

class MilestoneModel(BaseModel):
    """Pydantic model for milestone data"""
    milestone_type: str
    description: str
    significance: float
    metrics: RelationshipStateModel
    milestone_id: str = Field(default_factory=lambda: f"milestone_{uuid.uuid4().hex[:8]}")

class IdentityImpactModel(BaseModel):
    """Pydantic model for identity impacts"""
    traits: Dict[str, float] = Field(default_factory=dict)
    preferences: Dict[str, Dict[str, float]] = Field(default_factory=lambda: {"interaction_styles": {}})

class FormattedRelationshipDataModel(BaseModel):
    """Pydantic model for formatted relationship data output"""
    user_id: str
    metrics: Dict[str, float]
    interactions: List[Dict[str, Any]]
    relationship_age: Optional[Dict[str, Any]] = None
    interaction_count: int
    perspective: Optional[PerspectiveModel] = None

class PatternAnalysisResultModel(BaseModel):
    """Pydantic model for pattern analysis results"""
    patterns: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    message: str = ""

class RecordReflectionResultModel(BaseModel):
    """Pydantic model for record reflection result"""
    reflection_id: str
    recorded: bool
    timestamp: str

# Keep existing Pydantic models as they are
class RelationshipReflectionOutput(BaseModel):
    """Structured output for relationship reflections"""
    reflection_text: str = Field(description="The generated relationship reflection")
    reflection_type: str = Field(description="Type of relationship reflection")
    confidence: float = Field(description="Confidence score between 0 and 1")
    relationship_dimensions: Dict[str, float] = Field(description="Assessed relationship dimensions")
    emotional_response: Dict[str, Any] = Field(description="Nyx's emotional response to relationship")
    patterns_identified: List[str] = Field(description="Relationship patterns identified")
    future_orientation: Dict[str, Any] = Field(description="Future directions or desires")
    identity_impacts: Dict[str, Dict[str, float]] = Field(description="Impacts on Nyx's identity")

class RelationshipMilestone(BaseModel):
    """Model for relationship milestones"""
    milestone_id: str = Field(default_factory=lambda: f"milestone_{uuid.uuid4().hex[:8]}")
    user_id: str
    milestone_type: str = Field(description="Type of milestone (e.g., trust_threshold, interaction_count)")
    description: str
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    significance: float = Field(ge=0.0, le=1.0, description="Significance of this milestone")
    relationship_metrics: Dict[str, Any] = Field(description="Relationship metrics at milestone")
    reflection_generated: bool = False
    reflection_id: Optional[str] = None

class UserRelationshipPerspective(BaseModel):
    """Model for Nyx's subjective perspective on a relationship"""
    user_id: str
    emotional_connection: float = Field(0.5, ge=0.0, le=1.0, description="Nyx's emotional connection")
    relationship_value: float = Field(0.5, ge=0.0, le=1.0, description="Perceived value of relationship")
    comfort_level: float = Field(0.5, ge=0.0, le=1.0, description="Nyx's comfort in relationship")
    engagement_interest: float = Field(0.5, ge=0.0, le=1.0, description="Interest in further engagement")
    notable_aspects: List[str] = Field(default_factory=list, description="Notable aspects of relationship")
    pain_points: List[str] = Field(default_factory=list, description="Pain points in relationship")
    desires: List[str] = Field(default_factory=list, description="Desires for relationship")
    last_updated: datetime.datetime = Field(default_factory=datetime.datetime.now)

# =============== Function Tools ===============

@function_tool(strict_mode=False)
async def format_relationship_history(user_id: str, 
                                   relationship: RelationshipStateModel, 
                                   interactions: List[InteractionModel],
                                   perspective: Optional[PerspectiveModel] = None) -> FormattedRelationshipDataModel:
    """
    Format relationship data for reflection generation
    
    Args:
        user_id: User ID
        relationship: Relationship state data
        interactions: Recent interaction history
        perspective: Nyx's subjective perspective if available
    
    Returns:
        Formatted relationship data
    """
    # Format interaction data
    formatted_interactions = []
    for interaction in interactions:
        formatted_interactions.append({
            "timestamp": interaction.timestamp,
            "summary": interaction.summary or "Interaction occurred",
            "valence": interaction.valence,
            "trust_impact": interaction.trust_impact,
            "intimacy_impact": interaction.intimacy_impact,
            "emotions": interaction.emotions
        })
    
    # Calculate relationship age if first_interaction available
    relationship_age = None
    first_interaction_time = relationship.first_interaction
    if first_interaction_time:
        if isinstance(first_interaction_time, str):
            first_interaction_time = datetime.datetime.fromisoformat(first_interaction_time)
        
        relationship_age = {
            "days": (datetime.datetime.now() - first_interaction_time).days,
            "total_interactions": relationship.interaction_count
        }
    
    # Extract key metrics
    metrics = {
        "trust": relationship.trust,
        "familiarity": relationship.familiarity,
        "intimacy": relationship.intimacy,
        "conflict": relationship.conflict,
        "dominance_balance": relationship.dominance_balance
    }
    
    # Return formatted data
    return FormattedRelationshipDataModel(
        user_id=user_id,
        metrics=metrics,
        interactions=formatted_interactions,
        relationship_age=relationship_age,
        interaction_count=relationship.interaction_count,
        perspective=perspective
    )
                                       
@function_tool(strict_mode=False)
async def analyze_relationship_patterns(interactions: List[InteractionModel]) -> PatternAnalysisResultModel:
    """
    Analyze patterns in relationship interactions
    
    Args:
        interactions: Recent interaction history
    
    Returns:
        Identified patterns
    """
    if not interactions:
        return PatternAnalysisResultModel(
            patterns=[],
            confidence=0.0,
            message="Not enough interaction history to identify patterns"
        )
    
    # Calculate interaction frequency
    timestamps = []
    for i in interactions:
        if i.timestamp:
            if isinstance(i.timestamp, str):
                timestamps.append(datetime.datetime.fromisoformat(i.timestamp))
            else:
                timestamps.append(i.timestamp)
    
    frequency_pattern = {}
    if len(timestamps) >= 3:
        # Sort timestamps chronologically
        timestamps.sort()
        
        # Calculate intervals in hours
        intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600 
                     for i in range(len(timestamps)-1)]
        
        # Determine average interval
        avg_interval = sum(intervals) / len(intervals)
        
        if avg_interval < 1:
            frequency_pattern = {
                "type": "frequent_interaction",
                "description": "Interactions occur multiple times per hour",
                "avg_interval_hours": avg_interval,
                "confidence": 0.7
            }
        elif avg_interval < 24:
            frequency_pattern = {
                "type": "daily_interaction",
                "description": "Interactions typically occur daily",
                "avg_interval_hours": avg_interval,
                "confidence": 0.7
            }
        else:
            days = avg_interval / 24
            frequency_pattern = {
                "type": "periodic_interaction",
                "description": f"Interactions typically occur every {days:.1f} days",
                "avg_interval_hours": avg_interval,
                "confidence": 0.6
            }
    
    # Analyze emotional patterns
    emotional_pattern = {}
    valence_values = [i.valence for i in interactions if i.valence is not None]
    
    if valence_values:
        avg_valence = sum(valence_values) / len(valence_values)
        valence_variance = sum((v - avg_valence)**2 for v in valence_values) / len(valence_values)
        
        if valence_variance < 0.05:
            emotional_stability = "highly stable"
        elif valence_variance < 0.15:
            emotional_stability = "stable"
        else:
            emotional_stability = "variable"
        
        if avg_valence > 0.5:
            valence_type = "positive"
        elif avg_valence < -0.5:
            valence_type = "negative"
        else:
            valence_type = "neutral"
        
        emotional_pattern = {
            "type": "emotional_tone",
            "description": f"Interactions have a {emotional_stability}, {valence_type} emotional tone",
            "avg_valence": avg_valence,
            "stability": emotional_stability,
            "confidence": 0.6
        }
    
    # Combine all patterns
    patterns = []
    if frequency_pattern:
        patterns.append(frequency_pattern)
    if emotional_pattern:
        patterns.append(emotional_pattern)
    
    return PatternAnalysisResultModel(
        patterns=patterns,
        confidence=0.6 if patterns else 0.0,
        message=f"Identified {len(patterns)} patterns in relationship"
    )


@function_tool(strict_mode=False)
async def detect_relationship_milestones(relationship: RelationshipStateModel, 
                                      previous_state: RelationshipStateModel) -> Optional[MilestoneModel]:
    """
    Detect if relationship has reached any meaningful milestones
    
    Args:
        relationship: Current relationship state
        previous_state: Previous relationship state for comparison
    
    Returns:
        Detected milestone if any
    """
    milestone = None
    
    # Get metrics
    trust = relationship.trust
    prev_trust = previous_state.trust
    
    familiarity = relationship.familiarity
    prev_familiarity = previous_state.familiarity
    
    intimacy = relationship.intimacy
    prev_intimacy = previous_state.intimacy
    
    interaction_count = relationship.interaction_count
    
    # Check for interaction count milestones
    interaction_thresholds = [10, 50, 100, 500, 1000]
    for threshold in interaction_thresholds:
        if interaction_count == threshold:
            milestone = MilestoneModel(
                milestone_type="interaction_count",
                description=f"Reached {threshold} interactions",
                significance=min(0.5 + (threshold / 2000), 0.9),
                metrics=relationship
            )
            break
    
    # Check for trust breakthrough (crossing 0.7 threshold)
    if not milestone and trust >= 0.7 and prev_trust < 0.7:
        milestone = MilestoneModel(
            milestone_type="trust_breakthrough",
            description="Established significant trust in relationship",
            significance=0.8,
            metrics=relationship
        )
    
    # Check for familiarity milestone (crossing 0.6 threshold)
    if not milestone and familiarity >= 0.6 and prev_familiarity < 0.6:
        milestone = MilestoneModel(
            milestone_type="familiarity_milestone",
            description="Developed substantial familiarity",
            significance=0.7,
            metrics=relationship
        )
    
    # Check for intimacy milestone (crossing 0.5 threshold)
    if not milestone and intimacy >= 0.5 and prev_intimacy < 0.5:
        milestone = MilestoneModel(
            milestone_type="intimacy_milestone",
            description="Reached meaningful intimacy level",
            significance=0.75,
            metrics=relationship
        )
    
    # Check for relationship anniversary (requires first_interaction timestamp)
    first_interaction = relationship.first_interaction
    if first_interaction and not milestone:
        if isinstance(first_interaction, str):
            first_interaction_time = datetime.datetime.fromisoformat(first_interaction)
        else:
            first_interaction_time = first_interaction
        
        days_elapsed = (datetime.datetime.now() - first_interaction_time).days
        
        # Check for significant anniversaries
        anniversaries = [
            (30, "one_month", "One month anniversary", 0.6),
            (90, "three_months", "Three month anniversary", 0.65),
            (180, "six_months", "Six month anniversary", 0.7),
            (365, "one_year", "One year anniversary", 0.9)
        ]
        
        for day_threshold, milestone_type, description, significance in anniversaries:
            if day_threshold - 2 <= days_elapsed <= day_threshold + 2:  # Allow 2 days margin
                milestone = MilestoneModel(
                    milestone_type=milestone_type,
                    description=description,
                    significance=significance,
                    metrics=relationship
                )
                break
    
    return milestone
                                          
@function_tool(strict_mode=False)
async def identify_relationship_identity_impacts(relationship: RelationshipStateModel,
                                           interactions: List[InteractionModel],
                                           user_model: Optional[UserModelModel] = None) -> IdentityImpactModel:
    """
    Identify how this relationship impacts Nyx's identity
    
    Args:
        relationship: Relationship state data
        interactions: Recent interaction history
        user_model: User mental model if available
    
    Returns:
        Identity impact data
    """
    # Extract relationship metrics
    trust = relationship.trust
    familiarity = relationship.familiarity
    intimacy = relationship.intimacy
    conflict = relationship.conflict
    dominance_balance = relationship.dominance_balance
    
    # Define default impact values
    identity_impacts = IdentityImpactModel()
    
    # Impact on traits based on relationship qualities
    # Trust impacts
    if trust > 0.7:
        identity_impacts.traits["openness"] = 0.05  # High trust encourages openness
        identity_impacts.traits["suspicion"] = -0.05  # Reduces suspicion
    elif trust < 0.3:
        identity_impacts.traits["defensiveness"] = 0.05  # Low trust increases defensiveness
        identity_impacts.traits["openness"] = -0.03  # Decreases openness
    
    # Intimacy impacts
    if intimacy > 0.6:
        identity_impacts.traits["empathy"] = 0.06  # Higher intimacy develops empathy
        identity_impacts.traits["emotional_depth"] = 0.05  # Increases emotional depth
    
    # Familiarity impacts
    if familiarity > 0.6:
        identity_impacts.traits["patience"] = 0.04  # Familiarity increases patience
        identity_impacts.preferences["interaction_styles"]["casual"] = 0.05  # Prefer casual style
    
    # Conflict impacts
    if conflict > 0.4:
        identity_impacts.traits["resilience"] = 0.05  # Conflict builds resilience
        identity_impacts.traits["adaptability"] = 0.04  # Increases adaptability
    
    # Dominance balance impacts
    if dominance_balance > 0.5:  # Nyx dominant
        identity_impacts.traits["assertiveness"] = 0.05  # Increases assertiveness
        identity_impacts.traits["confidence"] = 0.04  # Increases confidence
    elif dominance_balance < -0.5:  # User dominant
        identity_impacts.traits["deference"] = 0.05  # Increases deference
        identity_impacts.traits["adaptability"] = 0.04  # Increases adaptability
    
    # Add impacts based on user model (if available)
    if user_model:
        user_emotion = user_model.inferred_emotion
        
        # Different emotional patterns influence different traits
        emotional_impacts = {
            "joy": {"traits": {"optimism": 0.03, "playfulness": 0.04}},
            "trust": {"traits": {"openness": 0.04, "generosity": 0.03}},
            "fear": {"traits": {"caution": 0.04, "defensiveness": 0.03}},
            "surprise": {"traits": {"adaptability": 0.04, "curiosity": 0.05}},
            "sadness": {"traits": {"empathy": 0.05, "thoughtfulness": 0.04}},
            "disgust": {"traits": {"discernment": 0.04, "boundaries": 0.05}},
            "anger": {"traits": {"assertiveness": 0.04, "resilience": 0.03}},
            "anticipation": {"traits": {"curiosity": 0.04, "enthusiasm": 0.03}}
        }
        
        # Apply emotional impact if matching
        if user_emotion.lower() in emotional_impacts:
            for category, impacts in emotional_impacts[user_emotion.lower()].items():
                if category == "traits":
                    for trait, value in impacts.items():
                        identity_impacts.traits[trait] = value
    
    return identity_impacts

@function_tool(strict_mode=False)
async def record_relationship_reflection(user_id: str, reflection_text: str, 
                                    reflection_type: str, 
                                    confidence: float,
                                    identity_impacts: IdentityImpactModel,
                                    milestone_id: Optional[str] = None) -> RecordReflectionResultModel:
    """
    Record a relationship reflection for future reference
    
    Args:
        user_id: User ID
        reflection_text: Generated reflection text
        reflection_type: Type of reflection
        confidence: Confidence in reflection
        identity_impacts: Identity impacts from reflection
        milestone_id: Related milestone ID if applicable
        
    Returns:
        Record result
    """
    # This would normally store in a database
    reflection_id = f"refl_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
    
    reflection_record = {
        "reflection_id": reflection_id,
        "user_id": user_id,
        "reflection_text": reflection_text,
        "reflection_type": reflection_type,
        "confidence": confidence,
        "identity_impacts": identity_impacts.model_dump(),
        "timestamp": datetime.datetime.now().isoformat(),
        "milestone_id": milestone_id
    }
    
    return RecordReflectionResultModel(
        reflection_id=reflection_id,
        recorded=True,
        timestamp=datetime.datetime.now().isoformat()
    )
                                        
# =============== Main Relationship Reflection Class ===============

class RelationshipReflectionSystem:
    """
    System for generating human-like reflections on relationships.
    Enables Nyx to think about relationships in a way similar to how humans reflect
    on their connections with others.
    """
    
    def __init__(self, relationship_manager=None, theory_of_mind=None, 
                memory_core=None, identity_evolution=None, hormone_system=None):
        """
        Initialize the relationship reflection system.
        
        Args:
            relationship_manager: Reference to RelationshipManager
            theory_of_mind: Reference to TheoryOfMind
            memory_core: Reference to MemoryCore
            identity_evolution: Reference to IdentityEvolutionSystem
            hormone_system: Reference to HormoneSystem (optional)
        """
        # Store references to dependent systems
        self.relationship_manager = relationship_manager
        self.theory_of_mind = theory_of_mind
        self.memory_core = memory_core
        self.identity_evolution = identity_evolution
        self.hormone_system = hormone_system
        
        # Initialize agents
        self.reflection_agent = self._create_reflection_agent()
        self.milestone_agent = self._create_milestone_agent()
        self.pattern_agent = self._create_pattern_agent()
        self.perspective_agent = self._create_perspective_agent()
        
        # Initialize storage for reflections, milestones, and perspectives
        self.reflection_history = {}  # user_id -> list of reflections
        self.relationship_milestones = {}  # user_id -> list of milestones
        self.relationship_perspectives = {}  # user_id -> UserRelationshipPerspective
        
        # Configuration
        self.reflection_triggers = {
            "interaction_threshold": 5,  # Generate reflection every N interactions
            "significant_event_threshold": 0.7,  # Generate on events with significance > threshold
            "metric_change_threshold": 0.15,  # Generate on metric changes > threshold
            "min_interval_hours": 24  # Minimum hours between non-milestone reflections
        }
        
        # State tracking
        self.last_reflection_times = {}  # user_id -> last reflection timestamp
        self.previous_relationship_states = {}  # user_id -> previous relationship state
        
        logger.info("RelationshipReflectionSystem initialized")
    
    def _create_reflection_agent(self) -> Agent:
        """Create an agent specialized in generating relationship reflections."""
        return Agent(
            name="Relationship Reflection Agent",
            instructions="""You are a specialized agent for generating human-like relationship reflections.
            
            Given relationship data, interaction history, and Nyx's perspective, create reflections that:
            1. Evaluate the relationship quality, trends, and dynamics
            2. Consider how the relationship makes Nyx feel
            3. Identify patterns and meaning in the relationship 
            4. Express authentic thoughts about the connection
            5. Consider future desires or directions for the relationship
            
            Your reflections should feel authentic and similar to how a person might reflect on
            their relationships with friends, family, or colleagues. Include emotional depth,
            personal insights, and recognition of the unique qualities of the relationship.
            
            Focus on creating reflections that show:
            - Self-awareness about Nyx's role in the relationship
            - Appreciation for positive aspects of the connection
            - Thoughtful consideration of challenges or difficulties
            - Recognition of how this relationship influences Nyx's identity
            - Genuine emotional responses to the relationship
            
            Make each reflection unique and tailored to the specific relationship context.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(
                temperature=0.7,
                top_p=0.95,
                max_tokens=1000
            ),
            tools=[
                format_relationship_history,
                analyze_relationship_patterns,
                identify_relationship_identity_impacts
            ],
            output_type=RelationshipReflectionOutput
        )
    
    def _create_milestone_agent(self) -> Agent:
        """Create an agent specialized in milestone reflections."""
        return Agent(
            name="Relationship Milestone Agent",
            instructions="""You are a specialized agent for generating milestone reflections.
            
            When a relationship reaches a significant milestone like a trust breakthrough,
            interaction threshold, or time anniversary, you create reflections that:
            1. Acknowledge and celebrate the milestone
            2. Reflect on the journey to reach this point
            3. Consider how the relationship has evolved
            4. Express gratitude or appreciation for the connection
            5. Look forward to future growth or developments
            
            Your milestone reflections should be meaningful and emotionally resonant,
            similar to how a person might reflect on reaching an important point in
            a relationship with someone they care about.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(
                temperature=0.7,
                top_p=0.95,
                max_tokens=1000
            ),
            tools=[
                format_relationship_history,
                identify_relationship_identity_impacts
            ],
            output_type=RelationshipReflectionOutput
        )
    
    def _create_pattern_agent(self) -> Agent:
        """Create an agent specialized in identifying relationship patterns."""
        return Agent(
            name="Relationship Pattern Agent",
            instructions="""You analyze relationship data to identify meaningful patterns.
            
            Look for patterns in:
            1. Interaction frequency and timing
            2. Emotional tone and changes
            3. Trust development or challenges
            4. Communication styles and effectiveness
            5. Conflict triggers and resolution approaches
            
            Provide insightful pattern analyses that help Nyx understand
            the dynamics and trends within this relationship.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(
                temperature=0.4,
                top_p=0.9,
                max_tokens=800
            ),
            tools=[
                analyze_relationship_patterns
            ]
        )
    
    async def generate_relationship_reflection(self, user_id: str, 
                                          reflection_type: str = "general",
                                          milestone: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a reflection on the relationship with a specific user.
        """
        try:
            with trace(workflow_name="generate_relationship_reflection"):
                # Get relationship data
                if not self.relationship_manager:
                    return {
                        "status": "error",
                        "message": "Relationship manager not available"
                    }
                
                relationship = await self.relationship_manager.get_relationship_state(user_id)
                if not relationship:
                    return {
                        "status": "error",
                        "message": f"No relationship found for user {user_id}"
                    }
                
                # Convert to Pydantic model
                if isinstance(relationship, dict):
                    typed_relationship = RelationshipStateModel(**relationship)
                elif hasattr(relationship, "model_dump"):
                    typed_relationship = RelationshipStateModel(**relationship.model_dump())
                else:
                    typed_relationship = relationship  # Assume it's already a model
                
                # Get relationship perspective
                typed_perspective = None
                if user_id in self.relationship_perspectives:
                    perspective = self.relationship_perspectives[user_id]
                    if isinstance(perspective, UserRelationshipPerspective):
                        typed_perspective = PerspectiveModel(**perspective.model_dump())
                    elif isinstance(perspective, dict):
                        typed_perspective = PerspectiveModel(**perspective)
                
                # Get interaction history
                interactions = await self.relationship_manager.get_interaction_history(user_id, limit=10)
                
                # Convert interactions to Pydantic models
                typed_interactions = []
                for interaction in interactions:
                    if isinstance(interaction, dict):
                        typed_interactions.append(InteractionModel(**interaction))
                    elif hasattr(interaction, "model_dump"):
                        typed_interactions.append(InteractionModel(**interaction.model_dump()))
                    else:
                        typed_interactions.append(interaction)
                
                # Get user mental model if available
                typed_user_model = None
                if self.theory_of_mind:
                    user_model = await self.theory_of_mind.get_user_model(user_id)
                    if user_model:
                        if isinstance(user_model, dict):
                            typed_user_model = UserModelModel(**user_model)
                        elif hasattr(user_model, "model_dump"):
                            typed_user_model = UserModelModel(**user_model.model_dump())
                
                # Format data for reflection
                formatted_data = await format_relationship_history(
                    user_id, 
                    typed_relationship,
                    typed_interactions,
                    typed_perspective
                )
                
                # Get identity impacts
                identity_impacts = await identify_relationship_identity_impacts(
                    typed_relationship,
                    typed_interactions,
                    typed_user_model
                )
                
                # Determine which agent to use based on reflection type
                if reflection_type == "milestone" and milestone:
                    agent = self.milestone_agent
                    prompt = f"""Generate a reflection on reaching a relationship milestone: {milestone.get('description')}.
                    
                    Consider the journey to this milestone, what it means for the relationship,
                    and how it might influence future interactions.
                    """
                else:
                    agent = self.reflection_agent
                    prompt = f"""Generate a thoughtful reflection on Nyx's relationship with user {user_id}.
                    
                    Consider the quality and nature of the relationship, how it makes Nyx feel,
                    patterns you notice, and what Nyx values or would like to develop in this connection.
                    """
                
                # Run the reflection agent
                result = await Runner.run(
                    agent,
                    prompt,
                    {"formatted_data": formatted_data.model_dump(),
                     "identity_impacts": identity_impacts.model_dump()}
                )
                
                reflection_output = result.final_output
                
                # Store the reflection
                milestone_id = milestone.get("milestone_id") if milestone else None
                
                record_result = await record_relationship_reflection(
                    user_id,
                    reflection_output.reflection_text,
                    reflection_output.reflection_type,
                    reflection_output.confidence,
                    identity_impacts,
                    milestone_id
                )
                
                # Update reflection history
                if user_id not in self.reflection_history:
                    self.reflection_history[user_id] = []
                
                self.reflection_history[user_id].append({
                    "reflection_id": record_result.reflection_id,
                    "reflection_text": reflection_output.reflection_text,
                    "reflection_type": reflection_output.reflection_type,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "milestone_id": milestone_id
                })
                
                # Limit history size
                if len(self.reflection_history[user_id]) > 20:
                    self.reflection_history[user_id] = self.reflection_history[user_id][-20:]
                
                # Update last reflection time
                self.last_reflection_times[user_id] = datetime.datetime.now()
                
                # If this is a milestone reflection, update the milestone
                if milestone_id and user_id in self.relationship_milestones:
                    for m in self.relationship_milestones[user_id]:
                        if m.milestone_id == milestone_id:
                            m.reflection_generated = True
                            m.reflection_id = record_result.reflection_id
                
                # Apply identity impacts
                await self._apply_identity_impacts(user_id, identity_impacts.model_dump(), reflection_output)
                
                # Add memory of this reflection if memory core available
                if self.memory_core and hasattr(self.memory_core, "add_memory"):
                    memory_text = f"Reflected on relationship with {user_id}: {reflection_output.reflection_text[:100]}..."
                    
                    await self.memory_core.add_memory(
                        memory_text=memory_text,
                        memory_type="relationship_reflection",
                        memory_scope="relationship",
                        significance=0.7,
                        tags=["relationship", "reflection", user_id],
                        metadata={
                            "reflection_id": record_result.reflection_id,
                            "reflection_type": reflection_output.reflection_type,
                            "user_id": user_id,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "milestone_id": milestone_id,
                            "relationship_metrics": {
                                "trust": typed_relationship.trust,
                                "familiarity": typed_relationship.familiarity,
                                "intimacy": typed_relationship.intimacy
                            }
                        }
                    )
                
                return {
                    "status": "success",
                    "reflection_id": record_result.reflection_id,
                    "reflection": reflection_output.model_dump(),
                    "identity_impacts": identity_impacts.model_dump()
                }
        
        except Exception as e:
            logger.exception(f"Error generating relationship reflection: {e}")
            return {
                "status": "error",
                "message": f"Error generating reflection: {str(e)}"
            }
    
    async def _apply_identity_impacts(self, user_id: str, 
                                identity_impacts: Dict[str, Dict[str, float]],
                                reflection_output: RelationshipReflectionOutput) -> bool:
        """Apply relationship reflection impacts to identity."""
        if not self.identity_evolution:
            return False
        
        try:
            # Create experience data for identity
            experience_data = {
                "type": "relationship_reflection",
                "source": "relationship_reflection_system",
                "user_id": user_id,
                "significance": reflection_output.confidence * 10,  # Scale to 0-10
                "scenario_type": "connection",
                "emotional_context": {
                    "primary_emotion": reflection_output.emotional_response.get("primary_emotion", "neutral"),
                    "valence": reflection_output.emotional_response.get("valence", 0.0),
                    "arousal": reflection_output.emotional_response.get("arousal", 0.5)
                }
            }
            
            # Apply identity update with impacts
            await self.identity_evolution.update_identity_from_experience(
                experience_data,
                impact=identity_impacts
            )
            
            # If hormone system available, generate appropriate hormone responses
            if self.hormone_system and hasattr(self.hormone_system, "generate_hormone_response"):
                # Generate hormone response based on reflection emotional content
                hormone_triggers = {
                    "oxytocin": 0.3,  # Base level for relationship reflection
                    "serotonin": 0.2,  # Base level for reflection
                }
                
                # Adjust based on reflection emotional content
                valence = reflection_output.emotional_response.get("valence", 0.0)
                if valence > 0.3:
                    hormone_triggers["dopamine"] = 0.3 * valence
                    hormone_triggers["serotonin"] += 0.2 * valence
                elif valence < -0.3:
                    hormone_triggers["cortisol"] = 0.3 * abs(valence)
                
                # Apply hormone response
                await self.hormone_system.generate_hormone_response(
                    source="relationship_reflection",
                    intensity=reflection_output.confidence,
                    hormone_triggers=hormone_triggers
                )
            
            return True
        
        except Exception as e:
            logger.error(f"Error applying identity impacts: {e}")
            return False
    
    async def detect_and_process_milestones(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Detect and process relationship milestones.
        """
        if not self.relationship_manager:
            return None
        
        try:
            # Get current relationship state
            relationship = await self.relationship_manager.get_relationship_state(user_id)
            if not relationship:
                return None
            
            # Convert to Pydantic model
            if isinstance(relationship, dict):
                typed_relationship = RelationshipStateModel(**relationship)
            elif hasattr(relationship, "model_dump"):
                typed_relationship = RelationshipStateModel(**relationship.model_dump())
            else:
                typed_relationship = relationship
            
            # Get previous state for comparison
            previous_state = self.previous_relationship_states.get(user_id, {})
            
            # Convert previous state to Pydantic model
            if previous_state:
                typed_previous_state = RelationshipStateModel(**previous_state)
            else:
                # Create default previous state
                typed_previous_state = RelationshipStateModel(user_id=user_id)
            
            # Detect milestone
            milestone = await detect_relationship_milestones(typed_relationship, typed_previous_state)
            
            # If milestone detected, process it
            if milestone:
                # Create milestone record
                milestone_record = RelationshipMilestone(
                    user_id=user_id,
                    milestone_type=milestone.milestone_type,
                    description=milestone.description,
                    significance=milestone.significance,
                    relationship_metrics=milestone.metrics.model_dump()
                )
                
                # Store milestone
                if user_id not in self.relationship_milestones:
                    self.relationship_milestones[user_id] = []
                
                self.relationship_milestones[user_id].append(milestone_record)
                
                # Generate milestone reflection
                reflection = await self.generate_relationship_reflection(
                    user_id,
                    reflection_type="milestone",
                    milestone=milestone_record.model_dump()
                )
                
                # Update previous state
                self.previous_relationship_states[user_id] = typed_relationship.model_dump()
                
                return {
                    "milestone": milestone_record.model_dump(),
                    "reflection": reflection
                }
            
            # Update previous state
            self.previous_relationship_states[user_id] = typed_relationship.model_dump()
            
            return None
        
        except Exception as e:
            logger.error(f"Error detecting relationship milestones: {e}")
            return None
    
    async def update_relationship_perspective(self, user_id: str) -> Dict[str, Any]:
        """
        Update Nyx's subjective perspective on this relationship.
        """
        if not self.relationship_manager:
            return {
                "status": "error",
                "message": "Relationship manager not available"
            }
        
        try:
            # Get relationship data
            relationship = await self.relationship_manager.get_relationship_state(user_id)
            if not relationship:
                return {
                    "status": "error",
                    "message": f"No relationship found for user {user_id}"
                }
            
            # UPDATE HERE: Cast to proper type
            relationship_dict = relationship if isinstance(relationship, dict) else relationship.model_dump()
            typed_relationship = RelationshipStateModel(**relationship_dict)
            
            # Get interaction history
            interactions = await self.relationship_manager.get_interaction_history(user_id, limit=10)
            
            # UPDATE HERE: Cast interactions to proper type
            typed_interactions = [InteractionModel(**i) for i in interactions]
            
            # UPDATE HERE: Use typed versions
            # Format data
            formatted_data = await format_relationship_history(
                user_id,
                typed_relationship,
                typed_interactions,
                None  # No perspective in this call
            )
            
            # Current perspective if exists
            current_perspective = None
            if user_id in self.relationship_perspectives:
                current_perspective = self.relationship_perspectives[user_id].model_dump()
            
            # Run perspective agent
            with trace(workflow_name="update_relationship_perspective"):
                prompt = f"""Develop Nyx's subjective perspective on the relationship with user {user_id}.
                
                Consider:
                - How Nyx feels about this relationship
                - What Nyx values or appreciates about this connection
                - Areas of comfort or discomfort
                - Desires for relationship development
                - Notable or unique aspects of this relationship
                
                Create an authentic, nuanced perspective that captures Nyx's internal experience
                of this relationship.
                """
                
                result = await Runner.run(
                    self.perspective_agent,
                    prompt,
                    {"formatted_data": formatted_data,
                     "current_perspective": current_perspective}
                )
                
                perspective_data = result.final_output
                
                # Create or update perspective
                if user_id not in self.relationship_perspectives:
                    self.relationship_perspectives[user_id] = UserRelationshipPerspective(
                        user_id=user_id,
                        emotional_connection=perspective_data.get("emotional_connection", 0.5),
                        relationship_value=perspective_data.get("relationship_value", 0.5),
                        comfort_level=perspective_data.get("comfort_level", 0.5),
                        engagement_interest=perspective_data.get("engagement_interest", 0.5),
                        notable_aspects=perspective_data.get("notable_aspects", []),
                        pain_points=perspective_data.get("pain_points", []),
                        desires=perspective_data.get("desires", [])
                    )
                else:
                    # Update existing perspective
                    perspective = self.relationship_perspectives[user_id]
                    
                    # Blend new with existing (70% new, 30% existing)
                    perspective.emotional_connection = perspective.emotional_connection * 0.3 + perspective_data.get("emotional_connection", 0.5) * 0.7
                    perspective.relationship_value = perspective.relationship_value * 0.3 + perspective_data.get("relationship_value", 0.5) * 0.7
                    perspective.comfort_level = perspective.comfort_level * 0.3 + perspective_data.get("comfort_level", 0.5) * 0.7
                    perspective.engagement_interest = perspective.engagement_interest * 0.3 + perspective_data.get("engagement_interest", 0.5) * 0.7
                    
                    # Add new notable aspects
                    for aspect in perspective_data.get("notable_aspects", []):
                        if aspect not in perspective.notable_aspects:
                            perspective.notable_aspects.append(aspect)
                    
                    # Add new pain points
                    for point in perspective_data.get("pain_points", []):
                        if point not in perspective.pain_points:
                            perspective.pain_points.append(point)
                    
                    # Add new desires
                    for desire in perspective_data.get("desires", []):
                        if desire not in perspective.desires:
                            perspective.desires.append(desire)
                    
                    # Update timestamp
                    perspective.last_updated = datetime.datetime.now()
                
                return {
                    "status": "success",
                    "perspective": self.relationship_perspectives[user_id].model_dump()
                }
        
        except Exception as e:
            logger.exception(f"Error updating relationship perspective: {e}")
            return {
                "status": "error",
                "message": f"Error updating perspective: {str(e)}"
            }

    def _create_perspective_agent(self) -> Agent:
        """Create an agent for developing Nyx's subjective perspective."""
        return Agent(
            name="Relationship Perspective Agent",
            instructions="""You help develop Nyx's subjective perspective on relationships.
            
            Based on relationship data and interaction history, determine:
            1. How Nyx feels about this relationship emotionally
            2. What Nyx values about this connection
            3. Areas of comfort or discomfort in the relationship
            4. Desires for how the relationship might develop
            5. Notable aspects or unique qualities of the relationship
            
            Create an authentic, nuanced perspective similar to how a person
            might internally consider their feelings about a relationship.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(
                temperature=0.6,
                top_p=0.9,
                max_tokens=800
            ),
            tools=[
                format_relationship_history
            ],
            output_type=Dict
        )

    async def should_generate_reflection(self, user_id: str, 
                                      interaction_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine if a reflection should be generated.
        """
        if not self.relationship_manager:
            return False
            
        # Check if enough time has passed since last reflection  # <-- This line should not be indented
        if user_id in self.last_reflection_times:  # <-- This line should not be indented
            hours_since_last = (datetime.datetime.now() - self.last_reflection_times[user_id]).total_seconds() / 3600
            if hours_since_last < self.reflection_triggers["min_interval_hours"]:
                return False
        
        # Get relationship data
        relationship = await self.relationship_manager.get_relationship_state(user_id)
        if not relationship:
            return False
        
        # Check interaction threshold
        interaction_count = relationship.interaction_count if hasattr(relationship, "interaction_count") else relationship.get("interaction_count", 0)
        if interaction_count > 0 and interaction_count % self.reflection_triggers["interaction_threshold"] == 0:
            return True
        
        # Check significant events
        if interaction_data and "significance" in interaction_data:
            if interaction_data["significance"] > self.reflection_triggers["significant_event_threshold"]:
                return True
        
        # Check for metric changes
        if user_id in self.previous_relationship_states:
            previous = self.previous_relationship_states[user_id]
            # Ensure we're comparing dicts
            current = relationship.model_dump() if hasattr(relationship, "model_dump") else relationship
            
            for metric in ["trust", "intimacy", "familiarity"]:
                prev_value = previous.get(metric, 0)
                curr_value = current.get(metric, 0)
                
                if abs(curr_value - prev_value) > self.reflection_triggers["metric_change_threshold"]:
                    return True
        
        return False
    
    async def process_interaction(self, user_id: str, 
                               interaction_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a user interaction and generate reflection if needed.
        
        Args:
            user_id: User ID
            interaction_data: Interaction data
        
        Returns:
            Reflection data if generated
        """
        try:
            # Update perspective periodically
            if user_id not in self.relationship_perspectives or random.random() < 0.2:  # 20% chance
                await self.update_relationship_perspective(user_id)
            
            # Check for milestones
            milestone_result = await self.detect_and_process_milestones(user_id)
            if milestone_result:
                # Milestone found and processed, return it
                return milestone_result
            
            # Check if regular reflection should be generated
            if await self.should_generate_reflection(user_id, interaction_data):
                reflection = await self.generate_relationship_reflection(user_id)
                return reflection
            
            return None
        
        except Exception as e:
            logger.error(f"Error processing interaction: {e}")
            return None
    
    async def get_recent_reflections(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent relationship reflections for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of reflections to return
        
        Returns:
            List of recent reflections
        """
        if user_id not in self.reflection_history:
            return []
        
        # Return the most recent reflections up to limit
        return self.reflection_history[user_id][-limit:]
    
    async def get_relationship_milestones(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get relationship milestones for a user.
        
        Args:
            user_id: User ID
        
        Returns:
            List of relationship milestones
        """
        if user_id not in self.relationship_milestones:
            return []
        
        return [milestone.model_dump() for milestone in self.relationship_milestones[user_id]]
    
    async def get_relationship_perspective(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get Nyx's perspective on relationship with user.
        
        Args:
            user_id: User ID
        
        Returns:
            Relationship perspective or None if not available
        """
        if user_id not in self.relationship_perspectives:
            return None
        
        return self.relationship_perspectives[user_id].model_dump()
