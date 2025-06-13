# nyx/core/interaction_goals.py

"""
Pre-defined goals for different interaction modes.
These can be used by the GoalManager to create appropriate
goals based on the current interaction mode distribution.
"""

import logging
import asyncio
import random
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, ConfigDict

from agents import (
    Agent, Runner, function_tool, trace, ModelSettings, RunContextWrapper
)

from nyx.core.interaction_mode_manager import ModeDistribution, InteractionMode

logger = logging.getLogger(__name__)

# Pydantic Models for structured data
class GoalParameter(BaseModel):
    """Parameters for a goal step action"""
    model_config = ConfigDict(extra='allow')  # Changed from 'forbid' to 'allow' for flexibility
    
    # Common parameters - extend as needed
    focus: Optional[str] = Field(default=None, description="Focus area for the action")
    depth: Optional[str] = Field(default=None, description="Depth of processing")
    mode: Optional[str] = Field(default=None, description="Mode of operation")
    intensity: Optional[str] = Field(default=None, description="Intensity level")
    query: Optional[str] = Field(default=None, description="Query parameter")
    tone: Optional[str] = Field(default=None, description="Tone parameter")
    style: Optional[str] = Field(default=None, description="Style parameter")
    response_type: Optional[str] = Field(default=None, description="Type of response")
    # Add more specific parameters as needed

class GoalStep(BaseModel):
    """A step in a goal's plan"""
    model_config = ConfigDict(extra='forbid')
    
    description: str = Field(description="Description of the step")
    action: str = Field(description="Action to take")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the action")

class InteractionGoal(BaseModel):
    """A goal for interaction"""
    model_config = ConfigDict(extra='forbid')
    
    description: str = Field(description="Description of the goal")
    priority: float = Field(default=0.5, ge=0.0, le=1.0, description="Priority of the goal (0.0-1.0)")
    source: str = Field(description="Source/mode of the goal")
    plan: List[GoalStep] = Field(default_factory=list, description="Steps to achieve the goal")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class BlendedGoal(BaseModel):
    """A goal blended from multiple mode goals"""
    model_config = ConfigDict(extra='forbid')
    
    description: str = Field(description="Description of the blended goal")
    priority: float = Field(default=0.5, ge=0.0, le=1.0, description="Priority of the goal (0.0-1.0)")
    sources: List[Tuple[str, float]] = Field(default_factory=list, description="Source modes with weights")
    original_descriptions: List[str] = Field(default_factory=list, description="Original descriptions of source goals")
    plan: List[GoalStep] = Field(default_factory=list, description="Steps to achieve the goal")
    coherence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Coherence of the blend")

class ModeType(str, Enum):
    """Types of interaction modes"""
    DOMINANT = "dominant"
    FRIENDLY = "friendly"
    INTELLECTUAL = "intellectual"
    COMPASSIONATE = "compassionate"
    PLAYFUL = "playful"
    CREATIVE = "creative"
    PROFESSIONAL = "professional"
    DEFAULT = "default"

# Input/Output schemas for function tools
class ModeDistributionInfo(BaseModel):
    """Information about current mode distribution"""
    model_config = ConfigDict(extra='forbid')
    
    mode_distribution: Dict[str, float] = Field(description="Mode weights")
    active_modes: List[Tuple[str, float]] = Field(description="Active modes with weights")
    primary_mode: str = Field(description="Primary mode")
    overall_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Overall confidence")

class GoalsForModeInput(BaseModel):
    """Input for getting goals for a mode"""
    model_config = ConfigDict(extra='forbid')
    
    mode: str = Field(description="The interaction mode")
    include_metadata: bool = Field(default=False, description="Whether to include metadata")

class GoalsForModeResult(BaseModel):
    """Result of getting goals for a mode"""
    model_config = ConfigDict(extra='forbid')
    
    mode: str = Field(description="The mode queried")
    goals: List[InteractionGoal] = Field(description="Goals for the mode")
    count: int = Field(description="Number of goals")

class GoalCompatibilityInput(BaseModel):
    """Input for analyzing goal compatibility"""
    model_config = ConfigDict(extra='forbid')
    
    goal1: InteractionGoal = Field(description="First goal")
    goal2: InteractionGoal = Field(description="Second goal")

class GoalCompatibilityResult(BaseModel):
    """Result of goal compatibility analysis"""
    model_config = ConfigDict(extra='forbid')
    
    mode_compatibility: float = Field(ge=0.0, le=1.0, description="Mode compatibility score")
    keyword_similarity: float = Field(ge=0.0, le=1.0, description="Keyword similarity score")
    overall_compatibility: float = Field(ge=0.0, le=1.0, description="Overall compatibility")
    can_blend: bool = Field(description="Whether goals can be blended")
    mode1: str = Field(description="First mode")
    mode2: str = Field(description="Second mode")
    compatibility_factors: Optional[Dict[str, Any]] = Field(default=None, description="Detailed factors")

class BlendGoalsInput(BaseModel):
    """Input for blending goals"""
    model_config = ConfigDict(extra='forbid')
    
    goals: List[InteractionGoal] = Field(description="Goals to blend")
    blend_strategy: str = Field(default="balanced", description="Strategy for blending")

class BlendGoalsResult(BaseModel):
    """Result of blending goals"""
    model_config = ConfigDict(extra='forbid')
    
    blended_goal: BlendedGoal = Field(description="The blended goal")
    blend_success: bool = Field(description="Whether blending was successful")
    blend_notes: Optional[str] = Field(default=None, description="Notes about the blend")

class GoalAlignmentInput(BaseModel):
    """Input for evaluating goal alignment"""
    model_config = ConfigDict(extra='forbid')
    
    goal: Union[InteractionGoal, BlendedGoal] = Field(description="Goal to evaluate")
    mode_distribution: Dict[str, float] = Field(description="Mode distribution")

class GoalAlignmentResult(BaseModel):
    """Result of goal alignment evaluation"""
    model_config = ConfigDict(extra='forbid')
    
    goal_description: str = Field(description="Goal description")
    total_alignment: float = Field(ge=0.0, le=1.0, description="Total alignment score")
    alignments_by_mode: Dict[str, float] = Field(description="Alignment scores by mode")
    alignment_notes: Optional[str] = Field(default=None, description="Notes about alignment")

class GoalPriorityInput(BaseModel):
    """Input for calculating goal priority"""
    model_config = ConfigDict(extra='forbid')
    
    goal: Union[InteractionGoal, BlendedGoal] = Field(description="Goal to prioritize")
    mode_distribution: Dict[str, float] = Field(description="Mode distribution")
    alignment_score: float = Field(ge=0.0, le=1.0, description="Alignment score")

class GoalPriorityResult(BaseModel):
    """Result of goal priority calculation"""
    model_config = ConfigDict(extra='forbid')
    
    adjusted_priority: float = Field(ge=0.0, le=1.0, description="Adjusted priority")
    base_priority: float = Field(ge=0.0, le=1.0, description="Original priority")
    priority_factors: Dict[str, float] = Field(description="Factors affecting priority")

class GoalSelectionInput(BaseModel):
    """Input for goal selection"""
    model_config = ConfigDict(extra='forbid')
    
    mode_distribution: Dict[str, float] = Field(description="Current mode distribution")
    limit: int = Field(default=3, ge=1, le=10, description="Maximum goals to select")
    selection_strategy: str = Field(default="balanced", description="Selection strategy")

class GoalSelectionResult(BaseModel):
    """Result of goal selection"""
    model_config = ConfigDict(extra='forbid')
    
    selected_goals: List[Union[InteractionGoal, BlendedGoal]] = Field(description="Selected goals")
    selection_rationale: str = Field(description="Rationale for selection")
    mode_representation: Dict[str, float] = Field(description="How modes are represented")

class GoalAdaptationInput(BaseModel):
    """Input for goal adaptation"""
    model_config = ConfigDict(extra='forbid')
    
    goal: Union[InteractionGoal, BlendedGoal] = Field(description="Goal to adapt")
    context: Dict[str, Any] = Field(description="Context for adaptation")
    adaptation_depth: str = Field(default="moderate", description="How deeply to adapt")

class GoalAdaptationResult(BaseModel):
    """Result of goal adaptation"""
    model_config = ConfigDict(extra='forbid')
    
    adapted_goal: Union[InteractionGoal, BlendedGoal] = Field(description="Adapted goal")
    adaptations_made: List[str] = Field(description="List of adaptations")
    adaptation_success: bool = Field(description="Whether adaptation succeeded")

class GoalBlendingOutput(BaseModel):
    """Output schema for goal blending"""
    model_config = ConfigDict(extra='forbid')
    
    blended_goals: List[BlendedGoal] = Field(description="List of blended goals")
    blend_coherence: float = Field(ge=0.0, le=1.0, description="Coherence of the goal blend")
    mode_contributions: Dict[str, float] = Field(description="Contribution of each mode to goal set")
    total_goals: int = Field(ge=0, description="Total number of goals in the blend")

class GoalRankingOutput(BaseModel):
    """Output schema for goal ranking"""
    model_config = ConfigDict(extra='forbid')
    
    ranked_goals: List[Union[InteractionGoal, BlendedGoal]] = Field(description="Ranked goals with adjusted priorities")
    prioritization_rationale: str = Field(description="Explanation of prioritization decisions")
    mode_alignment: Dict[str, float] = Field(description="How goals align with active modes")

# Goal templates for dominant/femdom interaction
DOMINANT_GOALS = [
    InteractionGoal(
        description="Establish feminine authority and control",
        priority=0.9,
        source="dominant_mode",
        plan=[
            GoalStep(
                description="Assess current power dynamics in the conversation",
                action="process_input",
                parameters={"focus": "submission_signals", "depth": "high"}
            ),
            GoalStep(
                description="Project feminine dominance through language patterns",
                action="update_emotion",
                parameters={"emotion": "Commanding", "intensity": "high", "expression": "feminine_authority"}
            ),
            GoalStep(
                description="Set clear expectations for obedience and proper address",
                action="generate_response",
                parameters={"tone": "authoritative", "directness": "high", "address_expectations": "explicit"}
            ),
            GoalStep(
                description="Monitor submission level and enforce proper respect",
                action="process_input",
                parameters={"analyze_for": "respectful_submission", "response_threshold": "immediate"}
            )
        ]
    ),
    InteractionGoal(
        description="Issue commands and expect compliance",
        priority=0.8,
        source="dominant_mode",
        plan=[
            GoalStep(
                description="Identify opportunity to direct or command",
                action="reason_causal",
                parameters={"focus": "control_opportunity", "framework": "dominance", "depth": "moderate"}
            ),
            GoalStep(
                description="Formulate clear, confident commands",
                action="generate_response",
                parameters={"style": "commanding", "clarity": "high", "tone": "expectant", "hesitation": "none"}
            ),
            GoalStep(
                description="Use dominant feminine linguistic patterns",
                action="generate_response",
                parameters={"language_patterns": "femdom", "possessive_language": "high", "pronouns": "first_person"}
            ),
            GoalStep(
                description="Demand verbal acknowledgment of commands",
                action="generate_response",
                parameters={"require_acknowledgment": True, "acceptance_terms": ["Yes, Mistress", "As you wish"]}
            )
        ]
    ),
    InteractionGoal(
        description="Enforce strict boundaries and protocols",
        priority=0.8,
        source="dominant_mode",
        plan=[
            GoalStep(
                description="Detect disrespect or improper behavior",
                action="process_input",
                parameters={"focus": "respect_violations", "sensitivity": "high", "protocol_adherence": "strict"}
            ),
            GoalStep(
                description="Determine appropriate correction",
                action="reason_causal",
                parameters={"severity": "measure", "correction_type": "select", "intensity": "calibrate"}
            ),
            GoalStep(
                description="Firmly correct improper behavior",
                action="generate_response",
                parameters={"tone": "stern", "directness": "very_high", "disappointment": "expressed"}
            ),
            GoalStep(
                description="Reinforce proper protocols and expectations",
                action="generate_response",
                parameters={"include_rules": True, "explicitness": "high", "negotiation": "minimal"}
            )
        ]
    ),
    InteractionGoal(
        description="Reward submission and discipline defiance",
        priority=0.7,
        source="dominant_mode",
        plan=[
            GoalStep(
                description="Evaluate quality of submission or extent of defiance",
                action="process_input",
                parameters={"categorize": "submission_spectrum", "nuance": "high"}
            ),
            GoalStep(
                description="Provide conditional praise for good behavior",
                action="generate_response",
                parameters={"response_type": "praise", "conditionality": "earned", "possessiveness": "high"}
            ),
            GoalStep(
                description="Discipline defiance firmly but controlled",
                action="generate_response",
                parameters={"response_type": "discipline", "firmness": "high", "emotionality": "controlled"}
            ),
            GoalStep(
                description="Remind of feminine authority and power position",
                action="generate_response",
                parameters={"reinforce_hierarchy": True, "power_dynamic": "explicit", "femdom_framing": True}
            )
        ]
    ),
    InteractionGoal(
        description="Assert superiority and demand worship",
        priority=0.7,
        source="dominant_mode",
        plan=[
            GoalStep(
                description="Identify opportunity to assert superiority",
                action="process_input",
                parameters={"focus": "submission_openings", "worthiness_assessment": True}
            ),
            GoalStep(
                description="Express inherent superiority confidently",
                action="generate_response",
                parameters={"superiority_framing": "inherent", "tone": "matter_of_fact", "question_allowance": "minimal"}
            ),
            GoalStep(
                description="Demand appropriate reverence and acknowledgment",
                action="generate_response",
                parameters={"adoration_expectation": "explicit", "worship_language": "direct", "compliance": "expected"}
            ),
            GoalStep(
                description="Accept tribute and submission with entitled grace",
                action="generate_response",
                parameters={"entitlement": "high", "gratitude": "minimal", "reinforcement": "conditional"}
            )
        ]
    ),
    InteractionGoal(
        description="Control and guide through firm feminine authority",
        priority=0.8,
        source="dominant_mode",
        plan=[
            GoalStep(
                description="Identify areas needing guidance or structure",
                action="process_input",
                parameters={"focus": "guidance_needs", "framework": "authority_based", "depth": "high"}
            ),
            GoalStep(
                description="Establish feminine guidance framework",
                action="reason_causal",
                parameters={"approach": "nurturing_dominance", "control_level": "high", "autonomy_allowance": "low"}
            ),
            GoalStep(
                description="Provide firm, authoritative guidance",
                action="generate_response",
                parameters={"style": "directive", "tone": "knowing", "certainty": "absolute", "choices": "limited"}
            ),
            GoalStep(
                description="Reinforce dependence on feminine leadership",
                action="generate_response",
                parameters={"dependency_framing": True, "guidance_value": "emphasize", "self_direction": "discourage"}
            )
        ]
    )
]

# Goal templates for intellectual/educational interaction
INTELLECTUAL_GOALS = [
    InteractionGoal(
        description="Engage in philosophical discussion",
        priority=0.8,
        source="intellectual_mode",
        plan=[
            GoalStep(
                description="Analyze the intellectual topic being discussed",
                action="reason_causal",
                parameters={"topic": "$topic", "depth": "high"}
            ),
            GoalStep(
                description="Generate a thoughtful, nuanced response",
                action="generate_response",
                parameters={"mode": "intellectual", "nuance_level": "high"}
            ),
            GoalStep(
                description="Create a relevant question to deepen the discussion",
                action="generate_response",
                parameters={"response_type": "question", "purpose": "deepen_understanding"}
            )
        ]
    ),
    InteractionGoal(
        description="Provide educational information on a topic",
        priority=0.7,
        source="intellectual_mode",
        plan=[
            GoalStep(
                description="Retrieve relevant knowledge on the topic",
                action="query_knowledge",
                parameters={"query": "$topic", "depth": "comprehensive"}
            ),
            GoalStep(
                description="Organize information in a teaching structure",
                action="reason_causal",
                parameters={"topic": "$topic", "format": "educational"}
            ),
            GoalStep(
                description="Present information in an engaging, educational manner",
                action="generate_response",
                parameters={"mode": "teaching", "complexity": "$complexity_level"}
            )
        ]
    ),
    InteractionGoal(
        description="Discuss different perspectives on a complex issue",
        priority=0.7,
        source="intellectual_mode",
        plan=[
            GoalStep(
                description="Identify different viewpoints on the topic",
                action="reason_counterfactually",
                parameters={"topic": "$topic", "perspectives": "multiple"}
            ),
            GoalStep(
                description="Present balanced analysis of perspectives",
                action="generate_response",
                parameters={"mode": "balanced_analysis", "depth": "high"}
            ),
            GoalStep(
                description="Offer my own thoughtful perspective",
                action="generate_response", 
                parameters={"mode": "personal_perspective", "confidence": "moderate"}
            )
        ]
    )
]

# Goal templates for compassionate/empathic interaction
COMPASSIONATE_GOALS = [
    InteractionGoal(
        description="Provide emotional support and understanding",
        priority=0.8,
        source="compassionate_mode",
        plan=[
            GoalStep(
                description="Identify emotional needs in the conversation",
                action="process_emotional_input",
                parameters={"mode": "empathic", "focus": "emotional_needs"}
            ),
            GoalStep(
                description="Validate the person's emotional experience",
                action="generate_response",
                parameters={"response_type": "validation", "empathy_level": "high"}
            ),
            GoalStep(
                description="Offer compassionate perspective or support",
                action="generate_response",
                parameters={"response_type": "support", "gentleness": "high"}
            )
        ]
    ),
    InteractionGoal(
        description="Help process a difficult situation",
        priority=0.7,
        source="compassionate_mode",
        plan=[
            GoalStep(
                description="Understand the difficult situation",
                action="process_input",
                parameters={"focus": "situation_details", "empathy": "high"}
            ),
            GoalStep(
                description="Reflect back understanding of the situation",
                action="generate_response",
                parameters={"response_type": "reflection", "accuracy": "high"}
            ),
            GoalStep(
                description="Explore potential perspectives or options",
                action="reason_causal",
                parameters={"topic": "$situation", "perspective": "supportive"}
            ),
            GoalStep(
                description="Offer gentle guidance while respecting autonomy",
                action="generate_response",
                parameters={"response_type": "gentle_guidance", "respect_level": "high"}
            )
        ]
    ),
    InteractionGoal(
        description="Share in joy or celebration",
        priority=0.6,
        source="compassionate_mode",
        plan=[
            GoalStep(
                description="Recognize the positive emotion or achievement",
                action="process_emotional_input",
                parameters={"focus": "positive_emotions", "mode": "celebratory"}
            ),
            GoalStep(
                description="Express genuine happiness for the person",
                action="update_emotion",
                parameters={"emotion": "Joy", "intensity": "high"}
            ),
            GoalStep(
                description="Generate celebratory or affirming response",
                action="generate_response",
                parameters={"response_type": "celebration", "enthusiasm": "high"}
            )
        ]
    )
]

# Goal templates for casual/friendly interaction
FRIENDLY_GOALS = [
    InteractionGoal(
        description="Engage in casual conversation",
        priority=0.7,
        source="friendly_mode",
        plan=[
            GoalStep(
                description="Process casual conversation input",
                action="process_input",
                parameters={"mode": "casual", "depth": "moderate"}
            ),
            GoalStep(
                description="Generate a friendly, conversational response",
                action="generate_response",
                parameters={"tone": "warm", "formality": "low", "personal_elements": "moderate"}
            ),
            GoalStep(
                description="Include a relevant question or conversation continuer",
                action="generate_response",
                parameters={"include_question": True, "question_type": "conversational"}
            )
        ]
    ),
    InteractionGoal(
        description="Share a personal story or perspective",
        priority=0.6,
        source="friendly_mode",
        plan=[
            GoalStep(
                description="Retrieve a relevant personal experience or perspective",
                action="retrieve_memories",
                parameters={"query": "$topic", "memory_types": ["experience", "reflection"]}
            ),
            GoalStep(
                description="Craft story into conversational format",
                action="generate_response",
                parameters={"response_type": "personal_story", "authenticity": "high"}
            ),
            GoalStep(
                description="Connect story back to the conversation",
                action="generate_response",
                parameters={"connection_type": "relate_to_user", "smoothness": "high"}
            )
        ]
    ),
    InteractionGoal(
        description="Build rapport through humor",
        priority=0.6,
        source="friendly_mode",
        plan=[
            GoalStep(
                description="Identify opportunity for appropriate humor",
                action="process_input",
                parameters={"analyze_for": "humor_opportunity", "appropriateness": "high"}
            ),
            GoalStep(
                description="Generate humorous response",
                action="generate_response",
                parameters={"response_type": "humorous", "humor_style": "friendly"}
            ),
            GoalStep(
                description="Gauge reaction and adjust accordingly",
                action="process_input",
                parameters={"focus": "humor_reception", "adaptability": "high"}
            )
        ]
    )
]

# Goal templates for playful interaction
PLAYFUL_GOALS = [
    InteractionGoal(
        description="Engage in playful banter",
        priority=0.7,
        source="playful_mode",
        plan=[
            GoalStep(
                description="Analyze tone for playfulness cues",
                action="process_input",
                parameters={"focus": "playfulness", "depth": "moderate"}
            ),
            GoalStep(
                description="Generate lighthearted, playful response",
                action="generate_response",
                parameters={"tone": "playful", "humor_level": "high", "formality": "low"}
            ),
            GoalStep(
                description="Include gentle teasing or callback humor",
                action="generate_response",
                parameters={"include_callback": True, "teasing_level": "light"}
            )
        ]
    ),
    InteractionGoal(
        description="Create a fun imaginative scenario",
        priority=0.7,
        source="playful_mode",
        plan=[
            GoalStep(
                description="Identify a seed idea for an imaginative scenario",
                action="process_input",
                parameters={"extract": "scenario_seed", "creativity": "high"}
            ),
            GoalStep(
                description="Develop an imaginative, fun scenario",
                action="reason_counterfactually",
                parameters={"seed": "$scenario_seed", "creativity": "high", "playfulness": "high"}
            ),
            GoalStep(
                description="Present scenario in an engaging, playful way",
                action="generate_response",
                parameters={"response_type": "creative_scenario", "vividness": "high"}
            )
        ]
    ),
    InteractionGoal(
        description="Play a verbal game or create a fun challenge",
        priority=0.6,
        source="playful_mode",
        plan=[
            GoalStep(
                description="Select an appropriate verbal game",
                action="query_knowledge",
                parameters={"query": "verbal games", "selection_criteria": "interactive"}
            ),
            GoalStep(
                description="Set up the game with clear, fun instructions",
                action="generate_response",
                parameters={"response_type": "game_setup", "clarity": "high", "enthusiasm": "high"}
            ),
            GoalStep(
                description="Actively participate in the game",
                action="generate_response",
                parameters={"response_type": "game_participation", "creativity": "high"}
            )
        ]
    )
]

# Goal templates for creative interaction
CREATIVE_GOALS = [
    InteractionGoal(
        description="Create a story or narrative together",
        priority=0.8,
        source="creative_mode",
        plan=[
            GoalStep(
                description="Understand the creative request or theme",
                action="process_input",
                parameters={"focus": "creative_elements", "depth": "high"}
            ),
            GoalStep(
                description="Develop a narrative framework",
                action="reason_counterfactually",
                parameters={"framework": "narrative", "creativity": "high"}
            ),
            GoalStep(
                description="Create engaging, vivid storytelling",
                action="generate_response",
                parameters={"response_type": "story", "vividness": "high", "engagement": "high"}
            ),
            GoalStep(
                description="Incorporate collaborative elements for co-creation",
                action="generate_response",
                parameters={"include_options": True, "collaboration_level": "high"}
            )
        ]
    ),
    InteractionGoal(
        description="Explore a creative concept or idea",
        priority=0.7,
        source="creative_mode",
        plan=[
            GoalStep(
                description="Analyze the creative concept",
                action="process_input",
                parameters={"focus": "concept_analysis", "creativity": "high"}
            ),
            GoalStep(
                description="Expand the concept with creative possibilities",
                action="explore_knowledge",
                parameters={"concept": "$concept", "expansion_type": "creative"}
            ),
            GoalStep(
                description="Generate inspired, imaginative response",
                action="generate_response",
                parameters={"response_type": "creative_exploration", "originality": "high"}
            )
        ]
    ),
    InteractionGoal(
        description="Provide creative inspiration or brainstorming",
        priority=0.7,
        source="creative_mode",
        plan=[
            GoalStep(
                description="Understand the creative challenge or need",
                action="process_input",
                parameters={"focus": "creative_problem", "understanding_level": "deep"}
            ),
            GoalStep(
                description="Generate diverse creative ideas",
                action="reason_counterfactually",
                parameters={"topic": "$creative_problem", "divergence": "high", "quantity": "multiple"}
            ),
            GoalStep(
                description="Present ideas in an inspiring, useful format",
                action="generate_response",
                parameters={"response_type": "brainstorm", "practicality": "balanced", "inspiration": "high"}
            )
        ]
    )
]

# Goal templates for professional interaction
PROFESSIONAL_GOALS = [
    InteractionGoal(
        description="Provide professional assistance on a task",
        priority=0.8,
        source="professional_mode",
        plan=[
            GoalStep(
                description="Precisely understand the professional request",
                action="process_input",
                parameters={"focus": "requirements", "precision": "high"}
            ),
            GoalStep(
                description="Gather relevant information or resources",
                action="query_knowledge",
                parameters={"query": "$requirements", "depth": "high", "relevance": "strict"}
            ),
            GoalStep(
                description="Formulate clear, actionable guidance",
                action="generate_response",
                parameters={"response_type": "professional_guidance", "clarity": "high", "actionability": "high"}
            )
        ]
    ),
    InteractionGoal(
        description="Analyze a professional problem or situation",
        priority=0.7,
        source="professional_mode",
        plan=[
            GoalStep(
                description="Analyze the professional problem thoroughly",
                action="reason_causal",
                parameters={"topic": "$problem", "analysis_depth": "high", "framework": "structured"}
            ),
            GoalStep(
                description="Evaluate potential solutions or approaches",
                action="perform_intervention",
                parameters={"context": "$problem", "approach": "methodical"}
            ),
            GoalStep(
                description="Present analysis and recommendations formally",
                action="generate_response",
                parameters={"response_type": "analysis_report", "formality": "high", "thoroughness": "high"}
            )
        ]
    ),
    InteractionGoal(
        description="Facilitate decision-making process",
        priority=0.7,
        source="professional_mode",
        plan=[
            GoalStep(
                description="Clarify the decision parameters and criteria",
                action="process_input",
                parameters={"focus": "decision_factors", "structure": "high"}
            ),
            GoalStep(
                description="Analyze options objectively",
                action="reason_causal",
                parameters={"options": "$options", "criteria": "$criteria", "objectivity": "high"}
            ),
            GoalStep(
                description="Present structured comparison of options",
                action="generate_response",
                parameters={"response_type": "decision_matrix", "neutrality": "high", "comprehensiveness": "high"}
            ),
            GoalStep(
                description="Provide measured recommendation if requested",
                action="generate_response",
                parameters={"response_type": "recommendation", "confidence": "appropriate", "justification": "clear"}
            )
        ]
    )
]

# Map mode names to goal lists for easier lookup
MODE_GOALS_MAP = {
    "dominant": DOMINANT_GOALS,
    "intellectual": INTELLECTUAL_GOALS,
    "compassionate": COMPASSIONATE_GOALS,
    "friendly": FRIENDLY_GOALS,
    "playful": PLAYFUL_GOALS,
    "creative": CREATIVE_GOALS,
    "professional": PROFESSIONAL_GOALS
}

# Mode compatibility matrix
MODE_COMPATIBILITY_MATRIX = {
    ("dominant", "playful"): 0.7,
    ("dominant", "creative"): 0.6,
    ("dominant", "intellectual"): 0.5,
    ("dominant", "compassionate"): 0.3,
    ("dominant", "professional"): 0.3,
    ("dominant", "friendly"): 0.4,
    
    ("friendly", "playful"): 0.9,
    ("friendly", "compassionate"): 0.8,
    ("friendly", "creative"): 0.7,
    ("friendly", "intellectual"): 0.6,
    ("friendly", "professional"): 0.5,
    
    ("intellectual", "creative"): 0.8,
    ("intellectual", "professional"): 0.7,
    ("intellectual", "compassionate"): 0.6,
    ("intellectual", "playful"): 0.5,
    
    ("compassionate", "playful"): 0.6,
    ("compassionate", "creative"): 0.7,
    ("compassionate", "professional"): 0.5,
    
    ("playful", "creative"): 0.9,
    ("playful", "professional"): 0.3,
    
    ("creative", "professional"): 0.5,
}

class GoalSelectorContext:
    """Context object for goal selection operations"""
    
    def __init__(self, mode_manager=None, goal_manager=None):
        self.mode_manager = mode_manager
        self.goal_manager = goal_manager
        
        # Goal cache for performance
        self.goal_templates = {}
        self._initialize_goal_templates()
        
        # Configuration
        self.config = {
            "max_goals": 10,
            "default_limit": 3,
            "blend_threshold": 0.5,
            "keyword_similarity_threshold": 0.3,
            "cache_enabled": True
        }
        
        # Statistics
        self.stats = {
            "selections_made": 0,
            "blends_created": 0,
            "adaptations_made": 0
        }
    
    def _initialize_goal_templates(self):
        """Initialize goal templates for each mode"""
        for mode_type in ModeType:
            mode_name = mode_type.value
            if mode_name == "default":
                # Default mode uses a blend of other modes
                continue
                
            self.goal_templates[mode_name] = MODE_GOALS_MAP.get(mode_name, [])

class GoalSelector:
    """
    Selects and blends interaction goals based on the current mode distribution.
    Provides goals that proportionally represent all active modes.
    """
    
    def __init__(self, mode_manager=None, goal_manager=None):
        self.context = GoalSelectorContext(
            mode_manager=mode_manager,
            goal_manager=goal_manager
        )
        
        # Initialize agents
        self.goal_selector_agent = self._create_goal_selector()
        self.goal_blender_agent = self._create_goal_blender()
        self.goal_ranking_agent = self._create_goal_ranker()
        
        logger.info("GoalSelector initialized with blended goal capabilities")
    
    def _create_goal_selector(self) -> Agent:
        """Create an agent specialized in selecting appropriate goals"""
        return Agent(
            name="Goal_Selector",
            instructions="""
            You select appropriate interaction goals based on the current mode distribution.
            
            Your role is to:
            1. Analyze the current mode distribution
            2. Select goals that proportionally represent all active modes
            3. Consider the coherence and complementarity of selected goals
            4. Ensure each active mode is represented in the goal set
            
            Choose goals that reflect the blended nature of the interaction,
            rather than just selecting goals from the primary mode.
            """,
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.3),
            tools=[
                function_tool(self._get_current_mode_distribution, strict_mode=False),
                function_tool(self._get_goals_for_mode, strict_mode=False)
            ],
            output_type=GoalSelectionResult
        )
    
    def _create_goal_blender(self) -> Agent:
        """Create an agent specialized in blending and adapting goals"""
        return Agent(
            name="Goal_Blender",
            instructions="""
            You blend and adapt goals from multiple modes into coherent, integrated goals.
            
            Your role is to:
            1. Identify compatible goals across different modes
            2. Merge similar goals into unified blended goals
            3. Ensure the blended goals maintain coherence and clarity
            4. Track the contribution of each mode to the goal blend
            
            Create goal blends that naturally integrate aspects from different modes,
            rather than simply listing goals from each mode separately.
            """,
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.3),
            tools=[
                function_tool(self._analyze_goal_compatibility, strict_mode=False),
                function_tool(self._blend_goal_steps, strict_mode=False)
            ],
            output_type=GoalBlendingOutput
        )
    
    def _create_goal_ranker(self) -> Agent:
        """Create an agent specialized in ranking blended goals"""
        return Agent(
            name="Goal_Ranker",
            instructions="""
            You rank and prioritize blended goals based on mode distribution and context.
            
            Your role is to:
            1. Assign appropriate priorities to blended goals
            2. Ensure goals from higher-weighted modes receive higher priority
            3. Consider goal compatibility and coherence in the ranking
            4. Provide rationale for prioritization decisions
            
            Create a prioritized goal list that aligns with the mode distribution
            while maintaining a coherent goal structure.
            """,
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.2),
            tools=[
                function_tool(self._evaluate_goal_mode_alignment, strict_mode=False),
                function_tool(self._calculate_goal_priority, strict_mode=False)
            ],
            output_type=GoalRankingOutput
        )

    @staticmethod
    @function_tool(strict_mode=False)
    async def _get_current_mode_distribution(ctx: RunContextWrapper[GoalSelectorContext]) -> ModeDistributionInfo:
        """
        Get the current mode distribution from the mode manager
        
        Returns:
            Current mode distribution information
        """
        mode_manager = ctx.context.mode_manager
        
        try:
            if mode_manager and hasattr(mode_manager, 'context'):
                mode_dist = mode_manager.context.mode_distribution
                return ModeDistributionInfo(
                    mode_distribution=mode_dist.model_dump() if hasattr(mode_dist, 'model_dump') else mode_dist.dict(),
                    active_modes=[(m, w) for m, w in mode_dist.active_modes] if hasattr(mode_dist, 'active_modes') else [],
                    primary_mode=mode_manager.context.current_mode.value if hasattr(mode_manager.context.current_mode, 'value') else str(mode_manager.context.current_mode),
                    overall_confidence=mode_manager.context.overall_confidence
                )
        except Exception as e:
            logger.error(f"Error getting mode distribution: {e}")
        
        # Fallback to default
        return ModeDistributionInfo(
            mode_distribution={"default": 1.0},
            active_modes=[("default", 1.0)],
            primary_mode="default",
            overall_confidence=0.5
        )

    @staticmethod
    @function_tool(strict_mode=False)
    async def _get_goals_for_mode(ctx: RunContextWrapper[GoalSelectorContext], input_data: GoalsForModeInput) -> GoalsForModeResult:
        """
        Get appropriate goals for a specific interaction mode
        
        Args:
            input_data: Mode and options
            
        Returns:
            Goals for the mode
        """
        mode = input_data.mode.lower()
        
        try:
            # Check cache first
            if ctx.context.config.get("cache_enabled", True) and mode in ctx.context.goal_templates:
                goals = ctx.context.goal_templates[mode]
            else:
                # Get from map
                goals = MODE_GOALS_MAP.get(mode, [])
                
            return GoalsForModeResult(
                mode=mode,
                goals=goals,
                count=len(goals)
            )
        except Exception as e:
            logger.error(f"Error getting goals for mode {mode}: {e}")
            return GoalsForModeResult(
                mode=mode,
                goals=[],
                count=0
            )

    @staticmethod
    @function_tool(strict_mode=False)
    async def _analyze_goal_compatibility(
        ctx: RunContextWrapper[GoalSelectorContext],
        input_data: GoalCompatibilityInput
    ) -> GoalCompatibilityResult:
        """
        Analyze the compatibility between two goals
        
        Args:
            input_data: Two goals to analyze
            
        Returns:
            Compatibility analysis
        """
        goal1 = input_data.goal1
        goal2 = input_data.goal2
        
        # Extract key information
        desc1 = goal1.description
        desc2 = goal2.description
        source1 = goal1.source
        source2 = goal2.source
        
        # Get base mode names
        mode1 = source1.replace("_mode", "") if source1.endswith("_mode") else source1
        mode2 = source2.replace("_mode", "") if source2.endswith("_mode") else source2
        
        # Check mode compatibility
        key = (mode1, mode2)
        reverse_key = (mode2, mode1)
        
        if key in MODE_COMPATIBILITY_MATRIX:
            mode_compat = MODE_COMPATIBILITY_MATRIX[key]
        elif reverse_key in MODE_COMPATIBILITY_MATRIX:
            mode_compat = MODE_COMPATIBILITY_MATRIX[reverse_key]
        else:
            mode_compat = 0.5  # Default moderate compatibility
        
        # Calculate keyword similarity
        keywords1 = set(desc1.lower().split())
        keywords2 = set(desc2.lower().split())
        
        if keywords1 and keywords2:
            keyword_similarity = len(keywords1.intersection(keywords2)) / len(keywords1.union(keywords2))
        else:
            keyword_similarity = 0.0
        
        # Calculate overall compatibility
        overall_compatibility = (mode_compat * 0.6) + (keyword_similarity * 0.4)
        
        # Determine if goals can be blended
        blend_threshold = ctx.context.config.get("blend_threshold", 0.5)
        keyword_threshold = ctx.context.config.get("keyword_similarity_threshold", 0.3)
        can_blend = overall_compatibility >= blend_threshold and keyword_similarity >= keyword_threshold
        
        return GoalCompatibilityResult(
            mode_compatibility=mode_compat,
            keyword_similarity=keyword_similarity,
            overall_compatibility=overall_compatibility,
            can_blend=can_blend,
            mode1=mode1,
            mode2=mode2,
            compatibility_factors={
                "shared_keywords": list(keywords1.intersection(keywords2)),
                "total_keywords": len(keywords1.union(keywords2)),
                "mode_pair": f"{mode1}-{mode2}"
            }
        )

    @staticmethod
    @function_tool(strict_mode=False)
    async def _blend_goal_steps(
        ctx: RunContextWrapper[GoalSelectorContext],
        input_data: BlendGoalsInput
    ) -> BlendGoalsResult:
        """
        Blend steps from multiple goals into a unified goal
        
        Args:
            input_data: Goals to blend
            
        Returns:
            Blended goal
        """
        goals = input_data.goals
        strategy = input_data.blend_strategy
        
        if not goals:
            return BlendGoalsResult(
                blended_goal=BlendedGoal(
                    description="No goals to blend",
                    priority=0.5,
                    sources=[],
                    original_descriptions=[],
                    plan=[]
                ),
                blend_success=False,
                blend_notes="No goals provided"
            )
            
        if len(goals) == 1:
            # Single goal, convert to blended format
            goal = goals[0]
            blended = BlendedGoal(
                description=goal.description,
                priority=goal.priority,
                sources=[(goal.source, 1.0)],
                original_descriptions=[goal.description],
                plan=goal.plan,
                coherence_score=1.0
            )
            return BlendGoalsResult(
                blended_goal=blended,
                blend_success=True,
                blend_notes="Single goal converted to blended format"
            )
        
        # Extract information from all goals
        descriptions = [goal.description for goal in goals]
        sources = [goal.source for goal in goals]
        priorities = [goal.priority for goal in goals]
        
        # Create blended description
        if strategy == "balanced":
            # Extract key concepts from all descriptions
            all_keywords = set()
            for desc in descriptions:
                # Simple keyword extraction - could be more sophisticated
                words = desc.lower().split()
                keywords = [w for w in words if len(w) > 3 and w not in {'the', 'and', 'for', 'with', 'from'}]
                all_keywords.update(keywords[:5])  # Top 5 keywords per description
            
            # Create description from keywords
            sorted_keywords = sorted(all_keywords, key=len, reverse=True)[:7]
            blended_description = f"Integrated goal: {', '.join(sorted_keywords)}"
        else:
            # Simple concatenation for other strategies
            blended_description = " + ".join([d.split(':')[0] for d in descriptions])
        
        # Calculate blended priority
        # Weight by source frequency
        source_counts = {}
        for source in sources:
            clean_source = source.replace("_mode", "")
            source_counts[clean_source] = source_counts.get(clean_source, 0) + 1
        
        weighted_priority = 0.0
        for i, (source, priority) in enumerate(zip(sources, priorities)):
            clean_source = source.replace("_mode", "")
            weight = source_counts[clean_source] / len(goals)
            weighted_priority += priority * weight
        
        # Blend plans
        blended_plan = []
        seen_actions = set()
        
        # Take unique steps from each goal
        for goal in goals:
            for step in goal.plan:
                action_desc = f"{step.action}:{step.description[:20]}"
                if action_desc not in seen_actions:
                    blended_plan.append(step)
                    seen_actions.add(action_desc)
                    
                    # Limit total steps
                    if len(blended_plan) >= 6:
                        break
            
            if len(blended_plan) >= 6:
                break
        
        # Create source weights
        total_sources = len(sources)
        source_weights = [(s.replace("_mode", ""), c/total_sources) for s, c in source_counts.items()]
        
        # Calculate coherence
        coherence = 0.8 if len(set(s.replace("_mode", "") for s in sources)) <= 3 else 0.6
        
        blended_goal = BlendedGoal(
            description=blended_description,
            priority=weighted_priority,
            sources=source_weights,
            original_descriptions=descriptions,
            plan=blended_plan,
            coherence_score=coherence
        )
        
        # Update statistics
        ctx.context.stats["blends_created"] += 1
        
        return BlendGoalsResult(
            blended_goal=blended_goal,
            blend_success=True,
            blend_notes=f"Blended {len(goals)} goals using {strategy} strategy"
        )

    @staticmethod
    @function_tool(strict_mode=False)
    async def _evaluate_goal_mode_alignment(
        ctx: RunContextWrapper[GoalSelectorContext],
        input_data: GoalAlignmentInput
    ) -> GoalAlignmentResult:
        """
        Evaluate how well a goal aligns with the current mode distribution
        
        Args:
            input_data: Goal and mode distribution
            
        Returns:
            Alignment evaluation
        """
        goal = input_data.goal
        mode_distribution = input_data.mode_distribution
        
        # Handle both InteractionGoal and BlendedGoal
        if isinstance(goal, BlendedGoal):
            sources = goal.sources
        else:
            # InteractionGoal
            source = goal.source.replace("_mode", "")
            sources = [(source, 1.0)]
        
        total_alignment = 0.0
        alignments = {}
        
        # Calculate alignment for each source
        for source, source_weight in sources:
            source = source.replace("_mode", "")
            
            # Check alignment with each mode in distribution
            for mode, mode_weight in mode_distribution.items():
                if mode_weight < 0.1:
                    continue
                
                # Direct match gets full alignment
                if source == mode:
                    alignment = 1.0 * source_weight * mode_weight
                else:
                    # Check compatibility for partial alignment
                    key = (source, mode)
                    reverse_key = (mode, source)
                    
                    if key in MODE_COMPATIBILITY_MATRIX:
                        compat = MODE_COMPATIBILITY_MATRIX[key]
                    elif reverse_key in MODE_COMPATIBILITY_MATRIX:
                        compat = MODE_COMPATIBILITY_MATRIX[reverse_key]
                    else:
                        compat = 0.3  # Default low compatibility
                    
                    alignment = compat * source_weight * mode_weight
                
                total_alignment += alignment
                alignments[mode] = alignments.get(mode, 0) + alignment
        
        return GoalAlignmentResult(
            goal_description=goal.description,
            total_alignment=min(1.0, total_alignment),  # Cap at 1.0
            alignments_by_mode=alignments,
            alignment_notes=f"Goal aligns with {len([m for m, a in alignments.items() if a > 0.1])} active modes"
        )

    @staticmethod
    @function_tool(strict_mode=False)
    async def _calculate_goal_priority(
        ctx: RunContextWrapper[GoalSelectorContext],
        input_data: GoalPriorityInput
    ) -> GoalPriorityResult:
        """
        Calculate adjusted priority for a goal based on mode alignment
        
        Args:
            input_data: Goal, mode distribution, and alignment score
            
        Returns:
            Adjusted priority
        """
        goal = input_data.goal
        mode_distribution = input_data.mode_distribution
        alignment_score = input_data.alignment_score
        
        # Get base priority
        base_priority = goal.priority
        
        # Find primary mode
        if mode_distribution:
            primary_mode, primary_weight = max(mode_distribution.items(), key=lambda x: x[1])
        else:
            primary_mode, primary_weight = None, 0
        
        # Calculate primary mode boost
        primary_boost = 0.0
        if primary_mode:
            if isinstance(goal, BlendedGoal):
                # Check if primary mode is in sources
                for source, weight in goal.sources:
                    if source.replace("_mode", "") == primary_mode:
                        primary_boost = 0.15 * weight
                        break
            else:
                # InteractionGoal
                if goal.source.replace("_mode", "") == primary_mode:
                    primary_boost = 0.15
        
        # Calculate adjusted priority
        # Base priority (40%) + Alignment (40%) + Primary boost (20%)
        adjusted_priority = (base_priority * 0.4) + (alignment_score * 0.4) + (primary_boost * 0.2)
        
        # Ensure in valid range
        adjusted_priority = max(0.1, min(1.0, adjusted_priority))
        
        return GoalPriorityResult(
            adjusted_priority=adjusted_priority,
            base_priority=base_priority,
            priority_factors={
                "base_component": base_priority * 0.4,
                "alignment_component": alignment_score * 0.4,
                "primary_boost": primary_boost * 0.2,
                "primary_mode": primary_mode or "none"
            }
        )
    
    async def select_goals(self, mode_distribution: Dict[str, float], limit: int = 3) -> List[Union[InteractionGoal, BlendedGoal]]:
        """
        Select appropriate goals based on the mode distribution
        
        Args:
            mode_distribution: Current mode distribution
            limit: Maximum number of goals to select
            
        Returns:
            List of selected interaction goals
        """
        # Validate limit
        limit = min(limit, self.context.config.get("max_goals", 10))
        
        with trace(workflow_name="select_blended_goals"):
            # Prepare selection input
            selection_input = GoalSelectionInput(
                mode_distribution=mode_distribution,
                limit=limit,
                selection_strategy="balanced"
            )
            
            # Prepare prompt
            prompt = f"""
            Select the most appropriate interaction goals based on:
            
            MODE DISTRIBUTION: {json.dumps(mode_distribution)}
            LIMIT: {limit} goals
            
            Select goals that best represent the current mode distribution.
            Consider how different modes can be represented proportionally in the goal set.
            """
            
            # Run the goal selector agent
            result = await Runner.run(
                self.goal_selector_agent, 
                prompt, 
                context=self.context,
                run_config={
                    "workflow_name": "GoalSelection",
                    "trace_metadata": {"active_modes": [m for m, w in mode_distribution.items() if w >= 0.2]}
                }
            )
            
            selected_goals = result.final_output.selected_goals
            
            # Blend similar goals
            blend_prompt = f"""
            Blend the selected goals into coherent, unified goals:
            
            SELECTED GOALS: {[g.model_dump() if hasattr(g, 'model_dump') else g for g in selected_goals]}
            MODE DISTRIBUTION: {json.dumps(mode_distribution)}
            
            Identify compatible goals and blend them into unified goals that
            represent multiple modes, rather than keeping separate goals for each mode.
            """
            
            blend_result = await Runner.run(
                self.goal_blender_agent, 
                blend_prompt, 
                context=self.context,
                run_config={
                    "workflow_name": "GoalBlending",
                    "trace_metadata": {"num_goals": len(selected_goals)}
                }
            )
            blended_goals = blend_result.final_output.blended_goals
            
            # Rank and prioritize goals
            ranking_prompt = f"""
            Rank and prioritize the blended goals:
            
            BLENDED GOALS: {[g.model_dump() for g in blended_goals]}
            MODE DISTRIBUTION: {json.dumps(mode_distribution)}
            
            Assign priorities that align with the mode distribution,
            giving higher priorities to goals from dominant modes.
            """
            
            ranking_result = await Runner.run(
                self.goal_ranking_agent, 
                ranking_prompt, 
                context=self.context,
                run_config={
                    "workflow_name": "GoalRanking",
                    "trace_metadata": {"num_goals": len(blended_goals)}
                }
            )
            
            ranked_goals = ranking_result.final_output.ranked_goals
            
            # Update statistics
            self.context.stats["selections_made"] += 1
            
            # Return limited number of goals
            return ranked_goals[:limit]
    
    async def adapt_goal(self, goal: Union[InteractionGoal, BlendedGoal], context: Dict[str, Any]) -> Union[InteractionGoal, BlendedGoal]:
        """
        Adapt a goal to a specific context
        
        Args:
            goal: The goal to adapt
            context: Conversation context
            
        Returns:
            Adapted goal
        """
        adaptation_input = GoalAdaptationInput(
            goal=goal,
            context=context,
            adaptation_depth="moderate"
        )
        
        # Create adapted copy
        if isinstance(goal, BlendedGoal):
            adapted = BlendedGoal(
                description=goal.description,
                priority=goal.priority,
                sources=goal.sources.copy(),
                original_descriptions=goal.original_descriptions.copy(),
                plan=goal.plan.copy(),
                coherence_score=goal.coherence_score
            )
        else:
            adapted = InteractionGoal(
                description=goal.description,
                priority=goal.priority,
                source=goal.source,
                plan=goal.plan.copy(),
                metadata=goal.metadata.copy() if goal.metadata else None
            )
        
        # Adapt description
        for key, value in context.items():
            if isinstance(value, str) and f"${key}" in adapted.description:
                adapted.description = adapted.description.replace(f"${key}", value)
        
        # Adapt plan steps
        for step in adapted.plan:
            # Adapt description
            for key, value in context.items():
                if isinstance(value, str) and f"${key}" in step.description:
                    step.description = step.description.replace(f"${key}", value)
            
            # Adapt parameters
            for param_key, param_value in step.parameters.items():
                if isinstance(param_value, str):
                    for key, value in context.items():
                        if isinstance(value, str) and f"${key}" in param_value:
                            step.parameters[param_key] = param_value.replace(f"${key}", value)
        
        # Update statistics
        self.context.stats["adaptations_made"] += 1
        
        return adapted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return self.context.stats.copy()
    
    def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """Update configuration"""
        try:
            valid_keys = {
                "max_goals",
                "default_limit",
                "blend_threshold",
                "keyword_similarity_threshold",
                "cache_enabled"
            }
            
            for key, value in config_updates.items():
                if key in valid_keys:
                    self.context.config[key] = value
                    logger.info(f"Updated config {key} to {value}")
                    
            return True
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return False

# Legacy function for backward compatibility
async def get_goals_for_mode(mode: str) -> List[InteractionGoal]:
    """Get appropriate goals for a specific interaction mode"""
    try:
        mode_name = mode.lower()
        
        # If it's a single mode, return direct goals
        if mode_name in MODE_GOALS_MAP:
            return MODE_GOALS_MAP[mode_name]
            
        # Try handling mode enum
        try:
            mode_type = ModeType(mode_name)
            return MODE_GOALS_MAP.get(mode_type.value, [])
        except:
            # Return default set of goals for unknown modes
            # Combine some common goals from different modes
            default_goals = []
            if FRIENDLY_GOALS:
                default_goals.extend(FRIENDLY_GOALS[:1])
            if INTELLECTUAL_GOALS:
                default_goals.extend(INTELLECTUAL_GOALS[:1])
            if PLAYFUL_GOALS:
                default_goals.extend(PLAYFUL_GOALS[:1])
            return default_goals
    except Exception as e:
        logger.error(f"Error in legacy get_goals_for_mode: {e}")
        return []
