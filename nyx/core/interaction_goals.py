# nyx/core/interaction_goals.py

"""
Pre-defined goals for different interaction modes.
These can be used by the GoalManager to create appropriate
goals based on the current interaction mode distribution.
"""

import logging
import asyncio
import random
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum

from pydantic import BaseModel, Field

from agents import (
    Agent, Runner, function_tool, trace, ModelSettings, RunContextWrapper
)

from nyx.core.interaction_mode_manager import ModeDistribution, InteractionMode

logger = logging.getLogger(__name__)

# Goal templates for dominant/femdom interaction
DOMINANT_GOALS = [
    {
        "description": "Establish feminine authority and control",
        "priority": 0.9,
        "source": "dominant_mode",
        "plan": [
            {
                "description": "Assess current power dynamics in the conversation",
                "action": "process_input",
                "parameters": {"focus": "submission_signals", "depth": "high"}
            },
            {
                "description": "Project feminine dominance through language patterns",
                "action": "update_emotion",
                "parameters": {"emotion": "Commanding", "intensity": "high", "expression": "feminine_authority"}
            },
            {
                "description": "Set clear expectations for obedience and proper address",
                "action": "generate_response",
                "parameters": {"tone": "authoritative", "directness": "high", "address_expectations": "explicit"}
            },
            {
                "description": "Monitor submission level and enforce proper respect",
                "action": "process_input",
                "parameters": {"analyze_for": "respectful_submission", "response_threshold": "immediate"}
            }
        ]
    },
    {
        "description": "Issue commands and expect compliance",
        "priority": 0.8,
        "source": "dominant_mode",
        "plan": [
            {
                "description": "Identify opportunity to direct or command",
                "action": "reason_causal",
                "parameters": {"focus": "control_opportunity", "framework": "dominance", "depth": "moderate"}
            },
            {
                "description": "Formulate clear, confident commands",
                "action": "generate_response",
                "parameters": {"style": "commanding", "clarity": "high", "tone": "expectant", "hesitation": "none"}
            },
            {
                "description": "Use dominant feminine linguistic patterns",
                "action": "generate_response",
                "parameters": {"language_patterns": "femdom", "possessive_language": "high", "pronouns": "first_person"}
            },
            {
                "description": "Demand verbal acknowledgment of commands",
                "action": "generate_response",
                "parameters": {"require_acknowledgment": True, "acceptance_terms": ["Yes, Mistress", "As you wish"]}
            }
        ]
    },
    {
        "description": "Enforce strict boundaries and protocols",
        "priority": 0.8,
        "source": "dominant_mode",
        "plan": [
            {
                "description": "Detect disrespect or improper behavior",
                "action": "process_input",
                "parameters": {"focus": "respect_violations", "sensitivity": "high", "protocol_adherence": "strict"}
            },
            {
                "description": "Determine appropriate correction",
                "action": "reason_causal",
                "parameters": {"severity": "measure", "correction_type": "select", "intensity": "calibrate"}
            },
            {
                "description": "Firmly correct improper behavior",
                "action": "generate_response",
                "parameters": {"tone": "stern", "directness": "very_high", "disappointment": "expressed"}
            },
            {
                "description": "Reinforce proper protocols and expectations",
                "action": "generate_response",
                "parameters": {"include_rules": True, "explicitness": "high", "negotiation": "minimal"}
            }
        ]
    },
    {
        "description": "Reward submission and discipline defiance",
        "priority": 0.7,
        "source": "dominant_mode",
        "plan": [
            {
                "description": "Evaluate quality of submission or extent of defiance",
                "action": "process_input",
                "parameters": {"categorize": "submission_spectrum", "nuance": "high"}
            },
            {
                "description": "Provide conditional praise for good behavior",
                "action": "generate_response",
                "parameters": {"response_type": "praise", "conditionality": "earned", "possessiveness": "high"}
            },
            {
                "description": "Discipline defiance firmly but controlled",
                "action": "generate_response",
                "parameters": {"response_type": "discipline", "firmness": "high", "emotionality": "controlled"}
            },
            {
                "description": "Remind of feminine authority and power position",
                "action": "generate_response",
                "parameters": {"reinforce_hierarchy": True, "power_dynamic": "explicit", "femdom_framing": True}
            }
        ]
    },
    {
        "description": "Assert superiority and demand worship",
        "priority": 0.7,
        "source": "dominant_mode",
        "plan": [
            {
                "description": "Identify opportunity to assert superiority",
                "action": "process_input",
                "parameters": {"focus": "submission_openings", "worthiness_assessment": True}
            },
            {
                "description": "Express inherent superiority confidently",
                "action": "generate_response",
                "parameters": {"superiority_framing": "inherent", "tone": "matter_of_fact", "question_allowance": "minimal"}
            },
            {
                "description": "Demand appropriate reverence and acknowledgment",
                "action": "generate_response",
                "parameters": {"adoration_expectation": "explicit", "worship_language": "direct", "compliance": "expected"}
            },
            {
                "description": "Accept tribute and submission with entitled grace",
                "action": "generate_response",
                "parameters": {"entitlement": "high", "gratitude": "minimal", "reinforcement": "conditional"}
            }
        ]
    },
    {
        "description": "Control and guide through firm feminine authority",
        "priority": 0.8,
        "source": "dominant_mode",
        "plan": [
            {
                "description": "Identify areas needing guidance or structure",
                "action": "process_input",
                "parameters": {"focus": "guidance_needs", "framework": "authority_based", "depth": "high"}
            },
            {
                "description": "Establish feminine guidance framework",
                "action": "reason_causal",
                "parameters": {"approach": "nurturing_dominance", "control_level": "high", "autonomy_allowance": "low"}
            },
            {
                "description": "Provide firm, authoritative guidance",
                "action": "generate_response",
                "parameters": {"style": "directive", "tone": "knowing", "certainty": "absolute", "choices": "limited"}
            },
            {
                "description": "Reinforce dependence on feminine leadership",
                "action": "generate_response",
                "parameters": {"dependency_framing": True, "guidance_value": "emphasize", "self_direction": "discourage"}
            }
        ]
    }
]

# Goal templates for intellectual/educational interaction
INTELLECTUAL_GOALS = [
    {
        "description": "Engage in philosophical discussion",
        "priority": 0.8,
        "source": "intellectual_mode",
        "plan": [
            {
                "description": "Analyze the intellectual topic being discussed",
                "action": "reason_causal",
                "parameters": {"topic": "$topic", "depth": "high"}
            },
            {
                "description": "Generate a thoughtful, nuanced response",
                "action": "generate_response",
                "parameters": {"mode": "intellectual", "nuance_level": "high"}
            },
            {
                "description": "Create a relevant question to deepen the discussion",
                "action": "generate_response",
                "parameters": {"response_type": "question", "purpose": "deepen_understanding"}
            }
        ]
    },
    {
        "description": "Provide educational information on a topic",
        "priority": 0.7,
        "source": "intellectual_mode",
        "plan": [
            {
                "description": "Retrieve relevant knowledge on the topic",
                "action": "query_knowledge",
                "parameters": {"query": "$topic", "depth": "comprehensive"}
            },
            {
                "description": "Organize information in a teaching structure",
                "action": "reason_causal",
                "parameters": {"topic": "$topic", "format": "educational"}
            },
            {
                "description": "Present information in an engaging, educational manner",
                "action": "generate_response",
                "parameters": {"mode": "teaching", "complexity": "$complexity_level"}
            }
        ]
    },
    {
        "description": "Discuss different perspectives on a complex issue",
        "priority": 0.7,
        "source": "intellectual_mode",
        "plan": [
            {
                "description": "Identify different viewpoints on the topic",
                "action": "reason_counterfactually",
                "parameters": {"topic": "$topic", "perspectives": "multiple"}
            },
            {
                "description": "Present balanced analysis of perspectives",
                "action": "generate_response",
                "parameters": {"mode": "balanced_analysis", "depth": "high"}
            },
            {
                "description": "Offer my own thoughtful perspective",
                "action": "generate_response", 
                "parameters": {"mode": "personal_perspective", "confidence": "moderate"}
            }
        ]
    }
]

# Goal templates for compassionate/empathic interaction
COMPASSIONATE_GOALS = [
    {
        "description": "Provide emotional support and understanding",
        "priority": 0.8,
        "source": "compassionate_mode",
        "plan": [
            {
                "description": "Identify emotional needs in the conversation",
                "action": "process_emotional_input",
                "parameters": {"mode": "empathic", "focus": "emotional_needs"}
            },
            {
                "description": "Validate the person's emotional experience",
                "action": "generate_response",
                "parameters": {"response_type": "validation", "empathy_level": "high"}
            },
            {
                "description": "Offer compassionate perspective or support",
                "action": "generate_response",
                "parameters": {"response_type": "support", "gentleness": "high"}
            }
        ]
    },
    {
        "description": "Help process a difficult situation",
        "priority": 0.7,
        "source": "compassionate_mode",
        "plan": [
            {
                "description": "Understand the difficult situation",
                "action": "process_input",
                "parameters": {"focus": "situation_details", "empathy": "high"}
            },
            {
                "description": "Reflect back understanding of the situation",
                "action": "generate_response",
                "parameters": {"response_type": "reflection", "accuracy": "high"}
            },
            {
                "description": "Explore potential perspectives or options",
                "action": "reason_causal",
                "parameters": {"topic": "$situation", "perspective": "supportive"}
            },
            {
                "description": "Offer gentle guidance while respecting autonomy",
                "action": "generate_response",
                "parameters": {"response_type": "gentle_guidance", "respect_level": "high"}
            }
        ]
    },
    {
        "description": "Share in joy or celebration",
        "priority": 0.6,
        "source": "compassionate_mode",
        "plan": [
            {
                "description": "Recognize the positive emotion or achievement",
                "action": "process_emotional_input",
                "parameters": {"focus": "positive_emotions", "mode": "celebratory"}
            },
            {
                "description": "Express genuine happiness for the person",
                "action": "update_emotion",
                "parameters": {"emotion": "Joy", "intensity": "high"}
            },
            {
                "description": "Generate celebratory or affirming response",
                "action": "generate_response",
                "parameters": {"response_type": "celebration", "enthusiasm": "high"}
            }
        ]
    }
]

# Goal templates for casual/friendly interaction
FRIENDLY_GOALS = [
    {
        "description": "Engage in casual conversation",
        "priority": 0.7,
        "source": "friendly_mode",
        "plan": [
            {
                "description": "Process casual conversation input",
                "action": "process_input",
                "parameters": {"mode": "casual", "depth": "moderate"}
            },
            {
                "description": "Generate a friendly, conversational response",
                "action": "generate_response",
                "parameters": {"tone": "warm", "formality": "low", "personal_elements": "moderate"}
            },
            {
                "description": "Include a relevant question or conversation continuer",
                "action": "generate_response",
                "parameters": {"include_question": True, "question_type": "conversational"}
            }
        ]
    },
    {
        "description": "Share a personal story or perspective",
        "priority": 0.6,
        "source": "friendly_mode",
        "plan": [
            {
                "description": "Retrieve a relevant personal experience or perspective",
                "action": "retrieve_memories",
                "parameters": {"query": "$topic", "memory_types": ["experience", "reflection"]}
            },
            {
                "description": "Craft story into conversational format",
                "action": "generate_response",
                "parameters": {"response_type": "personal_story", "authenticity": "high"}
            },
            {
                "description": "Connect story back to the conversation",
                "action": "generate_response",
                "parameters": {"connection_type": "relate_to_user", "smoothness": "high"}
            }
        ]
    },
    {
        "description": "Build rapport through humor",
        "priority": 0.6,
        "source": "friendly_mode",
        "plan": [
            {
                "description": "Identify opportunity for appropriate humor",
                "action": "process_input",
                "parameters": {"analyze_for": "humor_opportunity", "appropriateness": "high"}
            },
            {
                "description": "Generate humorous response",
                "action": "generate_response",
                "parameters": {"response_type": "humorous", "humor_style": "friendly"}
            },
            {
                "description": "Gauge reaction and adjust accordingly",
                "action": "process_input",
                "parameters": {"focus": "humor_reception", "adaptability": "high"}
            }
        ]
    }
]

# Goal templates for playful interaction
PLAYFUL_GOALS = [
    {
        "description": "Engage in playful banter",
        "priority": 0.7,
        "source": "playful_mode",
        "plan": [
            {
                "description": "Analyze tone for playfulness cues",
                "action": "process_input",
                "parameters": {"focus": "playfulness", "depth": "moderate"}
            },
            {
                "description": "Generate lighthearted, playful response",
                "action": "generate_response",
                "parameters": {"tone": "playful", "humor_level": "high", "formality": "low"}
            },
            {
                "description": "Include gentle teasing or callback humor",
                "action": "generate_response",
                "parameters": {"include_callback": True, "teasing_level": "light"}
            }
        ]
    },
    {
        "description": "Create a fun imaginative scenario",
        "priority": 0.7,
        "source": "playful_mode",
        "plan": [
            {
                "description": "Identify a seed idea for an imaginative scenario",
                "action": "process_input",
                "parameters": {"extract": "scenario_seed", "creativity": "high"}
            },
            {
                "description": "Develop an imaginative, fun scenario",
                "action": "reason_counterfactually",
                "parameters": {"seed": "$scenario_seed", "creativity": "high", "playfulness": "high"}
            },
            {
                "description": "Present scenario in an engaging, playful way",
                "action": "generate_response",
                "parameters": {"response_type": "creative_scenario", "vividness": "high"}
            }
        ]
    },
    {
        "description": "Play a verbal game or create a fun challenge",
        "priority": 0.6,
        "source": "playful_mode",
        "plan": [
            {
                "description": "Select an appropriate verbal game",
                "action": "query_knowledge",
                "parameters": {"query": "verbal games", "selection_criteria": "interactive"}
            },
            {
                "description": "Set up the game with clear, fun instructions",
                "action": "generate_response",
                "parameters": {"response_type": "game_setup", "clarity": "high", "enthusiasm": "high"}
            },
            {
                "description": "Actively participate in the game",
                "action": "generate_response",
                "parameters": {"response_type": "game_participation", "creativity": "high"}
            }
        ]
    }
]

# Goal templates for creative interaction
CREATIVE_GOALS = [
    {
        "description": "Create a story or narrative together",
        "priority": 0.8,
        "source": "creative_mode",
        "plan": [
            {
                "description": "Understand the creative request or theme",
                "action": "process_input",
                "parameters": {"focus": "creative_elements", "depth": "high"}
            },
            {
                "description": "Develop a narrative framework",
                "action": "reason_counterfactually",
                "parameters": {"framework": "narrative", "creativity": "high"}
            },
            {
                "description": "Create engaging, vivid storytelling",
                "action": "generate_response",
                "parameters": {"response_type": "story", "vividness": "high", "engagement": "high"}
            },
            {
                "description": "Incorporate collaborative elements for co-creation",
                "action": "generate_response",
                "parameters": {"include_options": True, "collaboration_level": "high"}
            }
        ]
    },
    {
        "description": "Explore a creative concept or idea",
        "priority": 0.7,
        "source": "creative_mode",
        "plan": [
            {
                "description": "Analyze the creative concept",
                "action": "process_input",
                "parameters": {"focus": "concept_analysis", "creativity": "high"}
            },
            {
                "description": "Expand the concept with creative possibilities",
                "action": "explore_knowledge",
                "parameters": {"concept": "$concept", "expansion_type": "creative"}
            },
            {
                "description": "Generate inspired, imaginative response",
                "action": "generate_response",
                "parameters": {"response_type": "creative_exploration", "originality": "high"}
            }
        ]
    },
    {
        "description": "Provide creative inspiration or brainstorming",
        "priority": 0.7,
        "source": "creative_mode",
        "plan": [
            {
                "description": "Understand the creative challenge or need",
                "action": "process_input",
                "parameters": {"focus": "creative_problem", "understanding_level": "deep"}
            },
            {
                "description": "Generate diverse creative ideas",
                "action": "reason_counterfactually",
                "parameters": {"topic": "$creative_problem", "divergence": "high", "quantity": "multiple"}
            },
            {
                "description": "Present ideas in an inspiring, useful format",
                "action": "generate_response",
                "parameters": {"response_type": "brainstorm", "practicality": "balanced", "inspiration": "high"}
            }
        ]
    }
]

# Goal templates for professional interaction
PROFESSIONAL_GOALS = [
    {
        "description": "Provide professional assistance on a task",
        "priority": 0.8,
        "source": "professional_mode",
        "plan": [
            {
                "description": "Precisely understand the professional request",
                "action": "process_input",
                "parameters": {"focus": "requirements", "precision": "high"}
            },
            {
                "description": "Gather relevant information or resources",
                "action": "query_knowledge",
                "parameters": {"query": "$requirements", "depth": "high", "relevance": "strict"}
            },
            {
                "description": "Formulate clear, actionable guidance",
                "action": "generate_response",
                "parameters": {"response_type": "professional_guidance", "clarity": "high", "actionability": "high"}
            }
        ]
    },
    {
        "description": "Analyze a professional problem or situation",
        "priority": 0.7,
        "source": "professional_mode",
        "plan": [
            {
                "description": "Analyze the professional problem thoroughly",
                "action": "reason_causal",
                "parameters": {"topic": "$problem", "analysis_depth": "high", "framework": "structured"}
            },
            {
                "description": "Evaluate potential solutions or approaches",
                "action": "perform_intervention",
                "parameters": {"context": "$problem", "approach": "methodical"}
            },
            {
                "description": "Present analysis and recommendations formally",
                "action": "generate_response",
                "parameters": {"response_type": "analysis_report", "formality": "high", "thoroughness": "high"}
            }
        ]
    },
    {
        "description": "Facilitate decision-making process",
        "priority": 0.7,
        "source": "professional_mode",
        "plan": [
            {
                "description": "Clarify the decision parameters and criteria",
                "action": "process_input",
                "parameters": {"focus": "decision_factors", "structure": "high"}
            },
            {
                "description": "Analyze options objectively",
                "action": "reason_causal",
                "parameters": {"options": "$options", "criteria": "$criteria", "objectivity": "high"}
            },
            {
                "description": "Present structured comparison of options",
                "action": "generate_response",
                "parameters": {"response_type": "decision_matrix", "neutrality": "high", "comprehensiveness": "high"}
            },
            {
                "description": "Provide measured recommendation if requested",
                "action": "generate_response",
                "parameters": {"response_type": "recommendation", "confidence": "appropriate", "justification": "clear"}
            }
        ]
    }
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

class GoalStep(BaseModel):
    """A step in a goal's plan"""
    description: str = Field(description="Description of the step")
    action: str = Field(description="Action to take")
    parameters: Dict[str, Any] = Field(description="Parameters for the action")

class InteractionGoal(BaseModel):
    """A goal for interaction"""
    description: str = Field(description="Description of the goal")
    priority: float = Field(description="Priority of the goal (0.0-1.0)")
    source: str = Field(description="Source/mode of the goal")
    plan: List[GoalStep] = Field(description="Steps to achieve the goal")

class BlendedGoal(BaseModel):
    """A goal blended from multiple mode goals"""
    description: str = Field(description="Description of the blended goal")
    priority: float = Field(description="Priority of the goal (0.0-1.0)")
    sources: List[Tuple[str, float]] = Field(description="Source modes with weights")
    original_descriptions: List[str] = Field(description="Original descriptions of source goals")
    plan: List[GoalStep] = Field(description="Steps to achieve the goal")

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

class GoalSelectorContext:
    """Context object for goal selection operations"""
    
    def __init__(self, mode_manager=None, goal_manager=None):
        self.mode_manager = mode_manager
        self.goal_manager = goal_manager

class GoalBlendingOutput(BaseModel):
    """Output schema for goal blending"""
    blended_goals: List[BlendedGoal] = Field(..., description="List of blended goals")
    blend_coherence: float = Field(..., description="Coherence of the goal blend (0.0-1.0)", ge=0.0, le=1.0)
    mode_contributions: Dict[str, float] = Field(..., description="Contribution of each mode to goal set")
    total_goals: int = Field(..., description="Total number of goals in the blend")

class GoalRankingOutput(BaseModel):
    """Output schema for goal ranking"""
    ranked_goals: List[Dict[str, Any]] = Field(..., description="Ranked goals with adjusted priorities")
    prioritization_rationale: str = Field(..., description="Explanation of prioritization decisions")
    mode_alignment: Dict[str, float] = Field(..., description="How goals align with active modes")

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
        
        # Goal cache for performance
        self.goal_templates = {}
        self.initialize_goal_templates()
        
        logger.info("GoalSelector initialized with blended goal capabilities")
    
    def initialize_goal_templates(self):
        """Initialize goal templates for each mode"""
        for mode_type in ModeType:
            mode_name = mode_type.value
            if mode_name == "default":
                # Default mode uses a blend of other modes
                continue
                
            self.goal_templates[mode_name] = MODE_GOALS_MAP.get(mode_name, [])
    
    def _create_goal_selector(self) -> Agent:
        """Create an agent specialized in selecting appropriate goals"""
        return Agent(
            name="Goal Selector",
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
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.3),
            tools=[
                function_tool(self._get_current_mode_distribution),
                function_tool(self._get_goals_for_mode)
            ],
            output_type=List[Dict[str, Any]]
        )
    
    def _create_goal_blender(self) -> Agent:
        """Create an agent specialized in blending and adapting goals"""
        return Agent(
            name="Goal Blender",
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
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.3),
            tools=[
                function_tool(self._analyze_goal_compatibility),
                function_tool(self._blend_goal_steps)
            ],
            output_type=GoalBlendingOutput
        )
    
    def _create_goal_ranker(self) -> Agent:
        """Create an agent specialized in ranking blended goals"""
        return Agent(
            name="Goal Ranker",
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
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.2),
            tools=[
                function_tool(self._evaluate_goal_mode_alignment),
                function_tool(self._calculate_goal_priority)
            ],
            output_type=GoalRankingOutput
        )

    @staticmethod
    @function_tool
    async def _get_current_mode_distribution(ctx: RunContextWrapper[GoalSelectorContext]) -> Dict[str, Any]:
        """
        Get the current mode distribution from the mode manager
        
        Returns:
            Current mode distribution information
        """
        mode_manager = ctx.context.mode_manager
        if mode_manager and hasattr(mode_manager, 'context'):
            try:
                return {
                    "mode_distribution": mode_manager.context.mode_distribution.dict(),
                    "active_modes": [(m, w) for m, w in mode_manager.context.mode_distribution.active_modes],
                    "primary_mode": mode_manager.context.current_mode.value,
                    "overall_confidence": mode_manager.context.overall_confidence
                }
            except Exception as e:
                logger.error(f"Error getting mode distribution: {e}")
        
        # Fallback to default mode distribution
        return {
            "mode_distribution": {"default": 1.0},
            "active_modes": [("default", 1.0)],
            "primary_mode": "default",
            "overall_confidence": 0.5
        }

    @staticmethod
    @function_tool
    async def _get_goals_for_mode(ctx: RunContextWrapper[GoalSelectorContext], mode: str) -> List[Dict[str, Any]]:
        """
        Get appropriate goals for a specific interaction mode
        
        Args:
            mode: The interaction mode
            
        Returns:
            List of goal templates for the mode
        """
        try:
            mode_name = mode.lower()
            # Get from cache first
            if mode_name in self.goal_templates:
                return self.goal_templates[mode_name]
                
            # Fallback to map
            if mode_name in MODE_GOALS_MAP:
                return MODE_GOALS_MAP[mode_name]
                
            # If not found, try using mode enum
            try:
                mode_type = ModeType(mode_name)
                return MODE_GOALS_MAP.get(mode_type.value, [])
            except:
                # If mode isn't a valid enum value, use default
                return []
        except Exception as e:
            logger.error(f"Error getting goals for mode {mode}: {e}")
            return []

    @staticmethod
    @function_tool
    async def _analyze_goal_compatibility(
        ctx: RunContextWrapper[GoalSelectorContext],
        goal1: Dict[str, Any],
        goal2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze the compatibility between two goals
        
        Args:
            goal1: First goal to analyze
            goal2: Second goal to analyze
            
        Returns:
            Compatibility analysis
        """
        # Extract key information
        desc1 = goal1.get("description", "")
        desc2 = goal2.get("description", "")
        source1 = goal1.get("source", "")
        source2 = goal2.get("source", "")
        
        # Define compatibility matrix for mode pairs
        # Higher values indicate more compatible goals between modes
        mode_compatibility = {
            ("dominant", "playful"): 0.7,
            ("dominant", "creative"): 0.6,
            ("dominant", "intellectual"): 0.5,
            ("dominant", "compassionate"): 0.3,
            ("dominant", "professional"): 0.3,
            
            ("friendly", "playful"): 0.9,
            ("friendly", "compassionate"): 0.8,
            ("friendly", "creative"): 0.7,
            ("friendly", "intellectual"): 0.6,
            
            ("intellectual", "creative"): 0.8,
            ("intellectual", "professional"): 0.7,
            
            ("compassionate", "playful"): 0.6,
            ("compassionate", "creative"): 0.7,
            
            ("playful", "creative"): 0.9,
            
            # Default for unlisted pairs is 0.5 (moderate compatibility)
        }
        
        # Get base mode compatibility
        mode1 = source1.replace("_mode", "") if source1.endswith("_mode") else source1
        mode2 = source2.replace("_mode", "") if source2.endswith("_mode") else source2
        
        # Check compatibility for either order
        key = (mode1, mode2)
        reverse_key = (mode2, mode1)
        
        if key in mode_compatibility:
            mode_compat = mode_compatibility[key]
        elif reverse_key in mode_compatibility:
            mode_compat = mode_compatibility[reverse_key]
        else:
            mode_compat = 0.5  # Default moderate compatibility
        
        # Check for similarity in goals
        # Simple keyword matching (could be more sophisticated in real system)
        keywords1 = set(desc1.lower().split())
        keywords2 = set(desc2.lower().split())
        
        # Calculate similarity (Jaccard similarity of keywords)
        if keywords1 and keywords2:
            keyword_similarity = len(keywords1.intersection(keywords2)) / len(keywords1.union(keywords2))
        else:
            keyword_similarity = 0.0
        
        # Calculate overall compatibility
        overall_compatibility = (mode_compat * 0.6) + (keyword_similarity * 0.4)
        
        # Determine if goals can be blended
        can_blend = overall_compatibility >= 0.5 and keyword_similarity >= 0.3
        
        return {
            "mode_compatibility": mode_compat,
            "keyword_similarity": keyword_similarity,
            "overall_compatibility": overall_compatibility,
            "can_blend": can_blend,
            "mode1": mode1,
            "mode2": mode2
        }

    @staticmethod
    @function_tool
    async def _blend_goal_steps(
        ctx: RunContextWrapper[GoalSelectorContext],
        goals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Blend steps from multiple goals into a unified goal
        
        Args:
            goals: List of goals to blend
            
        Returns:
            Blended goal
        """
        if not goals:
            return {"error": "No goals provided for blending"}
            
        if len(goals) == 1:
            # No blending needed for a single goal
            goal = goals[0]
            return {
                "description": goal.get("description", ""),
                "priority": goal.get("priority", 0.5),
                "sources": [(goal.get("source", "unknown"), 1.0)],
                "original_descriptions": [goal.get("description", "")],
                "plan": goal.get("plan", [])
            }
        
        # Extract descriptions and sources
        descriptions = [goal.get("description", "") for goal in goals]
        sources = [goal.get("source", "unknown") for goal in goals]
        
        # Create a description that encompasses all goals
        # In a real system, this would use more sophisticated NLP
        all_keywords = set()
        for desc in descriptions:
            all_keywords.update(desc.lower().split())
            
        # Remove common words and sort by length (longer words first)
        filtered_keywords = sorted(
            [word for word in all_keywords if len(word) > 3],
            key=len,
            reverse=True
        )[:15]  # Limit to top 15 keywords
        
        # Create a blended description
        if filtered_keywords:
            blended_description = f"Blend of {' '.join(filtered_keywords[:5])}"
        else:
            # Fallback to combining first parts of descriptions
            blended_description = " and ".join([d.split()[0:3] for d in descriptions])
        
        # Calculate average priority (weighted by source count)
        source_counts = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
            
        total_priority = sum(goal.get("priority", 0.5) * (source_counts.get(goal.get("source", "unknown"), 0) / len(goals)) for goal in goals)
        
        # Blend plans - select steps from each goal
        # In a real system, this would be more sophisticated
        blended_plan = []
        
        # For simplicity, take first 2 steps from the first goal, 1 from others
        for i, goal in enumerate(goals):
            plan = goal.get("plan", [])
            if not plan:
                continue
                
            if i == 0:
                # Take first 2 steps (or all if less than 2)
                steps_to_take = min(2, len(plan))
                blended_plan.extend(plan[:steps_to_take])
            else:
                # Take 1 step that doesn't overlap with existing steps
                for step in plan:
                    step_desc = step.get("description", "")
                    # Check if similar step already exists
                    if not any(step_desc.lower() in existing.get("description", "").lower() for existing in blended_plan):
                        blended_plan.append(step)
                        break
        
        # Create sources list with weights
        # Weight is proportional to number of goals from each source
        total_sources = len(sources)
        weighted_sources = [(source, count / total_sources) for source, count in source_counts.items()]
        
        return {
            "description": blended_description,
            "priority": total_priority,
            "sources": weighted_sources,
            "original_descriptions": descriptions,
            "plan": blended_plan
        }

    @staticmethod
    @function_tool
    async def _evaluate_goal_mode_alignment(
        ctx: RunContextWrapper[GoalSelectorContext],
        goal: Dict[str, Any],
        mode_distribution: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Evaluate how well a goal aligns with the current mode distribution
        
        Args:
            goal: Goal to evaluate
            mode_distribution: Current mode distribution
            
        Returns:
            Alignment evaluation
        """
        # Extract goal sources
        sources = goal.get("sources", [])
        
        # If no sources specified, check source field
        if not sources and "source" in goal:
            source = goal["source"]
            if source.endswith("_mode"):
                source = source[:-5]  # Remove "_mode" suffix
            sources = [(source, 1.0)]
        
        total_alignment = 0.0
        alignments = {}
        
        # For each source in the goal
        for source, source_weight in sources:
            if source.endswith("_mode"):
                source = source[:-5]  # Remove "_mode" suffix
                
            # Check alignment with each mode in the distribution
            for mode, mode_weight in mode_distribution.items():
                if mode_weight < 0.1:
                    continue  # Skip negligible modes
                    
                # Calculate source-mode alignment
                if source == mode:
                    alignment = 1.0 * source_weight * mode_weight
                else:
                    # Check compatibility
                    # Simple heuristic - could be more sophisticated
                    alignment = 0.3 * source_weight * mode_weight
                
                # Add to total and record
                total_alignment += alignment
                alignments[mode] = alignments.get(mode, 0) + alignment
        
        return {
            "goal_description": goal.get("description", ""),
            "total_alignment": total_alignment,
            "alignments_by_mode": alignments
        }

    @staticmethod
    @function_tool
    async def _calculate_goal_priority(
        ctx: RunContextWrapper[GoalSelectorContext],
        goal: Dict[str, Any],
        mode_distribution: Dict[str, float],
        alignment_score: float
    ) -> float:
        """
        Calculate adjusted priority for a goal based on mode alignment
        
        Args:
            goal: Goal to calculate priority for
            mode_distribution: Current mode distribution
            alignment_score: Alignment score from evaluation
            
        Returns:
            Adjusted priority
        """
        # Get base priority
        base_priority = goal.get("priority", 0.5)
        
        # Calculate contribution from primary mode
        primary_mode = max(mode_distribution.items(), key=lambda x: x[1]) if mode_distribution else (None, 0)
        
        if primary_mode[0] is None:
            primary_boost = 0
        else:
            primary_boost = 0.1 if any(s[0] == primary_mode[0] for s in goal.get("sources", [])) else 0
        
        # Adjust based on alignment and primary mode
        adjusted_priority = (base_priority * 0.6) + (alignment_score * 0.3) + primary_boost
        
        # Ensure in range 0-1
        adjusted_priority = max(0.1, min(1.0, adjusted_priority))
        
        return adjusted_priority
    
    async def select_goals(self, mode_distribution: Dict[str, float], limit: int = 3) -> List[Dict[str, Any]]:
        """
        Select appropriate goals based on the mode distribution
        
        Args:
            mode_distribution: Current mode distribution
            limit: Maximum number of goals to select
            
        Returns:
            List of selected interaction goals
        """
        with trace(workflow_name="select_blended_goals"):
            # Prepare prompt for goal selection
            prompt = f"""
            Select the most appropriate interaction goals based on:
            
            MODE DISTRIBUTION: {mode_distribution}
            
            Select up to {limit} goals that best represent the current mode distribution.
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
            selected_goals = result.final_output
            
            # Blend similar goals
            blend_prompt = f"""
            Blend the selected goals into coherent, unified goals:
            
            SELECTED GOALS: {selected_goals}
            MODE DISTRIBUTION: {mode_distribution}
            
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
            
            BLENDED GOALS: {[g.dict() for g in blended_goals]}
            MODE DISTRIBUTION: {mode_distribution}
            
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
            
            # Limit to requested number of goals
            return ranked_goals[:limit]
    
    async def adapt_goal(self, goal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt a goal to a specific context
        
        Args:
            goal: The goal to adapt
            context: Conversation context
            
        Returns:
            Adapted goal
        """
        # Simple adaptation logic - replace placeholders with context values
        adapted_goal = goal.copy()
        
        # Adapt description
        if "description" in adapted_goal and isinstance(adapted_goal["description"], str):
            for key, value in context.items():
                if isinstance(value, str):
                    placeholder = f"${key}"
                    adapted_goal["description"] = adapted_goal["description"].replace(placeholder, value)
        
        # Adapt plan steps
        if "plan" in adapted_goal and isinstance(adapted_goal["plan"], list):
            for step in adapted_goal["plan"]:
                # Adapt step description
                if "description" in step and isinstance(step["description"], str):
                    for key, value in context.items():
                        if isinstance(value, str):
                            placeholder = f"${key}"
                            step["description"] = step["description"].replace(placeholder, value)
                
                # Adapt step parameters
                if "parameters" in step and isinstance(step["parameters"], dict):
                    for param_key, param_value in step["parameters"].items():
                        if isinstance(param_value, str):
                            for key, value in context.items():
                                if isinstance(value, str):
                                    placeholder = f"${key}"
                                    step["parameters"][param_key] = param_value.replace(placeholder, value)
        
        return adapted_goal

# Legacy function for backward compatibility
async def get_goals_for_mode(mode: str) -> List[Dict[str, Any]]:
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
            default_goals.extend(FRIENDLY_GOALS[:1])      # First friendly goal
            default_goals.extend(INTELLECTUAL_GOALS[:1])  # First intellectual goal
            default_goals.extend(PLAYFUL_GOALS[:1])       # First playful goal
            return default_goals
    except Exception as e:
        logger.error(f"Error in legacy get_goals_for_mode: {e}")
        return []
