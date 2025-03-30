# nyx/core/interaction_goals.py

"""
Pre-defined goals for different interaction modes.
These can be used by the GoalManager to create appropriate
goals based on the current interaction context.
"""

from typing import Dict, List, Any

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
                "parameters": {"require_acknowledgment": true, "acceptance_terms": ["Yes, Mistress", "As you wish"]}
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
                "parameters": {"include_rules": true, "explicitness": "high", "negotiation": "minimal"}
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
                "parameters": {"reinforce_hierarchy": true, "power_dynamic": "explicit", "femdom_framing": true}
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
                "parameters": {"focus": "submission_openings", "worthiness_assessment": true}
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
                "parameters": {"dependency_framing": true, "guidance_value": "emphasize", "self_direction": "discourage"}
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

class GoalSelector:
    """
    Agent-based system for selecting and managing interaction goals
    based on the current mode and context.
    """
    
    def __init__(self):
        # Load goal templates
        self.goal_templates = {
            ModeType.INTELLECTUAL: INTELLECTUAL_GOALS,
            ModeType.COMPASSIONATE: COMPASSIONATE_GOALS,
            ModeType.DOMINANT: DOMINANT_GOALS,
            ModeType.FRIENDLY: FRIENDLY_GOALS,
            ModeType.PLAYFUL: PLAYFUL_GOALS,
            ModeType.CREATIVE: CREATIVE_GOALS,
            ModeType.PROFESSIONAL: PROFESSIONAL_GOALS,
            ModeType.DEFAULT: FRIENDLY_GOALS + DOMINANT_GOALS + PLAYFUL_GOALS
        }
        
        # Initialize agents
        self.goal_selector_agent = self._create_goal_selector()
        self.goal_adapter_agent = self._create_goal_adapter()
    
    def _create_goal_selector(self) -> Agent:
        """Create an agent specialized in selecting appropriate goals"""
        return Agent(
            name="Goal Selector",
            instructions="""
            You select appropriate interaction goals based on the current mode and context.
            
            Your role is to:
            1. Analyze the current interaction mode
            2. Consider the conversation context and user needs
            3. Select the most appropriate goals from available templates
            4. Prioritize goals based on relevance and importance
            
            Choose goals that align with the current interaction mode while
            addressing the specific needs of the conversation.
            """,
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.3),
            tools=[
                self._get_goals_for_mode
            ],
            output_type=List[InteractionGoal]
        )
    
    def _create_goal_adapter(self) -> Agent:
        """Create an agent specialized in adapting goals to specific contexts"""
        return Agent(
            name="Goal Adapter",
            instructions="""
            You adapt interaction goals to specific contexts and parameters.
            
            Your role is to:
            1. Take a selected goal and modify it for the current context
            2. Replace placeholder variables with actual values
            3. Adjust priorities based on context importance
            4. Refine step descriptions and parameters for clarity
            
            Ensure the adapted goal is concrete, actionable, and directly
            relevant to the current conversation.
            """,
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.2),
            output_type=InteractionGoal
        )
    
    @function_tool
    async def _get_goals_for_mode(self, mode: str) -> List[Dict[str, Any]]:
        """
        Get appropriate goals for a specific interaction mode
        
        Args:
            mode: The interaction mode
            
        Returns:
            List of goal templates for the mode
        """
        try:
            mode_type = ModeType(mode.lower())
            return self.goal_templates.get(mode_type, self.goal_templates[ModeType.DEFAULT])
        except ValueError:
            # If mode isn't a valid enum value, use default
            return self.goal_templates[ModeType.DEFAULT]
    
    async def select_goals(self, mode: str, context: Dict[str, Any], limit: int = 3) -> List[InteractionGoal]:
        """
        Select appropriate goals based on mode and context
        
        Args:
            mode: Current interaction mode
            context: Conversation context
            limit: Maximum number of goals to select
            
        Returns:
            List of selected interaction goals
        """
        with trace(workflow_name="select_interaction_goals"):
            # Prepare prompt for goal selection
            prompt = f"""
            Select the most appropriate interaction goals based on:
            
            MODE: {mode}
            
            CONTEXT: {context}
            
            Select up to {limit} goals that best match the current interaction.
            Prioritize goals based on relevance and importance to the current conversation.
            """
            
            # Run the goal selector agent
            result = await Runner.run(self.goal_selector_agent, prompt)
            selected_goals = result.final_output
            
            return selected_goals[:limit]  # Ensure we don't exceed the limit
    
    async def adapt_goal(self, goal: InteractionGoal, context: Dict[str, Any]) -> InteractionGoal:
        """
        Adapt a goal to a specific context
        
        Args:
            goal: The goal to adapt
            context: Conversation context
            
        Returns:
            Adapted goal
        """
        with trace(workflow_name="adapt_interaction_goal"):
            # Prepare prompt for goal adaptation
            prompt = f"""
            Adapt this interaction goal to the specific context:
            
            GOAL: {goal.dict()}
            
            CONTEXT: {context}
            
            Replace any placeholder variables with actual values.
            Adjust priorities and parameters based on the specific context.
            Ensure the goal is concrete and directly applicable to the current situation.
            """
            
            # Run the goal adapter agent
            result = await Runner.run(self.goal_adapter_agent, prompt)
            adapted_goal = result.final_output
            
            return adapted_goal

# Function to get goals for a mode (backward compatibility)
async def get_goals_for_mode(mode: str) -> List[Dict[str, Any]]:
    """Get appropriate goals for a specific interaction mode"""
    selector = GoalSelector()
    goals = await selector._get_goals_for_mode(mode)
    return goals
