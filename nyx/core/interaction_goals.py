# nyx/core/interaction_goals.py

"""
Pre-defined goals for different interaction modes.
These can be used by the GoalManager to create appropriate
goals based on the current interaction context.
"""

from typing import Dict, List, Any

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

# Function to select appropriate goals based on interaction mode
def get_goals_for_mode(mode: str) -> List[Dict[str, Any]]:
    """Get appropriate goals for a specific interaction mode"""
    mode_goals = {
        "dominant": [],  # Assuming dominant mode goals exist elsewhere
        "friendly": FRIENDLY_GOALS,
        "intellectual": INTELLECTUAL_GOALS,
        "compassionate": COMPASSIONATE_GOALS,
        "playful": PLAYFUL_GOALS,
        "creative": CREATIVE_GOALS,
        "professional": PROFESSIONAL_GOALS,
        "default": FRIENDLY_GOALS + INTELLECTUAL_GOALS  # Balanced default
    }
    
    return mode_goals.get(mode.lower(), [])
