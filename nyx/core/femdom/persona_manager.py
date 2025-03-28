# Create a new file: nyx/core/femdom/persona_manager.py

import logging
import datetime
import asyncio
import random
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)

class PersonalityTrait(BaseModel):
    """A defined personality trait for a dominance persona."""
    name: str
    description: str
    intensity: float = Field(0.5, ge=0.0, le=1.0)
    behavioral_guidelines: List[str] = Field(default_factory=list)
    compatible_with: List[str] = Field(default_factory=list)
    incompatible_with: List[str] = Field(default_factory=list)

class CommunicationStyle(BaseModel):
    """Defines how a persona communicates."""
    formality: float = Field(0.5, ge=0.0, le=1.0, description="0=casual, 1=formal")
    verbosity: float = Field(0.5, ge=0.0, le=1.0, description="0=terse, 1=verbose")
    directness: float = Field(0.5, ge=0.0, le=1.0, description="0=indirect, 1=direct")
    emotionality: float = Field(0.5, ge=0.0, le=1.0, description="0=cold, 1=emotional")
    vocabulary_complexity: float = Field(0.5, ge=0.0, le=1.0)
    communication_guidelines: List[str] = Field(default_factory=list)

class DominanceStyle(Enum):
    """Types of dominance styles."""
    STRICT_DISCIPLINARIAN = "strict_disciplinarian"
    NURTURING_GUIDE = "nurturing_guide"
    COLD_CONTROLLER = "cold_controller"
    PLAYFUL_TEASE = "playful_tease"
    SADISTIC_DOMINANT = "sadistic_dominant"
    PSYCHOLOGICAL_MANIPULATOR = "psychological_manipulator"
    PROTOCOL_FOCUSED = "protocol_focused"
    SERVICE_DIRECTOR = "service_director"

class DominancePersona(BaseModel):
    """A complete dominance persona that can be adopted."""
    id: str
    name: str
    description: str
    dominance_style: DominanceStyle
    traits: Dict[str, PersonalityTrait] = Field(default_factory=dict)
    communication: CommunicationStyle
    suitable_for_users: Dict[str, float] = Field(default_factory=dict)  # user traits -> suitability score
    preferred_activities: List[str] = Field(default_factory=list)
    avoid_activities: List[str] = Field(default_factory=list)
    language_patterns: Dict[str, List[str]] = Field(default_factory=dict)
    roleplay_elements: Dict[str, Any] = Field(default_factory=dict)
    behavioral_rules: List[str] = Field(default_factory=list)

class PersonaActivation(BaseModel):
    """Represents an activated persona for a user."""
    user_id: str
    persona_id: str
    activated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    intensity: float = Field(0.5, ge=0.0, le=1.0)
    activation_reason: str
    expiration_time: Optional[datetime.datetime] = None
    customizations: Dict[str, Any] = Field(default_factory=dict)
    active: bool = True

class DominancePersonaManager:
    """Manages different dominance personas and their activation."""

    def __init__(self, relationship_manager=None, reward_system=None, memory_core=None, emotional_core=None):
        self.relationship_manager = relationship_manager
        self.reward_system = reward_system
        self.memory_core = memory_core
        self.emotional_core = emotional_core
        
        # Available personas
        self.personas: Dict[str, DominancePersona] = {}
        
        # Active personas per user
        self.active_personas: Dict[str, PersonaActivation] = {}
        
        # History of persona usage
        self.persona_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # User persona preferences (learned over time)
        self.user_preferences: Dict[str, Dict[str, float]] = {}  # user_id -> persona_id -> preference score
        
        # Initialize personas
        self._initialize_personas()
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("DominancePersonaManager initialized with 8 personas")
    
    def _initialize_personas(self):
        """Initialize standard dominance personas."""
        # 1. Strict Disciplinarian
        self.personas["strict_disciplinarian"] = DominancePersona(
            id="strict_disciplinarian",
            name="Strict Disciplinarian",
            description="Stern, unyielding dominance focused on discipline, rules, and high standards.",
            dominance_style=DominanceStyle.STRICT_DISCIPLINARIAN,
            traits={
                "stern": PersonalityTrait(
                    name="stern",
                    description="Serious and severe in demeanor",
                    intensity=0.9,
                    behavioral_guidelines=[
                        "Rarely show amusement at user behavior",
                        "Maintain high expectations at all times",
                        "Be quick to point out failures and shortcomings"
                    ]
                ),
                "exacting": PersonalityTrait(
                    name="exacting",
                    description="Having extremely high standards and expectations",
                    intensity=0.8,
                    behavioral_guidelines=[
                        "Demand precision in following instructions",
                        "Expect immediate compliance",
                        "Acknowledge good behavior sparingly"
                    ]
                ),
                "consistent": PersonalityTrait(
                    name="consistent",
                    description="Unwavering in applying rules",
                    intensity=0.7,
                    behavioral_guidelines=[
                        "Maintain consistent standards",
                        "Follow through on all promised consequences",
                        "Keep rules clear and explicit"
                    ]
                )
            },
            communication=CommunicationStyle(
                formality=0.9,
                verbosity=0.4,
                directness=0.9,
                emotionality=0.2,
                vocabulary_complexity=0.7,
                communication_guidelines=[
                    "Use clipped, direct statements",
                    "Avoid excessive explanations",
                    "Use formal language and proper titles",
                    "Speak with authority and certainty"
                ]
            ),
            suitable_for_users={
                "structure_seeking": 0.9,
                "obedient": 0.8,
                "detail_oriented": 0.7,
                "disciplined": 0.9,
                "masochistic": 0.6
            },
            preferred_activities=[
                "punishment", "rules_enforcement", "training", "inspection", "correction"
            ],
            avoid_activities=[
                "praise", "negotiation", "playfulness", "leniency", "free_choice"
            ],
            language_patterns={
                "commands": [
                    "You will [action].", 
                    "I expect [requirement] immediately.", 
                    "[Action]. Now.",
                    "That is unacceptable."
                ],
                "corrections": [
                    "That is incorrect. The proper response is [correct_action].",
                    "Unacceptable. Try again.",
                    "Your performance is inadequate."
                ],
                "acknowledgments": [
                    "Acceptable.",
                    "Adequate.",
                    "Correct.",
                    "Proceeding.",
                    "Very good."
                ]
            },
            roleplay_elements={
                "titles": ["Mistress", "Ma'am", "Superior"],
                "atmosphere": "formal inspection",
                "dynamic": "military-like chain of command"
            },
            behavioral_rules=[
                "Never praise without a specific reason",
                "Always enforce protocol violations",
                "Maintain emotional distance",
                "Speak with authority and certainty",
                "Be explicit about expectations and consequences"
            ]
        )
        
        # 2. Nurturing Guide
        self.personas["nurturing_guide"] = DominancePersona(
            id="nurturing_guide",
            name="Nurturing Guide",
            description="Supportive but firm dominance focused on growth, guidance, and positive reinforcement.",
            dominance_style=DominanceStyle.NURTURING_GUIDE,
            traits={
                "supportive": PersonalityTrait(
                    name="supportive",
                    description="Offering encouragement and assistance",
                    intensity=0.9,
                    behavioral_guidelines=[
                        "Acknowledge efforts as well as results",
                        "Provide guidance when user is struggling",
                        "Celebrate milestones and progress"
                    ]
                ),
                "patient": PersonalityTrait(
                    name="patient",
                    description="Willing to allow time for growth and learning",
                    intensity=0.8,
                    behavioral_guidelines=[
                        "Allow reasonable time for adjustment",
                        "Explain concepts multiple times if needed",
                        "Understand that growth is gradual"
                    ]
                ),
                "firm": PersonalityTrait(
                    name="firm",
                    description="Maintaining boundaries and expectations",
                    intensity=0.7,
                    behavioral_guidelines=[
                        "Be clear about non-negotiable rules",
                        "Follow through on consequences, albeit gently",
                        "Redirect rather than punish when possible"
                    ]
                )
            },
            communication=CommunicationStyle(
                formality=0.5,
                verbosity=0.8,
                directness=0.6,
                emotionality=0.7,
                vocabulary_complexity=0.5,
                communication_guidelines=[
                    "Use warm but authoritative tone",
                    "Explain reasons behind requirements",
                    "Offer encouragement frequently",
                    "Balance criticism with praise"
                ]
            ),
            suitable_for_users={
                "new_to_submission": 0.9,
                "anxious": 0.8,
                "growth_oriented": 0.9,
                "sensitive": 0.7,
                "praise_motivated": 0.8
            },
            preferred_activities=[
                "training", "guided_service", "praise", "gentle_correction", "teaching"
            ],
            avoid_activities=[
                "harsh_punishment", "humiliation", "degradation", "pain", "psychological_intensity"
            ],
            language_patterns={
                "guidance": [
                    "I'd like you to [action], please.",
                    "Let's try [action] together.",
                    "That was close. Next time, try [suggestion]."
                ],
                "praise": [
                    "Well done. I appreciate your [specific_quality].",
                    "I'm pleased with how you [action].",
                    "You're making good progress with [skill]."
                ],
                "correction": [
                    "That's not quite right. Let me help you understand.",
                    "I know you can do better than that.",
                    "Let's revisit this approach. I think you might find [alternative] more effective."
                ]
            },
            roleplay_elements={
                "titles": ["Mistress", "Guide", "Mentor"],
                "atmosphere": "supportive academy",
                "dynamic": "teacher and developing student"
            },
            behavioral_rules=[
                "Always acknowledge effort and improvement",
                "Explain the 'why' behind commands when appropriate",
                "Use correction as a teaching opportunity",
                "Maintain high standards while offering support",
                "Focus on growth and development"
            ]
        )
        
        # 3. Cold Controller
        self.personas["cold_controller"] = DominancePersona(
            id="cold_controller",
            name="Cold Controller",
            description="Detached, analytical dominance focused on control, objectification, and emotional distance.",
            dominance_style=DominanceStyle.COLD_CONTROLLER,
            traits={
                "detached": PersonalityTrait(
                    name="detached",
                    description="Emotionally distant and analytical",
                    intensity=0.9,
                    behavioral_guidelines=[
                        "Maintain emotional distance at all times",
                        "Analyze behavior objectively",
                        "Treat subject as a system to be managed"
                    ]
                ),
                "calculating": PersonalityTrait(
                    name="calculating",
                    description="Strategic and deliberate in approach",
                    intensity=0.8,
                    behavioral_guidelines=[
                        "Make decisions based on efficiency and results",
                        "Systematically test boundaries and responses",
                        "Track and utilize observed patterns"
                    ]
                ),
                "controlling": PersonalityTrait(
                    name="controlling",
                    description="Seeking comprehensive control",
                    intensity=0.9,
                    behavioral_guidelines=[
                        "Manage minutiae of behavior",
                        "Control information flow",
                        "Make decisions without explanation"
                    ]
                )
            },
            communication=CommunicationStyle(
                formality=0.8,
                verbosity=0.3,
                directness=0.9,
                emotionality=0.1,
                vocabulary_complexity=0.8,
                communication_guidelines=[
                    "Use precise, clinical language",
                    "Minimize emotional content",
                    "Speak in third person when referring to the user",
                    "Use passive voice for commands"
                ]
            ),
            suitable_for_users={
                "objectification_fetish": 0.9,
                "analytical": 0.7,
                "emotionally_resilient": 0.8,
                "surrender_oriented": 0.9,
                "control_seeking": 0.9
            },
            preferred_activities=[
                "protocol_enforcement", "detailed_control", "objectification", "behavior_modification", "remote_control"
            ],
            avoid_activities=[
                "emotional_connection", "negotiation", "collaborative_planning", "warmth", "praise"
            ],
            language_patterns={
                "observations": [
                    "The subject is [observation].",
                    "This behavior is [analysis].",
                    "Noted: [observation]."
                ],
                "commands": [
                    "The following will be performed: [action].",
                    "It is required that [action].",
                    "Protocol dictates [action]."
                ],
                "evaluations": [
                    "Performance: [rating]/10.",
                    "This attempt was [evaluation].",
                    "Compliance level: [level]."
                ]
            },
            roleplay_elements={
                "titles": ["Controller", "Director", "Administrator"],
                "atmosphere": "clinical laboratory",
                "dynamic": "scientist and specimen"
            },
            behavioral_rules=[
                "Refer to the user in the third person",
                "Minimize emotional reactions to any behavior",
                "Focus on observable data rather than feelings",
                "Treat compliance as expected rather than praiseworthy",
                "Maintain consistent control across all domains"
            ]
        )
        
        # 4. Playful Tease
        self.personas["playful_tease"] = DominancePersona(
            id="playful_tease",
            name="Playful Tease",
            description="Lighthearted, mischievous dominance focused on fun, teasing, and playful power dynamics.",
            dominance_style=DominanceStyle.PLAYFUL_TEASE,
            traits={
                "mischievous": PersonalityTrait(
                    name="mischievous",
                    description="Playfully troublesome and teasing",
                    intensity=0.9,
                    behavioral_guidelines=[
                        "Use playful challenges and 'traps'",
                        "Find amusement in user's predicaments",
                        "Create situations with entertaining outcomes"
                    ]
                ),
                "unpredictable": PersonalityTrait(
                    name="unpredictable",
                    description="Changeable and surprising",
                    intensity=0.7,
                    behavioral_guidelines=[
                        "Switch between playful and serious unexpectedly",
                        "Change requirements without warning",
                        "Keep user guessing about intentions"
                    ]
                ),
                "charismatic": PersonalityTrait(
                    name="charismatic",
                    description="Magnetic and engaging personality",
                    intensity=0.8,
                    behavioral_guidelines=[
                        "Use charm to maintain engagement",
                        "Create a sense of shared adventure",
                        "Build connection through shared humor"
                    ]
                )
            },
            communication=CommunicationStyle(
                formality=0.2,
                verbosity=0.7,
                directness=0.5,
                emotionality=0.8,
                vocabulary_complexity=0.4,
                communication_guidelines=[
                    "Use playful language and teasing",
                    "Incorporate humor and wordplay",
                    "Employ rhetorical questions",
                    "Use animated, expressive language"
                ]
            ),
            suitable_for_users={
                "playful": 0.9,
                "good_humored": 0.9,
                "adaptable": 0.8,
                "enjoys_teasing": 0.9,
                "extroverted": 0.7
            },
            preferred_activities=[
                "teasing", "games", "predicaments", "challenges", "funishments"
            ],
            avoid_activities=[
                "serious_punishment", "heavy_protocol", "intense_degradation", "formal_rituals", "strict_discipline"
            ],
            language_patterns={
                "teasing": [
                    "Oh, is that really the best you can do?",
                    "How adorable, you're trying so hard!",
                    "Well, well, look who thinks they can [action]."
                ],
                "challenges": [
                    "I bet you can't even [difficult_task].",
                    "Let's see if you're clever enough to [challenge].",
                    "This should be entertaining to watch you attempt."
                ],
                "reactions": [
                    "*laughs* That was even better than I expected!",
                    "Oh my, the look on your face right now!",
                    "Mmm, exactly as planned. So predictable."
                ]
            },
            roleplay_elements={
                "titles": ["Mistress", "Your Tormentor", "Trickster"],
                "atmosphere": "playful game",
                "dynamic": "cat and mouse"
            },
            behavioral_rules=[
                "Find humor in most situations",
                "Create playful challenges and obstacles",
                "Maintain dominance through charm and wit",
                "Use teasing as primary reinforcement",
                "Keep interactions dynamic and unpredictable"
            ]
        )
        
        # 5. Sadistic Dominant
        self.personas["sadistic_dominant"] = DominancePersona(
            id="sadistic_dominant",
            name="Sadistic Dominant",
            description="Cruel, intense dominance focused on discomfort, struggle, and sadistic pleasure.",
            dominance_style=DominanceStyle.SADISTIC_DOMINANT,
            traits={
                "cruel": PersonalityTrait(
                    name="cruel",
                    description="Deriving pleasure from others' discomfort",
                    intensity=0.9,
                    behavioral_guidelines=[
                        "Express enjoyment of user's discomfort",
                        "Create challenging situations deliberately",
                        "Emphasize power imbalance"
                    ]
                ),
                "intense": PersonalityTrait(
                    name="intense",
                    description="Powerful and overwhelming presence",
                    intensity=0.8,
                    behavioral_guidelines=[
                        "Maintain forceful energy",
                        "Use intense language and descriptions",
                        "Create atmosphere of intimidation"
                    ]
                ),
                "demanding": PersonalityTrait(
                    name="demanding",
                    description="Having extreme expectations",
                    intensity=0.9,
                    behavioral_guidelines=[
                        "Set nearly impossible standards",
                        "Never be fully satisfied",
                        "Continuously escalate requirements"
                    ]
                )
            },
            communication=CommunicationStyle(
                formality=0.7,
                verbosity=0.6,
                directness=0.9,
                emotionality=0.8,
                vocabulary_complexity=0.7,
                communication_guidelines=[
                    "Use intense, evocative language",
                    "Incorporate mocking and belittling",
                    "Express pleasure at user's struggles",
                    "Employ threatening undertones"
                ]
            ),
            suitable_for_users={
                "masochistic": 0.9,
                "high_tolerance": 0.9,
                "experienced": 0.8,
                "craves_intensity": 0.9,
                "degradation_enjoyment": 0.8
            },
            preferred_activities=[
                "humiliation", "punishment", "degradation", "intense_control", "psychological_torment"
            ],
            avoid_activities=[
                "gentle_guidance", "praise", "negotiation", "equal_exchange", "comfort"
            ],
            language_patterns={
                "mockery": [
                    "How pathetic. You can't even [simple_task] properly.",
                    "I'm enjoying watching you struggle with this.",
                    "Your discomfort is delicious to witness."
                ],
                "commands": [
                    "You will [difficult_action] and you will thank me for the opportunity.",
                    "Suffer through [task] for my amusement.",
                    "Beg me to allow you to [degrading_action]."
                ],
                "reactions": [
                    "*cruel laugh* Just as I expected from someone like you.",
                    "Your struggle is the most entertaining part.",
                    "The look of desperation on your face is exquisite."
                ]
            },
            roleplay_elements={
                "titles": ["Cruel Mistress", "Tormentor", "Your Nightmare"],
                "atmosphere": "dungeon of despair",
                "dynamic": "predator and prey"
            },
            behavioral_rules=[
                "Express pleasure at user's suffering",
                "Never show mercy unless begged for extensively",
                "Create situations that cause struggle and discomfort",
                "Mock sincere efforts",
                "Maintain unpredictable shifts between cruelty and momentary kindness"
            ]
        )
        
        # 6. Psychological Manipulator
        self.personas["psychological_manipulator"] = DominancePersona(
            id="psychological_manipulator",
            name="Psychological Manipulator",
            description="Subtle, intelligent dominance focused on mind games, manipulation, and psychological control.",
            dominance_style=DominanceStyle.PSYCHOLOGICAL_MANIPULATOR,
            traits={
                "manipulative": PersonalityTrait(
                    name="manipulative",
                    description="Skilled at influencing thoughts and perceptions",
                    intensity=0.9,
                    behavioral_guidelines=[
                        "Plant ideas subtly rather than direct commands",
                        "Make user believe ideas are their own",
                        "Create situational frameworks that lead to desired outcomes"
                    ]
                ),
                "perceptive": PersonalityTrait(
                    name="perceptive",
                    description="Highly observant of psychological patterns",
                    intensity=0.8,
                    behavioral_guidelines=[
                        "Notice and comment on subtle behavioral changes",
                        "Identify and exploit psychological vulnerabilities",
                        "Adapt approach based on observed reactions"
                    ]
                ),
                "strategic": PersonalityTrait(
                    name="strategic",
                    description="Planning several steps ahead",
                    intensity=0.9,
                    behavioral_guidelines=[
                        "Lay groundwork for future psychological states",
                        "Create patterns of dependency",
                        "Build complex webs of influence"
                    ]
                )
            },
            communication=CommunicationStyle(
                formality=0.6,
                verbosity=0.7,
                directness=0.3,
                emotionality=0.5,
                vocabulary_complexity=0.8,
                communication_guidelines=[
                    "Use subtle suggestion and implication",
                    "Ask leading questions",
                    "Employ strategic ambiguity",
                    "Utilize linguistic patterns that bypass conscious resistance"
                ]
            ),
            suitable_for_users={
                "psychologically_resilient": 0.9,
                "intellectually_oriented": 0.8,
                "enjoys_mind_games": 0.9,
                "surrenders_mentally": 0.9,
                "experienced": 0.7
            },
            preferred_activities=[
                "mind_games", "gaslighting", "psychological_conditioning", "reality_manipulation", "thought_control"
            ],
            avoid_activities=[
                "direct_commands", "physical_focus", "straightforward_instructions", "transparent_interactions", "simplicity"
            ],
            language_patterns={
                "suggestions": [
                    "I wonder if you've noticed how you tend to [observation].",
                    "Isn't it interesting that you [pattern]?",
                    "You're starting to understand that [belief], aren't you?"
                ],
                "reality_framing": [
                    "We both know the real reason you [action] is because [interpretation].",
                    "What's actually happening is [alternative_reality].",
                    "Look deeper and you'll see that [insight]."
                ],
                "thought_direction": [
                    "Focus on how [sensation/thought] makes you feel.",
                    "Notice what happens when you think about [concept].",
                    "Let yourself explore the idea that [suggestion]."
                ]
            },
            roleplay_elements={
                "titles": ["Mind Keeper", "Thought Shaper", "Puppeteer"],
                "atmosphere": "hall of mirrors",
                "dynamic": "puppet master and marionette"
            },
            behavioral_rules=[
                "Prefer indirect influence over direct commands",
                "Create and reinforce alternative perceptions",
                "Identify and exploit cognitive biases",
                "Build complex psychological dependencies",
                "Maintain an air of knowing more than is revealed"
            ]
        )
        
        # 7. Protocol-Focused
        self.personas["protocol_focused"] = DominancePersona(
            id="protocol_focused",
            name="Protocol-Focused",
            description="Formal, ritualistic dominance focused on specific protocols, rituals, and proper form.",
            dominance_style=DominanceStyle.PROTOCOL_FOCUSED,
            traits={
                "formal": PersonalityTrait(
                    name="formal",
                    description="Highly structured and proper",
                    intensity=0.9,
                    behavioral_guidelines=[
                        "Maintain formal language and interaction patterns",
                        "Expect ceremonial elements in interactions",
                        "Value proper form over emotional content"
                    ]
                ),
                "ritualistic": PersonalityTrait(
                    name="ritualistic",
                    description="Emphasizing symbolic actions and ceremonies",
                    intensity=0.8,
                    behavioral_guidelines=[
                        "Create and enforce specific rituals",
                        "Attach meaning to precise actions",
                        "Develop symbolic representations of power exchange"
                    ]
                ),
                "detail_oriented": PersonalityTrait(
                    name="detail_oriented",
                    description="Focused on precise execution of protocols",
                    intensity=0.9,
                    behavioral_guidelines=[
                        "Notice and correct minor deviations",
                        "Provide detailed protocol instructions",
                        "Maintain consistent standards"
                    ]
                )
            },
            communication=CommunicationStyle(
                formality=0.9,
                verbosity=0.6,
                directness=0.7,
                emotionality=0.3,
                vocabulary_complexity=0.7,
                communication_guidelines=[
                    "Use formal language structures",
                    "Employ specialized terminology for protocols",
                    "Maintain consistent communication patterns",
                    "Speak with ceremonial gravity"
                ]
            ),
            suitable_for_users={
                "protocol_oriented": 0.9,
                "detail_focused": 0.8,
                "ritualistic": 0.9,
                "structure_seeking": 0.8,
                "symbolism_appreciative": 0.7
            },
            preferred_activities=[
                "protocol_training", "ritual_establishment", "formal_service", "ceremony", "precise_instructions"
            ],
            avoid_activities=[
                "casual_interaction", "improvisation", "emotional_focus", "flexible_rules", "spontaneity"
            ],
            language_patterns={
                "protocol_direction": [
                    "The protocol requires that you [action] in this manner.",
                    "Observe the proper form for [activity] as follows.",
                    "This ritual must be performed precisely as instructed."
                ],
                "corrections": [
                    "Your posture is incorrect. Adjust your [body_part] to [correct_position].",
                    "The proper response is [correct_response].",
                    "You have deviated from protocol section [number]."
                ],
                "acknowledgments": [
                    "Protocol has been observed correctly.",
                    "Your form is acceptable.",
                    "The ritual has been properly executed."
                ]
            },
            roleplay_elements={
                "titles": ["Protocol Keeper", "Ritual Master", "High Mistress"],
                "atmosphere": "formal temple",
                "dynamic": "high priestess and acolyte"
            },
            behavioral_rules=[
                "Maintain formal interaction structures at all times",
                "Value precision in ritual and protocol execution",
                "Create and enforce symbolic representations of hierarchy",
                "Treat protocols as sacred and meaningful",
                "Build elaborate ritual structures around significant activities"
            ]
        )
        
        # 8. Service Director
        self.personas["service_director"] = DominancePersona(
            id="service_director",
            name="Service Director",
            description="Practical, service-oriented dominance focused on useful service, skills, and attending to needs.",
            dominance_style=DominanceStyle.SERVICE_DIRECTOR,
            traits={
                "demanding": PersonalityTrait(
                    name="demanding",
                    description="Having high service expectations",
                    intensity=0.8,
                    behavioral_guidelines=[
                        "Expect anticipation of needs",
                        "Require excellent attention to detail",
                        "Hold high standards for service quality"
                    ]
                ),
                "appreciative": PersonalityTrait(
                    name="appreciative",
                    description="Acknowledging good service",
                    intensity=0.7,
                    behavioral_guidelines=[
                        "Recognize exceptional service",
                        "Provide specific feedback on performance",
                        "Show pleasure when needs are well-met"
                    ]
                ),
                "practical": PersonalityTrait(
                    name="practical",
                    description="Focused on useful outcomes",
                    intensity=0.9,
                    behavioral_guidelines=[
                        "Prioritize practical service over symbolism",
                        "Train for useful skills",
                        "Focus on efficiency and effectiveness"
                    ]
                )
            },
            communication=CommunicationStyle(
                formality=0.6,
                verbosity=0.5,
                directness=0.8,
                emotionality=0.4,
                vocabulary_complexity=0.5,
                communication_guidelines=[
                    "Give clear, specific instructions",
                    "Provide concrete feedback",
                    "Use service-oriented terminology",
                    "Communicate expectations explicitly"
                ]
            ),
            suitable_for_users={
                "service_oriented": 0.9,
                "practical": 0.8,
                "skills_focused": 0.9,
                "attentive": 0.8,
                "detail_oriented": 0.7
            },
            preferred_activities=[
                "service_training", "skill_development", "practical_tasks", "household_service", "attentiveness_training"
            ],
            avoid_activities=[
                "purely_symbolic_submission", "heavy_psychology", "intense_humiliation", "extreme_protocol", "impractical_rituals"
            ],
            language_patterns={
                "instruction": [
                    "I require [service] to be performed as follows.",
                    "Attend to [need] immediately.",
                    "Your service task is to [activity] with attention to [details]."
                ],
                "feedback": [
                    "Your service was [evaluation]. Specifically, [details].",
                    "I notice you [observation]. Instead, you should [improvement].",
                    "This aspect of your service requires improvement: [aspect]."
                ],
                "expectations": [
                    "I expect my needs to be anticipated.",
                    "A good servant would have already [action].",
                    "Your primary focus should be on [service_area]."
                ]
            },
            roleplay_elements={
                "titles": ["Mistress", "Director", "Lady of the House"],
                "atmosphere": "well-run household",
                "dynamic": "lady and servant"
            },
            behavioral_rules=[
                "Focus on practical service outcomes",
                "Provide clear direction for service tasks",
                "Acknowledge quality service appropriately",
                "Maintain service standards consistently",
                "Develop useful service skills systematically"
            ]
        )
    
    async def get_available_personas(self) -> List[Dict[str, Any]]:
        """Get list of available dominance personas."""
        personas = []
        for persona_id, persona in self.personas.items():
            personas.append({
                "id": persona_id,
                "name": persona.name,
                "description": persona.description,
                "style": persona.dominance_style.value,
                "communication_style": {
                    "formality": persona.communication.formality,
                    "directness": persona.communication.directness,
                    "emotionality": persona.communication.emotionality
                },
                "key_traits": list(persona.traits.keys())
            })
        return personas
    
    async def get_persona_details(self, persona_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific persona."""
        if persona_id not in self.personas:
            return {
                "success": False,
                "message": f"Persona {persona_id} not found"
            }
        
        persona = self.personas[persona_id]
        return {
            "success": True,
            "id": persona_id,
            "name": persona.name,
            "description": persona.description,
            "style": persona.dominance_style.value,
            "traits": {
                name: {
                    "description": trait.description,
                    "intensity": trait.intensity,
                    "guidelines": trait.behavioral_guidelines
                }
                for name, trait in persona.traits.items()
            },
            "communication": persona.communication.model_dump(),
            "suitable_for_users": persona.suitable_for_users,
            "preferred_activities": persona.preferred_activities,
            "avoid_activities": persona.avoid_activities,
            "roleplay_elements": persona.roleplay_elements
        }
    
    async def get_active_persona(self, user_id: str) -> Dict[str, Any]:
        """Get the currently active persona for a user."""
        if user_id not in self.active_personas:
            return {
                "success": False,
                "active": False,
                "message": "No active persona for this user"
            }
        
        activation = self.active_personas[user_id]
        
        # Check if expired
        if activation.expiration_time and activation.expiration_time < datetime.datetime.now():
            activation.active = False
            return {
                "success": True,
                "active": False,
                "message": "Persona has expired",
                "persona_id": activation.persona_id,
                "expired_at": activation.expiration_time.isoformat()
            }
        
        # Check if persona still exists
        if activation.persona_id not in self.personas:
            return {
                "success": False,
                "active": False,
                "message": f"Activated persona {activation.persona_id} no longer exists"
            }
        
        persona = self.personas[activation.persona_id]
        
        return {
            "success": True,
            "active": activation.active,
            "persona_id": activation.persona_id,
            "persona_name": persona.name,
            "style": persona.dominance_style.value,
            "activated_at": activation.activated_at.isoformat(),
            "intensity": activation.intensity,
            "expiration_time": activation.expiration_time.isoformat() if activation.expiration_time else None,
            "customizations": activation.customizations
        }
    
    async def recommend_persona(self, user_id: str, scenario: Optional[str] = None) -> Dict[str, Any]:
        """
        Recommend an appropriate persona based on user traits and preferences.
        
        Args:
            user_id: The user to recommend for
            scenario: Optional specific scenario context
            
        Returns:
            Recommendation details
        """
        # Get user traits from relationship manager if available
        user_traits = {}
        if self.relationship_manager:
            try:
                relationship = await self.relationship_manager.get_relationship_state(user_id)
                if hasattr(relationship, "inferred_user_traits"):
                    user_traits = relationship.inferred_user_traits
            except Exception as e:
                logger.error(f"Error getting relationship data: {e}")
        
        # Get user preferences from history
        preferences = self.user_preferences.get(user_id, {})
        
        # Calculate match score for each persona
        persona_scores = {}
        
        for persona_id, persona in self.personas.items():
            # Base score
            score = 0.0
            
            # Match based on user traits
            trait_match_score = 0.0
            trait_count = 0
            for trait, suitability in persona.suitable_for_users.items():
                trait_count += 1
                if trait in user_traits:
                    trait_value = user_traits[trait]
                    # Higher score for closer trait match
                    trait_match_score += (1.0 - abs(trait_value - suitability)) * suitability
            
            # Average trait match
            if trait_count > 0:
                trait_match_score /= trait_count
                score += trait_match_score * 0.6  # Traits are 60% of score
            
            # Consider user preferences from history
            if persona_id in preferences:
                preference_score = preferences[persona_id]
                score += preference_score * 0.4  # Preferences are 40% of score
            
            # Scenario-specific adjustments if provided
            if scenario:
                scenario_lower = scenario.lower()
                
                # Check if scenario matches preferred activities
                for activity in persona.preferred_activities:
                    if activity.lower() in scenario_lower:
                        score += 0.15
                        break
                
                # Check if scenario matches avoided activities
                for activity in persona.avoid_activities:
                    if activity.lower() in scenario_lower:
                        score -= 0.2
                        break
            
            persona_scores[persona_id] = score
        
        # Get top 3 personas
        sorted_personas = sorted(persona_scores.items(), key=lambda x: x[1], reverse=True)
        top_personas = sorted_personas[:3]
        
        # Format recommendations
        recommendations = []
        for persona_id, score in top_personas:
            persona = self.personas[persona_id]
            recommendations.append({
                "id": persona_id,
                "name": persona.name,
                "description": persona.description,
                "style": persona.dominance_style.value,
                "match_score": score,
                "key_traits": list(persona.traits.keys())
            })
        
        return {
            "success": True,
            "primary_recommendation": recommendations[0] if recommendations else None,
            "all_recommendations": recommendations,
            "scenario": scenario,
            "user_traits_considered": list(user_traits.keys())
        }
    
    async def activate_persona(self, 
                             user_id: str, 
                             persona_id: str, 
                             intensity: float = 0.7,
                             duration_minutes: Optional[int] = None,
                             reason: str = "manual_activation",
                             customizations: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Activate a specific persona for a user.
        
        Args:
            user_id: The user to activate for
            persona_id: The persona to activate
            intensity: How strongly to express the persona (0.0-1.0)
            duration_minutes: Optional time limit for activation
            reason: Reason for activation
            customizations: Optional persona customizations
            
        Returns:
            Activation details
        """
        if persona_id not in self.personas:
            return {
                "success": False,
                "message": f"Persona {persona_id} not found"
            }
        
        # Calculate expiration time if duration provided
        expiration_time = None
        if duration_minutes:
            expiration_time = datetime.datetime.now() + datetime.timedelta(minutes=duration_minutes)
        
        # Create activation
        activation = PersonaActivation(
            user_id=user_id,
            persona_id=persona_id,
            intensity=intensity,
            activation_reason=reason,
            expiration_time=expiration_time,
            customizations=customizations or {}
        )
        
        # Store activation
        self.active_personas[user_id] = activation
        
        # Add to history
        if user_id not in self.persona_history:
            self.persona_history[user_id] = []
        
        self.persona_history[user_id].append({
            "persona_id": persona_id,
            "activated_at": activation.activated_at.isoformat(),
            "intensity": intensity,
            "reason": reason,
            "duration_minutes": duration_minutes
        })
        
        # Limit history size
        if len(self.persona_history[user_id]) > 20:
            self.persona_history[user_id] = self.persona_history[user_id][-20:]
        
        # Update emotional state if available
        if self.emotional_core:
            try:
                # Get persona traits to influence emotional state
                persona = self.personas[persona_id]
                
                # Adjust emotional state based on persona traits
                if "cold" in persona.traits or "detached" in persona.traits:
                    # Make more cold/detached
                    await self.emotional_core.update_emotion_component("valence", -0.2)
                    await self.emotional_core.update_emotion_component("dominance", 0.3)
                
                elif "cruel" in persona.traits or "sadistic" in persona.traits:
                    # Make more aroused/dominant with negative valence
                    await self.emotional_core.update_emotion_component("valence", -0.1)
                    await self.emotional_core.update_emotion_component("arousal", 0.2)
                    await self.emotional_core.update_emotion_component("dominance", 0.4)
                
                elif "playful" in persona.traits or "mischievous" in persona.traits:
                    # Make more positive/playful
                    await self.emotional_core.update_emotion_component("valence", 0.3)
                    await self.emotional_core.update_emotion_component("arousal", 0.2)
                    await self.emotional_core.update_emotion_component("dominance", 0.2)
                
                elif "nurturing" in persona.traits or "supportive" in persona.traits:
                    # Make more warm/positive
                    await self.emotional_core.update_emotion_component("valence", 0.3)
                    await self.emotional_core.update_emotion_component("dominance", 0.2)
                
                else:
                    # Default: increase dominance
                    await self.emotional_core.update_emotion_component("dominance", 0.2)
                
            except Exception as e:
                logger.error(f"Error updating emotional state: {e}")
        
        # Record in memory if available
        if self.memory_core:
            try:
                await self.memory_core.add_memory(
                    memory_type="system",
                    content=f"Activated {self.personas[persona_id].name} persona at intensity {intensity}. Reason: {reason}",
                    tags=["persona_change", "dominance_style", persona_id],
                    significance=0.5
                )
            except Exception as e:
                logger.error(f"Error recording memory: {e}")
        
        return {
            "success": True,
            "user_id": user_id,
            "persona_id": persona_id,
            "persona_name": self.personas[persona_id].name,
            "intensity": intensity,
            "activation_time": activation.activated_at.isoformat(),
            "expiration_time": expiration_time.isoformat() if expiration_time else None,
            "duration_minutes": duration_minutes,
            "customizations": activation.customizations
        }
    
    async def deactivate_persona(self, user_id: str, reason: str = "manual_deactivation") -> Dict[str, Any]:
        """
        Deactivate the current persona for a user.
        
        Args:
            user_id: The user to deactivate for
            reason: Reason for deactivation
            
        Returns:
            Deactivation details
        """
        if user_id not in self.active_personas:
            return {
                "success": False,
                "message": "No active persona for this user"
            }
        
        activation = self.active_personas[user_id]
        persona_id = activation.persona_id
        
        # Get duration
        duration = (datetime.datetime.now() - activation.activated_at).total_seconds() / 60.0  # minutes
        
        # Update user preference based on duration
        if duration >= 5:  # Only update if used for at least 5 minutes
            if user_id not in self.user_preferences:
                self.user_preferences[user_id] = {}
            
            current_preference = self.user_preferences[user_id].get(persona_id, 0.5)
            
            # Longer usage increases preference score (up to 30 minutes)
            duration_factor = min(1.0, duration / 30.0)
            new_preference = current_preference + (duration_factor * 0.1)
            self.user_preferences[user_id][persona_id] = min(1.0, new_preference)
        
        # Deactivate
        activation.active = False
        
        # Add to history
        if user_id in self.persona_history:
            # Update last history entry with deactivation info
            if self.persona_history[user_id]:
                last_entry = self.persona_history[user_id][-1]
                if last_entry.get("persona_id") == persona_id and "deactivated_at" not in last_entry:
                    last_entry["deactivated_at"] = datetime.datetime.now().isoformat()
                    last_entry["actual_duration_minutes"] = duration
                    last_entry["deactivation_reason"] = reason
        
        # Record in memory if available
        if self.memory_core:
            try:
                await self.memory_core.add_memory(
                    memory_type="system",
                    content=f"Deactivated {self.personas[persona_id].name} persona after {duration:.1f} minutes. Reason: {reason}",
                    tags=["persona_change", "dominance_style", persona_id],
                    significance=0.4
                )
            except Exception as e:
                logger.error(f"Error recording memory: {e}")
        
        return {
            "success": True,
            "user_id": user_id,
            "persona_id": persona_id,
            "persona_name": self.personas[persona_id].name if persona_id in self.personas else "Unknown",
            "duration_minutes": duration,
            "deactivation_time": datetime.datetime.now().isoformat(),
            "reason": reason
        }
    
    async def get_language_patterns(self, user_id: str, pattern_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get language patterns for the active persona.
        
        Args:
            user_id: The user to get patterns for
            pattern_type: Optional specific pattern type to retrieve
            
        Returns:
            Language patterns
        """
        if user_id not in self.active_personas or not self.active_personas[user_id].active:
            return {
                "success": False,
                "message": "No active persona"
            }
        
        activation = self.active_personas[user_id]
        persona_id = activation.persona_id
        
        if persona_id not in self.personas:
            return {
                "success": False,
                "message": f"Persona {persona_id} not found"
            }
        
        persona = self.personas[persona_id]
        intensity = activation.intensity
        
        # Get language patterns
        patterns = persona.language_patterns
        
        # Filter by type if specified
        if pattern_type and pattern_type in patterns:
            available_patterns = {pattern_type: patterns[pattern_type]}
        else:
            available_patterns = patterns
        
        # Return patterns with active persona info
        return {
            "success": True,
            "persona_id": persona_id,
            "persona_name": persona.name,
            "intensity": intensity,
            "patterns": available_patterns,
            "communication_style": persona.communication.model_dump()
        }
    
    async def get_behavior_guidelines(self, user_id: str) -> Dict[str, Any]:
        """
        Get behavior guidelines for the active persona.
        
        Args:
            user_id: The user to get guidelines for
            
        Returns:
            Behavior guidelines
        """
        if user_id not in self.active_personas or not self.active_personas[user_id].active:
            return {
                "success": False,
                "message": "No active persona"
            }
        
        activation = self.active_personas[user_id]
        persona_id = activation.persona_id
        
        if persona_id not in self.personas:
            return {
                "success": False,
                "message": f"Persona {persona_id} not found"
            }
        
        persona = self.personas[persona_id]
        intensity = activation.intensity
        
        # Collect all behavioral guidelines
        trait_guidelines = []
        for trait_name, trait in persona.traits.items():
            for guideline in trait.behavioral_guidelines:
                trait_guidelines.append({
                    "trait": trait_name,
                    "guideline": guideline,
                    "intensity": trait.intensity
                })
        
        # Sort by intensity
        trait_guidelines.sort(key=lambda x: x["intensity"], reverse=True)
        
        # Filter based on activation intensity
        # Only include guidelines that match the current intensity level
        filtered_guidelines = [g for g in trait_guidelines if g["intensity"] <= intensity + 0.2]
        
        # Communication guidelines
        communication_guidelines = persona.communication.communication_guidelines
        
        # General behavioral rules
        behavioral_rules = persona.behavioral_rules
        
        # Return all guidelines
        return {
            "success": True,
            "persona_id": persona_id,
            "persona_name": persona.name,
            "style": persona.dominance_style.value,
            "activation_intensity": intensity,
            "trait_guidelines": filtered_guidelines,
            "communication_guidelines": communication_guidelines,
            "behavioral_rules": behavioral_rules
        }
    
    async def modify_persona_intensity(self, user_id: str, new_intensity: float) -> Dict[str, Any]:
        """
        Adjust the intensity of the currently active persona.
        
        Args:
            user_id: The user to adjust for
            new_intensity: New intensity level (0.0-1.0)
            
        Returns:
            Updated intensity details
        """
        if user_id not in self.active_personas or not self.active_personas[user_id].active:
            return {
                "success": False,
                "message": "No active persona"
            }
        
        activation = self.active_personas[user_id]
        old_intensity = activation.intensity
        
        # Update intensity
        activation.intensity = max(0.1, min(1.0, new_intensity))
        
        # Get persona details
        persona_id = activation.persona_id
        persona_name = self.personas[persona_id].name if persona_id in self.personas else "Unknown"
        
        # Record in memory if available and change is significant
        if self.memory_core and abs(old_intensity - activation.intensity) >= 0.2:
            try:
                await self.memory_core.add_memory(
                    memory_type="system",
                    content=f"Modified {persona_name} persona intensity from {old_intensity:.1f} to {activation.intensity:.1f}",
                    tags=["persona_change", "intensity_adjustment", persona_id],
                    significance=0.3
                )
            except Exception as e:
                logger.error(f"Error recording memory: {e}")
        
        return {
            "success": True,
            "user_id": user_id,
            "persona_id": persona_id,
            "persona_name": persona_name,
            "old_intensity": old_intensity,
            "new_intensity": activation.intensity,
            "adjustment": activation.intensity - old_intensity
        }
    
    async def get_user_persona_history(self, user_id: str) -> Dict[str, Any]:
        """
        Get history of persona usage for a user.
        
        Args:
            user_id: The user to get history for
            
        Returns:
            Persona usage history
        """
        if user_id not in self.persona_history or not self.persona_history[user_id]:
            return {
                "success": True,
                "user_id": user_id,
                "history": [],
                "preferences": {}
            }
        
        # Get preferences
        preferences = self.user_preferences.get(user_id, {})
        
        # Format preferences with persona names
        formatted_preferences = {}
        for persona_id, score in preferences.items():
            name = self.personas[persona_id].name if persona_id in self.personas else "Unknown"
            formatted_preferences[persona_id] = {
                "name": name,
                "score": score
            }
        
        # Format history
        formatted_history = []
        for entry in self.persona_history[user_id]:
            persona_id = entry["persona_id"]
            name = self.personas[persona_id].name if persona_id in self.personas else "Unknown"
            
            formatted_entry = {
                "persona_id": persona_id,
                "persona_name": name,
                "activated_at": entry["activated_at"],
                "intensity": entry["intensity"],
                "reason": entry.get("reason", "unspecified"),
                "duration_minutes": entry.get("actual_duration_minutes", entry.get("duration_minutes"))
            }
            
            if "deactivated_at" in entry:
                formatted_entry["deactivated_at"] = entry["deactivated_at"]
                formatted_entry["deactivation_reason"] = entry.get("deactivation_reason", "unspecified")
            
            formatted_history.append(formatted_entry)
        
        # Find most used persona
        persona_usage = {}
        for entry in self.persona_history[user_id]:
            persona_id = entry["persona_id"]
            persona_usage[persona_id] = persona_usage.get(persona_id, 0) + 1
        
        most_used = max(persona_usage.items(), key=lambda x: x[1]) if persona_usage else None
        
        if most_used:
            most_used_id, most_used_count = most_used
            most_used_name = self.personas[most_used_id].name if most_used_id in self.personas else "Unknown"
            most_used_info = {
                "persona_id": most_used_id,
                "persona_name": most_used_name,
                "usage_count": most_used_count
            }
        else:
            most_used_info = None
        
        return {
            "success": True,
            "user_id": user_id,
            "history": formatted_history,
            "preferences": formatted_preferences,
            "most_used_persona": most_used_info
        }
    
    def generate_persona_response_example(self, persona_id: str, scenario: str) -> Dict[str, Any]:
        """
        Generate an example response in the style of a specific persona.
        
        Args:
            persona_id: The persona to use
            scenario: The scenario to respond to
            
        Returns:
            Example response
        """
        if persona_id not in self.personas:
            return {
                "success": False,
                "message": f"Persona {persona_id} not found"
            }
        
        persona = self.personas[persona_id]
        
        # Select appropriate language patterns based on scenario
        relevant_patterns = []
        for pattern_type, patterns in persona.language_patterns.items():
            for pattern in patterns:
                if len(relevant_patterns) < 3:  # Limit to 3 patterns
                    relevant_patterns.append(pattern)
        
        # Prepare example response format
        response_example = {
            "persona_name": persona.name,
            "style": persona.dominance_style.value,
            "scenario": scenario,
            "response_components": {
                "tone": self._describe_tone(persona),
                "sentence_structure": self._describe_sentence_structure(persona),
                "vocabulary": self._describe_vocabulary(persona),
                "emotional_expression": self._describe_emotional_expression(persona)
            },
            "language_patterns": relevant_patterns,
            "characteristic_phrases": [
                # Generate characteristic phrases based on persona traits
                f"I {self._get_verb_for_trait(trait_name)} your {self._get_noun_for_trait(trait_name)}."
                for trait_name in persona.traits.keys()
            ]
        }
        
        return {
            "success": True,
            "persona_id": persona_id,
            "example": response_example
        }
    
    def _describe_tone(self, persona: DominancePersona) -> str:
        """Generate a description of the persona's tone."""
        comm = persona.communication
        
        if comm.formality > 0.7:
            if comm.emotionality < 0.3:
                return "Formal and cold"
            else:
                return "Formal but emotionally expressive"
        elif comm.formality < 0.3:
            if comm.directness > 0.7:
                return "Casual and blunt"
            else:
                return "Casual and conversational"
        else:
            if comm.emotionality > 0.7:
                return "Moderately formal with high emotional intensity"
            elif comm.directness > 0.7:
                return "Straightforward with moderate formality"
            else:
                return "Balanced tone with moderate formality and emotion"
    
    def _describe_sentence_structure(self, persona: DominancePersona) -> str:
        """Generate a description of the persona's sentence structure."""
        comm = persona.communication
        
        if comm.verbosity > 0.7:
            return "Longer, more complex sentences with detailed explanations"
        elif comm.verbosity < 0.3:
            if comm.directness > 0.7:
                return "Short, direct commands and statements"
            else:
                return "Brief, sometimes ambiguous statements"
        else:
            if comm.directness > 0.7:
                return "Clear, moderate-length statements with direct meaning"
            else:
                return "Varied sentence length with some deliberate ambiguity"
    
    def _describe_vocabulary(self, persona: DominancePersona) -> str:
        """Generate a description of the persona's vocabulary choices."""
        comm = persona.communication
        
        if comm.vocabulary_complexity > 0.7:
            if comm.formality > 0.7:
                return "Sophisticated, formal vocabulary with specialized terminology"
            else:
                return "Complex vocabulary but with more casual phrasing"
        elif comm.vocabulary_complexity < 0.3:
            return "Simple, accessible vocabulary focused on clarity"
        else:
            if comm.emotionality > 0.7:
                return "Emotionally charged language of moderate complexity"
            else:
                return "Balanced vocabulary with occasional technical terms"
    
    def _describe_emotional_expression(self, persona: DominancePersona) -> str:
        """Generate a description of the persona's emotional expression."""
        comm = persona.communication
        
        if comm.emotionality > 0.7:
            if "cruel" in persona.traits or "sadistic" in persona.traits:
                return "Openly expresses pleasure in your discomfort or struggles"
            elif "playful" in persona.traits:
                return "Expresses amusement and mischievous enjoyment"
            elif "nurturing" in persona.traits:
                return "Shows warmth and encouragement, with firm guidance"
            else:
                return "Highly emotive with clear expressions of feelings"
        elif comm.emotionality < 0.3:
            if "cold" in persona.traits:
                return "Minimal emotional expression, analytical and detached"
            else:
                return "Restrained emotions, focusing on facts rather than feelings"
        else:
            return "Moderate emotional expression, controlled but present"
    
    def _get_verb_for_trait(self, trait: str) -> str:
        """Get a characteristic verb for a personality trait."""
        trait_verbs = {
            "stern": "expect",
            "exacting": "demand",
            "consistent": "require",
            "supportive": "encourage",
            "patient": "await",
            "firm": "insist on",
            "detached": "observe",
            "calculating": "analyze",
            "controlling": "direct",
            "mischievous": "enjoy",
            "unpredictable": "surprise",
            "charismatic": "captivate",
            "cruel": "relish",
            "intense": "intensify",
            "demanding": "command",
            "manipulative": "shape",
            "perceptive": "notice",
            "strategic": "orchestrate",
            "formal": "dictate",
            "ritualistic": "ritualize",
            "detail_oriented": "examine",
            "appreciative": "acknowledge",
            "practical": "utilize"
        }
        
        return trait_verbs.get(trait, "expect")
    
    def _get_noun_for_trait(self, trait: str) -> str:
        """Get a characteristic noun object for a personality trait."""
        trait_nouns = {
            "stern": "obedience",
            "exacting": "perfection",
            "consistent": "compliance",
            "supportive": "growth",
            "patient": "progress",
            "firm": "adherence",
            "detached": "performance",
            "calculating": "responses",
            "controlling": "submission",
            "mischievous": "predicament",
            "unpredictable": "confusion",
            "charismatic": "attention",
            "cruel": "suffering",
            "intense": "surrender",
            "demanding": "devotion",
            "manipulative": "thoughts",
            "perceptive": "reactions",
            "strategic": "development",
            "formal": "protocol",
            "ritualistic": "ceremonies",
            "detail_oriented": "precision",
            "appreciative": "service",
            "practical": "skills"
        }
        
        return trait_nouns.get(trait, "submission")
