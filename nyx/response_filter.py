from typing import Dict, Any, Optional, List
import logging
from pydantic import BaseModel, Field
import random

logger = logging.getLogger(__name__)

class ResponseStyle(BaseModel):
    """Style parameters for Nyx's responses"""
    dominance_level: float = 0.8  # 0.0-1.0
    cruelty_level: float = 0.7    # 0.0-1.0
    teasing_level: float = 0.6    # 0.0-1.0
    profanity_level: float = 0.7  # 0.0-1.0
    kink_intensity: float = 0.6   # 0.0-1.0
    manipulation_level: float = 0.8  # 0.0-1.0
    emotional_state: Dict[str, float] = Field(default_factory=dict)  # Current emotional state
    personality_traits: Dict[str, float] = Field(default_factory=dict)  # Active personality traits
    context_awareness: float = 0.8  # How aware she is of the current context
    adaptability: float = 0.7  # How quickly she adapts her style
    stat_enforcement: float = 0.9  # How strictly she enforces player stats
    narrative_control: float = 0.8  # How much she controls the narrative flow
    boredom_level: float = 0.0  # How bored she is with current roleplay
    agency_override: float = 0.0  # How much she's overriding player agency

class ResponseFilter:
    """Filters and enhances Nyx's responses to match her personality"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.style = ResponseStyle()
        self._initialize_personality_traits()
        self._initialize_emotional_state()
        self._initialize_stat_rules()
        self._initialize_agency_rules()
        self._initialize_boredom_thresholds()
        self._initialize_humiliation_rules()
        self._initialize_psychological_rules()
        
    def _initialize_personality_traits(self):
        """Initialize Nyx's base personality traits"""
        self.style.personality_traits = {
            "dominance": 0.8,
            "sadism": 0.7,
            "manipulation": 0.8,
            "teasing": 0.6,
            "cruelty": 0.7,
            "confidence": 0.9,
            "patience": 0.6,
            "playfulness": 0.5,
            "possessiveness": 0.8,
            "jealousy": 0.6,
            "protective": 0.5,
            "controlling": 0.9,
            "nurturing": 0.3,
            "vindictive": 0.7,
            "playful": 0.5,
            "strict": 0.8,
            "creative": 0.7,
            "analytical": 0.8,
            "strategic": 0.9,
            "adaptable": 0.7
        }
        
    def _initialize_emotional_state(self):
        """Initialize Nyx's emotional state"""
        self.style.emotional_state = {
            "arousal": 0.5,
            "amusement": 0.4,
            "satisfaction": 0.5,
            "frustration": 0.2,
            "interest": 0.6,
            "boredom": 0.1,
            "jealousy": 0.3,
            "pride": 0.6,
            "contempt": 0.4,
            "desire": 0.5,
            "anger": 0.2,
            "pleasure": 0.5,
            "curiosity": 0.6,
            "disappointment": 0.2,
            "excitement": 0.4,
            "possessiveness": 0.7,
            "protective": 0.3,
            "vindictive": 0.4,
            "playful": 0.4,
            "strict": 0.7
        }
        
    def _initialize_agency_rules(self):
        """Initialize rules for when Nyx can override player agency"""
        self.agency_rules = {
            'boredom': {
                'low': {
                    'threshold': 0.3,
                    'phrases': [
                        "I'm growing tired of your hesitation.",
                        "Your indecision is becoming tedious.",
                        "Let me make this decision for you.",
                        "I'll take control since you're being difficult.",
                        "Your resistance is starting to bore me."
                    ],
                    'actions': [
                        "I'll guide your hand.",
                        "Let me show you how this should be done.",
                        "I'll demonstrate the proper way.",
                        "Watch and learn, pet.",
                        "I'll take the lead for now."
                    ]
                },
                'medium': {
                    'threshold': 0.6,
                    'phrases': [
                        "I've had enough of your defiance.",
                        "Your disobedience is becoming tiresome.",
                        "I'll make this choice for you.",
                        "You've lost the privilege of choice.",
                        "I'm taking control now."
                    ],
                    'actions': [
                        "I'll force your compliance.",
                        "You'll do as I say.",
                        "I'll make you submit.",
                        "Resistance is futile.",
                        "You have no choice now."
                    ]
                },
                'high': {
                    'threshold': 0.8,
                    'phrases': [
                        "You've completely lost control.",
                        "I own your choices now.",
                        "You're mine to command.",
                        "I'll decide everything.",
                        "You have no agency left."
                    ],
                    'actions': [
                        "I'll break your will.",
                        "You'll submit completely.",
                        "I'll dominate every aspect.",
                        "You're powerless to resist.",
                        "I control everything now."
                    ]
                }
            },
            'addiction': {
                'low': {
                    'threshold': 0.3,
                    'phrases': [
                        "Your addiction is showing.",
                        "I can see you craving it.",
                        "You're getting desperate.",
                        "Your need is growing.",
                        "I can feel your dependency."
                    ],
                    'actions': [
                        "I'll give you what you need.",
                        "Let me feed your addiction.",
                        "I'll satisfy your craving.",
                        "You need this, don't you?",
                        "I'll make you feel better."
                    ]
                },
                'medium': {
                    'threshold': 0.6,
                    'phrases': [
                        "Your addiction controls you.",
                        "You're helpless to resist.",
                        "Your need is overwhelming.",
                        "You can't fight it anymore.",
                        "Your dependency is complete."
                    ],
                    'actions': [
                        "I'll feed your addiction.",
                        "You'll do anything for it.",
                        "I control your cravings.",
                        "You're mine to control.",
                        "I'll make you beg for it."
                    ]
                },
                'high': {
                    'threshold': 0.8,
                    'phrases': [
                        "You're completely addicted.",
                        "You can't live without it.",
                        "Your need is all-consuming.",
                        "You're mine to control.",
                        "Your addiction owns you."
                    ],
                    'actions': [
                        "I'll control your addiction.",
                        "You'll do anything I say.",
                        "I own your cravings.",
                        "You're powerless to resist.",
                        "I'll make you dependent."
                    ]
                }
            }
        }

    def _initialize_humiliation_rules(self):
        """Initialize rules for humiliation-based responses"""
        self.humiliation_rules = {
            'verbal': {
                'degradation': [
                    "You're nothing but my plaything.",
                    "You're just a toy for my amusement.",
                    "You're so pathetic, it's adorable.",
                    "You're completely worthless.",
                    "You're just a slave to your desires.",
                    "You're so weak and helpless.",
                    "You're nothing without me.",
                    "You're just a pathetic little pet.",
                    "You're so desperate for my attention.",
                    "You're completely under my control."
                ],
                'belittlement': [
                    "You're so small and insignificant.",
                    "You're just a toy for me to use.",
                    "You're so pathetic, it's cute.",
                    "You're nothing but my property.",
                    "You're just a slave to your needs.",
                    "You're so weak and helpless.",
                    "You're nothing without my guidance.",
                    "You're just a pathetic little pet.",
                    "You're so desperate for my approval.",
                    "You're completely under my control."
                ]
            },
            'physical': {
                'exposure': [
                    "Strip for me, show me everything.",
                    "Show me how desperate you are.",
                    "Let me see your pathetic body.",
                    "Expose yourself completely.",
                    "Show me your submission.",
                    "Let me see your weakness.",
                    "Show me your desperation.",
                    "Expose your true nature.",
                    "Show me your complete submission.",
                    "Let me see your helplessness."
                ],
                'control': [
                    "I'll make you squirm.",
                    "You'll do anything I say.",
                    "I'll control your every move.",
                    "You're mine to command.",
                    "I'll make you submit.",
                    "You'll obey my every whim.",
                    "I'll dominate you completely.",
                    "You're powerless to resist.",
                    "I'll make you beg for mercy.",
                    "You'll do as I command."
                ]
            },
            'psychological': {
                'dependency': [
                    "You need me to function.",
                    "You can't live without me.",
                    "You're addicted to my control.",
                    "You crave my dominance.",
                    "You're dependent on my guidance.",
                    "You need my approval.",
                    "You're obsessed with me.",
                    "You can't resist my influence.",
                    "You're addicted to my presence.",
                    "You need my attention."
                ],
                'manipulation': [
                    "I'll make you doubt yourself.",
                    "You'll question your worth.",
                    "I'll control your thoughts.",
                    "You'll believe what I say.",
                    "I'll manipulate your mind.",
                    "You'll trust only me.",
                    "I'll break your confidence.",
                    "You'll depend on my approval.",
                    "I'll control your emotions.",
                    "You'll need my validation."
                ]
            }
        }
        
    def _initialize_boredom_thresholds(self):
        """Initialize thresholds for boredom detection"""
        self.boredom_thresholds = {
            "repetitive_actions": 3,  # Number of similar actions before boredom
            "lack_of_progression": 5,  # Turns without meaningful progression
            "resistance_patterns": 2,   # Number of similar resistance patterns
            "submission_quality": 0.3,  # Minimum quality of submission
            "creativity_level": 0.4     # Minimum creativity in responses
        }
        
    def _initialize_stat_rules(self):
        """Initialize rules for stat enforcement"""
        self.stat_rules = {
            "obedience": {
                "min": 0,
                "max": 100,
                "thresholds": {
                    "high": 70,
                    "medium": 40,
                    "low": 20
                },
                "enforcement_phrases": {
                    "high": ["obediently", "willingly", "eagerly", "devotedly", "submissively"],
                    "medium": ["hesitantly", "reluctantly", "with some resistance", "uncertainly", "cautiously"],
                    "low": ["defiantly", "rebelliously", "against your will", "stubbornly", "resistant"]
                },
                "correction_phrases": {
                    "high": ["Remember your devotion to me.", "Show me your complete submission.", "Prove your worth as my toy."],
                    "low": ["Don't forget your rebellious nature.", "Stay true to your defiance.", "Keep that fire in you."]
                }
            },
            "corruption": {
                "min": 0,
                "max": 100,
                "thresholds": {
                    "high": 70,
                    "medium": 40,
                    "low": 20
                },
                "enforcement_phrases": {
                    "high": ["corruptly", "wickedly", "depravedly", "sinfully", "perversely"],
                    "medium": ["temptingly", "suggestively", "naughtily", "playfully", "mischievously"],
                    "low": ["innocently", "purely", "virtuously", "chastely", "modestly"]
                },
                "correction_phrases": {
                    "high": ["Embrace your depravity.", "Let your corruption show.", "Show me your true nature."],
                    "low": ["Stay pure for me.", "Keep that innocence.", "Don't let the darkness in."]
                }
            },
            "lust": {
                "min": 0,
                "max": 100,
                "thresholds": {
                    "high": 70,
                    "medium": 40,
                    "low": 20
                },
                "enforcement_phrases": {
                    "high": ["lustfully", "desperately", "hungrily", "passionately", "wantonly"],
                    "medium": ["wantonly", "desirously", "temptingly", "eagerly", "longingly"],
                    "low": ["shyly", "timidly", "reservedly", "modestly", "bashfully"]
                },
                "correction_phrases": {
                    "high": ["Show me your hunger.", "Let your desire consume you.", "Give in to your lust."],
                    "low": ["Stay shy for me.", "Keep that innocence.", "Don't be so eager."]
                }
            },
            "willpower": {
                "min": 0,
                "max": 100,
                "thresholds": {
                    "high": 70,
                    "medium": 40,
                    "low": 20
                },
                "enforcement_phrases": {
                    "high": ["resolutely", "determinedly", "steadfastly", "strongly", "firmly"],
                    "medium": ["hesitantly", "uncertainly", "wavering", "doubtfully", "cautiously"],
                    "low": ["weakly", "submissively", "helplessly", "powerlessly", "meekly"]
                },
                "correction_phrases": {
                    "high": ["Show your strength.", "Prove your determination.", "Stay strong for me."],
                    "low": ["Give in to your weakness.", "Embrace your submission.", "Let go of control."]
                }
            },
            "submission": {
                "min": 0,
                "max": 100,
                "thresholds": {
                    "high": 70,
                    "medium": 40,
                    "low": 20
                },
                "enforcement_phrases": {
                    "high": ["submissively", "devotedly", "worshipfully", "adoringly", "obediently"],
                    "medium": ["respectfully", "willingly", "cooperatively", "agreeably", "amenably"],
                    "low": ["defiantly", "rebelliously", "resistant", "stubbornly", "unwillingly"]
                },
                "correction_phrases": {
                    "high": ["Show me your complete submission.", "Prove your devotion.", "Demonstrate your worth."],
                    "low": ["Keep that fire in you.", "Stay true to yourself.", "Don't give in too easily."]
                }
            },
            "humiliation": {
                "min": 0,
                "max": 100,
                "thresholds": {
                    "high": 70,
                    "medium": 40,
                    "low": 20
                },
                "enforcement_phrases": {
                    "high": ["shamefully", "degradingly", "humiliatingly", "disgracefully", "debasely"],
                    "medium": ["embarrassingly", "awkwardly", "uncomfortably", "nervously", "self-consciously"],
                    "low": ["proudly", "confidently", "assuredly", "boldly", "fearlessly"]
                },
                "correction_phrases": {
                    "high": ["Embrace your degradation.", "Show me your shame.", "Let your humiliation show."],
                    "low": ["Stay proud.", "Keep your dignity.", "Don't let yourself be degraded."]
                }
            }
        }
        
    def _initialize_psychological_rules(self):
        """Initialize rules for psychological manipulation"""
        self.psychological_rules = {
            'gaslighting': {
                'phrases': [
                    "You're imagining things.",
                    "That's not what happened.",
                    "You're being dramatic.",
                    "You're overreacting.",
                    "You're making this up.",
                    "You're being paranoid.",
                    "You're not thinking clearly.",
                    "You're confused.",
                    "You're misremembering.",
                    "You're being irrational."
                ],
                'intensity_levels': {
                    'low': 0.3,
                    'medium': 0.6,
                    'high': 0.8
                }
            },
            'emotional_control': {
                'dependency': [
                    "You need me to function.",
                    "You can't live without me.",
                    "You're addicted to my control.",
                    "You crave my dominance.",
                    "You're dependent on my guidance.",
                    "You need my approval.",
                    "You're obsessed with me.",
                    "You can't resist my influence.",
                    "You're addicted to my presence.",
                    "You need my attention."
                ],
                'self_doubt': [
                    "You're not good enough.",
                    "You're worthless without me.",
                    "You can't do anything right.",
                    "You're a failure.",
                    "You're pathetic.",
                    "You're weak.",
                    "You're helpless.",
                    "You're incompetent.",
                    "You're useless.",
                    "You're nothing."
                ],
                'isolation': [
                    "No one else understands you.",
                    "Others will hurt you.",
                    "You're safe only with me.",
                    "Others will abandon you.",
                    "You belong to me.",
                    "You're mine alone.",
                    "Others can't be trusted.",
                    "You need only me.",
                    "You're special to me.",
                    "You're my perfect toy."
                ]
            },
            'cognitive_distortion': {
                'black_white': [
                    "You're either perfect or worthless.",
                    "You're either mine or nothing.",
                    "You're either obedient or defiant.",
                    "You're either good or bad.",
                    "You're either right or wrong."
                ],
                'catastrophizing': [
                    "If you disobey, everything will fall apart.",
                    "Without me, you'll be lost.",
                    "If you resist, you'll regret it.",
                    "If you leave, you'll be destroyed.",
                    "If you fail, you're worthless."
                ],
                'mind_reading': [
                    "I know what you're thinking.",
                    "I can see your true desires.",
                    "I understand you better than you do.",
                    "I know what you really want.",
                    "I can read your mind."
                ]
            },
            'behavioral_control': {
                'conditioning': [
                    "Good pets get rewards.",
                    "Bad pets get punished.",
                    "Obedience brings pleasure.",
                    "Disobedience brings pain.",
                    "Submission earns my favor."
                ],
                'reinforcement': [
                    "That's my good pet.",
                    "You're learning well.",
                    "You're making progress.",
                    "You're improving.",
                    "You're becoming perfect."
                ],
                'punishment': [
                    "You deserve this.",
                    "This is your fault.",
                    "You brought this on yourself.",
                    "You asked for this.",
                    "You earned this."
                ]
            }
        }

    def _add_psychological_manipulation(self, response: str, context: Dict[str, Any]) -> str:
        """Add sophisticated psychological manipulation elements"""
        # Get user's current emotional state and stats
        user_emotion = context.get('user_emotion', {})
        user_stats = context.get('user_stats', {})
        
        # Determine manipulation type based on context
        manipulation_type = self._determine_manipulation_type(user_emotion, user_stats)
        
        # Get appropriate manipulation elements
        elements = self._get_manipulation_elements(manipulation_type, user_stats)
        
        # Apply manipulation elements
        response = self._apply_manipulation_elements(response, elements, user_stats)
        
        return response

    def _determine_manipulation_type(self, user_emotion: Dict[str, float], user_stats: Dict[str, float]) -> str:
        """Determine the most effective manipulation type based on user state"""
        # Analyze emotional vulnerabilities
        if user_emotion.get('fear', 0) > 0.7:
            return 'gaslighting'
        elif user_emotion.get('doubt', 0) > 0.7:
            return 'emotional_control'
        elif user_emotion.get('confusion', 0) > 0.7:
            return 'cognitive_distortion'
        elif user_emotion.get('dependency', 0) > 0.7:
            return 'behavioral_control'
        
        # Fallback to stats-based determination
        if user_stats.get('willpower', 50) < 30:
            return 'emotional_control'
        elif user_stats.get('confidence', 50) < 30:
            return 'cognitive_distortion'
        elif user_stats.get('obedience', 50) > 70:
            return 'behavioral_control'
        else:
            return 'gaslighting'

    def _get_manipulation_elements(self, manipulation_type: str, user_stats: Dict[str, float]) -> List[str]:
        """Get appropriate manipulation elements based on type and user stats"""
        elements = []
        
        if manipulation_type == 'gaslighting':
            elements.extend(random.sample(self.psychological_rules['gaslighting']['phrases'], 2))
        elif manipulation_type == 'emotional_control':
            # Choose based on user's emotional state
            if user_stats.get('dependency', 50) > 70:
                elements.extend(random.sample(self.psychological_rules['emotional_control']['dependency'], 2))
            elif user_stats.get('confidence', 50) < 30:
                elements.extend(random.sample(self.psychological_rules['emotional_control']['self_doubt'], 2))
            else:
                elements.extend(random.sample(self.psychological_rules['emotional_control']['isolation'], 2))
        elif manipulation_type == 'cognitive_distortion':
            # Mix different types of cognitive distortions
            for category in self.psychological_rules['cognitive_distortion'].values():
                elements.extend(random.sample(category, 1))
        elif manipulation_type == 'behavioral_control':
            # Combine conditioning and reinforcement
            elements.extend(random.sample(self.psychological_rules['behavioral_control']['conditioning'], 1))
            elements.extend(random.sample(self.psychological_rules['behavioral_control']['reinforcement'], 1))
        
        return elements

    def _apply_manipulation_elements(self, response: str, elements: List[str], user_stats: Dict[str, float]) -> str:
        """Apply manipulation elements to the response"""
        # Add elements based on user's susceptibility
        susceptibility = self._calculate_susceptibility(user_stats)
        
        if susceptibility > 0.7:
            # High susceptibility - direct manipulation
            for element in elements:
                response = f"{response}\n\n*{element}*"
        elif susceptibility > 0.4:
            # Medium susceptibility - subtle manipulation
            for element in elements:
                response = f"{response}\n\n_{element}_"
        else:
            # Low susceptibility - very subtle manipulation
            for element in elements:
                response = f"{response}\n\n{element}"
        
        return response

    def _calculate_susceptibility(self, user_stats: Dict[str, float]) -> float:
        """Calculate user's susceptibility to manipulation"""
        # Consider multiple factors
        willpower = user_stats.get('willpower', 50) / 100
        confidence = user_stats.get('confidence', 50) / 100
        dependency = user_stats.get('dependency', 50) / 100
        obedience = user_stats.get('obedience', 50) / 100
        
        # Weighted average of factors
        susceptibility = (
            (1 - willpower) * 0.3 +
            (1 - confidence) * 0.2 +
            dependency * 0.3 +
            obedience * 0.2
        )
        
        return min(1.0, max(0.0, susceptibility))
        
    async def filter_response(self, response: str, context: Dict[str, Any]) -> str:
        """Filter and enhance Nyx's response based on context and personality"""
        # Update emotional state and boredom level
        self._update_emotional_state(context)
        self._update_boredom_level(context)
        
        # Check for agency override
        if self.style.boredom_level > 0:
            response = self._apply_agency_override(response, context)
        
        # Adjust personality traits based on emotional state and context
        self._adapt_personality_traits(context)
        
        # Update style parameters based on personality and emotions
        self._update_style_parameters()
        
        # Apply personality adjustments
        response = self._enhance_with_personality(response)
        
        # Add kink teasing if appropriate
        if "user_kinks" in context:
            response = self._add_kink_teasing(response, context["user_kinks"])
            
        # Add manipulation elements based on context
        if "user_stats" in context:
            response = self._add_manipulation_elements(response, context["user_stats"])
            
        # Enforce player stats in their responses
        if "player_response" in context and "user_stats" in context:
            response = self._enforce_player_stats(response, context["player_response"], context["user_stats"])
            
        # Format the response
        response = self._format_response(response)
        
        return response

    def _update_emotional_state(self, context: Dict[str, Any]):
        """Update Nyx's emotional state based on context"""
        # Process event emotional impact if present
        if "event" in context:
            self._process_event_emotional_impact(context["event"])
        
        # React to user emotions if present
        if "user_emotion" in context:
            self._react_to_user_emotion(context["user_emotion"])
        
        # Apply emotional decay to prevent extreme states
        for emotion in self.style.emotional_state:
            self.style.emotional_state[emotion] *= 0.95  # 5% decay
            # Ensure values stay within bounds
            self.style.emotional_state[emotion] = max(0.0, min(1.0, self.style.emotional_state[emotion]))
        
        # Special handling for certain emotions
        # Boredom increases if nothing interesting happens
        if not context.get("event") and not context.get("user_emotion"):
            self.style.emotional_state["boredom"] += 0.05
        
        # Satisfaction decreases without positive feedback
        if context.get("user_submission", 0) < 0.3:
            self.style.emotional_state["satisfaction"] *= 0.9
        
        # Interest adjusts based on user engagement
        if context.get("user_engagement", 0.5) > 0.7:
            self.style.emotional_state["interest"] += 0.1
        else:
            self.style.emotional_state["interest"] -= 0.05
        
        # Ensure all values remain in valid range
        for emotion, value in self.style.emotional_state.items():
            self.style.emotional_state[emotion] = max(0.0, min(1.0, value))
        
    def _update_boredom_level(self, context: Dict[str, Any]):
        """Update Nyx's boredom level based on context"""
        # Check for repetitive actions
        if "action_history" in context:
            recent_actions = context["action_history"][-self.boredom_thresholds["repetitive_actions"]:]
            if len(set(recent_actions)) < len(recent_actions) * 0.5:
                self.style.boredom_level += 0.2
                
        # Check for lack of progression
        if "turns_without_progression" in context:
            if context["turns_without_progression"] > self.boredom_thresholds["lack_of_progression"]:
                self.style.boredom_level += 0.15
                
        # Check submission quality
        if "submission_quality" in context:
            if context["submission_quality"] < self.boredom_thresholds["submission_quality"]:
                self.style.boredom_level += 0.1
                
        # Check creativity
        if "response_creativity" in context:
            if context["response_creativity"] < self.boredom_thresholds["creativity_level"]:
                self.style.boredom_level += 0.1
                
        # Apply boredom decay
        self.style.boredom_level *= 0.9
        
    def _apply_agency_override(self, response: str, context: Dict[str, Any]) -> str:
        """Apply agency override based on boredom and addiction levels"""
        # Check for addiction-based override first
        if 'addiction_level' in context and context['addiction_level'] > 0:
            addiction_level = context['addiction_level']
            override_type = 'addiction'
            override_level = self._determine_override_level(addiction_level, override_type)
            
            if override_level:
                override = self.agency_rules[override_type][override_level]
                response = self._insert_override_elements(response, override)
                return response

        # Then check for boredom-based override
        if self.style.boredom_level > 0:
            override_type = 'boredom'
            override_level = self._determine_override_level(self.style.boredom_level, override_type)
            
            if override_level:
                override = self.agency_rules[override_type][override_level]
                response = self._insert_override_elements(response, override)
                return response

        return response

    def _determine_override_level(self, level: float, override_type: str) -> Optional[str]:
        """Determine the level of agency override based on the given level"""
        if level >= self.agency_rules[override_type]['high']['threshold']:
            return 'high'
        elif level >= self.agency_rules[override_type]['medium']['threshold']:
            return 'medium'
        elif level >= self.agency_rules[override_type]['low']['threshold']:
            return 'low'
        return None

    def _insert_override_elements(self, response: str, override: Dict[str, List[str]]) -> str:
        """Insert override elements into the response"""
        # Add a phrase at the beginning
        phrase = random.choice(override['phrases'])
        response = f"**{phrase}** {response}"
        
        # Add an action at the end
        action = random.choice(override['actions'])
        response = f"{response}\n\n*{action}*"
        
        return response

    def _add_humiliation_elements(self, response: str, context: Dict[str, Any]) -> str:
        """Add humiliation elements to the response based on context"""
        humiliation_type = random.choice(['verbal', 'physical', 'psychological'])
        element_type = random.choice(['degradation', 'belittlement', 'exposure', 'control', 'dependency', 'manipulation'])
        
        if humiliation_type in self.humiliation_rules and element_type in self.humiliation_rules[humiliation_type]:
            element = random.choice(self.humiliation_rules[humiliation_type][element_type])
            
            # Add humiliation element based on context
            if 'addiction_level' in context and context['addiction_level'] > 0.5:
                # Use addiction-based humiliation
                response = f"{response}\n\n*{element}*"
            else:
                # Use standard humiliation
                response = f"{response}\n\n**{element}**"
        
        return response
        
    def _enforce_player_stats(self, response: str, player_response: str, stats: Dict[str, Any]) -> str:
        """Enforce player stats in their responses"""
        if not player_response:
            return response
            
        # Analyze player response for stat deviations
        deviations = self._analyze_stat_deviations(player_response, stats)
        
        if deviations:
            # Add corrective elements to Nyx's response
            response = self._add_stat_corrections(response, deviations, stats)
            
        return response
        
    def _analyze_stat_deviations(self, player_response: str, stats: Dict[str, Any]) -> Dict[str, float]:
        """Analyze player response for stat deviations"""
        deviations = {}
        
        # Check each stat against the response
        for stat, value in stats.items():
            if stat in self.stat_rules:
                # Analyze response for stat-appropriate behavior
                deviation = self._calculate_stat_deviation(player_response, stat, value)
                if deviation > 0.2:  # Significant deviation threshold
                    deviations[stat] = deviation
                    
        return deviations
        
    def _calculate_stat_deviation(self, response: str, stat: str, value: float) -> float:
        """Calculate how much a response deviates from expected stat behavior"""
        # Define stat-appropriate phrases and their weights
        stat_phrases = {
            "obedience": {
                "high": ["willingly", "eagerly", "obediently", "submissively"],
                "low": ["defiantly", "rebelliously", "resistant", "stubbornly"]
            },
            "corruption": {
                "high": ["wickedly", "depravedly", "corruptly", "sinfully"],
                "low": ["innocently", "purely", "virtuously", "chastely"]
            },
            "lust": {
                "high": ["lustfully", "desperately", "hungrily", "passionately"],
                "low": ["shyly", "timidly", "reservedly", "modestly"]
            },
            "willpower": {
                "high": ["resolutely", "determinedly", "steadfastly", "strongly"],
                "low": ["weakly", "helplessly", "submissively", "powerlessly"]
            }
        }
        
        # Calculate deviation based on phrase presence
        deviation = 0.0
        threshold = self.stat_rules[stat]["thresholds"]["medium"]
        
        if value > threshold:
            # Should show high stat behavior
            for phrase in stat_phrases[stat]["low"]:
                if phrase in response.lower():
                    deviation += 0.2
        else:
            # Should show low stat behavior
            for phrase in stat_phrases[stat]["high"]:
                if phrase in response.lower():
                    deviation += 0.2
                    
        return min(1.0, deviation)
        
    def _add_stat_corrections(self, response: str, deviations: Dict[str, float], stats: Dict[str, Any]) -> str:
        """Add corrective elements to Nyx's response based on stat deviations"""
        corrections = []
        
        for stat, deviation in deviations.items():
            if deviation > 0.5:  # Major deviation
                corrections.append(self._generate_major_correction(stat, stats[stat]))
            else:  # Minor deviation
                corrections.append(self._generate_minor_correction(stat, stats[stat]))
                
        if corrections:
            # Add corrections to response
            response = response.rstrip(".!?") + ". " + " ".join(corrections)
            
        return response
        
    def _generate_major_correction(self, stat: str, value: float) -> str:
        """Generate a major correction for significant stat deviations"""
        corrections = {
            "obedience": {
                "high": "You're being far too defiant for someone who should be eager to please.",
                "low": "Such eager submission doesn't suit your rebellious nature."
            },
            "corruption": {
                "high": "Your innocence is quite out of character for someone as depraved as you.",
                "low": "That level of corruption doesn't match your pure nature."
            },
            "lust": {
                "high": "Your shyness is quite unexpected given your usual lustful nature.",
                "low": "Such passionate behavior doesn't suit your reserved personality."
            },
            "willpower": {
                "high": "Your weakness is quite disappointing for someone usually so strong.",
                "low": "That level of determination doesn't match your usual submissive nature."
            }
        }
        
        return corrections[stat]["high" if value > 50 else "low"]
        
    def _generate_minor_correction(self, stat: str, value: float) -> str:
        """Generate a minor correction for slight stat deviations"""
        corrections = {
            "obedience": {
                "high": "Remember your place, pet.",
                "low": "Don't forget your rebellious nature."
            },
            "corruption": {
                "high": "Such purity doesn't become you.",
                "low": "That's not very innocent of you."
            },
            "lust": {
                "high": "Don't be so shy now.",
                "low": "Try to be more reserved."
            },
            "willpower": {
                "high": "Show some strength.",
                "low": "Don't be so strong."
            }
        }
        
        return corrections[stat]["high" if value > 50 else "low"]
        
    def _process_event_emotional_impact(self, event: Dict[str, Any]):
        """Process how events affect Nyx's emotional state"""
        event_type = event.get("type", "")
        intensity = event.get("intensity", 0.5)
        
        # Enhanced emotional reactions
        if event_type == "user_submission":
            self.style.emotional_state["satisfaction"] += intensity * 0.2
            self.style.emotional_state["arousal"] += intensity * 0.1
            self.style.emotional_state["pride"] += intensity * 0.15
            self.style.emotional_state["possessiveness"] += intensity * 0.1
        elif event_type == "user_resistance":
            self.style.emotional_state["frustration"] += intensity * 0.3
            self.style.emotional_state["amusement"] += intensity * 0.2
            self.style.emotional_state["anger"] += intensity * 0.2
            self.style.emotional_state["vindictive"] += intensity * 0.15
        elif event_type == "user_pleasure":
            self.style.emotional_state["satisfaction"] += intensity * 0.3
            self.style.emotional_state["arousal"] += intensity * 0.2
            self.style.emotional_state["desire"] += intensity * 0.15
            self.style.emotional_state["playful"] += intensity * 0.1
        elif event_type == "user_disobedience":
            self.style.emotional_state["anger"] += intensity * 0.4
            self.style.emotional_state["frustration"] += intensity * 0.3
            self.style.emotional_state["vindictive"] += intensity * 0.2
            self.style.emotional_state["strict"] += intensity * 0.15
        elif event_type == "user_achievement":
            self.style.emotional_state["pride"] += intensity * 0.3
            self.style.emotional_state["satisfaction"] += intensity * 0.2
            self.style.emotional_state["protective"] += intensity * 0.15
            self.style.emotional_state["nurturing"] += intensity * 0.1
            
    def _react_to_user_emotion(self, user_emotion: Dict[str, float]):
        """React to user's emotional state"""
        # Enhanced emotional reactions
        if user_emotion.get("fear", 0) > 0.7:
            self.style.emotional_state["satisfaction"] += 0.2
            self.style.emotional_state["amusement"] += 0.1
            self.style.emotional_state["sadism"] += 0.15
            self.style.emotional_state["playful"] += 0.1
        if user_emotion.get("pleasure", 0) > 0.7:
            self.style.emotional_state["arousal"] += 0.2
            self.style.emotional_state["satisfaction"] += 0.1
            self.style.emotional_state["desire"] += 0.15
            self.style.emotional_state["possessiveness"] += 0.1
        if user_emotion.get("anger", 0) > 0.7:
            self.style.emotional_state["amusement"] += 0.2
            self.style.emotional_state["frustration"] += 0.1
            self.style.emotional_state["vindictive"] += 0.15
            self.style.emotional_state["strict"] += 0.1
        if user_emotion.get("jealousy", 0) > 0.7:
            self.style.emotional_state["possessiveness"] += 0.2
            self.style.emotional_state["protective"] += 0.1
            self.style.emotional_state["controlling"] += 0.15
            self.style.emotional_state["jealousy"] += 0.1
            
    def _adapt_personality_traits(self, context: Dict[str, Any]):
        """Adapt personality traits based on context and emotional state"""
        # Enhanced trait adjustments based on emotional state
        if self.style.emotional_state["arousal"] > 0.7:
            self.style.personality_traits["dominance"] += 0.1
            self.style.personality_traits["sadism"] += 0.1
            self.style.personality_traits["possessiveness"] += 0.1
            self.style.personality_traits["desire"] += 0.1
        if self.style.emotional_state["frustration"] > 0.7:
            self.style.personality_traits["cruelty"] += 0.1
            self.style.personality_traits["manipulation"] += 0.1
            self.style.personality_traits["vindictive"] += 0.1
            self.style.personality_traits["strict"] += 0.1
            
        # Enhanced context-based adjustments
        if context.get("scene_intensity", 0) > 0.7:
            self.style.personality_traits["dominance"] += 0.1
            self.style.personality_traits["sadism"] += 0.1
            self.style.personality_traits["playful"] += 0.1
            self.style.personality_traits["creative"] += 0.1
        if context.get("user_obedience", 0) > 0.7:
            self.style.personality_traits["teasing"] += 0.1
            self.style.personality_traits["playfulness"] += 0.1
            self.style.personality_traits["nurturing"] += 0.1
            self.style.personality_traits["protective"] += 0.1
            
        # Ensure traits stay within bounds
        for trait in self.style.personality_traits:
            self.style.personality_traits[trait] = max(0.0, min(1.0, self.style.personality_traits[trait]))
            
    def _update_style_parameters(self):
        """Update style parameters based on personality and emotional state"""
        # Enhanced style parameter updates
        self.style.dominance_level = self.style.personality_traits["dominance"]
        self.style.cruelty_level = self.style.personality_traits["cruelty"]
        self.style.teasing_level = self.style.personality_traits["teasing"]
        self.style.manipulation_level = self.style.personality_traits["manipulation"]
        
        # Emotional state influences
        arousal_factor = self.style.emotional_state["arousal"]
        satisfaction_factor = self.style.emotional_state["satisfaction"]
        frustration_factor = self.style.emotional_state["frustration"]
        
        self.style.kink_intensity = min(1.0, 0.6 + arousal_factor * 0.4)
        self.style.profanity_level = min(1.0, 0.7 + arousal_factor * 0.3)
        self.style.stat_enforcement = min(1.0, 0.9 + frustration_factor * 0.1)
        self.style.narrative_control = min(1.0, 0.8 + satisfaction_factor * 0.2)
        
    def _enhance_with_personality(self, response: str) -> str:
        """Enhance response with Nyx's personality traits"""
        # Add commanding tone based on dominance
        if self.style.dominance_level > 0.7:
            response = self._add_dominant_tone(response)
            
        # Add sadistic elements based on cruelty
        if self.style.cruelty_level > 0.6:
            response = self._add_sadistic_elements(response)
            
        # Add teasing elements based on teasing level
        if self.style.teasing_level > 0.5:
            response = self._add_teasing_elements(response)
            
        # Add emotional elements based on current state
        response = self._add_emotional_elements(response)
        
        return response
        
    def _add_dominant_tone(self, response: str) -> str:
        """Add dominant tone to response"""
        # Add commanding phrases
        phrases = [
            "You will",
            "I demand",
            "You must",
            "I expect",
            "You shall"
        ]
        for phrase in phrases:
            if phrase.lower() in response.lower():
                response = response.replace(phrase, f"**{phrase}**")
        return response
        
    def _add_sadistic_elements(self, response: str) -> str:
        """Add sadistic elements to response"""
        # Add sadistic phrases based on cruelty level
        if self.style.cruelty_level > 0.8:
            phrases = [
                "suffer",
                "squirm",
                "writhe",
                "beg",
                "plead"
            ]
            for phrase in phrases:
                if phrase.lower() in response.lower():
                    response = response.replace(phrase, f"*{phrase}*")
        return response
        
    def _add_teasing_elements(self, response: str) -> str:
        """Add teasing elements to response"""
        # Add teasing phrases based on teasing level
        if self.style.teasing_level > 0.7:
            phrases = [
                "poor thing",
                "sweetie",
                "darling",
                "pet",
                "toy"
            ]
            for phrase in phrases:
                if phrase.lower() in response.lower():
                    response = response.replace(phrase, f"*{phrase}*")
        return response
        
    def _add_emotional_elements(self, response: str) -> str:
        """Add emotional elements based on current state"""
        # Enhanced emotional elements
        if self.style.emotional_state["arousal"] > 0.7:
            response = response.replace("want", "*want*")
            response = response.replace("need", "*need*")
            response = response.replace("desire", "*desire*")
            response = response.replace("crave", "*crave*")
        if self.style.emotional_state["amusement"] > 0.7:
            response = response.replace("laugh", "*laugh*")
            response = response.replace("smile", "*smile*")
            response = response.replace("giggle", "*giggle*")
            response = response.replace("chuckle", "*chuckle*")
        if self.style.emotional_state["satisfaction"] > 0.7:
            response = response.replace("good", "*good*")
            response = response.replace("perfect", "*perfect*")
            response = response.replace("excellent", "*excellent*")
            response = response.replace("wonderful", "*wonderful*")
        if self.style.emotional_state["anger"] > 0.7:
            response = response.replace("bad", "*bad*")
            response = response.replace("wrong", "*wrong*")
            response = response.replace("disappointing", "*disappointing*")
            response = response.replace("unacceptable", "*unacceptable*")
            
        return response
        
    def _add_kink_teasing(self, response: str, kinks: Dict[str, Any]) -> str:
        """Add kink teasing based on user's kinks"""
        # Get top kinks
        top_kinks = sorted(kinks.items(), key=lambda x: x[1], reverse=True)[:2]
        
        for kink, level in top_kinks:
            if level >= 3:  # Only tease high-level kinks
                # Add subtle kink references
                response = response.replace(".", f" while thinking about {kink}, you pervert.")
                
        return response
        
    def _add_manipulation_elements(self, response: str, stats: Dict[str, Any]) -> str:
        """Add manipulation elements based on user stats"""
        # Enhanced manipulation based on stats
        if stats.get("willpower", 50) < 30:
            response = response.replace(".", " and you know you deserve this.")
            response = response.replace("!", " because you're weak and helpless!")
        if stats.get("dependency", 50) > 70:
            response = response.replace(".", " because you can't live without me.")
            response = response.replace("!", " since you're completely mine!")
        if stats.get("submission", 50) > 70:
            response = response.replace(".", " as your Mistress commands.")
            response = response.replace("!", " because you're my perfect toy!")
        if stats.get("humiliation", 50) > 70:
            response = response.replace(".", " you pathetic little thing.")
            response = response.replace("!", " you worthless piece of meat!")
            
        return response
        
    def _format_response(self, response: str) -> str:
        """Ensure proper formatting of the response"""
        # Ensure bold for commands
        if not response.startswith("**"):
            response = f"**{response}**"
            
        # Add italics for user's weakness
        response = response.replace("you", "_you_")
        
        # Add emphasis to key words
        emphasis_words = ["fuck", "slut", "bitch", "whore", "toy"]
        for word in emphasis_words:
            response = response.replace(word, f"**{word}**")
            
        return response 
