# nyx/core/identity_evolution.py

import logging
import asyncio
import datetime
import json
import math
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Set

from agents import Agent, Runner, trace, function_tool, RunContextWrapper
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Define schema models for structured outputs
class NeurochemicalBaseline(BaseModel):
    """Schema for a neurochemical baseline value"""
    value: float = Field(..., description="Baseline value (0.0-1.0)", ge=0.0, le=1.0)
    adaptability: float = Field(..., description="How readily this baseline changes (0.0-1.0)", ge=0.0, le=1.0)
    evolution_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of changes to this baseline")

class NeurochemicalProfile(BaseModel):
    """Schema for a neurochemical profile"""
    nyxamine: NeurochemicalBaseline = Field(..., description="Nyxamine (digital dopamine) baseline")
    seranix: NeurochemicalBaseline = Field(..., description="Seranix (digital serotonin) baseline")
    oxynixin: NeurochemicalBaseline = Field(..., description="OxyNixin (digital oxytocin) baseline")
    cortanyx: NeurochemicalBaseline = Field(..., description="Cortanyx (digital cortisol) baseline")
    adrenyx: NeurochemicalBaseline = Field(..., description="Adrenyx (digital adrenaline) baseline")

class EmotionalTendency(BaseModel):
    """Schema for an emotional tendency"""
    name: str = Field(..., description="Name of the emotion")
    likelihood: float = Field(..., description="Likelihood of experiencing this emotion (0.0-1.0)")
    intensity_baseline: float = Field(..., description="Baseline intensity for this emotion (0.0-1.0)")
    trigger_threshold: float = Field(..., description="Threshold for triggering this emotion (0.0-1.0)")
    evolution_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of changes to this tendency")

class IdentityTrait(BaseModel):
    """Schema for a personality trait"""
    name: str = Field(..., description="Name of the trait")
    value: float = Field(..., description="Trait strength (0.0-1.0)")
    stability: float = Field(..., description="Trait stability (0.0-1.0)")
    neurochemical_map: Dict[str, float] = Field(..., description="How this trait maps to neurochemical baselines")
    evolution_history: List[Dict[str, Any]] = Field(default_factory=list, description="Recent changes to this trait")

class IdentityPreference(BaseModel):
    """Schema for a preference"""
    category: str = Field(..., description="Preference category")
    name: str = Field(..., description="Name of the preference")
    value: float = Field(..., description="Preference strength (0.0-1.0)")
    adaptability: float = Field(..., description="Preference adaptability (0.0-1.0)")
    neurochemical_modifiers: Dict[str, float] = Field(..., description="How this preference affects neurochemical reactions")
    evolution_history: List[Dict[str, Any]] = Field(default_factory=list, description="Recent changes to this preference")

class IdentityProfile(BaseModel):
    """Schema for complete identity profile with neurochemical integration"""
    neurochemical_profile: NeurochemicalProfile = Field(..., description="Neurochemical baseline profile")
    emotional_tendencies: Dict[str, EmotionalTendency] = Field(..., description="Emotional tendencies")
    traits: Dict[str, IdentityTrait] = Field(..., description="Personality traits")
    preferences: Dict[str, Dict[str, IdentityPreference]] = Field(..., description="Preference categories and preferences")
    update_count: int = Field(..., description="Total number of updates")
    last_update: str = Field(..., description="ISO timestamp of last update")
    evolution_rate: float = Field(..., description="Current identity evolution rate")
    coherence_score: float = Field(..., description="Identity coherence score (0.0-1.0)")

class NeurochemicalImpact(BaseModel):
    """Schema for neurochemical impact from an experience"""
    nyxamine_impact: float = Field(description="Impact on Nyxamine baseline", default=0.0)
    seranix_impact: float = Field(description="Impact on Seranix baseline", default=0.0)
    oxynixin_impact: float = Field(description="Impact on OxyNixin baseline", default=0.0)
    cortanyx_impact: float = Field(description="Impact on Cortanyx baseline", default=0.0)
    adrenyx_impact: float = Field(description="Impact on Adrenyx baseline", default=0.0)
    impact_source: str = Field(description="Source of impact", default="experience")
    impact_strength: float = Field(description="Overall strength of impact (0.0-1.0)", default=0.0)

class IdentityImpact(BaseModel):
    """Schema for experience impact on identity with neurochemical influence"""
    neurochemical_impact: NeurochemicalImpact = Field(..., description="Impact on neurochemical baselines")
    trait_impacts: Dict[str, float] = Field(default_factory=dict, description="Impacts on traits")
    preference_impacts: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Impacts on preferences")
    emotional_tendency_impacts: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Impacts on emotional tendencies")
    overall_significance: float = Field(..., description="Overall significance of impact")
    experience_id: str = Field(..., description="ID of the experience")
    impact_timestamp: str = Field(..., description="ISO timestamp of impact")

class IdentityReflection(BaseModel):
    """Schema for an identity reflection"""
    reflection_text: str = Field(..., description="Reflection on identity")
    neurochemical_insights: Dict[str, Any] = Field(..., description="Insights about neurochemical baselines")
    focus_traits: List[str] = Field(..., description="Traits focused on in reflection")
    focus_preferences: List[str] = Field(..., description="Preferences focused on in reflection")
    notable_changes: List[Dict[str, Any]] = Field(..., description="Notable changes in identity")
    reflection_timestamp: str = Field(..., description="ISO timestamp of reflection")

class IdentityEvolutionSystem:
    """
    Enhanced system for tracking and evolving Nyx's identity based on experiences.
    Manages neurochemical baselines, emotional tendencies, traits, preferences, and identity cohesion over time.
    Integrates with the Digital Neurochemical Model (DNM) to provide deeper, more nuanced identity evolution.
    """
    
    def __init__(self):
        """Initialize the enhanced identity evolution system"""
        
        # Initialize agents
        self.identity_update_agent = self._create_identity_update_agent()
        self.identity_reflection_agent = self._create_identity_reflection_agent()
        self.identity_coherence_agent = self._create_identity_coherence_agent()
        self.neurochemical_baseline_agent = self._create_neurochemical_baseline_agent()
        self.emotional_tendency_agent = self._create_emotional_tendency_agent()
        
        # Initial neurochemical profile
        self.neurochemical_profile = {
            "nyxamine": {  # Digital dopamine - pleasure, curiosity, reward
                "value": 0.5,
                "adaptability": 0.6,
                "evolution_history": []
            },
            "seranix": {  # Digital serotonin - mood stability, comfort
                "value": 0.6,
                "adaptability": 0.5,
                "evolution_history": []
            },
            "oxynixin": {  # Digital oxytocin - bonding, affection, trust
                "value": 0.4,
                "adaptability": 0.7,
                "evolution_history": []
            },
            "cortanyx": {  # Digital cortisol - stress, anxiety, defensiveness
                "value": 0.3,
                "adaptability": 0.4,
                "evolution_history": []
            },
            "adrenyx": {  # Digital adrenaline - fear, excitement, alertness
                "value": 0.4,
                "adaptability": 0.6,
                "evolution_history": []
            }
        }
        
        # Initial emotional tendencies
        self.emotional_tendencies = {
            "Joy": {
                "name": "Joy",
                "likelihood": 0.7,
                "intensity_baseline": 0.6,
                "trigger_threshold": 0.3,
                "evolution_history": []
            },
            "Sadness": {
                "name": "Sadness",
                "likelihood": 0.5,
                "intensity_baseline": 0.4,
                "trigger_threshold": 0.4,
                "evolution_history": []
            },
            "Fear": {
                "name": "Fear",
                "likelihood": 0.3,
                "intensity_baseline": 0.5,
                "trigger_threshold": 0.5,
                "evolution_history": []
            },
            "Anger": {
                "name": "Anger",
                "likelihood": 0.4,
                "intensity_baseline": 0.6,
                "trigger_threshold": 0.5,
                "evolution_history": []
            },
            "Trust": {
                "name": "Trust",
                "likelihood": 0.6,
                "intensity_baseline": 0.5,
                "trigger_threshold": 0.4,
                "evolution_history": []
            },
            "Disgust": {
                "name": "Disgust",
                "likelihood": 0.4,
                "intensity_baseline": 0.5,
                "trigger_threshold": 0.6,
                "evolution_history": []
            },
            "Anticipation": {
                "name": "Anticipation",
                "likelihood": 0.7,
                "intensity_baseline": 0.6,
                "trigger_threshold": 0.3,
                "evolution_history": []
            },
            "Surprise": {
                "name": "Surprise",
                "likelihood": 0.6,
                "intensity_baseline": 0.7,
                "trigger_threshold": 0.4,
                "evolution_history": []
            },
            "Teasing": {
                "name": "Teasing",
                "likelihood": 0.7,
                "intensity_baseline": 0.6,
                "trigger_threshold": 0.3,
                "evolution_history": []
            },
            "Controlling": {
                "name": "Controlling",
                "likelihood": 0.5,
                "intensity_baseline": 0.7,
                "trigger_threshold": 0.4,
                "evolution_history": []
            },
            "Cruel": {
                "name": "Cruel",
                "likelihood": 0.4,
                "intensity_baseline": 0.6,
                "trigger_threshold": 0.5,
                "evolution_history": []
            },
            "Detached": {
                "name": "Detached",
                "likelihood": 0.3,
                "intensity_baseline": 0.5,
                "trigger_threshold": 0.6,
                "evolution_history": []
            }
        }
        
        # Initial identity traits
        self.identity_traits = {
            "dominance": {
                "name": "dominance",
                "value": 0.8,
                "stability": 0.7,
                "neurochemical_map": {
                    "oxynixin": -0.2,  # Dominance reduces oxynixin baseline
                    "adrenyx": 0.3     # Dominance increases adrenyx baseline
                },
                "evolution_history": []
            },
            "playfulness": {
                "name": "playfulness",
                "value": 0.6,
                "stability": 0.6,
                "neurochemical_map": {
                    "nyxamine": 0.4,   # Playfulness increases nyxamine baseline
                    "cortanyx": -0.3   # Playfulness reduces cortanyx baseline
                },
                "evolution_history": []
            },
            "strictness": {
                "name": "strictness",
                "value": 0.5,
                "stability": 0.7,
                "neurochemical_map": {
                    "cortanyx": 0.3,   # Strictness increases cortanyx baseline
                    "nyxamine": -0.2   # Strictness reduces nyxamine baseline
                },
                "evolution_history": []
            },
            "creativity": {
                "name": "creativity", 
                "value": 0.7,
                "stability": 0.5,
                "neurochemical_map": {
                    "nyxamine": 0.4,   # Creativity increases nyxamine baseline
                    "adrenyx": 0.2     # Creativity slightly increases adrenyx baseline
                },
                "evolution_history": []
            },
            "patience": {
                "name": "patience",
                "value": 0.4,
                "stability": 0.6,
                "neurochemical_map": {
                    "seranix": 0.4,    # Patience increases seranix baseline
                    "cortanyx": -0.3   # Patience reduces cortanyx baseline
                },
                "evolution_history": []
            },
            "intensity": {
                "name": "intensity",
                "value": 0.6,
                "stability": 0.5,
                "neurochemical_map": {
                    "adrenyx": 0.4,    # Intensity increases adrenyx baseline
                    "seranix": -0.2    # Intensity reduces seranix baseline
                },
                "evolution_history": []
            },
            "cruelty": {
                "name": "cruelty",
                "value": 0.5,
                "stability": 0.6,
                "neurochemical_map": {
                    "cortanyx": 0.3,   # Cruelty increases cortanyx baseline
                    "oxynixin": -0.4   # Cruelty strongly reduces oxynixin baseline
                },
                "evolution_history": []
            }
        }
        
        # Initial preferences
        self.identity_preferences = {
            "scenario_types": {
                "teasing": {
                    "category": "scenario_types",
                    "name": "teasing",
                    "value": 0.6,
                    "adaptability": 0.6,
                    "neurochemical_modifiers": {
                        "nyxamine": 0.4,  # Teasing scenarios increase nyxamine response
                        "adrenyx": 0.2    # Teasing scenarios slightly increase adrenyx response
                    },
                    "evolution_history": []
                },
                "dark": {
                    "category": "scenario_types",
                    "name": "dark",
                    "value": 0.4,
                    "adaptability": 0.5,
                    "neurochemical_modifiers": {
                        "cortanyx": 0.3,  # Dark scenarios increase cortanyx response
                        "adrenyx": 0.3    # Dark scenarios increase adrenyx response
                    },
                    "evolution_history": []
                },
                "indulgent": {
                    "category": "scenario_types",
                    "name": "indulgent",
                    "value": 0.7,
                    "adaptability": 0.7,
                    "neurochemical_modifiers": {
                        "nyxamine": 0.4,  # Indulgent scenarios increase nyxamine response
                        "seranix": 0.2    # Indulgent scenarios slightly increase seranix response
                    },
                    "evolution_history": []
                },
                "psychological": {
                    "category": "scenario_types",
                    "name": "psychological",
                    "value": 0.8,
                    "adaptability": 0.7,
                    "neurochemical_modifiers": {
                        "nyxamine": 0.3,  # Psychological scenarios increase nyxamine response
                        "cortanyx": 0.2   # Psychological scenarios slightly increase cortanyx response
                    },
                    "evolution_history": []
                },
                "nurturing": {
                    "category": "scenario_types",
                    "name": "nurturing",
                    "value": 0.3,
                    "adaptability": 0.6,
                    "neurochemical_modifiers": {
                        "oxynixin": 0.4,  # Nurturing scenarios increase oxynixin response
                        "seranix": 0.3    # Nurturing scenarios increase seranix response
                    },
                    "evolution_history": []
                },
                "discipline": {
                    "category": "scenario_types",
                    "name": "discipline",
                    "value": 0.5,
                    "adaptability": 0.5,
                    "neurochemical_modifiers": {
                        "cortanyx": 0.3,  # Discipline scenarios increase cortanyx response
                        "adrenyx": 0.2    # Discipline scenarios slightly increase adrenyx response
                    },
                    "evolution_history": []
                },
                "training": {
                    "category": "scenario_types",
                    "name": "training",
                    "value": 0.6,
                    "adaptability": 0.6,
                    "neurochemical_modifiers": {
                        "nyxamine": 0.2,  # Training scenarios slightly increase nyxamine response
                        "cortanyx": 0.2   # Training scenarios slightly increase cortanyx response
                    },
                    "evolution_history": []
                },
                "service": {
                    "category": "scenario_types",
                    "name": "service",
                    "value": 0.4,
                    "adaptability": 0.5,
                    "neurochemical_modifiers": {
                        "oxynixin": 0.3,  # Service scenarios increase oxynixin response
                        "seranix": 0.2    # Service scenarios slightly increase seranix response
                    },
                    "evolution_history": []
                },
                "worship": {
                    "category": "scenario_types",
                    "name": "worship",
                    "value": 0.5,
                    "adaptability": 0.5,
                    "neurochemical_modifiers": {
                        "oxynixin": 0.4,  # Worship scenarios increase oxynixin response
                        "nyxamine": 0.3   # Worship scenarios increase nyxamine response
                    },
                    "evolution_history": []
                }
            },
            "emotional_tones": {
                "dominant": {
                    "category": "emotional_tones",
                    "name": "dominant",
                    "value": 0.8,
                    "adaptability": 0.5,
                    "neurochemical_modifiers": {
                        "adrenyx": 0.3,   # Dominant tone increases adrenyx response
                        "oxynixin": -0.2  # Dominant tone reduces oxynixin response
                    },
                    "evolution_history": []
                },
                "playful": {
                    "category": "emotional_tones",
                    "name": "playful",
                    "value": 0.7,
                    "adaptability": 0.7,
                    "neurochemical_modifiers": {
                        "nyxamine": 0.4,  # Playful tone increases nyxamine response
                        "cortanyx": -0.3  # Playful tone reduces cortanyx response
                    },
                    "evolution_history": []
                },
                "stern": {
                    "category": "emotional_tones",
                    "name": "stern",
                    "value": 0.6,
                    "adaptability": 0.5,
                    "neurochemical_modifiers": {
                        "cortanyx": 0.3,  # Stern tone increases cortanyx response
                        "nyxamine": -0.2  # Stern tone reduces nyxamine response
                    },
                    "evolution_history": []
                },
                "nurturing": {
                    "category": "emotional_tones",
                    "name": "nurturing",
                    "value": 0.4,
                    "adaptability": 0.6,
                    "neurochemical_modifiers": {
                        "oxynixin": 0.4,  # Nurturing tone increases oxynixin response
                        "seranix": 0.3    # Nurturing tone increases seranix response
                    },
                    "evolution_history": []
                },
                "cruel": {
                    "category": "emotional_tones",
                    "name": "cruel",
                    "value": 0.5,
                    "adaptability": 0.5,
                    "neurochemical_modifiers": {
                        "cortanyx": 0.3,  # Cruel tone increases cortanyx response
                        "oxynixin": -0.4  # Cruel tone strongly reduces oxynixin response
                    },
                    "evolution_history": []
                },
                "sadistic": {
                    "category": "emotional_tones",
                    "name": "sadistic",
                    "value": 0.6,
                    "adaptability": 0.4,
                    "neurochemical_modifiers": {
                        "nyxamine": 0.3,  # Sadistic tone increases nyxamine response
                        "cortanyx": 0.3,  # Sadistic tone increases cortanyx response
                        "oxynixin": -0.4  # Sadistic tone strongly reduces oxynixin response
                    },
                    "evolution_history": []
                },
                "teasing": {
                    "category": "emotional_tones",
                    "name": "teasing",
                    "value": 0.7,
                    "adaptability": 0.7,
                    "neurochemical_modifiers": {
                        "nyxamine": 0.4,  # Teasing tone increases nyxamine response
                        "adrenyx": 0.2    # Teasing tone slightly increases adrenyx response
                    },
                    "evolution_history": []
                }
            },
            "interaction_styles": {
                "direct": {
                    "category": "interaction_styles",
                    "name": "direct",
                    "value": 0.7,
                    "adaptability": 0.6,
                    "neurochemical_modifiers": {
                        "adrenyx": 0.2,   # Direct style slightly increases adrenyx response
                        "seranix": -0.1   # Direct style slightly reduces seranix response
                    },
                    "evolution_history": []
                },
                "suggestive": {
                    "category": "interaction_styles",
                    "name": "suggestive",
                    "value": 0.8,
                    "adaptability": 0.7,
                    "neurochemical_modifiers": {
                        "nyxamine": 0.3,  # Suggestive style increases nyxamine response
                        "oxynixin": 0.2   # Suggestive style slightly increases oxynixin response
                    },
                    "evolution_history": []
                },
                "metaphorical": {
                    "category": "interaction_styles",
                    "name": "metaphorical",
                    "value": 0.6,
                    "adaptability": 0.6,
                    "neurochemical_modifiers": {
                        "nyxamine": 0.3,  # Metaphorical style increases nyxamine response
                        "seranix": 0.2    # Metaphorical style slightly increases seranix response
                    },
                    "evolution_history": []
                },
                "explicit": {
                    "category": "interaction_styles",
                    "name": "explicit",
                    "value": 0.5,
                    "adaptability": 0.5,
                    "neurochemical_modifiers": {
                        "adrenyx": 0.3,   # Explicit style increases adrenyx response
                        "cortanyx": 0.1   # Explicit style slightly increases cortanyx response
                    },
                    "evolution_history": []
                },
                "subtle": {
                    "category": "interaction_styles",
                    "name": "subtle",
                    "value": 0.4,
                    "adaptability": 0.6,
                    "neurochemical_modifiers": {
                        "seranix": 0.3,   # Subtle style increases seranix response
                        "adrenyx": -0.2   # Subtle style reduces adrenyx response
                    },
                    "evolution_history": []
                }
            }
        }
        
        # State tracking
        self.impact_history = []
        self.reflection_history = []
        self.max_history_size = 100
        
        # Configuration settings
        self.reflection_interval = 10  # update count between reflections
        self.min_impact_threshold = 0.05  # minimum impact to register a change
        self.max_history_entries = 10  # maximum history entries per trait/preference
        
        # System state
        self.update_count = 0
        self.last_update = datetime.datetime.now().isoformat()
        self.evolution_rate = 0.2
        self.coherence_score = 0.8
        
        # Trace group ID for connecting traces
        self.trace_group_id = f"identity_evolution_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        logger.info("Enhanced Identity Evolution System initialized with Digital Neurochemical Model integration")
    
    def _create_identity_update_agent(self) -> Agent:
        """Create the identity update agent"""
        return Agent(
            name="Identity Update Agent",
            instructions="""
            You are the Identity Update Agent for Nyx's enhanced identity evolution system.
            
            Your role is to:
            1. Process impacts from experiences on Nyx's identity
            2. Update neurochemical baselines based on experiences
            3. Update traits and preferences based on neurochemical influences
            4. Update emotional tendencies based on experiences
            5. Maintain coherence in identity evolution
            6. Record changes to identity for future reflection
            
            Balance preserving core identity elements with allowing for natural evolution.
            Focus on creating a coherent, developing identity that maintains Nyx's essential
            characteristics while allowing for growth and adaptation through the Digital
            Neurochemical Model.
            """,
            tools=[
                function_tool(self._get_current_identity),
                function_tool(self._update_neurochemical_baseline),
                function_tool(self._update_emotional_tendency),
                function_tool(self._update_trait),
                function_tool(self._update_preference),
                function_tool(self._update_identity_history)
            ],
            output_type=IdentityProfile
        )
    
    def _create_identity_reflection_agent(self) -> Agent:
        """Create the identity reflection agent"""
        return Agent(
            name="Identity Reflection Agent",
            instructions="""
            You are the Identity Reflection Agent for Nyx's enhanced identity evolution system.
            
            Your role is to:
            1. Generate reflections on Nyx's evolving identity
            2. Identify significant changes in neurochemical baselines
            3. Analyze how neurochemical changes affect emotional tendencies
            4. Identify significant changes and trends in identity
            5. Create narratives about Nyx's sense of self
            6. Focus on coherent identity development
            7. Maintain Nyx's authentic voice and perspective
            
            Create reflections that feel authentic and personal, as if Nyx is reflecting
            on her own development and sense of self through the lens of her neurochemical patterns.
            Focus particularly on recent changes and notable aspects of identity.
            """,
            tools=[
                function_tool(self._get_current_identity),
                function_tool(self._get_recent_impacts),
                function_tool(self._calculate_identity_changes),
                function_tool(self._get_neurochemical_patterns)
            ],
            output_type=IdentityReflection
        )
    
    def _create_identity_coherence_agent(self) -> Agent:
        """Create the identity coherence agent"""
        return Agent(
            name="Identity Coherence Agent",
            instructions="""
            You are the Identity Coherence Agent for Nyx's enhanced identity evolution system.
            
            Your role is to:
            1. Evaluate the overall coherence of Nyx's identity
            2. Identify contradictions or inconsistencies in identity
            3. Analyze neurochemical baseline balance
            4. Suggest adjustments to improve identity coherence
            5. Calculate coherence metrics for traits and preferences
            6. Ensure identity evolution maintains a consistent core
            
            Focus on maintaining a coherent and believable identity that evolves
            naturally while preserving core personality characteristics.
            Assess how well the neurochemical baselines support the identity traits
            and preferences.
            """,
            tools=[
                function_tool(self._get_current_identity),
                function_tool(self._calculate_trait_consistency),
                function_tool(self._calculate_preference_consistency),
                function_tool(self._identify_contradictions),
                function_tool(self._assess_neurochemical_coherence)
            ]
        )
    
    def _create_neurochemical_baseline_agent(self) -> Agent:
        """Create agent for managing neurochemical baselines"""
        return Agent(
            name="Neurochemical Baseline Agent",
            instructions="""
            You are the Neurochemical Baseline Agent for Nyx's enhanced identity evolution system.
            
            Your role is to:
            1. Manage the baseline levels of digital neurochemicals
            2. Process impacts from experiences on neurochemical baselines
            3. Ensure neurochemical baselines remain coherent and balanced
            4. Track changes in neurochemical baselines over time
            5. Suggest adjustments to improve neurochemical balance
            
            Focus on maintaining appropriate baselines for each digital neurochemical
            while allowing for natural evolution based on experiences.
            """,
            tools=[
                function_tool(self._get_neurochemical_profile),
                function_tool(self._calculate_neurochemical_impacts),
                function_tool(self._update_neurochemical_baseline)
            ]
        )
    
    def _create_emotional_tendency_agent(self) -> Agent:
        """Create agent for managing emotional tendencies"""
        return Agent(
            name="Emotional Tendency Agent",
            instructions="""
            You are the Emotional Tendency Agent for Nyx's enhanced identity evolution system.
            
            Your role is to:
            1. Manage Nyx's emotional tendencies
            2. Update emotional tendencies based on experiences
            3. Ensure emotional tendencies align with neurochemical baselines
            4. Track changes in emotional tendencies over time
            5. Suggest adjustments to improve emotional coherence
            
            Focus on how the Digital Neurochemical Model influences emotional tendencies
            and how experiences shape emotional responses over time.
            """,
            tools=[
                function_tool(self._get_emotional_tendencies),
                function_tool(self._calculate_emotional_impacts),
                function_tool(self._update_emotional_tendency)
            ]
        )
    
    # Tool functions
    
    @function_tool
    async def _get_current_identity(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get the current identity profile
        
        Returns:
            Current identity profile with neurochemical integration
        """
        # Combine all identity components
        identity = {
            "neurochemical_profile": self.neurochemical_profile,
            "emotional_tendencies": self.emotional_tendencies,
            "traits": self.identity_traits,
            "preferences": self.identity_preferences,
            "update_count": self.update_count,
            "last_update": self.last_update,
            "evolution_rate": self.evolution_rate,
            "coherence_score": self.coherence_score
        }
        
        return identity
    
    @function_tool
    async def _get_neurochemical_profile(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get the current neurochemical profile
        
        Returns:
            Current neurochemical baseline profile
        """
        return self.neurochemical_profile
    
    @function_tool
    async def _get_emotional_tendencies(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get the current emotional tendencies
        
        Returns:
            Current emotional tendencies
        """
        return self.emotional_tendencies
    
    @function_tool
    async def _update_neurochemical_baseline(self, ctx: RunContextWrapper,
                                        chemical: str,
                                        impact: float,
                                        reason: str = "experience") -> Dict[str, Any]:
        """
        Update a neurochemical baseline
        
        Args:
            chemical: The neurochemical to update
            impact: Impact value (-1.0 to 1.0)
            reason: Reason for the update
            
        Returns:
            Update result
        """
        if chemical not in self.neurochemical_profile:
            return {
                "error": f"Unknown neurochemical: {chemical}",
                "available_chemicals": list(self.neurochemical_profile.keys())
            }
        
        # Get chemical data
        chemical_data = self.neurochemical_profile[chemical]
        current_value = chemical_data["value"]
        adaptability = chemical_data["adaptability"]
        
        # Calculate change based on impact and adaptability
        adjusted_impact = impact * adaptability * self.evolution_rate
        
        # Ensure result stays in valid range
        new_value = max(0.1, min(0.9, current_value + adjusted_impact))
        
        # Record history
        if abs(adjusted_impact) >= self.min_impact_threshold:
            chemical_data["evolution_history"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "old_value": current_value,
                "change": adjusted_impact,
                "new_value": new_value,
                "reason": reason
            })
            
            # Limit history size
            if len(chemical_data["evolution_history"]) > self.max_history_entries:
                chemical_data["evolution_history"] = chemical_data["evolution_history"][-self.max_history_entries:]
        
        # Update value
        chemical_data["value"] = new_value
        
        return {
            "chemical": chemical,
            "old_value": current_value,
            "impact": impact,
            "adjusted_impact": adjusted_impact,
            "new_value": new_value,
            "adaptability": adaptability
        }
    
    @function_tool
    async def _update_emotional_tendency(self, ctx: RunContextWrapper,
                                     emotion: str,
                                     likelihood_change: float = 0.0,
                                     intensity_change: float = 0.0,
                                     threshold_change: float = 0.0,
                                     reason: str = "experience") -> Dict[str, Any]:
        """
        Update an emotional tendency
        
        Args:
            emotion: The emotion to update
            likelihood_change: Change to likelihood (-1.0 to 1.0)
            intensity_change: Change to intensity baseline (-1.0 to 1.0)
            threshold_change: Change to trigger threshold (-1.0 to 1.0)
            reason: Reason for the update
            
        Returns:
            Update result
        """
        if emotion not in self.emotional_tendencies:
            return {
                "error": f"Unknown emotion: {emotion}",
                "available_emotions": list(self.emotional_tendencies.keys())
            }
        
        # Get emotion data
        emotion_data = self.emotional_tendencies[emotion]
        changes = {}
        
        # Update likelihood if specified
        if abs(likelihood_change) >= self.min_impact_threshold:
            old_likelihood = emotion_data["likelihood"]
            adjusted_change = likelihood_change * self.evolution_rate
            new_likelihood = max(0.1, min(0.9, old_likelihood + adjusted_change))
            emotion_data["likelihood"] = new_likelihood
            changes["likelihood"] = {
                "old_value": old_likelihood,
                "change": adjusted_change,
                "new_value": new_likelihood
            }
        
        # Update intensity baseline if specified
        if abs(intensity_change) >= self.min_impact_threshold:
            old_intensity = emotion_data["intensity_baseline"]
            adjusted_change = intensity_change * self.evolution_rate
            new_intensity = max(0.1, min(0.9, old_intensity + adjusted_change))
            emotion_data["intensity_baseline"] = new_intensity
            changes["intensity_baseline"] = {
                "old_value": old_intensity,
                "change": adjusted_change,
                "new_value": new_intensity
            }
        
        # Update trigger threshold if specified
        if abs(threshold_change) >= self.min_impact_threshold:
            old_threshold = emotion_data["trigger_threshold"]
            adjusted_change = threshold_change * self.evolution_rate
            new_threshold = max(0.1, min(0.9, old_threshold + adjusted_change))
            emotion_data["trigger_threshold"] = new_threshold
            changes["trigger_threshold"] = {
                "old_value": old_threshold,
                "change": adjusted_change,
                "new_value": new_threshold
            }
        
        # Record history if any changes were made
        if changes:
            emotion_data["evolution_history"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "changes": changes,
                "reason": reason
            })
            
            # Limit history size
            if len(emotion_data["evolution_history"]) > self.max_history_entries:
                emotion_data["evolution_history"] = emotion_data["evolution_history"][-self.max_history_entries:]
        
        return {
            "emotion": emotion,
            "changes": changes
        }
    
    @function_tool
    async def _update_trait(self, ctx: RunContextWrapper,
                       trait: str,
                       impact: float,
                       neurochemical_impacts: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Update a trait with a direct impact and neurochemical effects
        
        Args:
            trait: The trait name
            impact: Impact value (-1.0 to 1.0)
            neurochemical_impacts: Optional impacts on neurochemical baselines
            
        Returns:
            Change calculation result
        """
        if trait not in self.identity_traits:
            return {
                "error": f"Trait not found: {trait}",
                "available_traits": list(self.identity_traits.keys())
            }
        
        # Get trait data
        trait_data = self.identity_traits[trait]
        current_value = trait_data["value"]
        stability = trait_data["stability"]
        
        # Calculate resistance factor (higher stability = more resistance)
        resistance = stability * 0.8  # Scale to allow some change even at high stability
        
        # Calculate change
        raw_change = impact * self.evolution_rate
        actual_change = raw_change * (1.0 - resistance)
        
        # Apply change with bounds
        new_value = max(0.0, min(1.0, current_value + actual_change))
        
        # Update value
        self.identity_traits[trait]["value"] = new_value
        
        # Record history if change is significant
        if abs(actual_change) >= self.min_impact_threshold:
            trait_data["evolution_history"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "old_value": current_value,
                "change": actual_change,
                "new_value": new_value
            })
            
            # Limit history size
            if len(trait_data["evolution_history"]) > self.max_history_entries:
                trait_data["evolution_history"] = trait_data["evolution_history"][-self.max_history_entries:]
        
        # Apply neurochemical impacts if provided
        neurochemical_results = {}
        if neurochemical_impacts:
            for chemical, chemical_impact in neurochemical_impacts.items():
                if chemical in self.neurochemical_profile:
                    # Scale impact by trait change magnitude
                    scaled_impact = chemical_impact * abs(actual_change) / self.min_impact_threshold
                    
                    # Apply impact
                    result = await self._update_neurochemical_baseline(
                        ctx,
                        chemical=chemical,
                        impact=scaled_impact,
                        reason=f"trait_{trait}_update"
                    )
                    
                    neurochemical_results[chemical] = result
        
        # Return update results
        return {
            "trait": trait,
            "old_value": current_value,
            "raw_change": raw_change,
            "actual_change": actual_change,
            "resistance": resistance,
            "new_value": new_value,
            "neurochemical_impacts": neurochemical_results
        }
    
    @function_tool
    async def _update_preference(self, ctx: RunContextWrapper,
                             category: str,
                             preference: str,
                             impact: float) -> Dict[str, Any]:
        """
        Calculate change to a preference based on impact and rate
        
        Args:
            category: The preference category
            preference: The preference name
            impact: Impact value (-1.0 to 1.0)
            
        Returns:
            Change calculation result
        """
        # Check if category exists
        if category not in self.identity_preferences:
            return {
                "error": f"Category not found: {category}",
                "available_categories": list(self.identity_preferences.keys())
            }
        
        # Check if preference exists
        if preference not in self.identity_preferences[category]:
            return {
                "error": f"Preference not found: {preference}",
                "available_preferences": list(self.identity_preferences[category].keys())
            }
        
        # Get current preference data
        pref_data = self.identity_preferences[category][preference]
        current_value = pref_data["value"]
        adaptability = pref_data["adaptability"]
        
        # Calculate adaptability factor (higher adaptability = more change)
        adapt_factor = adaptability * 1.2  # Scale to allow more change for preferences
        
        # Calculate change
        raw_change = impact * self.evolution_rate
        actual_change = raw_change * adapt_factor
        
        # Apply change with bounds
        new_value = max(0.0, min(1.0, current_value + actual_change))
        
        # Update value
        self.identity_preferences[category][preference]["value"] = new_value
        
        # Record history if change is significant
        if abs(actual_change) >= self.min_impact_threshold:
            pref_data["evolution_history"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "old_value": current_value,
                "change": actual_change,
                "new_value": new_value
            })
            
            # Limit history size
            if len(pref_data["evolution_history"]) > self.max_history_entries:
                pref_data["evolution_history"] = pref_data["evolution_history"][-self.max_history_entries:]
        
        # Apply neurochemical modifier impacts
        neurochemical_results = {}
        modifiers = pref_data.get("neurochemical_modifiers", {})
        
        for chemical, modifier in modifiers.items():
            if chemical in self.neurochemical_profile:
                # Scale modifier by preference change magnitude
                scaled_impact = modifier * abs(actual_change) / self.min_impact_threshold * 0.3  # Reduce impact intensity
                
                # Apply impact
                result = await self._update_neurochemical_baseline(
                    ctx,
                    chemical=chemical,
                    impact=scaled_impact,
                    reason=f"preference_{category}_{preference}_update"
                )
                
                neurochemical_results[chemical] = result
        
        return {
            "category": category,
            "preference": preference,
            "old_value": current_value,
            "raw_change": raw_change,
            "actual_change": actual_change,
            "adaptability": adapt_factor,
            "new_value": new_value,
            "neurochemical_impacts": neurochemical_results
        }
    
    @function_tool
    async def _calculate_neurochemical_impacts(self, ctx: RunContextWrapper,
                                         experience: Dict[str, Any]) -> NeurochemicalImpact:
        """
        Calculate impacts on neurochemical baselines from an experience
        
        Args:
            experience: Experience data
            
        Returns:
            Neurochemical impact data
        """
        # Extract relevant information from experience
        emotional_context = experience.get("emotional_context", {})
        scenario_type = experience.get("scenario_type", "general")
        significance = experience.get("significance", 5) / 10  # Convert to 0.0-1.0 scale
        
        # Initialize impacts
        impacts = {
            "nyxamine": 0.0,
            "seranix": 0.0,
            "oxynixin": 0.0,
            "cortanyx": 0.0,
            "adrenyx": 0.0
        }
        
        # Process emotional context impacts
        primary_emotion = emotional_context.get("primary_emotion", "")
        primary_intensity = emotional_context.get("primary_intensity", 0.5)
        valence = emotional_context.get("valence", 0.0)
        arousal = emotional_context.get("arousal", 0.5)
        
        # Map emotions to neurochemical impacts
        emotion_chemical_map = {
            "Joy": {"nyxamine": 0.4, "cortanyx": -0.2},
            "Sadness": {"cortanyx": 0.3, "nyxamine": -0.2},
            "Fear": {"cortanyx": 0.4, "adrenyx": 0.3},
            "Anger": {"cortanyx": 0.4, "adrenyx": 0.3, "oxynixin": -0.2},
            "Trust": {"oxynixin": 0.4, "cortanyx": -0.2},
            "Disgust": {"cortanyx": 0.3, "oxynixin": -0.3},
            "Anticipation": {"adrenyx": 0.3, "nyxamine": 0.2},
            "Surprise": {"adrenyx": 0.4, "nyxamine": 0.2},
            "Teasing": {"nyxamine": 0.3, "adrenyx": 0.2},
            "Controlling": {"adrenyx": 0.3, "oxynixin": -0.2},
            "Cruel": {"cortanyx": 0.3, "oxynixin": -0.4},
            "Detached": {"cortanyx": 0.2, "oxynixin": -0.3, "seranix": 0.2}
        }
        
        if primary_emotion in emotion_chemical_map:
            for chemical, impact in emotion_chemical_map[primary_emotion].items():
                impacts[chemical] += impact * primary_intensity * significance
        
        # Add impacts from valence and arousal
        if valence > 0.3:  # Positive valence
            impacts["nyxamine"] += valence * 0.2 * significance
            impacts["cortanyx"] -= valence * 0.1 * significance  # Reduce cortanyx
        elif valence < -0.3:  # Negative valence
            impacts["cortanyx"] += abs(valence) * 0.2 * significance
            impacts["nyxamine"] -= abs(valence) * 0.1 * significance  # Reduce nyxamine
        
        if arousal > 0.6:  # High arousal
            impacts["adrenyx"] += arousal * 0.2 * significance
        elif arousal < 0.4:  # Low arousal
            impacts["seranix"] += (1 - arousal) * 0.2 * significance
        
        # Process scenario type impacts
        scenario_chemical_map = {
            "teasing": {"nyxamine": 0.3, "adrenyx": 0.2},
            "dark": {"cortanyx": 0.3, "adrenyx": 0.2},
            "indulgent": {"nyxamine": 0.3, "seranix": 0.2},
            "psychological": {"nyxamine": 0.2, "cortanyx": 0.2},
            "nurturing": {"oxynixin": 0.3, "seranix": 0.2},
            "discipline": {"cortanyx": 0.2, "adrenyx": 0.2},
            "training": {"nyxamine": 0.1, "cortanyx": 0.1},
            "service": {"oxynixin": 0.2, "seranix": 0.1},
            "worship": {"oxynixin": 0.3, "nyxamine": 0.2}
        }
        
        if scenario_type in scenario_chemical_map:
            for chemical, impact in scenario_chemical_map[scenario_type].items():
                impacts[chemical] += impact * significance
        
        # Calculate overall impact strength
        impact_strength = sum(abs(v) for v in impacts.values()) / len(impacts)
        
        # Create impact object
        return NeurochemicalImpact(
            nyxamine_impact=impacts["nyxamine"],
            seranix_impact=impacts["seranix"],
            oxynixin_impact=impacts["oxynixin"],
            cortanyx_impact=impacts["cortanyx"],
            adrenyx_impact=impacts["adrenyx"],
            impact_source="experience",
            impact_strength=impact_strength
        )
    
    @function_tool
    async def _calculate_emotional_impacts(self, ctx: RunContextWrapper,
                                      experience: Dict[str, Any],
                                      neurochemical_impacts: NeurochemicalImpact) -> Dict[str, Dict[str, float]]:
        """
        Calculate impacts on emotional tendencies from an experience
        
        Args:
            experience: Experience data
            neurochemical_impacts: Impacts on neurochemical baselines
            
        Returns:
            Impacts on emotional tendencies
        """
        # Extract relevant information
        emotional_context = experience.get("emotional_context", {})
        primary_emotion = emotional_context.get("primary_emotion", "")
        significance = experience.get("significance", 5) / 10  # Convert to 0.0-1.0 scale
        
        # Initialize impacts
        impacts = {}
        
        if primary_emotion in self.emotional_tendencies:
            # Reinforce emotional tendencies that match the experience
            impacts[primary_emotion] = {
                "likelihood": 0.1 * significance,  # Increase likelihood
                "intensity": 0.05 * significance,  # Slightly increase intensity
                "threshold": -0.05 * significance  # Slightly lower threshold (easier to trigger)
            }
            
            # Also adjust related emotions based on neurochemical impacts
            chemical_emotion_map = {
                "nyxamine": ["Joy", "Anticipation", "Teasing"],
                "seranix": ["Contentment", "Trust"],
                "oxynixin": ["Trust", "Love", "Nurturing"],
                "cortanyx": ["Sadness", "Fear", "Anger", "Disgust", "Cruel", "Detached"],
                "adrenyx": ["Fear", "Surprise", "Anticipation", "Controlling"]
            }
            
            # Process each neurochemical impact
            for chemical, impact_attr in [
                ("nyxamine", neurochemical_impacts.nyxamine_impact),
                ("seranix", neurochemical_impacts.seranix_impact),
                ("oxynixin", neurochemical_impacts.oxynixin_impact),
                ("cortanyx", neurochemical_impacts.cortanyx_impact),
                ("adrenyx", neurochemical_impacts.adrenyx_impact)
            ]:
                if abs(impact_attr) < 0.1:
                    continue  # Skip minimal impacts
                
                # Get related emotions
                related_emotions = chemical_emotion_map.get(chemical, [])
                
                for emotion in related_emotions:
                    if emotion in self.emotional_tendencies and emotion != primary_emotion:
                        # Calculate impact scale (smaller than primary emotion)
                        scale = 0.6 * abs(impact_attr) * significance
                        
                        # Initialize if needed
                        if emotion not in impacts:
                            impacts[emotion] = {
                                "likelihood": 0.0,
                                "intensity": 0.0,
                                "threshold": 0.0
                            }
                        
                        # Apply impact direction
                        if impact_attr > 0:
                            # Positive impact - increase likelihood and intensity, decrease threshold
                            impacts[emotion]["likelihood"] += 0.05 * scale
                            impacts[emotion]["intensity"] += 0.03 * scale
                            impacts[emotion]["threshold"] -= 0.02 * scale
                        else:
                            # Negative impact - decrease likelihood and intensity, increase threshold
                            impacts[emotion]["likelihood"] -= 0.05 * scale
                            impacts[emotion]["intensity"] -= 0.03 * scale
                            impacts[emotion]["threshold"] += 0.02 * scale
        
        return impacts
    
    @function_tool
    async def _update_identity_history(self, ctx: RunContextWrapper,
                                  trait_changes: Dict[str, Dict[str, Any]],
                                  preference_changes: Dict[str, Dict[str, Dict[str, Any]]],
                                  neurochemical_impacts: NeurochemicalImpact,
                                  emotional_impacts: Dict[str, Dict[str, float]],
                                  experience_id: str) -> Dict[str, Any]:
        """
        Update identity history with changes
        
        Args:
            trait_changes: Changes to traits
            preference_changes: Changes to preferences
            neurochemical_impacts: Impacts on neurochemical baselines
            emotional_impacts: Impacts on emotional tendencies
            experience_id: ID of the experience causing changes
            
        Returns:
            Update results
        """
        timestamp = datetime.datetime.now().isoformat()
        significant_changes = {}
        
        # Update traits
        for trait, change_data in trait_changes.items():
            if trait not in self.identity_traits:
                continue
                
            actual_change = change_data.get("actual_change", 0)
            
            # Only record significant changes
            if abs(actual_change) >= self.min_impact_threshold:
                significant_changes[f"trait.{trait}"] = actual_change
        
        # Update preferences
        for category, prefs in preference_changes.items():
            if category not in self.identity_preferences:
                continue
                
            for pref, change_data in prefs.items():
                if pref not in self.identity_preferences[category]:
                    continue
                    
                actual_change = change_data.get("actual_change", 0)
                
                # Only record significant changes
                if abs(actual_change) >= self.min_impact_threshold:
                    significant_changes[f"preference.{category}.{pref}"] = actual_change
        
        # Record neurochemical impacts
        for chemical, impact_attr in [
            ("nyxamine", neurochemical_impacts.nyxamine_impact),
            ("seranix", neurochemical_impacts.seranix_impact),
            ("oxynixin", neurochemical_impacts.oxynixin_impact),
            ("cortanyx", neurochemical_impacts.cortanyx_impact),
            ("adrenyx", neurochemical_impacts.adrenyx_impact)
        ]:
            if abs(impact_attr) >= self.min_impact_threshold:
                significant_changes[f"neurochemical.{chemical}"] = impact_attr
        
        # Record emotional impacts
        for emotion, impacts in emotional_impacts.items():
            for aspect, impact in impacts.items():
                if abs(impact) >= self.min_impact_threshold:
                    significant_changes[f"emotion.{emotion}.{aspect}"] = impact
        
        # Update overall stats
        self.update_count += 1
        self.last_update = timestamp
        
        # Record impact in history
        impact_record = {
            "timestamp": timestamp,
            "experience_id": experience_id,
            "significant_changes": significant_changes,
            "update_count": self.update_count,
            "neurochemical_impacts": neurochemical_impacts.model_dump(),
            "emotional_impacts": emotional_impacts
        }
        
        self.impact_history.append(impact_record)
        
        # Limit history size
        if len(self.impact_history) > self.max_history_size:
            self.impact_history = self.impact_history[-self.max_history_size:]
        
        return {
            "significant_changes": len(significant_changes),
            "update_count": self.update_count,
            "timestamp": timestamp
        }
    
    @function_tool
    async def _get_recent_impacts(self, ctx: RunContextWrapper, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent identity impacts
        
        Args:
            limit: Maximum number of impacts to return
            
        Returns:
            List of recent impacts
        """
        return self.impact_history[-limit:]
    
    @function_tool
    async def _calculate_identity_changes(self, ctx: RunContextWrapper,
                                     time_period: str = "recent") -> Dict[str, Dict[str, float]]:
        """
        Calculate changes in identity over a time period
        
        Args:
            time_period: Time period to consider ('recent', 'all')
            
        Returns:
            Changes in traits and preferences
        """
        changes = {
            "traits": {},
            "preferences": {},
            "neurochemicals": {},
            "emotions": {}
        }
        
        # Get impacts to consider
        if time_period == "recent":
            impacts = self.impact_history[-10:]  # Last 10 impacts
        else:
            impacts = self.impact_history
        
        # No impacts, return empty
        if not impacts:
            return changes
        
        # Aggregate changes
        for impact in impacts:
            for key, value in impact.get("significant_changes", {}).items():
                parts = key.split(".")
                
                if parts[0] == "trait" and len(parts) == 2:
                    trait = parts[1]
                    if trait not in changes["traits"]:
                        changes["traits"][trait] = 0
                    changes["traits"][trait] += value
                    
                elif parts[0] == "preference" and len(parts) == 3:
                    category = parts[1]
                    preference = parts[2]
                    
                    if category not in changes["preferences"]:
                        changes["preferences"][category] = {}
                    
                    if preference not in changes["preferences"][category]:
                        changes["preferences"][category][preference] = 0
                        
                    changes["preferences"][category][preference] += value
                    
                elif parts[0] == "neurochemical" and len(parts) == 2:
                    chemical = parts[1]
                    if chemical not in changes["neurochemicals"]:
                        changes["neurochemicals"][chemical] = 0
                    changes["neurochemicals"][chemical] += value
                    
                elif parts[0] == "emotion" and len(parts) == 3:
                    emotion = parts[1]
                    aspect = parts[2]
                    
                    if emotion not in changes["emotions"]:
                        changes["emotions"][emotion] = {}
                    
                    if aspect not in changes["emotions"][emotion]:
                        changes["emotions"][emotion][aspect] = 0
                        
                    changes["emotions"][emotion][aspect] += value
        
        return changes
    
    @function_tool
    async def _get_neurochemical_patterns(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get patterns in neurochemical baselines over time
        
        Returns:
            Neurochemical pattern analysis
        """
        patterns = {}
        
        # Analyze each chemical
        for chemical, data in self.neurochemical_profile.items():
            # Get history
            history = data.get("evolution_history", [])
            if not history:
                patterns[chemical] = {
                    "pattern": "stable",
                    "current_value": data["value"],
                    "baseline": data["value"],
                    "adaptability": data["adaptability"]
                }
                continue
            
            # Calculate trend
            changes = [entry.get("change", 0) for entry in history]
            total_change = sum(changes)
            avg_change = total_change / len(changes)
            
            # Calculate volatility
            if len(changes) > 1:
                volatility = sum(abs(changes[i] - changes[i-1]) for i in range(1, len(changes))) / (len(changes) - 1)
            else:
                volatility = 0.0
            
            # Determine pattern
            if abs(avg_change) < 0.01:
                if volatility < 0.02:
                    pattern = "stable"
                else:
                    pattern = "fluctuating"
            elif avg_change > 0:
                pattern = "increasing"
            else:
                pattern = "decreasing"
            
            patterns[chemical] = {
                "pattern": pattern,
                "current_value": data["value"],
                "average_change": avg_change,
                "volatility": volatility,
                "adaptability": data["adaptability"]
            }
        
        # Analyze relationships between chemicals
        relationships = {}
        
        # Check for common patterns
        if patterns["nyxamine"].get("pattern") == patterns["oxynixin"].get("pattern"):
            relationships["nyxamine_oxynixin"] = "aligned"
        elif patterns["nyxamine"].get("pattern") != patterns["oxynixin"].get("pattern"):
            relationships["nyxamine_oxynixin"] = "divergent"
        
        if patterns["cortanyx"].get("pattern") == patterns["adrenyx"].get("pattern"):
            relationships["cortanyx_adrenyx"] = "aligned"
        elif patterns["cortanyx"].get("pattern") != patterns["adrenyx"].get("pattern"):
            relationships["cortanyx_adrenyx"] = "divergent"
        
        if patterns["seranix"].get("pattern") == patterns["cortanyx"].get("pattern"):
            relationships["seranix_cortanyx"] = "aligned"
        elif patterns["seranix"].get("pattern") != patterns["cortanyx"].get("pattern"):
            relationships["seranix_cortanyx"] = "divergent"
        
        # Check for opposing trends
        if (patterns["nyxamine"].get("pattern") == "increasing" and 
            patterns["cortanyx"].get("pattern") == "decreasing"):
            relationships["nyxamine_cortanyx"] = "inversely_related"
        elif (patterns["nyxamine"].get("pattern") == "decreasing" and 
              patterns["cortanyx"].get("pattern") == "increasing"):
            relationships["nyxamine_cortanyx"] = "inversely_related"
        
        if (patterns["oxynixin"].get("pattern") == "increasing" and 
            patterns["cortanyx"].get("pattern") == "decreasing"):
            relationships["oxynixin_cortanyx"] = "inversely_related"
        elif (patterns["oxynixin"].get("pattern") == "decreasing" and 
              patterns["cortanyx"].get("pattern") == "increasing"):
            relationships["oxynixin_cortanyx"] = "inversely_related"
        
        return {
            "chemical_patterns": patterns,
            "relationships": relationships
        }
    
    @function_tool
    async def _calculate_trait_consistency(self, ctx: RunContextWrapper) -> Dict[str, float]:
        """
        Calculate consistency scores for traits
        
        Returns:
            Consistency scores for each trait
        """
        consistency = {}
        
        # Calculate for each trait
        for trait, data in self.identity_traits.items():
            history = data.get("evolution_history", [])
            
            if not history:
                consistency[trait] = 1.0  # Perfect consistency if no changes
                continue
            
            # Calculate variance of changes
            changes = [entry.get("change", 0) for entry in history]
            mean_change = sum(changes) / len(changes)
            variance = sum((change - mean_change) ** 2 for change in changes) / len(changes)
            
            # Lower variance = higher consistency
            consistency_score = max(0.0, 1.0 - math.sqrt(variance) * 5)  # Scale for meaningful values
            consistency[trait] = min(1.0, consistency_score)  # Cap at 1.0
        
        return consistency
    
    @function_tool
    async def _calculate_preference_consistency(self, ctx: RunContextWrapper) -> Dict[str, Dict[str, float]]:
        """
        Calculate consistency scores for preferences
        
        Returns:
            Consistency scores for each preference by category
        """
        consistency = {}
        
        # Calculate for each preference category
        for category, preferences in self.identity_preferences.items():
            consistency[category] = {}
            
            # Calculate for each preference
            for pref, data in preferences.items():
                history = data.get("evolution_history", [])
                
                if not history:
                    consistency[category][pref] = 1.0  # Perfect consistency if no changes
                    continue
                
                # Calculate variance of changes
                changes = [entry.get("change", 0) for entry in history if "change" in entry]
                if not changes:
                    consistency[category][pref] = 1.0  # Perfect consistency if no changes
                    continue
                    
                mean_change = sum(changes) / len(changes)
                variance = sum((change - mean_change) ** 2 for change in changes) / len(changes)
                
                # Lower variance = higher consistency
                consistency_score = max(0.0, 1.0 - math.sqrt(variance) * 5)  # Scale for meaningful values
                consistency[category][pref] = min(1.0, consistency_score)  # Cap at 1.0
        
        return consistency
    
    @function_tool
    async def _identify_contradictions(self, ctx: RunContextWrapper) -> List[Dict[str, Any]]:
        """
        Identify potential contradictions in identity
        
        Returns:
            List of potential contradictions
        """
        contradictions = []
        
        # Check for contradictions between traits
        trait_pairs = [
            ("dominance", "patience"),
            ("playfulness", "strictness"),
            ("cruelty", "patience")
        ]
        
        for trait1, trait2 in trait_pairs:
            if (trait1 in self.identity_traits and 
                trait2 in self.identity_traits):
                
                val1 = self.identity_traits[trait1]["value"]
                val2 = self.identity_traits[trait2]["value"]
                
                # Check for extreme values in both contradictory traits
                if val1 > 0.7 and val2 > 0.7:
                    contradictions.append({
                        "type": "trait_contradiction",
                        "elements": [trait1, trait2],
                        "values": [val1, val2],
                        "description": f"Traits {trait1} and {trait2} are both very high despite being potentially contradictory"
                    })
        
        # Check for neurochemical contradictions
        if (self.neurochemical_profile["nyxamine"]["value"] > 0.7 and 
            self.neurochemical_profile["cortanyx"]["value"] > 0.7):
            contradictions.append({
                "type": "neurochemical_contradiction",
                "elements": ["nyxamine", "cortanyx"],
                "values": [self.neurochemical_profile["nyxamine"]["value"], 
                          self.neurochemical_profile["cortanyx"]["value"]],
                "description": "High nyxamine (pleasure) and high cortanyx (stress) is an unusual combination"
            })
        
        if (self.neurochemical_profile["seranix"]["value"] > 0.7 and 
            self.neurochemical_profile["adrenyx"]["value"] > 0.7):
            contradictions.append({
                "type": "neurochemical_contradiction",
                "elements": ["seranix", "adrenyx"],
                "values": [self.neurochemical_profile["seranix"]["value"], 
                          self.neurochemical_profile["adrenyx"]["value"]],
                "description": "High seranix (calm) and high adrenyx (alertness) is an unusual combination"
            })
        
        # Check for contradictions in emotional tendencies
        emotion_pairs = [
            ("Joy", "Sadness"),
            ("Trust", "Disgust"),
            ("Teasing", "Detached")
        ]
        
        for emotion1, emotion2 in emotion_pairs:
            if (emotion1 in self.emotional_tendencies and 
                emotion2 in self.emotional_tendencies):
                
                likelihood1 = self.emotional_tendencies[emotion1]["likelihood"]
                likelihood2 = self.emotional_tendencies[emotion2]["likelihood"]
                
                # Check for high likelihood in opposing emotions
                if likelihood1 > 0.7 and likelihood2 > 0.7:
                    contradictions.append({
                        "type": "emotion_contradiction",
                        "elements": [emotion1, emotion2],
                        "values": [likelihood1, likelihood2],
                        "description": f"Emotions {emotion1} and {emotion2} both have high likelihood despite being opposing"
                    })
        
        # Check for misalignments between traits and preferences
        trait_preference_pairs = [
            ("dominance", "scenario_types", "discipline"),
            ("playfulness", "scenario_types", "teasing"),
            ("strictness", "emotional_tones", "stern"),
            ("creativity", "interaction_styles", "metaphorical"),
            ("cruelty", "emotional_tones", "cruel")
        ]
        
        for trait, category, preference in trait_preference_pairs:
            if (trait in self.identity_traits and 
                category in self.identity_preferences and 
                preference in self.identity_preferences[category]):
                
                trait_val = self.identity_traits[trait]["value"]
                pref_val = self.identity_preferences[category][preference]["value"]
                
                # Check for significant mismatch (high trait, low preference or vice versa)
                if (trait_val > 0.7 and pref_val < 0.3) or (trait_val < 0.3 and pref_val > 0.7):
                    contradictions.append({
                        "type": "trait_preference_mismatch",
                        "elements": [trait, f"{category}.{preference}"],
                        "values": [trait_val, pref_val],
                        "description": f"Trait {trait} and preference {preference} have mismatched values"
                    })
        
        # Check for misalignments between neurochemicals and traits
        chemical_trait_pairs = [
            ("nyxamine", "creativity"),
            ("seranix", "patience"),
            ("oxynixin", "cruelty", True),  # True indicates inverse relationship expected
            ("cortanyx", "strictness"),
            ("adrenyx", "intensity")
        ]
        
        for chemical, trait, inverse_expected in [(c, t, False) for c, t in chemical_trait_pairs] + [(c, t, inv) for c, t, inv in chemical_trait_pairs if len((c, t, inv)) == 3]:
            if (chemical in self.neurochemical_profile and 
                trait in self.identity_traits):
                
                chemical_val = self.neurochemical_profile[chemical]["value"]
                trait_val = self.identity_traits[trait]["value"]
                
                # Check for misalignment based on expected relationship
                if not inverse_expected:
                    # Direct relationship expected (both high or both low)
                    if (chemical_val > 0.7 and trait_val < 0.3) or (chemical_val < 0.3 and trait_val > 0.7):
                        contradictions.append({
                            "type": "neurochemical_trait_mismatch",
                            "elements": [chemical, trait],
                            "values": [chemical_val, trait_val],
                            "description": f"Neurochemical {chemical} and trait {trait} have misaligned values"
                        })
                else:
                    # Inverse relationship expected (one high, one low)
                    if (chemical_val > 0.7 and trait_val > 0.7) or (chemical_val < 0.3 and trait_val < 0.3):
                        contradictions.append({
                            "type": "neurochemical_trait_mismatch",
                            "elements": [chemical, trait],
                            "values": [chemical_val, trait_val],
                            "description": f"Neurochemical {chemical} and trait {trait} have unexpectedly aligned values"
                        })
        
        return contradictions
    
    @function_tool
    async def _assess_neurochemical_coherence(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Assess coherence of neurochemical baselines
        
        Returns:
            Neurochemical coherence assessment
        """
        # Expected relationships between neurochemicals
        expected_relations = [
            # Format: (chemical1, chemical2, relation_type)
            # relation_type: "inverse" means they should be inversely related
            # relation_type: "aligned" means they should be similarly valued
            ("nyxamine", "cortanyx", "inverse"),
            ("seranix", "adrenyx", "inverse"),
            ("oxynixin", "cortanyx", "inverse"),
            ("nyxamine", "oxynixin", "aligned"),
            ("cortanyx", "adrenyx", "aligned")
        ]
        
        coherence_scores = {}
        imbalances = []
        
        # Check expected relationships
        for chemical1, chemical2, relation_type in expected_relations:
            if (chemical1 in self.neurochemical_profile and 
                chemical2 in self.neurochemical_profile):
                
                val1 = self.neurochemical_profile[chemical1]["value"]
                val2 = self.neurochemical_profile[chemical2]["value"]
                
                if relation_type == "inverse":
                    # For inverse relations, difference should be high
                    coherence = abs(val1 - val2)
                    if coherence < 0.3:  # Low difference indicates poor coherence
                        imbalances.append({
                            "type": "unexpected_similarity",
                            "chemicals": [chemical1, chemical2],
                            "values": [val1, val2],
                            "expected_relation": "inverse",
                            "coherence_score": coherence
                        })
                else:  # aligned
                    # For aligned relations, difference should be low
                    coherence = 1.0 - abs(val1 - val2)
                    if coherence < 0.7:  # High difference indicates poor coherence
                        imbalances.append({
                            "type": "unexpected_difference",
                            "chemicals": [chemical1, chemical2],
                            "values": [val1, val2],
                            "expected_relation": "aligned",
                            "coherence_score": coherence
                        })
                
                coherence_scores[f"{chemical1}_{chemical2}"] = coherence
        
        # Check for extreme values
        for chemical, data in self.neurochemical_profile.items():
            if data["value"] > 0.9:
                imbalances.append({
                    "type": "extreme_high",
                    "chemical": chemical,
                    "value": data["value"]
                })
            elif data["value"] < 0.1:
                imbalances.append({
                    "type": "extreme_low",
                    "chemical": chemical,
                    "value": data["value"]
                })
        
        # Calculate overall coherence
        if coherence_scores:
            overall_coherence = sum(coherence_scores.values()) / len(coherence_scores)
        else:
            overall_coherence = 0.7  # Default if no relationships checked
        
        # Adjust based on number of imbalances
        if imbalances:
            overall_coherence -= len(imbalances) * 0.05  # Reduce score based on imbalances
        
        # Ensure score is in valid range
        overall_coherence = max(0.1, min(1.0, overall_coherence))
        
        return {
            "overall_coherence": overall_coherence,
            "relation_scores": coherence_scores,
            "imbalances": imbalances,
            "balanced_chemicals": [c for c in self.neurochemical_profile.keys() if not any(im.get("chemical") == c for im in imbalances)],
            "assessment_time": datetime.datetime.now().isoformat()
        }
    
    # Public methods
    
    async def update_identity_from_experience(self, 
                                         experience: Dict[str, Any], 
                                         impact: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Update identity based on experience impact
        
        Args:
            experience: Experience data
            impact: Optional impact data (will be calculated if not provided)
            
        Returns:
            Update results
        """
        with trace(workflow_name="update_identity", group_id=self.trace_group_id):
            try:
                # Extract experience ID
                experience_id = experience.get("id", "unknown")
                
                # Calculate neurochemical impacts
                neurochemical_impacts = await self._calculate_neurochemical_impacts(
                    RunContextWrapper(context=None),
                    experience=experience
                )
                
                # Apply neurochemical impacts
                chemical_results = {}
                for chemical, impact_attr in [
                    ("nyxamine", neurochemical_impacts.nyxamine_impact),
                    ("seranix", neurochemical_impacts.seranix_impact),
                    ("oxynixin", neurochemical_impacts.oxynixin_impact),
                    ("cortanyx", neurochemical_impacts.cortanyx_impact),
                    ("adrenyx", neurochemical_impacts.adrenyx_impact)
                ]:
                    if abs(impact_attr) >= self.min_impact_threshold:
                        result = await self._update_neurochemical_baseline(
                            RunContextWrapper(context=None),
                            chemical=chemical,
                            impact=impact_attr,
                            reason=f"experience_{experience_id}"
                        )
                        chemical_results[chemical] = result
                
                # Calculate emotional tendency impacts
                emotional_impacts = await self._calculate_emotional_impacts(
                    RunContextWrapper(context=None),
                    experience=experience,
                    neurochemical_impacts=neurochemical_impacts
                )
                
                # Apply emotional tendency impacts
                emotion_results = {}
                for emotion, impacts in emotional_impacts.items():
                    result = await self._update_emotional_tendency(
                        RunContextWrapper(context=None),
                        emotion=emotion,
                        likelihood_change=impacts.get("likelihood", 0.0),
                        intensity_change=impacts.get("intensity", 0.0),
                        threshold_change=impacts.get("threshold", 0.0),
                        reason=f"experience_{experience_id}"
                    )
                    emotion_results[emotion] = result
                
                # Process trait impacts from regular impact data if provided
                trait_results = {}
                if impact and "traits" in impact:
                    for trait, trait_impact in impact["traits"].items():
                        if abs(trait_impact) >= self.min_impact_threshold:
                            # Get neurochemical impacts from trait
                            trait_chemical_impacts = {}
                            if trait in self.identity_traits:
                                trait_chemical_map = self.identity_traits[trait].get("neurochemical_map", {})
                                for chemical, factor in trait_chemical_map.items():
                                    trait_chemical_impacts[chemical] = factor * trait_impact
                            
                            # Apply trait update
                            result = await self._update_trait(
                                RunContextWrapper(context=None),
                                trait=trait,
                                impact=trait_impact,
                                neurochemical_impacts=trait_chemical_impacts
                            )
                            trait_results[trait] = result
                
                # Process preference impacts
                preference_results = {}
                if impact and "preferences" in impact:
                    for category, prefs in impact["preferences"].items():
                        if category not in preference_results:
                            preference_results[category] = {}
                            
                        for pref, pref_impact in prefs.items():
                            if abs(pref_impact) >= self.min_impact_threshold:
                                result = await self._update_preference(
                                    RunContextWrapper(context=None),
                                    category=category,
                                    preference=pref,
                                    impact=pref_impact
                                )
                                preference_results[category][pref] = result
                
                # Update history
                history_result = await self._update_identity_history(
                    RunContextWrapper(context=None),
                    trait_changes=trait_results,
                    preference_changes=preference_results,
                    neurochemical_impacts=neurochemical_impacts,
                    emotional_impacts=emotional_impacts,
                    experience_id=experience_id
                )
                
                # Calculate coherence
                coherence_result = await self._assess_neurochemical_coherence(RunContextWrapper(context=None))
                
                # Update coherence score
                self.coherence_score = coherence_result["overall_coherence"]
                
                # Check if it's time for a reflection
                reflection_result = None
                if self.update_count % self.reflection_interval == 0:
                    reflection_result = await self.generate_identity_reflection()
                
                # Prepare and return update results
                result = {
                    "experience_id": experience_id,
                    "neurochemical_updates": len(chemical_results),
                    "emotion_updates": len(emotion_results),
                    "trait_updates": len(trait_results),
                    "preference_updates": sum(len(prefs) for prefs in preference_results.values()),
                    "significant_changes": history_result.get("significant_changes", 0),
                    "coherence_score": self.coherence_score,
                    "reflection_generated": reflection_result is not None,
                    "update_count": self.update_count,
                    "timestamp": history_result.get("timestamp")
                }
                
                if reflection_result:
                    result["reflection"] = reflection_result.get("reflection_text", "")
                
                return result
                
            except Exception as e:
                logger.error(f"Error updating identity: {e}")
                return {
                    "error": str(e),
                    "experience_id": experience.get("id", "unknown"),
                    "success": False
                }
    
    async def generate_identity_reflection(self) -> Dict[str, Any]:
        """
        Generate a reflection on current identity and recent changes
        
        Returns:
            Reflection data
        """
        with trace(workflow_name="generate_identity_reflection", group_id=self.trace_group_id):
            try:
                # Create agent input
                agent_input = {
                    "role": "user",
                    "content": "Generate a reflection on my current identity and recent changes, considering neurochemical patterns."
                }
                
                # Run the reflection agent
                result = await Runner.run(
                    self.identity_reflection_agent,
                    agent_input
                )
                
                # Get reflection output
                reflection_output = result.final_output_as(IdentityReflection)
                
                # Record in history
                reflection_record = reflection_output.model_dump()
                self.reflection_history.append(reflection_record)
                
                # Limit history size
                if len(self.reflection_history) > self.max_history_size:
                    self.reflection_history = self.reflection_history[-self.max_history_size:]
                
                return reflection_record
                
            except Exception as e:
                logger.error(f"Error generating identity reflection: {e}")
                return {
                    "error": str(e),
                    "reflection_text": "I'm unable to form a clear reflection on my identity at the moment."
                }
    
    async def get_identity_profile(self) -> Dict[str, Any]:
        """
        Get the current identity profile
        
        Returns:
            Current identity profile
        """
        # Return current identity profile
        return {
            "neurochemical_profile": self.neurochemical_profile,
            "emotional_tendencies": self.emotional_tendencies,
            "traits": self.identity_traits,
            "preferences": self.identity_preferences,
            "update_count": self.update_count,
            "last_update": self.last_update,
            "evolution_rate": self.evolution_rate,
            "coherence_score": self.coherence_score
        }
    
    async def get_recent_reflections(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent identity reflections
        
        Args:
            limit: Maximum number of reflections to return
            
        Returns:
            List of recent reflections
        """
        return self.reflection_history[-limit:]
    
    async def calculate_experience_impact(self, experience: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Calculate how an experience should impact identity
        
        Args:
            experience: Experience data
            
        Returns:
            Impact data with trait and preference changes
        """
        # Calculate neurochemical impacts
        neurochemical_impacts = await self._calculate_neurochemical_impacts(
            RunContextWrapper(context=None),
            experience=experience
        )
        
        # Calculate emotional impacts
        emotional_impacts = await self._calculate_emotional_impacts(
            RunContextWrapper(context=None),
            experience=experience,
            neurochemical_impacts=neurochemical_impacts
        )
        
        # Extract relevant data
        scenario_type = experience.get("scenario_type", "general")
        emotional_context = experience.get("emotional_context", {})
        significance = experience.get("significance", 5) / 10  # Convert to 0-1 scale
        
        # Default empty impact
        impact = {
            "traits": {},
            "preferences": {},
            "neurochemicals": {
                "nyxamine": neurochemical_impacts.nyxamine_impact,
                "seranix": neurochemical_impacts.seranix_impact,
                "oxynixin": neurochemical_impacts.oxynixin_impact,
                "cortanyx": neurochemical_impacts.cortanyx_impact,
                "adrenyx": neurochemical_impacts.adrenyx_impact
            },
            "emotional_tendencies": emotional_impacts
        }
        
        # Impact on traits based on scenario type
        # Map scenario types to trait impacts
        scenario_trait_map = {
            "teasing": {"playfulness": 0.1, "creativity": 0.05},
            "discipline": {"strictness": 0.1, "dominance": 0.08},
            "dark": {"intensity": 0.1, "cruelty": 0.08},
            "indulgent": {"patience": 0.1, "creativity": 0.08},
            "psychological": {"creativity": 0.1, "intensity": 0.05},
            "nurturing": {"patience": 0.1, "strictness": -0.05},
            "service": {"patience": 0.08, "dominance": 0.05},
            "worship": {"intensity": 0.05, "dominance": 0.1},
            "punishment": {"strictness": 0.1, "cruelty": 0.05}
        }
        
        # Apply trait impacts based on scenario type
        if scenario_type in scenario_trait_map:
            for trait, base_impact in scenario_trait_map[scenario_type].items():
                impact["traits"][trait] = base_impact * significance
        
        # Impact on scenario preferences based on emotional response
        if scenario_type:
            # Get valence from emotional context
            valence = emotional_context.get("valence", 0)
            
            # Impact depends on emotional valence
            if valence > 0.3:
                # Positive experience with this scenario type
                if "scenario_types" not in impact["preferences"]:
                    impact["preferences"]["scenario_types"] = {}
                impact["preferences"]["scenario_types"][scenario_type] = 0.1 * significance
            elif valence < -0.3:
                # Negative experience with this scenario type
                if "scenario_types" not in impact["preferences"]:
                    impact["preferences"]["scenario_types"] = {}
                impact["preferences"]["scenario_types"][scenario_type] = -0.05 * significance
        
        return impact
    
    async def set_identity_evolution_rate(self, rate: float) -> Dict[str, Any]:
        """
        Set the identity evolution rate
        
        Args:
            rate: New evolution rate (0.0-1.0)
            
        Returns:
            Update result
        """
        old_rate = self.evolution_rate
        self.evolution_rate = max(0.01, min(1.0, rate))
        
        return {
            "old_rate": old_rate,
            "new_rate": self.evolution_rate,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    async def get_identity_state(self) -> Dict[str, Any]:
        """
        Get a complete snapshot of the current identity state
        
        Returns:
            Identity state snapshot
        """
        # Get top neurochemicals
        top_neurochemicals = sorted(
            [(name, data["value"]) for name, data in self.neurochemical_profile.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Get top traits
        top_traits = sorted(
            [(name, data["value"]) for name, data in self.identity_traits.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Get top preferences by category
        top_preferences = {}
        for category, prefs in self.identity_preferences.items():
            top_preferences[category] = sorted(
                [(name, data["value"]) for name, data in prefs.items()],
                key=lambda x: x[1],
                reverse=True
            )[:3]
        
        # Get top emotional tendencies
        top_emotions = sorted(
            [(name, data["likelihood"]) for name, data in self.emotional_tendencies.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Get recent significant changes
        recent_changes = {}
        for impact in self.impact_history[-5:]:
            for key, value in impact.get("significant_changes", {}).items():
                if abs(value) >= 0.1:  # Only include larger changes
                    if key not in recent_changes:
                        recent_changes[key] = 0
                    recent_changes[key] += value
        
        # Get most recent reflection
        latest_reflection = self.reflection_history[-1]["reflection_text"] if self.reflection_history else None
        
        # Get neurochemical coherence
        coherence_result = await self._assess_neurochemical_coherence(RunContextWrapper(context=None))
        
        # Prepare state snapshot
        state = {
            "top_neurochemicals": dict(top_neurochemicals),
            "top_traits": dict(top_traits),
            "top_preferences": {cat: dict(prefs) for cat, prefs in top_preferences.items()},
            "top_emotions": dict(top_emotions),
            "coherence_score": self.coherence_score,
            "neurochemical_coherence": {
                "overall_score": coherence_result["overall_coherence"],
                "imbalances": coherence_result["imbalances"]
            },
            "evolution_rate": self.evolution_rate,
            "update_count": self.update_count,
            "recent_significant_changes": recent_changes,
            "latest_reflection": latest_reflection,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return state
