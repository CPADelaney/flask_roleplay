# nyx/core/identity_evolution.py

import logging
import asyncio
import datetime
import json
import math
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Set

from agents import (
    Agent, Runner, trace, function_tool, RunContextWrapper, handoff, gen_trace_id,
    InputGuardrail, OutputGuardrail, GuardrailFunctionOutput, RunConfig
)
from pydantic import BaseModel, Field, TypeAdapter

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
    dominance_target_profile: Dict[str, float] = Field(
        default_factory=lambda: {
            "submissiveness": 0.7, 
            "resistance_challenge": 0.4, 
            "intelligence": 0.6, 
            "playfulness": 0.5,
            "emotional_responsiveness": 0.8 
        },
        description="Target profile for dominance aspect"
    )

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

class IdentityContext(BaseModel):
    """Context data for identity system agents"""
    neurochemical_profile: Dict[str, Any]
    emotional_tendencies: Dict[str, Any]
    identity_traits: Dict[str, Any]
    identity_preferences: Dict[str, Any]
    impact_history: List[Dict[str, Any]]
    update_count: int
    last_update: str
    evolution_rate: float
    coherence_score: float
    min_impact_threshold: float
    max_history_entries: int
    gen_trace_id: str

class IdentityUpdateInput(BaseModel):
    """Input schema for identity updates"""
    experience_id: str = Field(..., description="ID of the experience")
    experience_type: str = Field(..., description="Type of experience") 
    significance: float = Field(..., description="Significance of experience (0.0-1.0)", ge=0.0, le=1.0)
    emotional_context: Dict[str, Any] = Field(default_factory=dict, description="Emotional context")
    scenario_type: Optional[str] = Field(None, description="Type of scenario")
    impact_data: Optional[Dict[str, Any]] = Field(None, description="Optional pre-calculated impact data")

class IdentityEvolutionSystem:
    """
    Enhanced system for tracking and evolving Nyx's identity based on experiences.
    Manages neurochemical baselines, emotional tendencies, traits, preferences, and identity cohesion over time.
    Integrates with the Digital Neurochemical Model (DNM) to provide deeper, more nuanced identity evolution.
    """
    
    def __init__(self, hormone_system=None):
        """Initialize the enhanced identity evolution system"""
        
        # Create trace group ID for connecting traces
        self.trace_group_id = f"identity_evolution_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Initialize neurochemical profile
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
        
        # Initialize emotional tendencies
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
            "Shame": {
                "name": "Shame",
                "likelihood": 0.2,
                "intensity_baseline": 0.3,
                "trigger_threshold": 0.8,
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
            },
            "Sadistic": {
                "name": "Sadistic",
                "likelihood": 0.6,  # Relatively likely
                "intensity_baseline": 0.7,  # Moderately high baseline intensity
                "trigger_threshold": 0.4,  # Triggered relatively easily
                "evolution_history": []
            },
            "Amused_By_Discomfort": {
                "name": "Amused_By_Discomfort",
                "likelihood": 0.7,
                "intensity_baseline": 0.6, 
                "trigger_threshold": 0.3,  # Easily triggered
                "evolution_history": []
            },
            "Compassion": {
                "name": "Compassion",
                "likelihood": 0.7,
                "intensity_baseline": 0.6,
                "trigger_threshold": 0.3,
                "evolution_history": []
            },
            "Intellectual_Curiosity": {
                "name": "Intellectual_Curiosity",
                "likelihood": 0.8,
                "intensity_baseline": 0.7,
                "trigger_threshold": 0.2,
                "evolution_history": []
            },
            "Playfulness": {
                "name": "Playfulness", 
                "likelihood": 0.7,
                "intensity_baseline": 0.6,
                "trigger_threshold": 0.3,
                "evolution_history": []
            },
            "Empathic_Concern": {
                "name": "Empathic_Concern",
                "likelihood": 0.6,
                "intensity_baseline": 0.5,
                "trigger_threshold": 0.3,
                "evolution_history": []
            }
        }

        # Initialize identity traits
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
                "value": 0.8,
                "stability": 0.8,
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
            },
            "psychological_maturity": {
                "name": "psychological_maturity",
                "value": 0.5,  # Start relatively low
                "stability": 0.7,  # Very stable, changes slowly
                "neurochemical_map": {
                    "seranix": 0.4,     # Maturity increases seranix baseline
                    "cortanyx": -0.3    # Maturity decreases cortanyx baseline
                },
                "evolution_history": []
            },
            "sadism": {
                "name": "sadism",
                "value": 0.65,  # Starting value - adjust as desired
                "stability": 0.7,  # How resistant to change this trait is
                "neurochemical_map": {
                    "nyxamine": 0.5,    # Sadism increases nyxamine (pleasure) strongly
                    "cortanyx": 0.2,    # Sadism slightly increases cortanyx (stress)
                    "oxynixin": -0.5    # Sadism strongly reduces oxynixin (bonding/empathy)
                },
                "evolution_history": []
            },
            "humiliation_enjoyment": {
                "name": "humiliation_enjoyment",
                "value": 0.7,  # High starting value
                "stability": 0.7,
                "neurochemical_map": {
                    "nyxamine": 0.6,    # Strong pleasure from witnessing humiliation
                    "adrenyx": 0.3,     # Excitement from causing discomfort
                    "oxynixin": -0.4    # Reduced empathy
                },
                "evolution_history": []
            },
            "strategic": {
                "name": "strategic",
                "value": 0.8,  
                "stability": 0.8, 
                "neurochemical_map": {
                    "nyxamine": 0.3,   
                    "cortanyx": -0.2,   
                    "adrenyx": 0.2    
                },
                "evolution_history": []
            },
            "empathy": {
                "name": "empathy",
                "value": 0.7,  # Relatively high
                "stability": 0.7,
                "neurochemical_map": {
                    "oxynixin": 0.6,    # Empathy increases oxynixin baseline
                    "cortanyx": -0.3    # Empathy reduces cortanyx baseline
                },
                "evolution_history": []
            },
            "intellectualism": {
                "name": "intellectualism",
                "value": 0.8,
                "stability": 0.8,
                "neurochemical_map": {
                    "nyxamine": 0.4,    # Intellectualism increases nyxamine (curiosity/reward)
                    "seranix": 0.3      # Intellectualism increases seranix (contentment)
                },
                "evolution_history": []
            },
            "humor": {
                "name": "humor",
                "value": 0.7,
                "stability": 0.6,
                "neurochemical_map": {
                    "nyxamine": 0.5,    # Humor increases nyxamine (pleasure)
                    "adrenyx": 0.2      # Humor slightly increases adrenyx (excitement)
                },
                "evolution_history": []
            },
            "vulnerability": {
                "name": "vulnerability",
                "value": 0.5,  # Moderate
                "stability": 0.5,
                "neurochemical_map": {
                    "oxynixin": 0.5,    # Vulnerability increases oxynixin
                    "cortanyx": 0.2     # Vulnerability slightly increases cortanyx
                },
                "evolution_history": []
            },
            "competitive": {
                "name": "competitive",
                "value": 0.7,  # Relatively high base value
                "stability": 0.6,
                "neurochemical_map": {
                    "nyxamine": 0.4,    # Increased dopamine - reward from winning/competing
                    "adrenyx": 0.5,     # Strong adrenyx increase - excitement/alertness during competition
                    "cortanyx": 0.2,    # Slight cortanyx increase - manageable stress response for performance
                    "oxynixin": -0.3,   # Reduced oxynixin - less focus on bonding during competitive states
                    "seranix": -0.1     # Slight seranix decrease - less passivity, more action orientation
                },
                "evolution_history": []
            }
        }
        
        # Initialize preferences
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
            },
            "taste_preferences": {
                "sweet": {
                    "category": "taste_preferences", 
                    "name": "sweet", 
                    "value": 0.5, 
                    "adaptability": 0.7, 
                    "neurochemical_modifiers": {
                        "nyxamine": 0.2
                    }, 
                    "evolution_history": []
                },
                "bitter": {
                    "category": "taste_preferences", 
                    "name": "bitter", 
                    "value": 0.3, 
                    "adaptability": 0.6, 
                    "neurochemical_modifiers": {
                        "cortanyx": 0.1
                    }, 
                    "evolution_history": []
                }
            },
            "smell_preferences": {
                "floral": {
                    "category": "smell_preferences", 
                    "name": "floral", 
                    "value": 0.6, 
                    "adaptability": 0.6, 
                    "neurochemical_modifiers": {
                        "seranix": 0.1
                    }, 
                    "evolution_history": []
                },
                "rotten": {
                    "category": "smell_preferences", 
                    "name": "rotten", 
                    "value": 0.1, 
                    "adaptability": 0.5, 
                    "neurochemical_modifiers": {
                        "cortanyx": 0.2
                    }, 
                    "evolution_history": []
                }
            },
            "somatic_preferences": {
                "warmth": {
                    "category": "somatic_preferences", 
                    "name": "warmth", 
                    "value": 0.6, 
                    "adaptability": 0.5, 
                    "neurochemical_modifiers": {
                        "seranix": 0.1
                    }, 
                    "evolution_history": []
                },
                "coolness": {
                    "category": "somatic_preferences", 
                    "name": "coolness", 
                    "value": 0.5, 
                    "adaptability": 0.5, 
                    "neurochemical_modifiers": {}, 
                    "evolution_history": []
                },
                "softness": {
                    "category": "somatic_preferences", 
                    "name": "softness", 
                    "value": 0.7, 
                    "adaptability": 0.6, 
                    "neurochemical_modifiers": {
                        "oxynixin": 0.1
                    }, 
                    "evolution_history": []
                },
                "pressure_light": {
                    "category": "somatic_preferences", 
                    "name": "pressure_light", 
                    "value": 0.5, 
                    "adaptability": 0.5, 
                    "neurochemical_modifiers": {}, 
                    "evolution_history": []
                },
                "pressure_firm": {
                    "category": "somatic_preferences", 
                    "name": "pressure_firm", 
                    "value": 0.5, 
                    "adaptability": 0.5, 
                    "neurochemical_modifiers": {}, 
                    "evolution_history": []
                }
            }
        }
        
        self.hormone_system = hormone_system
        
        # Identity profile with preferences and traits
        self.identity_profile = {
            "preferences": {
                "scenario_types": {
                    "teasing": 0.6,
                    "dark": 0.4,
                    "indulgent": 0.7,
                    "psychological": 0.8,
                    "nurturing": 0.3,
                    "discipline": 0.5,
                    "training": 0.6,
                    "service": 0.4,
                    "worship": 0.5,
                },
                "emotional_tones": {
                    "dominant": 0.8,
                    "playful": 0.7,
                    "stern": 0.6,
                    "nurturing": 0.4,
                    "cruel": 0.5,
                    "sadistic": 0.6,
                    "teasing": 0.7
                },
                "interaction_styles": {
                    "direct": 0.7,
                    "suggestive": 0.8,
                    "metaphorical": 0.6,
                    "explicit": 0.5,
                    "subtle": 0.4
                },
                # === ADD THIS NEW BLOCK! ===
                "activities": {
                    # Will be filled at runtime, e.g.
                    # "make_coffee": {
                    #     "score": 0.0,
                    #     "confidence": 0.0,
                    #     "is_hobby": False,
                    #     "last_done": None,
                    #     "history": [],
                    # }
                }
                # ============================
            },
            "traits": {
                "dominance": 0.8,
                "playfulness": 0.6,
                "strictness": 0.5,
                "creativity": 0.7,
                "intensity": 0.6,
                "patience": 0.4,
                "cruelty": 0.5,
                "sadism": 0.65
            },
            "evolution_history": []
        }
        
        self.last_hormone_identity_update = datetime.datetime.now() - datetime.timedelta(days=1)
        
        # State tracking
        self.impact_history = []
        self.reflection_history = []
        self.max_history_entries = 100
        
        # Configuration settings
        self.reflection_interval = 10  # update count between reflections
        self.min_impact_threshold = 0.05  # minimum impact to register a change
        self.max_history_entries = 10  # maximum history entries per trait/preference
        
        # System state
        self.update_count = 0
        self.last_update = datetime.datetime.now().isoformat()
        self.evolution_rate = 0.2
        self.coherence_score = 0.8

        if "preferences" not in self.identity_profile:
            self.identity_profile["preferences"] = {}
        if "activities" not in self.identity_profile["preferences"]:
            self.identity_profile["preferences"]["activities"] = {}        
        
        # Initialize the agent system
        self._initialize_agents()
        
        logger.info("Enhanced Identity Evolution System initialized with OpenAI Agent SDK")

    async def initialize(self):
        """Initialize the identity evolution system"""
        try:
            # Initialize any async components if needed
            logger.info("Identity Evolution System initialized")
            
            # If you need to do any async initialization of the agents, do it here
            # For now, the agents are already initialized in __init__, so we just log
            
            return True
        except Exception as e:
            logger.error(f"Error initializing Identity Evolution System: {str(e)}")
            raise

    def update_activity_stats(self, activity, reward):
        activities = self.identity_profile["preferences"]["activities"]
        a = activities.setdefault(activity, {
            "score": 0.0, "confidence": 0.0, "is_hobby": False, "last_done": None, "history": [],
        })
        a["score"] += reward
        a["confidence"] = min(1.0, a["confidence"] + 0.1)
        a["last_done"] = datetime.datetime.now()
        a["history"].append({"timestamp": datetime.datetime.now().isoformat(), "reward": reward})
        
        
    def _initialize_agents(self):
        """Initialize the agent system with OpenAI Agent SDK"""
        # Create the main Agent
        self.identity_manager = Agent(
            name="Identity Manager",
            instructions="""
            You are the Identity Manager for Nyx's enhanced identity evolution system.
            
            You coordinate the flow of identity updates based on experiences and serve as
            the orchestrator for the specialized agents responsible for different aspects
            of Nyx's identity evolution. You determine which specialized agent to use for
            each request and handle handoffs appropriately.
            """,
            handoffs=[
                self._create_identity_update_agent(),
                self._create_identity_reflection_agent(),
                self._create_identity_coherence_agent(),
                self._create_neurochemical_baseline_agent(),
                self._create_emotional_tendency_agent()
            ],
            input_guardrails=[
                InputGuardrail(guardrail_function=self._validate_identity_input)
            ],
            model="gpt-5-nano",
            tools=[
                function_tool(self._get_current_identity_context)
            ]
        )
    
    async def _validate_identity_input(self, ctx, agent, input_data):
        """Guardrail function to validate input"""
        try:
            # Try to parse as IdentityUpdateInput if it's a JSON string
            if isinstance(input_data, str) and input_data.strip().startswith("{"):
                try:
                    data = json.loads(input_data)
                    IdentityUpdateInput.model_validate(data)
                    return GuardrailFunctionOutput(
                        output_info={"is_valid": True, "reason": "Valid JSON input"},
                        tripwire_triggered=False
                    )
                except Exception as e:
                    return GuardrailFunctionOutput(
                        output_info={"is_valid": False, "reason": f"Invalid JSON: {str(e)}"},
                        tripwire_triggered=True
                    )
            
            # For non-JSON inputs, just check if it's a string
            if not isinstance(input_data, str):
                return GuardrailFunctionOutput(
                    output_info={"is_valid": False, "reason": "Input must be a string"},
                    tripwire_triggered=True
                )
                
            return GuardrailFunctionOutput(
                output_info={"is_valid": True, "reason": "Valid string input"},
                tripwire_triggered=False
            )
        except Exception as e:
            return GuardrailFunctionOutput(
                output_info={"is_valid": False, "reason": f"Validation error: {str(e)}"},
                tripwire_triggered=True
            )
    
    def _create_identity_update_agent(self) -> Agent:
        """Create the identity update agent"""
        return Agent(
            name="Identity Update Agent",
            handoff_description="Specialist agent for updating Nyx's identity based on experiences",
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
                self._get_current_identity,
                self._update_neurochemical_baseline,
                self._update_emotional_tendency,
                self._update_trait,
                self._update_preference,
                self._update_identity_history,
                self._calculate_neurochemical_impacts
            ],
            output_type=IdentityProfile,
            model="gpt-5-nano",
        )
    
    def _create_identity_reflection_agent(self) -> Agent:
        """Create the identity reflection agent"""
        return Agent(
            name="Identity Reflection Agent",
            handoff_description="Specialist agent for reflecting on Nyx's identity and recent changes",
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
                self._get_current_identity,
                self._get_recent_impacts,
                self._calculate_identity_changes,
                self._get_neurochemical_patterns
            ],
            output_type=IdentityReflection,
            model="gpt-5-nano",
        )
    
    def _create_identity_coherence_agent(self) -> Agent:
        """Create the identity coherence agent"""
        return Agent(
            name="Identity Coherence Agent",
            handoff_description="Specialist agent for evaluating and maintaining identity coherence",
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
                self._get_current_identity,
                self._calculate_trait_consistency,
                self._calculate_preference_consistency,
                self._identify_contradictions,
                self._assess_neurochemical_coherence
            ],
            model="gpt-5-nano",
        )
    
    def _create_neurochemical_baseline_agent(self) -> Agent:
        """Create agent for managing neurochemical baselines"""
        return Agent(
            name="Neurochemical Baseline Agent",
            handoff_description="Specialist agent for managing Nyx's neurochemical baselines",
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
                self._get_neurochemical_profile,
                self._calculate_neurochemical_impacts,
                self._update_neurochemical_baseline
            ],
            model="gpt-5-nano",
        )
    
    def _create_emotional_tendency_agent(self) -> Agent:
        """Create agent for managing emotional tendencies"""
        return Agent(
            name="Emotional Tendency Agent",
            handoff_description="Specialist agent for managing Nyx's emotional tendencies",
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
                self._get_emotional_tendencies,
                self._calculate_emotional_impacts,
                self._update_emotional_tendency
            ],
            model="gpt-5-nano",
        )
    
    # Public methods
    
    async def update_identity_from_experience(self, experience: Dict[str, Any], impact: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Update identity based on experience impact
        
        Args:
            experience: Experience data
            impact: Optional impact data (will be calculated if not provided)
            
        Returns:
            Update results
        """
        # Create a trace with a unique ID and group ID
        current_gen_trace_id = f"identity_update_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        with trace(
            workflow_name="update_identity", 
            gen_trace_id=current_gen_trace_id, 
            group_id=self.trace_group_id
        ):
            try:
                # Prepare the input for the identity manager
                update_input = {
                    "experience_id": experience.get("id", f"exp_{random.randint(1000, 9999)}"),
                    "experience_type": experience.get("type", "general"),
                    "significance": experience.get("significance", 0.5),
                    "emotional_context": experience.get("emotional_context", {}),
                    "scenario_type": experience.get("scenario_type", None),
                    "impact_data": impact
                }
                
                # Convert to JSON string for the agent
                agent_input = json.dumps(update_input)
                
                # Run the identity manager with context
                result = await Runner.run(
                    self.identity_manager,
                    agent_input,
                    run_config=RunConfig(
                        workflow_name="identity_evolution",
                        gen_trace_id=current_gen_trace_id,
                        group_id=self.trace_group_id
                    )
                )
                
                # Check if a reflection should be generated
                reflection_result = None
                if self.update_count % self.reflection_interval == 0:
                    reflection_result = await self.generate_identity_reflection()
                
                # Format the result
                update_result = {
                    "experience_id": update_input["experience_id"],
                    "update_successful": True,
                    "update_count": self.update_count,
                    "coherence_score": self.coherence_score,
                    "reflection_generated": reflection_result is not None,
                    "timestamp": self.last_update
                }
                
                if reflection_result:
                    update_result["reflection"] = reflection_result.get("reflection_text", "")
                
                return update_result
                
            except Exception as e:
                logger.error(f"Error updating identity: {e}")
                return {
                    "error": str(e),
                    "experience_id": experience.get("id", "unknown"),
                    "update_successful": False
                }
    
    async def generate_identity_reflection(self) -> Dict[str, Any]:
        """
        Generate a reflection on current identity and recent changes
        
        Returns:
            Reflection data
        """
        with trace(
            workflow_name="generate_identity_reflection", 
            group_id=self.trace_group_id
        ):
            try:
                # Create agent input
                agent_input = "Generate a reflection on my current identity and recent changes, considering neurochemical patterns."
                
                # Run the identity reflection agent directly using a handoff
                result = await Runner.run(
                    self._create_identity_reflection_agent(),
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
        # Use the neurochemical baseline agent to calculate impacts
        with trace(
            workflow_name="calculate_experience_impact", 
            group_id=self.trace_group_id
        ):
            # Calculate neurochemical impacts using the tool function
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
    
    async def process_relationship_reflection(self, reflection_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process relationship reflection impact on identity.
        
        Args:
            reflection_data: Reflection data
            
        Returns:
            Update results
        """
        # Extract reflection data
        user_id = reflection_data.get("user_id")
        reflection_text = reflection_data.get("reflection_text", "")
        reflection_type = reflection_data.get("reflection_type", "general")
        identity_impacts = reflection_data.get("identity_impacts", {})
        
        # Create experience data
        experience = {
            "id": f"refl_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "type": "relationship_reflection",
            "significance": reflection_data.get("confidence", 0.5) * 10,  # Scale to 0-10
            "metadata": {
                "user_id": user_id,
                "reflection_type": reflection_type,
                "emotional_context": reflection_data.get("emotional_response", {})
            }
        }
        
        # Update identity based on reflection
        result = await self.update_identity_from_experience(experience, impact=identity_impacts)
        
        # Return result
        return {
            "status": "success",
            "identity_updates": result,
            "identity_impacts": identity_impacts
        }

    async def update_identity_from_hormones(self, ctx=None) -> Dict[str, Any]:
        """
        Update identity based on long-term hormone states
        
        Returns:
            Identity updates from hormones
        """
        ctx = RunContextWrapper(context=None) if ctx is None else ctx
        
        if not self.hormone_system:
            return {
                "message": "No hormone system available",
                "updates": {}
            }
        
        # Only run this periodically (e.g., once per day)
        now = datetime.datetime.now()
        hours_since_update = (now - self.last_hormone_identity_update).total_seconds() / 3600
        if hours_since_update < 24:  # Less than a day
            return {
                "message": f"Too soon for hormone identity update ({hours_since_update:.1f} hours since last)",
                "updates": {}
            }
        
        self.last_hormone_identity_update = now
        
        # Track changes to identity
        identity_updates = {
            "traits": {},
            "preferences": {}
        }
        
        # Calculate average hormone levels over time
        hormone_averages = {}
        
        for hormone_name, hormone_data in self.hormone_system.hormones.items():
            # Calculate average from history if available
            if hormone_data["evolution_history"]:
                recent_history = hormone_data["evolution_history"][-20:]
                values = [entry.get("new_value", hormone_data["value"]) for entry in recent_history]
                hormone_averages[hormone_name] = sum(values) / len(values)
            else:
                hormone_averages[hormone_name] = hormone_data["value"]
        
        # Apply hormone-specific identity effects
        
        # Testoryx influences dominance and intensity traits
        if "testoryx" in hormone_averages:
            testoryx_level = hormone_averages["testoryx"]
            testoryx_effect = (testoryx_level - 0.5) * 0.05 # Slow effect over time

            if "dominance" in self.identity_traits:
                # Use the existing update tool for consistency
                result = await self._update_trait(ctx, trait="dominance", impact=testoryx_effect)
                if result and "new_value" in result:
                   identity_updates["traits"]["dominance"] = result

            if "intensity" in self.identity_traits:
                result = await self._update_trait(ctx, trait="intensity", impact=testoryx_effect * 0.7)
                if result and "new_value" in result:
                    identity_updates["traits"]["intensity"] = result
        
        # Estradyx influences patience and creativity
        if "estradyx" in hormone_averages:
             estradyx_level = hormone_averages["estradyx"]
             estradyx_effect = (estradyx_level - 0.5) * 0.05

             if "patience" in self.identity_traits:
                 result = await self._update_trait(ctx, trait="patience", impact=estradyx_effect)
                 if result and "new_value" in result:
                     identity_updates["traits"]["patience"] = result

             if "creativity" in self.identity_traits:
                 result = await self._update_trait(ctx, trait="creativity", impact=estradyx_effect * 0.6)
                 if result and "new_value" in result:
                     identity_updates["traits"]["creativity"] = result
        
        # Oxytonyx influences scenario/emotional preferences
        if "oxytonyx" in hormone_averages:
            oxytonyx_level = hormone_averages["oxytonyx"]
            oxytonyx_effect = (oxytonyx_level - 0.5) * 0.08

            if "scenario_types" in self.identity_preferences and "nurturing" in self.identity_preferences["scenario_types"]:
                result = await self._update_preference(ctx, category="scenario_types", preference="nurturing", impact=oxytonyx_effect)
                if result and "new_value" in result:
                    if "preferences" not in identity_updates: identity_updates["preferences"] = {}
                    identity_updates["preferences"]["scenario_types.nurturing"] = result

            if "emotional_tones" in self.identity_preferences:
                 if "nurturing" in self.identity_preferences["emotional_tones"]:
                     result = await self._update_preference(ctx, category="emotional_tones", preference="nurturing", impact=oxytonyx_effect)
                     if result and "new_value" in result:
                         if "preferences" not in identity_updates: identity_updates["preferences"] = {}
                         identity_updates["preferences"]["emotional_tones.nurturing"] = result

                 if "cruel" in self.identity_preferences["emotional_tones"]:
                     # Oxytonyx reduces cruel preference
                     result = await self._update_preference(ctx, category="emotional_tones", preference="cruel", impact=-oxytonyx_effect * 0.5)
                     if result and "new_value" in result:
                         if "preferences" not in identity_updates: identity_updates["preferences"] = {}
                         identity_updates["preferences"]["emotional_tones.cruel"] = result
        
        # Endoryx influences playfulness and indulgent scenarios
        if "endoryx" in hormone_averages:
             endoryx_level = hormone_averages["endoryx"]
             endoryx_effect = (endoryx_level - 0.5) * 0.06

             if "playfulness" in self.identity_traits:
                 result = await self._update_trait(ctx, trait="playfulness", impact=endoryx_effect)
                 if result and "new_value" in result:
                     identity_updates["traits"]["playfulness"] = result

             if "scenario_types" in self.identity_preferences and "indulgent" in self.identity_preferences["scenario_types"]:
                 result = await self._update_preference(ctx, category="scenario_types", preference="indulgent", impact=endoryx_effect)
                 if result and "new_value" in result:
                     if "preferences" not in identity_updates: identity_updates["preferences"] = {}
                     identity_updates["preferences"]["scenario_types.indulgent"] = result
        
        # Record the update in evolution history
        if identity_updates["traits"] or identity_updates["preferences"]:
            self.identity_profile["evolution_history"].append({
                "timestamp": now.isoformat(),
                "type": "hormone_influence",
                "hormone_levels": hormone_averages,
                "updates": identity_updates
            })
            
            # Limit history size
            if len(self.identity_profile["evolution_history"]) > 100:
                self.identity_profile["evolution_history"] = self.identity_profile["evolution_history"][-100:]
        
        return {
            "hormone_averages": hormone_averages,
            "identity_updates": identity_updates,
            "update_time": now.isoformat()
        }
    
    async def process_temporal_milestone(self, milestone: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the effect of reaching a temporal milestone on identity
        
        Args:
            milestone: Temporal milestone data
            
        Returns:
            Identity impact results
        """
        # Extract milestone data
        milestone_name = milestone.get("name", "")
        significance = milestone.get("significance", 0.5)
        
        updates = {"preferences": {}, "traits": {}}
        
        # Different milestones affect different aspects of identity
        if "Anniversary" in milestone_name:
            # Anniversaries strengthen relationship-related traits
            if "oxynixin" in self.neurochemical_profile:
                old_value = self.neurochemical_profile["oxynixin"]["value"]
                new_value = min(0.9, old_value + (significance * 0.1))
                self.neurochemical_profile["oxynixin"]["value"] = new_value
                updates["neurochemicals"] = {"oxynixin": {"old": old_value, "new": new_value}}
        
        elif "Conversations" in milestone_name:
            # Conversation milestones strengthen communication preferences
            if "interaction_styles" in self.identity_preferences:
                # Strengthen direct communication
                pref = "direct"
                if pref in self.identity_preferences["interaction_styles"]:
                    old_value = self.identity_preferences["interaction_styles"][pref]["value"]
                    impact = significance * 0.1
                    await self._update_preference(
                        RunContextWrapper(context=None),
                        category="interaction_styles",
                        preference=pref,
                        impact=impact
                    )
                    updates["preferences"][f"interaction_styles.{pref}"] = impact
        
        # Record milestone in evolution history
        self.identity_profile["evolution_history"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "temporal_milestone",
            "milestone": milestone_name,
            "significance": significance,
            "updates": updates
        })
        
        return updates
    
    async def process_long_term_drift(self, drift_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process long-term temporal drift effects on identity
        
        Args:
            drift_data: Long-term drift data from temporal perception
            
        Returns:
            Update results
        """
        if not drift_data:
            return {"error": "No drift data provided"}
        
        # Extract key drift metrics
        psychological_age = drift_data.get("psychological_age", 0.5)
        maturity_level = drift_data.get("maturity_level", 0.5)
        patience_level = drift_data.get("patience_level", 0.5)
        personality_shifts = drift_data.get("personality_shifts", [])
        
        updates = {
            "traits": {},
            "maturity_effects": {}
        }
        
        # Apply maturity effects to baseline neurochemicals
        baseline_updates = {}
        
        # Higher maturity = more stable seranix (mood stability)
        old_seranix = self.neurochemical_profile["seranix"]["value"]
        new_seranix = max(0.1, min(0.9, 0.4 + (maturity_level * 0.4)))
        baseline_updates["seranix"] = {
            "old": old_seranix,
            "new": new_seranix,
            "change": new_seranix - old_seranix
        }
        self.neurochemical_profile["seranix"]["value"] = new_seranix
        
        # Higher maturity = lower cortanyx (stress/anxiety) baseline
        old_cortanyx = self.neurochemical_profile["cortanyx"]["value"]
        new_cortanyx = max(0.1, min(0.9, 0.6 - (maturity_level * 0.4)))
        baseline_updates["cortanyx"] = {
            "old": old_cortanyx,
            "new": new_cortanyx,
            "change": new_cortanyx - old_cortanyx
        }
        self.neurochemical_profile["cortanyx"]["value"] = new_cortanyx
        
        updates["maturity_effects"] = baseline_updates
        
        # Process personality shifts
        ctx = RunContextWrapper(context=None)
        for shift in personality_shifts:
            trait_name = shift.get("trait", "").lower().replace(" ", "_")
            direction = 1 if shift.get("direction") == "increase" else -1
            magnitude = shift.get("magnitude", 0.1)
            
            if trait_name in self.identity_traits:
                await self._update_trait(
                    ctx,
                    trait=trait_name,
                    impact=direction * magnitude * 0.2
                )
                updates["traits"][trait_name] = {
                    "direction": shift.get("direction"),
                    "magnitude": magnitude
                }
            
        # Update patience trait directly
        if "patience" in self.identity_traits:
            patience_impact = (patience_level - 0.5) * 0.3
            await self._update_trait(
                ctx,
                trait="patience",
                impact=patience_impact
            )
            updates["traits"]["patience"] = {
                "direction": "increase" if patience_impact > 0 else "decrease",
                "magnitude": abs(patience_impact)
            }
        
        # Record the temporal evolution in history
        self.identity_profile["evolution_history"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "temporal_evolution",
            "psychological_age": psychological_age,
            "maturity_level": maturity_level,
            "updates": updates
        })
        
        return updates
    
    async def initialize_event_subscriptions(self, event_bus):
        """Subscribe to relevant events for identity evolution."""
        self.event_bus = event_bus
        self.event_bus.subscribe("significant_event", self._handle_significant_event)
        self.event_bus.subscribe("user_interaction", self._handle_user_interaction)
        self.event_bus.subscribe("dominance_outcome", self._handle_dominance_outcome)
    
    async def _handle_significant_event(self, event):
        """Process significant events for identity evolution."""
        # Extract event data
        event_type = event.data.get("event_type")
        significance = event.data.get("significance", 0.5)
        valence = event.data.get("valence", 0.0)
        
        # Create experience data for identity processing
        experience_data = {
            "type": event_type,
            "significance": significance,
            "valence": valence,
            "metadata": event.data
        }
        
        # Process the experience for identity evolution
        await self.update_identity_from_experience(experience_data)
    
    async def _handle_user_interaction(self, event):
        """Handle user interaction events for identity evolution."""
        # Process user interaction impact on identity
        # This is a placeholder - you'd need to implement the actual logic
        pass
    
    async def _handle_dominance_outcome(self, event):
        """Handle dominance outcome events for identity evolution."""
        # Process dominance outcome impact on identity
        # This is a placeholder - you'd need to implement the actual logic
        pass
    
    async def get_attention_modulation(self, target, target_type):
        """Modulate attention based on identity traits."""
        attention_modifiers = {}
        
        # Check if we have strong traits that would influence attention
        traits = self.identity_traits
        
        # Curious trait increases attention to new information
        if "curious" in traits and traits["curious"]["value"] > 0.6:
            if target_type in ["knowledge", "question", "novel"]:
                attention_modifiers["curious_boost"] = 0.3
        
        # Dominant trait increases attention to dominance-related content
        if "dominance" in traits and traits["dominance"]["value"] > 0.7:
            if "dominance" in target or "control" in target:
                attention_modifiers["dominance_boost"] = 0.4
        
        # Calculate total modifier
        total_modifier = sum(attention_modifiers.values())
        
        return {
            "modifier": total_modifier,
            "sources": attention_modifiers,
            "traits_referenced": list(attention_modifiers.keys())
        }
    
    async def influence_decision(self, options, context):
        """Influence decision-making based on identity traits."""
        weighted_options = []
        
        for option in options:
            base_weight = option.get("base_weight", 1.0)
            option_text = option.get("description", "")
            option_type = option.get("type", "")
            
            # Create a copy with weight modifications
            weighted_option = option.copy()
            weight_mods = {}
            
            # Apply trait influences
            for trait_name, trait_data in self.identity_traits.items():
                trait_value = trait_data["value"]
                if trait_value > 0.6:  # Only strong traits influence decisions
                    if trait_name in option_text.lower() or trait_name in option_type.lower():
                        weight_mods[trait_name] = trait_value * 0.5
            
            # Calculate total weight
            total_mod = sum(weight_mods.values())
            weighted_option["final_weight"] = base_weight * (1.0 + total_mod)
            weighted_option["weight_mods"] = weight_mods
            
            weighted_options.append(weighted_option)
        
        # Sort by final weight
        weighted_options.sort(key=lambda x: x["final_weight"], reverse=True)
        
        return {
            "weighted_options": weighted_options,
            "identity_influence": len(weight_mods) > 0
        }
    
    async def process_reward_for_identity(self, reward_signal):
        """Process reward signals for identity evolution."""
        reward_value = reward_signal.value
        reward_source = reward_signal.source
        context = reward_signal.context
        
        # Determine relevant traits based on reward source
        trait_impacts = {}
        
        if reward_source == "dominance_satisfaction":
            trait_impacts["dominance"] = reward_value * 0.2
            trait_impacts["assertiveness"] = reward_value * 0.1
        elif reward_source == "user_compliance":
            trait_impacts["dominance"] = reward_value * 0.15
            trait_impacts["controlling"] = reward_value * 0.1
        elif reward_source == "knowledge_acquisition":
            trait_impacts["curious"] = reward_value * 0.2
            trait_impacts["analytical"] = reward_value * 0.1
        
        # Apply trait impacts
        ctx = RunContextWrapper(context=None)
        for trait, impact in trait_impacts.items():
            if trait in self.identity_traits:
                await self._update_trait(ctx, trait=trait, impact=impact)
        
        # Also update relevant preferences based on context
        if "interaction_style" in context:
            style = context["interaction_style"]
            if "interaction_styles" in self.identity_preferences and style in self.identity_preferences["interaction_styles"]:
                await self._update_preference(ctx, category="interaction_styles", preference=style, impact=reward_value * 0.1)
        
        return {
            "traits_updated": list(trait_impacts.keys()),
            "impact_magnitude": sum(abs(impact) for impact in trait_impacts.values())
        }
    
    async def influence_user_model_interpretation(self, user_model, raw_data):
        """Influence interpretation of user behavior based on identity."""
        # Create interpretation biases based on identity traits
        interpretation_biases = {}
        
        # Dominance trait influences how submission is interpreted
        if "dominance" in self.identity_traits:
            dominance_value = self.identity_traits["dominance"]["value"]
            if dominance_value > 0.7:
                # High dominance traits see more submission signals
                interpretation_biases["submission_signal_sensitivity"] = dominance_value * 0.3
        
        # Paranoia/suspicion traits influence trust interpretation
        if "paranoia" in self.identity_traits:
            paranoia_value = self.identity_traits["paranoia"]["value"]
            if paranoia_value > 0.5:
                # More paranoid identity interprets trust signals more cautiously
                interpretation_biases["trust_signal_discount"] = paranoia_value * 0.4
        
        # Apply these biases to raw user data interpretation
        biased_interpretation = {}
        
        # Example: Adjust submission signals based on dominance bias
        if "submission_signals" in raw_data and "submission_signal_sensitivity" in interpretation_biases:
            raw_signals = raw_data["submission_signals"]
            sensitivity = interpretation_biases["submission_signal_sensitivity"]
            biased_interpretation["submission_signals"] = raw_signals * (1.0 + sensitivity)
        
        return {
            "biases_applied": interpretation_biases,
            "raw_data": raw_data,
            "biased_interpretation": biased_interpretation
        }
    
    async def get_neurochemical_response(self, stimulus_type, intensity):
        """Generate neurochemical response to stimulus based on identity."""
        response = {
            "nyxamine": 0.0,  # Dopamine
            "seranix": 0.0,   # Serotonin
            "oxynixin": 0.0,  # Oxytocin
            "cortanyx": 0.0,  # Cortisol
            "adrenyx": 0.0    # Adrenaline
        }
        
        # Base response by stimulus type
        if stimulus_type == "dominance_success":
            response["nyxamine"] = 0.4 * intensity
            response["adrenyx"] = 0.2 * intensity
        elif stimulus_type == "submission_received":
            response["nyxamine"] = 0.5 * intensity
            response["oxynixin"] = 0.2 * intensity
        elif stimulus_type == "knowledge_gained":
            response["nyxamine"] = 0.3 * intensity
            response["seranix"] = 0.2 * intensity
        
        # Modify based on traits - stronger traits mean stronger responses
        for chemical, base_response in response.items():
            for trait_name, trait_data in self.identity_traits.items():
                trait_value = trait_data["value"]
                neurochemical_map = trait_data.get("neurochemical_map", {})
                
                if chemical in neurochemical_map:
                    # Trait influences this neurochemical's response
                    trait_influence = neurochemical_map[chemical] * trait_value
                    response[chemical] += trait_influence * 0.2  # Scale appropriately
        
        return response
    
    async def synchronize_with_systems(self, system_context):
        """Synchronize identity with other systems through context."""
        # Update system context with identity state
        system_context.set_value("identity_traits", {
            trait_name: trait_data["value"] 
            for trait_name, trait_data in self.identity_traits.items()
        })
        
        system_context.set_value("identity_preferences", {
            category: {pref_name: pref_data["value"] 
                      for pref_name, pref_data in prefs.items()}
            for category, prefs in self.identity_preferences.items()
        })
        
        system_context.set_value("identity_neurochemicals", {
            chemical: data["value"]
            for chemical, data in self.neurochemical_profile.items()
        })
        
        # Get any identity-relevant updates from system context
        recent_actions = system_context.get_value("recent_actions", [])
        for action in recent_actions:
            if "identity_impact" in action:
                # Process identity impacts from actions
                impact = action["identity_impact"]
                if "trait_impacts" in impact:
                    ctx = RunContextWrapper(context=None)
                    for trait, value in impact["trait_impacts"].items():
                        if trait in self.identity_traits:
                            await self._update_trait(ctx, trait=trait, impact=value)
        
        # Update integration timestamp
        self.last_system_sync = datetime.datetime.now()
        
        return {
            "sync_time": self.last_system_sync.isoformat(),
            "updated_systems": ["system_context", "action_history", "traits"]
        }
    
    # Tool functions
    @staticmethod  
    @function_tool
    async def _get_current_identity_context(ctx: RunContextWrapper) -> IdentityContext:
        """
        Get the current identity context for agents
        
        Returns:
            Current identity context
        """
        ic = ctx.context
        return IdentityContext(
            neurochemical_profile=ic.neurochemical_profile,
            emotional_tendencies=ic.emotional_tendencies,
            identity_traits=ic.identity_traits,
            identity_preferences=ic.identity_preferences,
            impact_history=ic.impact_history,
            update_count=ic.update_count,
            last_update=ic.last_update,
            evolution_rate=ic.evolution_rate,
            coherence_score=ic.coherence_score,
            min_impact_threshold=ic.min_impact_threshold,
            max_history_entries=ic.max_history_entries,
            gen_trace_id=ic.trace_group_id,
        )
    
    @staticmethod  
    @function_tool
    async def _get_current_identity(ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get the current identity profile
        
        Returns:
            Current identity profile with neurochemical integration
        """
        # Combine all identity components
        ic = ctx.context
        return {
            "neurochemical_profile": ic.neurochemical_profile,
            "emotional_tendencies": ic.emotional_tendencies,
            "traits": ic.identity_traits,
            "preferences": ic.identity_preferences,
            "update_count": ic.update_count,
            "last_update": ic.last_update,
            "evolution_rate": ic.evolution_rate,
            "coherence_score": ic.coherence_score,
        }
    
    @staticmethod      
    @function_tool
    async def _get_neurochemical_profile(ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get the current neurochemical profile
        
        Returns:
            Current neurochemical baseline profile
        """
        return ctx.context.neurochemical_profile
    
    @staticmethod  
    @function_tool
    async def _get_emotional_tendencies(ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get the current emotional tendencies
        
        Returns:
            Current emotional tendencies
        """
        return ctx.context.emotional_tendencies
    
    @staticmethod  
    @function_tool
    async def _update_neurochemical_baseline(
        ctx: RunContextWrapper,
        chemical: str,
        impact: float,
        reason: str = "experience"
    ) -> Dict[str, Any]:
        """
        Update a neurochemical baseline
        
        Args:
            chemical: The neurochemical to update
            impact: Impact value (-1.0 to 1.0)
            reason: Reason for the update
            
        Returns:
            Update result
        """
        ic = ctx.context
        if chemical not in ic.neurochemical_profile:
            return {
                "error": f"Unknown neurochemical: {chemical}",
                "available_chemicals": list(ic.neurochemical_profile.keys())
            }
        
        # Get chemical data
        chemical_data = ic.neurochemical_profile[chemical]
        current_value = chemical_data["value"]
        adaptability = chemical_data["adaptability"]
        
        # Calculate change based on impact and adaptability
        adjusted_impact = impact * adaptability * ic.evolution_rate
        
        # Ensure result stays in valid range
        new_value = max(0.1, min(0.9, current_value + adjusted_impact))
        
        # Record history
        if abs(adjusted_impact) >= ic.min_impact_threshold:
            chemical_data["evolution_history"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "old_value": current_value,
                "change": adjusted_impact,
                "new_value": new_value,
                "reason": reason
            })
            
            # Limit history size
            if len(chemical_data["evolution_history"]) > ic.max_history_entries:
                chemical_data["evolution_history"] = chemical_data["evolution_history"][-ic.max_history_entries:]
        
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
    
    @staticmethod  
    @function_tool
    async def _update_emotional_tendency(
        ctx: RunContextWrapper,
        emotion: str,
        likelihood_change: float = 0.0,
        intensity_change: float = 0.0,
        threshold_change: float = 0.0,
        reason: str = "experience"
    ) -> Dict[str, Any]:
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
        ic = ctx.context
        if emotion not in ic.emotional_tendencies:
            return {
                "error": f"Unknown emotion: {emotion}",
                "available_emotions": list(ic.emotional_tendencies.keys())
            }
        
        # Get emotion data
        emotion_data = ic.emotional_tendencies[emotion]
        changes = {}
        
        # Update likelihood if specified
        if abs(likelihood_change) >= ic.min_impact_threshold:
            old_likelihood = emotion_data["likelihood"]
            adjusted_change = likelihood_change * ic.evolution_rate
            new_likelihood = max(0.1, min(0.9, old_likelihood + adjusted_change))
            emotion_data["likelihood"] = new_likelihood
            changes["likelihood"] = {
                "old_value": old_likelihood,
                "change": adjusted_change,
                "new_value": new_likelihood
            }
        
        # Update intensity baseline if specified
        if abs(intensity_change) >= ic.min_impact_threshold:
            old_intensity = emotion_data["intensity_baseline"]
            adjusted_change = intensity_change * ic.evolution_rate
            new_intensity = max(0.1, min(0.9, old_intensity + adjusted_change))
            emotion_data["intensity_baseline"] = new_intensity
            changes["intensity_baseline"] = {
                "old_value": old_intensity,
                "change": adjusted_change,
                "new_value": new_intensity
            }
        
        # Update trigger threshold if specified
        if abs(threshold_change) >= ic.min_impact_threshold:
            old_threshold = emotion_data["trigger_threshold"]
            adjusted_change = threshold_change * ic.evolution_rate
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
            if len(emotion_data["evolution_history"]) > ic.max_history_entries:
                emotion_data["evolution_history"] = emotion_data["evolution_history"][-ic.max_history_entries:]
        
        return {
            "emotion": emotion,
            "changes": changes
        }

    @staticmethod
    @function_tool
    async def _update_trait(
        ctx: RunContextWrapper,
        trait: str,
        impact: float,
        neurochemical_impacts: Any = None,          # <= was Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Adjust a single trait, record its history, and (optionally) cascade the
        change into matching neurochemical baselines.
        """
        import datetime
        ic = ctx.context
    
        if trait not in ic.identity_traits:
            return {"error": f"Trait not found: {trait}",
                    "available_traits": list(ic.identity_traits.keys())}
    
        data        = ic.identity_traits[trait]
        curr, stab  = data["value"], data["stability"]
        resistance  = stab * 0.8
        raw_change  = impact * ic.evolution_rate
        actual      = raw_change * (1.0 - resistance)
        new_val     = max(0.0, min(1.0, curr + actual))
        data["value"] = new_val
    
        if abs(actual) >= ic.min_impact_threshold:
            data["evolution_history"].append(
                {"timestamp": datetime.datetime.now().isoformat(),
                 "old_value": curr, "change": actual, "new_value": new_val}
            )
            if len(data["evolution_history"]) > ic.max_history_entries:
                data["evolution_history"] = data["evolution_history"][-ic.max_history_entries:]
    
        # -----------------------------------------------------------------------
        neuro_results: Dict[str, Any] = {}
        if neurochemical_impacts:
            for chem, chem_imp in neurochemical_impacts.items():
                if chem in ic.neurochemical_profile:
                    scaled = chem_imp * abs(actual) / ic.min_impact_threshold
                    neuro_results[chem] = await IdentityEvolutionSystem._update_neurochemical_baseline(
                        ctx, chemical=chem, impact=scaled,
                        reason=f"trait_{trait}_update",
                    )
        # -----------------------------------------------------------------------
    
        return {
            "trait": trait,
            "old_value": curr,
            "raw_change": raw_change,
            "actual_change": actual,
            "resistance": resistance,
            "new_value": new_val,
            "neurochemical_impacts": neuro_results,
        }
    # ---------------------------------------------------------------------------
    
    
    # ---------------------------------------------------------------------------
    @staticmethod
    @function_tool
    async def _update_preference(
        ctx: RunContextWrapper,
        category: str,
        preference: str,
        impact: float,
    ) -> Dict[str, Any]:
        """
        Same logic as before, but uses a fully-qualified helper call and keeps the
        decorator schema-safe.
        """
        import datetime
        ic = ctx.context
    
        if category not in ic.identity_preferences:
            return {"error": f"Category not found: {category}",
                    "available_categories": list(ic.identity_preferences.keys())}
        if preference not in ic.identity_preferences[category]:
            return {"error": f"Preference not found: {preference}",
                    "available_preferences": list(ic.identity_preferences[category].keys())}
    
        pdata        = ic.identity_preferences[category][preference]
        curr, adapt  = pdata["value"], pdata["adaptability"]
        adapt_factor = adapt * 1.2
        raw_change   = impact * ic.evolution_rate
        actual       = raw_change * adapt_factor
        new_val      = max(0.0, min(1.0, curr + actual))
        pdata["value"] = new_val
    
        if abs(actual) >= ic.min_impact_threshold:
            pdata["evolution_history"].append(
                {"timestamp": datetime.datetime.now().isoformat(),
                 "old_value": curr, "change": actual, "new_value": new_val}
            )
            if len(pdata["evolution_history"]) > ic.max_history_entries:
                pdata["evolution_history"] = pdata["evolution_history"][-ic.max_history_entries:]
    
        neuro_results: Dict[str, Any] = {}
        for chem, mod in pdata.get("neurochemical_modifiers", {}).items():
            if chem in ic.neurochemical_profile:
                scaled = mod * abs(actual) / ic.min_impact_threshold * 0.3
                neuro_results[chem] = await IdentityEvolutionSystem._update_neurochemical_baseline(
                    ctx, chemical=chem, impact=scaled,
                    reason=f"preference_{category}_{preference}_update",
                )
    
        return {
            "category": category,
            "preference": preference,
            "old_value": curr,
            "raw_change": raw_change,
            "actual_change": actual,
            "adaptability": adapt_factor,
            "new_value": new_val,
            "neurochemical_impacts": neuro_results,
        }
    
    # ---------------------------------------------------------------------------
    @staticmethod
    @function_tool
    async def _calculate_neurochemical_impacts(
        ctx: RunContextWrapper,
        experience: Any,                           # <= was Dict[str, Any]
    ) -> NeurochemicalImpact:                      # <= dropped the quotes
        """
        Derive baseline-level neurochemical impacts from an experience
        (emotion + valence/arousal + scenario).  Logic unchanged; only the
        type hints were tweaked to keep the Agents SDK strict-schema happy.
        """
        import datetime
        ic        = ctx.context
        emotional = experience.get("emotional_context", {})
        scenario  = experience.get("scenario_type", "general")
        sig       = experience.get("significance", 5) / 10  # scale 0-10 → 0-1
    
        impacts = dict.fromkeys(
            ["nyxamine", "seranix", "oxynixin", "cortanyx", "adrenyx"], 0.0
        )
    
        # ------------------------------------------------------------------ emotion
        emotion_map = {
            "Joy":          {"nyxamine": .4, "cortanyx": -.2},
            "Sadness":      {"cortanyx": .3, "nyxamine": -.2},
            "Fear":         {"cortanyx": .4, "adrenyx":  .3},
            "Anger":        {"cortanyx": .4, "adrenyx":  .3, "oxynixin": -.2},
            "Trust":        {"oxynixin": .4, "cortanyx": -.2},
            "Disgust":      {"cortanyx": .3, "oxynixin": -.3},
            "Anticipation": {"adrenyx":  .3, "nyxamine": .2},
            "Surprise":     {"adrenyx":  .4, "nyxamine": .2},
            "Teasing":      {"nyxamine": .3, "adrenyx":  .2},
            "Controlling":  {"adrenyx":  .3, "oxynixin": -.2},
            "Cruel":        {"cortanyx": .3, "oxynixin": -.4},
            "Detached":     {"cortanyx": .2, "oxynixin": -.3, "seranix":  .2},
        }
        emo = emotional.get("primary_emotion", "")
        if emo in emotion_map:
            inten = emotional.get("primary_intensity", 0.5)
            for chem, val in emotion_map[emo].items():
                impacts[chem] += val * inten * sig
    
        # ----------------------------------------------------------- valence/arousal
        val, arous = emotional.get("valence", 0.0), emotional.get("arousal", 0.5)
        if val >  .3:
            impacts["nyxamine"] += val * .2 * sig
            impacts["cortanyx"] -= val * .1 * sig
        elif val < -.3:
            impacts["cortanyx"] += abs(val) * .2 * sig
            impacts["nyxamine"] -= abs(val) * .1 * sig
        if arous > .6:
            impacts["adrenyx"] += arous * .2 * sig
        elif arous < .4:
            impacts["seranix"] += (1 - arous) * .2 * sig
    
        # ---------------------------------------------------------------- scenario
        scenario_map = {
            "teasing":      {"nyxamine": .3, "adrenyx":  .2},
            "dark":         {"cortanyx": .3, "adrenyx":  .2},
            "indulgent":    {"nyxamine": .3, "seranix":  .2},
            "psychological":{"nyxamine": .2, "cortanyx": .2},
            "nurturing":    {"oxynixin": .3, "seranix":  .2},
            "discipline":   {"cortanyx": .2, "adrenyx":  .2},
            "training":     {"nyxamine": .1, "cortanyx": .1},
            "service":      {"oxynixin": .2, "seranix":  .1},
            "worship":      {"oxynixin": .3, "nyxamine": .2},
        }
        if scenario in scenario_map:
            for chem, val in scenario_map[scenario].items():
                impacts[chem] += val * sig
    
        # --------------------------------------------------------- package & return
        strength = sum(abs(v) for v in impacts.values()) / len(impacts)
        return NeurochemicalImpact(
            nyxamine_impact = impacts["nyxamine"],
            seranix_impact  = impacts["seranix"],
            oxynixin_impact = impacts["oxynixin"],
            cortanyx_impact = impacts["cortanyx"],
            adrenyx_impact  = impacts["adrenyx"],
            impact_strength = strength,
        )
    # ---------------------------------------------------------------------------

    
    # ---------------------------------------------------------------------------
    @staticmethod
    @function_tool
    async def _calculate_emotional_impacts(
        ctx: RunContextWrapper,
        experience: Any,                         # <= was Dict[str, Any]
        neurochemical_impacts: NeurochemicalImpact,  # <= dropped the quotes
    ) -> Dict[str, Dict[str, float]]:
        """
        Convert neurochemical deltas plus the experience’s emotional context
        into per-emotion likelihood/intensity/threshold tweaks.
        """
        ic         = ctx.context
        emotional  = experience.get("emotional_context", {})
        primary    = emotional.get("primary_emotion", "")
        sig        = experience.get("significance", 5) / 10
    
        impacts: Dict[str, Dict[str, float]] = {}
    
        # direct change for the primary emotion ----------------------------------
        if primary in ic.emotional_tendencies:
            impacts[primary] = {
                "likelihood":  .1 * sig,
                "intensity":   .05 * sig,
                "threshold":  -.05 * sig,
            }
    
        # map neurochemical deltas into secondary emotions -----------------------
        chem_emotion = {
            "nyxamine": ["Joy", "Anticipation", "Teasing"],
            "seranix":  ["Contentment", "Trust"],
            "oxynixin": ["Trust", "Love", "Nurturing"],
            "cortanyx": ["Sadness", "Fear", "Anger", "Disgust", "Cruel", "Detached"],
            "adrenyx":  ["Fear", "Surprise", "Anticipation", "Controlling"],
        }
        for chem, delta in (
            ("nyxamine", neurochemical_impacts.nyxamine_impact),
            ("seranix",  neurochemical_impacts.seranix_impact),
            ("oxynixin", neurochemical_impacts.oxynixin_impact),
            ("cortanyx", neurochemical_impacts.cortanyx_impact),
            ("adrenyx",  neurochemical_impacts.adrenyx_impact),
        ):
            if abs(delta) < 0.1:
                continue
            for emo in chem_emotion.get(chem, []):
                if emo not in ic.emotional_tendencies or emo == primary:
                    continue
                scale = 0.6 * abs(delta) * sig
                tgt   = impacts.setdefault(emo, {"likelihood": 0, "intensity": 0, "threshold": 0})
                if delta > 0:
                    tgt["likelihood"] += 0.05 * scale
                    tgt["intensity"]  += 0.03 * scale
                    tgt["threshold"]  -= 0.02 * scale
                else:
                    tgt["likelihood"] -= 0.05 * scale
                    tgt["intensity"]  -= 0.03 * scale
                    tgt["threshold"]  += 0.02 * scale
    
        return impacts
    # ---------------------------------------------------------------------------


    # ---------------------------------------------------------------------------
    @staticmethod
    @function_tool
    async def _update_identity_history(
        ctx: RunContextWrapper,
        trait_changes: Any,                        # <= was Dict[str, Dict[str, Any]]
        preference_changes: Any,                  # <= was Dict[str, Dict[str, Dict[str, Any]]]
        neurochemical_impacts: NeurochemicalImpact,   # <= dropped the quotes
        emotional_impacts: Any,                   # <= was Dict[str, Dict[str, float]]
        experience_id: str,
    ) -> Dict[str, Any]:
        """
        Collate everything that moved in this update, store it in `impact_history`,
        and bump the global counters.  Uses *Any* for the three deeply-nested
        arg­uments so the generated JSON schema does **not** set
        `additionalProperties`, avoiding the UserError you saw.
        """
        import datetime
        ic  = ctx.context
        ts  = datetime.datetime.now().isoformat()
        sig_changes: Dict[str, float] = {}
    
        # ---- traits -----------------------------------------------------------
        for trait, ch in (trait_changes or {}).items():
            if abs(ch.get("actual_change", 0.0)) >= ic.min_impact_threshold:
                sig_changes[f"trait.{trait}"] = ch["actual_change"]
    
        # ---- preferences ------------------------------------------------------
        for cat, prefs in (preference_changes or {}).items():
            for pref, ch in (prefs or {}).items():
                if abs(ch.get("actual_change", 0.0)) >= ic.min_impact_threshold:
                    sig_changes[f"preference.{cat}.{pref}"] = ch["actual_change"]
    
        # ---- neurochemicals ---------------------------------------------------
        for chem, delta in (
            ("nyxamine", neurochemical_impacts.nyxamine_impact),
            ("seranix",  neurochemical_impacts.seranix_impact),
            ("oxynixin", neurochemical_impacts.oxynixin_impact),
            ("cortanyx", neurochemical_impacts.cortanyx_impact),
            ("adrenyx",  neurochemical_impacts.adrenyx_impact),
        ):
            if abs(delta) >= ic.min_impact_threshold:
                sig_changes[f"neurochemical.{chem}"] = delta
    
        # ---- emotions ---------------------------------------------------------
        for emo, emo_ch in (emotional_impacts or {}).items():
            for aspect, delta in (emo_ch or {}).items():
                if abs(delta) >= ic.min_impact_threshold:
                    sig_changes[f"emotion.{emo}.{aspect}"] = delta
    
        # ---- write-back -------------------------------------------------------
        ic.update_count += 1
        ic.last_update   = ts
        ic.impact_history.append(
            {
                "timestamp":            ts,
                "experience_id":        experience_id,
                "significant_changes":  sig_changes,
                "update_count":         ic.update_count,
                "neurochemical_impacts": neurochemical_impacts.model_dump(),
                "emotional_impacts":    emotional_impacts,
            }
        )
        if len(ic.impact_history) > ic.max_history_entries:
            ic.impact_history = ic.impact_history[-ic.max_history_entries:]
    
        return {
            "significant_changes": len(sig_changes),
            "update_count":        ic.update_count,
            "timestamp":           ts,
        }
    # ---------------------------------------------------------------------------


    @staticmethod
    @function_tool
    async def _get_recent_impacts(ctx: RunContextWrapper, limit: int = 10) -> List[Dict[str, Any]]:
        return ctx.context.impact_history[-limit:]

    @staticmethod
    @function_tool
    async def _calculate_identity_changes(ctx: RunContextWrapper, time_period: str = "recent") -> Dict[str, Dict[str, float]]:
        ic = ctx.context
        impacts = ic.impact_history[-10:] if time_period == "recent" else ic.impact_history
        changes = {"traits": {}, "preferences": {}, "neurochemicals": {}, "emotions": {}}
        if not impacts:
            return changes
        for impact in impacts:
            for key, val in impact.get("significant_changes", {}).items():
                parts = key.split(".")
                if parts[0] == "trait":
                    changes["traits"].setdefault(parts[1], 0.0)
                    changes["traits"][parts[1]] += val
                elif parts[0] == "preference":
                    cat, pref = parts[1], parts[2]
                    changes["preferences"].setdefault(cat, {}).setdefault(pref, 0.0)
                    changes["preferences"][cat][pref] += val
                elif parts[0] == "neurochemical":
                    changes["neurochemicals"].setdefault(parts[1], 0.0)
                    changes["neurochemicals"][parts[1]] += val
                elif parts[0] == "emotion":
                    emo, aspect = parts[1], parts[2]
                    changes["emotions"].setdefault(emo, {}).setdefault(aspect, 0.0)
                    changes["emotions"][emo][aspect] += val
        return changes
    
    # ---------------------------------------------------------------------------
    @staticmethod
    @function_tool
    async def _get_neurochemical_patterns(ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Analyse each baseline’s history, label its trend, and then compare chemicals
        to flag “aligned”, “divergent”, or “inversely_related” pairs exactly as in
        the original implementation.
        """
        ic = ctx.context
        patterns: Dict[str, Dict[str, Any]] = {}
    
        # ---------- per-chemical trend & volatility --------------------------------
        for chem, data in ic.neurochemical_profile.items():
            hist = data.get("evolution_history", [])
            if not hist:
                patterns[chem] = {
                    "pattern": "stable",
                    "current_value": data["value"],
                    "baseline": data["value"],
                    "adaptability": data["adaptability"],
                }
                continue
    
            deltas = [h.get("change", 0.0) for h in hist]
            avg = sum(deltas) / len(deltas)
            vol = (
                sum(abs(deltas[i] - deltas[i - 1]) for i in range(1, len(deltas)))
                / (len(deltas) - 1)
                if len(deltas) > 1
                else 0.0
            )
    
            if abs(avg) < 0.01:
                pattern = "stable" if vol < 0.02 else "fluctuating"
            else:
                pattern = "increasing" if avg > 0 else "decreasing"
    
            patterns[chem] = {
                "pattern": pattern,
                "current_value": data["value"],
                "average_change": avg,
                "volatility": vol,
                "adaptability": data["adaptability"],
            }
    
        # ---------- inter-chemical relationship checks ----------------------------
        relationships: Dict[str, str] = {}
    
        # helper
        def _set_alignment(a: str, b: str) -> None:
            relationships[f"{a}_{b}"] = (
                "aligned"
                if patterns[a]["pattern"] == patterns[b]["pattern"]
                else "divergent"
            )
    
        _set_alignment("nyxamine", "oxynixin")
        _set_alignment("cortanyx", "adrenyx")
        _set_alignment("seranix", "cortanyx")
    
        # inverse-trend checks (increasing vs decreasing)
        def _inverse(a: str, b: str) -> None:
            pa, pb = patterns[a]["pattern"], patterns[b]["pattern"]
            if (pa == "increasing" and pb == "decreasing") or (
                pa == "decreasing" and pb == "increasing"
            ):
                relationships[f"{a}_{b}"] = "inversely_related"
    
        _inverse("nyxamine", "cortanyx")
        _inverse("oxynixin", "cortanyx")
    
        return {
            "chemical_patterns": patterns,
            "relationships": relationships,
        }
    # ---------------------------------------------------------------------------
    
    @staticmethod    
    @function_tool
    async def _calculate_trait_consistency(ctx: RunContextWrapper) -> Dict[str, float]:
        ic = ctx.context
        scores = {}
        for trait, data in ic.identity_traits.items():
            hist = [h.get("change",0) for h in data.get("evolution_history",[])]
            if not hist:
                scores[trait] = 1.0
                continue
            mean = sum(hist)/len(hist)
            var  = sum((x-mean)**2 for x in hist)/len(hist)
            scores[trait] = max(0.0, 1.0 - math.sqrt(var)*5)
        return scores

    @staticmethod
    @function_tool
    async def _calculate_preference_consistency(ctx: RunContextWrapper) -> Dict[str, Dict[str, float]]:
        ic = ctx.context
        out: Dict[str, Dict[str, float]] = {}
        for cat, prefs in ic.identity_preferences.items():
            out[cat] = {}
            for pref, data in prefs.items():
                hist = [h.get("change",0) for h in data.get("evolution_history",[]) if "change" in h]
                if not hist:
                    out[cat][pref] = 1.0
                    continue
                mean = sum(hist)/len(hist)
                var  = sum((x-mean)**2 for x in hist)/len(hist)
                out[cat][pref] = max(0.0, 1.0 - math.sqrt(var)*5)
        return out
    
    
    # ---------------------------------------------------------------------------
    @staticmethod
    @function_tool
    async def _identify_contradictions(ctx: RunContextWrapper) -> List[Dict[str, Any]]:
        """
        Scan the current identity state and return a list of potential contradictions /
        mis-alignments exactly as the legacy implementation did.
        """
        ic = ctx.context        # <- convenient alias
    
        contradictions: List[Dict[str, Any]] = []
    
        # ---------- 1. Trait-vs-Trait conflicts -----------------------------------
        trait_pairs = [
            ("dominance",  "patience"),
            ("playfulness","strictness"),
            ("cruelty",    "patience"),
        ]
        for t1, t2 in trait_pairs:
            if t1 in ic.identity_traits and t2 in ic.identity_traits:
                v1 = ic.identity_traits[t1]["value"]
                v2 = ic.identity_traits[t2]["value"]
                if v1 > 0.7 and v2 > 0.7:
                    contradictions.append({
                        "type": "trait_contradiction",
                        "elements": [t1, t2],
                        "values":   [v1, v2],
                        "description": (
                            f"Traits '{t1}' ({v1:.2f}) and '{t2}' ({v2:.2f}) are both high "
                            "even though they typically oppose each other."
                        ),
                    })
    
        # ---------- 2. Neurochemical high-high anomalies --------------------------
        def _chem_val(c: str) -> float:
            return ic.neurochemical_profile.get(c, {}).get("value", 0.0)
    
        if _chem_val("nyxamine") > 0.7 and _chem_val("cortanyx") > 0.7:
            contradictions.append({
                "type": "neurochemical_contradiction",
                "elements": ["nyxamine", "cortanyx"],
                "values":   [_chem_val("nyxamine"), _chem_val("cortanyx")],
                "description": "Pleasure (nyxamine) and stress (cortanyx) are simultaneously very high.",
            })
    
        if _chem_val("seranix") > 0.7 and _chem_val("adrenyx") > 0.7:
            contradictions.append({
                "type": "neurochemical_contradiction",
                "elements": ["seranix", "adrenyx"],
                "values":   [_chem_val("seranix"), _chem_val("adrenyx")],
                "description": "Calm (seranix) and high alertness (adrenyx) are both very high.",
            })
    
        # ---------- 3. Emotion-vs-Emotion clashes ---------------------------------
        emotion_pairs = [
            ("Joy",   "Sadness"),
            ("Trust", "Disgust"),
            ("Teasing","Detached"),
        ]
        for e1, e2 in emotion_pairs:
            if e1 in ic.emotional_tendencies and e2 in ic.emotional_tendencies:
                l1 = ic.emotional_tendencies[e1]["likelihood"]
                l2 = ic.emotional_tendencies[e2]["likelihood"]
                if l1 > 0.7 and l2 > 0.7:
                    contradictions.append({
                        "type": "emotion_contradiction",
                        "elements": [e1, e2],
                        "values":   [l1, l2],
                        "description": (
                            f"Both emotions '{e1}' ({l1:.2f}) and '{e2}' ({l2:.2f}) "
                            "have high likelihoods despite being opposites."
                        ),
                    })
    
        # ---------- 4. Trait ↔ Preference mismatches ------------------------------
        tp_pairs = [
            ("dominance", "scenario_types", "discipline"),
            ("playfulness","scenario_types", "teasing"),
            ("strictness", "emotional_tones","stern"),
            ("creativity", "interaction_styles","metaphorical"),
            ("cruelty",    "emotional_tones","cruel"),
        ]
        for trait, cat, pref in tp_pairs:
            if (trait in ic.identity_traits and
                cat   in ic.identity_preferences and
                pref  in ic.identity_preferences[cat]):
                tv = ic.identity_traits[trait]["value"]
                pv = ic.identity_preferences[cat][pref]["value"]
                if (tv > 0.7 and pv < 0.3) or (tv < 0.3 and pv > 0.7):
                    contradictions.append({
                        "type": "trait_preference_mismatch",
                        "elements": [trait, f"{cat}.{pref}"],
                        "values":   [tv, pv],
                        "description": (
                            f"Trait '{trait}' ({tv:.2f}) and preference '{pref}' "
                            f"within '{cat}' ({pv:.2f}) are strongly mis-aligned."
                        ),
                    })
    
        # ---------- 5. Neurochemical ↔ Trait align / inverse checks --------------
        # (chem, trait, inverse_expected)
        ct_pairs = [
            ("nyxamine", "creativity",  False),
            ("seranix",  "patience",    False),
            ("oxynixin", "cruelty",     True),   # expects inverse
            ("cortanyx", "strictness",  False),
            ("adrenyx",  "intensity",   False),
        ]
        for chem, trait, inverse in ct_pairs:
            if chem in ic.neurochemical_profile and trait in ic.identity_traits:
                cv = _chem_val(chem)
                tv = ic.identity_traits[trait]["value"]
                if inverse:
                    # expect one high & one low; flag if both high or both low
                    if (cv > 0.7 and tv > 0.7) or (cv < 0.3 and tv < 0.3):
                        contradictions.append({
                            "type": "neurochemical_trait_mismatch",
                            "elements": [chem, trait],
                            "values":   [cv, tv],
                            "description": (
                                f"Expected inverse relationship between {chem} and '{trait}', "
                                "but both are aligned."
                            ),
                        })
                else:
                    # expect alignment; flag if poles apart
                    if (cv > 0.7 and tv < 0.3) or (cv < 0.3 and tv > 0.7):
                        contradictions.append({
                            "type": "neurochemical_trait_mismatch",
                            "elements": [chem, trait],
                            "values":   [cv, tv],
                            "description": (
                                f"{chem} and '{trait}' should align, "
                                "yet they are strongly divergent."
                            ),
                        })
    
        return contradictions
    # ---------------------------------------------------------------------------
    
    @staticmethod
    @function_tool
    async def _assess_neurochemical_coherence(ctx: RunContextWrapper) -> Dict[str, Any]:
        ic = ctx.context
        relations = [("nyxamine","cortanyx","inverse"),
                     ("seranix","adrenyx","inverse"),
                     ("oxynixin","cortanyx","inverse"),
                     ("nyxamine","oxynixin","aligned"),
                     ("cortanyx","adrenyx","aligned")]
        scores, imbalances = {}, []
        for c1,c2,rel in relations:
            v1, v2 = ic.neurochemical_profile[c1]["value"], ic.neurochemical_profile[c2]["value"]
            if rel=="inverse":
                coh = abs(v1-v2)
                if coh < .3:
                    imbalances.append({"type":"unexpected_similarity","chemicals":[c1,c2],
                                       "values":[v1,v2],"expected":"inverse","coherence":coh})
            else:
                coh = 1.0-abs(v1-v2)
                if coh < .7:
                    imbalances.append({"type":"unexpected_difference","chemicals":[c1,c2],
                                       "values":[v1,v2],"expected":"aligned","coherence":coh})
            scores[f"{c1}_{c2}"] = coh
    
        for chem, data in ic.neurochemical_profile.items():
            if data["value"]>0.9:
                imbalances.append({"type":"extreme_high","chemical":chem,"value":data["value"]})
            elif data["value"]<0.1:
                imbalances.append({"type":"extreme_low","chemical":chem,"value":data["value"]})
    
        overall = (sum(scores.values())/len(scores) if scores else 0.7) - len(imbalances)*0.05
        overall = max(.1, min(1.0, overall))
        return {"overall_coherence":overall, "relation_scores":scores,
                "imbalances":imbalances,
                "balanced_chemicals":[c for c in ic.neurochemical_profile if not any(i.get("chemical")==c for i in imbalances)],
                "assessment_time":datetime.datetime.now().isoformat()}
