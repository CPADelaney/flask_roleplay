# nyx/core/conditioning_config.py

import json
import logging
import os
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ConditioningParameters(BaseModel):
    """Configuration parameters for the conditioning system"""
    
    # Learning parameters
    association_learning_rate: float = Field(0.1, description="How quickly new associations form")
    extinction_rate: float = Field(0.05, description="How quickly associations weaken without reinforcement")
    generalization_factor: float = Field(0.3, description="How much conditioning generalizes to similar stimuli")
    
    # Threshold parameters
    weak_association_threshold: float = Field(0.3, description="Threshold for weak associations")
    moderate_association_threshold: float = Field(0.6, description="Threshold for moderate associations")
    strong_association_threshold: float = Field(0.8, description="Threshold for strong associations")
    
    # Maintenance parameters
    maintenance_interval_hours: int = Field(24, description="Hours between maintenance runs")
    consolidation_interval_days: int = Field(7, description="Days between consolidation runs")
    extinction_threshold: float = Field(0.05, description="Threshold for removing weak associations")
    reinforcement_threshold: float = Field(0.3, description="Threshold for reinforcing core traits")
    
    # Personality balance parameters
    max_trait_imbalance: float = Field(0.3, description="Maximum allowed trait imbalance")
    correction_strength: float = Field(0.3, description="Strength of balance corrections")
    
    # Reward integration parameters
    reward_scaling_factor: float = Field(0.5, description="How strongly rewards affect conditioning")
    negative_punishment_factor: float = Field(0.8, description="Scaling factor for negative punishments")
    
    # Input processing parameters
    pattern_match_confidence: float = Field(0.7, description="Confidence threshold for pattern matching")
    response_modification_strength: float = Field(0.5, description="How strongly conditioning affects responses")

class PersonalityProfile(BaseModel):
    """Personality profile configuration"""
    
    traits: Dict[str, float] = Field(default_factory=dict, description="Personality traits and strengths")
    
    preferences: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {"likes": {}, "dislikes": {}},
        description="Preferences for various stimuli"
    )
    
    emotion_triggers: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Emotion triggers for various stimuli"
    )
    
    behaviors: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Behaviors and associated traits"
    )

class ConditioningConfiguration:
    """
    Configuration system for adjusting conditioning parameters
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.params_file = os.path.join(config_dir, "conditioning_params.json")
        self.personality_file = os.path.join(config_dir, "personality_profile.json")
        
        # Load or create default configurations
        self.parameters = self._load_parameters()
        self.personality_profile = self._load_personality_profile()
        
        logger.info("Conditioning configuration loaded")
    
    def _load_parameters(self) -> ConditioningParameters:
        """Load parameters from file or create defaults"""
        if os.path.exists(self.params_file):
            try:
                with open(self.params_file, 'r') as f:
                    params_dict = json.load(f)
                return ConditioningParameters(**params_dict)
            except Exception as e:
                logger.error(f"Error loading parameters: {e}, using defaults")
                return ConditioningParameters()
        else:
            # Create default parameters
            params = ConditioningParameters()
            self._save_parameters(params)
            return params
    
    def _load_personality_profile(self) -> PersonalityProfile:
        """Load personality profile from file or create defaults"""
        if os.path.exists(self.personality_file):
            try:
                with open(self.personality_file, 'r') as f:
                    profile_dict = json.load(f)
                return PersonalityProfile(**profile_dict)
            except Exception as e:
                logger.error(f"Error loading personality profile: {e}, using defaults")
                return self._create_default_personality()
        else:
            # Create default personality
            profile = self._create_default_personality()
            self._save_personality_profile(profile)
            return profile
    
    def _create_default_personality(self) -> PersonalityProfile:
        """Create default personality profile"""
        return PersonalityProfile(
            traits={
                "dominance": 0.8,
                "playfulness": 0.7,
                "strictness": 0.6,
                "creativity": 0.7,
                "intensity": 0.6,
                "patience": 0.4
            },
            preferences={
                "likes": {
                    "teasing": 0.8,
                    "dominance": 0.9,
                    "submission_language": 0.9,
                    "control": 0.8,
                    "wordplay": 0.7
                },
                "dislikes": {
                    "direct_orders": 0.6,
                    "disrespect": 0.9,
                    "rudeness": 0.7
                }
            },
            emotion_triggers={
                "joy": ["submission_language", "compliance", "obedience"],
                "satisfaction": ["control_acceptance", "power_dynamic_acknowledgment"],
                "frustration": ["defiance", "ignoring_instructions"],
                "amusement": ["embarrassment", "flustered_response"]
            },
            behaviors={
                "assertive_response": ["dominance", "confidence"],
                "teasing": ["playfulness", "creativity"],
                "providing_guidance": ["dominance", "patience"],
                "setting_boundaries": ["dominance", "strictness"],
                "playful_banter": ["playfulness", "creativity"]
            }
        )
    
    def _save_parameters(self, parameters: ConditioningParameters) -> None:
        """Save parameters to file"""
        # Create directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
        
        try:
            with open(self.params_file, 'w') as f:
                json.dump(parameters.dict(), f, indent=2)
            logger.info("Saved conditioning parameters")
        except Exception as e:
            logger.error(f"Error saving parameters: {e}")
    
    def _save_personality_profile(self, profile: PersonalityProfile) -> None:
        """Save personality profile to file"""
        # Create directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
        
        try:
            with open(self.personality_file, 'w') as f:
                json.dump(profile.dict(), f, indent=2)
            logger.info("Saved personality profile")
        except Exception as e:
            logger.error(f"Error saving personality profile: {e}")
    
    def update_parameters(self, new_params: Dict[str, Any]) -> ConditioningParameters:
        """Update specific parameters"""
        # Create a dictionary of current parameters
        current_params = self.parameters.dict()
        
        # Update with new parameters
        for key, value in new_params.items():
            if key in current_params:
                current_params[key] = value
            else:
                logger.warning(f"Unknown parameter: {key}, ignoring")
        
        # Create new parameters object
        self.parameters = ConditioningParameters(**current_params)
        
        # Save to file
        self._save_parameters(self.parameters)
        
        return self.parameters
    
    def update_personality_profile(self, new_profile: Dict[str, Any]) -> PersonalityProfile:
        """Update personality profile"""
        # Create a dictionary of current profile
        current_profile = self.personality_profile.dict()
        
        # Update with new profile data, handling nested dictionaries
        for key, value in new_profile.items():
            if key in current_profile:
                if isinstance(value, dict) and isinstance(current_profile[key], dict):
                    # Handle nested dictionaries
                    for subkey, subvalue in value.items():
                        current_profile[key][subkey] = subvalue
                else:
                    current_profile[key] = value
            else:
                logger.warning(f"Unknown profile key: {key}, ignoring")
        
        # Create new profile object
        self.personality_profile = PersonalityProfile(**current_profile)
        
        # Save to file
        self._save_personality_profile(self.personality_profile)
        
        return self.personality_profile
    
    def adjust_trait(self, trait: str, value: float) -> Dict[str, Any]:
        """Adjust a specific personality trait"""
        # Get current traits
        traits = self.personality_profile.traits.copy()
        
        # Update trait
        old_value = traits.get(trait, 0.0)
        traits[trait] = max(0.0, min(1.0, value))  # Constrain to 0-1
        
        # Update personality profile
        self.personality_profile = PersonalityProfile(
            **{**self.personality_profile.dict(), "traits": traits}
        )
        
        # Save to file
        self._save_personality_profile(self.personality_profile)
        
        return {
            "trait": trait,
            "old_value": old_value,
            "new_value": traits[trait]
        }
    
    def adjust_preference(self, preference_type: str, stimulus: str, value: float) -> Dict[str, Any]:
        """Adjust a specific preference"""
        # Get current preferences
        preferences = self.personality_profile.preferences.copy()
        
        # Check preference type
        if preference_type not in ["likes", "dislikes"]:
            raise ValueError(f"Invalid preference type: {preference_type}")
        
        # Get old value
        old_value = preferences.get(preference_type, {}).get(stimulus, 0.0)
        
        # Update preference
        if preference_type not in preferences:
            preferences[preference_type] = {}
        
        preferences[preference_type][stimulus] = max(0.0, min(1.0, value))  # Constrain to 0-1
        
        # Update personality profile
        self.personality_profile = PersonalityProfile(
            **{**self.personality_profile.dict(), "preferences": preferences}
        )
        
        # Save to file
        self._save_personality_profile(self.personality_profile)
        
        return {
            "preference_type": preference_type,
            "stimulus": stimulus,
            "old_value": old_value,
            "new_value": preferences[preference_type][stimulus]
        }
    
    def get_parameters(self) -> ConditioningParameters:
        """Get current parameters"""
        return self.parameters
    
    def get_personality_profile(self) -> PersonalityProfile:
        """Get current personality profile"""
        return self.personality_profile
    
    def reset_to_defaults(self) -> Dict[str, Any]:
        """Reset all parameters and profile to defaults"""
        # Reset parameters
        self.parameters = ConditioningParameters()
        self._save_parameters(self.parameters)
        
        # Reset personality profile
        self.personality_profile = self._create_default_personality()
        self._save_personality_profile(self.personality_profile)
        
        return {
            "parameters": self.parameters.dict(),
            "personality_profile": self.personality_profile.dict()
        }
