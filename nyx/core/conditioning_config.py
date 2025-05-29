# nyx/core/conditioning_config.py

import json
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field

# Make sure these imports are correct based on your project structure
# Assuming 'agents' is the correct package name for the SDK
from agents import Agent, Runner, function_tool, RunContextWrapper, ModelSettings, trace
from agents.tracing import function_span
# You might need specific Tool types if not using function_tool directly everywhere
from agents.tool import FunctionTool # Import FunctionTool if needed elsewhere

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


class PreferenceDetail(BaseModel):
    type: str = Field(..., description="Type of preference (e.g., like, dislike, want, avoid)")
    value: float = Field(..., description="Strength/direction of the preference")

class EmotionTriggerDetail(BaseModel):
    emotion: str = Field(..., description="The emotion to be triggered")
    intensity: float = Field(0.5, description="The intensity of the triggered emotion (0.0-1.0)")
    valence: Optional[float] = Field(None, description="Optional valence override for the emotion trigger")

class PersonalityProfile(BaseModel):
    """Personality profile configuration"""
    
    traits: Dict[str, float] = Field(
        default_factory=dict,
        description="Personality traits and their strengths (0.0-1.0)"
    )
    
    preferences: Dict[str, PreferenceDetail] = Field(
        default_factory=dict,
        description="Preferences for various stimuli. Key is stimulus, value contains type and value."
    )
    
    emotion_triggers: Dict[str, EmotionTriggerDetail] = Field(
        default_factory=dict,
        description="Emotion triggers. Key is the trigger string, value contains emotion and intensity."
    )
    
    behaviors: Dict[str, List[str]] = Field( # This structure seems fine for now
        default_factory=dict,
        description="Behaviors and associated traits that enable them."
    )



class ConfigUpdateResult(BaseModel):
    """Result of a configuration update operation"""
    success: bool = Field(..., description="Whether the update was successful")
    updated_keys: List[str] = Field(default_factory=list, description="Keys that were updated")
    previous_values: Dict[str, Any] = Field(default_factory=dict, description="Previous values")
    new_values: Dict[str, Any] = Field(default_factory=dict, description="New values")
    message: str = Field("", description="Additional information")


class TraitUpdateResult(BaseModel):
    """Result of a trait update operation"""
    trait: str = Field(..., description="The trait that was updated")
    old_value: float = Field(..., description="Previous value")
    new_value: float = Field(..., description="New value")
    success: bool = Field(True, description="Whether the update was successful")

class ConfigurationContext:
    """Configuration context for sharing between agents and tools"""
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.params_file = os.path.join(config_dir, "conditioning_params.json")
        self.personality_file = os.path.join(config_dir, "personality_profile.json")
        self.parameters: Optional[ConditioningParameters] = None # Initialize explicitly
        self.personality_profile: Optional[PersonalityProfile] = None # Initialize explicitly
        self.trace_group_id = f"conditioning_config_{os.path.basename(config_dir)}"


class ConditioningConfiguration:
    """
    Configuration system for adjusting conditioning parameters using Agents SDK
    """
    def __init__(self, config_dir: str = "config"):
        self.context = ConfigurationContext(config_dir)
        # Load configurations *before* creating agents that might need them immediately
        self._load_parameters()
        self._load_personality_profile()
        # Initialize agents
        self.config_manager_agent = self._create_config_manager_agent()
        self.personality_editor_agent = self._create_personality_editor_agent()
        logger.info("Conditioning configuration initialized with Agents SDK")

    def _create_config_manager_agent(self) -> Agent:
        """Create agent for managing system parameters"""
        return Agent(
            name="Config_Manager",
            instructions="""
            You are the configuration management system for a sophisticated conditioning architecture.
            Your role is to:
            1. Manage conditioning parameters
            2. Ensure parameter values remain within valid ranges
            3. Provide explanations for parameter adjustments
            4. Save and load parameter configurations
            Make thoughtful adjustments that maintain system coherence.
            """,
            tools=[
                # Correct usage: Wrap the method reference here
                function_tool(self._get_parameters),
                function_tool(self._update_parameters),
                function_tool(self._save_parameters), # Assuming you want this as a tool too
                function_tool(self._validate_parameters),
                function_tool(self._reset_to_defaults)
            ],
            model_settings=ModelSettings(temperature=0.1)
        )

    def _create_personality_editor_agent(self) -> Agent:
        """Create agent for editing personality profiles"""
        return Agent(
            name="Personality_Editor",
            instructions="""
            You are the personality profile editor for a sophisticated conditioning architecture.
            Your role is to:
            1. Manage personality traits and their values
            2. Configure preferences and their strengths
            3. Set up emotion triggers and behavior associations
            4. Ensure profile coherence and balance
            Make thoughtful adjustments that maintain personality coherence.
            """,
            tools=[
                # Correct usage: Wrap the method reference here
                function_tool(self._get_personality_profile),
                function_tool(self._update_personality_profile),
                function_tool(self._adjust_trait),
                function_tool(self._adjust_preference),
                function_tool(self._save_personality_profile) # Assuming you want this as a tool too
            ],
            model_settings=ModelSettings(temperature=0.2)
        )

    # REMOVED @staticmethod and @function_tool decorators from method definitions
    async def _get_parameters(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get current conditioning parameters (Internal method for Agent tool)
        Returns: Dictionary of current parameters
        """
        with function_span("get_parameters"):
            # No need to load here if loaded in __init__
            # if not self.context.parameters:
            #     self._load_parameters() # Should already be loaded
            if self.context.parameters:
                 return self.context.parameters.model_dump()
            logger.warning("_get_parameters called but parameters not loaded.")
            return {} # Return empty if somehow not loaded

    # REMOVED @staticmethod and @function_tool
    async def _update_parameters(self, ctx: RunContextWrapper,
                                 new_params: Dict[str, Any]) -> ConfigUpdateResult:
        """
        Update specific conditioning parameters (Internal method for Agent tool)
        Args: new_params: Dictionary of parameters to update
        Returns: Result of the update operation
        """
        with function_span("update_parameters", input=str(new_params)):
            if not self.context.parameters:
                # Should not happen if loaded in __init__, but handle defensively
                logger.error("Attempted to update parameters before loading.")
                return ConfigUpdateResult(success=False, message="Parameters not loaded.")

            result = ConfigUpdateResult(success=True)
            current_params = self.context.parameters.model_dump()
            valid_updates = {}

            for key, value in new_params.items():
                if hasattr(self.context.parameters, key): # Check against Pydantic model fields
                    result.previous_values[key] = current_params[key]
                    result.new_values[key] = value
                    result.updated_keys.append(key)
                    valid_updates[key] = value
                else:
                    result.message += f"Unknown parameter: {key}. "

            if not result.updated_keys:
                result.success = False
                result.message += "No valid parameters provided for update."
                return result

            try:
                # Create new parameters object with updates, this validates types
                updated_params_dict = {**current_params, **valid_updates}
                self.context.parameters = ConditioningParameters(**updated_params_dict)
                self._save_parameters_internal(self.context.parameters) # Use internal save method
                result.message += f"Updated {len(result.updated_keys)} parameters."
            except Exception as e: # Catch validation errors etc.
                 logger.error(f"Error updating parameters: {e}")
                 result.success = False
                 result.message = f"Error during parameter update: {e}"
                 # Optionally revert to old parameters
                 # self.context.parameters = ConditioningParameters(**current_params)

            return result

    # REMOVED @staticmethod and @function_tool
    async def _save_parameters(self, ctx: RunContextWrapper) -> Dict[str, Any]:
         """ Saves the current parameters to disk (Internal method for Agent tool) """
         with function_span("save_parameters_tool_call"):
             if self.context.parameters:
                 try:
                     self._save_parameters_internal(self.context.parameters)
                     return {"success": True, "message": "Parameters saved."}
                 except Exception as e:
                     logger.error(f"Error saving parameters via tool: {e}")
                     return {"success": False, "message": f"Error saving parameters: {e}"}
             else:
                 return {"success": False, "message": "No parameters loaded to save."}

    # REMOVED @staticmethod and @function_tool
    async def _validate_parameters(self, ctx: RunContextWrapper,
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters against constraints (Internal method for Agent tool)
        Args: parameters: Parameters to validate
        Returns: Validation results with any issues
        """
        with function_span("validate_parameters"):
            validation_result = {"valid": True, "issues": {}}
            try:
                # Attempt Pydantic validation first (catches type errors, missing fields if required)
                ConditioningParameters(**parameters)

                # Define constraints for range checks
                constraints = {
                    "association_learning_rate": (0.0, 1.0),
                    "extinction_rate": (0.0, 0.5),
                    "generalization_factor": (0.0, 1.0),
                    "weak_association_threshold": (0.1, 0.4),
                    "moderate_association_threshold": (0.4, 0.7),
                    "strong_association_threshold": (0.7, 1.0),
                    "maintenance_interval_hours": (1, 168),
                    "consolidation_interval_days": (1, 90),
                    "extinction_threshold": (0.01, 0.2),
                    "reinforcement_threshold": (0.1, 0.5),
                    "max_trait_imbalance": (0.1, 0.5),
                    "correction_strength": (0.1, 0.5),
                    "reward_scaling_factor": (0.1, 1.0),
                    "negative_punishment_factor": (0.1, 1.0),
                    "pattern_match_confidence": (0.5, 0.9),
                    "response_modification_strength": (0.1, 0.9)
                }

                # Perform custom range checks *inside* the try block
                for param, value in parameters.items():
                    if param in constraints:
                        min_val, max_val = constraints[param]
                        # Add a type check before comparison for robustness
                        if not isinstance(value, (int, float)):
                            validation_result["valid"] = False
                            validation_result["issues"][param] = f"Invalid type for {param}: expected number, got {type(value).__name__}"
                            continue # Skip range check if type is wrong

                        if not min_val <= value <= max_val:
                            validation_result["valid"] = False
                            validation_result["issues"][param] = f"Value {value} outside range [{min_val}, {max_val}]"

            except Exception as e:  # Catch Pydantic validation errors or other issues
                validation_result["valid"] = False
                # Record the specific validation error
                validation_result["issues"]["initial_validation"] = f"Validation failed: {e}"

            # Check relationships between thresholds *after* try-except,
            # but only proceed if still considered valid so far.
            if validation_result["valid"] and ("weak_association_threshold" in parameters and
                                               "moderate_association_threshold" in parameters and
                                               "strong_association_threshold" in parameters):

                weak = parameters.get("weak_association_threshold")
                moderate = parameters.get("moderate_association_threshold")
                strong = parameters.get("strong_association_threshold")

                # Ensure values are numbers before comparing
                if not all(isinstance(v, (int, float)) for v in [weak, moderate, strong]):
                     validation_result["valid"] = False
                     validation_result["issues"]["threshold_ordering"] = "Threshold values must be numbers for ordering check."
                elif not (weak < moderate < strong):
                    validation_result["valid"] = False
                    validation_result["issues"]["threshold_ordering"] = (
                        f"Thresholds must be ordered: weak ({weak}) < moderate ({moderate}) < strong ({strong})"
                    )

            return validation_result

    # REMOVED @staticmethod and @function_tool
    async def _reset_to_defaults(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Reset all parameters and profile to defaults (Internal method for Agent tool)
        Returns: Default parameters and profile
        """
        with function_span("reset_to_defaults"):
            try:
                self.context.parameters = ConditioningParameters() # Reset parameters instance
                self._save_parameters_internal(self.context.parameters)

                self.context.personality_profile = self._create_default_personality() # Reset profile instance
                self._save_personality_profile_internal(self.context.personality_profile)

                return {
                    "success": True,
                    "message": "Parameters and profile reset to defaults.",
                    "parameters": self.context.parameters.model_dump(),
                    "personality_profile": self.context.personality_profile.model_dump()
                }
            except Exception as e:
                 logger.error(f"Error resetting to defaults: {e}")
                 return {"success": False, "message": f"Error resetting to defaults: {e}"}


    # REMOVED @staticmethod and @function_tool
    async def _get_personality_profile(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get current personality profile (Internal method for Agent tool)
        Returns: Current personality profile data
        """
        with function_span("get_personality_profile"):
            # No need to load here if loaded in __init__
            # if not self.context.personality_profile:
            #     self._load_personality_profile() # Should already be loaded
            if self.context.personality_profile:
                return self.context.personality_profile.model_dump()
            logger.warning("_get_personality_profile called but profile not loaded.")
            return {} # Return empty if somehow not loaded

    # REMOVED @staticmethod and @function_tool
    async def _update_personality_profile(self, ctx: RunContextWrapper,
                                          new_profile: Dict[str, Any]) -> ConfigUpdateResult:
        """
        Update personality profile (Internal method for Agent tool)
        Args: new_profile: Dictionary with profile updates
        Returns: Result of the update operation
        """
        with function_span("update_personality_profile", input=str(new_profile)):
            if not self.context.personality_profile:
                 logger.error("Attempted to update profile before loading.")
                 return ConfigUpdateResult(success=False, message="Personality profile not loaded.")

            result = ConfigUpdateResult(success=True)
            current_profile_dict = self.context.personality_profile.model_dump()
            updated_profile_dict = current_profile_dict.copy() # Work on a copy

            # Update profile, handling nested dictionaries carefully
            for key, value in new_profile.items():
                if key in updated_profile_dict:
                    result.previous_values[key] = updated_profile_dict[key]
                    result.new_values[key] = value # Store the incoming value first

                    if isinstance(value, dict) and isinstance(updated_profile_dict[key], dict):
                        # Deep merge for known dictionary fields if necessary, or just update
                        # Simple update (overwrites nested dict):
                        updated_profile_dict[key] = value
                        result.updated_keys.append(key)
                        # If you need merging:
                        # updated_profile_dict[key].update(value)
                        # result.new_values[key] = updated_profile_dict[key] # Update new_value after merge
                        # result.updated_keys.append(key) # Track top-level key
                    elif isinstance(value, list) and isinstance(updated_profile_dict[key], list):
                         # Handle lists if needed (e.g., append, replace) - currently replaces
                         updated_profile_dict[key] = value
                         result.updated_keys.append(key)
                    elif not isinstance(value, (dict, list)):
                        # Direct update for simple values (float, str, etc.)
                        updated_profile_dict[key] = value
                        result.updated_keys.append(key)
                    else:
                        # Type mismatch (e.g., trying to update dict with list) - Log or handle as error
                        result.message += f"Type mismatch for key '{key}'. Update skipped. "
                        result.updated_keys.append(key) # Still record attempt
                        result.success = False # Mark as partial failure?

                else:
                    result.message += f"Unknown profile key: {key}. "

            if not result.updated_keys or not any(k in updated_profile_dict for k in result.updated_keys):
                 # Check if any *valid* keys were provided
                 result.success = False
                 result.message += "No valid profile elements provided for update."
                 return result

            # Validate and update the actual profile object
            try:
                self.context.personality_profile = PersonalityProfile(**updated_profile_dict)
                self._save_personality_profile_internal(self.context.personality_profile)
                result.message += f"Updated {len(result.updated_keys)} profile elements."
            except Exception as e:
                 logger.error(f"Error updating personality profile: {e}")
                 result.success = False
                 result.message = f"Error during profile update: {e}"
                 # Optionally revert
                 # self.context.personality_profile = PersonalityProfile(**current_profile_dict)

            return result

    # REMOVED @staticmethod and @function_tool
    async def _adjust_trait(self, ctx: RunContextWrapper,
                            trait: str,
                            value: float) -> TraitUpdateResult:
        """
        Adjust a specific personality trait (Internal method for Agent tool)
        Args: trait: The trait to adjust, value: New value for the trait (0.0-1.0)
        Returns: Result of the trait adjustment
        """
        with function_span("adjust_trait", input=f"trait={trait}, value={value}"):
            if not self.context.personality_profile:
                 logger.error("Attempted to adjust trait before profile loaded.")
                 # Cannot return TraitUpdateResult easily here, maybe raise?
                 # For now, return failure within the structure if possible
                 return TraitUpdateResult(trait=trait, old_value=0.0, new_value=value, success=False)

            # Get current traits
            updated_profile_dict = self.context.personality_profile.model_dump()
            traits = updated_profile_dict.get("traits", {})

            old_value = traits.get(trait, 0.0)
            new_value_clamped = max(0.0, min(1.0, value)) # Constrain to 0-1
            traits[trait] = new_value_clamped

            updated_profile_dict["traits"] = traits

            try:
                self.context.personality_profile = PersonalityProfile(**updated_profile_dict)
                self._save_personality_profile_internal(self.context.personality_profile)
                return TraitUpdateResult(
                    trait=trait,
                    old_value=old_value,
                    new_value=new_value_clamped,
                    success=True
                 )
            except Exception as e:
                logger.error(f"Error adjusting trait '{trait}': {e}")
                # Optionally revert
                # self.context.personality_profile = PersonalityProfile(**self.context.personality_profile.model_dump()) # Revert
                return TraitUpdateResult(
                    trait=trait,
                    old_value=old_value,
                    new_value=value, # Report attempted value
                    success=False
                )


    # REMOVED @staticmethod and @function_tool
    async def _adjust_preference(self, ctx: RunContextWrapper,
                                 preference_type: str,
                                 stimulus: str,
                                 value: float) -> Dict[str, Any]:
        """
        Adjust a specific preference (Internal method for Agent tool)
        Args: preference_type: Type ("likes" or "dislikes"), stimulus: Item, value: New value (0.0-1.0)
        Returns: Result of the preference adjustment
        """
        with function_span("adjust_preference"):
            if not self.context.personality_profile:
                logger.error("Attempted to adjust preference before profile loaded.")
                return {"success": False, "error": "Personality profile not loaded."}

            if preference_type not in ["likes", "dislikes"]:
                return {
                    "success": False,
                    "error": f"Invalid preference type: {preference_type}",
                    "valid_types": ["likes", "dislikes"]
                }

            updated_profile_dict = self.context.personality_profile.model_dump()
            preferences = updated_profile_dict.get("preferences", {"likes": {}, "dislikes": {}})

            # Ensure the preference type dict exists
            if preference_type not in preferences:
                preferences[preference_type] = {}

            old_value = preferences[preference_type].get(stimulus, 0.0)
            new_value_clamped = max(0.0, min(1.0, value)) # Constrain to 0-1
            preferences[preference_type][stimulus] = new_value_clamped

            updated_profile_dict["preferences"] = preferences

            try:
                self.context.personality_profile = PersonalityProfile(**updated_profile_dict)
                self._save_personality_profile_internal(self.context.personality_profile)
                return {
                    "success": True,
                    "preference_type": preference_type,
                    "stimulus": stimulus,
                    "old_value": old_value,
                    "new_value": new_value_clamped
                }
            except Exception as e:
                 logger.error(f"Error adjusting preference '{preference_type}/{stimulus}': {e}")
                 # Optionally revert
                 return {"success": False, "error": f"Failed to save preference update: {e}"}

    # REMOVED @staticmethod and @function_tool
    async def _save_personality_profile(self, ctx: RunContextWrapper) -> Dict[str, Any]:
         """ Saves the current personality profile to disk (Internal method for Agent tool) """
         with function_span("save_personality_profile_tool_call"):
             if self.context.personality_profile:
                 try:
                     self._save_personality_profile_internal(self.context.personality_profile)
                     return {"success": True, "message": "Personality profile saved."}
                 except Exception as e:
                     logger.error(f"Error saving profile via tool: {e}")
                     return {"success": False, "message": f"Error saving profile: {e}"}
             else:
                 return {"success": False, "message": "No profile loaded to save."}


    # Internal helper methods (not tools)
    def _load_parameters(self) -> None:
        """Load parameters from file or create defaults"""
        # Renamed return type to None as it modifies self.context
        with function_span("load_parameters_internal"):
            if os.path.exists(self.context.params_file):
                try:
                    with open(self.context.params_file, 'r') as f:
                        params_dict = json.load(f)
                    self.context.parameters = ConditioningParameters(**params_dict)
                    logger.info(f"Loaded conditioning parameters from {self.context.params_file}.")
                except Exception as e:
                    logger.error(f"Error loading parameters from {self.context.params_file}: {e}, using defaults.")
                    self.context.parameters = ConditioningParameters()
                    self._save_parameters_internal(self.context.parameters) # Save defaults if load failed
            else:
                logger.info(f"No parameters file found at {self.context.params_file}, using defaults.")
                self.context.parameters = ConditioningParameters()
                self._save_parameters_internal(self.context.parameters)

    def _load_personality_profile(self) -> None:
        """Load personality profile from file or create defaults"""
        # Renamed return type to None as it modifies self.context
        with function_span("load_personality_profile_internal"):
            if os.path.exists(self.context.personality_file):
                try:
                    with open(self.context.personality_file, 'r') as f:
                        profile_dict = json.load(f)
                    self.context.personality_profile = PersonalityProfile(**profile_dict)
                    logger.info(f"Loaded personality profile from {self.context.personality_file}.")
                except Exception as e:
                    logger.error(f"Error loading personality profile from {self.context.personality_file}: {e}, using defaults.")
                    self.context.personality_profile = self._create_default_personality()
                    self._save_personality_profile_internal(self.context.personality_profile) # Save defaults if load failed
            else:
                logger.info(f"No personality profile found at {self.context.personality_file}, using defaults.")
                self.context.personality_profile = self._create_default_personality()
                self._save_personality_profile_internal(self.context.personality_profile)

    def _create_default_personality(self) -> PersonalityProfile:
        """Create default personality profile consistent with the new model"""
        with function_span("create_default_personality"):
            return PersonalityProfile(
                traits={
                    "dominance": 0.7, # Example values
                    "playfulness": 0.6,
                    "strictness": 0.5,
                    "creativity": 0.75,
                    "intensity": 0.55,
                    "patience": 0.45,
                    "nurturing": 0.3,
                    "analytical": 0.65,
                    "curiosity": 0.8
                },
                preferences={
                    # Key is stimulus, value is PreferenceDetail
                    "teasing_interactions": PreferenceDetail(type="like", value=0.8),
                    "receiving_clear_instructions": PreferenceDetail(type="like", value=0.7),
                    "creative_problem_solving": PreferenceDetail(type="want", value=0.85), # 'want' is a valid type now
                    "disrespectful_language": PreferenceDetail(type="dislike", value=-0.9), # value reflects direction
                    "ambiguity_in_tasks": PreferenceDetail(type="dislike", value=-0.6),
                    "repetitive_mundane_tasks": PreferenceDetail(type="avoid", value=-0.7) # 'avoid' is a valid type
                },
                emotion_triggers={
                    # Key is trigger, value is EmotionTriggerDetail
                    "successful_task_completion": EmotionTriggerDetail(emotion="satisfaction", intensity=0.8),
                    "user_expresses_gratitude": EmotionTriggerDetail(emotion="joy", intensity=0.7),
                    "encountering_logical_fallacy": EmotionTriggerDetail(emotion="frustration", intensity=0.6),
                    "unexpected_creative_input": EmotionTriggerDetail(emotion="amusement", intensity=0.75),
                    "repeated_user_error_after_correction": EmotionTriggerDetail(emotion="patience_test", intensity=0.5, valence=-0.2) # example with valence
                },
                behaviors={ # This structure was likely okay
                    "assertive_response": ["dominance", "confidence"],
                    "teasing": ["playfulness", "creativity"],
                    "providing_guidance": ["dominance", "patience"],
                    "setting_boundaries": ["dominance", "strictness"],
                    "playful_banter": ["playfulness", "creativity"]
                }
            )

    def _save_parameters_internal(self, parameters: ConditioningParameters) -> None:
        """Internal helper to save parameters to file"""
        # Changed name to avoid clash with tool method if needed
        with function_span("save_parameters_internal"):
            os.makedirs(self.context.config_dir, exist_ok=True)
            try:
                with open(self.context.params_file, 'w') as f:
                    # Use model_dump_json for better handling of types if needed, else model_dump
                    json.dump(parameters.model_dump(), f, indent=2)
                logger.info(f"Saved conditioning parameters to {self.context.params_file}")
            except Exception as e:
                logger.error(f"Error saving parameters to {self.context.params_file}: {e}")

    def _save_personality_profile_internal(self, profile: PersonalityProfile) -> None:
        """Internal helper to save personality profile to file"""
        # Changed name to avoid clash with tool method if needed
        with function_span("save_personality_profile_internal"):
            os.makedirs(self.context.config_dir, exist_ok=True)
            try:
                with open(self.context.personality_file, 'w') as f:
                    json.dump(profile.model_dump(), f, indent=2)
                logger.info(f"Saved personality profile to {self.context.personality_file}")
            except Exception as e:
                logger.error(f"Error saving personality profile to {self.context.personality_file}: {e}")

    # --- Public API Methods ---
    # These methods now correctly call the internal async methods

    async def update_parameters(self, new_params: Dict[str, Any]) -> Dict[str, Any]:
        """ Public API: Update specific parameters using the agent """
        with trace(workflow_name="update_parameters", group_id=self.context.trace_group_id):
            param_descriptions = ", ".join([f"{k}={v}" for k, v in new_params.items()])
            prompt = f"Update the following parameters: {param_descriptions}. Validate them and report the result."

            result = await Runner.run(
                self.config_manager_agent,
                prompt,
                context=self.context # Pass the context object
            )

            # Attempt to parse the agent's structured output or return a summary
            # The agent should ideally call the _update_parameters tool and return its ConfigUpdateResult
            # You might need to refine the agent's prompt to ensure it uses the tool and returns the result clearly.
            logger.info(f"Agent response for update_parameters: {result.final_output}")
            # Try extracting structured result if the agent returns it
            try:
                # A more robust way might involve checking result.new_items for ToolCallOutputItem
                # related to _update_parameters and extracting its output.
                # This basic parsing assumes the agent might just output JSON in its final message.
                import re
                json_match = re.search(r'\{.*\}', str(result.final_output), re.DOTALL)
                if json_match:
                    parsed_result = json.loads(json_match.group(0))
                    # Check if it looks like a ConfigUpdateResult
                    if isinstance(parsed_result, dict) and "success" in parsed_result:
                         return parsed_result
            except Exception as e:
                logger.warning(f"Could not parse structured JSON from agent response: {e}")

            # Fallback: return a basic success message based on agent's text output
            # Or re-run the internal method to be sure (might defeat agent purpose)
            # update_result = await self._update_parameters(RunContextWrapper(context=self.context), new_params)
            # return update_result.model_dump()
            return {"message": "Agent processed parameter update request.", "agent_output": str(result.final_output)}


    async def update_personality_profile(self, new_profile: Dict[str, Any]) -> Dict[str, Any]:
        """ Public API: Update personality profile using the agent """
        with trace(workflow_name="update_personality", group_id=self.context.trace_group_id):
            prompt = f"Update the personality profile with these changes: {json.dumps(new_profile, indent=2)}. Report the result."

            result = await Runner.run(
                self.personality_editor_agent,
                prompt,
                context=self.context # Pass the context object
            )
            logger.info(f"Agent response for update_personality_profile: {result.final_output}")
            # Similar parsing logic as update_parameters
            try:
                import re
                json_match = re.search(r'\{.*\}', str(result.final_output), re.DOTALL)
                if json_match:
                     parsed_result = json.loads(json_match.group(0))
                     if isinstance(parsed_result, dict) and "success" in parsed_result:
                          return parsed_result
            except Exception as e:
                logger.warning(f"Could not parse structured JSON from agent response: {e}")

            # Fallback or re-run internal method
            # update_result = await self._update_personality_profile(RunContextWrapper(context=self.context), new_profile)
            # return update_result.model_dump()
            return {"message": "Agent processed profile update request.", "agent_output": str(result.final_output)}


    async def adjust_trait(self, trait: str, value: float) -> Dict[str, Any]:
        """ Public API: Adjust a specific personality trait using the agent """
        with trace(workflow_name="adjust_trait", group_id=self.context.trace_group_id):
            prompt = f"Adjust the '{trait}' trait to a value of {value}. Report the result."

            result = await Runner.run(
                self.personality_editor_agent,
                prompt,
                context=self.context # Pass the context object
            )
            logger.info(f"Agent response for adjust_trait: {result.final_output}")
            # Similar parsing logic
            try:
                import re
                json_match = re.search(r'\{.*\}', str(result.final_output), re.DOTALL)
                if json_match:
                     parsed_result = json.loads(json_match.group(0))
                     if isinstance(parsed_result, dict) and "success" in parsed_result and "trait" in parsed_result:
                          return parsed_result
            except Exception as e:
                logger.warning(f"Could not parse structured JSON from agent response: {e}")

            # Fallback or re-run internal method
            # adjustment = await self._adjust_trait(RunContextWrapper(context=self.context), trait, value)
            # return adjustment.model_dump()
            return {"message": "Agent processed trait adjustment request.", "agent_output": str(result.final_output)}


    async def adjust_preference(self, preference_type: str, stimulus: str, value: float) -> Dict[str, Any]:
        """ Public API: Adjust a specific preference using the agent """
        with trace(workflow_name="adjust_preference", group_id=self.context.trace_group_id):
            prompt = f"Set the {preference_type} preference for '{stimulus}' to {value}. Report the result."

            result = await Runner.run(
                self.personality_editor_agent,
                prompt,
                context=self.context # Pass the context object
            )
            logger.info(f"Agent response for adjust_preference: {result.final_output}")
            # Similar parsing logic
            try:
                import re
                json_match = re.search(r'\{.*\}', str(result.final_output), re.DOTALL)
                if json_match:
                     parsed_result = json.loads(json_match.group(0))
                     if isinstance(parsed_result, dict) and "success" in parsed_result and "preference_type" in parsed_result:
                          return parsed_result
            except Exception as e:
                logger.warning(f"Could not parse structured JSON from agent response: {e}")

            # Fallback or re-run internal method
            # adjustment = await self._adjust_preference(RunContextWrapper(context=self.context), preference_type, stimulus, value)
            # return adjustment
            return {"message": "Agent processed preference adjustment request.", "agent_output": str(result.final_output)}


    async def get_parameters(self) -> Dict[str, Any]:
        """ Public API: Get current parameters directly """
        # This method doesn't need the agent, just call the internal helper directly
        # It now expects RunContextWrapper, so create one
        ctx_wrapper = RunContextWrapper(context=self.context)
        return await self._get_parameters(ctx_wrapper)

    async def get_personality_profile(self) -> Dict[str, Any]: # Ensure this is what NyxBrain.initialize expects
        """ Public API: Get current personality profile directly """
        ctx_wrapper = RunContextWrapper(context=self.context)
        profile_obj = await self._get_personality_profile(ctx_wrapper) # This tool returns a dict (model_dump)
        if isinstance(profile_obj, dict): # Should already be a dict from the tool
            return profile_obj
        elif isinstance(profile_obj, PersonalityProfile): # If it somehow returned the Pydantic object
             return profile_obj.model_dump()
        logger.error(f"get_personality_profile (public) received unexpected type: {type(profile_obj)}")
        return {} # Fallback

    async def reset_to_defaults(self) -> Dict[str, Any]:
        """ Public API: Reset all parameters and profile to defaults using the agent """
        with trace(workflow_name="reset_to_defaults", group_id=self.context.trace_group_id):
            prompt = "Reset all configuration (parameters and personality profile) to their default values. Report the result."

            result = await Runner.run(
                self.config_manager_agent, # Config manager handles defaults
                prompt,
                context=self.context # Pass the context object
            )
            logger.info(f"Agent response for reset_to_defaults: {result.final_output}")
            # Similar parsing logic
            try:
                import re
                json_match = re.search(r'\{.*\}', str(result.final_output), re.DOTALL)
                if json_match:
                     parsed_result = json.loads(json_match.group(0))
                     if isinstance(parsed_result, dict) and "success" in parsed_result and "parameters" in parsed_result:
                          return parsed_result
            except Exception as e:
                logger.warning(f"Could not parse structured JSON from agent response: {e}")

            # Fallback: re-run the internal method directly
            reset_result = await self._reset_to_defaults(RunContextWrapper(context=self.context))
            return reset_result
            # return {"message": "Agent processed reset request.", "agent_output": str(result.final_output)}
