# nyx/core/conditioning_config.py

import json
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field

from agents import Agent, Runner, function_tool, RunContextWrapper, ModelSettings, trace
from agents.tracing import function_span

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
        
        # Initialize parameters and profile
        self.parameters = None
        self.personality_profile = None
        
        # Trace information
        self.trace_group_id = f"conditioning_config_{os.path.basename(config_dir)}"


class ConditioningConfiguration:
    """
    Configuration system for adjusting conditioning parameters using Agents SDK
    """
    
    def __init__(self, config_dir: str = "config"):
        self.context = ConfigurationContext(config_dir)
        
        # Initialize agents
        self.config_manager_agent = self._create_config_manager_agent()
        self.personality_editor_agent = self._create_personality_editor_agent()
        
        # Load configurations
        self._load_parameters()
        self._load_personality_profile()
        
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
                function_tool(self._get_parameters),
                function_tool(self._update_parameters),
                function_tool(self._save_parameters),
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
                function_tool(self._get_personality_profile),
                function_tool(self._update_personality_profile),
                function_tool(self._adjust_trait),
                function_tool(self._adjust_preference),
                function_tool(self._save_personality_profile)
            ],
            model_settings=ModelSettings(temperature=0.2)
        )

    @staticmethod  
    @function_tool
    async def _get_parameters(ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get current conditioning parameters
        
        Returns:
            Dictionary of current parameters
        """
        with function_span("get_parameters"):
            if not self.context.parameters:
                self._load_parameters()
            
            return self.context.parameters.model_dump() if self.context.parameters else {}

    @staticmethod  
    @function_tool
    async def _update_parameters(ctx: RunContextWrapper, 
                           new_params: Dict[str, Any]) -> ConfigUpdateResult:
        """
        Update specific conditioning parameters
        
        Args:
            new_params: Dictionary of parameters to update
            
        Returns:
            Result of the update operation
        """
        with function_span("update_parameters", input=str(new_params)):
            if not self.context.parameters:
                self._load_parameters()
            
            # Track changes
            result = ConfigUpdateResult(success=True)
            
            # Get current parameters
            current_params = self.context.parameters.model_dump()
            
            # Update with new parameters
            for key, value in new_params.items():
                if key in current_params:
                    result.previous_values[key] = current_params[key]
                    result.new_values[key] = value
                    result.updated_keys.append(key)
                else:
                    result.message += f"Unknown parameter: {key}. "
            
            if not result.updated_keys:
                result.success = False
                result.message = "No valid parameters provided for update."
                return result
            
            # Create new parameters object with updates
            updated_params = {**current_params, **{k: v for k, v in new_params.items() if k in current_params}}
            self.context.parameters = ConditioningParameters(**updated_params)
            
            # Save to file
            self._save_parameters(self.context.parameters)
            
            result.message += f"Updated {len(result.updated_keys)} parameters."
            return result

    @staticmethod  
    @function_tool
    async def _validate_parameters(ctx: RunContextWrapper, 
                             parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters against constraints
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Validation results with any issues
        """
        with function_span("validate_parameters"):
            validation_result = {
                "valid": True,
                "issues": {}
            }
            
            # Define constraints
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
            
            # Check each parameter against constraints
            for param, value in parameters.items():
                if param in constraints:
                    min_val, max_val = constraints[param]
                    if not min_val <= value <= max_val:
                        validation_result["valid"] = False
                        validation_result["issues"][param] = f"Value {value} outside range [{min_val}, {max_val}]"
            
            # Check relationships between thresholds
            if ("weak_association_threshold" in parameters and 
                "moderate_association_threshold" in parameters and
                "strong_association_threshold" in parameters):
                
                weak = parameters["weak_association_threshold"]
                moderate = parameters["moderate_association_threshold"]
                strong = parameters["strong_association_threshold"]
                
                if not (weak < moderate < strong):
                    validation_result["valid"] = False
                    validation_result["issues"]["threshold_ordering"] = (
                        f"Thresholds must be ordered: weak ({weak}) < moderate ({moderate}) < strong ({strong})"
                    )
            
            return validation_result

    @staticmethod  
    @function_tool
    async def _reset_to_defaults(ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Reset all parameters and profile to defaults
        
        Returns:
            Default parameters and profile
        """
        with function_span("reset_to_defaults"):
            # Reset parameters
            self.context.parameters = ConditioningParameters()
            self._save_parameters(self.context.parameters)
            
            # Reset personality profile
            self.context.personality_profile = self._create_default_personality()
            self._save_personality_profile(self.context.personality_profile)
            
            return {
                "parameters": self.context.parameters.model_dump(),
                "personality_profile": self.context.personality_profile.model_dump()
            }

    @staticmethod  
    @function_tool
    async def _get_personality_profile(ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get current personality profile
        
        Returns:
            Current personality profile data
        """
        with function_span("get_personality_profile"):
            if not self.context.personality_profile:
                self._load_personality_profile()
            
            return self.context.personality_profile.model_dump() if self.context.personality_profile else {}

    @staticmethod  
    @function_tool
    async def _update_personality_profile(ctx: RunContextWrapper,
                                    new_profile: Dict[str, Any]) -> ConfigUpdateResult:
        """
        Update personality profile
        
        Args:
            new_profile: Dictionary with profile updates
            
        Returns:
            Result of the update operation
        """
        with function_span("update_personality_profile", input=str(new_profile)):
            if not self.context.personality_profile:
                self._load_personality_profile()
            
            # Track changes
            result = ConfigUpdateResult(success=True)
            
            # Get current profile
            current_profile = self.context.personality_profile.model_dump()
            
            # Update profile, handling nested dictionaries
            for key, value in new_profile.items():
                if key in current_profile:
                    # For top-level keys
                    result.previous_values[key] = current_profile[key]
                    
                    if isinstance(value, dict) and isinstance(current_profile[key], dict):
                        # Handle nested dictionaries by merging
                        for subkey, subvalue in value.items():
                            if subkey in current_profile[key]:
                                # Track changes at the nested level
                                nested_key = f"{key}.{subkey}"
                                result.previous_values[nested_key] = current_profile[key][subkey]
                                result.new_values[nested_key] = subvalue
                                result.updated_keys.append(nested_key)
                            
                        # Update with merged dictionary
                        current_profile[key].update(value)
                        result.new_values[key] = current_profile[key]
                    else:
                        # Direct update for non-dict values
                        current_profile[key] = value
                        result.new_values[key] = value
                        result.updated_keys.append(key)
                else:
                    result.message += f"Unknown profile key: {key}. "
            
            if not result.updated_keys:
                result.success = False
                result.message = "No valid profile elements provided for update."
                return result
            
            # Create new profile object
            self.context.personality_profile = PersonalityProfile(**current_profile)
            
            # Save to file
            self._save_personality_profile(self.context.personality_profile)
            
            result.message += f"Updated {len(result.updated_keys)} profile elements."
            return result

    @staticmethod  
    @function_tool
    async def _adjust_trait(ctx: RunContextWrapper,
                      trait: str, 
                      value: float) -> TraitUpdateResult:
        """
        Adjust a specific personality trait
        
        Args:
            trait: The trait to adjust
            value: New value for the trait (0.0-1.0)
            
        Returns:
            Result of the trait adjustment
        """
        with function_span("adjust_trait", input=f"trait={trait}, value={value}"):
            if not self.context.personality_profile:
                self._load_personality_profile()
            
            # Get current traits
            traits = self.context.personality_profile.traits.copy()
            
            # Update trait
            old_value = traits.get(trait, 0.0)
            traits[trait] = max(0.0, min(1.0, value))  # Constrain to 0-1
            
            # Update personality profile
            updated_profile = self.context.personality_profile.model_dump()
            updated_profile["traits"] = traits
            self.context.personality_profile = PersonalityProfile(**updated_profile)
            
            # Save to file
            self._save_personality_profile(self.context.personality_profile)
            
            return TraitUpdateResult(
                trait=trait,
                old_value=old_value,
                new_value=traits[trait]
            )

    @staticmethod  
    @function_tool
    async def _adjust_preference(ctx: RunContextWrapper,
                           preference_type: str, 
                           stimulus: str, 
                           value: float) -> Dict[str, Any]:
        """
        Adjust a specific preference
        
        Args:
            preference_type: Type of preference ("likes" or "dislikes")
            stimulus: The stimulus to adjust preference for
            value: New value for the preference (0.0-1.0)
            
        Returns:
            Result of the preference adjustment
        """
        with function_span("adjust_preference"):
            if not self.context.personality_profile:
                self._load_personality_profile()
            
            # Check preference type
            if preference_type not in ["likes", "dislikes"]:
                return {
                    "success": False,
                    "error": f"Invalid preference type: {preference_type}",
                    "valid_types": ["likes", "dislikes"]
                }
            
            # Get current preferences
            preferences = self.context.personality_profile.preferences.copy()
            
            # Get old value
            old_value = preferences.get(preference_type, {}).get(stimulus, 0.0)
            
            # Update preference
            if preference_type not in preferences:
                preferences[preference_type] = {}
            
            value = max(0.0, min(1.0, value))  # Constrain to 0-1
            preferences[preference_type][stimulus] = value
            
            # Update personality profile
            updated_profile = self.context.personality_profile.model_dump()
            updated_profile["preferences"] = preferences
            self.context.personality_profile = PersonalityProfile(**updated_profile)
            
            # Save to file
            self._save_personality_profile(self.context.personality_profile)
            
            return {
                "success": True,
                "preference_type": preference_type,
                "stimulus": stimulus,
                "old_value": old_value,
                "new_value": value
            }
    
    def _load_parameters(self) -> ConditioningParameters:
        """Load parameters from file or create defaults"""
        with function_span("load_parameters"):
            if os.path.exists(self.context.params_file):
                try:
                    with open(self.context.params_file, 'r') as f:
                        params_dict = json.load(f)
                    self.context.parameters = ConditioningParameters(**params_dict)
                    logger.info("Loaded conditioning parameters from file.")
                except Exception as e:
                    logger.error(f"Error loading parameters: {e}, using defaults")
                    self.context.parameters = ConditioningParameters()
            else:
                # Create default parameters
                logger.info("No parameters file found, using defaults.")
                self.context.parameters = ConditioningParameters()
                self._save_parameters(self.context.parameters)
            
            return self.context.parameters
    
    def _load_personality_profile(self) -> PersonalityProfile:
        """Load personality profile from file or create defaults"""
        with function_span("load_personality_profile"):
            if os.path.exists(self.context.personality_file):
                try:
                    with open(self.context.personality_file, 'r') as f:
                        profile_dict = json.load(f)
                    self.context.personality_profile = PersonalityProfile(**profile_dict)
                    logger.info("Loaded personality profile from file.")
                except Exception as e:
                    logger.error(f"Error loading personality profile: {e}, using defaults")
                    self.context.personality_profile = self._create_default_personality()
            else:
                # Create default personality
                logger.info("No personality profile found, using defaults.")
                self.context.personality_profile = self._create_default_personality()
                self._save_personality_profile(self.context.personality_profile)
            
            return self.context.personality_profile
    
    def _create_default_personality(self) -> PersonalityProfile:
        """Create default personality profile"""
        with function_span("create_default_personality"):
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
        with function_span("save_parameters"):
            # Create directory if it doesn't exist
            os.makedirs(self.context.config_dir, exist_ok=True)
            
            try:
                with open(self.context.params_file, 'w') as f:
                    json.dump(parameters.model_dump(), f, indent=2)
                logger.info("Saved conditioning parameters")
            except Exception as e:
                logger.error(f"Error saving parameters: {e}")
    
    def _save_personality_profile(self, profile: PersonalityProfile) -> None:
        """Save personality profile to file"""
        with function_span("save_personality_profile"):
            # Create directory if it doesn't exist
            os.makedirs(self.context.config_dir, exist_ok=True)
            
            try:
                with open(self.context.personality_file, 'w') as f:
                    json.dump(profile.model_dump(), f, indent=2)
                logger.info("Saved personality profile")
            except Exception as e:
                logger.error(f"Error saving personality profile: {e}")
    
    async def update_parameters(self, new_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Public API: Update specific parameters
        
        Args:
            new_params: Dictionary of parameters to update
            
        Returns:
            Update results
        """
        with trace(workflow_name="update_parameters", group_id=self.context.trace_group_id):
            # Construct a prompt asking the agent to update the parameters
            param_descriptions = ", ".join([f"{k}={v}" for k, v in new_params.items()])
            prompt = f"Update the following parameters: {param_descriptions}"
            
            # Run the config manager agent
            result = await Runner.run(
                self.config_manager_agent,
                prompt,
                context=self.context
            )
            
            # Extract the results from the agent's response
            try:
                # Try to parse JSON response if available
                import re
                json_match = re.search(r'\{.*\}', result.final_output, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                else:
                    # Get the updates directly
                    update_result = await self._update_parameters(RunContextWrapper(context=self.context), new_params)
                    return update_result.model_dump()
            except Exception as e:
                logger.error(f"Error processing update result: {e}")
                # Return a simple result
                return {
                    "success": True,
                    "updated": list(new_params.keys())
                }
    
    async def update_personality_profile(self, new_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Public API: Update personality profile
        
        Args:
            new_profile: Dictionary with profile updates
            
        Returns:
            Update results
        """
        with trace(workflow_name="update_personality", group_id=self.context.trace_group_id):
            # Construct a prompt explaining the profile updates
            prompt = f"Update the personality profile with these changes: {json.dumps(new_profile, indent=2)}"
            
            # Run the personality editor agent
            result = await Runner.run(
                self.personality_editor_agent,
                prompt,
                context=self.context
            )
            
            # Extract the results from the agent's response
            try:
                # Try to parse JSON response if available
                import re
                json_match = re.search(r'\{.*\}', result.final_output, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                else:
                    # Get the updates directly
                    update_result = await self._update_personality_profile(RunContextWrapper(context=self.context), new_profile)
                    return update_result.model_dump()
            except Exception as e:
                logger.error(f"Error processing personality update: {e}")
                # Return a simple result
                return {
                    "success": True,
                    "updated": list(new_profile.keys())
                }
    
    async def adjust_trait(self, trait: str, value: float) -> Dict[str, Any]:
        """
        Public API: Adjust a specific personality trait
        
        Args:
            trait: The trait to adjust
            value: New value for the trait (0.0-1.0)
            
        Returns:
            Trait adjustment results
        """
        with trace(workflow_name="adjust_trait", group_id=self.context.trace_group_id):
            # Construct a prompt for trait adjustment
            prompt = f"Adjust the '{trait}' trait to a value of {value}"
            
            # Run the personality editor agent
            result = await Runner.run(
                self.personality_editor_agent,
                prompt,
                context=self.context
            )
            
            # Extract and return results
            try:
                import re
                json_match = re.search(r'\{.*\}', result.final_output, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                else:
                    # Get the updates directly
                    adjustment = await self._adjust_trait(RunContextWrapper(context=self.context), trait, value)
                    return adjustment.model_dump()
            except Exception as e:
                logger.error(f"Error processing trait adjustment: {e}")
                # Return direct result
                return {
                    "trait": trait,
                    "old_value": self.context.personality_profile.traits.get(trait, 0.0),
                    "new_value": value
                }
    
    async def adjust_preference(self, preference_type: str, stimulus: str, value: float) -> Dict[str, Any]:
        """
        Public API: Adjust a specific preference
        
        Args:
            preference_type: Type of preference ("likes" or "dislikes")
            stimulus: The stimulus to adjust preference for
            value: New value for the preference (0.0-1.0)
            
        Returns:
            Preference adjustment results
        """
        with trace(workflow_name="adjust_preference", group_id=self.context.trace_group_id):
            # Construct a prompt for preference adjustment
            prompt = f"Set the {preference_type} preference for '{stimulus}' to {value}"
            
            # Run the personality editor agent
            result = await Runner.run(
                self.personality_editor_agent,
                prompt,
                context=self.context
            )
            
            # Extract and return results
            try:
                import re
                json_match = re.search(r'\{.*\}', result.final_output, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                else:
                    # Get the updates directly
                    return await self._adjust_preference(RunContextWrapper(context=self.context), 
                                                         preference_type, stimulus, value)
            except Exception as e:
                logger.error(f"Error processing preference adjustment: {e}")
                # Return direct result
                return {
                    "preference_type": preference_type,
                    "stimulus": stimulus,
                    "new_value": value
                }
    
    async def get_parameters(self) -> Dict[str, Any]:
        """
        Public API: Get current parameters
        
        Returns:
            Current parameters
        """
        return await self._get_parameters(RunContextWrapper(context=self.context))
    
    async def get_personality_profile(self) -> Dict[str, Any]:
        """
        Public API: Get current personality profile
        
        Returns:
            Current personality profile
        """
        return await self._get_personality_profile(RunContextWrapper(context=self.context))
    
    async def reset_to_defaults(self) -> Dict[str, Any]:
        """
        Public API: Reset all parameters and profile to defaults
        
        Returns:
            Default parameters and profile
        """
        with trace(workflow_name="reset_to_defaults", group_id=self.context.trace_group_id):
            prompt = "Reset all configuration to default values"
            
            # Run the config manager agent
            result = await Runner.run(
                self.config_manager_agent,
                prompt,
                context=self.context
            )
            
            # Return the default values
            return await self._reset_to_defaults(RunContextWrapper(context=self.context))
