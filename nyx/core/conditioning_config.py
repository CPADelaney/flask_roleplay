# nyx/core/conditioning_config.py

import json
import logging
import os
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from agents import Agent, Runner, ModelSettings, trace, function_tool, FunctionTool, RunContextWrapper, handoff
from nyx.core.conditioning_models import *
from nyx.core.conditioning_tools import check_trait_balance

logger = logging.getLogger(__name__)

# Pydantic models for function parameters and returns
class ParametersResult(BaseModel, extra="forbid"):
    """Result of getting parameters"""
    association_learning_rate: float = 0.01
    extinction_rate: float = 0.005
    generalization_factor: float = 0.3
    context_weight: float = 0.4
    temporal_decay_rate: float = 0.02
    max_association_strength: float = 1.0
    min_association_strength: float = 0.0
    reinforcement_threshold: float = 0.3
    punishment_modifier: float = 0.8
    # Add other fields as they appear in ConditioningParameters

class UpdateParametersParams(BaseModel, extra="forbid"):
    """Parameters for updating conditioning parameters"""
    association_learning_rate: Optional[float] = None
    extinction_rate: Optional[float] = None
    generalization_factor: Optional[float] = None
    context_weight: Optional[float] = None
    temporal_decay_rate: Optional[float] = None
    max_association_strength: Optional[float] = None
    min_association_strength: Optional[float] = None
    reinforcement_threshold: Optional[float] = None
    punishment_modifier: Optional[float] = None

class UpdateParametersResult(BaseModel, extra="forbid"):
    """Result of updating parameters"""
    success: bool
    updated_keys: List[str] = Field(default_factory=list)
    previous_values: Dict[str, float] = Field(default_factory=dict)
    new_values: Dict[str, float] = Field(default_factory=dict)
    message: Optional[str] = None

class SaveResult(BaseModel, extra="forbid"):
    """Result of save operation"""
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None

class ValidateParametersResult(BaseModel, extra="forbid"):
    """Result of parameter validation"""
    valid: bool
    issues: Dict[str, str] = Field(default_factory=dict)

class ResetResult(BaseModel, extra="forbid"):
    """Result of reset operation"""
    success: bool
    message: str
    parameters: ParametersResult
    personality_profile: 'PersonalityProfileResult'
    error: Optional[str] = None

class PersonalityProfileResult(BaseModel, extra="forbid"):
    """Result of getting personality profile"""
    traits: Dict[str, float] = Field(default_factory=dict)
    preferences: Dict[str, 'PreferenceDetailResult'] = Field(default_factory=dict)
    emotion_triggers: Dict[str, 'EmotionTriggerDetailResult'] = Field(default_factory=dict)
    behaviors: Dict[str, List[str]] = Field(default_factory=dict)

class PreferenceDetailResult(BaseModel, extra="forbid"):
    """Preference detail result"""
    type: str
    value: float

class EmotionTriggerDetailResult(BaseModel, extra="forbid"):
    """Emotion trigger detail result"""
    emotion: str
    intensity: float

class UpdatePersonalityProfileParams(BaseModel, extra="forbid"):
    """Parameters for updating personality profile"""
    traits: Optional[Dict[str, float]] = None
    preferences: Optional[Dict[str, PreferenceDetail]] = None
    emotion_triggers: Optional[Dict[str, EmotionTriggerDetail]] = None
    behaviors: Optional[Dict[str, List[str]]] = None

class UpdatePersonalityResult(BaseModel, extra="forbid"):
    """Result of updating personality profile"""
    success: bool
    updated_keys: List[str] = Field(default_factory=list)
    message: Optional[str] = None
    error: Optional[str] = None

class AdjustTraitParams(BaseModel, extra="forbid"):
    """Parameters for adjusting a trait"""
    trait: str
    value: float

class AdjustTraitResult(BaseModel, extra="forbid"):
    """Result of adjusting a trait"""
    success: bool
    trait: str
    old_value: float
    new_value: float
    error: Optional[str] = None

class AdjustPreferenceParams(BaseModel, extra="forbid"):
    """Parameters for adjusting a preference"""
    stimulus: str
    preference_type: str
    value: float

class AdjustPreferenceResult(BaseModel, extra="forbid"):
    """Result of adjusting a preference"""
    success: bool
    stimulus: str
    preference_type: str
    value: float
    error: Optional[str] = None

# Configuration-specific tools

@function_tool
async def get_parameters(ctx: RunContextWrapper) -> ParametersResult:
    """Get current conditioning parameters"""
    if ctx.context.parameters:
        params_dict = ctx.context.parameters.model_dump()
        return ParametersResult(**params_dict)
    return ParametersResult()

@function_tool
async def update_parameters(ctx: RunContextWrapper, params: UpdateParametersParams) -> UpdateParametersResult:
    """Update conditioning parameters"""
    if not ctx.context.parameters:
        return UpdateParametersResult(success=False, message="Parameters not loaded")
    
    result = UpdateParametersResult(success=True)
    current_params = ctx.context.parameters.model_dump()
    
    # Update only the provided fields
    update_dict = params.model_dump(exclude_unset=True)
    
    for key, value in update_dict.items():
        if hasattr(ctx.context.parameters, key):
            result.previous_values[key] = current_params[key]
            result.new_values[key] = value
            result.updated_keys.append(key)
            setattr(ctx.context.parameters, key, value)
    
    return result

@function_tool
async def save_parameters(ctx: RunContextWrapper) -> SaveResult:
    """Save current parameters to disk"""
    try:
        os.makedirs(ctx.context.config_dir, exist_ok=True)
        with open(ctx.context.params_file, 'w') as f:
            json.dump(ctx.context.parameters.model_dump(), f, indent=2)
        return SaveResult(success=True, message="Parameters saved")
    except Exception as e:
        return SaveResult(success=False, error=str(e))

@function_tool
async def validate_parameters(ctx: RunContextWrapper, parameters: ParametersResult) -> ValidateParametersResult:
    """Validate parameters against constraints"""
    validation_result = ValidateParametersResult(valid=True)
    
    try:
        # Convert back to ConditioningParameters to validate
        params_dict = parameters.model_dump()
        ConditioningParameters(**params_dict)
        
        # Range checks
        constraints = {
            "association_learning_rate": (0.0, 1.0),
            "extinction_rate": (0.0, 0.5),
            "generalization_factor": (0.0, 1.0)
        }
        
        for param, value in params_dict.items():
            if param in constraints:
                min_val, max_val = constraints[param]
                if not min_val <= value <= max_val:
                    validation_result.valid = False
                    validation_result.issues[param] = f"Value outside range [{min_val}, {max_val}]"
    
    except Exception as e:
        validation_result.valid = False
        validation_result.issues["validation_error"] = str(e)
    
    return validation_result

@function_tool
async def reset_to_defaults(ctx: RunContextWrapper) -> ResetResult:
    """Reset all parameters and profile to defaults"""
    try:
        ctx.context.parameters = ConditioningParameters()
        ctx.context.personality_profile = PersonalityProfile(
            traits={"dominance": 0.7, "playfulness": 0.6, "strictness": 0.5},
            preferences={},
            emotion_triggers={},
            behaviors={}
        )
        
        # Save both
        await save_parameters(ctx)
        await save_personality_profile(ctx)
        
        # Convert to result models
        params_result = ParametersResult(**ctx.context.parameters.model_dump())
        profile_result = PersonalityProfileResult(
            traits=ctx.context.personality_profile.traits,
            preferences={k: PreferenceDetailResult(type=v.type, value=v.value) 
                        for k, v in ctx.context.personality_profile.preferences.items()},
            emotion_triggers={k: EmotionTriggerDetailResult(emotion=v.emotion, intensity=v.intensity)
                            for k, v in ctx.context.personality_profile.emotion_triggers.items()},
            behaviors=ctx.context.personality_profile.behaviors
        )
        
        return ResetResult(
            success=True,
            message="Reset to defaults",
            parameters=params_result,
            personality_profile=profile_result
        )
    except Exception as e:
        return ResetResult(
            success=False,
            message="",
            parameters=ParametersResult(),
            personality_profile=PersonalityProfileResult(),
            error=str(e)
        )

@function_tool
async def get_personality_profile(ctx: RunContextWrapper) -> PersonalityProfileResult:
    """Get current personality profile"""
    if ctx.context.personality_profile:
        profile = ctx.context.personality_profile
        return PersonalityProfileResult(
            traits=profile.traits,
            preferences={k: PreferenceDetailResult(type=v.type, value=v.value) 
                        for k, v in profile.preferences.items()},
            emotion_triggers={k: EmotionTriggerDetailResult(emotion=v.emotion, intensity=v.intensity)
                            for k, v in profile.emotion_triggers.items()},
            behaviors=profile.behaviors
        )
    return PersonalityProfileResult()

@function_tool
async def update_personality_profile(ctx: RunContextWrapper, params: UpdatePersonalityProfileParams) -> UpdatePersonalityResult:
    """Update personality profile"""
    if not ctx.context.personality_profile:
        return UpdatePersonalityResult(success=False, message="Profile not loaded")
    
    result = UpdatePersonalityResult(success=True)
    
    try:
        current_dict = ctx.context.personality_profile.model_dump()
        update_dict = params.model_dump(exclude_unset=True)
        
        # Update the current dict with new values
        for key, value in update_dict.items():
            if value is not None:
                current_dict[key] = value
                result.updated_keys.append(key)
        
        ctx.context.personality_profile = PersonalityProfile(**current_dict)
        return result
    except Exception as e:
        return UpdatePersonalityResult(success=False, error=str(e))

@function_tool
async def adjust_trait(ctx: RunContextWrapper, params: AdjustTraitParams) -> AdjustTraitResult:
    """Adjust a specific personality trait"""
    if not ctx.context.personality_profile:
        return AdjustTraitResult(
            success=False, 
            trait=params.trait, 
            old_value=0.0, 
            new_value=0.0,
            error="Profile not loaded"
        )
    
    old_value = ctx.context.personality_profile.traits.get(params.trait, 0.0)
    new_value = max(0.0, min(1.0, params.value))
    ctx.context.personality_profile.traits[params.trait] = new_value
    
    return AdjustTraitResult(
        success=True,
        trait=params.trait,
        old_value=old_value,
        new_value=new_value
    )

@function_tool
async def adjust_preference(ctx: RunContextWrapper, params: AdjustPreferenceParams) -> AdjustPreferenceResult:
    """Adjust a preference"""
    if not ctx.context.personality_profile:
        return AdjustPreferenceResult(
            success=False,
            stimulus=params.stimulus,
            preference_type=params.preference_type,
            value=params.value,
            error="Profile not loaded"
        )
    
    ctx.context.personality_profile.preferences[params.stimulus] = PreferenceDetail(
        type=params.preference_type,
        value=params.value
    )
    
    return AdjustPreferenceResult(
        success=True,
        stimulus=params.stimulus,
        preference_type=params.preference_type,
        value=params.value
    )

@function_tool
async def save_personality_profile(ctx: RunContextWrapper) -> SaveResult:
    """Save personality profile to disk"""
    try:
        os.makedirs(ctx.context.config_dir, exist_ok=True)
        with open(ctx.context.personality_file, 'w') as f:
            json.dump(ctx.context.personality_profile.model_dump(), f, indent=2)
        return SaveResult(success=True, message="Profile saved")
    except Exception as e:
        return SaveResult(success=False, error=str(e))

class ConditioningConfiguration:
    """Configuration system for adjusting conditioning parameters"""
    
    def __init__(self, config_dir: str = "config"):
        self.context = ConfigurationContext(config_dir)
        self._load_configurations()
        self._create_agents()
        logger.info("Conditioning configuration initialized")
    
    def _load_configurations(self):
        """Load parameters and personality profile"""
        # Load parameters
        if os.path.exists(self.context.params_file):
            try:
                with open(self.context.params_file, 'r') as f:
                    params_dict = json.load(f)
                self.context.parameters = ConditioningParameters(**params_dict)
            except Exception as e:
                logger.error(f"Error loading parameters: {e}")
                self.context.parameters = ConditioningParameters()
        else:
            self.context.parameters = ConditioningParameters()
        
        # Load personality profile
        if os.path.exists(self.context.personality_file):
            try:
                with open(self.context.personality_file, 'r') as f:
                    profile_dict = json.load(f)
                self.context.personality_profile = PersonalityProfile(**profile_dict)
            except Exception as e:
                logger.error(f"Error loading profile: {e}")
                self.context.personality_profile = self._create_default_personality()
        else:
            self.context.personality_profile = self._create_default_personality()
    
    def _create_default_personality(self) -> PersonalityProfile:
        """Create default personality profile"""
        return PersonalityProfile(
            traits={
                "dominance": 0.7,
                "playfulness": 0.6,
                "strictness": 0.5,
                "creativity": 0.75,
                "intensity": 0.55
            },
            preferences={
                "teasing_interactions": PreferenceDetail(type="like", value=0.8),
                "creative_problem_solving": PreferenceDetail(type="want", value=0.85)
            },
            emotion_triggers={
                "successful_task_completion": EmotionTriggerDetail(emotion="satisfaction", intensity=0.8),
                "user_expresses_gratitude": EmotionTriggerDetail(emotion="joy", intensity=0.7)
            },
            behaviors={
                "assertive_response": ["dominance", "confidence"],
                "teasing": ["playfulness", "creativity"]
            }
        )
    
    def _create_agents(self):
        """Create configuration management agents"""
        
        self.config_manager_agent = Agent(
            name="Config_Manager",
            instructions="""
            You manage conditioning system parameters.
            Ensure values remain within valid ranges and save changes.
            Use the tools to get, update, validate, and save parameters.
            """,
            tools=[
                get_parameters,
                update_parameters,
                save_parameters,
                validate_parameters,
                reset_to_defaults
            ],
            model_settings=ModelSettings(temperature=0.1)
        )
        
        self.personality_editor_agent = Agent(
            name="Personality_Editor",
            instructions="""
            You manage personality profiles including traits, preferences, and triggers.
            Use the tools to get, update, and save personality data.
            When checking trait balance, pass traits dictionary to check_trait_balance.
            """,
            tools=[
                get_personality_profile,
                update_personality_profile,
                adjust_trait,
                adjust_preference,
                save_personality_profile,
                check_trait_balance
            ],
            model_settings=ModelSettings(temperature=0.2)
        )
    
    # Public API methods
    
    async def update_parameters(self, new_params: Dict[str, Any]) -> Dict[str, Any]:
        """Update specific parameters"""
        with trace(workflow_name="update_parameters", group_id=self.context.trace_group_id):
            # Convert dict to UpdateParametersParams
            params_obj = UpdateParametersParams(**new_params)
            prompt = f"Update these parameters: {params_obj.model_dump_json()}"
            
            result = await Runner.run(
                self.config_manager_agent,
                prompt,
                context=self.context
            )
            
            return {"success": True, "message": str(result.final_output)}
    
    async def update_personality_profile(self, new_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Update personality profile"""
        with trace(workflow_name="update_personality", group_id=self.context.trace_group_id):
            # Convert dict to UpdatePersonalityProfileParams
            params_obj = UpdatePersonalityProfileParams(**new_profile)
            prompt = f"Update the personality profile: {params_obj.model_dump_json()}"
            
            result = await Runner.run(
                self.personality_editor_agent,
                prompt,
                context=self.context
            )
            
            return {"success": True, "message": str(result.final_output)}
    
    async def adjust_trait(self, trait: str, value: float) -> Dict[str, Any]:
        """Adjust a specific personality trait"""
        with trace(workflow_name="adjust_trait", group_id=self.context.trace_group_id):
            prompt = f"Adjust trait '{trait}' to value {value}"
            
            result = await Runner.run(
                self.personality_editor_agent,
                prompt,
                context=self.context
            )
            
            return {"success": True, "message": str(result.final_output)}
    
    async def adjust_preference(self, preference_type: str, stimulus: str, value: float) -> Dict[str, Any]:
        """Adjust a preference"""
        with trace(workflow_name="adjust_preference", group_id=self.context.trace_group_id):
            prompt = f"Set {preference_type} preference for '{stimulus}' to {value}"
            
            result = await Runner.run(
                self.personality_editor_agent,
                prompt,
                context=self.context
            )
            
            return {"success": True, "message": str(result.final_output)}
    
    async def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters directly"""
        return self.context.parameters.model_dump() if self.context.parameters else {}
    
    async def get_personality_profile(self) -> Dict[str, Any]:
        """Get current personality profile directly"""
        return self.context.personality_profile.model_dump() if self.context.personality_profile else {}
    
    async def reset_to_defaults(self) -> Dict[str, Any]:
        """Reset all to defaults"""
        with trace(workflow_name="reset_to_defaults", group_id=self.context.trace_group_id):
            prompt = "Reset all configuration to defaults"
            
            result = await Runner.run(
                self.config_manager_agent,
                prompt,
                context=self.context
            )
            
            return {"success": True, "message": str(result.final_output)}

# Update model forward references
PersonalityProfileResult.model_rebuild()
ResetResult.model_rebuild()
