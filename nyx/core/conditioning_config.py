# nyx/core/conditioning_config.py

import json
import logging
import os
from typing import Dict, List, Any, Optional

from agents import Agent, Runner, ModelSettings, trace
from nyx.core.conditioning_models import *
from nyx.core.conditioning_tools import check_trait_balance

logger = logging.getLogger(__name__)

# Configuration-specific tools

@function_tool
async def get_parameters(ctx: RunContextWrapper) -> Dict[str, Any]:
    """Get current conditioning parameters"""
    if ctx.context.parameters:
        return ctx.context.parameters.model_dump()
    return {}

@function_tool
async def update_parameters(ctx: RunContextWrapper, new_params: Dict[str, Any]) -> Dict[str, Any]:
    """Update conditioning parameters"""
    if not ctx.context.parameters:
        return {"success": False, "message": "Parameters not loaded"}
    
    result = {"success": True, "updated_keys": [], "previous_values": {}, "new_values": {}}
    current_params = ctx.context.parameters.model_dump()
    
    for key, value in new_params.items():
        if hasattr(ctx.context.parameters, key):
            result["previous_values"][key] = current_params[key]
            result["new_values"][key] = value
            result["updated_keys"].append(key)
            setattr(ctx.context.parameters, key, value)
    
    return result

@function_tool
async def save_parameters(ctx: RunContextWrapper) -> Dict[str, Any]:
    """Save current parameters to disk"""
    try:
        os.makedirs(ctx.context.config_dir, exist_ok=True)
        with open(ctx.context.params_file, 'w') as f:
            json.dump(ctx.context.parameters.model_dump(), f, indent=2)
        return {"success": True, "message": "Parameters saved"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@function_tool
async def validate_parameters(ctx: RunContextWrapper, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Validate parameters against constraints"""
    validation_result = {"valid": True, "issues": {}}
    
    try:
        ConditioningParameters(**parameters)
        
        # Range checks
        constraints = {
            "association_learning_rate": (0.0, 1.0),
            "extinction_rate": (0.0, 0.5),
            "generalization_factor": (0.0, 1.0)
        }
        
        for param, value in parameters.items():
            if param in constraints:
                min_val, max_val = constraints[param]
                if not min_val <= value <= max_val:
                    validation_result["valid"] = False
                    validation_result["issues"][param] = f"Value outside range [{min_val}, {max_val}]"
    
    except Exception as e:
        validation_result["valid"] = False
        validation_result["issues"]["validation_error"] = str(e)
    
    return validation_result

@function_tool
async def reset_to_defaults(ctx: RunContextWrapper) -> Dict[str, Any]:
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
        
        return {
            "success": True,
            "message": "Reset to defaults",
            "parameters": ctx.context.parameters.model_dump(),
            "personality_profile": ctx.context.personality_profile.model_dump()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@function_tool
async def get_personality_profile(ctx: RunContextWrapper) -> Dict[str, Any]:
    """Get current personality profile"""
    if ctx.context.personality_profile:
        return ctx.context.personality_profile.model_dump()
    return {}

@function_tool
async def update_personality_profile(ctx: RunContextWrapper, new_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Update personality profile"""
    if not ctx.context.personality_profile:
        return {"success": False, "message": "Profile not loaded"}
    
    result = {"success": True, "updated_keys": []}
    
    try:
        current_dict = ctx.context.personality_profile.model_dump()
        current_dict.update(new_profile)
        ctx.context.personality_profile = PersonalityProfile(**current_dict)
        result["updated_keys"] = list(new_profile.keys())
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

@function_tool
async def adjust_trait(ctx: RunContextWrapper, trait: str, value: float) -> Dict[str, Any]:
    """Adjust a specific personality trait"""
    if not ctx.context.personality_profile:
        return {"success": False, "error": "Profile not loaded"}
    
    old_value = ctx.context.personality_profile.traits.get(trait, 0.0)
    new_value = max(0.0, min(1.0, value))
    ctx.context.personality_profile.traits[trait] = new_value
    
    return {
        "success": True,
        "trait": trait,
        "old_value": old_value,
        "new_value": new_value
    }

@function_tool
async def adjust_preference(ctx: RunContextWrapper, stimulus: str, preference_type: str, value: float) -> Dict[str, Any]:
    """Adjust a preference"""
    if not ctx.context.personality_profile:
        return {"success": False, "error": "Profile not loaded"}
    
    ctx.context.personality_profile.preferences[stimulus] = PreferenceDetail(
        type=preference_type,
        value=value
    )
    
    return {
        "success": True,
        "stimulus": stimulus,
        "preference_type": preference_type,
        "value": value
    }

@function_tool
async def save_personality_profile(ctx: RunContextWrapper) -> Dict[str, Any]:
    """Save personality profile to disk"""
    try:
        os.makedirs(ctx.context.config_dir, exist_ok=True)
        with open(ctx.context.personality_file, 'w') as f:
            json.dump(ctx.context.personality_profile.model_dump(), f, indent=2)
        return {"success": True, "message": "Profile saved"}
    except Exception as e:
        return {"success": False, "error": str(e)}

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
            prompt = f"Update these parameters: {json.dumps(new_params)}"
            
            result = await Runner.run(
                self.config_manager_agent,
                prompt,
                context=self.context
            )
            
            return {"success": True, "message": str(result.final_output)}
    
    async def update_personality_profile(self, new_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Update personality profile"""
        with trace(workflow_name="update_personality", group_id=self.context.trace_group_id):
            prompt = f"Update the personality profile: {json.dumps(new_profile)}"
            
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
