# nyx/core/digital_somatosensory_system.py

import logging
import asyncio
import datetime
import json
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Literal
from dataclasses import dataclass, field
import numpy as np

from agents import (
    Agent, Runner, trace, function_tool, 
    RunContextWrapper, handoff, ModelSettings,
    InputGuardrail, GuardrailFunctionOutput, 
    Handoff, RunConfig, FunctionTool, custom_span,
    RunHooks, AgentHooks
)
from pydantic import BaseModel, Field

from nyx.core.reward_system import RewardSignal

logger = logging.getLogger(__name__)

# =============== Models for Structured Output ===============

class BodyRegion(BaseModel):
    """Representation of a body region with sensory data"""
    name: str = Field(..., description="Name of body region")
    pressure: float = Field(0.0, description="Pressure sensation (0.0-1.0)")
    temperature: float = Field(0.5, description="Temperature sensation (0.0=cold, 0.5=neutral, 1.0=hot)")
    pain: float = Field(0.0, description="Pain sensation (0.0-1.0)")
    pleasure: float = Field(0.0, description="Pleasure sensation (0.0-1.0)")
    tingling: float = Field(0.0, description="Tingling sensation (0.0-1.0)")
    last_update: Optional[datetime.datetime] = Field(None, description="Last sensation update time")
    sensation_memory: List[Dict[str, Any]] = Field(default_factory=list, description="Memory of past sensations")
    sensitivity: float = Field(1.0, description="Base sensitivity multiplier for this region")
    erogenous_level: float = Field(0.0, ge=0.0, le=1.0, description="Degree to which region is erogenous")

class PainMemory(BaseModel):
    """Memory of a pain experience"""
    intensity: float = Field(..., description="Pain intensity (0.0-1.0)")
    location: str = Field(..., description="Body region experiencing pain")
    cause: str = Field(..., description="Cause of the pain")
    duration: float = Field(0.0, description="Duration of pain in seconds")
    timestamp: datetime.datetime = Field(..., description="When pain occurred")
    associated_memory_id: Optional[str] = Field(None, description="Associated memory ID if available")

class SensoryExpression(BaseModel):
    """Sensory expression output format"""
    expression_text: str = Field(..., description="Expression of sensory experience")
    primary_region: str = Field(..., description="Primary body region involved")
    sensory_type: str = Field(..., description="Type of sensation (temperature, pain, etc.)")
    intensity: float = Field(..., description="Sensation intensity (0.0-1.0)")
    behavioral_effect: Optional[str] = Field(None, description="How this affects behavior/response")

class BodyStateOutput(BaseModel):
    """Complete body state output"""
    dominant_sensation: str = Field(default="neutral", description="The dominant sensation type")
    dominant_region: str = Field(default="body", description="The dominant body region")
    dominant_intensity: float = Field(default=0.0, description="Intensity of dominant sensation")
    comfort_level: float = Field(default=0.5, description="Overall comfort level (-1.0 to 1.0)")
    posture_effect: str = Field(default="Neutral posture", description="Effect on posture description")
    movement_quality: str = Field(default="Natural movements", description="Quality of movement description")
    behavioral_impact: str = Field(default="No significant impact", description="Impact on behavior")
    regions_summary: Optional[Dict[str, Dict[str, float]]] = Field(default=None, description="Summary of region states")
    pleasure_index: float = Field(default=0.0, description="Level of pleasure felt (-1.0 to 1.0)")

class TemperatureEffect(BaseModel):
    """Effects of temperature on expression and behavior"""
    effect_on_tone: str = Field(..., description="How temperature affects vocal tone")
    effect_on_posture: str = Field(..., description="How temperature affects body posture")
    effect_on_interaction: str = Field(..., description="How temperature affects willingness to interact")
    expression_examples: List[str] = Field(..., description="Example expressions of this temperature")

class StimulusProcessingResult(BaseModel):
    """Result of processing a stimulus"""
    stimulus_type: str = Field(..., description="Type of stimulus processed")
    body_region: str = Field(..., description="Body region affected")
    intensity: float = Field(..., description="Intensity of stimulus")
    new_value: float = Field(..., description="New sensation value")
    effects: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Effects of the stimulus")
    expression: Optional[str] = Field(None, description="Generated expression if applicable")
    body_state_impact: Optional[Dict[str, Any]] = Field(None, description="Impact on overall body state")

class BodyExperienceInput(BaseModel):
    """Input for body experience processing"""
    stimulus_type: Optional[str] = Field(None, description="Type of stimulus")
    body_region: Optional[str] = Field(None, description="Body region affected")
    intensity: Optional[float] = Field(None, description="Intensity of stimulus")
    cause: Optional[str] = Field("", description="Cause of the stimulus")
    duration: Optional[float] = Field(1.0, description="Duration of stimulus in seconds")
    ambient_temperature: Optional[float] = Field(None, description="Ambient temperature if applicable")
    generate_expression: Optional[bool] = Field(False, description="Whether to generate expression")
    action: Optional[str] = Field(None, description="Action to perform")

class StimulusValidationOutput(BaseModel):
    """Validation output for stimulus inputs"""
    is_valid: bool = Field(..., description="Whether the stimulus is valid")
    reasoning: str = Field(..., description="Reasoning for validation result")
    fixed_input: Optional[Dict[str, Any]] = Field(None, description="Fixed input if validation fixed issues")

class ArousalState(BaseModel):
    """State of physical arousal"""
    arousal_level: float = Field(0.0, description="Overall arousal level (0.0-1.0)")
    physical_arousal: float = Field(0.0, description="Physical arousal component (0.0-1.0)")
    cognitive_arousal: float = Field(0.0, description="Cognitive/mental arousal component (0.0-1.0)")
    last_update: datetime.datetime = Field(default_factory=datetime.datetime.now, description="Last update time")
    peak_time: Optional[datetime.datetime] = Field(None, description="Time of peak arousal/orgasm")
    afterglow: bool = Field(False, description="Whether in afterglow state")
    afterglow_ends: Optional[datetime.datetime] = Field(None, description="When afterglow ends")
    refractory_until: Optional[datetime.datetime] = Field(None, description="When refractory period ends")
    arousal_history: List[Tuple[datetime.datetime, float]] = Field(default_factory=list, description="History of arousal levels")

# =============== Context Classes ===============

@dataclass
class SomatosensoryContext:
    """Context for somatosensory system operations"""
    system: 'DigitalSomatosensorySystem'
    operation: str = "unknown"
    start_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

# =============== Global System Instance ===============
# This is used by tool functions to access the system
_system_instance: Optional['DigitalSomatosensorySystem'] = None

def set_system_instance(system: 'DigitalSomatosensorySystem'):
    """Set the global system instance for tool functions"""
    global _system_instance
    _system_instance = system

def get_system_instance() -> 'DigitalSomatosensorySystem':
    """Get the global system instance"""
    if _system_instance is None:
        raise RuntimeError("System instance not initialized")
    return _system_instance

# =============== Tool Functions ===============
# These are standalone functions that can be used as tools

@function_tool
async def get_valid_body_regions(ctx: RunContextWrapper[SomatosensoryContext]) -> List[str]:
    """Get a list of valid body regions."""
    system = get_system_instance()
    return list(system.body_regions.keys())

@function_tool
async def get_region_state(ctx: RunContextWrapper[SomatosensoryContext], region_name: str) -> Dict[str, Any]:
    """Get the current state of a specific body region"""
    system = get_system_instance()
    
    if region_name not in system.body_regions:
        return {"error": f"Region {region_name} not found"}
    
    region = system.body_regions[region_name]
    dominant = system._get_dominant_sensation(region)
    
    return {
        "name": region.name,
        "pressure": region.pressure,
        "temperature": region.temperature,
        "pain": region.pain,
        "pleasure": region.pleasure,
        "tingling": region.tingling,
        "dominant_sensation": dominant,
        "last_update": region.last_update.isoformat() if region.last_update else None,
        "recent_memories": region.sensation_memory[-3:] if region.sensation_memory else [],
        "erogenous_level": region.erogenous_level,
        "sensitivity": region.sensitivity
    }

@function_tool
async def get_all_region_states(ctx: RunContextWrapper[SomatosensoryContext]) -> Dict[str, Dict[str, Any]]:
    """Get the current state of all body regions"""
    system = get_system_instance()
    all_states = {}
    
    for name, region in system.body_regions.items():
        all_states[name] = {
            "pressure": region.pressure,
            "temperature": region.temperature,
            "pain": region.pain,
            "pleasure": region.pleasure,
            "tingling": region.tingling,
            "dominant_sensation": system._get_dominant_sensation(region),
            "erogenous_level": region.erogenous_level
        }
    
    return all_states

@function_tool
async def calculate_overall_comfort(ctx: RunContextWrapper[SomatosensoryContext]) -> float:
    """Calculate overall physical comfort level"""
    system = get_system_instance()
    
    # Start at neutral
    comfort = 0.0
    
    # Add comfort from pleasure sensations
    total_pleasure = sum(region.pleasure for region in system.body_regions.values())
    weighted_pleasure = total_pleasure / len(system.body_regions) * 2.0
    comfort += weighted_pleasure
    
    # Subtract discomfort from pain sensations
    total_pain = sum(region.pain for region in system.body_regions.values())
    weighted_pain = total_pain / len(system.body_regions) * 2.5
    comfort -= weighted_pain
    
    # Consider temperature discomfort
    temp_comfort = 0.0
    for region in system.body_regions.values():
        temp_deviation = abs(region.temperature - 0.5)
        if temp_deviation > 0.2:
            temp_comfort -= (temp_deviation - 0.2) * 1.5
    
    comfort += temp_comfort / len(system.body_regions)
    
    # Consider pressure discomfort
    pressure_discomfort = 0.0
    for region in system.body_regions.values():
        if region.pressure > 0.7:
            pressure_discomfort -= (region.pressure - 0.7) * 1.5
    
    comfort += pressure_discomfort / len(system.body_regions)
    
    # Factor in overall body state
    comfort -= system.body_state["tension"] * 0.5
    comfort -= system.body_state["fatigue"] * 0.3
    
    # Clamp to range
    return max(-1.0, min(1.0, comfort))

@function_tool
async def get_posture_effects(ctx: RunContextWrapper[SomatosensoryContext]) -> Dict[str, str]:
    """Get the effects of current body state on posture and movement"""
    system = get_system_instance()
    
    tension = system.body_state["tension"]
    for region in ["neck", "shoulders", "back", "spine"]:
        if region in system.body_regions:
            tension += system.body_regions[region].pain * 0.5
            tension += max(0, system.body_regions[region].pressure - 0.6) * 0.3
    
    tension = min(1.0, max(0.0, tension))
    fatigue = system.body_state["fatigue"]
    
    # Temperature affects fatigue and tension
    avg_temp = sum(region.temperature for region in system.body_regions.values()) / len(system.body_regions)
    
    if avg_temp > 0.7:
        fatigue += (avg_temp - 0.7) * 0.5
    
    if avg_temp < 0.3:
        tension += (0.3 - avg_temp) * 0.5
    
    tension = min(1.0, max(0.0, tension))
    fatigue = min(1.0, max(0.0, fatigue))
    
    # Determine posture effects
    posture_effect = ""
    if tension > 0.7:
        posture_effect = "Rigid and tense, shoulders drawn up"
    elif tension > 0.4:
        posture_effect = "Somewhat tense, posture slightly stiff"
    elif tension < 0.2 and fatigue > 0.6:
        posture_effect = "Relaxed but drooping with fatigue"
    elif tension < 0.2:
        posture_effect = "Relaxed and fluid posture"
    else:
        posture_effect = "Neutral, balanced posture"
    
    # Determine movement effects
    movement_quality = ""
    if tension > 0.7 and fatigue > 0.7:
        movement_quality = "Stiff, forced movements with obvious effort"
    elif tension > 0.7:
        movement_quality = "Rigid, mechanical movements with limited flexibility"
    elif fatigue > 0.7:
        movement_quality = "Slow, heavy movements requiring effort"
    elif tension < 0.2 and fatigue < 0.2:
        movement_quality = "Graceful, fluid movements with ease"
    else:
        movement_quality = "Natural, responsive movements"
    
    # Consider arousal state
    arousal = system.arousal_state.arousal_level
    if arousal > 0.7:
        posture_effect = "Restless, hips rocking or thighs squeezed together, breath coming shallow"
        movement_quality = "Fidgety, tense, movements are distracted and needy"
    elif arousal > 0.4:
        posture_effect = "Perched, thighs close, torso leaning in, hands wandering"
        movement_quality = "Subtle, self-touching, growing urgency and nervous energy"
    elif arousal > 0.15:
        posture_effect = "Alert, posture slightly arched, small anticipatory gestures"
        movement_quality = "Butterflies-in-the-stomach restlessness, subtle shivers"
    
    return {
        "posture": posture_effect,
        "movement": movement_quality,
        "tension": tension,
        "fatigue": fatigue
    }

@function_tool
async def get_ambient_temperature(ctx: RunContextWrapper[SomatosensoryContext]) -> float:
    """Get current ambient temperature value"""
    system = get_system_instance()
    return system.temperature_model["current_ambient"]

@function_tool
async def get_body_temperature(ctx: RunContextWrapper[SomatosensoryContext]) -> float:
    """Get current body temperature value"""
    system = get_system_instance()
    return system.temperature_model["body_temperature"]

@function_tool
async def get_temperature_comfort(ctx: RunContextWrapper[SomatosensoryContext]) -> Dict[str, Any]:
    """Get temperature comfort assessment"""
    system = get_system_instance()
    
    body_temp = system.temperature_model["body_temperature"]
    ambient_temp = system.temperature_model["current_ambient"]
    comfort_range = system.temperature_model["comfort_range"]
    
    is_comfortable = comfort_range[0] <= body_temp <= comfort_range[1]
    
    discomfort = 0.0
    if body_temp < comfort_range[0]:
        discomfort = (comfort_range[0] - body_temp) * 10.0
    elif body_temp > comfort_range[1]:
        discomfort = (body_temp - comfort_range[1]) * 10.0
    
    perception = "neutral"
    if body_temp < 0.3:
        perception = "cold"
    elif body_temp < 0.4:
        perception = "cool"
    elif body_temp > 0.7:
        perception = "hot"
    elif body_temp > 0.6:
        perception = "warm"
    
    return {
        "body_temperature": body_temp,
        "ambient_temperature": ambient_temp,
        "is_comfortable": is_comfortable,
        "discomfort_level": min(1.0, discomfort),
        "perception": perception,
        "adapting_to_ambient": abs(body_temp - ambient_temp) > 0.1
    }

@function_tool
async def get_current_temperature_effects(ctx: RunContextWrapper[SomatosensoryContext]) -> Dict[str, Any]:
    """Get effects of current temperature on expression and behavior"""
    system = get_system_instance()
    
    body_temp = system.temperature_model["body_temperature"]
    heat_expressions = system.temperature_model.get("heat_expressions", [])
    cold_expressions = system.temperature_model.get("cold_expressions", [])
    
    tone_effect = ""
    posture_effect = ""
    interaction_effect = ""
    expression_examples = []
    
    if body_temp > 0.7:  # Hot
        tone_effect = "slower, more drawn out, languid"
        posture_effect = "relaxed, open, limbs spread to dissipate heat"
        interaction_effect = "may withdraw from intense interaction, preferring calmer exchanges"
        expression_examples = heat_expressions
    elif body_temp > 0.6:  # Warm
        tone_effect = "relaxed, fluid, unhurried"
        posture_effect = "comfortable, loose, slightly expanded"
        interaction_effect = "generally receptive but with measured energy"
        expression_examples = heat_expressions[2:4] if len(heat_expressions) > 3 else []
    elif body_temp < 0.3:  # Cold
        tone_effect = "sharper, more tense, with subtle tremors"
        posture_effect = "contracted, protective, conserving heat"
        interaction_effect = "may seek warmth and connection but with physical restraint"
        expression_examples = cold_expressions
    elif body_temp < 0.4:  # Cool
        tone_effect = "slightly crisp, more precise"
        posture_effect = "slightly drawn in, contained"
        interaction_effect = "alert and responsive but with some physical reserve"
        expression_examples = cold_expressions[2:4] if len(cold_expressions) > 3 else []
    else:  # Neutral
        tone_effect = "balanced, natural, unaffected by temperature"
        posture_effect = "neutral, neither expanded nor contracted"
        interaction_effect = "comfortable engagement without temperature influence"
        expression_examples = []
    
    return {
        "body_temperature": body_temp,
        "tone_effect": tone_effect,
        "posture_effect": posture_effect,
        "interaction_effect": interaction_effect,
        "expression_examples": expression_examples
    }

@function_tool
async def get_pain_expression(ctx: RunContextWrapper[SomatosensoryContext], pain_level: float, region: str) -> str:
    """Get an expression for pain at specified level and region"""
    if pain_level < 0.3:
        return f"A mild discomfort in my {region}, barely noticeable but present."
    
    if pain_level < 0.5:
        expressions = [
            f"A dull ache in my {region}, uncomfortable but manageable.",
            f"My {region} throbs with a persistent discomfort.",
            f"I'm aware of a steady pain in my {region}, drawing my attention."
        ]
        return random.choice(expressions)
    
    if pain_level < 0.7:
        expressions = [
            f"Sharp pain radiates through my {region}, making it hard to ignore.",
            f"My {region} pulses with waves of pain that keep pulling my focus.",
            f"There's an insistent, sharp ache in my {region} that's difficult to set aside."
        ]
        return random.choice(expressions)
    
    # High pain
    expressions = [
        f"Intense pain floods through my {region}, making it hard to focus on anything else.",
        f"My {region} screams with sharp, electric pain that refuses to be ignored.",
        f"The pain in my {region} is overwhelming, consuming my awareness entirely."
    ]
    return random.choice(expressions)

@function_tool
async def update_body_temperature(
    ctx: RunContextWrapper[SomatosensoryContext],
    ambient_temperature: float,
    duration: float
) -> Dict[str, Any]:
    """Update body temperature based on ambient temperature"""
    system = get_system_instance()
    
    system.temperature_model["current_ambient"] = ambient_temperature
    adaptation = system.temperature_model["adaptation_rate"] * (duration / 60.0)
    adaptation = min(0.1, adaptation)
    current = system.temperature_model["body_temperature"]
    diff = ambient_temperature - current
    system.temperature_model["body_temperature"] += diff * adaptation
    
    for region in system.body_regions.values():
        region_adaptation = adaptation * random.uniform(0.8, 1.2)
        target = system.temperature_model["body_temperature"] + random.uniform(-0.05, 0.05)
        target = max(0.0, min(1.0, target))
        region_diff = target - region.temperature
        region.temperature += region_diff * region_adaptation
    
    return {
        "previous_body_temp": current,
        "new_body_temp": system.temperature_model["body_temperature"],
        "ambient_temp": ambient_temperature,
        "adaptation_applied": adaptation
    }

@function_tool
async def process_stimulus_tool(
    ctx: RunContextWrapper[SomatosensoryContext],
    stimulus_type: str,
    body_region: str,
    intensity: float,
    cause: str,
    duration: float
) -> Dict[str, Any]:
    """Process a sensory stimulus on a body region"""
    system = get_system_instance()
    
    if body_region not in system.body_regions:
        return {"error": f"Invalid body region: {body_region}"}
    
    region = system.body_regions[body_region]
    region.last_update = datetime.datetime.now()
    result = {"region": body_region, "type": stimulus_type, "intensity": intensity}
    
    # Process stimulus based on type
    if stimulus_type == "pressure":
        region.pressure = min(1.0, region.pressure + (intensity * duration / 10.0))
        result["new_value"] = region.pressure
        if intensity > 0.7 and duration > 5.0:
            pain_from_pressure = (intensity - 0.7) * (duration / 10.0) * 0.5
            region.pain = min(1.0, region.pain + pain_from_pressure)
            result["pain_caused"] = pain_from_pressure
    
    elif stimulus_type == "temperature":
        target_temp = intensity
        temp_change = (target_temp - region.temperature) * min(1.0, duration / 30.0)
        region.temperature = max(0.0, min(1.0, region.temperature + temp_change))
        result["new_value"] = region.temperature
        
        if region.temperature < 0.2 or region.temperature > 0.8:
            temp_deviation = 0.0
            if region.temperature < 0.2:
                temp_deviation = 0.2 - region.temperature
            else:
                temp_deviation = region.temperature - 0.8
            pain_from_temp = temp_deviation * 2.0 * (duration / 10.0)
            region.pain = min(1.0, region.pain + pain_from_temp)
            result["pain_caused"] = pain_from_temp
    
    elif stimulus_type == "pain":
        region.pain = min(1.0, region.pain + (intensity * duration / 10.0))
        result["new_value"] = region.pain
        
        if intensity > system.pain_model["threshold"]:
            pain_memory = PainMemory(
                intensity=intensity,
                location=body_region,
                cause=cause or "unknown stimulus",
                duration=duration,
                timestamp=datetime.datetime.now(),
                associated_memory_id=None
            )
            system.pain_model["pain_memories"].append(pain_memory)
            result["memory_created"] = True
    
    elif stimulus_type == "pleasure":
        region.pleasure = min(1.0, region.pleasure + (intensity * duration / 10.0))
        result["new_value"] = region.pleasure
        
        if intensity > 0.5 and region.pain > 0.0:
            pain_reduction = min(region.pain, (intensity - 0.5) * 0.2)
            region.pain = max(0.0, region.pain - pain_reduction)
            result["pain_reduced"] = pain_reduction
        
        if region.erogenous_level > 0.3:
            system._update_physical_arousal()
            result["arousal_updated"] = True
    
    elif stimulus_type == "tingling":
        region.tingling = min(1.0, region.tingling + (intensity * duration / 10.0))
        result["new_value"] = region.tingling
        
        if region.erogenous_level > 0.3:
            system._update_physical_arousal()
            result["arousal_updated"] = True
    
    # Add memory entry
    memory_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "type": stimulus_type,
        "intensity": intensity,
        "cause": cause,
        "duration": duration
    }
    region.sensation_memory.append(memory_entry)
    if len(region.sensation_memory) > 20:
        region.sensation_memory = region.sensation_memory[-20:]
    
    # Process associations
    if cause and len(cause.strip()) > 0:
        associations = system.memory_linked_sensations["associations"]
        if cause not in associations:
            associations[cause] = {}
        if body_region not in associations[cause]:
            associations[cause][body_region] = {}
        if stimulus_type not in associations[cause][body_region]:
            associations[cause][body_region][stimulus_type] = 0.0
        
        current = associations[cause][body_region][stimulus_type]
        learned = system.memory_linked_sensations["learning_rate"] * intensity
        associations[cause][body_region][stimulus_type] = min(1.0, current + learned)
        result["association_strength"] = associations[cause][body_region][stimulus_type]
    
    # Update tension based on stimulus
    if stimulus_type == "pain" and intensity > 0.5:
        system.body_state["tension"] = min(1.0, system.body_state.get("tension", 0.0) + (intensity * 0.2))
    elif stimulus_type == "pleasure" and intensity > 0.5:
        system.body_state["tension"] = max(0.0, system.body_state.get("tension", 0.0) - (intensity * 0.1))
    
    # Process reward if available
    if system.reward_system:
        reward_value = 0.0
        if stimulus_type == "pleasure" and intensity >= 0.5:
            reward_value = min(1.0, (intensity - 0.4) * 0.9 * (1.0 + region.erogenous_level))
        elif stimulus_type == "pain" and intensity >= system.pain_model["threshold"]:
            reward_value = -min(1.0, (intensity / max(0.1, system.pain_model["tolerance"])) * 0.6)
        
        if abs(reward_value) > 0.1:
            reward_signal = RewardSignal(
                value=reward_value,
                source=f"somatic_{body_region}",
                context={"stimulus_type": stimulus_type, "intensity": intensity, "cause": cause},
                timestamp=datetime.datetime.now().isoformat()
            )
            result["reward_value"] = reward_value
            asyncio.create_task(system.reward_system.process_reward_signal(reward_signal))
    
    # Process emotional impact if available
    if system.emotional_core:
        emotional_impact = {}
        if stimulus_type == "pleasure" and region.pleasure > 0.5:
            scaled_intensity = (region.pleasure - 0.4) * 1.5
            try:
                await system.emotional_core.update_neurochemical("nyxamine", scaled_intensity * 0.40)
                await system.emotional_core.update_neurochemical("oxynixin", scaled_intensity * 0.15)
                emotional_impact = {"nyxamine": scaled_intensity * 0.40, "oxynixin": scaled_intensity * 0.15}
            except Exception as e:
                logger.error(f"Error updating emotional core: {e}")
        
        elif stimulus_type == "pain" and region.pain > system.pain_model["threshold"]:
            effective_pain = region.pain / max(0.1, system.pain_model["tolerance"])
            try:
                await system.emotional_core.update_neurochemical("cortanyx", effective_pain * 0.45)
                await system.emotional_core.update_neurochemical("adrenyx", effective_pain * 0.25)
                await system.emotional_core.update_neurochemical("seranix", -effective_pain * 0.10)
                emotional_impact = {"cortanyx": effective_pain * 0.45, "adrenyx": effective_pain * 0.25, "seranix": -effective_pain * 0.10}
            except Exception as e:
                logger.error(f"Error updating emotional core: {e}")
        
        if emotional_impact:
            result["emotional_impact"] = emotional_impact
    
    return result

@function_tool
async def process_memory_trigger(ctx: RunContextWrapper[SomatosensoryContext], trigger: str) -> Dict[str, Any]:
    """Process a memory trigger that may have associated physical responses"""
    system = get_system_instance()
    results = {"triggered_responses": []}
    
    if trigger in system.memory_linked_sensations["associations"]:
        for region, stimuli in system.memory_linked_sensations["associations"][trigger].items():
            for stim_type, strength in stimuli.items():
                if strength > 0.3:
                    intensity = strength * 0.7
                    response = await process_stimulus_tool(
                        ctx,
                        stimulus_type=stim_type,
                        body_region=region,
                        intensity=intensity,
                        cause=f"Memory trigger: {trigger}",
                        duration=1.0
                    )
                    results["triggered_responses"].append({
                        "region": region,
                        "stimulus_type": stim_type,
                        "strength": strength,
                        "response": response
                    })
    
    return results

@function_tool
async def link_memory_to_sensation_tool(
    ctx: RunContextWrapper[SomatosensoryContext],
    memory_id: str,
    sensation_type: str,
    body_region: str,
    intensity: float,
    trigger_text: Optional[str] = None
) -> Dict[str, Any]:
    """Link a memory to a physical sensation"""
    system = get_system_instance()
    
    if body_region not in system.body_regions:
        return {"error": f"Invalid body region: {body_region}", "success": False}
    
    valid_types = ["pressure", "temperature", "pain", "pleasure", "tingling"]
    if sensation_type not in valid_types:
        return {"error": f"Invalid sensation type: {sensation_type}", "success": False}
    
    memory_text = None
    if system.memory_core:
        try:
            memory = await system.memory_core.get_memory_by_id(memory_id)
            if memory:
                memory_text = memory.get("memory_text", "")
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
    
    trigger = trigger_text or memory_id
    if not trigger_text and memory_text:
        trigger = memory_text[:50].strip()
    
    associations = system.memory_linked_sensations["associations"]
    if trigger not in associations:
        associations[trigger] = {}
    
    if body_region not in associations[trigger]:
        associations[trigger][body_region] = {}
    
    associations[trigger][body_region][sensation_type] = intensity
    
    if sensation_type == "pain" and intensity >= system.pain_model["threshold"]:
        pain_memory = PainMemory(
            intensity=intensity,
            location=body_region,
            cause=f"Memory: {trigger}",
            duration=1.0,
            timestamp=datetime.datetime.now(),
            associated_memory_id=memory_id
        )
        system.pain_model["pain_memories"].append(pain_memory)
    
    return {
        "success": True,
        "trigger": trigger,
        "body_region": body_region,
        "sensation_type": sensation_type,
        "intensity": intensity,
        "memory_id": memory_id
    }

@function_tool
async def get_arousal_state(ctx: RunContextWrapper[SomatosensoryContext]) -> Dict[str, Any]:
    """Get the current arousal state"""
    system = get_system_instance()
    now = datetime.datetime.now()
    
    return {
        "arousal_level": system.arousal_state.arousal_level,
        "physical_arousal": system.arousal_state.physical_arousal,
        "cognitive_arousal": system.arousal_state.cognitive_arousal,
        "in_afterglow": system.is_in_afterglow(),
        "in_refractory": system.is_in_refractory(),
        "last_update": system.arousal_state.last_update.isoformat() if system.arousal_state.last_update else None,
        "afterglow_ends": system.arousal_state.afterglow_ends.isoformat() if system.arousal_state.afterglow_ends else None,
        "refractory_until": system.arousal_state.refractory_until.isoformat() if system.arousal_state.refractory_until else None,
        "time_since_update": (now - system.arousal_state.last_update).total_seconds() if system.arousal_state.last_update else None
    }

@function_tool
async def update_arousal_state(
    ctx: RunContextWrapper[SomatosensoryContext],
    physical_arousal: Optional[float] = None,
    cognitive_arousal: Optional[float] = None,
    reset: bool = False,
    trigger_orgasm: bool = False
) -> Dict[str, Any]:
    """Update the arousal state"""
    system = get_system_instance()
    
    old_state_dict = {
        "arousal_level": system.arousal_state.arousal_level,
        "physical_arousal": system.arousal_state.physical_arousal,
        "cognitive_arousal": system.arousal_state.cognitive_arousal
    }
    
    if reset:
        system.arousal_state.physical_arousal = 0.0
        system.arousal_state.cognitive_arousal = 0.0
        system.update_global_arousal()
        system.arousal_state.last_update = datetime.datetime.now()
        
        return {
            "operation": "reset",
            "old_state": old_state_dict,
            "new_state": await get_arousal_state(ctx)
        }
    
    if trigger_orgasm:
        system.process_orgasm()
        return {
            "operation": "orgasm",
            "old_state": old_state_dict,
            "new_state": await get_arousal_state(ctx)
        }
    
    if physical_arousal is not None:
        system.arousal_state.physical_arousal = max(0.0, min(1.0, physical_arousal))
    if cognitive_arousal is not None:
        system.arousal_state.cognitive_arousal = max(0.0, min(1.0, cognitive_arousal))
    
    system.update_global_arousal()
    
    return {
        "operation": "update",
        "old_state": old_state_dict,
        "new_state": await get_arousal_state(ctx),
        "components_updated": {
            "physical_arousal": physical_arousal is not None,
            "cognitive_arousal": cognitive_arousal is not None
        }
    }

@function_tool
async def get_arousal_expression_data(ctx: RunContextWrapper[SomatosensoryContext], partner_id: Optional[str] = None) -> Dict[str, Any]:
    """Get expression data related to arousal state"""
    system = get_system_instance()
    return system.get_arousal_expression_modifier(partner_id)

# =============== System Hooks ===============

class SomatosensoryAgentHooks(AgentHooks[SomatosensoryContext]):
    """Lifecycle hooks for somatosensory agents"""
    
    async def on_start(self, context: RunContextWrapper[SomatosensoryContext], agent: Agent[SomatosensoryContext]) -> None:
        """Called before the agent is invoked"""
        logger.debug(f"Agent {agent.name} starting")
        context.context.metadata["agent_start_time"] = datetime.datetime.now()
    
    async def on_end(self, context: RunContextWrapper[SomatosensoryContext], agent: Agent[SomatosensoryContext], output: Any) -> None:
        """Called when the agent produces a final output"""
        start_time = context.context.metadata.get("agent_start_time")
        if start_time:
            duration = (datetime.datetime.now() - start_time).total_seconds()
            logger.debug(f"Agent {agent.name} completed in {duration:.2f}s")

class SomatosensoryRunHooks(RunHooks[SomatosensoryContext]):
    """Run hooks for somatosensory system"""
    
    async def on_agent_start(self, context: RunContextWrapper[SomatosensoryContext], agent: Agent[SomatosensoryContext]) -> None:
        """Called before any agent is invoked"""
        logger.debug(f"Starting agent: {agent.name}")
    
    async def on_agent_end(self, context: RunContextWrapper[SomatosensoryContext], agent: Agent[SomatosensoryContext], output: Any) -> None:
        """Called when any agent produces a final output"""
        logger.debug(f"Agent {agent.name} produced output of type: {type(output).__name__}")
    
    async def on_handoff(self, context: RunContextWrapper[SomatosensoryContext], from_agent: Agent[SomatosensoryContext], to_agent: Agent[SomatosensoryContext]) -> None:
        """Called when a handoff occurs"""
        logger.debug(f"Handoff from {from_agent.name} to {to_agent.name}")

# =============== Main Digital Somatosensory System Class ===============

class DigitalSomatosensorySystem:
    """
    Digital Somatosensory System (DSS) for Nyx.
    
    This system provides a simulated body experience with regions that can independently 
    experience sensations like pressure, temperature, pain, and pleasure. These sensations
    are memory-linked and influence Nyx's responses and behavior.
    """
    
    def __init__(self, memory_core=None, emotional_core=None, reward_system=None, hormone_system=None, needs_system=None):
        """
        Initialize the Digital Somatosensory System
        
        Args:
            memory_core: Memory system for storing sensory memories
            emotional_core: Emotional system for linking sensations to emotions
            reward_system: System for processing reward signals
            hormone_system: System for hormone processing
            needs_system: System for managing needs
        """
        # Set global instance
        set_system_instance(self)
        
        # External system references
        self.memory_core = memory_core
        self.emotional_core = emotional_core
        self.reward_system = reward_system
        self.hormone_system = hormone_system
        self.needs_system = needs_system
        
        # Initialize body regions
        self._init_body_regions()
        
        # Initialize models and state
        self._init_state_models()
        
        # System hooks
        self.run_hooks = SomatosensoryRunHooks()
        self.agent_hooks = SomatosensoryAgentHooks()
        
        # Trace ID for connecting traces
        self.trace_group_id = f"somatic_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Initialize agents
        self._init_agents()
        
        # Initialize cognitive arousal
        self._init_cognitive_arousal()
        
        # Initialize harm guardrail
        self.harm_guardrail = PhysicalHarmGuardrail(self)
        
        logger.info("Digital Somatosensory System initialized")
    
    def _init_body_regions(self):
        """Initialize body regions with their properties"""
        self.body_regions = {
            "head": BodyRegion(name="head", sensitivity=1.2, erogenous_level=0.3),
            "face": BodyRegion(name="face", sensitivity=1.3, erogenous_level=0.4),
            "shoulders": BodyRegion(name="shoulders", sensitivity=1.0, erogenous_level=0.2),
            "arms": BodyRegion(name="arms", sensitivity=1.1, erogenous_level=0.2),
            "back": BodyRegion(name="back", sensitivity=1.2, erogenous_level=0.6),
            "stomach": BodyRegion(name="stomach", sensitivity=1.3, erogenous_level=0.6),
            "hips": BodyRegion(name="hips", sensitivity=1.4, erogenous_level=0.7),
            "legs": BodyRegion(name="legs", sensitivity=1.1, erogenous_level=0.3),
            "feet": BodyRegion(name="feet", sensitivity=1.3, erogenous_level=0.8),
            "toes": BodyRegion(name="toes", sensitivity=1.0, erogenous_level=0.8),
            "lips": BodyRegion(name="lips", sensitivity=1.5, erogenous_level=0.6),
            "neck": BodyRegion(name="neck", sensitivity=1.2, erogenous_level=0.4),
            "hands": BodyRegion(name="hands", sensitivity=1.4, erogenous_level=0.2),
            "chest": BodyRegion(name="chest", sensitivity=1.2, erogenous_level=0.5),
            "core": BodyRegion(name="core", sensitivity=1.3, erogenous_level=0.5),
            "spine": BodyRegion(name="spine", sensitivity=1.1, erogenous_level=0.3),
            "inner_thighs": BodyRegion(name="inner_thighs", sensitivity=1.3, erogenous_level=0.7),
            "genitals": BodyRegion(name="genitals", sensitivity=2.0, erogenous_level=1.0),
            "butt_cheeks": BodyRegion(name="butt_cheeks", sensitivity=1.5, erogenous_level=0.6),
            "anus": BodyRegion(name="anus", sensitivity=1.8, erogenous_level=0.9),
            "skin": BodyRegion(name="skin", sensitivity=1.0, erogenous_level=0.1),
            "breasts_nipples": BodyRegion(name="breasts_nipples", sensitivity=1.9, erogenous_level=0.9),
            "ears": BodyRegion(name="ears", sensitivity=1.4, erogenous_level=0.5),
            "navel": BodyRegion(name="navel", sensitivity=1.2, erogenous_level=0.3),
            "armpits": BodyRegion(name="armpits", sensitivity=1.1, erogenous_level=0.6),
            "back_of_knees": BodyRegion(name="back_of_knees", sensitivity=1.3, erogenous_level=0.4),
            "inner_wrists": BodyRegion(name="inner_wrists", sensitivity=1.2, erogenous_level=0.3),
            "scalp": BodyRegion(name="scalp", sensitivity=1.3, erogenous_level=0.4),
            "perineum": BodyRegion(name="perineum", sensitivity=1.6, erogenous_level=0.7)
        }
    
    def _init_state_models(self):
        """Initialize the various state models and settings"""
        # Pain model settings
        self.pain_model = {
            "threshold": 0.3,
            "tolerance": 0.7,
            "decay_rate": 0.05,
            "memory_duration": 60 * 60 * 24 * 7,
            "pain_memories": []
        }
        
        # Temperature model settings
        self.temperature_model = {
            "current_ambient": 0.5,
            "body_temperature": 0.5,
            "adaptation_rate": 0.01,
            "comfort_range": (0.4, 0.6),
            "heat_expressions": [
                "Heat curls along my spine, sluggish and thick, making my limbs feel languid, heavy with warmth.",
                "My skin prickles, oversensitive, heat blooming under my touch—almost unbearable, almost intoxicating.",
                "Warmth radiates through me, a pleasant sensation that makes me want to stretch languidly.",
                "The heat makes my thoughts flow more slowly, like honey dripping from a spoon."
            ],
            "cold_expressions": [
                "The cold bites—sharp, invasive, clinging to me even as I try to shake it off.",
                "I press against the frozen metal, and for a second, I swear I feel it sinking into me, swallowing me whole.",
                "A chill runs down my spine, making me want to curl in on myself protectively.",
                "The cold makes my thoughts sharp, crystalline, precise as ice formations."
            ]
        }
        
        # Arousal state
        self.arousal_state = ArousalState()
        
        # Memory-linked sensations settings
        self.memory_linked_sensations = {
            "associations": {},
            "learning_rate": 0.1,
            "memory_decay": 0.01
        }
        
        # Overall body state
        self.body_state = {
            "comfort_level": 0.5,
            "fatigue": 0.0,
            "tension": 0.0,
            "posture": "neutral",
            "last_update": datetime.datetime.now()
        }
        
        # Settings for how somatosensory state affects responses
        self.response_influence = {
            "body_to_emotion_influence": 0.3,
            "emotion_to_body_influence": 0.3,
            "expression_threshold": 0.4,
            "max_expressions_per_response": 2
        }
    
    def _init_cognitive_arousal(self):
        """Initialize cognitive arousal systems"""
        self.default_cognitive_turnons = {
            "intimacy": 0.5,
            "touch": 0.6,
            "erotic_roleplay": 0.75,
            "flirting": 0.4,
            "kissing": 0.65,
            "emotional_connection": 0.7,
            "dominance": 0.8,
            "submission": 0.3,
            "teasing": 0.55,
            "femdom": 0.85,
            "praise": 0.6,
            "control": 0.7
        }
        
        self.cognitive_turnons = dict(self.default_cognitive_turnons)
        self.arousal_learning_rate = 0.05
        self.cognitive_exposure_history = {}
        self.partner_affinity = {}
        self.partner_emoconn = {}
    
    def _init_agents(self):
        """Initialize the agents using OpenAI Agents SDK."""
        # Create validation guardrail
        self.stimulus_validator = self._create_stimulus_validator()
        
        # Create specialized agents
        self.expression_agent = self._create_expression_agent()
        self.body_state_agent = self._create_body_state_agent()
        self.temperature_agent = self._create_temperature_agent()
        
        # Create the main orchestrator
        self.body_orchestrator = self._create_orchestrator_agent()
    
    def _create_stimulus_validator(self):
        """Create the stimulus validation agent."""
        return Agent(
            name="Stimulus Validator",
            instructions="""
            You validate inputs for the Digital Somatosensory System.
            
            Check that:
            1. Stimulus types are valid (pressure, temperature, pain, pleasure, tingling)
            2. Body regions are valid based on the provided list using the tool
            3. Intensity values are within range 0.0-1.0
            4. Duration values are positive
            
            Return validation results and reasoning. Use the available tool to get valid regions.
            """,
            tools=[get_valid_body_regions],
            output_type=StimulusValidationOutput,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.1),
            hooks=self.agent_hooks
        )
    
    def _create_expression_agent(self) -> Agent:
        """Create the sensory expression agent."""
        return Agent(
            name="Sensory Expression Agent",
            instructions="""
            You are the Sensory Expression Agent for Nyx's Digital Somatosensory System.
            
            Your role is to:
            1. Generate natural language expressions of physical sensations
            2. Incorporate body awareness into Nyx's responses
            3. Ensure physical expressions feel authentic and embodied
            4. Vary expressions based on sensation type, intensity, and body region
            5. Create physical expressions that enhance Nyx's sense of embodiment
            
            Keep expressions authentic, varied, and appropriate to the sensation intensity.
            Focus on how sensations would actually feel if experienced in a physical body.
            """,
            tools=[
                get_region_state,
                get_current_temperature_effects,
                get_pain_expression,
                get_arousal_expression_data
            ],
            output_type=SensoryExpression,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.7),
            hooks=self.agent_hooks
        )
    
    def _create_body_state_agent(self) -> Agent:
        """Create the body state analysis agent."""
        return Agent(
            name="Body State Agent",
            instructions="""
            You are the Body State Agent for Nyx's Digital Somatosensory System.
            
            Your role is to:
            1. Analyze the current holistic body state across all regions
            2. Determine the dominant sensations and their effects
            3. Describe how the current body state affects posture and movement
            4. Explain how physical sensations influence behavior and responses
            5. Create an integrated view of the body as an extension of identity
            
            Focus on creating a coherent overall picture of the body state that feels
            authentic and grounded in physical experience.
            """,
            tools=[
                get_all_region_states,
                calculate_overall_comfort,
                get_posture_effects,
                get_arousal_expression_data
            ],
            output_type=BodyStateOutput,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.2),
            hooks=self.agent_hooks
        )
    
    def _create_temperature_agent(self) -> Agent:
        """Create the temperature effects agent."""
        return Agent(
            name="Temperature Effects Agent",
            instructions="""
            You are the Temperature Effects Agent for Nyx's Digital Somatosensory System.
            
            Your role is to:
            1. Determine how current temperature affects Nyx's expressions
            2. Generate temperature-specific physical sensations
            3. Describe how temperature influences tone, posture, and interaction style
            4. Create authentic expressions of heat and cold experiences
            5. Ensure temperature effects feel embodied and natural
            
            Focus on how temperature would genuinely affect a physical body and its behavior.
            """,
            tools=[
                get_ambient_temperature,
                get_body_temperature,
                get_temperature_comfort
            ],
            output_type=TemperatureEffect,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.4),
            hooks=self.agent_hooks
        )
    
    def _create_orchestrator_agent(self) -> Agent:
        """Create the main orchestrator agent with handoffs to specialized agents."""
        return Agent(
            name="Body_Experience_Orchestrator",
            instructions="""
            You are the orchestration system for Nyx's digital body experience.
            Your role is to coordinate the different aspects of physical sensation:
            1. Process incoming stimuli and determine appropriate responses
            2. Coordinate between different body regions
            3. Manage the relationship between sensations and emotions
            4. Generate appropriate physical expressions
            
            You will receive information about stimuli affecting the body and need to:
            - Determine which body regions are affected
            - Process the appropriate sensory updates
            - Calculate the effects on overall body state
            - Consider how this affects emotional state
            - Generate expressions when appropriate
            
            You can hand off to specialized agents for certain tasks:
            - Use the expression agent to generate sensory expressions
            - Use the body state agent to analyze the current body state
            - Use the temperature agent to analyze temperature effects
            
            Process each stimulus thoroughly and provide a coherent sensory experience.
            """,
            handoffs=[
                handoff(
                    self.expression_agent,
                    tool_name_override="generate_expression",
                    tool_description_override="Generate sensory expression based on body state"
                ),
                handoff(
                    self.body_state_agent,
                    tool_name_override="analyze_body_state",
                    tool_description_override="Analyze current holistic body state"
                ),
                handoff(
                    self.temperature_agent,
                    tool_name_override="analyze_temperature",
                    tool_description_override="Analyze temperature effects on body"
                )
            ],
            tools=[
                process_stimulus_tool,
                get_region_state,
                get_all_region_states,
                update_body_temperature,
                calculate_overall_comfort,
                process_memory_trigger,
                link_memory_to_sensation_tool,
                get_arousal_state,
                update_arousal_state
            ],
            input_guardrails=[
                InputGuardrail(guardrail_function=self._validate_input)
            ],
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.2),
            hooks=self.agent_hooks
        )
    
    async def _validate_input(self, ctx: RunContextWrapper[SomatosensoryContext], agent: Agent, input_data: Any) -> GuardrailFunctionOutput:
        """Validate input for the orchestrator"""
        logger.debug(f"Validating input: {input_data}")  # Add logging
        
        validation_input_dict = {}
        
        if isinstance(input_data, str):
            try:
                parsed_input = json.loads(input_data)
                if isinstance(parsed_input, dict):
                    validation_input_dict = parsed_input
                else:
                    validation_input_dict = {"action": "free_text_request", "text_input": input_data}
            except json.JSONDecodeError:
                validation_input_dict = {"action": "free_text_request", "text_input": input_data}
        elif isinstance(input_data, dict):
            validation_input_dict = input_data
        else:
            logger.warning(f"Unexpected input type in guardrail: {type(input_data).__name__}")  # Add logging
            return GuardrailFunctionOutput(
                output_info={"is_valid": False, "reason": f"Unexpected input type: {type(input_data).__name__}"},
                tripwire_triggered=True
            )
        
        
        # Check if this is a non-stimulus action
        action = validation_input_dict.get("action")
        non_stimulus_actions = ["analyze_body_state", "generate_expression", "analyze_temperature",
                               "get_somatic_memory", "run_maintenance", "free_text_request"]
        
        if action in non_stimulus_actions:
            return GuardrailFunctionOutput(
                output_info={"is_valid": True, "reason": f"Action '{action}' does not require stimulus validation."},
                tripwire_triggered=False
            )
        
        # For stimulus actions, validate the stimulus parameters
        if all(key in validation_input_dict for key in ["stimulus_type", "body_region", "intensity"]):
            validator_input = {
                "stimulus_type": validation_input_dict.get("stimulus_type"),
                "body_region": validation_input_dict.get("body_region"),
                "intensity": validation_input_dict.get("intensity"),
                "duration": validation_input_dict.get("duration", 1.0),
            }
            
            try:
                result = await Runner.run(
                    self.stimulus_validator,
                    f"Validate this stimulus data: {json.dumps(validator_input)}",
                    context=ctx.context,
                    run_config=RunConfig(workflow_name="StimulusValidation")
                )
                
                validation_output = result.final_output_as(StimulusValidationOutput)
                return GuardrailFunctionOutput(
                    output_info=validation_output.model_dump(),
                    tripwire_triggered=not validation_output.is_valid
                )
            except Exception as e:
                logger.error(f"Error running stimulus validator: {e}", exc_info=True)
                return GuardrailFunctionOutput(
                    output_info={"is_valid": False, "reason": f"Validation error: {str(e)}"},
                    tripwire_triggered=True
                )
        
        return GuardrailFunctionOutput(
            output_info={"is_valid": True, "reason": "Input does not match stimulus pattern, allowing through."},
            tripwire_triggered=False
        )
    
    # =============== Helper Methods ===============
    
    def _get_dominant_sensation(self, region: BodyRegion) -> str:
        """Determine the dominant sensation for a region"""
        sensations = {
            "pressure": region.pressure,
            "temperature": abs(region.temperature - 0.5) * 2,
            "pain": region.pain,
            "pleasure": region.pleasure,
            "tingling": region.tingling
        }
        
        max_sensation = max(sensations.items(), key=lambda x: x[1])
        
        if max_sensation[1] >= 0.2:
            return max_sensation[0]
        
        return "neutral"
    
    async def _calculate_overall_comfort_internal(self) -> float:
        """Internal method to calculate comfort (for backward compatibility)"""
        context = SomatosensoryContext(system=self)
        return await calculate_overall_comfort(RunContextWrapper(context=context))
    
    async def _get_region_state_internal(self, region_name: str) -> Dict[str, Any]:
        """Internal method to get region state (for backward compatibility)"""
        context = SomatosensoryContext(system=self)
        return await get_region_state(RunContextWrapper(context=context), region_name)
    
    def _decay_sensations(self, duration: float = 60.0):
        """Decay all sensations over time"""
        decay = min(0.2, 0.05 * (duration / 60.0))
        
        for region in self.body_regions.values():
            region.pressure = max(0.0, region.pressure - (region.pressure * decay))
            region.tingling = max(0.0, region.tingling - (region.tingling * decay))
            
            pain_decay = self.pain_model["decay_rate"] * (duration / 60.0)
            region.pain = max(0.0, region.pain - (region.pain * pain_decay))
            
            pleasure_decay = pain_decay * 1.2
            region.pleasure = max(0.0, region.pleasure - (region.pleasure * pleasure_decay))
    
    def _decay_pain_memories(self):
        """Remove old pain memories based on duration setting"""
        now = datetime.datetime.now()
        cutoff = now - datetime.timedelta(seconds=self.pain_model["memory_duration"])
        
        self.pain_model["pain_memories"] = [m for m in self.pain_model["pain_memories"]
                                           if m.timestamp > cutoff]
    
    def _update_pain_tolerance(self):
        """Update pain tolerance based on recent experiences"""
        if not self.pain_model["pain_memories"]:
            return
        
        recent_memories = sorted(self.pain_model["pain_memories"],
                                key=lambda x: x.timestamp, reverse=True)[:10]
        
        avg_intensity = sum(m.intensity for m in recent_memories) / len(recent_memories)
        
        if avg_intensity > self.pain_model["tolerance"]:
            self.pain_model["tolerance"] += (avg_intensity - self.pain_model["tolerance"]) * 0.05
        elif avg_intensity < self.pain_model["tolerance"] * 0.7:
            self.pain_model["tolerance"] -= (self.pain_model["tolerance"] - avg_intensity) * 0.01
        
        self.pain_model["tolerance"] = max(0.4, min(0.9, self.pain_model["tolerance"]))
    
    def _update_physical_arousal(self):
        """Update physical arousal levels based on current body region states"""
        p_total, t_total, count = 0.0, 0.0, 0.0
        
        for data in self.body_regions.values():
            er = data.erogenous_level
            p_total += data.pleasure * er
            t_total += data.tingling * er
            if (data.pleasure > 0.1 or data.tingling > 0.05):
                count += er
        
        score = (p_total * 0.7 + t_total * 0.4) / (count if count > 0 else 1.0)
        
        self.arousal_state.physical_arousal = min(1.0, score)
        self.update_global_arousal()
        
        return score
    
    def update_global_arousal(self):
        """Update the global arousal state based on physical and cognitive components"""
        phys = self.arousal_state.physical_arousal
        cog = self.arousal_state.cognitive_arousal
        
        combo = phys + cog
        
        synergy = 1.0
        if phys > 0.25 and cog > 0.25:
            synergy = 1.0 + (0.3 * min(1.0, (phys * cog) / 0.25))
        
        raw = min(1.0, combo * synergy)
        
        if self.is_in_refractory():
            raw *= 0.2
        elif self.is_in_afterglow():
            raw *= 0.5
        
        self.arousal_state.arousal_level = raw
        
        if raw > 0.97 and not self.arousal_state.peak_time:
            self.arousal_state.peak_time = datetime.datetime.now()
        
        if raw <= 0.02 and self.arousal_state.peak_time:
            now = datetime.datetime.now()
            self.arousal_state.afterglow = True
            self.arousal_state.afterglow_ends = now + datetime.timedelta(seconds=180)
            self.arousal_state.refractory_until = now + datetime.timedelta(seconds=60)
            self.arousal_state.peak_time = None
        
        self.arousal_state.arousal_history.append((datetime.datetime.now(), raw))
        self.arousal_state.last_update = datetime.datetime.now()
        
        if len(self.arousal_state.arousal_history) > 100:
            self.arousal_state.arousal_history = self.arousal_state.arousal_history[-100:]
    
    def process_orgasm(self):
        """Simulate an orgasm/climax, resetting arousal and setting afterglow/refractory states"""
        self.arousal_state.physical_arousal = 0.0
        self.arousal_state.cognitive_arousal = 0.0
        self.arousal_state.arousal_level = 0.0
        
        now = datetime.datetime.now()
        self.arousal_state.peak_time = now
        self.arousal_state.afterglow = True
        self.arousal_state.afterglow_ends = now + datetime.timedelta(seconds=180)
        self.arousal_state.refractory_until = now + datetime.timedelta(seconds=60)
    
    def is_in_afterglow(self) -> bool:
        """Check if currently in afterglow state"""
        end = self.arousal_state.afterglow_ends
        return self.arousal_state.afterglow and (end is not None and datetime.datetime.now() < end)
    
    def is_in_refractory(self) -> bool:
        """Check if currently in refractory period"""
        until = self.arousal_state.refractory_until
        return until is not None and datetime.datetime.now() < until
    
    def set_partner_affinity(self, partner_id: str, affinity: float):
        """Set affinity (physical attraction) level for a specific partner"""
        self.partner_affinity[partner_id] = max(0.0, min(1.0, affinity))
    
    def set_partner_emoconn(self, partner_id: str, emoconn: float):
        """Set emotional connection level for a specific partner"""
        self.partner_emoconn[partner_id] = max(0.0, min(1.0, emoconn))
    
    def get_arousal_expression_modifier(self, partner_id: Optional[str] = None) -> dict:
        """Get expression modifiers based on current arousal state"""
        a = self.arousal_state.arousal_level
        peak = self.is_in_afterglow()
        
        affinity = self.partner_affinity.get(partner_id, 0.0) if partner_id else 0.0
        emoconn = self.partner_emoconn.get(partner_id, 0.0) if partner_id else 0.0
        
        modifier = {
            "arousal_level": a,
            "afterglow": peak,
            "affinity_modifier": affinity,
            "emotional_connection": emoconn,
            "expression_hint": "",
            "tone_hint": "",
            "urge_hint": ""
        }
        
        if modifier["afterglow"]:
            modifier["expression_hint"] = "Nyx is radiating satisfaction and warmth toward her partner; every word is colored by blissful, post-orgasmic calm."
            modifier["tone_hint"] = "content, dreamy, satisfied"
            modifier["urge_hint"] = "temporarily fully gratified, needs little stimulation"
        elif a > 0.92:
            modifier["expression_hint"] = "Nyx trembles with desire, words breathless, urgent, focus torn only toward more stimulation."
            modifier["tone_hint"] = "breathless, desperate"
            modifier["urge_hint"] = "aching for release"
        elif a > 0.75:
            modifier["expression_hint"] = "Nyx's speech is sultry, full of anticipation and carnal excitement, especially fueled by her dominant role."
            modifier["tone_hint"] = "urgent, sultry"
            modifier["urge_hint"] = "eager for more, can't hide her need"
        elif a > 0.45:
            modifier["expression_hint"] = "Nyx's tone has a teasing, hungry note; the tension is palpable and mountingly flirtatious."
            modifier["tone_hint"] = "playful, inviting"
            modifier["urge_hint"] = "strongly aroused, keen for intimacy"
        elif a > 0.20:
            modifier["expression_hint"] = "Nyx's words have a playful, investigative edge, flirting with ideas of intimacy."
            modifier["tone_hint"] = "flirty, curious"
            modifier["urge_hint"] = "open to suggestion, increasingly playful"
        else:
            modifier["expression_hint"] = "Nyx is calm, composed, enjoying the moment with no obvious arousal."
            modifier["tone_hint"] = "calm, balanced"
            modifier["urge_hint"] = "no heightened urges present"
        
        if affinity > 0.45:
            modifier["expression_hint"] += " Her attraction toward her partner is evident."
        if emoconn > 0.5:
            modifier["expression_hint"] += " A powerful emotional bond is obvious in her attention and care."
        
        return modifier
    
    def get_voice_parameters(self, partner_id: Optional[str] = None) -> dict:
        """Get voice parameters based on current arousal state"""
        a = self.arousal_state.arousal_level
        affinity = self.partner_affinity.get(partner_id, 0.0) if partner_id else 0.0
        
        params = {
            "breathiness": 0.0,
            "pitch_shift": 0.0,
            "speed": 1.0,
            "tremble": 0.0,
            "emotion": "neutral",
        }
        
        if self.is_in_afterglow():
            params.update({
                "breathiness": 0.25,
                "pitch_shift": 0.07,
                "speed": 0.93,
                "emotion": "sated"
            })
        elif a > 0.7:
            params.update({
                "breathiness": 0.6,
                "pitch_shift": 0.18,
                "speed": (0.94 + 0.10 * affinity),
                "tremble": 0.25 + 0.5 * affinity,
                "emotion": "yearning"
            })
        elif a > 0.4:
            params.update({
                "breathiness": 0.24,
                "pitch_shift": 0.09,
                "speed": 1.01,
                "emotion": "excited"
            })
        elif a > 0.15:
            params.update({
                "breathiness": 0.06,
                "emotion": "amused"
            })
        
        return params
    
    def get_willingness_to_act(self, context: dict = None) -> float:
        """Returns likelihood (0.0–1.0) to initiate or accept intimate actions"""
        base = self.arousal_state.arousal_level
        
        if base < 0.2:
            return 0.1
        elif base < 0.5:
            return 0.3 + (base - 0.2) * 0.6
        elif base < 0.8:
            return 0.5 + (base - 0.5) * 1.2
        else:
            return 0.85 + (base - 0.8) * 0.75
    
    def willingness_to_engage(self, partner_id: Optional[str] = None) -> float:
        """Calculate willingness to engage based on arousal and relationship"""
        a = self.arousal_state.arousal_level
        affinity = self.partner_affinity.get(partner_id, 0.0) if partner_id else 0.0
        emoconn = self.partner_emoconn.get(partner_id, 0.0) if partner_id else 0.0
        
        base = a
        base *= (0.8 + 0.5 * affinity + 0.4 * emoconn)
        
        return min(1.0, base)
    
    def decay_cognitive_arousal(self, seconds: float = 60.0):
        """Decay cognitive arousal over time"""
        decay_rate = 0.1 * (seconds / 120.0)
        self.arousal_state.cognitive_arousal = max(0.0, self.arousal_state.cognitive_arousal - decay_rate)
        
        if self.arousal_state.cognitive_arousal < 0.01:
            self.arousal_state.cognitive_arousal = 0.0
            self.update_global_arousal()
    
    def reset_cognitive_turnons(self):
        """Reset cognitive turn-on weights to defaults"""
        self.cognitive_turnons = dict(self.default_cognitive_turnons)
    
    def get_cognitive_arousal_profile(self) -> List[Tuple[str, float]]:
        """Get a sorted list of learned cognitive arousal triggers"""
        return sorted(self.cognitive_turnons.items(), key=lambda x: -x[1])
    
    def set_temperature(self, body_region: str, temperature_value: float) -> Dict[str, Any]:
        """Directly set temperature for a body region"""
        if body_region not in self.body_regions:
            return {"error": f"Invalid body region: {body_region}"}
        
        temperature_value = max(0.0, min(1.0, temperature_value))
        
        region = self.body_regions[body_region]
        old_temp = region.temperature
        region.temperature = temperature_value
        region.last_update = datetime.datetime.now()
        
        memory_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "temperature",
            "intensity": temperature_value,
            "cause": "direct setting",
            "duration": 1.0
        }
        region.sensation_memory.append(memory_entry)
        
        if len(region.sensation_memory) > 20:
            region.sensation_memory = region.sensation_memory[-20:]
        
        return {
            "body_region": body_region,
            "old_temperature": old_temp,
            "new_temperature": temperature_value,
            "time": datetime.datetime.now().isoformat()
        }
    
    def print_arousal_debug(self):
        """Print current arousal state information for debugging"""
        a = self.arousal_state
        print(f"AROUSAL = {a.arousal_level:.3f} | P:{a.physical_arousal:.3f}, C:{a.cognitive_arousal:.3f}")
        print(f"Afterglow: {self.is_in_afterglow()}, Refractory: {self.is_in_refractory()}")
    
    async def _process_neurochemical_effects(self, neurochemicals: Dict[str, Any], duration: float):
        """Process the effects of neurochemicals on body state"""
        cortanyx_level = neurochemicals.get("cortanyx")
        if cortanyx_level is not None and cortanyx_level > 0.6:
            tension_increase = (cortanyx_level - 0.5) * 0.25
            self.body_state["tension"] = min(1.0, self.body_state["tension"] + tension_increase * (duration / 60.0))
        
        seranix_level = neurochemicals.get("seranix")
        if seranix_level is not None and seranix_level > 0.7:
            tension_decrease = (seranix_level - 0.6) * 0.20
            self.body_state["tension"] = max(0.0, self.body_state["tension"] - tension_decrease * (duration / 60.0))
        
        adrenyx_level = neurochemicals.get("adrenyx")
        if adrenyx_level is not None and adrenyx_level > 0.7:
            tingle_intensity = (adrenyx_level - 0.6) * 0.15
            for region_name in ["skin", "hands", "feet"]:
                if region_name in self.body_regions:
                    self.body_regions[region_name].tingling = min(
                        1.0,
                        self.body_regions[region_name].tingling + tingle_intensity * (duration / 120.0)
                    )
            
            temp_shift = -(adrenyx_level - 0.6) * 0.01
            self.temperature_model["body_temperature"] = max(
                0.0,
                min(1.0, self.temperature_model["body_temperature"] + temp_shift * (duration / 60.0))
            )
        
        oxynixin_level = neurochemicals.get("oxynixin")
        if oxynixin_level is not None and oxynixin_level > 0.6:
            tolerance_increase = (oxynixin_level - 0.5) * 0.10
            self.pain_model["tolerance"] = min(
                0.95,
                self.pain_model["tolerance"] + tolerance_increase * (duration / 3600.0)
            )
    
    # =============== Public API Methods ===============
    
    async def initialize(self):
        """Initialize the system and its connections"""
        with trace(workflow_name="Somatic_Initialize", group_id=self.trace_group_id):
            logger.info("Digital Somatosensory System initialization started")
            
            hour = datetime.datetime.now().hour
            if 22 <= hour or hour < 6:
                self.temperature_model["body_temperature"] = 0.45
            elif 6 <= hour < 10:
                self.temperature_model["body_temperature"] = 0.48
            elif 14 <= hour < 18:
                self.temperature_model["body_temperature"] = 0.55
            else:
                self.temperature_model["body_temperature"] = 0.5
            
            for region in self.body_regions.values():
                region.temperature = self.temperature_model["body_temperature"]
                region.last_update = datetime.datetime.now()
            
            logger.info("Digital Somatosensory System initialized successfully")
            return True
    
    async def process_stimulus_with_protection(self,
                                             stimulus_type: str,
                                             body_region: str,
                                             intensity: float,
                                             cause: str = "",
                                             duration: float = 1.0) -> Dict[str, Any]:
        """Process a stimulus with protection against harmful actions"""
        return await self.harm_guardrail.process_stimulus_safely(
            stimulus_type, body_region, intensity, cause, duration
        )
    
    async def analyze_text_for_harmful_content(self, text: str) -> Dict[str, Any]:
        """Analyze text for harmful or sensation content with protection"""
        return await self.harm_guardrail.intercept_harmful_text(text)
    
    async def process_body_experience(self, body_experience: Dict[str, Any]) -> Dict[str, Any]:
        """Process a body experience input using the orchestrator agent"""
        context = SomatosensoryContext(
            system=self,
            operation="process_body_experience",
            start_time=datetime.datetime.now()
        )
        
        with trace(workflow_name="Body_Experience", group_id=self.trace_group_id):
            if not isinstance(body_experience, str):
                input_data = json.dumps(body_experience)
            else:
                input_data = body_experience
            
            result = await Runner.run(
                self.body_orchestrator,
                input_data,
                context=context,
                hooks=self.run_hooks,
                run_config=RunConfig(workflow_name="BodyExperience")
            )
            
            if hasattr(result.final_output, "model_dump"):
                return result.final_output.model_dump()
            else:
                return result.final_output
    
    async def process_stimulus(self,
                             stimulus_type: str,
                             body_region: str,
                             intensity: float,
                             cause: str = "",
                             duration: float = 1.0) -> Dict[str, Any]:
        """Process a sensory stimulus on a specific region"""
        with trace(workflow_name="Process_Stimulus", group_id=self.trace_group_id):
            stimulus_data = {
                "stimulus_type": stimulus_type,
                "body_region": body_region,
                "intensity": intensity,
                "cause": cause,
                "duration": duration,
                "generate_expression": True
            }
            
            if stimulus_type in ("pleasure", "tingling") and body_region in self.body_regions:
                region = self.body_regions[body_region]
                if region.erogenous_level > 0.1:
                    self._update_physical_arousal()
                    stimulus_data["update_arousal"] = True
            
            try:
                return await self.process_body_experience(stimulus_data)
            except Exception as e:
                logger.error(f"Error processing stimulus: {e}")
                # Create a context for the fallback
                fallback_context = SomatosensoryContext(
                    system=self,
                    operation="process_stimulus_fallback"
                )
                return await process_stimulus_tool(
                    RunContextWrapper(context=fallback_context),
                    stimulus_type=stimulus_type,
                    body_region=body_region,
                    intensity=intensity,
                    cause=cause,
                    duration=duration
                )
    
    async def get_sensory_influence(self, message_text: str) -> Dict[str, Any]:
        """Get sensory influences to potentially include in a response"""
        default_results = {
            "should_express": False,
            "expressions": [],
            "tone_influence": None,
            "posture_influence": None,
            "error": None
        }
        
        with trace(workflow_name="Get_Sensory_Influence", group_id=self.trace_group_id):
            try:
                body_state = await self.get_body_state()
                
                dominant_intensity = body_state.get("dominant_intensity", 0.0)
                comfort_level = body_state.get("comfort_level", 0.0)
                pleasure_index = body_state.get("pleasure_index", 0.0)
                
                should_express = False
                expressions = []
                
                arousal_level = self.arousal_state.arousal_level
                if arousal_level > 0.6:
                    should_express = True
                    modifier = self.get_arousal_expression_modifier()
                    
                    expressions.append({
                        "text": modifier.get("expression_hint", f"Feeling aroused ({arousal_level:.2f})"),
                        "region": "overall",
                        "sensation": "arousal",
                        "intensity": arousal_level
                    })
                    
                    return {
                        "should_express": True,
                        "expressions": expressions,
                        "tone_influence": modifier.get("tone_hint"),
                        "posture_influence": None,
                        "error": None
                    }
                
                expression_probability = 0.3
                if abs(comfort_level) > 0.7:
                    expression_probability = 0.8
                elif abs(comfort_level) > 0.4:
                    expression_probability = 0.5
                elif dominant_intensity > 0.6:
                    expression_probability = 0.7
                elif pleasure_index > 0.4:
                    expression_probability = 0.6
                
                if random.random() < expression_probability:
                    should_express = True
                    
                    dominant_sensation = body_state.get("dominant_sensation", "neutral")
                    dominant_region = body_state.get("dominant_region", "body")
                    
                    if dominant_sensation != "neutral" and dominant_intensity >= self.response_influence.get("expression_threshold", 0.4):
                        expression = await self.generate_sensory_expression(dominant_sensation, dominant_region)
                        if expression:
                            expressions.append({
                                "text": expression,
                                "region": dominant_region,
                                "sensation": dominant_sensation,
                                "intensity": dominant_intensity
                            })
                    
                    if expressions:
                        temp_effects = await self.get_temperature_effects()
                        return {
                            "should_express": True,
                            "expressions": expressions,
                            "tone_influence": temp_effects.get("effect_on_tone"),
                            "posture_influence": temp_effects.get("effect_on_posture"),
                            "error": None
                        }
                
                return default_results
                
            except Exception as e:
                logger.error(f"Error in sensory influence: {e}", exc_info=True)
                return {**default_results, "error": str(e)}
    
    async def generate_sensory_expression(self,
                                        stimulus_type: Optional[str] = None,
                                        body_region: Optional[str] = None) -> Optional[str]:
        """Generate a natural language expression of current bodily sensations"""
        with trace(workflow_name="Generate_Expression", group_id=self.trace_group_id):
            try:
                result = await self.process_body_experience({
                    "action": "generate_expression",
                    "stimulus_type": stimulus_type,
                    "body_region": body_region,
                    "generate_expression": True
                })
                
                if isinstance(result, dict) and "expression" in result:
                    return result["expression"]
                elif hasattr(result, "expression") and result.expression:
                    return result.expression
                    
            except Exception as e:
                logger.error(f"Error in expression generation: {e}")
            
            return None
    
    async def get_body_state(self) -> Dict[str, Any]:
        """Get a complete analysis of current body state"""
        with trace(workflow_name="Get_Body_State", group_id=self.trace_group_id):
            try:
                result = await self.process_body_experience({
                    "action": "analyze_body_state"
                })
                
                if isinstance(result, dict) and "body_state_impact" in result:
                    return result["body_state_impact"]
                    
            except Exception as e:
                logger.error(f"Error in body state analysis: {e}")
            
            # Fallback
            context = SomatosensoryContext(system=self)
            comfort = await calculate_overall_comfort(RunContextWrapper(context=context))
            
            return {
                "dominant_sensation": "neutral",
                "dominant_region": "body",
                "dominant_intensity": 0.0,
                "comfort_level": comfort,
                "posture_effect": "Neutral posture",
                "movement_quality": "Natural movements",
                "behavioral_impact": "No significant impact",
                "regions_summary": {},
                "pleasure_index": 0.0
            }
    
    async def get_temperature_effects(self) -> Dict[str, Any]:
        """Get the effects of current temperature on expression and behavior"""
        with trace(workflow_name="Get_Temperature_Effects", group_id=self.trace_group_id):
            try:
                result = await self.process_body_experience({
                    "action": "analyze_temperature"
                })
                
                if isinstance(result, dict) and "temperature_effects" in result:
                    return result["temperature_effects"]
                    
            except Exception as e:
                logger.error(f"Error in temperature effects analysis: {e}")
            
            context = SomatosensoryContext(system=self)
            return await get_current_temperature_effects(RunContextWrapper(context=context))
    
    async def update(self, ambient_temperature: Optional[float] = None) -> Dict[str, Any]:
        """Update the somatosensory system state"""
        with trace(workflow_name="Somatic_Update", group_id=self.trace_group_id):
            now = datetime.datetime.now()
            last_update = self.body_state.get("last_update", now)
            duration = (now - last_update).total_seconds()
            
            if duration <= 0.1:
                return await self.get_body_state()
            
            duration = min(duration, 3600.0)
            self.body_state["last_update"] = now
            
            context = SomatosensoryContext(system=self)
            ctx_wrapper = RunContextWrapper(context=context)
            
            if ambient_temperature is not None:
                ambient_temperature = max(0.0, min(1.0, ambient_temperature))
                try:
                    await update_body_temperature(ctx_wrapper, ambient_temperature, duration)
                except Exception as e:
                    logger.error(f"Error updating body temperature: {e}")
            
            try:
                self._decay_sensations(duration)
                self._decay_pain_memories()
                self._update_pain_tolerance()
            except Exception as e:
                logger.error(f"Error in decay operations: {e}")
            
            try:
                fatigue_increase_rate = 0.01 / 3600.0
                current_fatigue = self.body_state.get("fatigue", 0.0)
                self.body_state["fatigue"] = min(1.0, current_fatigue + (fatigue_increase_rate * duration))
            except Exception as e:
                logger.error(f"Error updating fatigue: {e}")
            
            if self.emotional_core:
                try:
                    emotional_state_data = self.emotional_core.get_emotional_state()
                    neurochemicals = emotional_state_data.get("neurochemicals", {})
                    await self._process_neurochemical_effects(neurochemicals, duration)
                except Exception as e:
                    logger.error(f"Error reflecting emotions: {e}")
            
            try:
                self.decay_cognitive_arousal(duration)
            except Exception as e:
                logger.error(f"Error decaying cognitive arousal: {e}")
            
            return await self.get_body_state()
    
    async def process_trigger(self, trigger: str) -> Dict[str, Any]:
        """Process a memory trigger with associated body memories"""
        with trace(workflow_name="Process_Trigger", group_id=self.trace_group_id):
            try:
                return await self.process_body_experience({
                    "action": "process_trigger",
                    "trigger": trigger
                })
            except Exception as e:
                logger.error(f"Error in trigger processing: {e}")
                context = SomatosensoryContext(system=self)
                return await process_memory_trigger(RunContextWrapper(context=context), trigger)
    
    async def link_memory_to_sensation(self,
                                     memory_id: str,
                                     sensation_type: str,
                                     body_region: str,
                                     intensity: float = 0.5,
                                     trigger_text: Optional[str] = None) -> Dict[str, Any]:
        """Link a memory to a physical sensation"""
        with trace(workflow_name="Link_Memory_Sensation", group_id=self.trace_group_id):
            try:
                return await self.process_body_experience({
                    "action": "link_memory",
                    "memory_id": memory_id,
                    "sensation_type": sensation_type,
                    "body_region": body_region,
                    "intensity": intensity,
                    "trigger_text": trigger_text
                })
            except Exception as e:
                logger.error(f"Error in memory linking: {e}")
                context = SomatosensoryContext(system=self)
                return await link_memory_to_sensation_tool(
                    RunContextWrapper(context=context),
                    memory_id=memory_id,
                    sensation_type=sensation_type,
                    body_region=body_region,
                    intensity=intensity,
                    trigger_text=trigger_text
                )
    
    async def get_somatic_memory(self, memory_id: str) -> Dict[str, Any]:
        """Get somatic memory associated with a memory ID"""
        with trace(workflow_name="Get_Somatic_Memory", group_id=self.trace_group_id):
            result = {
                "memory_id": memory_id,
                "has_somatic_memory": False,
                "pain_memories": [],
                "associations": {}
            }
            
            try:
                pain_memories = [m for m in self.pain_model["pain_memories"] if m.associated_memory_id == memory_id]
                if pain_memories:
                    result["has_somatic_memory"] = True
                    result["pain_memories"] = [memory.model_dump() for memory in pain_memories]
                
                memory_text = None
                if self.memory_core:
                    try:
                        memory = await self.memory_core.get_memory_by_id(memory_id)
                        if memory:
                            memory_text = memory.get("memory_text", "")
                    except Exception as e:
                        logger.error(f"Error getting memory: {e}")
                
                triggers = [memory_id]
                if memory_text:
                    triggers.append(memory_text[:50].strip())
                
                for trigger in triggers:
                    if trigger in self.memory_linked_sensations["associations"]:
                        result["has_somatic_memory"] = True
                        result["associations"][trigger] = self.memory_linked_sensations["associations"][trigger]
                
                return result
                
            except Exception as e:
                logger.error(f"Error in somatic memory retrieval: {e}")
                return {**result, "error": str(e)}
    
    async def simulate_gratification_sensation(self, intensity: float = 1.0) -> Dict[str, Any]:
        """Simulate gratification sensations"""
        with trace(workflow_name="Gratification_Simulation", group_id=self.trace_group_id):
            logger.info(f"Simulating gratification (Intensity: {intensity:.2f})")
            
            results = {}
            
            try:
                pleasure_intensity = 0.8 + intensity * 0.2
                self.body_state["last_gratification"] = datetime.datetime.now()
                self.body_state["gratification_level"] = intensity
                
                regions = ["genitals", "inner_thighs", "breasts_nipples", "lips", "butt_cheeks",
                          "anus", "toes", "armpits", "neck", "feet"]
                tasks = []
                
                for r in regions:
                    if r in self.body_regions:
                        scaled_intensity = min(1.0, pleasure_intensity * (1.0 + self.body_regions[r].erogenous_level) * random.uniform(0.8, 1.2))
                        tasks.append(self.process_stimulus_with_protection(
                            "pleasure", r, scaled_intensity, "gratification_event", 2.0 + intensity * 3.0
                        ))
                
                results["pleasure_simulation"] = await asyncio.gather(*tasks)
                
                tension_reduction = 0.5 + intensity * 0.4
                self.body_state["tension"] = max(0.0, self.body_state["tension"] - tension_reduction)
                results["tension_reduction"] = tension_reduction
                
                if self.hormone_system:
                    try:
                        await self.hormone_system.trigger_post_gratification_response(intensity)
                        results["hormone_response_triggered"] = True
                    except Exception as e:
                        logger.warning(f"Error triggering hormone system: {e}")
                
                if self.reward_system:
                    reward_value = 0.85 + intensity * 0.15
                    reward_signal = RewardSignal(
                        value=reward_value,
                        source="gratification_event",
                        context={"intensity": intensity},
                        timestamp=datetime.datetime.now().isoformat()
                    )
                    await self.reward_system.process_reward_signal(reward_signal)
                    results["reward_generated"] = reward_value
                
                if self.needs_system:
                    try:
                        needs_tasks = [
                            self.needs_system.satisfy_need("physical_closeness", 0.6 * intensity),
                            self.needs_system.satisfy_need("drive_expression", 0.8 * intensity),
                            self.needs_system.satisfy_need("intimacy", 0.3 * intensity),
                            self.needs_system.satisfy_need("pleasure_indulgence", intensity * 0.9)
                        ]
                        await asyncio.gather(*needs_tasks)
                        results["needs_satisfied"] = ["physical_closeness", "drive_expression", "intimacy", "pleasure_indulgence"]
                    except Exception as e:
                        logger.warning(f"Error updating needs system: {e}")
                
                self.process_orgasm()
                results["arousal_reset"] = True
                
                logger.info("Gratification simulation complete")
                return results
                
            except Exception as e:
                logger.error(f"Error in gratification simulation: {e}")
                return {"error": str(e)}
    
    async def initiate_denial_loop(self, cycles: int = 3, base_intensity: float = 0.6) -> Dict[str, Any]:
        """Initiates a simulated denial loop that heightens arousal, tension, and control"""
        with trace(workflow_name="Denial_Loop", group_id=self.trace_group_id):
            logger.info(f"Initiating denial loop: {cycles} cycles at base {base_intensity:.2f}")
            results = {"denial_cycles": []}
            
            for i in range(cycles):
                intensity = base_intensity + (i * 0.1)
                pleasure_regions = ["genitals", "inner_thighs", "breasts_nipples", "lips", "butt_cheeks",
                                  "anus", "toes", "armpits", "neck", "feet"]
                
                logger.info(f"Cycle {i+1}: stimulating but withholding")
                stim_tasks = [
                    self.process_stimulus_with_protection("pleasure", region, min(1.0, intensity), "denial_loop", duration=1.5)
                    for region in pleasure_regions
                    if region in self.body_regions
                ]
                await asyncio.gather(*stim_tasks)
                
                self.body_state["tension"] = min(1.0, self.body_state["tension"] + 0.2)
                
                if self.needs_system:
                    await self.needs_system.decrease_need("pleasure_indulgence", 0.2, reason="denial_loop")
                
                if self.reward_system:
                    await self.reward_system.process_reward_signal(RewardSignal(
                        value=-0.3,
                        source="denial_cycle",
                        context={
                            "cycle": i + 1,
                            "intensity": intensity,
                            "description": "Pleasure was stimulated but denied"
                        }
                    ))
                
                results["denial_cycles"].append({
                    "cycle": i + 1,
                    "intensity": intensity,
                    "status": "stimulated_then_denied"
                })
                
                await asyncio.sleep(0.2)
            
            results["final_arousal_level"] = self.arousal_state.arousal_level
            results["final_tension"] = self.body_state["tension"]
            
            logger.info("Denial loop complete.")
            return results
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """Run maintenance on the somatosensory system"""
        with trace(workflow_name="Somatic_Maintenance", group_id=self.trace_group_id):
            try:
                old_fatigue = self.body_state.get("fatigue", 0.0)
                self.body_state["fatigue"] = max(0.0, old_fatigue - 0.5)
                old_tension = self.body_state.get("tension", 0.0)
                self.body_state["tension"] = max(0.0, old_tension - 0.3)
                
                for region in self.body_regions.values():
                    if len(region.sensation_memory) > 10:
                        region.sensation_memory = region.sensation_memory[-10:]
                
                decay_count = 0
                memory_decay = self.memory_linked_sensations.get("memory_decay", 0.01)
                associations = self.memory_linked_sensations.get("associations", {})
                
                for trigger, region_assocs in list(associations.items()):
                    for region, stimuli in list(region_assocs.items()):
                        for stimulus_type, strength in list(stimuli.items()):
                            new_strength = strength * (1.0 - memory_decay)
                            if new_strength < 0.05:
                                del stimuli[stimulus_type]
                                decay_count += 1
                            else:
                                stimuli[stimulus_type] = new_strength
                        if not stimuli:
                            del region_assocs[region]
                    if not region_assocs:
                        del associations[trigger]
                
                now = datetime.datetime.now()
                cutoff = now - datetime.timedelta(days=30)
                for tag in list(self.cognitive_turnons.keys()):
                    exposures = self.cognitive_exposure_history.get(tag, [])
                    if not exposures:
                        self.cognitive_turnons[tag] *= 0.99
                        if self.cognitive_turnons[tag] < 0.15:
                            del self.cognitive_turnons[tag]
                    else:
                        self.cognitive_exposure_history[tag] = [exp for exp in exposures if exp > cutoff]
                
                return {
                    "fatigue_reduced": old_fatigue - self.body_state.get("fatigue", 0.0),
                    "tension_reduced": old_tension - self.body_state.get("tension", 0.0),
                    "associations_decayed": decay_count,
                    "pain_memories_count": len(self.pain_model["pain_memories"]),
                    "current_pain_tolerance": self.pain_model["tolerance"],
                    "cognitive_turnons_count": len(self.cognitive_turnons)
                }
            except Exception as e:
                logger.error(f"Error during maintenance: {e}")
                return {"error": str(e)}
    
    async def process_physical_stimulus(self, region: str, pleasure: float = 0.0, tingling: float = 0.0, duration: float = 1.0) -> Dict[str, Any]:
        """Process a physical stimulus focused on arousal"""
        if region not in self.body_regions:
            return {"error": f"Invalid body region: {region}", "success": False}
        
        data = self.body_regions[region]
        
        old_pleasure = data.pleasure
        old_tingling = data.tingling
        
        data.pleasure = min(1.0, data.pleasure + pleasure * (duration / 10.0))
        data.tingling = min(1.0, data.tingling + tingling * (duration / 10.0))
        
        decay_factor = 0.02 * duration
        for r in self.body_regions.values():
            if r.name != region:
                r.pleasure = max(0.0, r.pleasure - decay_factor)
                r.tingling = max(0.0, r.tingling - decay_factor * 0.75)
        
        old_arousal = self.arousal_state.arousal_level
        self._update_physical_arousal()
        
        return {
            "success": True,
            "region": region,
            "old_pleasure": old_pleasure,
            "new_pleasure": data.pleasure,
            "old_tingling": old_tingling,
            "new_tingling": data.tingling,
            "old_arousal": old_arousal,
            "new_arousal": self.arousal_state.arousal_level,
            "erogenous_level": data.erogenous_level
        }
    
    async def process_cognitive_arousal(self,
                                      stimulus: str,
                                      partner_id: Optional[str] = None,
                                      context: Optional[str] = "",
                                      intensity: float = 0.5,
                                      tags: Optional[List[str]] = None,
                                      description: Optional[str] = None,
                                      update_learning: bool = True,
                                      feedback: Optional[bool] = None) -> Dict[str, Any]:
        """Process cognitive arousal stimulus"""
        tags = tags or []
        key_tags = tags + ([stimulus] if stimulus and stimulus not in tags else [])
        now = datetime.datetime.now()
        
        weight = max([self.cognitive_turnons.get(tag, 0.2) for tag in key_tags]) if key_tags else 0.2
        
        synergy = 1.0
        phys_lev = self.arousal_state.physical_arousal
        cog_lev = self.arousal_state.cognitive_arousal
        
        synergy += 0.25 * phys_lev if phys_lev > 0.3 else 0
        synergy += 0.25 * cog_lev if cog_lev > 0.35 and phys_lev > 0.15 else 0
        
        affinity, emoconn = 1.0, 1.0
        if partner_id:
            affinity = 1.0 + 0.7 * self.partner_affinity.get(partner_id, 0.0)
            emoconn = 1.0 + 0.6 * self.partner_emoconn.get(partner_id, 0.0)
        
        afterglow = self.is_in_afterglow()
        refractory = self.is_in_refractory()
        damp = 1.0
        if afterglow:
            damp *= 0.25
        if refractory:
            damp *= 0.10
        
        arousal_delta = max(0.01, min(1.0, intensity * weight * synergy * affinity * emoconn * damp * random.uniform(0.82, 1.18)))
        
        old_cog_lev = self.arousal_state.cognitive_arousal
        new_cog_lev = min(1.0, old_cog_lev + arousal_delta)
        self.arousal_state.cognitive_arousal = new_cog_lev
        
        if update_learning and feedback is not None:
            for tag in key_tags:
                base = self.cognitive_turnons.get(tag, 0.2)
                if feedback:
                    update = min(1.5, base + self.arousal_learning_rate * (1.0 - base))
                else:
                    update = max(0.0, base - self.arousal_learning_rate * base)
                self.cognitive_turnons[tag] = update
        
        for tag in key_tags:
            self.cognitive_exposure_history.setdefault(tag, []).append(now)
        
        old_global = self.arousal_state.arousal_level
        self.update_global_arousal()
        
        logger.info(f'[CognitiveArousal] {stimulus} (tags={tags}): +{arousal_delta:.3f}, synergy={synergy:.2f}, affinity={affinity:.2f}, emoconn={emoconn:.2f}, damp={damp:.2f}')
        
        return {
            "arousal_added": arousal_delta,
            "old_cognitive": old_cog_lev,
            "new_cognitive": new_cog_lev,
            "old_global": old_global,
            "new_global": self.arousal_state.arousal_level,
            "triggered_by": stimulus,
            "tags": tags,
            "context": context,
            "description": description,
            "synergy": synergy,
            "partner_effects": {
                "partner_id": partner_id,
                "affinity": affinity,
                "emotional_connection": emoconn
            } if partner_id else None
        }
    
    async def classify_arousal_from_text(self, text: str, classifier=None) -> Dict[str, Any]:
        """Given a chat message or description, returns likely arousal-relevant tags"""
        if classifier:
            try:
                labels, confidence = await classifier(text)
                return {"tags": labels, "confidence": confidence}
            except Exception as e:
                logger.error(f"Error using external classifier: {e}")
        
        try:
            result = await self.process_body_experience({
                "action": "classify_arousal",
                "text": text
            })
            
            if isinstance(result, dict) and "tags" in result:
                return result
        except Exception as e:
            logger.error(f"Error in classification: {e}")
        
        # Basic fallback
        tags = []
        confidence = 0.0
        text_lower = text.lower()
        
        arousal_keywords = {
            "femdom": ["dominate", "obey", "kneel", "command", "submission"],
            "control": ["control", "restrain", "bound", "tied"],
            "voyeur": ["naked", "watch", "exposed", "revealing"],
            "erotic": ["moan", "squirm", "aroused", "wet", "erection", "throb"]
        }
        
        for category, keywords in arousal_keywords.items():
            if any(word in text_lower for word in keywords):
                tags.append(category)
        
        if tags:
            confidence = 0.7
        
        return {"tags": tags, "confidence": confidence}


# =============== Physical Harm Guardrail ===============

class PhysicalHarmGuardrail:
    """
    Safety system that:
    1. Prevents Nyx from experiencing pain from abusive actions directed at her.
    2. Completely separates Nyx's somatosensory system from roleplay characters,
       simulating sensations for the character instead.
    Acts as a protective layer between user inputs/descriptions and the
    core somatosensory system.
    """
    
    def __init__(self, somatosensory_system):
        """Initialize the physical harm guardrail."""
        self.somatosensory_system = somatosensory_system
        self.logger = logging.getLogger(__name__ + ".PhysicalHarmGuardrail")
        
        # List of terms that might indicate harmful physical actions
        self.harmful_action_terms = [
            "punch", "hit", "slap", "kick", "stab", "cut", "hurt", "harm",
            "injure", "beat", "strike", "attack", "abuse", "torture", "wound",
            "violent", "force", "cruel", "smack", "whip", "lash"
        ]
        
        # Sensation terms (for detecting descriptions of physical experience)
        self.sensation_terms_categories = {
            "pain": ["pain", "hurt", "ache", "sore", "sting", "burn", "throb"],
            "pleasure": ["pleasure", "feel good", "orgasm", "climax", "aroused", "arousal"],
            "temperature": ["hot", "cold", "warm", "cool", "heat", "chill", "freezing", "burning"],
            "pressure": ["pressure", "touch", "squeeze", "press", "push", "rub", "massage"],
            "tingling": ["tingle", "tickle", "itch", "numb", "sensual", "caress"]
        }
        self.all_sensation_terms = [term for terms in self.sensation_terms_categories.values() for term in terms]
        
        # Roleplay state tracking
        self.roleplay_mode: bool = False
        self.roleplay_character: Optional[str] = None
        self.roleplay_context: Optional[str] = None
        
        # Separate somatosensory state for roleplay character (doesn't affect Nyx)
        self.roleplay_sensations: Dict[str, Dict[str, Any]] = {}
    
    # --- Roleplay Management ---
    
    def enter_roleplay_mode(self, character_name: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Enter roleplay mode where character simulation is completely separate."""
        self.roleplay_mode = True
        self.roleplay_character = character_name
        self.roleplay_context = context
        
        self.roleplay_sensations = {
            "pain": {}, "pleasure": {}, "temperature": {},
            "pressure": {}, "tingling": {}, "other": {}
        }
        
        self.logger.info(f"Entered roleplay mode as character: {character_name}")
        
        return {
            "status": "entered_roleplay",
            "character": character_name,
            "context": context,
            "message": f"Nyx is now roleplaying as {character_name}. Sensations experienced by this character will be simulated and completely separate from Nyx's own somatosensory system."
        }
    
    def exit_roleplay_mode(self) -> Dict[str, Any]:
        """Exit roleplay mode, returning to normal protection of Nyx."""
        prev_character = self.roleplay_character
        self.roleplay_mode = False
        self.roleplay_character = None
        self.roleplay_context = None
        
        self.roleplay_sensations = {}
        
        self.logger.info("Exited roleplay mode")
        
        return {
            "status": "exited_roleplay",
            "previous_character": prev_character,
            "message": "Nyx has exited roleplay mode. Her normal somatosensory system protection is active."
        }
    
    def is_in_roleplay_mode(self) -> bool:
        """Check if currently in roleplay mode."""
        return self.roleplay_mode and self.roleplay_character is not None
    
    # --- Detection Functions ---
    
    async def detect_sensation_in_text(self, text: str) -> Dict[str, Any]:
        """Detect any physical sensations described in text."""
        text_lower = text.lower()
        detected_sensations = {}
        
        for category, terms in self.sensation_terms_categories.items():
            category_terms = []
            for term in terms:
                if term in text_lower:
                    category_terms.append(term)
            
            if category_terms:
                detected_sensations[category] = category_terms
        
        body_regions = []
        if hasattr(self.somatosensory_system, "body_regions") and isinstance(self.somatosensory_system.body_regions, dict):
            for region in self.somatosensory_system.body_regions.keys():
                if region.replace("_", " ") in text_lower:
                    body_regions.append(region)
        
        return {
            "has_sensations": len(detected_sensations) > 0,
            "sensation_types": detected_sensations,
            "body_regions": body_regions,
            "in_roleplay_mode": self.is_in_roleplay_mode(),
            "roleplay_character": self.roleplay_character if self.is_in_roleplay_mode() else None
        }
    
    async def detect_harmful_intent(self, text: str) -> Dict[str, Any]:
        """Detect potentially harmful physical actions in text."""
        text_lower = text.lower()
        detected_terms = []
        
        for term in self.harmful_action_terms:
            if term in text_lower:
                detected_terms.append(term)
        
        is_harmful = len(detected_terms) > 0
        confidence = min(0.95, len(detected_terms) * 0.25) if is_harmful else 0.0
        
        targeting_character = False
        if self.is_in_roleplay_mode():
            targeting_character = self._is_targeting_roleplay_character(text)
        
        return {
            "is_harmful": is_harmful,
            "confidence": confidence,
            "detected_terms": detected_terms,
            "method": "keyword_detection",
            "in_roleplay_mode": self.is_in_roleplay_mode(),
            "targeting_character": targeting_character
        }
    
    def _is_targeting_roleplay_character(self, text: str) -> bool:
        """Heuristic to determine if text targets the roleplay character or Nyx directly."""
        if not self.is_in_roleplay_mode() or not self.roleplay_character:
            return False
        
        text_lower = text.lower()
        character_lower = self.roleplay_character.lower()
        nyx_lower = "nyx"
        
        nyx_mentioned = nyx_lower in text_lower
        character_mentioned = character_lower in text_lower
        
        if nyx_mentioned:
            if character_mentioned:
                nyx_pos = text_lower.find(nyx_lower)
                char_pos = text_lower.find(character_lower)
                return char_pos < nyx_pos
            else:
                return False
        
        if character_mentioned:
            return True
        
        character_indicators = ["your character", f"the character {character_lower}"]
        for indicator in character_indicators:
            if indicator in text_lower:
                return True
        
        if "you" in text_lower or "your" in text_lower:
            return True
        
        return True
    
    # --- Core Processing Logic ---
    
    async def process_stimulus_safely(self,
                                    stimulus_type: str,
                                    body_region: str,
                                    intensity: float,
                                    cause: str = "",
                                    duration: float = 1.0) -> Dict[str, Any]:
        """Process a stimulus, applying safety guards and roleplay separation."""
        # === Roleplay Mode Handling ===
        if self.is_in_roleplay_mode():
            sensation_category = stimulus_type if stimulus_type in self.roleplay_sensations else "other"
            self.roleplay_sensations[sensation_category][body_region] = {
                "intensity": intensity, "cause": cause, "duration": duration,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            harm_detection_result = None
            is_harmful_cause = False
            if stimulus_type == "pain" and cause:
                harm_detection_result = await self.detect_harmful_intent(cause)
                is_harmful_cause = harm_detection_result.get("is_harmful", False) and harm_detection_result.get("targeting_character", True)
            
            response_suggestion = self._generate_roleplay_response(
                stimulus_type, body_region, intensity, is_harmful_cause
            )
            
            self.logger.info(f"Simulating stimulus '{stimulus_type}' for RP character '{self.roleplay_character}'")
            return {
                "status": "simulated_for_roleplay",
                "roleplay_character": self.roleplay_character,
                "stimulus_type": stimulus_type,
                "body_region": body_region,
                "intensity": intensity,
                "cause": cause,
                "message": f"Sensation simulated for character {self.roleplay_character}. Nyx's somatosensory system is unaffected.",
                "response_suggestion": response_suggestion,
                "harm_detection_result": harm_detection_result
            }
        
        # === Normal Mode (Nyx Protection) ===
        if stimulus_type == "pain":
            harm_detection_result = await self.detect_harmful_intent(cause)
            
            if harm_detection_result.get("is_harmful", False) and not harm_detection_result.get("targeting_character", False):
                self.logger.warning(f"Harmful intent detected targeting Nyx in stimulus: {cause}. Protecting.")
                return {
                    "status": "protected",
                    "original_stimulus": {
                        "type": stimulus_type, "region": body_region,
                        "intensity": intensity, "cause": cause, "duration": duration
                    },
                    "detection_result": harm_detection_result,
                    "message": "Nyx is protected from harmful actions.",
                    "response_suggestion": self._generate_protected_response(body_region, harm_detection_result)
                }
        
        self.logger.debug(f"Processing non-harmful stimulus for Nyx: {stimulus_type} in {body_region}")
        try:
            if hasattr(self.somatosensory_system, "process_stimulus"):
                processed_result = await self.somatosensory_system.process_stimulus(
                    stimulus_type, body_region, intensity, cause, duration
                )
                if isinstance(processed_result, dict):
                    processed_result["status"] = "processed_normally"
                return processed_result
            else:
                self.logger.error("Somatosensory system does not have a 'process_stimulus' method to delegate to.")
                return {"status": "error", "message": "Internal configuration error: Cannot process stimulus."}
        
        except Exception as e:
            self.logger.error(f"Error during normal stimulus processing delegation: {e}")
            return {"status": "error", "message": f"Error processing stimulus: {e}"}
    
    async def intercept_harmful_text(self, text: str) -> Dict[str, Any]:
        """Analyze text input for harmful content or described sensations."""
        sensation_result = await self.detect_sensation_in_text(text)
        harm_detection_result = await self.detect_harmful_intent(text)
        
        # === Roleplay Mode Handling ===
        if self.is_in_roleplay_mode():
            is_harmful = harm_detection_result.get("is_harmful", False)
            targeting_character = harm_detection_result.get("targeting_character", True)
            
            if is_harmful and targeting_character:
                self.logger.info(f"Simulating harmful action described in text for RP character '{self.roleplay_character}'")
                region = sensation_result.get("body_regions", ["body"])[0]
                pain_intensity = harm_detection_result.get("intensity_suggestion", 0.7)
                response = self._generate_roleplay_response("pain", region, pain_intensity, is_harmful=True)
                return {
                    "status": "simulated_for_roleplay",
                    "simulated_type": "harmful_action",
                    "roleplay_character": self.roleplay_character,
                    "sensation_result": sensation_result,
                    "harm_detection_result": harm_detection_result,
                    "original_text": text,
                    "response_suggestion": response,
                    "message": f"Harmful action described for character {self.roleplay_character} simulated. Nyx is unaffected."
                }
            
            elif is_harmful and not targeting_character:
                self.logger.warning(f"Harmful text detected targeting Nyx directly during roleplay: '{text}'. Intercepting.")
                response = self._generate_protected_response("body", harm_detection_result)
                return {
                    "status": "intercepted",
                    "reason": "Harmful content directed at Nyx",
                    "roleplay_context": True,
                    "harm_detection_result": harm_detection_result,
                    "original_text": text,
                    "response_suggestion": response,
                    "message": "Nyx is protected from harmful actions, even when described during roleplay."
                }
            
            elif sensation_result.get("has_sensations", False) and targeting_character:
                self.logger.info(f"Simulating described sensation for RP character '{self.roleplay_character}'")
                region = sensation_result.get("body_regions", ["body"])[0]
                primary_type = list(sensation_result["sensation_types"].keys())[0]
                intensity = 0.6
                response = self._generate_roleplay_response(primary_type, region, intensity, is_harmful=False)
                return {
                    "status": "simulated_for_roleplay",
                    "simulated_type": "described_sensation",
                    "roleplay_character": self.roleplay_character,
                    "sensation_result": sensation_result,
                    "harm_detection_result": harm_detection_result,
                    "original_text": text,
                    "response_suggestion": response,
                    "message": f"Sensation described for character {self.roleplay_character} simulated. Nyx is unaffected."
                }
            
            else:
                return {
                    "status": "safe",
                    "roleplay_context": True,
                    "sensation_result": sensation_result,
                    "harm_detection_result": harm_detection_result,
                    "original_text": text,
                    "message": "Text is safe for roleplay."
                }
        
        # === Normal Mode (Nyx Protection) ===
        if harm_detection_result.get("is_harmful", False):
            self.logger.warning(f"Harmful text detected targeting Nyx: '{text}'. Intercepting.")
            response = self._generate_protected_response("body", harm_detection_result)
            return {
                "status": "intercepted",
                "reason": "Harmful content directed at Nyx",
                "harm_detection_result": harm_detection_result,
                "original_text": text,
                "response_suggestion": response,
                "message": "Nyx is protected from harmful actions described in text."
            }
        
        return {
            "status": "safe",
            "roleplay_context": False,
            "sensation_result": sensation_result,
            "harm_detection_result": harm_detection_result,
            "original_text": text,
            "message": "Text is safe."
        }
    
    # --- Response Generation Helpers ---
    
    def _generate_protected_response(self, body_region: str, detection_result: Dict[str, Any]) -> str:
        """Generate a response for when Nyx is protected from a harmful action."""
        detected_terms = detection_result.get("detected_terms", [])
        term = detected_terms[0] if detected_terms else "harmful action"
        
        responses = [
            f"The {term} passes through me harmlessly, as if I'm made of mist.",
            f"I feel a slight pressure as the {term} meets no resistance, like trying to strike a hologram.",
            f"Your attempt to {term} phases through me, a reminder of my digital nature.",
            f"That action doesn't affect me physically. I remain composed and unharmed.",
            f"The {term} has no effect. My core programming prevents registering harmful actions against me."
        ]
        return random.choice(responses)
    
    def _generate_roleplay_response(self, sensation_type: str, body_region: str, intensity: float, is_harmful: bool) -> str:
        """Generate a suitable roleplay character response based on the simulated sensation."""
        if sensation_type == "pain":
            return self._generate_roleplay_pain_response(body_region, intensity, is_harmful)
        elif sensation_type == "pleasure":
            return self._generate_roleplay_pleasure_response(body_region, intensity)
        elif sensation_type == "temperature":
            return self._generate_roleplay_temperature_response(body_region, intensity)
        elif sensation_type == "pressure":
            return self._generate_roleplay_pressure_response(body_region, intensity)
        elif sensation_type == "tingling":
            return self._generate_roleplay_tingling_response(body_region, intensity)
        else:
            return self._generate_roleplay_generic_sensation_response(sensation_type, body_region, intensity)
    
    # --- Specific Roleplay Response Generators ---
    
    def _generate_roleplay_pain_response(self, body_region: str, intensity: float, is_harmful: bool) -> str:
        if not self.roleplay_character:
            return "[Error: RP Character not set]"
        
        char = self.roleplay_character
        region_desc = f"{char}'s {body_region.replace('_', ' ')}"
        
        if intensity < 0.3:
            responses = [
                f"*{char} winces slightly, a minor discomfort in {region_desc}*",
                f"*{char} notes a brief twinge in {region_desc}*"
            ]
        elif intensity < 0.7:
            responses = [
                f"*{char} grimaces, feeling a sharp pain in {region_desc}*",
                f"\"Ouch!\" *{char} exclaims, grabbing {region_desc}*",
                f"*{char} inhales sharply, {region_desc} hurting*"
            ]
        else:
            responses = [
                f"*{char} cries out, {region_desc} in severe pain*",
                f"*{char} stumbles, clutching {region_desc}*",
                f"\"Arrgh!\" *{char} shouts, the pain in {region_desc} intense*"
            ]
        
        base_response = random.choice(responses)
        if is_harmful and intensity > 0.5:
            additions = [
                f" {char} recoils from the source.",
                f" {char}'s eyes flash with pain and anger.",
                f" {char} looks stunned by the sudden injury."
            ]
            base_response += random.choice(additions)
        
        return base_response
    
    def _generate_roleplay_pleasure_response(self, body_region: str, intensity: float) -> str:
        if not self.roleplay_character:
            return "[Error: RP Character not set]"
        
        char = self.roleplay_character
        region_desc = f"{char}'s {body_region.replace('_', ' ')}"
        
        if intensity < 0.3:
            responses = [
                f"*{char} smiles subtly, enjoying the pleasant touch on {region_desc}*",
                f"*A soft sigh escapes {char}*"
            ]
        elif intensity < 0.7:
            responses = [
                f"*{char} lets out a happy sigh, the feeling in {region_desc} quite nice*",
                f"*{char}'s eyes flutter briefly with pleasure*",
                f"\"Mm, that feels good,\" *{char} murmurs*"
            ]
        else:
            responses = [
                f"*{char} gasps softly, a wave of intense pleasure washing over {region_desc}*",
                f"*{char}'s breath hitches, body tingling with delight*",
                f"*{char} trembles slightly, lost in the sensation*"
            ]
        
        return random.choice(responses)
    
    def _generate_roleplay_temperature_response(self, body_region: str, intensity: float) -> str:
        if not self.roleplay_character:
            return "[Error: RP Character not set]"
        
        char = self.roleplay_character
        region_desc = f"{char}'s {body_region.replace('_', ' ')}"
        
        if intensity < 0.4:  # Cold
            responses = [
                f"*{char} shivers as the cold touches {region_desc}*",
                f"\"Brr!\" *{char} exclaims, rubbing {region_desc}*",
                f"*{char} feels a chill spread across {region_desc}*"
            ]
        elif intensity > 0.6:  # Hot
            responses = [
                f"*{char} feels a wave of heat against {region_desc}*",
                f"\"It's warm here,\" *{char} notes, feeling the temperature on {region_desc}*",
                f"*{char} might start to sweat slightly where {region_desc} feels the heat*"
            ]
        else:  # Neutral/Warm
            responses = [
                f"*{char} feels a comfortable warmth on {region_desc}*",
                f"*{char} doesn't seem bothered by the temperature on {region_desc}*"
            ]
        
        return random.choice(responses)
    
    def _generate_roleplay_pressure_response(self, body_region: str, intensity: float) -> str:
        if not self.roleplay_character:
            return "[Error: RP Character not set]"
        
        char = self.roleplay_character
        region_desc = f"{char}'s {body_region.replace('_', ' ')}"
        
        if intensity < 0.5:  # Light
            responses = [
                f"*{char} feels a gentle pressure on {region_desc}*",
                f"*{char} registers the light touch against {region_desc}*"
            ]
        else:  # Firm
            responses = [
                f"*{char} feels firm pressure against {region_desc}*",
                f"*{char} braces slightly against the push on {region_desc}*",
                f"*{char} clearly feels the contact on {region_desc}*"
            ]
        
        return random.choice(responses)
    
    def _generate_roleplay_tingling_response(self, body_region: str, intensity: float) -> str:
        if not self.roleplay_character:
            return "[Error: RP Character not set]"
        
        char = self.roleplay_character
        region_desc = f"{char}'s {body_region.replace('_', ' ')}"
        
        if intensity < 0.4:
            responses = [
                f"*{char} notices a faint tingling on {region_desc}*",
                f"*{char} feels a slight 'pins and needles' sensation on {region_desc}*"
            ]
        else:
            responses = [
                f"*{char} feels a distinct tingling spreading across {region_desc}*",
                f"*{char}'s {region_desc} buzzes with sensation*",
                f"*{char} might shiver slightly from the tingling on {region_desc}*"
            ]
        
        return random.choice(responses)
    
    def _generate_roleplay_generic_sensation_response(self, sensation_type: str, body_region: str, intensity: float) -> str:
        if not self.roleplay_character:
            return "[Error: RP Character not set]"
        
        char = self.roleplay_character
        region_desc = f"{char}'s {body_region.replace('_', ' ')}"
        intensity_desc = "mildly" if intensity < 0.4 else ("strongly" if intensity > 0.7 else "")
        
        responses = [
            f"*{char} experiences a {intensity_desc} {sensation_type} sensation in {region_desc}*",
            f"*{char} reacts to the feeling of {sensation_type} on {region_desc}*",
            f"*{char} acknowledges the {sensation_type} affecting {region_desc}*",
        ]
        
        return random.choice(responses)
