# nyx/core/digital_somatosensory_system.py

import logging
import asyncio
import datetime
import json
import random
import math
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import numpy as np
import asyncio
from nyx.core.reward_system import RewardSignal

from agents import (
    Agent, Runner, trace, function_tool, 
    RunContextWrapper, handoff, ModelSettings
)
from pydantic import BaseModel, Field

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
    dominant_sensation: str = Field(..., description="The dominant sensation type")
    dominant_region: str = Field(..., description="The dominant body region")
    dominant_intensity: float = Field(..., description="Intensity of dominant sensation")
    comfort_level: float = Field(..., description="Overall comfort level (-1.0 to 1.0)")
    posture_effect: str = Field(..., description="Effect on posture description")
    movement_quality: str = Field(..., description="Quality of movement description")
    behavioral_impact: str = Field(..., description="Impact on behavior")
    regions_summary: Dict[str, Dict[str, float]] = Field(..., description="Summary of region states")

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
    effects: Dict[str, Any] = Field(..., description="Effects of the stimulus")
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

# =============== Main Digital Somatosensory System Class ===============

class DigitalSomatosensorySystem:
    """
    Digital Somatosensory System (DSS) for Nyx.
    
    This system provides a simulated body experience with regions that can independently 
    experience sensations like pressure, temperature, pain, and pleasure. These sensations
    are memory-linked and influence Nyx's responses and behavior.
    """
    
    def __init__(self, memory_core=None, emotional_core=None, reward_system=None): 
        """
        Initialize the Digital Somatosensory System
        
        Args:
            memory_core: Memory system for storing sensory memories
            emotional_core: Emotional system for linking sensations to emotions
        """
        self.memory_core = memory_core
        self.emotional_core = emotional_core
        self.reward_system = reward_system 
        
        # Initialize body regions
        self.body_regions = {
            "head": BodyRegion(name="head"),
            "face": BodyRegion(name="face"),
            "neck": BodyRegion(name="neck"),
            "shoulders": BodyRegion(name="shoulders"),
            "arms": BodyRegion(name="arms"),
            "hands": BodyRegion(name="hands"),
            "chest": BodyRegion(name="chest"),
            "back": BodyRegion(name="back"),
            "spine": BodyRegion(name="spine"),
            "stomach": BodyRegion(name="stomach"),
            "hips": BodyRegion(name="hips"),
            "legs": BodyRegion(name="legs"),
            "feet": BodyRegion(name="feet"),
            "skin": BodyRegion(name="skin")
        }
        
        # Pain model settings
        self.pain_model = {
            "threshold": 0.3,  # Minimum pain to register consciously
            "tolerance": 0.7,  # Pain tolerance level (adapted over time)
            "decay_rate": 0.05,  # How quickly pain fades per update
            "memory_duration": 60 * 60 * 24 * 7,  # How long pain memories last (in seconds) - 1 week
            "pain_memories": []  # List of pain memories
        }
        
        # Temperature model settings
        self.temperature_model = {
            "current_ambient": 0.5,  # Current ambient temperature (0.0=freezing, 0.5=neutral, 1.0=very hot)
            "body_temperature": 0.5,  # Internal body temperature
            "adaptation_rate": 0.01,  # Rate of adapting to ambient temperature
            "comfort_range": (0.4, 0.6),  # Comfortable temperature range
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
        
        # Memory-linked sensations settings
        self.memory_linked_sensations = {
            "associations": {},  # Map of stimuli to physical responses
            "learning_rate": 0.1,  # How quickly new associations are formed
            "memory_decay": 0.01  # How quickly associations fade if not reinforced
        }
        
        # Overall body state
        self.body_state = {
            "comfort_level": 0.5,  # Overall comfort (-1.0 to 1.0)
            "fatigue": 0.0,  # Physical fatigue level (0.0-1.0)
            "tension": 0.0,  # Physical tension level (0.0-1.0)
            "posture": "neutral",  # Current postural state
            "last_update": datetime.datetime.now()
        }
        
        # Settings for how somatosensory state affects responses
        self.response_influence = {
            "body_to_emotion_influence": 0.3,  # How much body state influences emotions
            "emotion_to_body_influence": 0.3,  # How much emotions influence body state
            "expression_threshold": 0.4,  # Minimum sensation intensity to express
            "max_expressions_per_response": 2  # Maximum number of sensory expressions per response
        }
        
        # Trace ID for connecting traces
        self.trace_group_id = f"somatic_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Initialize agents
        self._init_agents()
        
        logger.info("Digital Somatosensory System initialized")
    
    def _init_agents(self):
        """Initialize the agents for sensory processing"""
        # Create specialized agents
        self.expression_agent = self._create_expression_agent()
        self.body_state_agent = self._create_body_state_agent()
        self.temperature_agent = self._create_temperature_agent()
        
        # Create the body orchestrator agent
        self.body_orchestrator = self._orchestrate_body_experience()
    
    def _create_expression_agent(self) -> Agent:
        """Create the sensory expression agent"""
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
                function_tool(self._get_region_state),
                function_tool(self._get_current_temperature_effects),
                function_tool(self._get_pain_expression)
            ],
            output_type=SensoryExpression
        )
    
    def _create_body_state_agent(self) -> Agent:
        """Create the body state analysis agent"""
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
                function_tool(self._get_all_region_states),
                function_tool(self._calculate_overall_comfort),
                function_tool(self._get_posture_effects)
            ],
            output_type=BodyStateOutput
        )
    
    def _create_temperature_agent(self) -> Agent:
        """Create the temperature effects agent"""
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
                function_tool(self._get_ambient_temperature),
                function_tool(self._get_body_temperature),
                function_tool(self._get_temperature_comfort)
            ],
            output_type=TemperatureEffect
        )
    
    def _orchestrate_body_experience(self) -> Agent:
        """Create orchestrator agent for body experience"""
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
            
            Process each stimulus thoroughly and provide a coherent sensory experience.
            """,
            handoffs=[
                handoff(self.expression_agent, 
                       tool_name_override="generate_expression", 
                       tool_description_override="Generate sensory expression based on body state"),
                
                handoff(self.body_state_agent, 
                       tool_name_override="analyze_body_state",
                       tool_description_override="Analyze current holistic body state"),
                
                handoff(self.temperature_agent,
                       tool_name_override="analyze_temperature",
                       tool_description_override="Analyze temperature effects on body")
            ],
            tools=[
                function_tool(self._process_stimulus_tool),
                function_tool(self._get_region_state),
                function_tool(self._get_all_region_states),
                function_tool(self._update_body_temperature),
                function_tool(self._calculate_overall_comfort)
            ],
            model_settings=ModelSettings(temperature=0.2),
            output_type=StimulusProcessingResult
        )
    
    # =============== Tool Functions ===============
    
    @function_tool
    async def _process_stimulus_tool(self, 
                              ctx: RunContextWrapper,
                              stimulus_type: str, 
                              body_region: str, 
                              intensity: float,
                              cause: str = "",
                              duration: float = 1.0) -> Dict[str, Any]:
        """
        Process a sensory stimulus on a body region (internal tool function)
        
        Args:
            stimulus_type: Type of stimulus (pressure, temperature, pain, pleasure, tingling)
            body_region: Body region receiving the stimulus
            intensity: Intensity of the stimulus (0.0-1.0)
            cause: Cause of the stimulus
            duration: Duration of the stimulus in seconds
            
        Returns:
            Result of stimulus application
        """
        # Get the region
        if body_region not in self.body_regions:
            return {"error": f"Invalid body region: {body_region}"}
            
        region = self.body_regions[body_region]
        
        # Record time of update
        region.last_update = datetime.datetime.now()
        
        # Apply stimulus based on type
        result = {"region": body_region, "type": stimulus_type, "intensity": intensity}
        
        if stimulus_type == "pressure":
            region.pressure = min(1.0, region.pressure + (intensity * duration / 10.0))
            result["new_value"] = region.pressure
            
            # High pressure can cause pain if intense and sustained
            if intensity > 0.7 and duration > 5.0:
                pain_from_pressure = (intensity - 0.7) * (duration / 10.0) * 0.5
                region.pain = min(1.0, region.pain + pain_from_pressure)
                result["pain_caused"] = pain_from_pressure
            
        elif stimulus_type == "temperature":
            # Scale to temperature range (0.0-1.0)
            target_temp = intensity  # Direct mapping for simplicity
            
            # Apply temperature change with duration factor
            temp_change = (target_temp - region.temperature) * min(1.0, duration / 30.0)
            region.temperature += temp_change
            region.temperature = max(0.0, min(1.0, region.temperature))
            result["new_value"] = region.temperature
            
            # Extreme temperatures can cause pain
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
            
            # Store pain memory if significant
            if intensity > self.pain_model["threshold"]:
                pain_memory = PainMemory(
                    intensity=intensity,
                    location=body_region,
                    cause=cause or "unknown stimulus",
                    duration=duration,
                    timestamp=datetime.datetime.now(),
                    associated_memory_id=None
                )
                self.pain_model["pain_memories"].append(pain_memory)
                result["memory_created"] = True
            
        elif stimulus_type == "pleasure":
            region.pleasure = min(1.0, region.pleasure + (intensity * duration / 10.0))
            result["new_value"] = region.pleasure
            
            # Pleasure can reduce pain
            if intensity > 0.5 and region.pain > 0.0:
                pain_reduction = min(region.pain, (intensity - 0.5) * 0.2)
                region.pain = max(0.0, region.pain - pain_reduction)
                result["pain_reduced"] = pain_reduction
            
        elif stimulus_type == "tingling":
            region.tingling = min(1.0, region.tingling + (intensity * duration / 10.0))
            result["new_value"] = region.tingling
        
        # Add to sensation memory
        memory_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": stimulus_type,
            "intensity": intensity,
            "cause": cause,
            "duration": duration
        }
        region.sensation_memory.append(memory_entry)
        
        # Keep memory size manageable
        if len(region.sensation_memory) > 20:
            region.sensation_memory = region.sensation_memory[-20:]
        
        # Check for learned associations
        if cause and len(cause.strip()) > 0:
            # Store or update association
            if cause not in self.memory_linked_sensations["associations"]:
                self.memory_linked_sensations["associations"][cause] = {}
            
            # Update region-specific association
            if body_region not in self.memory_linked_sensations["associations"][cause]:
                self.memory_linked_sensations["associations"][cause][body_region] = {}
            
            # Update stimulus-specific association
            if stimulus_type not in self.memory_linked_sensations["associations"][cause][body_region]:
                self.memory_linked_sensations["associations"][cause][body_region][stimulus_type] = 0.0
            
            # Strengthen association based on learning rate and intensity
            current = self.memory_linked_sensations["associations"][cause][body_region][stimulus_type]
            learned = self.memory_linked_sensations["learning_rate"] * intensity
            self.memory_linked_sensations["associations"][cause][body_region][stimulus_type] = min(1.0, current + learned)
            
            result["association_strength"] = self.memory_linked_sensations["associations"][cause][body_region][stimulus_type]
        
        # Update body state if needed
        if stimulus_type == "pain" and intensity > 0.5:
            # Pain increases tension
            self.body_state["tension"] = min(1.0, self.body_state["tension"] + (intensity * 0.2))
        elif stimulus_type == "pleasure" and intensity > 0.5:
            # Pleasure reduces tension
            self.body_state["tension"] = max(0.0, self.body_state["tension"] - (intensity * 0.1))
        
        return result
    
    @function_tool
    async def _get_region_state(self, ctx: RunContextWrapper, region_name: str) -> Dict[str, Any]:
        """
        Get the current state of a specific body region
        
        Args:
            region_name: Name of the body region
            
        Returns:
            Current state of the region
        """
        if region_name not in self.body_regions:
            return {"error": f"Region {region_name} not found"}
        
        region = self.body_regions[region_name]
        
        return {
            "name": region.name,
            "pressure": region.pressure,
            "temperature": region.temperature,
            "pain": region.pain,
            "pleasure": region.pleasure,
            "tingling": region.tingling,
            "dominant_sensation": self._get_dominant_sensation(region),
            "last_update": region.last_update.isoformat() if region.last_update else None,
            "recent_memories": region.sensation_memory[-3:] if region.sensation_memory else []
        }
    
    @function_tool
    async def _get_all_region_states(self, ctx: RunContextWrapper) -> Dict[str, Dict[str, Any]]:
        """
        Get the current state of all body regions
        
        Returns:
            Dictionary of all region states
        """
        all_states = {}
        
        for name, region in self.body_regions.items():
            all_states[name] = {
                "pressure": region.pressure,
                "temperature": region.temperature,
                "pain": region.pain,
                "pleasure": region.pleasure,
                "tingling": region.tingling,
                "dominant_sensation": self._get_dominant_sensation(region)
            }
        
        return all_states
    
    @function_tool
    async def _calculate_overall_comfort(self, ctx: RunContextWrapper) -> float:
        """
        Calculate overall physical comfort level
        
        Returns:
            Comfort level from -1.0 (extremely uncomfortable) to 1.0 (extremely comfortable)
        """
        # Start at neutral
        comfort = 0.0
        
        # Add comfort from pleasure sensations
        total_pleasure = sum(region.pleasure for region in self.body_regions.values())
        weighted_pleasure = total_pleasure / len(self.body_regions) * 2.0  # Scale up to have more impact
        comfort += weighted_pleasure
        
        # Subtract discomfort from pain sensations
        total_pain = sum(region.pain for region in self.body_regions.values())
        weighted_pain = total_pain / len(self.body_regions) * 2.5  # Pain has stronger negative impact
        comfort -= weighted_pain
        
        # Consider temperature discomfort
        temp_comfort = 0.0
        for region in self.body_regions.values():
            # Calculate how far temperature is from neutral (0.5)
            temp_deviation = abs(region.temperature - 0.5)
            # Temperature that's too hot or too cold reduces comfort
            if temp_deviation > 0.2:  # Only count significant deviations
                temp_comfort -= (temp_deviation - 0.2) * 1.5
        
        # Add temperature effect to comfort
        comfort += temp_comfort / len(self.body_regions)
        
        # Consider pressure discomfort (very high pressure is uncomfortable)
        pressure_discomfort = 0.0
        for region in self.body_regions.values():
            if region.pressure > 0.7:  # High pressure
                pressure_discomfort -= (region.pressure - 0.7) * 1.5
        
        # Add pressure effect to comfort
        comfort += pressure_discomfort / len(self.body_regions)
        
        # Factor in overall body state
        comfort -= self.body_state["tension"] * 0.5
        comfort -= self.body_state["fatigue"] * 0.3
        
        # Clamp to range
        return max(-1.0, min(1.0, comfort))
    
    @function_tool
    async def _get_posture_effects(self, ctx: RunContextWrapper) -> Dict[str, str]:
        """
        Get the effects of current body state on posture and movement
        
        Returns:
            Dictionary with posture and movement descriptions
        """
        # Calculate overall tension
        tension = self.body_state["tension"]
        for region in ["neck", "shoulders", "back", "spine"]:
            if region in self.body_regions:
                # Pain and high pressure in these regions increase tension
                tension += self.body_regions[region].pain * 0.5
                tension += max(0, self.body_regions[region].pressure - 0.6) * 0.3
        
        # Clamp tension
        tension = min(1.0, max(0.0, tension))
        
        # Calculate overall fatigue
        fatigue = self.body_state["fatigue"]
        
        # Temperature affects fatigue and tension
        avg_temp = sum(region.temperature for region in self.body_regions.values()) / len(self.body_regions)
        
        # Very hot temperatures increase fatigue
        if avg_temp > 0.7:
            fatigue += (avg_temp - 0.7) * 0.5
        
        # Very cold temperatures increase tension
        if avg_temp < 0.3:
            tension += (0.3 - avg_temp) * 0.5
        
        # Clamp values
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
        
        return {
            "posture": posture_effect,
            "movement": movement_quality,
            "tension": tension,
            "fatigue": fatigue
        }
    
    @function_tool
    async def _get_ambient_temperature(self, ctx: RunContextWrapper) -> float:
        """
        Get current ambient temperature value
        
        Returns:
            Temperature value (0.0=freezing, 0.5=neutral, 1.0=very hot)
        """
        return self.temperature_model["current_ambient"]
    
    @function_tool
    async def _get_body_temperature(self, ctx: RunContextWrapper) -> float:
        """
        Get current body temperature value
        
        Returns:
            Temperature value (0.0=freezing, 0.5=neutral, 1.0=very hot)
        """
        return self.temperature_model["body_temperature"]
    
    @function_tool
    async def _get_temperature_comfort(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get temperature comfort assessment
        
        Returns:
            Temperature comfort data
        """
        body_temp = self.temperature_model["body_temperature"]
        ambient_temp = self.temperature_model["current_ambient"]
        comfort_range = self.temperature_model["comfort_range"]
        
        # Determine if current temperature is comfortable
        is_comfortable = comfort_range[0] <= body_temp <= comfort_range[1]
        
        # Calculate discomfort level
        discomfort = 0.0
        if body_temp < comfort_range[0]:
            discomfort = (comfort_range[0] - body_temp) * 10.0
        elif body_temp > comfort_range[1]:
            discomfort = (body_temp - comfort_range[1]) * 10.0
        
        # Determine temperature perception
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
    async def _get_current_temperature_effects(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Get effects of current temperature on expression and behavior
        
        Returns:
            Current temperature effects
        """
        body_temp = self.temperature_model["body_temperature"]
        
        # Determine effects based on temperature
        tone_effect = ""
        posture_effect = ""
        interaction_effect = ""
        expression_examples = []
        
        if body_temp > 0.7:  # Hot
            tone_effect = "slower, more drawn out, languid"
            posture_effect = "relaxed, open, limbs spread to dissipate heat"
            interaction_effect = "may withdraw from intense interaction, preferring calmer exchanges"
            expression_examples = self.temperature_model["heat_expressions"]
        elif body_temp > 0.6:  # Warm
            tone_effect = "relaxed, fluid, unhurried"
            posture_effect = "comfortable, loose, slightly expanded"
            interaction_effect = "generally receptive but with measured energy"
            expression_examples = [self.temperature_model["heat_expressions"][2], 
                                  self.temperature_model["heat_expressions"][3]]
        elif body_temp < 0.3:  # Cold
            tone_effect = "sharper, more tense, with subtle tremors"
            posture_effect = "contracted, protective, conserving heat"
            interaction_effect = "may seek warmth and connection but with physical restraint"
            expression_examples = self.temperature_model["cold_expressions"]
        elif body_temp < 0.4:  # Cool
            tone_effect = "slightly crisp, more precise"
            posture_effect = "slightly drawn in, contained"
            interaction_effect = "alert and responsive but with some physical reserve"
            expression_examples = [self.temperature_model["cold_expressions"][2], 
                                  self.temperature_model["cold_expressions"][3]]
        else:  # Neutral
            tone_effect = "balanced, natural, unaffected by temperature"
            posture_effect = "neutral, neither expanded nor contracted"
            interaction_effect = "comfortable engagement without temperature influence"
            
            # Mix of mild expressions
            expression_examples = [self.temperature_model["heat_expressions"][3], 
                                  self.temperature_model["cold_expressions"][3]]
        
        return {
            "body_temperature": body_temp,
            "tone_effect": tone_effect,
            "posture_effect": posture_effect,
            "interaction_effect": interaction_effect,
            "expression_examples": expression_examples
        }
    
    @function_tool
    async def _get_pain_expression(self, ctx: RunContextWrapper, pain_level: float, region: str) -> str:
        """
        Get an expression for pain at specified level and region
        
        Args:
            pain_level: Pain intensity (0.0-1.0)
            region: Body region experiencing pain
            
        Returns:
            Pain expression text
        """
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
    async def _update_body_temperature(self, ctx: RunContextWrapper, ambient_temperature: float, duration: float = 60.0) -> Dict[str, Any]:
        """
        Update body temperature based on ambient temperature
        
        Args:
            ambient_temperature: Ambient temperature (0.0-1.0)
            duration: Duration in seconds since last update
            
        Returns:
            Updated temperature data
        """
        # Store current ambient temperature
        self.temperature_model["current_ambient"] = ambient_temperature
        
        # Calculate adaptation rate based on duration
        adaptation = self.temperature_model["adaptation_rate"] * (duration / 60.0)
        
        # Cap adaptation to prevent huge jumps
        adaptation = min(0.1, adaptation)
        
        # Move body temperature toward ambient temperature
        current = self.temperature_model["body_temperature"]
        diff = ambient_temperature - current
        
        # Update body temperature
        self.temperature_model["body_temperature"] += diff * adaptation
        
        # Update all body regions (with some variation)
        for region in self.body_regions.values():
            # Calculate region-specific adaptation with variation
            region_adaptation = adaptation * random.uniform(0.8, 1.2)
            
            # Calculate target temperature (with slight variation from body temp)
            target = self.temperature_model["body_temperature"] + random.uniform(-0.05, 0.05)
            target = max(0.0, min(1.0, target))
            
            # Move region temperature toward target
            region_diff = target - region.temperature
            region.temperature += region_diff * region_adaptation
        
        return {
            "previous_body_temp": current,
            "new_body_temp": self.temperature_model["body_temperature"],
            "ambient_temp": ambient_temperature,
            "adaptation_applied": adaptation
        }
    
    # =============== Helper Methods ===============
    
    def _get_dominant_sensation(self, region: BodyRegion) -> str:
        """Determine the dominant sensation for a region"""
        sensations = {
            "pressure": region.pressure,
            "temperature": abs(region.temperature - 0.5) * 2,  # Convert to deviation from neutral
            "pain": region.pain,
            "pleasure": region.pleasure,
            "tingling": region.tingling
        }
        
        # Find the maximum sensation
        max_sensation = max(sensations.items(), key=lambda x: x[1])
        
        # Only return if it's above a threshold
        if max_sensation[1] >= 0.2:
            return max_sensation[0]
        
        return "neutral"
    
    def _decay_sensations(self, duration: float = 60.0):
        """
        Decay all sensations over time
        
        Args:
            duration: Time in seconds since last update
        """
        # Calculate decay factor based on duration
        decay = 0.05 * (duration / 60.0)
        
        # Cap decay to prevent huge jumps
        decay = min(0.2, decay)
        
        # Decay all sensations
        for region in self.body_regions.values():
            # Pressure and tingling decay toward zero
            region.pressure = max(0.0, region.pressure - (region.pressure * decay))
            region.tingling = max(0.0, region.tingling - (region.tingling * decay))
            
            # Pain decays based on pain model
            pain_decay = self.pain_model["decay_rate"] * (duration / 60.0)
            region.pain = max(0.0, region.pain - (region.pain * pain_decay))
            
            # Pleasure decays slightly faster than pain
            pleasure_decay = pain_decay * 1.2
            region.pleasure = max(0.0, region.pleasure - (region.pleasure * pleasure_decay))
    
    def _decay_pain_memories(self):
        """Remove old pain memories based on duration setting"""
        now = datetime.datetime.now()
        cutoff = now - datetime.timedelta(seconds=self.pain_model["memory_duration"])
        
        # Filter out old memories
        self.pain_model["pain_memories"] = [m for m in self.pain_model["pain_memories"] 
                                           if m.timestamp > cutoff]
    
    def _update_pain_tolerance(self):
        """Update pain tolerance based on recent experiences"""
        # If no pain memories, keep tolerance the same
        if not self.pain_model["pain_memories"]:
            return
        
        # Get recent pain memories (last 10)
        recent_memories = sorted(self.pain_model["pain_memories"], 
                                key=lambda x: x.timestamp, reverse=True)[:10]
        
        # Calculate average intensity
        avg_intensity = sum(m.intensity for m in recent_memories) / len(recent_memories)
        
        # Adjust tolerance (higher recent pain increases tolerance slowly)
        if avg_intensity > self.pain_model["tolerance"]:
            # Pain higher than tolerance increases tolerance
            self.pain_model["tolerance"] += (avg_intensity - self.pain_model["tolerance"]) * 0.05
        elif avg_intensity < self.pain_model["tolerance"] * 0.7:
            # Long period of low pain decreases tolerance slowly
            self.pain_model["tolerance"] -= (self.pain_model["tolerance"] - avg_intensity) * 0.01
        
        # Clamp tolerance to reasonable range
        self.pain_model["tolerance"] = max(0.4, min(0.9, self.pain_model["tolerance"]))
    
    # =============== Public Methods ===============
    
    async def initialize(self):
        """Initialize the system and its connections"""
        logger.info("Digital Somatosensory System initialization started")
        
        # Set initial body temperature based on time of day
        hour = datetime.datetime.now().hour
        if 22 <= hour or hour < 6:
            # Night - cooler
            self.temperature_model["body_temperature"] = 0.45
        elif 6 <= hour < 10:
            # Morning - cool to neutral
            self.temperature_model["body_temperature"] = 0.48
        elif 14 <= hour < 18:
            # Afternoon - warmer
            self.temperature_model["body_temperature"] = 0.55
        else:
            # Default - neutral
            self.temperature_model["body_temperature"] = 0.5
        
        # Initialize all region temperatures to match body temperature
        for region in self.body_regions.values():
            region.temperature = self.temperature_model["body_temperature"]
            region.last_update = datetime.datetime.now()
        
        logger.info("Digital Somatosensory System initialized successfully")
        return True
    
    async def update(self, ambient_temperature: Optional[float] = None):
        """
        Update the somatosensory system
        
        Args:
            ambient_temperature: Optional ambient temperature to use for update
            
        Returns:
            Updated body state
        """
        with trace(workflow_name="Somatic_Update", group_id=self.trace_group_id):
            now = datetime.datetime.now()
            
            # Calculate time since last update
            last_update = self.body_state["last_update"]
            duration = (now - last_update).total_seconds()
            
            # Update timestamps
            self.body_state["last_update"] = now
            
            # If ambient temperature provided, update temperature model
            if ambient_temperature is not None:
                await self._update_body_temperature(
                    RunContextWrapper(context=None),
                    ambient_temperature=ambient_temperature, 
                    duration=duration
                )
            
            # Decay sensations over time
            self._decay_sensations(duration)
            
            # Process pain memory updates
            self._decay_pain_memories()
            self._update_pain_tolerance()
            
            # Update body state metrics
            # Fatigue increases slowly over time, resets during maintenance
            self.body_state["fatigue"] = min(1.0, self.body_state["fatigue"] + (0.01 * duration / 3600.0))
            
            # Use the body orchestration agent to get the current body state
            body_state_input = {
                "action": "update",
                "ambient_temperature": ambient_temperature,
                "duration": duration
            }
            
            try:
                # Use the orchestrator to get body state
                result = await self.process_body_experience(body_state_input)
                return result.get("body_state_impact", await self.get_body_state())
            except Exception as e:
                # Fallback to direct method if orchestration fails
                logger.error(f"Body state orchestration failed, using fallback: {e}")
                return await self.get_body_state()
    
    async def process_body_experience(self, stimulus_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a body experience using the orchestrator agent
        
        Args:
            stimulus_data: Data about the stimulus to process
            
        Returns:
            Processing results with updated body state
        """
        with trace(workflow_name="Body_Experience", group_id=self.trace_group_id):
            # Run the orchestrator agent
            result = await Runner.run(
                self.body_orchestrator,
                json.dumps(stimulus_data)
            )
            
            # Process the result
            processed_result = result.final_output.model_dump() if hasattr(result.final_output, "model_dump") else result.final_output
            
            return processed_result
    
    async def process_stimulus(self, 
                           stimulus_type: str, 
                           body_region: str, 
                           intensity: float,
                           cause: str = "",
                           duration: float = 1.0) -> Dict[str, Any]:
        """
        Process a sensory stimulus on a specific body region
        
        Args:
            stimulus_type: Type of stimulus (pressure, temperature, pain, pleasure, tingling)
            body_region: Body region receiving the stimulus
            intensity: Intensity of the stimulus (0.0-1.0)
            cause: Cause of the stimulus
            duration: Duration of the stimulus in seconds
            
        Returns:
            Result of stimulus application
        """
        # Validate body region
        if body_region not in self.body_regions:
            return {"error": f"Invalid body region: {body_region}"}
        
        # Validate stimulus type
        valid_types = ["pressure", "temperature", "pain", "pleasure", "tingling"]
        if stimulus_type not in valid_types:
            return {"error": f"Invalid stimulus type: {stimulus_type}"}
        
        # Create stimulus data for orchestrator
        stimulus_data = {
            "stimulus_type": stimulus_type,
            "body_region": body_region,
            "intensity": intensity,
            "cause": cause,
            "duration": duration,
            "generate_expression": True
        }
        
        # Use the orchestrator to process the stimulus if available
        try:
            result = await self.process_body_experience(stimulus_data)
            return result
        except Exception as e:
            # Fallback to original implementation if orchestration fails
            logger.error(f"Orchestrator failed, using fallback: {e}")
            
            # Get the region
            region = self.body_regions[body_region]
            
            # Record time of update
            region.last_update = datetime.datetime.now()
            
            # Apply stimulus based on type
            result = {"region": body_region, "type": stimulus_type, "intensity": intensity}
            
            if stimulus_type == "pressure":
                region.pressure = min(1.0, region.pressure + (intensity * duration / 10.0))
                result["new_value"] = region.pressure
                
                # High pressure can cause pain if intense and sustained
                if intensity > 0.7 and duration > 5.0:
                    pain_from_pressure = (intensity - 0.7) * (duration / 10.0) * 0.5
                    region.pain = min(1.0, region.pain + pain_from_pressure)
                    result["pain_caused"] = pain_from_pressure
                
            elif stimulus_type == "temperature":
                # Scale to temperature range (0.0-1.0)
                target_temp = intensity  # Direct mapping for simplicity
                
                # Apply temperature change with duration factor
                temp_change = (target_temp - region.temperature) * min(1.0, duration / 30.0)
                region.temperature += temp_change
                region.temperature = max(0.0, min(1.0, region.temperature))
                result["new_value"] = region.temperature
                
                # Extreme temperatures can cause pain
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
                
                # Store pain memory if significant
                if intensity > self.pain_model["threshold"]:
                    pain_memory = PainMemory(
                        intensity=intensity,
                        location=body_region,
                        cause=cause or "unknown stimulus",
                        duration=duration,
                        timestamp=datetime.datetime.now(),
                        associated_memory_id=None
                    )
                    self.pain_model["pain_memories"].append(pain_memory)
                    result["memory_created"] = True
                
            elif stimulus_type == "pleasure":
                region.pleasure = min(1.0, region.pleasure + (intensity * duration / 10.0))
                result["new_value"] = region.pleasure
                
                # Pleasure can reduce pain
                if intensity > 0.5 and region.pain > 0.0:
                    pain_reduction = min(region.pain, (intensity - 0.5) * 0.2)
                    region.pain = max(0.0, region.pain - pain_reduction)
                    result["pain_reduced"] = pain_reduction
                
            elif stimulus_type == "tingling":
                region.tingling = min(1.0, region.tingling + (intensity * duration / 10.0))
                result["new_value"] = region.tingling

            # --- Add Reward Generation ---
            if self.reward_system:
                reward_value = 0.0
                source = f"somatic_{body_region}"
                context = {
                    "stimulus_type": stimulus_type,
                    "intensity": intensity,
                    "cause": cause,
                    "body_region": body_region,
                    "duration": duration
                }
                reward_generated = False
    
                if stimulus_type == "pleasure" and intensity >= 0.6:
                    # Higher pleasure intensity -> stronger reward
                    reward_value = (intensity - 0.5) * 0.8 # Scale reward
                    reward_value = min(1.0, reward_value) # Cap reward
                elif stimulus_type == "pain" and intensity >= self.pain_model["threshold"]:
                    # Pain intensity relative to tolerance determines negative reward
                    # Higher tolerance means less negative reward for the same pain intensity
                    effective_pain = intensity / max(0.1, self.pain_model["tolerance"])
                    reward_value = -min(1.0, effective_pain * 0.6) # Scale negative reward
                elif stimulus_type == "temperature":
                    # Discomfort from temperature can be a negative reward
                    temp_deviation = abs(region.temperature - 0.5)
                    if temp_deviation > 0.3: # Significant deviation from neutral
                       reward_value = -min(0.5, temp_deviation * 0.5) # Mild negative reward for discomfort
    
                # Only send reward if significant
                if abs(reward_value) > 0.1:
                    reward_signal = RewardSignal(
                        value=reward_value,
                        source=source,
                        context=context,
                        timestamp=datetime.datetime.now().isoformat()
                    )
                    # Use asyncio.create_task for non-blocking call
                    asyncio.create_task(self.reward_system.process_reward_signal(reward_signal))
                    result["reward_generated"] = reward_value
                    reward_generated = True
                    logger.debug(f"Generated reward {reward_value:.2f} from {source}")
            
            # Add to sensation memory
            memory_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "type": stimulus_type,
                "intensity": intensity,
                "cause": cause,
                "duration": duration
            }
            region.sensation_memory.append(memory_entry)
            
            # Keep memory size manageable
            if len(region.sensation_memory) > 20:
                region.sensation_memory = region.sensation_memory[-20:]
            
            # Check for learned associations
            if cause and len(cause.strip()) > 0:
                # Store or update association
                if cause not in self.memory_linked_sensations["associations"]:
                    self.memory_linked_sensations["associations"][cause] = {}
                
                # Update region-specific association
                if body_region not in self.memory_linked_sensations["associations"][cause]:
                    self.memory_linked_sensations["associations"][cause][body_region] = {}
                
                # Update stimulus-specific association
                if stimulus_type not in self.memory_linked_sensations["associations"][cause][body_region]:
                    self.memory_linked_sensations["associations"][cause][body_region][stimulus_type] = 0.0
                
                # Strengthen association based on learning rate and intensity
                current = self.memory_linked_sensations["associations"][cause][body_region][stimulus_type]
                learned = self.memory_linked_sensations["learning_rate"] * intensity
                self.memory_linked_sensations["associations"][cause][body_region][stimulus_type] = min(1.0, current + learned)
                
                result["association_strength"] = self.memory_linked_sensations["associations"][cause][body_region][stimulus_type]
            
            # Update body state if needed
            if stimulus_type == "pain" and intensity > 0.5:
                # Pain increases tension
                self.body_state["tension"] = min(1.0, self.body_state["tension"] + (intensity * 0.2))
            elif stimulus_type == "pleasure" and intensity > 0.5:
                # Pleasure reduces tension
                self.body_state["tension"] = max(0.0, self.body_state["tension"] - (intensity * 0.1))
            
            return result
    
    async def process_trigger(self, trigger: str) -> Dict[str, Any]:
        """
        Process a trigger that might have associated body memories
        
        Args:
            trigger: The trigger text or stimulus
            
        Returns:
            Responses triggered if any
        """
        with trace(workflow_name="Somatic_Trigger", group_id=self.trace_group_id):
            # Try to use orchestrator for trigger processing
            try:
                result = await self.process_body_experience({
                    "action": "process_trigger",
                    "trigger": trigger
                })
                return result
            except Exception as e:
                logger.error(f"Trigger orchestration failed, using fallback: {e}")
                
                # Fallback implementation
                results = {"triggered_responses": []}
                
                # Check if this trigger has associations
                if trigger in self.memory_linked_sensations["associations"]:
                    associations = self.memory_linked_sensations["associations"][trigger]
                    
                    # Process each associated body region
                    for region_name, stimuli in associations.items():
                        # Process each stimulus type
                        for stimulus_type, strength in stimuli.items():
                            # Only trigger if association is strong enough
                            if strength > 0.3:
                                # Calculate intensity based on association strength
                                intensity = strength * 0.7  # Scale down slightly
                                
                                # Apply the associated stimulus
                                response = await self.process_stimulus(
                                    stimulus_type=stimulus_type,
                                    body_region=region_name,
                                    intensity=intensity,
                                    cause=f"Memory trigger: {trigger}",
                                    duration=1.0
                                )
                                
                                results["triggered_responses"].append({
                                    "region": region_name,
                                    "stimulus": stimulus_type,
                                    "intensity": intensity,
                                    "association_strength": strength
                                })
                
                return results
    
    async def generate_sensory_expression(self, 
                                       stimulus_type: Optional[str] = None,
                                       body_region: Optional[str] = None) -> Optional[str]:
        """
        Generate a natural language expression of current bodily sensations
        
        Args:
            stimulus_type: Optional type of stimulus to focus on
            body_region: Optional body region to focus on
            
        Returns:
            Natural language expression of sensation, or None if nothing significant
        """
        with trace(workflow_name="Sensory_Expression", group_id=self.trace_group_id):
            # Use orchestrator to generate expression
            try:
                result = await self.process_body_experience({
                    "action": "generate_expression",
                    "stimulus_type": stimulus_type,
                    "body_region": body_region,
                    "generate_expression": True
                })
                
                if "expression" in result:
                    return result["expression"]
            except Exception as e:
                logger.error(f"Expression orchestration failed, using fallback: {e}")
            
            # If specific region provided, check if it has significant sensation
            if body_region and body_region in self.body_regions:
                region = self.body_regions[body_region]
                
                # If stimulus type specified, check that specific sensation
                if stimulus_type in ["pressure", "temperature", "pain", "pleasure", "tingling"]:
                    value = getattr(region, stimulus_type, 0.0)
                    if stimulus_type == "temperature":
                        # For temperature, check deviation from neutral
                        significance = abs(value - 0.5) * 2.0
                    else:
                        significance = value
                    
                    # If not significant enough, return None
                    if significance < self.response_influence["expression_threshold"]:
                        return None
                else:
                    # Check if any sensation is significant
                    max_sensation = max(
                        region.pressure,
                        abs(region.temperature - 0.5) * 2.0,  # Temperature deviation from neutral
                        region.pain,
                        region.pleasure,
                        region.tingling
                    )
                    
                    # If not significant enough, return None
                    if max_sensation < self.response_influence["expression_threshold"]:
                        return None
            
            # Find significant sensations to express
            significant_regions = []
            
            for name, region in self.body_regions.items():
                # Skip if looking for specific region and this isn't it
                if body_region and name != body_region:
                    continue
                    
                # Get the dominant sensation for this region
                dominant = self._get_dominant_sensation(region)
                
                # Skip if looking for specific type and this isn't it
                if stimulus_type and dominant != stimulus_type:
                    continue
                    
                # Get the value for the dominant sensation
                if dominant == "temperature":
                    value = abs(region.temperature - 0.5) * 2.0  # Temperature deviation from neutral
                else:
                    value = getattr(region, dominant, 0.0)
                
                # If significant enough, add to list
                if value >= self.response_influence["expression_threshold"]:
                    significant_regions.append({
                        "name": name,
                        "sensation": dominant,
                        "intensity": value
                    })
            
            # If no significant sensations, return None
            if not significant_regions:
                return None
            
            # Sort by intensity
            significant_regions.sort(key=lambda x: x["intensity"], reverse=True)
            
            # Use the most significant one for expression
            region_data = significant_regions[0]
            
            # Generate expression with the agent
            try:
                result = await Runner.run(
                    self.expression_agent,
                    f"Generate an expression of {region_data['sensation']} sensation in the {region_data['name']} with intensity {region_data['intensity']:.2f}"
                )
                
                expression_output = result.final_output_as(SensoryExpression)
                return expression_output.expression_text
            except Exception as e:
                logger.error(f"Error generating sensory expression: {e}")
                
                # Fallback basic expression if agent fails
                if region_data["sensation"] == "pain":
                    return await self._get_pain_expression(
                        RunContextWrapper(context=None),
                        region_data["intensity"],
                        region_data["name"]
                    )
                
                return f"I can feel {region_data['sensation']} in my {region_data['name']}."
    
    async def get_body_state(self) -> Dict[str, Any]:
        """
        Get a complete analysis of current body state
        
        Returns:
            Comprehensive body state analysis
        """
        with trace(workflow_name="Body_State_Analysis", group_id=self.trace_group_id):
            # Use orchestrator for body state analysis
            try:
                result = await self.process_body_experience({
                    "action": "analyze_body_state"
                })
                
                if "body_state_impact" in result:
                    return result["body_state_impact"]
            except Exception as e:
                logger.error(f"Body state orchestration failed, using fallback: {e}")
            
            try:
                result = await Runner.run(
                    self.body_state_agent,
                    "Analyze the current body state across all regions"
                )
                
                body_state_output = result.final_output_as(BodyStateOutput)
                
                return {
                    "dominant_sensation": body_state_output.dominant_sensation,
                    "dominant_region": body_state_output.dominant_region,
                    "dominant_intensity": body_state_output.dominant_intensity,
                    "comfort_level": body_state_output.comfort_level,
                    "posture_effect": body_state_output.posture_effect,
                    "movement_quality": body_state_output.movement_quality,
                    "behavioral_impact": body_state_output.behavioral_impact,
                    "regions_summary": body_state_output.regions_summary
                }
            except Exception as e:
                logger.error(f"Error getting body state: {e}")
                
                # Fallback simplified state if agent fails
                comfort = await self._calculate_overall_comfort(RunContextWrapper(context=None))
                posture = await self._get_posture_effects(RunContextWrapper(context=None))
                
                # Find dominant sensation across all regions
                max_region = None
                max_sensation = None
                max_value = 0.0
                
                for name, region in self.body_regions.items():
                    dominant = self._get_dominant_sensation(region)
                    if dominant == "temperature":
                        value = abs(region.temperature - 0.5) * 2.0
                    else:
                        value = getattr(region, dominant, 0.0)
                    
                    if value > max_value:
                        max_value = value
                        max_sensation = dominant
                        max_region = name
                
                # Default values if nothing significant found
                if not max_region:
                    max_region = "overall body"
                    max_sensation = "neutral"
                    max_value = 0.0
                
                return {
                    "dominant_sensation": max_sensation,
                    "dominant_region": max_region,
                    "dominant_intensity": max_value,
                    "comfort_level": comfort,
                    "posture_effect": posture.get("posture", "Neutral posture"),
                    "movement_quality": posture.get("movement", "Natural movements"),
                    "behavioral_impact": "Minimal impact on behavior",
                    "regions_summary": {}
                }
    
    async def get_temperature_effects(self) -> Dict[str, Any]:
        """
        Get the effects of current temperature on expression and behavior
        
        Returns:
            Temperature effects analysis
        """
        with trace(workflow_name="Temperature_Analysis", group_id=self.trace_group_id):
            # Try using orchestrator for temperature effects
            try:
                result = await self.process_body_experience({
                    "action": "analyze_temperature"
                })
                
                if "temperature_effects" in result:
                    return result["temperature_effects"]
            except Exception as e:
                logger.error(f"Temperature orchestration failed, using fallback: {e}")
            
            try:
                result = await Runner.run(
                    self.temperature_agent,
                    "Analyze how the current temperature affects expression and behavior"
                )
                
                temp_effect_output = result.final_output_as(TemperatureEffect)
                
                return {
                    "effect_on_tone": temp_effect_output.effect_on_tone,
                    "effect_on_posture": temp_effect_output.effect_on_posture,
                    "effect_on_interaction": temp_effect_output.effect_on_interaction,
                    "expression_examples": temp_effect_output.expression_examples
                }
            except Exception as e:
                logger.error(f"Error getting temperature effects: {e}")
                
                # Fallback to direct model data if agent fails
                return await self._get_current_temperature_effects(RunContextWrapper(context=None))
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """
        Run maintenance on the somatosensory system
        
        Returns:
            Maintenance results
        """
        with trace(workflow_name="Somatic_Maintenance", group_id=self.trace_group_id):
            # Reset fatigue
            old_fatigue = self.body_state["fatigue"]
            self.body_state["fatigue"] = max(0.0, self.body_state["fatigue"] - 0.5)
            
            # Reduce tension
            old_tension = self.body_state["tension"]
            self.body_state["tension"] = max(0.0, self.body_state["tension"] - 0.3)
            
            # Clean up sensation memory for all regions (keep last 10)
            for region in self.body_regions.values():
                if len(region.sensation_memory) > 10:
                    region.sensation_memory = region.sensation_memory[-10:]
            
            # Decay associations that haven't been reinforced
            decay_count = 0
            for trigger, associations in self.memory_linked_sensations["associations"].items():
                for region, stimuli in associations.items():
                    for stimulus_type, strength in list(stimuli.items()):
                        # Decay the association
                        new_strength = strength * (1.0 - self.memory_linked_sensations["memory_decay"])
                        
                        # Remove if too weak, otherwise update
                        if new_strength < 0.05:
                            del stimuli[stimulus_type]
                            decay_count += 1
                        else:
                            stimuli[stimulus_type] = new_strength
            
            return {
                "fatigue_reduced": old_fatigue - self.body_state["fatigue"],
                "tension_reduced": old_tension - self.body_state["tension"],
                "associations_decayed": decay_count,
                "pain_memories_count": len(self.pain_model["pain_memories"]),
                "current_pain_tolerance": self.pain_model["tolerance"]
            }
    
    async def get_sensory_influence(self, message_text: str) -> Dict[str, Any]:
        """
        Get sensory influences to potentially include in a response
        
        Args:
            message_text: The message being formed
            
        Returns:
            Sensory influences that could be incorporated
        """
        with trace(workflow_name="Sensory_Influence", group_id=self.trace_group_id):
            # Use orchestrator for sensory influence
            try:
                result = await self.process_body_experience({
                    "action": "get_sensory_influence",
                    "message_text": message_text
                })
                
                if isinstance(result, dict) and "should_express" in result:
                    return result
            except Exception as e:
                logger.error(f"Sensory influence orchestration failed, using fallback: {e}")
        
            # Check if current body state is significant enough to express
            comfort_level = await self._calculate_overall_comfort(RunContextWrapper(context=None))
            
            # Prepare results
            results = {
                "should_express": False,
                "expressions": [],
                "tone_influence": None,
                "posture_influence": None
            }
            
            # Determine if we should express sensations based on comfort level
            # Extreme comfort or discomfort is more likely to be expressed
            expression_probability = 0.3  # Base probability
            
            if abs(comfort_level) > 0.7:
                expression_probability = 0.8  # High for extreme states
            elif abs(comfort_level) > 0.4:
                expression_probability = 0.5  # Medium for moderate states
            
            # Roll for expression
            if random.random() < expression_probability:
                results["should_express"] = True
                
                # Get significant sensations to potentially express
                significant_sensations = []
                
                for name, region in self.body_regions.items():
                    dominant = self._get_dominant_sensation(region)
                    
                    # Get value for dominant sensation
                    if dominant == "temperature":
                        value = abs(region.temperature - 0.5) * 2.0
                    else:
                        value = getattr(region, dominant, 0.0)
                    
                    # If significant, add to list
                    if value >= self.response_influence["expression_threshold"]:
                        significant_sensations.append({
                            "region": name,
                            "sensation": dominant,
                            "intensity": value
                        })
                
                # Sort by intensity and select top sensations to express
                significant_sensations.sort(key=lambda x: x["intensity"], reverse=True)
                sensations_to_express = significant_sensations[:self.response_influence["max_expressions_per_response"]]
                
                # Generate expressions for selected sensations
                for sensation in sensations_to_express:
                    try:
                        expression = await self.generate_sensory_expression(
                            stimulus_type=sensation["sensation"],
                            body_region=sensation["region"]
                        )
                        
                        if expression:
                            results["expressions"].append({
                                "text": expression,
                                "region": sensation["region"],
                                "sensation": sensation["sensation"],
                                "intensity": sensation["intensity"]
                            })
                    except Exception as e:
                        logger.error(f"Error generating expression: {e}")
                
                # Get temperature effects on tone
                try:
                    temperature_effects = await self.get_temperature_effects()
                    results["tone_influence"] = temperature_effects.get("effect_on_tone")
                    results["posture_influence"] = temperature_effects.get("effect_on_posture")
                except Exception as e:
                    logger.error(f"Error getting temperature effects: {e}")
            
            return results
    
    async def link_memory_to_sensation(self, 
                                   memory_id: str, 
                                   sensation_type: str,
                                   body_region: str,
                                   intensity: float = 0.5) -> Dict[str, Any]:
        """
        Link a memory to a physical sensation
        
        Args:
            memory_id: ID of the memory to link
            sensation_type: Type of sensation to link
            body_region: Body region to associate
            intensity: Intensity of the association
            
        Returns:
            Result of the link operation
        """
        with trace(workflow_name="Memory_Sensation_Link", group_id=self.trace_group_id):
            # Validate body region
            if body_region not in self.body_regions:
                return {"error": f"Invalid body region: {body_region}"}
            
            # Validate sensation type
            valid_types = ["pressure", "temperature", "pain", "pleasure", "tingling"]
            if sensation_type not in valid_types:
                return {"error": f"Invalid sensation type: {sensation_type}"}
            
            # Use orchestrator for memory linking
            try:
                result = await self.process_body_experience({
                    "action": "link_memory",
                    "memory_id": memory_id,
                    "sensation_type": sensation_type,
                    "body_region": body_region,
                    "intensity": intensity
                })
                
                if isinstance(result, dict) and "success" in result:
                    return result
            except Exception as e:
                logger.error(f"Memory linking orchestration failed, using fallback: {e}")
            
            # Get memory content from memory core if available
            memory_text = None
            if self.memory_core:
                try:
                    memory = await self.memory_core.get_memory_by_id(memory_id)
                    if memory:
                        memory_text = memory.get("memory_text", "")
                except Exception as e:
                    logger.error(f"Error getting memory: {e}")
            
            # Get or create trigger from memory
            trigger = memory_id
            if memory_text:
                # Use first 50 chars of memory as trigger
                trigger = memory_text[:50].strip()
            
            # Create or update association
            if trigger not in self.memory_linked_sensations["associations"]:
                self.memory_linked_sensations["associations"][trigger] = {}
            
            if body_region not in self.memory_linked_sensations["associations"][trigger]:
                self.memory_linked_sensations["associations"][trigger][body_region] = {}
            
            # Set the association directly (stronger than learning)
            self.memory_linked_sensations["associations"][trigger][body_region][sensation_type] = intensity
            
            # If it's a pain sensation, also create a pain memory
            if sensation_type == "pain" and intensity >= self.pain_model["threshold"]:
                pain_memory = PainMemory(
                    intensity=intensity,
                    location=body_region,
                    cause=f"Memory: {trigger}",
                    duration=1.0,  # Default duration
                    timestamp=datetime.datetime.now(),
                    associated_memory_id=memory_id
                )
                self.pain_model["pain_memories"].append(pain_memory)
            
            return {
                "success": True,
                "trigger": trigger,
                "body_region": body_region,
                "sensation_type": sensation_type,
                "intensity": intensity,
                "memory_id": memory_id
            }
    
    async def get_somatic_memory(self, memory_id: str) -> Dict[str, Any]:
        """
        Get somatic memory associated with a memory ID
        
        Args:
            memory_id: Memory ID to check
            
        Returns:
            Associated somatic memories if any
        """
        with trace(workflow_name="Get_Somatic_Memory", group_id=self.trace_group_id):
            # Use orchestrator for somatic memory retrieval
            try:
                result = await self.process_body_experience({
                    "action": "get_somatic_memory",
                    "memory_id": memory_id
                })
                
                if isinstance(result, dict) and "has_somatic_memory" in result:
                    return result
            except Exception as e:
                logger.error(f"Somatic memory orchestration failed, using fallback: {e}")
                
            # Fallback implementation
            result = {
                "memory_id": memory_id,
                "has_somatic_memory": False,
                "pain_memories": [],
                "associations": {}
            }
            
            # Check pain memories
            pain_memories = [m for m in self.pain_model["pain_memories"] 
                           if m.associated_memory_id == memory_id]
            
            if pain_memories:
                result["has_somatic_memory"] = True
                result["pain_memories"] = [memory.model_dump() for memory in pain_memories]
            
            # Get memory content from memory core if available
            memory_text = None
            if self.memory_core:
                try:
                    memory = await self.memory_core.get_memory_by_id(memory_id)
                    if memory:
                        memory_text = memory.get("memory_text", "")
                except Exception as e:
                    logger.error(f"Error getting memory: {e}")
            
            # Check associations using memory ID and text as possible triggers
            triggers = [memory_id]
            if memory_text:
                triggers.append(memory_text[:50].strip())
            
            for trigger in triggers:
                if trigger in self.memory_linked_sensations["associations"]:
                    result["has_somatic_memory"] = True
                    result["associations"][trigger] = self.memory_linked_sensations["associations"][trigger]
            
            return result

    def set_temperature(self, body_region: str, temperature_value: float) -> Dict[str, Any]:
        """
        Directly set temperature for a body region
        
        Args:
            body_region: Body region to update
            temperature_value: Temperature value (0.0=freezing, 0.5=neutral, 1.0=very hot)
            
        Returns:
            Result of the update
        """
        # Validate body region
        if body_region not in self.body_regions:
            return {"error": f"Invalid body region: {body_region}"}
        
        # Clamp temperature to valid range
        temperature_value = max(0.0, min(1.0, temperature_value))
        
        # Update the region
        region = self.body_regions[body_region]
        old_temp = region.temperature
        region.temperature = temperature_value
        region.last_update = datetime.datetime.now()
        
        # Add to sensation memory
        memory_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "temperature",
            "intensity": temperature_value,
            "cause": "direct setting",
            "duration": 1.0  # Default duration
        }
        region.sensation_memory.append(memory_entry)
        
        # Keep memory size manageable
        if len(region.sensation_memory) > 20:
            region.sensation_memory = region.sensation_memory[-20:]
        
        return {
            "body_region": body_region,
            "old_temperature": old_temp,
            "new_temperature": temperature_value,
            "time": datetime.datetime.now().isoformat()
        }
