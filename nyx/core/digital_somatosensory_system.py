# nyx/core/digital_somatosensory_system.py

import logging
import asyncio
import datetime
import json
import random
import math
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import numpy as np
from nyx.core.reward_system import RewardSignal

from agents import (
    Agent, Runner, trace, function_tool, 
    RunContextWrapper, handoff, ModelSettings,
    InputGuardrail, GuardrailFunctionOutput
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
    sensitivity: float = Field(1.0, description="Base sensitivity multiplier for this region")
    erogenous_level: float = Field(0.0, ge=0.0, le=1.0, description="Degree to which region is erogenous") # 0 = none, 1 = high

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
    effects: Dict[str, Any] = Field({}, description="Effects of the stimulus")
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

# =============== Context Classes ===============

class SomatosensorySystemContext(BaseModel):
    """Context for somatosensory system operations"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    current_operation: Optional[str] = None
    operation_start_time: Optional[datetime.datetime] = None
    memory_references: List[str] = Field(default_factory=list)
    emotional_state: Dict[str, Any] = Field(default_factory=dict)
    hormone_data: Dict[str, float] = Field(default_factory=dict)

# =============== System Hooks ===============

class SomatosensorySystemHooks(RunHooks):
    """Lifecycle hooks for the somatosensory system"""
    
    async def on_agent_start(self, context, agent):
        """Called before the agent is invoked"""
        logger.debug(f"Starting agent: {agent.name}")
        with custom_span(
            name="somatosensory_agent_start",
            data={"agent_name": agent.name, "context_data": str(context.context)}
        ) as span:
            # Update operation tracking
            if hasattr(context.context, "operation_start_time") and context.context.operation_start_time is None:
                context.context.operation_start_time = datetime.datetime.now()
    
    async def on_agent_end(self, context, agent, output):
        """Called when the agent produces a final output"""
        logger.debug(f"Agent {agent.name} finished with output type: {type(output).__name__}")
        with custom_span(
            name="somatosensory_agent_end",
            data={"agent_name": agent.name, "output_type": type(output).__name__}
        ) as span:
            # Calculate execution time if we have start time
            if hasattr(context.context, "operation_start_time") and context.context.operation_start_time:
                execution_time = (datetime.datetime.now() - context.context.operation_start_time).total_seconds()
                logger.debug(f"Agent execution time: {execution_time:.2f}s")
    
    async def on_handoff(self, context, from_agent, to_agent):
        """Called when a handoff occurs"""
        logger.debug(f"Handoff from {from_agent.name} to {to_agent.name}")
        with custom_span(
            name="somatosensory_handoff",
            data={"from_agent": from_agent.name, "to_agent": to_agent.name}
        ) as span:
            # You could add additional logic here, like updating context
            pass

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
        self.memory_core = memory_core
        self.emotional_core = emotional_core
        self.reward_system = reward_system 
        self.hormone_system = hormone_system
        self.needs_system = needs_system
        
        # Initialize body regions
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
        
        # Initialize agents using the OpenAI Agents SDK
        self._init_agents()
        
        logger.info("Digital Somatosensory System initialized")
        
    # =============== Agent Initialization ===============
    
    def _init_agents(self):
        """Initialize the agents using OpenAI Agents SDK."""
        # Create stimulus validation guardrail
        self.stimulus_validator = self._create_stimulus_validator()
        
        # Create specialized agents for different aspects of body experience
        self.expression_agent = self._create_expression_agent()
        self.body_state_agent = self._create_body_state_agent()
        self.temperature_agent = self._create_temperature_agent()
        
        # Create the main orchestrator agent with handoffs to specialized agents
        self.body_orchestrator = self._create_orchestrator_agent()
        
    def _create_stimulus_validator(self):
        """Create guardrail for validating stimulus inputs."""
        # Create a validation agent that can be used as a guardrail
        validation_agent = Agent(
            name="Stimulus Validator",
            instructions="""
            You validate inputs for the Digital Somatosensory System.
            
            Check that:
            1. Stimulus types are valid (pressure, temperature, pain, pleasure, tingling)
            2. Body regions are valid based on the provided list
            3. Intensity values are within range 0.0-1.0
            4. Duration values are positive
            
            Return validation results and reasoning.
            """,
            tools=[function_tool(self._get_valid_body_regions)],
            output_type=StimulusValidationOutput,
            model="gpt-4o", # Explicitly specify model
            model_settings=ModelSettings(temperature=0.1) # Low temperature for consistency
        )
        
        return validation_agent
    
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
                function_tool(self._get_region_state),
                function_tool(self._get_current_temperature_effects),
                function_tool(self._get_pain_expression)
            ],
            output_type=SensoryExpression,
            model="gpt-4o", # Explicitly specify model
            model_settings=ModelSettings(temperature=0.7)  # Higher creativity for expressions
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
                function_tool(self._get_all_region_states),
                function_tool(self._calculate_overall_comfort),
                function_tool(self._get_posture_effects)
            ],
            output_type=BodyStateOutput,
            model="gpt-4o", # Explicitly specify model
            model_settings=ModelSettings(temperature=0.2)  # Lower temperature for consistency
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
                function_tool(self._get_ambient_temperature),
                function_tool(self._get_body_temperature),
                function_tool(self._get_temperature_comfort)
            ],
            output_type=TemperatureEffect,
            model="gpt-4o", # Explicitly specify model
            model_settings=ModelSettings(temperature=0.4)  # Moderate temperature for variation
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
                function_tool(self._process_stimulus_tool),
                function_tool(self._get_region_state),
                function_tool(self._get_all_region_states),
                function_tool(self._update_body_temperature),
                function_tool(self._calculate_overall_comfort),
                function_tool(self._process_memory_trigger),
                function_tool(self._link_memory_to_sensation_tool)
            ],
            input_guardrails=[
                InputGuardrail(guardrail_function=self._validate_input)
            ],
            model="gpt-4o", # Explicitly specify model
            model_settings=ModelSettings(temperature=0.2),
            output_type=StimulusProcessingResult
        )
    
    # =============== Guardrail Functions ===============
    
    async def _validate_input(self, ctx, agent, input_data):
        """Validate input data for the body orchestrator."""
        # Parse input into the appropriate format
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                # If not JSON, treat as plain text request
                input_data = {"action": "free_text_request", "text": input_data}
        
        # If the input is a free text request, no need to validate
        if input_data.get("action") == "free_text_request":
            return GuardrailFunctionOutput(
                output_info={"is_valid": True, "reason": "Free text request"},
                tripwire_triggered=False
            )
        
        # Create validation context
        validation_input = {
            "stimulus_type": input_data.get("stimulus_type"),
            "body_region": input_data.get("body_region"),
            "intensity": input_data.get("intensity"),
            "action": input_data.get("action")
        }
        
        # Run validation agent with tracing
        with trace(workflow_name="Stimulus_Validation", group_id=self.trace_group_id):
            result = await Runner.run(
                self.stimulus_validator, 
                validation_input,
                run_config=RunConfig(
                    workflow_name="StimulusValidation",
                    trace_id=None,  # Auto-generate
                    trace_metadata={"input_type": input_data.get("action", "unknown")}
                )
            )
            
            validation_output = result.final_output
            
            # Return guardrail output based on validation
            return GuardrailFunctionOutput(
                output_info=validation_output,
                tripwire_triggered=not validation_output.is_valid
            )
    
    # =============== Tool Functions ===============
    
    @function_tool
    async def _get_valid_body_regions(self, ctx: RunContextWrapper) -> List[str]:
        """Get a list of valid body regions."""
        return list(self.body_regions.keys())
    
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
        # Create a custom span for tracing this operation
        with custom_span(
            name="process_stimulus", 
            data={
                "stimulus_type": stimulus_type,
                "body_region": body_region,
                "intensity": intensity,
                "cause": cause,
                "duration": duration
            }
        ):
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
            
            # Update reward system if available
            if self.reward_system:
                reward_value = 0.0
                if stimulus_type == "pleasure" and intensity >= 0.5:
                    reward_value = min(1.0, (intensity - 0.4) * 0.9 * (1.0 + region.erogenous_level))
                elif stimulus_type == "pain" and intensity >= self.pain_model["threshold"]:
                    reward_value = -min(1.0, (intensity / max(0.1, self.pain_model["tolerance"])) * 0.6)
                
                if abs(reward_value) > 0.1:
                    reward_signal = RewardSignal(
                        value=reward_value,
                        source=f"somatic_{body_region}",
                        context={"stimulus_type": stimulus_type, "intensity": intensity, "cause": cause},
                        timestamp=datetime.datetime.now().isoformat()
                    )
                    
                    # Add reward signal to result for tracking
                    result["reward_value"] = reward_value
                    
                    # Process reward signal asynchronously
                    asyncio.create_task(self.reward_system.process_reward_signal(reward_signal))
            
            # Update emotional core if available
            if self.emotional_core:
                emotional_impact = {}
                
                if stimulus_type == "pleasure" and region.pleasure > 0.5:
                    scaled_intensity = (region.pleasure - 0.4) * 1.5
                    self.emotional_core.update_neurochemical("nyxamine", scaled_intensity * 0.40)
                    self.emotional_core.update_neurochemical("oxynixin", scaled_intensity * 0.15)
                    emotional_impact = {"nyxamine": scaled_intensity * 0.40, "oxynixin": scaled_intensity * 0.15}
                    
                elif stimulus_type == "pain" and region.pain > self.pain_model["threshold"]:
                    effective_pain = region.pain / max(0.1, self.pain_model["tolerance"])
                    self.emotional_core.update_neurochemical("cortanyx", effective_pain * 0.45)
                    self.emotional_core.update_neurochemical("adrenyx", effective_pain * 0.25)
                    self.emotional_core.update_neurochemical("seranix", -effective_pain * 0.10)
                    emotional_impact = {"cortanyx": effective_pain * 0.45, "adrenyx": effective_pain * 0.25, "seranix": -effective_pain * 0.10}
                
                if emotional_impact:
                    result["emotional_impact"] = emotional_impact
            
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
    
    @function_tool
    async def _process_memory_trigger(self, ctx: RunContextWrapper, trigger: str) -> Dict[str, Any]:
        """
        Process a memory trigger that may have associated physical responses
        
        Args:
            trigger: The memory trigger text
            
        Returns:
            Results of processing the trigger
        """
        results = {"triggered_responses": []}
        
        # Check if trigger exists in associations
        if trigger in self.memory_linked_sensations["associations"]:
            for region, stimuli in self.memory_linked_sensations["associations"][trigger].items():
                for stim_type, strength in stimuli.items():
                    # Only trigger if association is strong enough
                    if strength > 0.3:
                        # Scale intensity by association strength
                        intensity = strength * 0.7
                        
                        # Process the stimulus
                        response = await self._process_stimulus_tool(
                            ctx,
                            stimulus_type=stim_type,
                            body_region=region,
                            intensity=intensity,
                            cause=f"Memory trigger: {trigger}",
                            duration=1.0
                        )
                        
                        # Add to triggered responses
                        results["triggered_responses"].append({
                            "region": region,
                            "stimulus": stim_type,
                            "intensity": intensity,
                            "association_strength": strength,
                            "response": response
                        })
        
        return results
    
    @function_tool
    async def _link_memory_to_sensation_tool(
        self, 
        ctx: RunContextWrapper,
        memory_id: str,
        sensation_type: str,
        body_region: str,
        intensity: float = 0.5,
        trigger_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Link a memory to a physical sensation
        
        Args:
            memory_id: ID of the memory to link
            sensation_type: Type of sensation to link (pressure, temperature, pain, pleasure, tingling)
            body_region: Body region to associate
            intensity: Intensity of the association (0.0-1.0)
            trigger_text: Optional text to use as trigger (defaults to memory_id if not provided)
            
        Returns:
            Result of the link operation
        """
        # Validate body region
        if body_region not in self.body_regions:
            return {"error": f"Invalid body region: {body_region}", "success": False}
        
        # Validate sensation type
        valid_types = ["pressure", "temperature", "pain", "pleasure", "tingling"]
        if sensation_type not in valid_types:
            return {"error": f"Invalid sensation type: {sensation_type}", "success": False}
        
        # Get memory content from memory core if available
        memory_text = None
        if self.memory_core:
            try:
                memory = await self.memory_core.get_memory_by_id(memory_id)
                if memory:
                    memory_text = memory.get("memory_text", "")
            except Exception as e:
                logger.error(f"Error getting memory: {e}")
        
        # Use provided trigger or create from memory
        trigger = trigger_text or memory_id
        if not trigger_text and memory_text:
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
        with trace(workflow_name="Somatic_Initialize", group_id=self.trace_group_id):
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
    
    async def update(self, ambient_temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Update the somatosensory system, including decay, environmental effects,
        and interplay with the emotional core.

        Args:
            ambient_temperature: Optional ambient temperature to use for update (0.0-1.0).

        Returns:
            Updated comprehensive body state analysis dictionary.
        """
        # Use trace for the whole update process
        with trace(workflow_name="Somatic_Update", group_id=self.trace_group_id):
            now = datetime.datetime.now()

            # 1. Calculate time since last update
            last_update = self.body_state.get("last_update", now)
            duration = (now - last_update).total_seconds()

            # Avoid excessive updates or large jumps if duration is very small or large
            if duration <= 0.1:  # Less than 0.1 second, negligible change
                return await self.get_body_state()  # Return current state analysis

            duration = min(duration, 3600.0)  # Cap duration at 1 hour to prevent huge jumps

            # Update timestamp immediately
            self.body_state["last_update"] = now

            # 2. Update Temperature Model if ambient temperature provided
            if ambient_temperature is not None:
                # Ensure ambient temp is within valid range
                ambient_temperature = max(0.0, min(1.0, ambient_temperature))
                try:
                    await self._update_body_temperature(
                        RunContextWrapper(context=None),
                        ambient_temperature=ambient_temperature,
                        duration=duration
                    )
                except Exception as e:
                    logger.error(f"Error updating body temperature: {e}")

            # 3. Decay Sensations Over Time
            try:
                self._decay_sensations(duration)
            except Exception as e:
                logger.error(f"Error decaying sensations: {e}")

            # 4. Process Pain Memory Updates
            try:
                self._decay_pain_memories()
                self._update_pain_tolerance()
            except Exception as e:
                logger.error(f"Error processing pain memories: {e}")

            # 5. Update Intrinsic Body State Metrics (e.g., fatigue)
            try:
                fatigue_increase_rate = 0.01 / 3600.0  # Per second rate (0.01 per hour)
                current_fatigue = self.body_state.get("fatigue", 0.0)
                self.body_state["fatigue"] = min(1.0, current_fatigue + (fatigue_increase_rate * duration))
            except Exception as e:
                logger.error(f"Error updating fatigue: {e}")

            # 6. Reflect Emotions onto Body State (Top-down: Emotion -> Body)
            if self.emotional_core:
                try:
                    # Get emotional state data
                    emotional_state_data = self.emotional_core.get_emotional_state()
                    neurochemicals = emotional_state_data.get("neurochemicals", {})
                    
                    # Process neurochemical effects on body state
                    self._process_neurochemical_effects(neurochemicals, duration)
                except Exception as e:
                    logger.error(f"Error reflecting emotions onto body state: {e}")

            # 7. Use the body orchestration agent to get the final, integrated body state analysis
            body_state_input = {
                "action": "analyze_body_state",
                "ambient_temperature": ambient_temperature,
                "duration_since_last": duration
            }

            try:
                # Use the orchestrator agent to get the analysis
                result = await Runner.run(
                    self.body_orchestrator,
                    json.dumps(body_state_input)
                )
                
                output = result.final_output
                
                # Extract body state analysis if available
                if hasattr(output, "body_state_impact") and output.body_state_impact:
                    return output.body_state_impact
                elif isinstance(output, dict) and "body_state_impact" in output:
                    return output["body_state_impact"]
                
                # Fallback to direct method if needed
                return await self.get_body_state()
            except Exception as e:
                logger.error(f"Error running body state analysis: {e}")
                return await self.get_body_state()
    
    def _process_neurochemical_effects(self, neurochemicals: Dict[str, Any], duration: float):
        """
        Process the effects of neurochemicals on body state
        
        Args:
            neurochemicals: Dictionary of neurochemical levels
            duration: Duration in seconds since last update
        """
        # Cortanyx (Stress) -> Increase Tension
        cortanyx_level = neurochemicals.get("cortanyx")
        if cortanyx_level is not None and cortanyx_level > 0.6:
            tension_increase = (cortanyx_level - 0.5) * 0.25  # Higher stress = more tension
            self.body_state["tension"] = min(1.0, self.body_state["tension"] + tension_increase * (duration / 60.0))
        
        # Seranix (Calm) -> Reduce Tension
        seranix_level = neurochemicals.get("seranix")
        if seranix_level is not None and seranix_level > 0.7:
            tension_decrease = (seranix_level - 0.6) * 0.20  # Higher calm = less tension
            self.body_state["tension"] = max(0.0, self.body_state["tension"] - tension_decrease * (duration / 60.0))
        
        # Adrenyx (Fear/Excitement) -> Tingling / Temp Shift (Subtle)
        adrenyx_level = neurochemicals.get("adrenyx")
        if adrenyx_level is not None and adrenyx_level > 0.7:
            tingle_intensity = (adrenyx_level - 0.6) * 0.15
            # Apply subtle tingling to specific regions
            for region_name in ["skin", "hands", "feet"]:
                if region_name in self.body_regions:
                    self.body_regions[region_name].tingling = min(
                        1.0, 
                        self.body_regions[region_name].tingling + tingle_intensity * (duration / 120.0)
                    )
            
            # Optional: Simulate feeling cold (subtle effect)
            temp_shift = -(adrenyx_level - 0.6) * 0.01
            self.temperature_model["body_temperature"] = max(
                0.0, 
                min(1.0, self.temperature_model["body_temperature"] + temp_shift * (duration / 60.0))
            )
        
        # Oxynixin (Bonding) -> Pain Tolerance (Very Slow Adjustment)
        oxynixin_level = neurochemicals.get("oxynixin")
        if oxynixin_level is not None and oxynixin_level > 0.6:
            tolerance_increase = (oxynixin_level - 0.5) * 0.10
            self.pain_model["tolerance"] = min(
                0.95, 
                self.pain_model["tolerance"] + tolerance_increase * (duration / 3600.0)
            )
    
    async def process_body_experience(self, body_experience: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a body experience input using the orchestrator agent
        
        Args:
            body_experience: Dictionary containing body experience data
            
        Returns:
            Results of processing the body experience
        """
        # Create context object
        context = SomatosensorySystemContext(
            current_operation="process_body_experience",
            operation_start_time=datetime.datetime.now()
        )
        
        with trace(workflow_name="Body_Experience", group_id=self.trace_group_id):
            # Add trace metadata
            trace_metadata({
                "operation": "process_body_experience",
                "input_type": body_experience.get("action", "stimulus")
            })
            
            # Convert input to JSON string if needed
            if not isinstance(body_experience, str):
                input_data = json.dumps(body_experience)
            else:
                input_data = body_experience
            
            # Run through orchestrator agent with hooks
            result = await Runner.run(
                self.body_orchestrator, 
                input_data,
                context=context,
                hooks=self.hooks,
                run_config=RunConfig(
                    workflow_name="BodyExperience",
                    trace_id=None,  # Auto-generate
                    trace_metadata={
                        "action": body_experience.get("action", "unknown"),
                        "stimulus_type": body_experience.get("stimulus_type"),
                        "body_region": body_experience.get("body_region")
                    }
                )
            )
            
            # Extract output
            if hasattr(result.final_output, "model_dump"):
                return result.final_output.model_dump()
            else:
                return result.final_output
                
    async def process_stimulus(
        self, 
        stimulus_type: str, 
        body_region: str, 
        intensity: float, 
        cause: str = "", 
        duration: float = 1.0
    ) -> Dict[str, Any]:
        """
        Process a sensory stimulus on a specific region
        
        Args:
            stimulus_type: Type of stimulus (pressure, temperature, pain, pleasure, tingling)
            body_region: Body region receiving the stimulus
            intensity: Intensity of the stimulus (0.0-1.0)
            cause: Cause of the stimulus
            duration: Duration of the stimulus in seconds
            
        Returns:
            Results of processing the stimulus
        """
        with trace(workflow_name="Process_Stimulus", group_id=self.trace_group_id):
            # Add trace metadata
            trace_metadata({
                "stimulus_type": stimulus_type,
                "body_region": body_region,
                "intensity": intensity,
                "duration": duration
            })
            
            # Build input for orchestrator
            stimulus_data = {
                "stimulus_type": stimulus_type,
                "body_region": body_region,
                "intensity": intensity,
                "cause": cause,
                "duration": duration,
                "generate_expression": True
            }
            
            # Create context object
            context = SomatosensorySystemContext(
                current_operation="process_stimulus",
                operation_start_time=datetime.datetime.now()
            )
            
            # Process through orchestrator
            try:
                return await self.process_body_experience(stimulus_data)
            except Exception as e:
                logger.error(f"Error processing stimulus: {e}")
                
                # Fallback to direct processing if orchestrator fails
                try:
                    # Use the tool function directly
                    return await self._process_stimulus_tool(
                        RunContextWrapper(context=context),
                        stimulus_type=stimulus_type,
                        body_region=body_region,
                        intensity=intensity,
                        cause=cause,
                        duration=duration
                    )
                except Exception as e2:
                    logger.error(f"Error in fallback stimulus processing: {e2}")
                    return {
                        "error": str(e2),
                        "stimulus_type": stimulus_type,
                        "body_region": body_region,
                        "intensity": intensity
                    }
    
    async def simulate_gratification_sensation(self, intensity: float = 1.0) -> Dict[str, Any]:
        """
        Simulate gratification sensations
        
        Args:
            intensity: Intensity of gratification (0.0-1.0)
            
        Returns:
            Results of simulation
        """
        with trace(workflow_name="Gratification_Simulation", group_id=self.trace_group_id):
            logger.info(f"Simulating gratification (Intensity: {intensity:.2f})")
            
            # Process through orchestrator
            try:
                return await self.process_body_experience({
                    "action": "simulate_gratification",
                    "intensity": intensity
                })
            except Exception as e:
                logger.error(f"Error in gratification simulation orchestration: {e}")
                
                # Fallback implementation
                results = {}
                
                # Calculate pleasure intensity based on input intensity
                pleasure_intensity = 0.8 + intensity * 0.2
                
                # Apply pleasure to erogenous regions with varying intensity
                regions = ["genitals", "inner_thighs", "skin", "chest"]
                tasks = []
                
                # Create tasks for parallel processing
                for r in regions:
                    if r in self.body_regions:
                        # Scale intensity by erogenous level
                        scaled_intensity = min(
                            1.0, 
                            pleasure_intensity * (1.0 + self.body_regions[r].erogenous_level) * random.uniform(0.8, 1.2)
                        )
                        
                        # Add task
                        tasks.append(
                            self.process_stimulus(
                                "pleasure", 
                                r, 
                                scaled_intensity,
                                "gratification_event", 
                                2.0 + intensity * 3.0
                            )
                        )
                
                # Run all tasks in parallel
                results["pleasure_simulation"] = await asyncio.gather(*tasks)
                
                # Reduce tension
                old_tension = self.body_state["tension"]
                tension_reduction = 0.5 + intensity * 0.4
                self.body_state["tension"] = max(0.0, self.body_state["tension"] - tension_reduction)
                results["tension_reduction"] = tension_reduction
                
                # Trigger hormone system if available
                if self.hormone_system:
                    await self.hormone_system.trigger_post_gratification_response(
                        RunContextWrapper(context=None), 
                        intensity
                    )
                    results["hormone_response_triggered"] = True
                
                # Generate reward if reward system available
                if self.reward_system:
                    reward_value = 0.85 + intensity * 0.15
                    reward_signal = RewardSignal(
                        reward_value, 
                        "gratification_event", 
                        {"intensity": intensity},
                        datetime.datetime.now().isoformat()
                    )
                    await self.reward_system.process_reward_signal(reward_signal)
                    results["reward_generated"] = reward_value
                
                # Update needs if needs system available
                if self.needs_system:
                    needs_tasks = [
                        self.needs_system.satisfy_need("physical_closeness", 0.6 * intensity),
                        self.needs_system.satisfy_need("drive_expression", 0.8 * intensity),
                        self.needs_system.satisfy_need("intimacy", 0.3 * intensity)
                    ]
                    await asyncio.gather(*needs_tasks)
                    results["needs_satisfied"] = ["physical_closeness", "drive_expression", "intimacy"]
                
                logger.info("Gratification simulation complete")
                return results
    
    async def process_trigger(self, trigger: str) -> Dict[str, Any]:
        """
        Process a trigger with associated body memories
        
        Args:
            trigger: Trigger text to process
            
        Returns:
            Results of processing the trigger
        """
        with trace(workflow_name="Process_Trigger", group_id=self.trace_group_id):
            try:
                # Process through orchestrator
                return await self.process_body_experience({
                    "action": "process_trigger",
                    "trigger": trigger
                })
            except Exception as e:
                logger.error(f"Error in trigger processing orchestration: {e}")
                
                # Fallback using tool directly
                return await self._process_memory_trigger(
                    RunContextWrapper(context=None),
                    trigger=trigger
                )
    
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
        with trace(workflow_name="Generate_Expression", group_id=self.trace_group_id):
            try:
                # Process through orchestrator with expression request
                result = await self.process_body_experience({
                    "action": "generate_expression",
                    "stimulus_type": stimulus_type,
                    "body_region": body_region,
                    "generate_expression": True
                })
                
                # Extract expression from result
                if isinstance(result, dict) and "expression" in result:
                    return result["expression"]
                elif hasattr(result, "expression") and result.expression:
                    return result.expression
            except Exception as e:
                logger.error(f"Error in expression orchestration: {e}")
            
            # Fallback: Use expression agent directly
            try:
                input_text = "Generate an expression of "
                
                if stimulus_type:
                    input_text += f"{stimulus_type} sensation "
                else:
                    input_text += "the dominant sensation "
                
                if body_region:
                    input_text += f"in the {body_region} "
                else:
                    input_text += "in the most significant body region "
                
                # Run expression agent
                result = await Runner.run(self.expression_agent, input_text)
                
                # Extract expression
                if result.final_output and hasattr(result.final_output, "expression_text"):
                    return result.final_output.expression_text
            except Exception as e:
                logger.error(f"Error in direct expression generation: {e}")
            
            # If all else fails, generate a simple expression
            if body_region and body_region in self.body_regions:
                region = self.body_regions[body_region]
                dominant = self._get_dominant_sensation(region)
                
                if dominant == "neutral":
                    return None
                
                value = 0.0
                if dominant == "temperature":
                    value = abs(region.temperature - 0.5) * 2.0
                else:
                    value = getattr(region, dominant, 0.0)
                
                if value < self.response_influence["expression_threshold"]:
                    return None
                
                if dominant == "pain":
                    return await self._get_pain_expression(
                        RunContextWrapper(context=None),
                        value,
                        body_region
                    )
                
                return f"I feel a {dominant} sensation in my {body_region}."
            
            return None
    
    async def get_body_state(self) -> Dict[str, Any]:
        """
        Get a complete analysis of current body state
        
        Returns:
            Comprehensive body state analysis
        """
        with trace(workflow_name="Get_Body_State", group_id=self.trace_group_id):
            try:
                # Process through orchestrator with body state request
                result = await self.process_body_experience({
                    "action": "analyze_body_state"
                })
                
                # Extract body state from result
                if isinstance(result, dict) and "body_state_impact" in result:
                    return result["body_state_impact"]
            except Exception as e:
                logger.error(f"Error in body state orchestration: {e}")
            
            # Fallback: Use body state agent directly
            try:
                result = await Runner.run(
                    self.body_state_agent,
                    "Analyze the current body state across all regions"
                )
                
                if result.final_output and hasattr(result.final_output, "model_dump"):
                    return result.final_output.model_dump()
                elif hasattr(result.final_output, "__dict__"):
                    return result.final_output.__dict__
            except Exception as e:
                logger.error(f"Error in direct body state analysis: {e}")
            
            # Extreme fallback: Generate minimal state manually
            comfort = await self._calculate_overall_comfort(RunContextWrapper(context=None))
            
            # Find dominant sensation
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
                "posture_effect": "Neutral posture",
                "movement_quality": "Natural movements",
                "behavioral_impact": "Minimal impact on behavior",
                "regions_summary": {}
            }
    
    async def get_temperature_effects(self) -> Dict[str, Any]:
        """
        Get the effects of current temperature on expression and behavior
        
        Returns:
            Temperature effects analysis
        """
        with trace(workflow_name="Get_Temperature_Effects", group_id=self.trace_group_id):
            try:
                # Process through orchestrator with temperature effects request
                result = await self.process_body_experience({
                    "action": "analyze_temperature"
                })
                
                # Extract temperature effects from result
                if isinstance(result, dict) and "temperature_effects" in result:
                    return result["temperature_effects"]
            except Exception as e:
                logger.error(f"Error in temperature effects orchestration: {e}")
            
            # Fallback: Use temperature agent directly
            try:
                result = await Runner.run(
                    self.temperature_agent,
                    "Analyze how the current temperature affects expression and behavior"
                )
                
                if result.final_output and hasattr(result.final_output, "model_dump"):
                    return result.final_output.model_dump()
                elif hasattr(result.final_output, "__dict__"):
                    return result.final_output.__dict__
            except Exception as e:
                logger.error(f"Error in direct temperature effects analysis: {e}")
            
            # Extreme fallback: Get effects directly
            return await self._get_current_temperature_effects(RunContextWrapper(context=None))
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """
        Run maintenance on the somatosensory system
        
        Returns:
            Maintenance results
        """
        with trace(workflow_name="Somatic_Maintenance", group_id=self.trace_group_id):
            try:
                # Process through orchestrator with maintenance request
                result = await self.process_body_experience({
                    "action": "run_maintenance"
                })
                
                # Return orchestrator results if available
                if isinstance(result, dict) and "maintenance_results" in result:
                    return result["maintenance_results"]
            except Exception as e:
                logger.error(f"Error in maintenance orchestration: {e}")
            
            # Fallback maintenance implementation
            
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
        with trace(workflow_name="Get_Sensory_Influence", group_id=self.trace_group_id):
            try:
                # Create context
                context = SomatosensorySystemContext(
                    current_operation="get_sensory_influence",
                    operation_start_time=datetime.datetime.now()
                )
                
                # Process through orchestrator with sensory influence request with hooks
                result = await Runner.run(
                    self.body_orchestrator,
                    {
                        "action": "get_sensory_influence",
                        "message_text": message_text
                    },
                    context=context,
                    hooks=self.hooks,
                    run_config=RunConfig(
                        workflow_name="SensoryInfluence",
                        trace_id=None,  # Auto-generate
                        trace_metadata={"message_length": len(message_text)}
                    )
                )
                
                # Return orchestrator results if available
                if isinstance(result.final_output, dict) and "should_express" in result.final_output:
                    return result.final_output
            except Exception as e:
                logger.error(f"Error in sensory influence orchestration: {e}")
            
            # Fallback implementation
            
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
                                   intensity: float = 0.5,
                                   trigger_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Link a memory to a physical sensation
        
        Args:
            memory_id: ID of the memory to link
            sensation_type: Type of sensation to link
            body_region: Body region to associate
            intensity: Intensity of the association
            trigger_text: Optional text to use as trigger
            
        Returns:
            Result of the link operation
        """
        with trace(workflow_name="Link_Memory_Sensation", group_id=self.trace_group_id):
            try:
                # Process through orchestrator
                return await self.process_body_experience({
                    "action": "link_memory",
                    "memory_id": memory_id,
                    "sensation_type": sensation_type,
                    "body_region": body_region,
                    "intensity": intensity,
                    "trigger_text": trigger_text
                })
            except Exception as e:
                logger.error(f"Error in memory linking orchestration: {e}")
                
                # Fallback using tool directly
                return await self._link_memory_to_sensation_tool(
                    RunContextWrapper(context=None),
                    memory_id=memory_id,
                    sensation_type=sensation_type,
                    body_region=body_region,
                    intensity=intensity,
                    trigger_text=trigger_text
                )
    
    async def get_somatic_memory(self, memory_id: str) -> Dict[str, Any]:
        """
        Get somatic memory associated with a memory ID
        
        Args:
            memory_id: Memory ID to check
            
        Returns:
            Associated somatic memories if any
        """
        with trace(workflow_name="Get_Somatic_Memory", group_id=self.trace_group_id):
            try:
                # Process through orchestrator
                return await self.process_body_experience({
                    "action": "get_somatic_memory",
                    "memory_id": memory_id
                })
            except Exception as e:
                logger.error(f"Error in somatic memory retrieval orchestration: {e}")
                
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
