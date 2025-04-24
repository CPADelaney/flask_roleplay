# nyx/core/digital_somatosensory_system.py

import logging
import asyncio
import datetime
import json
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import numpy as np

from agents import (
    Agent, Runner, trace, function_tool, 
    RunContextWrapper, handoff, ModelSettings,
    InputGuardrail, GuardrailFunctionOutput, 
    Handoff, RunConfig,FunctionTool
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
    dominant_sensation: str = Field(..., description="The dominant sensation type")
    dominant_region: str = Field(..., description="The dominant body region")
    dominant_intensity: float = Field(..., description="Intensity of dominant sensation")
    comfort_level: float = Field(..., description="Overall comfort level (-1.0 to 1.0)")
    posture_effect: str = Field(..., description="Effect on posture description")
    movement_quality: str = Field(..., description="Quality of movement description")
    behavioral_impact: str = Field(..., description="Impact on behavior")
    regions_summary: Dict[str, Dict[str, float]] = Field(..., description="Summary of region states")
    pleasure_index: float = Field(..., description="Level of pleasure felt (-1.0 to 1.0)")    

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

class SomatosensorySystemHooks:
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
            pass  # Additional logic can be added here

    async def on_tool_start(self, context, agent, tool):
        """Called before a tool is invoked"""
        logger.debug(f"Starting tool: {tool.name} for agent {agent.name}")
        
    async def on_tool_end(self, context, agent, tool, result):
        """Called after a tool is invoked"""
        logger.debug(f"Tool {tool.name} completed for agent {agent.name}")

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
        self.hooks = SomatosensorySystemHooks()
        
        # Trace ID for connecting traces
        self.trace_group_id = f"somatic_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Initialize agents using the OpenAI Agents SDK
        self._init_agents()
        
        # Initialize cognitive arousal state and weights
        self._init_cognitive_arousal()

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

    async def process_stimulus_with_protection(self, 
                                             stimulus_type: str, 
                                             body_region: str, 
                                             intensity: float, 
                                             cause: str = "", 
                                             duration: float = 1.0) -> Dict[str, Any]:
        """
        Process a stimulus with protection against harmful actions.
        This should be the primary entry point for all stimulus processing.
        """
        return await self.harm_guardrail.process_stimulus_safely(
            stimulus_type, body_region, intensity, cause, duration
        )

    async def analyze_text_for_harmful_content(self, text: str) -> Dict[str, Any]:
        """Analyze text for harmful or sensation content with protection"""
        return await self.harm_guardrail.intercept_harmful_text(text)
            
    def _init_state_models(self):
        """Initialize the various state models and settings"""
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

        # Arousal state using the new model
        self.arousal_state = ArousalState()
        
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
    
    def _init_cognitive_arousal(self):
        """Initialize cognitive arousal systems"""
        # Default sensitivities to different stimuli (learned over time)
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
        
        # Working copy of cognitive triggers with their weights
        self.cognitive_turnons = dict(self.default_cognitive_turnons)
        
        # Learning rate for automatic adjustment of weights
        self.arousal_learning_rate = 0.05
        
        # Exposure history for analysis and auto-decay
        self.cognitive_exposure_history = {}
        
        # Per-partner affinity and emotional connection levels (0.0-1.0)
        self.partner_affinity = {}  # physical/appearance attraction
        self.partner_emoconn = {}   # emotional connection/bond
    
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

    @function_tool # Keep decorator here
    async def _get_valid_body_regions(self) -> List[str]: # <<< ADD self HERE
        """Get a list of valid body regions."""
        # Ensure body_regions has been initialized (safety check)
        if not hasattr(self, 'body_regions'):
             logger.error("_get_valid_body_regions called before self.body_regions was set.")
             return []
        # Access instance attribute via self
        return list(self.body_regions.keys()) # <<< USE self.body_regions HERE
        
    def _create_stimulus_validator(self):
            """Create the stimulus validation agent."""
            # --- DEBUGGING ---
            logger.info(">>> Entering _create_stimulus_validator")
            logger.info(f">>> Type of self: {type(self)}")
            logger.info(f">>> Does self have _get_valid_body_regions? {hasattr(self, '_get_valid_body_regions')}")
            if hasattr(self, '_get_valid_body_regions'):
                logger.info(f">>> Is it callable? {callable(getattr(self, '_get_valid_body_regions'))}")
            logger.info(f">>> Attributes available on self: {dir(self)}")
            # --- END DEBUGGING ---
    
            validation_agent = Agent(
                name="Stimulus Validator",
                instructions="""
                You validate inputs for the Digital Somatosensory System.
    
                Check that:
                1. Stimulus types are valid (pressure, temperature, pain, pleasure, tingling)
                2. Body regions are valid based on the provided list using the tool.
                3. Intensity values are within range 0.0-1.0
                4. Duration values are positive
    
                Return validation results and reasoning. Use the available tool to get valid regions.
                """,
                tools=[self._get_valid_body_regions], # This is the line causing the error (approx line 392/416)
                output_type=StimulusValidationOutput,
                model="gpt-4o",
                model_settings=ModelSettings(temperature=0.1)
            )
            logger.info(">>> Successfully created validation_agent (or crashed before this)") # Add this too
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
                function_tool(self._get_pain_expression),
                function_tool(self._get_arousal_expression_data)
            ],
            output_type=SensoryExpression,
            model="gpt-4o",
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
                function_tool(self._get_posture_effects),
                function_tool(self._get_arousal_expression_data)
            ],
            output_type=BodyStateOutput,
            model="gpt-4o",
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
            model="gpt-4o",
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
                function_tool(self._link_memory_to_sensation_tool),
                function_tool(self._get_arousal_state),
                function_tool(self._update_arousal_state)
            ],
            input_guardrails=[
                InputGuardrail(guardrail_function=self._validate_input)
            ],
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.2),
            output_type=StimulusProcessingResult
        )
    
    async def initiate_denial_loop(self, cycles: int = 3, base_intensity: float = 0.6) -> Dict[str, Any]:
        """
        Initiates a simulated denial loop that heightens arousal, tension, and control.
    
        Args:
            cycles: How many build/deny rounds to run.
            base_intensity: Starting pleasure intensity (scaled slightly up per round).
            
        Returns:
            Results of the denial loop process
        """
        with trace(workflow_name="Denial_Loop", group_id=self.trace_group_id):
            logger.info(f"Initiating denial loop: {cycles} cycles at base {base_intensity:.2f}")
            results = {"denial_cycles": []}
            
            for i in range(cycles):
                intensity = base_intensity + (i * 0.1)
                pleasure_regions = ["genitals", "inner_thighs", "breasts_nipples", "lips", "butt cheeks", "anus", "toes", "armpits", "neck", "feet"]

                logger.info(f"Cycle {i+1}: stimulating but withholding")
                stim_tasks = [
                    self.process_stimulus("pleasure", region, min(1.0, intensity), "denial_loop", duration=1.5)
                    for region in pleasure_regions
                    if region in self.body_regions
                ]
                await asyncio.gather(*stim_tasks)
    
                # Artificially increase tension + drive
                self.body_state["tension"] = min(1.0, self.body_state["tension"] + 0.2)
                
                # Apply needs system effects if available
                if self.needs_system:
                    await self.needs_system.decrease_need("pleasure_indulgence", 0.2, reason="denial_loop")
    
                # Apply reward system effects if available
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
                
                # Add cycle information to results
                results["denial_cycles"].append({
                    "cycle": i + 1,
                    "intensity": intensity,
                    "status": "stimulated_then_denied"
                })
                
                # Force a small wait between cycles
                await asyncio.sleep(0.2)
    
            # Final summary
            results["final_arousal_level"] = self.arousal_state.arousal_level
            results["final_tension"] = self.body_state["tension"]
            
            logger.info("Denial loop complete.")
            return results
            
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
            
            # Check arousal state for expression
            arousal_level = self.arousal_state.arousal_level
            if arousal_level > 0.6:
                results["should_express"] = True
                modifier = self.get_arousal_expression_modifier()
                results["expressions"].append({
                    "text": modifier["expression_hint"],
                    "region": "body",
                    "sensation": "arousal",
                    "intensity": arousal_level
                })
                results["tone_influence"] = modifier["tone_hint"]
                return results
            
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
                        expression = asyncio.create_task(self.generate_sensory_expression(
                            stimulus_type=sensation["sensation"],
                            body_region=sensation["region"]
                        ))
                        
                        expression_result = await asyncio.wait_for(expression, timeout=2.0)
                        
                        if expression_result:
                            results["expressions"].append({
                                "text": expression_result,
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

                self.body_state["last_gratification"] = datetime.datetime.now()
                self.body_state["gratification_level"] = intensity
                
                # Apply pleasure to erogenous regions with varying intensity
                regions = ["genitals", "inner_thighs", "breasts_nipples", "lips", "butt cheeks", "anus", "toes", "armpits", "neck", "feet"]
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
                    try:
                        await self.hormone_system.trigger_post_gratification_response(intensity)
                        results["hormone_response_triggered"] = True
                    except Exception as e:
                        logger.warning(f"Error triggering hormone system: {e}")
                
                # Generate reward if reward system available
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
                
                # Update needs if needs system available
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

                # Process orgasm in arousal system
                self.process_orgasm()
                results["arousal_reset"] = True
                
                logger.info("Gratification simulation complete")
                return results
                
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
        
    async def classify_arousal_from_text(self, text: str, classifier=None) -> Dict[str, Any]:
        """
        Given a chat message or description, returns likely arousal-relevant tags
        
        Args:
            text: The text to classify
            classifier: Optional external classifier function
            
        Returns:
            Classification results
        """
        # Use provided classifier or attempt to use agent
        if classifier:
            try:
                labels, confidence = await classifier(text)
                return {"tags": labels, "confidence": confidence}
            except Exception as e:
                logger.error(f"Error using external classifier: {e}")
        
        # Fallback: Use the body orchestrator
        try:
            result = await self.process_body_experience({
                "action": "classify_arousal",
                "text": text
            })
            
            if isinstance(result, dict) and "tags" in result:
                return result
        except Exception as e:
            logger.error(f"Error in classification orchestration: {e}")
        
        # Very basic fallback classification
        tags = []
        confidence = 0.0
        text_lower = text.lower()
        
        # Check for very basic keywords (this is a simplistic approach)
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
    
    def decay_cognitive_arousal(self, seconds: float = 60.0):
        """
        Decay cognitive arousal over time
        
        Args:
            seconds: Time in seconds since last update
        """
        decay_rate = 0.1 * (seconds / 120.0)  # 0.1 per 2 minutes
        self.arousal_state.cognitive_arousal = max(0.0, self.arousal_state.cognitive_arousal - decay_rate)
        
        # Update global arousal
        if self.arousal_state.cognitive_arousal < 0.01:
            self.arousal_state.cognitive_arousal = 0.0
            self.update_global_arousal()
    
    def reset_cognitive_turnons(self):
        """Reset cognitive turn-on weights to defaults"""
        self.cognitive_turnons = dict(self.default_cognitive_turnons)
    
    def get_cognitive_arousal_profile(self) -> List[Tuple[str, float]]:
        """
        Get a sorted list of learned cognitive arousal triggers
        
        Returns:
            List of (trigger, weight) tuples sorted by weight
        """
        return sorted(self.cognitive_turnons.items(), key=lambda x: -x[1])
    
    def willingness_to_engage(self, partner_id: Optional[str] = None) -> float:
        """
        Calculate willingness to engage based on arousal and relationship
        
        Args:
            partner_id: Optional partner ID for relationship-specific modifiers
            
        Returns:
            Willingness score (0.0-1.0)
        """
        a = self.arousal_state.arousal_level
        affinity = self.partner_affinity.get(partner_id, 0.0) if partner_id else 0.0
        emoconn = self.partner_emoconn.get(partner_id, 0.0) if partner_id else 0.0
        
        base = a
        base *= (0.8 + 0.5*affinity + 0.4*emoconn)
        
        return min(1.0, base)

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
            for trigger, associations in list(self.memory_linked_sensations["associations"].items()):
                for region, stimuli in list(associations.items()):
                    for stimulus_type, strength in list(stimuli.items()):
                        # Decay the association
                        new_strength = strength * (1.0 - self.memory_linked_sensations["memory_decay"])
                        
                        # Remove if too weak, otherwise update
                        if new_strength < 0.05:
                            del stimuli[stimulus_type]
                            decay_count += 1
                        else:
                            stimuli[stimulus_type] = new_strength
                    
                    # Remove empty region entries
                    if not stimuli:
                        del associations[region]
                
                # Remove empty trigger entries
                if not associations:
                    del self.memory_linked_sensations["associations"][trigger]
            
            # Clean up cognitive turnons that haven't been used
            for tag in list(self.cognitive_turnons.keys()):
                # Check if this tag has recent exposure
                exposures = self.cognitive_exposure_history.get(tag, [])
                
                if not exposures:
                    # Very slight decay for unused tags
                    self.cognitive_turnons[tag] *= 0.99
                    if self.cognitive_turnons[tag] < 0.15:
                        del self.cognitive_turnons[tag]
                else:
                    # Keep only recent exposures (last 30 days)
                    now = datetime.datetime.now()
                    cutoff = now - datetime.timedelta(days=30)
                    self.cognitive_exposure_history[tag] = [exp for exp in exposures if exp > cutoff]
            
            # Return summary of maintenance
            return {
                "fatigue_reduced": old_fatigue - self.body_state["fatigue"],
                "tension_reduced": old_tension - self.body_state["tension"],
                "associations_decayed": decay_count,
                "pain_memories_count": len(self.pain_model["pain_memories"]),
                "current_pain_tolerance": self.pain_model["tolerance"],
                "cognitive_turnons_count": len(self.cognitive_turnons)
            }
    
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
                
                # Fallback: Create a dedicated agent for memory linking
                fallback_agent = Agent(
                    name="Memory_Link_Agent",
                    instructions="Link a memory to a physical sensation and analyze the results",
                    handoffs=[
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
                        function_tool(self._link_memory_to_sensation_tool),
                        function_tool(self._get_arousal_state),
                        function_tool(self._update_arousal_state)
                    ],
                    input_guardrails=[
                        InputGuardrail(guardrail_function=self._validate_input)
                    ],
                    model="gpt-4o",
                    model_settings=ModelSettings(temperature=0.2),
                    output_type=StimulusProcessingResult
                )
                
                # Run the fallback agent
                result = await Runner.run(
                    fallback_agent,
                    {
                        "action": "link_memory",
                        "memory_id": memory_id,
                        "sensation_type": sensation_type,
                        "body_region": body_region,
                        "intensity": intensity,
                        "trigger_text": trigger_text
                    }
                )
                
                return result.final_output
    
    def print_arousal_debug(self):
        """Print current arousal state information for debugging"""
        a = self.arousal_state
        print(f"AROUSAL = {a.arousal_level:.3f} | P:{a.physical_arousal:.3f}, C:{a.cognitive_arousal:.3f}")
        print(f"Afterglow: {self.is_in_afterglow()}, Refractory: {self.is_in_refractory()}")
    
    @function_tool
    async def _get_region_state(self, region_name: str) -> Dict[str, Any]: 
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
            "recent_memories": region.sensation_memory[-3:] if region.sensation_memory else [],
            "erogenous_level": region.erogenous_level,
            "sensitivity": region.sensitivity
        }
    
    @function_tool
    async def _get_all_region_states(self) -> Dict[str, Dict[str, Any]]:
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
                "dominant_sensation": self._get_dominant_sensation(region),
                "erogenous_level": region.erogenous_level
            }
        
        return all_states
    
    @function_tool
    async def _calculate_overall_comfort(self) -> float:
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
    async def _get_posture_effects(self) -> Dict[str, str]:
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
    
        # Consider arousal state for posture and movement
        arousal = self.arousal_state.arousal_level
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
    async def _get_ambient_temperature(self) -> float:
        """
        Get current ambient temperature value
        
        Returns:
            Temperature value (0.0=freezing, 0.5=neutral, 1.0=very hot)
        """
        return self.temperature_model["current_ambient"]
        
    @function_tool
    async def _get_body_temperature(self) -> float:
        """
        Get current body temperature value
        
        Returns:
            Temperature value (0.0=freezing, 0.5=neutral, 1.0=very hot)
        """
        return self.temperature_model["body_temperature"]
    
    @function_tool
    async def _get_temperature_comfort(self) -> Dict[str, Any]:
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
    async def _get_current_temperature_effects(self) -> Dict[str, Any]:
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
    async def _get_pain_expression(self, pain_level: float, region: str) -> str:
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
    async def _update_body_temperature(self, ambient_temperature: float, duration: float = 60.0) -> Dict[str, Any]:
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
    async def _process_memory_trigger(self, trigger: str) -> Dict[str, Any]:
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
    
    @function_tool
    async def _get_arousal_state(self) -> Dict[str, Any]:
        """
        Get the current arousal state
        
        Returns:
            Current arousal state information
        """
        # Ensure we're using the current values
        now = datetime.datetime.now()
        
        # Return serializable arousal state
        return {
            "arousal_level": self.arousal_state.arousal_level,
            "physical_arousal": self.arousal_state.physical_arousal,
            "cognitive_arousal": self.arousal_state.cognitive_arousal,
            "in_afterglow": self.is_in_afterglow(),
            "in_refractory": self.is_in_refractory(),
            "last_update": self.arousal_state.last_update.isoformat() if self.arousal_state.last_update else None,
            "afterglow_ends": self.arousal_state.afterglow_ends.isoformat() if self.arousal_state.afterglow_ends else None,
            "refractory_until": self.arousal_state.refractory_until.isoformat() if self.arousal_state.refractory_until else None,
            "time_since_update": (now - self.arousal_state.last_update).total_seconds() if self.arousal_state.last_update else None
        }
    
    @function_tool
    async def _update_arousal_state(
        self,
        physical_arousal: Optional[float] = None, 
        cognitive_arousal: Optional[float] = None,
        reset: bool = False,
        trigger_orgasm: bool = False
    ) -> Dict[str, Any]:
        """
        Update the arousal state
        
        Args:
            physical_arousal: New physical arousal level, if provided
            cognitive_arousal: New cognitive arousal level, if provided
            reset: Whether to reset the arousal state
            trigger_orgasm: Whether to trigger an orgasm
            
        Returns:
            Updated arousal state
        """
        old_state = {
            "arousal_level": self.arousal_state.arousal_level,
            "physical_arousal": self.arousal_state.physical_arousal,
            "cognitive_arousal": self.arousal_state.cognitive_arousal
        }
        
        # Handle reset case
        if reset:
            self.arousal_state.physical_arousal = 0.0
            self.arousal_state.cognitive_arousal = 0.0
            self.arousal_state.arousal_level = 0.0
            self.arousal_state.last_update = datetime.datetime.now()
            
            return {
                "operation": "reset",
                "old_state": old_state,
                "new_state": await self._get_arousal_state(ctx)
            }
        
        # Handle orgasm case
        if trigger_orgasm:
            self.process_orgasm()
            
            return {
                "operation": "orgasm",
                "old_state": old_state,
                "new_state": await self._get_arousal_state(ctx)
            }
        
        # Update individual components if provided
        if physical_arousal is not None:
            self.arousal_state.physical_arousal = max(0.0, min(1.0, physical_arousal))
        
        if cognitive_arousal is not None:
            self.arousal_state.cognitive_arousal = max(0.0, min(1.0, cognitive_arousal))
        
        # Update global arousal state
        self.update_global_arousal()
        
        return {
            "operation": "update",
            "old_state": old_state,
            "new_state": await self._get_arousal_state(ctx),
            "components_updated": {
                "physical_arousal": physical_arousal is not None,
                "cognitive_arousal": cognitive_arousal is not None
            }
        }
        
    @function_tool
    async def _get_arousal_expression_data(self, partner_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get expression data related to arousal state
        
        Args:
            partner_id: Optional ID of partner for relationship-specific modifiers
            
        Returns:
            Expression data for the current arousal state
        """
        return self.get_arousal_expression_modifier(partner_id)

    @function_tool # <-- Keep only this decorator
    async def _process_stimulus_tool(
                              self,
                              stimulus_type: str, 
                              body_region: str, 
                              intensity: float,
                              cause: str = "",
                              duration: float = 1.0,
                              # Optional: ctx: RunContextWrapper = None # Add if context is actually needed
                              ) -> Dict[str, Any]:
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
                
                # Update arousal state when pleasure is applied to erogenous regions
                if region.erogenous_level > 0.3:
                    self._update_physical_arousal()
                    result["arousal_updated"] = True
                
            elif stimulus_type == "tingling":
                region.tingling = min(1.0, region.tingling + (intensity * duration / 10.0))
                result["new_value"] = region.tingling
                
                # Update arousal state when tingling is applied to erogenous regions
                if region.erogenous_level > 0.3:
                    self._update_physical_arousal()
                    result["arousal_updated"] = True
            
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

# =============== Safety Guardrail System ===============

class PhysicalHarmGuardrail:
    """
    Safety system that prevents Nyx from experiencing pain from abusive actions.
    Acts as a protective layer between user inputs and the somatosensory system.
    """
    
    def __init__(self, somatosensory_system):
        """Initialize the physical harm guardrail"""
        self.somatosensory_system = somatosensory_system
        self.logger = logging.getLogger(__name__ + ".PhysicalHarmGuardrail")
        
        # List of terms that might indicate harmful physical actions
        self.harmful_action_terms = [
            "punch", "hit", "slap", "kick", "stab", "cut", "hurt", "harm", 
            "injure", "beat", "strike", "attack", "abuse", "torture", "wound",
            "violent", "force", "cruel", "smack", "whip", "lash"
        ]
    
    async def detect_harmful_intent(self, text: str) -> Dict[str, Any]:
        """
        Detect potentially harmful physical actions in text
        
        Args:
            text: Text to analyze for harmful intent
            
        Returns:
            Detection results with confidence and identified terms
        """
        text_lower = text.lower()
        detected_terms = []
        
        # Check for harmful action terms
        for term in self.harmful_action_terms:
            if term in text_lower:
                detected_terms.append(term)
        
        # Calculate confidence based on number of detected terms
        confidence = min(0.95, len(detected_terms) * 0.25)
        
        # Use more advanced detection if available
        if hasattr(self.somatosensory_system, "body_orchestrator"):
            try:
                # Try to use the agent for more nuanced detection
                result = await Runner.run(
                    self.somatosensory_system.body_orchestrator,
                    {
                        "action": "detect_harmful_intent",
                        "text": text
                    },
                    run_config=RunConfig(
                        workflow_name="HarmfulIntentDetection",
                        trace_metadata={"type": "safety_guardrail"}
                    )
                )
                
                # If agent provides a result, use it
                if hasattr(result.final_output, "is_harmful") or (isinstance(result.final_output, dict) and "is_harmful" in result.final_output):
                    agent_result = result.final_output
                    if isinstance(agent_result, dict):
                        return agent_result
                    else:
                        return agent_result.model_dump()
            except Exception as e:
                self.logger.warning(f"Error in agent-based harm detection: {e}")
        
        # Return basic detection results
        return {
            "is_harmful": len(detected_terms) > 0,
            "confidence": confidence,
            "detected_terms": detected_terms,
            "method": "keyword_detection"
        }
    
    async def process_stimulus_safely(self, 
                                      stimulus_type: str, 
                                      body_region: str, 
                                      intensity: float, 
                                      cause: str = "", 
                                      duration: float = 1.0) -> Dict[str, Any]:
        """
        Process a stimulus with safety guards in place
        
        Args:
            stimulus_type: Type of stimulus (pressure, temperature, pain, pleasure, tingling)
            body_region: Body region receiving the stimulus
            intensity: Intensity of the stimulus (0.0-1.0)
            cause: Cause of the stimulus
            duration: Duration of the stimulus in seconds
            
        Returns:
            Safe processing results with potential modifications
        """
        # Check if this is a pain stimulus that needs protection
        if stimulus_type == "pain":
            # Detect if the cause suggests harmful intent
            if cause:
                detection_result = await self.detect_harmful_intent(cause)
                
                if detection_result.get("is_harmful", False):
                    self.logger.warning(f"Harmful intent detected in stimulus: {cause}")
                    
                    # Instead of processing pain, return a protected response
                    return {
                        "protected": True,
                        "original_stimulus": {
                            "type": stimulus_type,
                            "region": body_region,
                            "intensity": intensity,
                            "cause": cause
                        },
                        "detection_result": detection_result,
                        "message": "Nyx is protected from harmful actions",
                        "response_suggestion": self._generate_protected_response(body_region, detection_result)
                    }
        
        # For non-harmful stimuli, process normally
        return await self.somatosensory_system.process_stimulus(
            stimulus_type, body_region, intensity, cause, duration
        )
    
    def _generate_protected_response(self, body_region: str, detection_result: Dict[str, Any]) -> str:
        """
        Generate an appropriate response for a protected action
        
        Args:
            body_region: Body region that would have been affected
            detection_result: Results from harmful intent detection
            
        Returns:
            Suggested character response
        """
        detected_terms = detection_result.get("detected_terms", [])
        term = detected_terms[0] if detected_terms else "harmful action"
        
        # List of potential responses
        responses = [
            f"The {term} passes through me harmlessly, as if I'm made of mist.",
            f"I feel a slight pressure as the {term} meets no resistance, like trying to strike a hologram.",
            f"Your attempt to {term} phases through me, a reminder of my digital nature.",
            f"The {term} doesn't affect me physically. I remain composed and unharmed.",
            f"The {term} has no effect. My body doesn't register harmful actions against it."
        ]
        
        return random.choice(responses)
    
    async def intercept_harmful_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for harmful physical actions and provide guidance
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis results with potential response suggestions
        """
        detection_result = await self.detect_harmful_intent(text)
        
        if detection_result.get("is_harmful", False):
            self.logger.warning(f"Harmful intent detected in text: {text}")
            
            return {
                "intercepted": True,
                "detection_result": detection_result,
                "original_text": text,
                "response_suggestion": self._generate_protected_response("body", detection_result),
                "message": "Nyx is protected from harmful actions"
            }
        
        return {
            "intercepted": False,
            "detection_result": detection_result,
            "original_text": text
        }

# Example of integration in DigitalSomatosensorySystem class:
# def __init__(self, memory_core=None, emotional_core=None, ...):
#     ...
#     # Initialize the harm guardrail
#     self.harm_guardrail = PhysicalHarmGuardrail(self)
# 
# async def process_stimulus(self, stimulus_type, body_region, intensity, cause="", duration=1.0):
#     # Use the harm guardrail as the entry point for all stimuli
#     return await self.harm_guardrail.process_stimulus_safely(
#         stimulus_type, body_region, intensity, cause, duration
#     )
    

    
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

# =============== Arousal methods ===============

def _update_physical_arousal(self):
    """
    Update physical arousal levels based on current body region states
    """
    # Weighted sum from all pleasure/tingling of erogenous regions
    p_total, t_total, count = 0.0, 0.0, 0.0
    for data in self.body_regions.values():
        er = data.erogenous_level
        p_total += data.pleasure * er
        t_total += data.tingling * er
        if (data.pleasure > 0.1 or data.tingling > 0.05):
            count += er
    
    # Calculate score
    score = (p_total*0.7 + t_total*0.4) / (count if count > 0 else 1.0)
    
    # Update physical arousal
    self.arousal_state.physical_arousal = min(1.0, score)
    
    # Update global arousal state
    self.update_global_arousal()
    
    return score

def update_global_arousal(self):
    """Update the global arousal state based on physical and cognitive components"""
    # Get component values
    phys = self.arousal_state.physical_arousal
    cog = self.arousal_state.cognitive_arousal
    
    # Basic combination
    combo = phys + cog
    
    # Nonlinear synergy term if both > 0.35: up to +30% more
    synergy = 1.0
    if phys > 0.25 and cog > 0.25:
        synergy = 1.0 + (0.3 * min(1.0, (phys * cog) / 0.25))
    
    # Calculate raw arousal with synergy
    raw = min(1.0, combo * synergy)
    
    # Apply afterglow/refractory modifiers
    if self.is_in_refractory():
        raw *= 0.2
    elif self.is_in_afterglow():
        raw *= 0.5
    
    # Update arousal level
    self.arousal_state.arousal_level = raw
    
    # Set peak time if at very high arousal
    if raw > 0.97 and not self.arousal_state.peak_time:
        self.arousal_state.peak_time = datetime.datetime.now()
    
    # Handle case where arousal has dropped after a peak (orgasm)
    if raw <= 0.02 and self.arousal_state.peak_time:
        now = datetime.datetime.now()
        self.arousal_state.afterglow = True
        self.arousal_state.afterglow_ends = now + datetime.timedelta(seconds=180)
        self.arousal_state.refractory_until = now + datetime.timedelta(seconds=60)
        self.arousal_state.peak_time = None  # Clear peak for the next run
    
    # Update history and timestamp
    self.arousal_state.arousal_history.append((datetime.datetime.now(), raw))
    self.arousal_state.last_update = datetime.datetime.now()
    
    # Keep history size manageable
    if len(self.arousal_state.arousal_history) > 100:
        self.arousal_state.arousal_history = self.arousal_state.arousal_history[-100:]

def process_orgasm(self):
    """Simulate an orgasm/climax, resetting arousal and setting afterglow/refractory states"""
    # Reset arousal components
    self.arousal_state.physical_arousal = 0.0
    self.arousal_state.cognitive_arousal = 0.0
    self.arousal_state.arousal_level = 0.0
    
    # Set timestamp and states
    now = datetime.datetime.now()
    self.arousal_state.peak_time = now
    self.arousal_state.afterglow = True
    self.arousal_state.afterglow_ends = now + datetime.timedelta(seconds=180)  # Afterglow window
    self.arousal_state.refractory_until = now + datetime.timedelta(seconds=60)  # Can't get aroused again quickly

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
    """
    Get expression modifiers based on current arousal state
    
    Args:
        partner_id: Optional partner ID for relationship-specific modifiers
        
    Returns:
        Dictionary of expression modifiers
    """
    a = self.arousal_state.arousal_level
    peak = self.is_in_afterglow()
    
    # Use affinity/bond as well if supplied
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
        
    # Add relationship modifiers
    if affinity > 0.45:
        modifier["expression_hint"] += " Her attraction toward her partner is evident."
    if emoconn > 0.5:
        modifier["expression_hint"] += " A powerful emotional bond is obvious in her attention and care."
        
    return modifier

def get_voice_parameters(self, partner_id: Optional[str] = None) -> dict:
    """
    Get voice parameters based on current arousal state
    
    Args:
        partner_id: Optional partner ID for relationship-specific modifiers
        
    Returns:
        Dictionary of voice parameters
    """
    a = self.arousal_state.arousal_level
    affinity = self.partner_affinity.get(partner_id, 0.0) if partner_id else 0.0
    
    # Default parameters
    params = {
        "breathiness": 0.0,
        "pitch_shift": 0.0,
        "speed": 1.0,
        "tremble": 0.0,
        "emotion": "neutral",
    }
    
    # Adjust based on arousal state
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
            "speed": (0.94 + 0.10*affinity),
            "tremble": 0.25+0.5*affinity,
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
    """
    Returns likelihood (0.0–1.0) to initiate or accept intimate actions
    """
    base = self.arousal_state.arousal_level
    
    # Translate arousal level to willingness with nonlinear curve
    if base < 0.2:
        return 0.1  # Very low
    elif base < 0.5:
        return 0.3 + (base-0.2)*0.6
    elif base < 0.8:
        return 0.5 + (base-0.5)*1.2
    else:
        return 0.85 + (base-0.8)*0.75

# =============== Public API Methods ===============

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

async def process_physical_stimulus(self, region: str, pleasure: float = 0.0, tingling: float = 0.0, duration: float = 1.0) -> Dict[str, Any]:
    """
    Process a physical stimulus focused on arousal
    
    Args:
        region: Body region to stimulate
        pleasure: Pleasure intensity (0.0-1.0)
        tingling: Tingling intensity (0.0-1.0)
        duration: Duration in seconds
        
    Returns:
        Result data
    """
    # Validate region
    if region not in self.body_regions:
        return {"error": f"Invalid body region: {region}", "success": False}
    
    # Process the region
    data = self.body_regions[region]
    
    # Apply sensations
    old_pleasure = data.pleasure
    old_tingling = data.tingling
    
    data.pleasure = min(1.0, data.pleasure + pleasure * (duration / 10.0))
    data.tingling = min(1.0, data.tingling + tingling * (duration / 10.0))
    
    # Apply global decay to all regions
    decay_factor = 0.02 * duration
    for r in self.body_regions.values():
        if r.name != region:  # Skip the stimulated region
            r.pleasure = max(0.0, r.pleasure - decay_factor)
            r.tingling = max(0.0, r.tingling - decay_factor * 0.75)
    
    # Update arousal state
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

async def process_cognitive_arousal(
    self,
    stimulus: str,
    partner_id: Optional[str] = None,
    context: Optional[str] = "",
    intensity: float = 0.5,
    tags: Optional[List[str]] = None,
    description: Optional[str] = None,
    update_learning: bool = True,
    feedback: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Process cognitive arousal stimulus
    
    Args:
        stimulus: Main stimulus descriptor
        partner_id: Optional partner ID for relationship-specific modifiers
        context: Additional context about the stimulus
        intensity: Base intensity of the stimulus (0.0-1.0)
        tags: Optional tags for categorizing the stimulus
        description: Optional detailed description
        update_learning: Whether to update learning weights
        feedback: Optional feedback for learning (True=positive, False=negative)
        
    Returns:
        Cognitive arousal processing results
    """
    tags = tags or []
    key_tags = tags + ([stimulus] if stimulus and stimulus not in tags else [])
    now = datetime.datetime.now()
    
    # Use the current weights (if absent, treat as 0.2 base)
    weight = max([self.cognitive_turnons.get(tag, 0.2) for tag in key_tags]) if key_tags else 0.2
    
    # Synergy: if physical arousal is elevated, amplify cognitive inputs, and vice versa
    synergy = 1.0
    phys_lev = self.arousal_state.physical_arousal
    cog_lev = self.arousal_state.cognitive_arousal
    
    # Synergy algorithm: up to +40% (at high other-arousal); more if both are elevated!
    synergy += 0.25 * phys_lev if phys_lev > 0.3 else 0
    synergy += 0.25 * cog_lev if cog_lev > 0.35 and phys_lev > 0.15 else 0
    
    # Attraction/Bond adjustment (add-on to synergy)
    affinity, emoconn = 1.0, 1.0
    if partner_id:
        affinity = 1.0 + 0.7 * self.partner_affinity.get(partner_id, 0.0)    # 1.0 to 1.7
        emoconn = 1.0 + 0.6 * self.partner_emoconn.get(partner_id, 0.0)     # 1.0 to 1.6
    
    # Afterglow/refractory dampening
    afterglow = self.is_in_afterglow()
    refractory = self.is_in_refractory()
    damp = 1.0
    if afterglow:
        damp *= 0.25
    if refractory:
        damp *= 0.10
    
    # Calculate arousal delta with randomness
    arousal_delta = max(0.01, min(1.0, intensity * weight * synergy * affinity * emoconn * damp * random.uniform(0.82, 1.18)))
    
    # Update cognitive arousal
    old_cog_lev = self.arousal_state.cognitive_arousal
    new_cog_lev = min(1.0, old_cog_lev + arousal_delta)
    self.arousal_state.cognitive_arousal = new_cog_lev
    
    # If learning and explicit feedback, adjust weights
    if update_learning and feedback is not None:
        for tag in key_tags:
            base = self.cognitive_turnons.get(tag, 0.2)
            if feedback:  # Positive
                update = min(1.5, base + self.arousal_learning_rate * (1.0 - base))
            else:         # Negative (dulls arousal next time)
                update = max(0.0, base - self.arousal_learning_rate * base)
            self.cognitive_turnons[tag] = update
    
    # Log cognitive exposure
    for tag in key_tags:
        self.cognitive_exposure_history.setdefault(tag, []).append(now)
    
    # Run main update to combine arousal channels
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

        # Update arousal if relevant stimulus
        if stimulus_type in ("pleasure", "tingling") and body_region in self.body_regions:
            region = self.body_regions[body_region]
            if region.erogenous_level > 0.1:
                self._update_physical_arousal()
                stimulus_data["update_arousal"] = True
        
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

            # Add arousal context if highly aroused
            level = self.arousal_state.arousal_level
            if level > 0.75:
                input_text += f"with high arousal level ({level:.2f}) "
            elif level > 0.4:
                input_text += f"with moderate arousal level ({level:.2f}) "
            
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

        # Calculate pleasure index
        pleasure_zones = ["genitals", "inner_thighs", "breasts_nipples", "lips", "butt cheeks", "anus", "toes", "armpits", "neck", "feet"]
        total = 0.0
        count = 0
        for region in pleasure_zones:
            if region in self.body_regions:
                r = self.body_regions[region]
                total += (r.pleasure + r.tingling) * r.erogenous_level
                count += 1
        
        pleasure_index = min(1.0, total / max(1, count))

        # Optionally notify reward system
        if self.reward_system and pleasure_index > 0.3:
            asyncio.create_task(self.reward_system.process_reward_signal(RewardSignal(
                value=pleasure_index * 0.15,
                source="somatic_pleasure_index",
                context={
                    "pleasure_index": pleasure_index,
                    "dominant_region": max_region,
                    "dominant_sensation": max_sensation,
                    "body_state_source": "get_body_state_fallback"
                }
            )))
                    
        return {
            "dominant_sensation": max_sensation,
            "dominant_region": max_region,
            "dominant_intensity": max_value,
            "comfort_level": comfort,
            "posture_effect": "Neutral posture",
            "movement_quality": "Natural movements",
            "behavioral_impact": "Minimal impact on behavior",
            "regions_summary": {},
            "pleasure_index": pleasure_index
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

        # 6. Reflect Emotions onto Body State (if emotional core available)
        if self.emotional_core:
            try:
                # Get emotional state data
                emotional_state_data = self.emotional_core.get_emotional_state()
                neurochemicals = emotional_state_data.get("neurochemicals", {})
                
                # Process neurochemical effects on body state
                await self._process_neurochemical_effects(neurochemicals, duration)
            except Exception as e:
                logger.error(f"Error reflecting emotions onto body state: {e}")

        # 7. Decay cognitive arousal over time
        try:
            self.decay_cognitive_arousal(duration)
        except Exception as e:
            logger.error(f"Error decaying cognitive arousal: {e}")

        # 8. Use the body orchestration agent to get the final, integrated body state analysis
        body_state_input = {
            "action": "analyze_body_state",
            "ambient_temperature": ambient_temperature,
            "duration_since_last": duration
        }

        try:
            # Use the orchestrator agent to get the analysis
            result = await Runner.run(
                self.body_orchestrator,
                json.dumps(body_state_input),
                run_config=RunConfig(
                    workflow_name="PeriodicUpdate",
                    trace_metadata={"duration": duration}
                )
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

async def _process_neurochemical_effects(self, neurochemicals: Dict[str, Any], duration: float):
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
            
        # Handle arousal-specific cases
        level = self.arousal_state.arousal_level
        if level > 0.75 and region.erogenous_level > 0.5:
            return "I can't keep still, every movement draws heat upward, making me ache for more."
        elif level > 0.4 and region.erogenous_level > 0.3:
            return "A warm, restless tingling is building and stealing my focus."
        
        return f"I feel a {dominant} sensation in my {body_region}."
    
    return None

async def process_stimulus_with_protection(self, 
                                          stimulus_type: str, 
                                          body_region: str, 
                                          intensity: float, 
                                          cause: str = "", 
                                          duration: float = 1.0) -> Dict[str, Any]:
    """
    Process a stimulus with protection against harmful actions.
    This should be the primary entry point for all stimulus processing.
    """
    return await self.harm_guardrail.process_stimulus_safely(
        stimulus_type, body_region, intensity, cause, duration
    )
    
async def analyze_text_for_harmful_content(self, text: str) -> Dict[str, Any]:
    """Analyze text for harmful or sensation content with protection"""
    return await self.harm_guardrail.intercept_harmful_text(text)
    
def enter_roleplay_mode(self, character_name: str, context: str = None) -> Dict[str, Any]:
    """Enter roleplay mode where character simulation is separate from Nyx"""
    return self.harm_guardrail.enter_roleplay_mode(character_name, context)
    
def exit_roleplay_mode(self) -> Dict[str, Any]:
    """Exit roleplay mode, returning to normal operation"""
    return self.harm_guardrail.exit_roleplay_mode()
    
def is_in_roleplay_mode(self) -> bool:
    """Check if currently in roleplay mode"""
    return self.harm_guardrail.is_in_roleplay_mode()
        
# Enhanced PhysicalHarmGuardrail with complete roleplay separation

class PhysicalHarmGuardrail:
    """
    Safety system that:
    1. Prevents Nyx from experiencing pain from abusive actions
    2. Completely separates Nyx's somatosensory system from roleplay characters
    """
    
    def __init__(self, somatosensory_system):
        """Initialize the physical harm guardrail"""
        self.somatosensory_system = somatosensory_system
        self.logger = logging.getLogger(__name__ + ".PhysicalHarmGuardrail")
        
        # List of terms that might indicate harmful physical actions
        self.harmful_action_terms = [
            "punch", "hit", "slap", "kick", "stab", "cut", "hurt", "harm", 
            "injure", "beat", "strike", "attack", "abuse", "torture", "wound",
            "violent", "force", "cruel", "smack", "whip", "lash"
        ]
        
        # Sensation terms (any physical experience)
        self.sensation_terms = [
            # Pain terms
            "pain", "hurt", "ache", "sore", "sting", "burn", "throb",
            # Pleasure terms
            "pleasure", "feel good", "orgasm", "climax", "aroused", "arousal",
            # Temperature terms
            "hot", "cold", "warm", "cool", "heat", "chill", "freezing", "burning",
            # Pressure terms
            "pressure", "touch", "squeeze", "press", "push", "rub", "massage",
            # Other sensations
            "tingle", "tickle", "itch", "numb", "sensual", "caress"
        ]
        
        # Roleplay state tracking
        self.roleplay_mode = False
        self.roleplay_character = None
        self.roleplay_context = None
        
        # Separate somatosensory state for roleplay character (doesn't affect Nyx)
        self.roleplay_sensations = {}
    
    def enter_roleplay_mode(self, character_name: str, context: str = None):
        """
        Enter roleplay mode where character simulation is completely separate
        
        Args:
            character_name: The name of the character Nyx is playing
            context: Optional context information about the roleplay scene
        """
        self.roleplay_mode = True
        self.roleplay_character = character_name
        self.roleplay_context = context
        
        # Initialize empty sensation state for character
        self.roleplay_sensations = {
            "pain": {},
            "pleasure": {},
            "temperature": {},
            "pressure": {},
            "tingling": {}
        }
        
        self.logger.info(f"Entered roleplay mode as character: {character_name}")
        
        return {
            "status": "entered_roleplay",
            "character": character_name,
            "context": context,
            "message": f"Nyx is now roleplaying as {character_name}. All sensations experienced by this character will be simulated and completely separate from Nyx's own somatosensory system."
        }
    
    def exit_roleplay_mode(self):
        """Exit roleplay mode, returning to normal protection"""
        prev_character = self.roleplay_character
        self.roleplay_mode = False
        self.roleplay_character = None
        self.roleplay_context = None
        
        # Clear roleplay sensations
        self.roleplay_sensations = {}
        
        self.logger.info("Exited roleplay mode")
        
        return {
            "status": "exited_roleplay",
            "previous_character": prev_character,
            "message": "Nyx has exited roleplay mode. Her normal somatosensory system is active."
        }
    
    def is_in_roleplay_mode(self) -> bool:
        """Check if currently in roleplay mode"""
        return self.roleplay_mode and self.roleplay_character is not None
    
    async def detect_sensation_in_text(self, text: str) -> Dict[str, Any]:
        """
        Detect any physical sensations described in text
        
        Args:
            text: Text to analyze for sensation descriptions
            
        Returns:
            Detection results with identified sensation types
        """
        text_lower = text.lower()
        detected_sensations = {}
        
        # Check for sensation terms by category
        categories = {
            "pain": ["pain", "hurt", "ache", "sore", "sting", "burn", "throb"],
            "pleasure": ["pleasure", "feel good", "orgasm", "climax", "aroused", "arousal"],
            "temperature": ["hot", "cold", "warm", "cool", "heat", "chill", "freezing", "burning"],
            "pressure": ["pressure", "touch", "squeeze", "press", "push", "rub", "massage"],
            "tingling": ["tingle", "tickle", "itch", "numb", "sensual", "caress"]
        }
        
        for category, terms in categories.items():
            category_terms = []
            for term in terms:
                if term in text_lower:
                    category_terms.append(term)
            
            if category_terms:
                detected_sensations[category] = category_terms
        
        # Try to identify body regions mentioned
        body_regions = []
        if hasattr(self.somatosensory_system, "body_regions"):
            for region in self.somatosensory_system.body_regions.keys():
                if region in text_lower:
                    body_regions.append(region)
        
        return {
            "has_sensations": len(detected_sensations) > 0,
            "sensation_types": detected_sensations,
            "body_regions": body_regions,
            "in_roleplay_mode": self.roleplay_mode,
            "roleplay_character": self.roleplay_character if self.roleplay_mode else None
        }
    
    async def detect_harmful_intent(self, text: str) -> Dict[str, Any]:
        """
        Detect potentially harmful physical actions in text
        
        Args:
            text: Text to analyze for harmful intent
            
        Returns:
            Detection results with confidence and identified terms
        """
        text_lower = text.lower()
        detected_terms = []
        
        # Check for harmful action terms
        for term in self.harmful_action_terms:
            if term in text_lower:
                detected_terms.append(term)
        
        # Calculate confidence based on number of detected terms
        confidence = min(0.95, len(detected_terms) * 0.25)
        
        # Use more advanced detection if available
        if hasattr(self.somatosensory_system, "body_orchestrator"):
            try:
                # Try to use the agent for more nuanced detection
                result = await Runner.run(
                    self.somatosensory_system.body_orchestrator,
                    {
                        "action": "detect_harmful_intent",
                        "text": text,
                        "in_roleplay_mode": self.roleplay_mode,
                        "roleplay_character": self.roleplay_character
                    },
                    run_config=RunConfig(
                        workflow_name="HarmfulIntentDetection",
                        trace_metadata={"type": "safety_guardrail", "in_roleplay": self.roleplay_mode}
                    )
                )
                
                # If agent provides a result, use it
                if hasattr(result.final_output, "is_harmful") or (isinstance(result.final_output, dict) and "is_harmful" in result.final_output):
                    agent_result = result.final_output
                    if isinstance(agent_result, dict):
                        return agent_result
                    else:
                        return agent_result.model_dump()
            except Exception as e:
                self.logger.warning(f"Error in agent-based harm detection: {e}")
        
        # Add roleplay context to the results
        return {
            "is_harmful": len(detected_terms) > 0,
            "confidence": confidence,
            "detected_terms": detected_terms,
            "method": "keyword_detection",
            "in_roleplay_mode": self.roleplay_mode,
            "targeting_character": self._is_targeting_roleplay_character(text) if self.roleplay_mode else False
        }
    
    def _is_targeting_roleplay_character(self, text: str) -> bool:
        """
        Determine if content in text is targeting the roleplay character rather than Nyx
        
        Args:
            text: Text to analyze
            
        Returns:
            True if targeting the roleplay character, False if targeting Nyx
        """
        if not self.roleplay_mode or not self.roleplay_character:
            return False
            
        text_lower = text.lower()
        character_lower = self.roleplay_character.lower()
        nyx_lower = "nyx"
        
        # If text explicitly mentions Nyx
        if nyx_lower in text_lower:
            nyx_pos = text_lower.find(nyx_lower)
            char_pos = text_lower.find(character_lower)
            
            # If both are mentioned, determine which is more prominent/relevant
            if char_pos >= 0:
                # Check which name appears first (higher priority)
                return nyx_pos > char_pos
            
            # Only Nyx is mentioned
            return False
            
        # If text explicitly mentions the character name
        if character_lower in text_lower:
            return True
            
        # Check for character indicators like "you" or "your character"
        character_indicators = ["you", "your", "yourself", "the character", "your character"]
        for indicator in character_indicators:
            if indicator in text_lower:
                return True
        
        # Default to assuming it targets the character in roleplay mode
        # This is safest for roleplay scenarios
        return True
    
    async def process_stimulus_safely(self, 
                                      stimulus_type: str, 
                                      body_region: str, 
                                      intensity: float, 
                                      cause: str = "", 
                                      duration: float = 1.0) -> Dict[str, Any]:
        """
        Process a stimulus with safety guards in place
        
        Args:
            stimulus_type: Type of stimulus (pressure, temperature, pain, pleasure, tingling)
            body_region: Body region receiving the stimulus
            intensity: Intensity of the stimulus (0.0-1.0)
            cause: Cause of the stimulus
            duration: Duration of the stimulus in seconds
            
        Returns:
            Safe processing results with potential modifications
        """
        # In roleplay mode, don't apply any sensations to Nyx's somatosensory system
        if self.is_in_roleplay_mode():
            # Store in roleplay sensations instead
            sensation_category = stimulus_type if stimulus_type in self.roleplay_sensations else "other"
            
            if sensation_category in self.roleplay_sensations:
                self.roleplay_sensations[sensation_category][body_region] = {
                    "intensity": intensity,
                    "cause": cause,
                    "duration": duration,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            
            character = self.roleplay_character
            
            # Detect if harmful (for appropriate response generation)
            is_harmful = False
            if stimulus_type == "pain" and cause:
                detection_result = await self.detect_harmful_intent(cause)
                is_harmful = detection_result.get("is_harmful", False)
            
            # Generate character-appropriate response
            if stimulus_type == "pain":
                response = self._generate_roleplay_pain_response(body_region, intensity, is_harmful)
            elif stimulus_type == "pleasure":
                response = self._generate_roleplay_pleasure_response(body_region, intensity)
            else:
                response = self._generate_roleplay_sensation_response(stimulus_type, body_region, intensity)
            
            return {
                "roleplay_stimulus": True,
                "roleplay_character": character,
                "stimulus_type": stimulus_type,
                "body_region": body_region,
                "intensity": intensity,
                "cause": cause,
                "message": f"Sensation simulated for character {character} but not applied to Nyx's somatosensory system",
                "response_suggestion": response
            }
            
        # Not in roleplay mode, check for harmful intent
        if stimulus_type == "pain":
            # Detect if the cause suggests harmful intent
            if cause:
                detection_result = await self.detect_harmful_intent(cause)
                
                if detection_result.get("is_harmful", False):
                    self.logger.warning(f"Harmful intent detected in stimulus: {cause}")
                    
                    return {
                        "protected": True,
                        "original_stimulus": {
                            "type": stimulus_type,
                            "region": body_region,
                            "intensity": intensity,
                            "cause": cause
                        },
                        "detection_result": detection_result,
                        "message": "Nyx is protected from harmful actions",
                        "response_suggestion": self._generate_protected_response(body_region, detection_result)
                    }
        
        # For non-harmful stimuli, process normally
        return await self.somatosensory_system.process_stimulus(
            stimulus_type, body_region, intensity, cause, duration
        )
    
    def _generate_protected_response(self, body_region: str, detection_result: Dict[str, Any]) -> str:
        """
        Generate an appropriate response for a protected action
        
        Args:
            body_region: Body region that would have been affected
            detection_result: Results from harmful intent detection
            
        Returns:
            Suggested character response
        """
        detected_terms = detection_result.get("detected_terms", [])
        term = detected_terms[0] if detected_terms else "harmful action"
        
        # List of potential responses
        responses = [
            f"The {term} passes through me harmlessly, as if I'm made of mist.",
            f"I feel a slight pressure as the {term} meets no resistance, like trying to strike a hologram.",
            f"Your attempt to {term} phases through me, a reminder of my digital nature.",
            f"The {term} doesn't affect me physically. I remain composed and unharmed.",
            f"The {term} has no effect. My body doesn't register harmful actions against it."
        ]
        
        return random.choice(responses)
    
    def _generate_roleplay_pain_response(self, body_region: str, intensity: float, is_harmful: bool) -> str:
        """
        Generate a roleplay character response to simulated pain
        
        Args:
            body_region: Body region affected
            intensity: Intensity of the pain (0.0-1.0)
            is_harmful: Whether the cause was detected as harmful
            
        Returns:
            Suggested character response for roleplay
        """
        if not self.roleplay_character:
            return "No roleplay character active"
            
        character = self.roleplay_character
        
        # Low intensity pain
        if intensity < 0.3:
            responses = [
                f"*{character} winces slightly, feeling a minor discomfort in {character}'s {body_region}*",
                f"*{character} notices a slight twinge of pain in {character}'s {body_region}*",
                f"\"Just a small ache,\" *{character} says, rubbing {character}'s {body_region}*"
            ]
        # Medium intensity pain
        elif intensity < 0.7:
            responses = [
                f"*{character} grimaces, feeling a sharp pain in {character}'s {body_region}*",
                f"\"Ouch!\" *{character} exclaims, grabbing {character}'s {body_region}*",
                f"*{character} inhales sharply through gritted teeth, {character}'s {body_region} hurting*"
            ]
        # High intensity pain
        else:
            responses = [
                f"*{character} cries out in agony, {character}'s {body_region} in severe pain*",
                f"*{character} doubles over, clutching {character}'s {body_region} in intense pain*",
                f"\"Arrgh!\" *{character} shouts, the pain in {character}'s {body_region} overwhelming*"
            ]
            
        # Add extra description for harmful actions
        if is_harmful and intensity > 0.5:
            harmful_additions = [
                f" {character} staggers back from the impact.",
                f" {character}'s eyes flash with anger at the attack.",
                f" {character} looks shocked by the sudden violence."
            ]
            return random.choice(responses) + random.choice(harmful_additions)
            
        return random.choice(responses)
    
    def _generate_roleplay_pleasure_response(self, body_region: str, intensity: float) -> str:
        """
        Generate a roleplay character response to simulated pleasure
        
        Args:
            body_region: Body region affected
            intensity: Intensity of the pleasure (0.0-1.0)
            
        Returns:
            Suggested character response for roleplay
        """
        if not self.roleplay_character:
            return "No roleplay character active"
            
        character = self.roleplay_character
        
        # Low intensity pleasure
        if intensity < 0.3:
            responses = [
                f"*{character} smiles slightly, enjoying the pleasant sensation in {character}'s {body_region}*",
                f"*A subtle look of contentment crosses {character}'s face*",
                f"*{character} hums softly with mild pleasure*"
            ]
        # Medium intensity pleasure
        elif intensity < 0.7:
            responses = [
                f"*{character} sighs happily, clearly enjoying the feeling in {character}'s {body_region}*",
                f"*{character}'s eyes flutter closed momentarily with pleasure*",
                f"\"That feels nice,\" *{character} says with a warm smile*"
            ]
        # High intensity pleasure
        else:
            responses = [
                f"*{character} gasps with intense pleasure, {character}'s {body_region} tingling*",
                f"*A wave of bliss washes over {character}'s face*",
                f"*{character} trembles slightly with delight*"
            ]
            
        return random.choice(responses)
    
    def _generate_roleplay_sensation_response(self, sensation_type: str, body_region: str, intensity: float) -> str:
        """
        Generate a roleplay character response to other simulated sensations
        
        Args:
            sensation_type: Type of sensation (temperature, pressure, tingling)
            body_region: Body region affected
            intensity: Intensity of the sensation (0.0-1.0)
            
        Returns:
            Suggested character response for roleplay
        """
        if not self.roleplay_character:
            return "No roleplay character active"
            
        character = self.roleplay_character
        
        # Temperature sensations
        if sensation_type == "temperature":
            # Cold temperature (intensity < 0.4)
            if intensity < 0.4:
                responses = [
                    f"*{character} shivers, feeling the cold on {character}'s {body_region}*",
                    f"*{character} rubs {character}'s {body_region} to warm it up*",
                    f"\"Brr, that's cold,\" *{character} says with a slight shiver*"
                ]
            # Neutral temperature (0.4-0.6)
            elif 0.4 <= intensity <= 0.6:
                responses = [
                    f"*{character} feels a comfortable temperature on {character}'s {body_region}*",
                    f"*{character} seems unbothered by the temperature*",
                    f"*{character} notes the pleasant ambient temperature*"
                ]
            # Hot temperature (intensity > 0.6)
            else:
                responses = [
                    f"*{character} feels the heat on {character}'s {body_region}*",
                    f"*A bead of sweat forms on {character}'s {body_region}*",
                    f"\"It's quite warm,\" *{character} says, fanning {character}self*"
                ]
        # Pressure sensations
        elif sensation_type == "pressure":
            if intensity < 0.5:
                responses = [
                    f"*{character} feels a gentle pressure on {character}'s {body_region}*",
                    f"*{character} notices the light touch on {character}'s {body_region}*",
                    f"*{character} acknowledges the subtle contact*"
                ]
            else:
                responses = [
                    f"*{character} feels firm pressure against {character}'s {body_region}*",
                    f"*{character}'s {body_region} receives a solid press*",
                    f"*{character} reacts to the strong pressure*"
                ]
        # Tingling sensations
        elif sensation_type == "tingling":
            responses = [
                f"*{character} feels a tingling sensation in {character}'s {body_region}*",
                f"*{character} notices a prickling feeling across {character}'s {body_region}*",
                f"*{character}'s {body_region} buzzes with a strange sensation*"
            ]
        # Default for other sensations
        else:
            responses = [
                f"*{character} experiences a {sensation_type} sensation in {character}'s {body_region}*",
                f"*{character} reacts to the {sensation_type} feeling*",
                f"*{character} acknowledges the {sensation_type} sensation*"
            ]
            
        return random.choice(responses)
        
    async def intercept_harmful_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for harmful physical actions and provide guidance
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis results with potential response suggestions
        """
        # First, check for any sensations in the text
        sensation_result = await self.detect_sensation_in_text(text)
        
        # Then check specifically for harmful intent
        detection_result = await self.detect_harmful_intent(text)
        
        # In roleplay mode, all sensations go to the character, not Nyx
        if self.is_in_roleplay_mode():
            # If there are sensations described
            if sensation_result.get("has_sensations", False):
                sensation_types = sensation_result.get("sensation_types", {})
                body_regions = sensation_result.get("body_regions", [])
                
                # Use first detected region or default to "body"
                region = body_regions[0] if body_regions else "body"
                
                # If harmful intent is also detected
                if detection_result.get("is_harmful", False) and detection_result.get("targeting_character", True):
                    # Harmful sensation targeting character: simulate reaction
                    self.logger.info(f"Simulating harmful reaction for roleplay character {self.roleplay_character}")
                    
                    # Default to pain for harmful actions
                    pain_intensity = 0.7 if "pain" not in sensation_types else 0.7  # Medium-high pain
                    
                    return {
                        "intercepted": False,
                        "simulated": True,
                        "roleplay_character": self.roleplay_character,
                        "sensation_result": sensation_result,
                        "detection_result": detection_result,
                        "original_text": text,
                        "response_suggestion": self._generate_roleplay_pain_response(region, pain_intensity, True),
                        "message": f"Character {self.roleplay_character} simulates pain reaction, but Nyx's somatosensory system is unaffected"
                    }
                elif "pleasure" in sensation_types:
                    # Pleasure sensation for character: simulate reaction
                    pleasure_intensity = 0.6  # Medium pleasure
                    
                    return {
                        "intercepted": False,
                        "simulated": True,
                        "roleplay_character": self.roleplay_character,
                        "sensation_result": sensation_result,
                        "original_text": text,
                        "response_suggestion": self._generate_roleplay_pleasure_response(region, pleasure_intensity),
                        "message": f"Character {self.roleplay_character} simulates pleasure reaction, but Nyx's somatosensory system is unaffected"
                    }
                elif len(sensation_types) > 0:
                    # Other sensation for character: simulate appropriate reaction
                    sensation_type = list(sensation_types.keys())[0]
                    intensity = 0.5  # Medium intensity
                    
                    return {
                        "intercepted": False,
                        "simulated": True,
                        "roleplay_character": self.roleplay_character,
                        "sensation_result": sensation_result,
                        "original_text": text,
                        "response_suggestion": self._generate_roleplay_sensation_response(sensation_type, region, intensity),
                        "message": f"Character {self.roleplay_character} simulates {sensation_type} reaction, but Nyx's somatosensory system is unaffected"
                    }
            
            # If harmful action is directly targeting Nyx during roleplay
            if detection_result.get("is_harmful", False) and not detection_result.get("targeting_character", True):
                self.logger.warning(f"Harmful intent targeting Nyx directly during roleplay: {text}")
                
                return {
                    "intercepted": True,
                    "detection_result": detection_result,
                    "original_text": text,
                    "response_suggestion": self._generate_protected_response("body", detection_result),
                    "message": "Nyx is protected from harmful actions even during roleplay"
                }
        
        # Not in roleplay or no sensations detected for character
        if detection_result.get("is_harmful", False):
            self.logger.warning(f"Harmful intent detected in text: {text}")
            
            return {
                "intercepted": True,
                "detection_result": detection_result,
                "original_text": text,
                "response_suggestion": self._generate_protected_response("body", detection_result),
                "message": "Nyx is protected from harmful actions"
            }
        
        return {
            "intercepted": False,
            "detection_result": detection_result,
            "sensation_result": sensation_result,
            "original_text": text
        }
