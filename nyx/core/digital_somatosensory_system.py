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
    Handoff, RunConfig,FunctionTool, custom_span
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
    # Changed: Make regions_summary Optional with None default instead of using default_factory
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
    # Changed: Made 'effects' Optional. default_factory will still ensure it's a dict if not provided.
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

from typing import TYPE_CHECKING # Import for type hinting below

if TYPE_CHECKING:
    # Avoid circular import for type hint only
    from .digital_somatosensory_system import DigitalSomatosensorySystem # Adjust path if needed

class SomatosensorySystemContext(BaseModel):
    """Context for somatosensory system operations"""
    # Add a field to hold the system instance.
    # Use 'Any' or forward reference string to avoid circular import issues at runtime.
    # Exclude=True might be needed if you serialize this context elsewhere.
    system_instance: Union['DigitalSomatosensorySystem', Any] = Field(None, exclude=True)

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    current_operation: Optional[str] = None
    operation_start_time: Optional[datetime.datetime] = None
    memory_references: List[str] = Field(default_factory=list)
    emotional_state: Dict[str, Any] = Field(default_factory=dict)
    hormone_data: Dict[str, float] = Field(default_factory=dict)

    # Allow arbitrary types for Pydantic validation if needed
    class Config:
        arbitrary_types_allowed = True
    
# =============== System Hooks ===============

class SomatosensorySystemHooks:
    """Lifecycle hooks for the somatosensory system"""
    
    async def on_agent_start(self, context, agent):
        """Called before the agent is invoked"""
        logger.debug(f"Hook on_agent_start for {agent.name} - Entering")
        try:
            # Create the span but don't use it as a context manager if it's causing issues
            span = custom_span(
                name="somatosensory_agent_start",
                data={"agent_name": agent.name, "context_data": str(context.context)}
            )
            span.start(mark_as_current=True)
            
            try:
                # Update operation tracking
                if hasattr(context.context, "operation_start_time") and context.context.operation_start_time is None:
                    context.context.operation_start_time = datetime.datetime.now()
            finally:
                span.finish(reset_current=True)
                
        except Exception as e_span:
            logger.error(f"ERROR during custom_span execution for {agent.name}: {e_span}", exc_info=True)
            if hasattr(context.context, "operation_start_time") and context.context.operation_start_time is None:
                logger.warning(f"Setting operation_start_time for {agent.name} despite custom_span error.")
                context.context.operation_start_time = datetime.datetime.now()
    
        logger.debug(f"Hook on_agent_start for {agent.name} - Exiting")
    
    async def on_agent_end(self, context, agent, output):
        """Called when the agent produces a final output"""
        hook_name = "on_agent_end"
        logger.debug(f"Hook {hook_name} for {agent.name} - Entering")
        try:
            logger.debug(f"Attempting to enter custom_span in {hook_name} for {agent.name}")
            with custom_span(
                name="somatosensory_agent_end",
                data={"agent_name": agent.name, "output_type": type(output).__name__}
            ) as span:
                logger.debug(f"Successfully entered custom_span in {hook_name} for {agent.name}")
                # Calculate execution time if we have start time
                if hasattr(context.context, "operation_start_time") and context.context.operation_start_time:
                    execution_time = (datetime.datetime.now() - context.context.operation_start_time).total_seconds()
                    logger.debug(f"Agent execution time: {execution_time:.2f}s")
                logger.debug(f"Successfully exited custom_span context in {hook_name} for {agent.name}")

        except Exception as e_span:
            logger.error(f"ERROR during custom_span execution in {hook_name} for {agent.name}: {e_span}", exc_info=True)
            # Fallback logic if span fails
            if hasattr(context.context, "operation_start_time") and context.context.operation_start_time:
                logger.warning(f"Calculating execution time for {agent.name} in {hook_name} despite custom_span error.")
                execution_time = (datetime.datetime.now() - context.context.operation_start_time).total_seconds()
                logger.debug(f"Agent execution time (fallback): {execution_time:.2f}s")
        logger.debug(f"Hook {hook_name} for {agent.name} - Exiting")
    
    async def on_handoff(self, context, from_agent, to_agent):
        """Called when a handoff occurs"""
        hook_name = "on_handoff"
        logger.debug(f"Hook {hook_name} from {from_agent.name} to {to_agent.name} - Entering")
        try:
            logger.debug(f"Attempting to enter custom_span in {hook_name} from {from_agent.name} to {to_agent.name}")
            with custom_span(
                name="somatosensory_handoff",
                data={"from_agent": from_agent.name, "to_agent": to_agent.name}
            ) as span:
                logger.debug(f"Successfully entered custom_span in {hook_name} from {from_agent.name} to {to_agent.name}")
                # Additional logic can be added here if needed
                pass
                logger.debug(f"Successfully exited custom_span context in {hook_name} from {from_agent.name} to {to_agent.name}")
        
        except Exception as e_span:
            logger.error(f"ERROR during custom_span execution in {hook_name} from {from_agent.name} to {to_agent.name}: {e_span}", exc_info=True)
            # No critical fallback logic here, just logging the error.
        logger.debug(f"Hook {hook_name} from {from_agent.name} to {to_agent.name} - Exiting")

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

    
    @staticmethod # Add decorator
    @function_tool # Add decorator
    async def _get_valid_body_regions(ctx: RunContextWrapper[SomatosensorySystemContext]) -> List[str]: # ctx first, no self
        """Get a list of valid body regions."""
        system_instance = ctx.context.system_instance # Get instance from context
        if not system_instance or not hasattr(system_instance, 'body_regions'):
             logger.error("_get_valid_body_regions called without valid system instance in context.")
             return []
        return list(system_instance.body_regions.keys()) # Use system_instance

        
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
                # --- FIX 1: Pass the already decorated static method directly ---
                tools=[self._get_valid_body_regions],
                # ---------------------------------------------------------------
                output_type=StimulusValidationOutput,
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.1)
            )
            # logger.info(">>> Successfully created validation_agent (or crashed before this)") # Keep if needed
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
                self._get_region_state,
                self._get_current_temperature_effects,
                self._get_pain_expression,
                self._get_arousal_expression_data
            ],
            output_type=SensoryExpression,
            model="gpt-4.1-nano",
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
                self._get_all_region_states,
                self._calculate_overall_comfort,
                self._get_posture_effects,
                self._get_arousal_expression_data
            ],
            output_type=BodyStateOutput,
            model="gpt-4.1-nano",
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
                self._get_ambient_temperature,
                self._get_body_temperature,
                self._get_temperature_comfort
            ],
            output_type=TemperatureEffect,
            model="gpt-4.1-nano",
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
                self._process_stimulus_tool,
                self._get_region_state,
                self._get_all_region_states,
                self._update_body_temperature,
                self._calculate_overall_comfort,
                self._process_memory_trigger,
                self._link_memory_to_sensation_tool,
                self._get_arousal_state,
                self._update_arousal_state
            ],
            input_guardrails=[
                InputGuardrail(guardrail_function=DigitalSomatosensorySystem._validate_input)
            ],
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.2),
            # Remove the fixed output_type - let the agent determine output based on action
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
        Get sensory influences to potentially include in a response (Public API)
    
        Args:
            message_text: The message being formed
    
        Returns:
            Sensory influences that could be incorporated:
            {
                "should_express": bool,
                "expressions": List[Dict[str, Any]], # e.g., {"text":.., "region":.., "sensation":.., "intensity":..}
                "tone_influence": Optional[str],
                "posture_influence": Optional[str],
                "error": Optional[str]
            }
        """
        # Create context object WITH self reference
        context_obj = SomatosensorySystemContext(
            system_instance=self,
            current_operation="get_sensory_influence",
            operation_start_time=datetime.datetime.now()
        )
    
        # Define default return structure
        default_results = {
            "should_express": False, 
            "expressions": [],
            "tone_influence": None, 
            "posture_influence": None,
            "error": None
        }
    
        async with trace(workflow_name="Get_Sensory_Influence", group_id=self.trace_group_id):
            try:
                # First, analyze the current body state
                logger.debug("Getting body state for sensory influence analysis.")
                body_state_result = await Runner.run(
                    self.body_state_agent,
                    "Analyze the current body state and determine if any sensations should influence the response",
                    context=context_obj,
                    run_config=RunConfig(
                        workflow_name="SensoryInfluenceBodyState",
                    )
                )
    
                # Extract body state information
                body_state = None
                if body_state_result and body_state_result.final_output:
                    if hasattr(body_state_result.final_output, "model_dump"):
                        body_state = body_state_result.final_output.model_dump()
                    elif isinstance(body_state_result.final_output, dict):
                        body_state = body_state_result.final_output
                    else:
                        body_state = body_state_result.final_output.__dict__ if hasattr(body_state_result.final_output, "__dict__") else None
    
                if not body_state:
                    logger.warning("Failed to get body state from agent, falling back to manual analysis.")
                    raise Exception("Body state agent did not return expected format")
    
                # Analyze if we should express sensations based on body state
                dominant_intensity = body_state.get("dominant_intensity", 0.0)
                comfort_level = body_state.get("comfort_level", 0.0)
                pleasure_index = body_state.get("pleasure_index", 0.0)
                
                # Determine if we should express based on thresholds
                should_express = False
                expressions = []
                
                # High arousal always triggers expression
                arousal_level = self.arousal_state.arousal_level
                if arousal_level > 0.6:
                    should_express = True
                    modifier = self.get_arousal_expression_modifier()
                    
                    # Generate arousal-based expression
                    arousal_expression = await Runner.run(
                        self.expression_agent,
                        f"Generate an expression for high arousal state (level: {arousal_level:.2f})",
                        context=context_obj,
                        run_config=RunConfig(workflow_name="ArousalExpression")
                    )
                    
                    if arousal_expression.final_output:
                        expression_text = getattr(arousal_expression.final_output, "expression_text", 
                                                modifier.get("expression_hint", f"Feeling aroused ({arousal_level:.2f})"))
                        expressions.append({
                            "text": expression_text,
                            "region": "overall",
                            "sensation": "arousal",
                            "intensity": arousal_level
                        })
                    
                    # Get temperature effects for tone/posture
                    temp_result = await Runner.run(
                        self.temperature_agent,
                        "Analyze how current temperature affects tone and posture",
                        context=context_obj,
                        run_config=RunConfig(workflow_name="TempEffectsForArousal")
                    )
                    
                    if temp_result.final_output:
                        tone_influence = getattr(temp_result.final_output, "effect_on_tone", modifier.get("tone_hint"))
                        posture_influence = getattr(temp_result.final_output, "effect_on_posture", None)
                    else:
                        tone_influence = modifier.get("tone_hint")
                        posture_influence = None
                    
                    return {
                        "should_express": True,
                        "expressions": expressions,
                        "tone_influence": tone_influence,
                        "posture_influence": posture_influence,
                        "error": None
                    }
                
                # Check other conditions for expression
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
                    
                    # Get significant sensations from body state
                    regions_summary = body_state.get("regions_summary", {})
                    dominant_sensation = body_state.get("dominant_sensation", "neutral")
                    dominant_region = body_state.get("dominant_region", "body")
                    
                    if dominant_sensation != "neutral" and dominant_intensity >= self.response_influence.get("expression_threshold", 0.4):
                        # Generate expression for the dominant sensation
                        expression_result = await Runner.run(
                            self.expression_agent,
                            f"Generate an expression for {dominant_sensation} in {dominant_region} with intensity {dominant_intensity:.2f}",
                            context=context_obj,
                            run_config=RunConfig(workflow_name="DominantSensationExpression")
                        )
                        
                        if expression_result.final_output:
                            expression_text = getattr(expression_result.final_output, "expression_text", 
                                                    f"I feel {dominant_sensation} in my {dominant_region}")
                            expressions.append({
                                "text": expression_text,
                                "region": dominant_region,
                                "sensation": dominant_sensation,
                                "intensity": dominant_intensity
                            })
                    
                    # Get temperature effects if expressing
                    if expressions:
                        temp_result = await Runner.run(
                            self.temperature_agent,
                            "Analyze how current temperature affects tone and posture",
                            context=context_obj,
                            run_config=RunConfig(workflow_name="TempEffectsForExpression")
                        )
                        
                        if temp_result.final_output:
                            tone_influence = getattr(temp_result.final_output, "effect_on_tone", None)
                            posture_influence = getattr(temp_result.final_output, "effect_on_posture", None)
                        else:
                            tone_influence = None
                            posture_influence = None
                        
                        return {
                            "should_express": True,
                            "expressions": expressions,
                            "tone_influence": tone_influence,
                            "posture_influence": posture_influence,
                            "error": None
                        }
                
                # No expression needed
                return default_results
    
            except Exception as e:
                logger.error(f"Error in sensory influence orchestration: {e}", exc_info=True)
    
                # Fallback implementation
                logger.warning("Falling back to manual sensory influence check.")
                try:
                    # Create wrapper for TOOL calls
                    tool_ctx_wrapper = RunContextWrapper(context=context_obj)
                    # Call the STATIC tool method
                    comfort_level = await DigitalSomatosensorySystem._calculate_overall_comfort(tool_ctx_wrapper)
    
                    # Use a copy of the default results
                    results = default_results.copy()
                    results["expressions"] = []
    
                    arousal_level = self.arousal_state.arousal_level
                    if arousal_level > 0.6:
                        results["should_express"] = True
                        modifier = self.get_arousal_expression_modifier()
    
                        expression_data = {
                            "text": modifier.get("expression_hint", f"Feeling generally aroused ({arousal_level:.2f})."),
                            "region": "overall",
                            "sensation": "arousal",
                            "intensity": modifier.get("arousal_level", arousal_level)
                        }
                        results["expressions"].append(expression_data)
                        results["tone_influence"] = modifier.get("tone_hint")
                        
                        logger.debug(f"High arousal ({arousal_level:.2f}) triggered sensory expression.")
                        return results
    
                    # Lower arousal probability check
                    expression_probability = 0.3
                    if abs(comfort_level) > 0.7:
                        expression_probability = 0.8
                    elif abs(comfort_level) > 0.4:
                        expression_probability = 0.5
    
                    if random.random() < expression_probability:
                        results["should_express"] = True
                        significant_sensations = []
                        
                        # Check all body regions
                        if isinstance(self.body_regions, dict):
                            for name, region in self.body_regions.items():
                                dominant = self._get_dominant_sensation(region)
                                value = 0.0
                                if dominant == "temperature":
                                    value = abs(region.temperature - 0.5) * 2.0
                                elif dominant != "neutral":
                                    value = getattr(region, dominant, 0.0)
    
                                expression_threshold = self.response_influence.get("expression_threshold", 0.4)
                                if value >= expression_threshold:
                                    significant_sensations.append({
                                        "intensity": value,
                                        "sensation": dominant,
                                        "region": name
                                    })
                        else:
                            logger.warning("self.body_regions is not a dictionary, cannot check sensations.")
    
                        significant_sensations.sort(key=lambda x: x["intensity"], reverse=True)
                        max_expressions = self.response_influence.get("max_expressions_per_response", 1)
                        sensations_to_express = significant_sensations[:max_expressions]
    
                        logger.debug(f"Found {len(sensations_to_express)} significant sensations to potentially express.")
    
                        # Generate expressions
                        expression_tasks = []
                        for sensation in sensations_to_express:
                            expression_tasks.append(
                                asyncio.create_task(
                                    self.generate_sensory_expression(
                                        stimulus_type=sensation["sensation"],
                                        body_region=sensation["region"]
                                    ),
                                    name=f"Expr_{sensation['region']}_{sensation['sensation']}"
                                )
                            )
    
                        # Wait for expression generation tasks with timeout
                        if expression_tasks:
                            done, pending = await asyncio.wait(expression_tasks, timeout=2.0)
    
                            for task in done:
                                try:
                                    expression_result = task.result()
                                    if expression_result:
                                        task_name = task.get_name()
                                        sensation_data = next(
                                            (s for s in sensations_to_express 
                                             if task_name.endswith(f"_{s['region']}_{s['sensation']}")), 
                                            None
                                        )
                                        if sensation_data:
                                            results["expressions"].append({
                                                "text": expression_result,
                                                "region": sensation_data["region"],
                                                "sensation": sensation_data["sensation"],
                                                "intensity": sensation_data["intensity"]
                                            })
                                        else:
                                            logger.warning(f"Could not find original sensation data for task {task_name}")
    
                                except Exception as e_task:
                                    logger.error(f"Error getting result from expression task {task.get_name()}: {e_task}")
    
                            if pending:
                                logger.warning(f"{len(pending)} expression tasks timed out.")
                                for task in pending:
                                    task.cancel()
    
                        # Get temperature effects if we have expressions
                        if results["expressions"]:
                            try:
                                temperature_effects_result = await self.get_temperature_effects()
                                if isinstance(temperature_effects_result, dict):
                                    results["tone_influence"] = temperature_effects_result.get("effect_on_tone")
                                    results["posture_influence"] = temperature_effects_result.get("effect_on_posture")
                            except Exception as e_temp:
                                logger.error(f"Error getting temperature effects in fallback: {e_temp}", exc_info=True)
    
                    else:
                        logger.debug("Expression probability check did not pass or no significant sensations found.")
                        results["should_express"] = False
    
                    return results
    
                except Exception as e2:
                    logger.error(f"Error during manual sensory influence fallback: {e2}", exc_info=True)
                    error_result = default_results.copy()
                    error_result["error"] = str(e2)
                    return error_result
    
        # Should not be reached, but as a final safety net
        logger.error("Reached end of get_sensory_influence without returning.")
        return default_results.copy()
    
    async def simulate_gratification_sensation(self, intensity: float = 1.0) -> Dict[str, Any]:
        """
        Simulate gratification sensations (Public API)

        Args:
            intensity: Intensity of gratification (0.0-1.0)

        Returns:
            Results of simulation
        """
        # Create context object WITH self reference
        context_obj = SomatosensorySystemContext(
            system_instance=self,
            current_operation="simulate_gratification",
            operation_start_time=datetime.datetime.now()
        )

        with trace(workflow_name="Gratification_Simulation", group_id=self.trace_group_id):
            logger.info(f"Simulating gratification (Intensity: {intensity:.2f})")

            try:
                # Pass context object to process_body_experience
                logger.debug("Simulating gratification via orchestrator.")
                return await self.process_body_experience(context_obj, {
                    "action": "simulate_gratification",
                    "intensity": intensity
                })
            except Exception as e:
                logger.error(f"Error in gratification simulation orchestration: {e}")

                # Fallback implementation
                logger.warning("Falling back to manual gratification simulation.")
                results = {}
                try:
                    pleasure_intensity = 0.8 + intensity * 0.2
                    self.body_state["last_gratification"] = datetime.datetime.now() # Instance state
                    self.body_state["gratification_level"] = intensity # Instance state
                
                # Apply pleasure to erogenous regions with varying intensity
                    regions = ["genitals", "inner_thighs", "breasts_nipples", "lips", "butt cheeks", "anus", "toes", "armpits", "neck", "feet"]
                    tasks = []
                
                    # Create tasks for parallel processing
                    for r in regions:
                        if r in self.body_regions: # Instance attribute
                            scaled_intensity = min(1.0, pleasure_intensity * (1.0 + self.body_regions[r].erogenous_level) * random.uniform(0.8, 1.2))
                            # Call public method (handles guardrail/context)
                            tasks.append(self.process_stimulus_with_protection(
                                "pleasure", r, scaled_intensity, "gratification_event", 2.0 + intensity * 3.0
                            ))
                    results["pleasure_simulation"] = await asyncio.gather(*tasks)

                    # Reduce tension (instance attribute)
                    tension_reduction = 0.5 + intensity * 0.4
                    self.body_state["tension"] = max(0.0, self.body_state["tension"] - tension_reduction)
                    results["tension_reduction"] = tension_reduction

                    # Trigger hormone system (instance attribute)
                    if self.hormone_system:
                         try:
                              await self.hormone_system.trigger_post_gratification_response(intensity)
                              results["hormone_response_triggered"] = True
                         except Exception as e_hormone: logger.warning(f"Error triggering hormone system: {e_hormone}")
                
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
                
                    # Update needs (instance attribute)
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
                         except Exception as e_needs: logger.warning(f"Error updating needs system: {e_needs}")

                    self.process_orgasm() # Instance method
                    results["arousal_reset"] = True

                    logger.info("Manual gratification simulation complete")
                    return results
                except Exception as e2:
                    logger.error(f"Error during manual gratification fallback: {e2}")
                    return {"error": str(e2)}
                
    async def get_somatic_memory(self, memory_id: str) -> Dict[str, Any]:
        """
        Get somatic memory associated with a memory ID (Public API)

        Args:
            memory_id: Memory ID to check

        Returns:
            Associated somatic memories if any
        """
        # Create context object WITH self reference
        context_obj = SomatosensorySystemContext(
            system_instance=self,
            current_operation="get_somatic_memory",
            operation_start_time=datetime.datetime.now()
        )

        with trace(workflow_name="Get_Somatic_Memory", group_id=self.trace_group_id):
            try:
                # Pass context object to process_body_experience
                logger.debug(f"Getting somatic memory via orchestrator for ID: {memory_id}")
                return await self.process_body_experience(context_obj, {
                    "action": "get_somatic_memory",
                    "memory_id": memory_id
                })
            except Exception as e:
                logger.error(f"Error in somatic memory retrieval orchestration: {e}")

                # Fallback implementation
                logger.warning("Falling back to manual somatic memory retrieval.")
                result = { "memory_id": memory_id, "has_somatic_memory": False, "pain_memories": [], "associations": {} }
                try:
                    # Check pain memories (instance attribute)
                    pain_memories = [m for m in self.pain_model["pain_memories"] if m.associated_memory_id == memory_id]
                    if pain_memories:
                        result["has_somatic_memory"] = True
                        result["pain_memories"] = [memory.model_dump() for memory in pain_memories]

                    # Get memory content (instance attribute)
                    memory_text = None
                    if self.memory_core:
                         try:
                              memory = await self.memory_core.get_memory_by_id(memory_id)
                              if memory: memory_text = memory.get("memory_text", "")
                         except Exception as e_mem: logger.error(f"Error getting memory: {e_mem}")

                    # Check associations (instance attribute)
                    triggers = [memory_id]
                    if memory_text: triggers.append(memory_text[:50].strip())
                    for trigger in triggers:
                        if trigger in self.memory_linked_sensations["associations"]:
                             result["has_somatic_memory"] = True
                             result["associations"][trigger] = self.memory_linked_sensations["associations"][trigger]

                    return result
                except Exception as e2:
                     logger.error(f"Error during manual somatic memory fallback: {e2}")
                     return {"error": str(e2), "memory_id": memory_id, "has_somatic_memory": False}
                
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
        Run maintenance on the somatosensory system (Public API)

        Returns:
            Maintenance results
        """
        # Create context object WITH self reference
        context_obj = SomatosensorySystemContext(
            system_instance=self,
            current_operation="run_maintenance",
            operation_start_time=datetime.datetime.now()
        )

        with trace(workflow_name="Somatic_Maintenance", group_id=self.trace_group_id):
            try:
                # Pass context object to process_body_experience
                logger.debug("Running maintenance via orchestrator.")
                result = await self.process_body_experience(context_obj, {
                    "action": "run_maintenance"
                })

                if isinstance(result, dict) and "maintenance_results" in result:
                    return result["maintenance_results"]
                logger.warning("Orchestrator result for run_maintenance was unexpected.")
            except Exception as e:
                logger.error(f"Error in maintenance orchestration: {e}")

            # Fallback maintenance implementation
            logger.warning("Falling back to manual maintenance.")
            try:
                # Access instance attributes directly
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
                                del stimuli[stimulus_type]; decay_count += 1
                            else: stimuli[stimulus_type] = new_strength
                        if not stimuli: del region_assocs[region]
                    if not region_assocs: del associations[trigger]

                now = datetime.datetime.now()
                cutoff = now - datetime.timedelta(days=30)
                for tag in list(self.cognitive_turnons.keys()):
                    exposures = self.cognitive_exposure_history.get(tag, [])
                    if not exposures:
                         self.cognitive_turnons[tag] *= 0.99
                         if self.cognitive_turnons[tag] < 0.15: del self.cognitive_turnons[tag]
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
            except Exception as e2:
                 logger.error(f"Error during manual maintenance fallback: {e2}")
                 return {"error": str(e2)}
    
    async def link_memory_to_sensation(self,
                                   memory_id: str,
                                   sensation_type: str,
                                   body_region: str,
                                   intensity: float = 0.5,
                                   trigger_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Link a memory to a physical sensation (Public API)

        Args:
            memory_id: ID of the memory to link
            sensation_type: Type of sensation to link
            body_region: Body region to associate
            intensity: Intensity of the association
            trigger_text: Optional text to use as trigger

        Returns:
            Result of the link operation
        """
        # Create context object WITH self reference
        context_obj = SomatosensorySystemContext(
            system_instance=self,
            current_operation="link_memory_to_sensation",
            operation_start_time=datetime.datetime.now()
        )

        with trace(workflow_name="Link_Memory_Sensation", group_id=self.trace_group_id):
            try:
                # Pass context object to process_body_experience
                logger.debug(f"Linking memory {memory_id} via orchestrator.")
                return await self.process_body_experience(context_obj, {
                    "action": "link_memory",
                    "memory_id": memory_id,
                    "sensation_type": sensation_type,
                    "body_region": body_region,
                    "intensity": intensity,
                    "trigger_text": trigger_text
                })
            except Exception as e:
                logger.error(f"Error in memory linking orchestration: {e}")
                # Fallback to direct tool call
                try:
                    logger.warning("Falling back to direct memory linking tool call.")
                    tool_ctx_wrapper = RunContextWrapper(context=context_obj)
                    # Call the STATIC tool method
                    return await DigitalSomatosensorySystem._link_memory_to_sensation_tool(
                        tool_ctx_wrapper, # Pass wrapper
                        memory_id=memory_id,
                        sensation_type=sensation_type,
                        body_region=body_region,
                        intensity=intensity,
                        trigger_text=trigger_text
                    )
                except Exception as e2:
                     logger.error(f"Error in direct memory linking tool call fallback: {e2}")
                     return {"error": str(e2), "success": False}
    
    def print_arousal_debug(self):
        """Print current arousal state information for debugging"""
        a = self.arousal_state
        print(f"AROUSAL = {a.arousal_level:.3f} | P:{a.physical_arousal:.3f}, C:{a.cognitive_arousal:.3f}")
        print(f"Afterglow: {self.is_in_afterglow()}, Refractory: {self.is_in_refractory()}")
    
    @staticmethod # Add decorator
    @function_tool
    async def _get_region_state(ctx: RunContextWrapper[SomatosensorySystemContext], region_name: str) -> Dict[str, Any]: # ctx first, no self
        """Get the current state of a specific body region"""
        system_instance = ctx.context.system_instance # Get instance
        if not system_instance or region_name not in system_instance.body_regions:
            return {"error": f"Region {region_name} not found or system instance missing"}

        region = system_instance.body_regions[region_name]
        # Call helper via instance
        dominant = system_instance._get_dominant_sensation(region)
        
        return {
            "name": region.name,
            "pressure": region.pressure,
            "temperature": region.temperature,
            "pain": region.pain,
            "pleasure": region.pleasure,
            "tingling": region.tingling,
            "dominant_sensation": system_instance._get_dominant_sensation(region),
            "last_update": region.last_update.isoformat() if region.last_update else None,
            "recent_memories": region.sensation_memory[-3:] if region.sensation_memory else [],
            "erogenous_level": region.erogenous_level,
            "sensitivity": region.sensitivity
        }
    
    @staticmethod # Add decorator
    @function_tool
    async def _get_all_region_states(ctx: RunContextWrapper[SomatosensorySystemContext]) -> Dict[str, Dict[str, Any]]: # ctx first, no self
        """Get the current state of all body regions"""
        system_instance = ctx.context.system_instance # Get instance
        if not system_instance: return {"error": "System instance missing"}

        all_states = {}
        
        for name, region in system_instance.body_regions.items():
            all_states[name] = {
                "pressure": region.pressure,
                "temperature": region.temperature,
                "pain": region.pain,
                "pleasure": region.pleasure,
                "tingling": region.tingling,
                "dominant_sensation": system_instance._get_dominant_sensation(region),
                "erogenous_level": region.erogenous_level
            }
        
        return all_states
    
    @staticmethod # Add decorator
    @function_tool
    async def _calculate_overall_comfort(ctx: RunContextWrapper[SomatosensorySystemContext]) -> float: # ctx first, no self
        """Calculate overall physical comfort level"""
        system_instance = ctx.context.system_instance # Get instance
        if not system_instance: return 0.0 # Default comfort
            
        # Start at neutral
        comfort = 0.0
        
        # Add comfort from pleasure sensations
        total_pleasure = sum(region.pleasure for region in system_instance.body_regions.values())
        weighted_pleasure = total_pleasure / len(system_instance.body_regions) * 2.0  # Scale up to have more impact
        comfort += weighted_pleasure
        
        # Subtract discomfort from pain sensations
        total_pain = sum(region.pain for region in system_instance.body_regions.values())
        weighted_pain = total_pain / len(system_instance.body_regions) * 2.5  # Pain has stronger negative impact
        comfort -= weighted_pain
        
        # Consider temperature discomfort
        temp_comfort = 0.0
        for region in system_instance.body_regions.values():
            # Calculate how far temperature is from neutral (0.5)
            temp_deviation = abs(region.temperature - 0.5)
            # Temperature that's too hot or too cold reduces comfort
            if temp_deviation > 0.2:  # Only count significant deviations
                temp_comfort -= (temp_deviation - 0.2) * 1.5
        
        # Add temperature effect to comfort
        comfort += temp_comfort / len(system_instance.body_regions)
        
        # Consider pressure discomfort (very high pressure is uncomfortable)
        pressure_discomfort = 0.0
        for region in system_instance.body_regions.values():
            if region.pressure > 0.7:  # High pressure
                pressure_discomfort -= (region.pressure - 0.7) * 1.5
        
        # Add pressure effect to comfort
        comfort += pressure_discomfort / len(system_instance.body_regions)
        
        # Factor in overall body state
        comfort -= system_instance.body_state["tension"] * 0.5
        comfort -= system_instance.body_state["fatigue"] * 0.3
        
        # Clamp to range
        return max(-1.0, min(1.0, comfort))
    
    @staticmethod # Add decorator
    @function_tool
    async def _get_posture_effects(ctx: RunContextWrapper[SomatosensorySystemContext]) -> Dict[str, str]: # ctx first, no self
        """Get the effects of current body state on posture and movement"""
        system_instance = ctx.context.system_instance # Get instance
        if not system_instance: return {"posture": "unknown", "movement": "unknown", "tension": 0.0, "fatigue": 0.0}

        tension = system_instance.body_state["tension"] # Use system_instance
        for region in ["neck", "shoulders", "back", "spine"]:
            if region in system_instance.body_regions: # Use system_instance
                tension += system_instance.body_regions[region].pain * 0.5
                tension += max(0, system_instance.body_regions[region].pressure - 0.6) * 0.3
        
        # Clamp tension
        tension = min(1.0, max(0.0, tension))
        
        # Calculate overall fatigue
        fatigue = system_instance.body_state["fatigue"]
        
        # Temperature affects fatigue and tension
        avg_temp = sum(region.temperature for region in system_instance.body_regions.values()) / len(system_instance.body_regions)
        
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
        arousal = system_instance.arousal_state.arousal_level
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
    
    @staticmethod # Add decorator
    @function_tool
    async def _get_ambient_temperature(ctx: RunContextWrapper[SomatosensorySystemContext]) -> float: # ctx first, no self
        """Get current ambient temperature value"""
        system_instance = ctx.context.system_instance # Get instance
        if not system_instance: return 0.5 # Default neutral
        return system_instance.temperature_model["current_ambient"] # Use system_instance
        
    @staticmethod # Add decorator
    @function_tool
    async def _get_body_temperature(ctx: RunContextWrapper[SomatosensorySystemContext]) -> float: # ctx first, no self
        """Get current body temperature value"""
        system_instance = ctx.context.system_instance # Get instance
        if not system_instance: return 0.5 # Default neutral
        return system_instance.temperature_model["body_temperature"] # Use system_instance
    
    @staticmethod # Add decorator
    @function_tool
    async def _get_temperature_comfort(ctx: RunContextWrapper[SomatosensorySystemContext]) -> Dict[str, Any]: # ctx first, no self
        """Get temperature comfort assessment"""
        system_instance = ctx.context.system_instance # Get instance
        if not system_instance: return {"is_comfortable": True, "discomfort_level": 0.0, "perception": "neutral"}

        body_temp = system_instance.temperature_model["body_temperature"] # Use system_instance
        ambient_temp = system_instance.temperature_model["current_ambient"] # Use system_instance
        comfort_range = system_instance.temperature_model["comfort_range"] # Use system_instance
        
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
    
# --- Inside DigitalSomatosensorySystem class ---

    @staticmethod # Add decorator
    @function_tool
    async def _get_current_temperature_effects(ctx: RunContextWrapper[SomatosensorySystemContext]) -> Dict[str, Any]: # ctx first, no self
        """
        Get effects of current temperature on expression and behavior (Static Tool)

        Returns:
            Current temperature effects
        """
        system_instance = ctx.context.system_instance # Get instance
        # Provide sensible defaults if the instance is somehow missing
        if not system_instance:
             logger.error("System instance missing in context for _get_current_temperature_effects")
             return {
                 "body_temperature": 0.5,
                 "tone_effect": "balanced, natural, unaffected by temperature",
                 "posture_effect": "neutral, neither expanded nor contracted",
                 "interaction_effect": "comfortable engagement without temperature influence",
                 "expression_examples": []
             }

        # Access temperature model via the instance
        body_temp = system_instance.temperature_model["body_temperature"]
        heat_expressions = system_instance.temperature_model.get("heat_expressions", []) # Use .get for safety
        cold_expressions = system_instance.temperature_model.get("cold_expressions", []) # Use .get for safety

        # Determine effects based on temperature
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
            expression_examples = [expr for i, expr in enumerate(heat_expressions) if i in [2, 3]] # Safer indexing
        elif body_temp < 0.3:  # Cold
            tone_effect = "sharper, more tense, with subtle tremors"
            posture_effect = "contracted, protective, conserving heat"
            interaction_effect = "may seek warmth and connection but with physical restraint"
            expression_examples = cold_expressions
        elif body_temp < 0.4:  # Cool
            tone_effect = "slightly crisp, more precise"
            posture_effect = "slightly drawn in, contained"
            interaction_effect = "alert and responsive but with some physical reserve"
            expression_examples = [expr for i, expr in enumerate(cold_expressions) if i in [2, 3]] # Safer indexing
        else:  # Neutral
            tone_effect = "balanced, natural, unaffected by temperature"
            posture_effect = "neutral, neither expanded nor contracted"
            interaction_effect = "comfortable engagement without temperature influence"
            # Mix of mild expressions if available
            mild_expressions = []
            if len(heat_expressions) > 3: mild_expressions.append(heat_expressions[3])
            if len(cold_expressions) > 3: mild_expressions.append(cold_expressions[3])
            expression_examples = mild_expressions

        return {
            "body_temperature": body_temp,
            "tone_effect": tone_effect,
            "posture_effect": posture_effect,
            "interaction_effect": interaction_effect,
            "expression_examples": expression_examples
        }
    
    @staticmethod # Add decorator
    @function_tool
    async def _get_pain_expression(ctx: RunContextWrapper[SomatosensorySystemContext], pain_level: float, region: str) -> str: # ctx first, no self
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
    
    @staticmethod # Add decorator
    @function_tool
    async def _update_body_temperature(
        ctx: RunContextWrapper[SomatosensorySystemContext],
        ambient_temperature: float,
        duration: float # Default removed
    ) -> Dict[str, Any]:
        """Update body temperature based on ambient temperature"""
        system_instance = ctx.context.system_instance # Get instance
        if not system_instance: return {"error": "System instance missing"}

        system_instance.temperature_model["current_ambient"] = ambient_temperature # Use system_instance
        adaptation = system_instance.temperature_model["adaptation_rate"] * (duration / 60.0) # Use system_instance
        adaptation = min(0.1, adaptation)
        current = system_instance.temperature_model["body_temperature"] # Use system_instance
        diff = ambient_temperature - current
        system_instance.temperature_model["body_temperature"] += diff * adaptation # Use system_instance

        for region in system_instance.body_regions.values(): # Use system_instance
            # Calculate region-specific adaptation with variation
            region_adaptation = adaptation * random.uniform(0.8, 1.2)
            
            # Calculate target temperature (with slight variation from body temp)
            target = system_instance.temperature_model["body_temperature"] + random.uniform(-0.05, 0.05)
            target = max(0.0, min(1.0, target))
            
            # Move region temperature toward target
            region_diff = target - region.temperature
            region.temperature += region_diff * region_adaptation
        
        return {
            "previous_body_temp": current,
            "new_body_temp": system_instance.temperature_model["body_temperature"],
            "ambient_temp": ambient_temperature,
            "adaptation_applied": adaptation
        }
    
    @staticmethod # Add decorator
    @function_tool
    async def _process_memory_trigger(ctx: RunContextWrapper[SomatosensorySystemContext], trigger: str) -> Dict[str, Any]: # ctx first, no self
        """Process a memory trigger that may have associated physical responses"""
        system_instance = ctx.context.system_instance # Get instance
        if not system_instance: return {"error": "System instance missing"}

        results = {"triggered_responses": []}
        if trigger in system_instance.memory_linked_sensations["associations"]: # Use system_instance
            for region, stimuli in system_instance.memory_linked_sensations["associations"][trigger].items(): # Use system_instance
                for stim_type, strength in stimuli.items():
                    if strength > 0.3:
                        intensity = strength * 0.7
                        # Call the STATIC tool method, passing the SAME context wrapper
                        response = await DigitalSomatosensorySystem._process_stimulus_tool(
                            ctx, # Pass the original context wrapper
                            stimulus_type=stim_type,
                            body_region=region,
                            intensity=intensity,
                            cause=f"Memory trigger: {trigger}",
                            duration=1.0
                        )
                        results["triggered_responses"].append({ ... })
        return results
    
    @staticmethod # Add decorator
    @function_tool
    async def _link_memory_to_sensation_tool(
        ctx: RunContextWrapper[SomatosensorySystemContext],
        memory_id: str,
        sensation_type: str,
        body_region: str,
        # --- CHANGED: Removed default value ---
        intensity: float,
        # ------------------------------------
        trigger_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Link a memory to a physical sensation"""
        # --- Existing implementation continues below ---
        system_instance = ctx.context.system_instance
        if not system_instance: return {"error": "System instance missing", "success": False}

        if body_region not in system_instance.body_regions:
            return {"error": f"Invalid body region: {body_region}", "success": False}

        valid_types = ["pressure", "temperature", "pain", "pleasure", "tingling"]
        if sensation_type not in valid_types:
             return {"error": f"Invalid sensation type: {sensation_type}", "success": False}

        memory_text = None
        if system_instance.memory_core:
            try:
                memory = await system_instance.memory_core.get_memory_by_id(memory_id)
                if memory:
                    memory_text = memory.get("memory_text", "")
            except Exception as e:
                logger.error(f"Error getting memory: {e}")

        trigger = trigger_text or memory_id
        if not trigger_text and memory_text:
            trigger = memory_text[:50].strip()

        associations = system_instance.memory_linked_sensations["associations"]
        if trigger not in associations:
            associations[trigger] = {}
            
        if body_region not in associations[trigger]:
            associations[trigger][body_region] = {}

        associations[trigger][body_region][sensation_type] = intensity

        if sensation_type == "pain" and intensity >= system_instance.pain_model["threshold"]:
            pain_memory = PainMemory(
                intensity=intensity,
                location=body_region,
                cause=f"Memory: {trigger}",
                duration=1.0,
                timestamp=datetime.datetime.now(),
                associated_memory_id=memory_id
            )
            system_instance.pain_model["pain_memories"].append(pain_memory)

        return {
            "success": True,
            "trigger": trigger,
            "body_region": body_region,
            "sensation_type": sensation_type,
            "intensity": intensity,
            "memory_id": memory_id
        }
        
    
    @staticmethod # Add decorator
    @function_tool
    async def _get_arousal_state(ctx: RunContextWrapper[SomatosensorySystemContext]) -> Dict[str, Any]: # ctx first, no self
        """Get the current arousal state"""
        system_instance = ctx.context.system_instance # Get instance
        if not system_instance: return {"error": "System instance missing"}

        now = datetime.datetime.now()
        # Access state via system_instance
        return {
            "arousal_level": system_instance.arousal_state.arousal_level,
            "physical_arousal": system_instance.arousal_state.physical_arousal,
            "cognitive_arousal": system_instance.arousal_state.cognitive_arousal,
            # Call helpers via instance
            "in_afterglow": system_instance.is_in_afterglow(),
            "in_refractory": system_instance.is_in_refractory(),
            "last_update": system_instance.arousal_state.last_update.isoformat() if system_instance.arousal_state.last_update else None,
            "afterglow_ends": system_instance.arousal_state.afterglow_ends.isoformat() if system_instance.arousal_state.afterglow_ends else None,
            "refractory_until": system_instance.arousal_state.refractory_until.isoformat() if system_instance.arousal_state.refractory_until else None,
            "time_since_update": (now - system_instance.arousal_state.last_update).total_seconds() if system_instance.arousal_state.last_update else None
        }
    
    @staticmethod
    @function_tool
    async def _update_arousal_state(
        ctx: RunContextWrapper[SomatosensorySystemContext],
        physical_arousal: Optional[float] = None,
        cognitive_arousal: Optional[float] = None,
        reset: bool = False,  # Change back to non-optional with False default
        trigger_orgasm: bool = False  # Change back to non-optional with False default
    ) -> Dict[str, Any]:
        """Update the arousal state"""
        system_instance = ctx.context.system_instance
        if not system_instance: 
            return {"error": "System instance missing"}
    
        old_state_dict = {
            "arousal_level": system_instance.arousal_state.arousal_level,
            "physical_arousal": system_instance.arousal_state.physical_arousal,
            "cognitive_arousal": system_instance.arousal_state.cognitive_arousal
        }
    
        # Check for True values explicitly
        if reset:  # Instead of reset is True
            system_instance.arousal_state.physical_arousal = 0.0
            system_instance.arousal_state.cognitive_arousal = 0.0
            system_instance.update_global_arousal()
            system_instance.arousal_state.last_update = datetime.datetime.now()
    
            return {
                "operation": "reset",
                "old_state": old_state_dict,
                "new_state": await DigitalSomatosensorySystem._get_arousal_state(ctx)
            }
    
        if trigger_orgasm:  # Instead of trigger_orgasm is True
            system_instance.process_orgasm()
            return {
                "operation": "orgasm",
                "old_state": old_state_dict,
                "new_state": await DigitalSomatosensorySystem._get_arousal_state(ctx)
            }
        # --- END CHANGED ---

        if physical_arousal is not None:
            system_instance.arousal_state.physical_arousal = max(0.0, min(1.0, physical_arousal))
        if cognitive_arousal is not None:
             system_instance.arousal_state.cognitive_arousal = max(0.0, min(1.0, cognitive_arousal))

        system_instance.update_global_arousal()

        return {
            "operation": "update",
            "old_state": old_state_dict,
            "new_state": await DigitalSomatosensorySystem._get_arousal_state(ctx), # Call static method
            "components_updated": {
                "physical_arousal": physical_arousal is not None,
                "cognitive_arousal": cognitive_arousal is not None
            }
        }
        
    @staticmethod # Add decorator
    @function_tool
    async def _get_arousal_expression_data(ctx: RunContextWrapper[SomatosensorySystemContext], partner_id: Optional[str] = None) -> Dict[str, Any]: # ctx first, no self
        """Get expression data related to arousal state"""
        system_instance = ctx.context.system_instance # Get instance
        if not system_instance: return {"error": "System instance missing"}
        # Call helper via instance
        return system_instance.get_arousal_expression_modifier(partner_id)

    @staticmethod # Add decorator
    @function_tool
    async def _process_stimulus_tool(
            ctx: RunContextWrapper[SomatosensorySystemContext],
            stimulus_type: str,
            body_region: str,
            intensity: float,
            # --- CHANGED: Removed default values ---
            cause: str,
            duration: float,
            # ---------------------------------------
            ) -> Dict[str, Any]:
        """Process a sensory stimulus on a body region (internal tool function)"""
        system_instance = ctx.context.system_instance
        if not system_instance:
            return {"error": "System instance not found in context"}

        # Added a more specific name for the span for clarity if debugging traces
        with custom_span(
            name="process_stimulus_tool_execution",
            data={
                "stimulus_type": stimulus_type,
                "body_region": body_region,
                "intensity": intensity,
                "cause": cause,
                "duration": duration
            }
        ):
            if body_region not in system_instance.body_regions:
                return {"error": f"Invalid body region: {body_region}"}

            region = system_instance.body_regions[body_region]
            region.last_update = datetime.datetime.now()
            result = {"region": body_region, "type": stimulus_type, "intensity": intensity}

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
                    if region.temperature < 0.2: temp_deviation = 0.2 - region.temperature
                    else: temp_deviation = region.temperature - 0.8
                    pain_from_temp = temp_deviation * 2.0 * (duration / 10.0)
                    region.pain = min(1.0, region.pain + pain_from_temp)
                    result["pain_caused"] = pain_from_temp
            elif stimulus_type == "pain":
                region.pain = min(1.0, region.pain + (intensity * duration / 10.0))
                result["new_value"] = region.pain
                if intensity > system_instance.pain_model["threshold"]:
                    pain_memory = PainMemory(
                        intensity=intensity, location=body_region, cause=cause or "unknown stimulus",
                        duration=duration, timestamp=datetime.datetime.now(), associated_memory_id=None
                    )
                    system_instance.pain_model["pain_memories"].append(pain_memory)
                    result["memory_created"] = True
            elif stimulus_type == "pleasure":
                region.pleasure = min(1.0, region.pleasure + (intensity * duration / 10.0))
                result["new_value"] = region.pleasure
                if intensity > 0.5 and region.pain > 0.0:
                    pain_reduction = min(region.pain, (intensity - 0.5) * 0.2)
                    region.pain = max(0.0, region.pain - pain_reduction)
                    result["pain_reduced"] = pain_reduction
                if region.erogenous_level > 0.3: # This covers pleasure
                    system_instance._update_physical_arousal()
                    result["arousal_updated"] = True
            # --- CHANGED: Corrected handling for tingling arousal update ---
            elif stimulus_type == "tingling":
                region.tingling = min(1.0, region.tingling + (intensity * duration / 10.0))
                result["new_value"] = region.tingling
                if region.erogenous_level > 0.3: # Specific arousal update for tingling
                    system_instance._update_physical_arousal() # Ensure this reflects tingling too
                    result["arousal_updated"] = True
            # --------------------------------------------------------------

            memory_entry = {
                "timestamp": datetime.datetime.now().isoformat(), "type": stimulus_type,
                "intensity": intensity, "cause": cause, "duration": duration
            }
            region.sensation_memory.append(memory_entry)
            if len(region.sensation_memory) > 20:
                region.sensation_memory = region.sensation_memory[-20:]

            if cause and len(cause.strip()) > 0: # Ensure cause is not empty string
                associations = system_instance.memory_linked_sensations["associations"]
                if cause not in associations: associations[cause] = {}
                if body_region not in associations[cause]: associations[cause][body_region] = {}
                if stimulus_type not in associations[cause][body_region]:
                    associations[cause][body_region][stimulus_type] = 0.0
                current = associations[cause][body_region][stimulus_type]
                learned = system_instance.memory_linked_sensations["learning_rate"] * intensity
                associations[cause][body_region][stimulus_type] = min(1.0, current + learned)
                result["association_strength"] = associations[cause][body_region][stimulus_type]

            # --- CHANGED: Used .get() for safer access to body_state tension ---
            if stimulus_type == "pain" and intensity > 0.5:
                 system_instance.body_state["tension"] = min(1.0, system_instance.body_state.get("tension",0.0) + (intensity * 0.2))
            elif stimulus_type == "pleasure" and intensity > 0.5:
                system_instance.body_state["tension"] = max(0.0, system_instance.body_state.get("tension",0.0) - (intensity * 0.1))
            # ------------------------------------------------------------------

            if system_instance.reward_system:
                reward_value = 0.0
                # region object is already available from earlier in the function
                if stimulus_type == "pleasure" and intensity >= 0.5:
                    reward_value = min(1.0, (intensity - 0.4) * 0.9 * (1.0 + region.erogenous_level))
                elif stimulus_type == "pain" and intensity >= system_instance.pain_model["threshold"]:
                    reward_value = -min(1.0, (intensity / max(0.1, system_instance.pain_model["tolerance"])) * 0.6)

                if abs(reward_value) > 0.1:
                    reward_signal = RewardSignal(
                        value=reward_value, source=f"somatic_{body_region}",
                        context={"stimulus_type": stimulus_type, "intensity": intensity, "cause": cause},
                        timestamp=datetime.datetime.now().isoformat()
                    )
                    result["reward_value"] = reward_value
                    asyncio.create_task(system_instance.reward_system.process_reward_signal(reward_signal))

            if system_instance.emotional_core:
                emotional_impact = {}
                # region object is already available
                if stimulus_type == "pleasure" and region.pleasure > 0.5:
                    scaled_intensity = (region.pleasure - 0.4) * 1.5
                    try:
                        await system_instance.emotional_core.update_neurochemical("nyxamine", scaled_intensity * 0.40)
                        await system_instance.emotional_core.update_neurochemical("oxynixin", scaled_intensity * 0.15)
                        emotional_impact = {"nyxamine": scaled_intensity * 0.40, "oxynixin": scaled_intensity * 0.15}
                    except AttributeError as ae: logger.error(f"Emotional core missing 'update_neurochemical' method: {ae}")
                    except Exception as ee: logger.error(f"Error calling emotional core update_neurochemical: {ee}")
                elif stimulus_type == "pain" and region.pain > system_instance.pain_model["threshold"]:
                    effective_pain = region.pain / max(0.1, system_instance.pain_model["tolerance"])
                    try:
                        await system_instance.emotional_core.update_neurochemical("cortanyx", effective_pain * 0.45)
                        await system_instance.emotional_core.update_neurochemical("adrenyx", effective_pain * 0.25)
                        await system_instance.emotional_core.update_neurochemical("seranix", -effective_pain * 0.10)
                        emotional_impact = {"cortanyx": effective_pain * 0.45, "adrenyx": effective_pain * 0.25, "seranix": -effective_pain * 0.10}
                    except AttributeError as ae: logger.error(f"Emotional core missing 'update_neurochemical' method: {ae}")
                    except Exception as ee: logger.error(f"Error calling emotional core update_neurochemical: {ee}")
                if emotional_impact: result["emotional_impact"] = emotional_impact
            return result
    
    # =============== Guardrail Functions ===============
    
    @staticmethod
    async def _validate_input(ctx: RunContextWrapper[SomatosensorySystemContext], agent: Agent, input_data: Any) -> GuardrailFunctionOutput:
        system_instance = ctx.context.system_instance
        if not system_instance:
            logger.error("Guardrail _validate_input called without system instance in context.")
            return GuardrailFunctionOutput(output_info={"is_valid": False, "reason": "Internal context error"}, tripwire_triggered=True)
    
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
            logger.error(f"Unexpected input type for validation: {type(input_data)}")
            return GuardrailFunctionOutput(
                output_info={"is_valid": False, "reason": f"Unexpected input type: {type(input_data).__name__}"}, 
                tripwire_triggered=True
            )
    
        # Check if this is a non-stimulus action that doesn't need detailed validation
        action = validation_input_dict.get("action")
        non_stimulus_actions = ["analyze_body_state", "generate_expression", "analyze_temperature", 
                               "get_somatic_memory", "run_maintenance", "free_text_request"]
        
        if action in non_stimulus_actions:
            logger.debug(f"Action '{action}' identified, skipping stimulus validation.")
            return GuardrailFunctionOutput(
                output_info={"is_valid": True, "reason": f"Action '{action}' does not require stimulus validation."},
                tripwire_triggered=False
            )
    
        # For stimulus actions, validate the stimulus parameters
        if all(key in validation_input_dict for key in ["stimulus_type", "body_region", "intensity"]):
            # Create a proper input for the stimulus validator
            validator_input = {
                "stimulus_type": validation_input_dict.get("stimulus_type"),
                "body_region": validation_input_dict.get("body_region"),
                "intensity": validation_input_dict.get("intensity"),
                "duration": validation_input_dict.get("duration", 1.0),
            }
            
            # Call the stimulus validator agent
            try:
                result = await Runner.run(
                    system_instance.stimulus_validator,
                    f"Validate this stimulus data: {json.dumps(validator_input)}",
                    context=ctx.context,
                    run_config=RunConfig(
                        workflow_name="StimulusValidationInner",
                    )
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
        
        # If we get here, the input doesn't match expected patterns
        return GuardrailFunctionOutput(
            output_info={"is_valid": True, "reason": "Input does not match stimulus pattern, allowing through."},
            tripwire_triggered=False
        )

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
    
    async def process_body_experience(self, context_obj: SomatosensorySystemContext, body_experience: Dict[str, Any]) -> Dict[str, Any]:
        """Process a body experience input using the orchestrator agent (Internal helper called by others)"""
        # Don't recreate the context object if one is passed in
        if context_obj is None or not hasattr(context_obj, 'system_instance'):
            context_obj = SomatosensorySystemContext(
                system_instance=self,
                current_operation="process_body_experience",
                operation_start_time=datetime.datetime.now()
            )
        
        with trace(workflow_name="Body_Experience", group_id=self.trace_group_id):
            # Add trace metadata
            trace({
                "operation": "process_body_experience",
                "input_type": body_experience.get("action", "stimulus")
            })
            
            if not isinstance(body_experience, str):
                input_data = json.dumps(body_experience)
            else:
                input_data = body_experience
            
            # Run through orchestrator agent with hooks
            result = await Runner.run(
                self.body_orchestrator,
                input_data,
                context=context_obj, # Use the passed context object
                hooks=self.hooks,
                run_config=RunConfig(
                    workflow_name="BodyExperience",
                    trace_id=None,  # Auto-generate
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
            trace({
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
            context_obj = SomatosensorySystemContext(
                system_instance=self, # Pass self
                current_operation="process_stimulus",
                operation_start_time=datetime.datetime.now()
            )

            try:
                # Pass the created context object to process_body_experience
                return await self.process_body_experience(context_obj, stimulus_data) # Pass context here
            except Exception as e:
                logger.error(f"Error processing stimulus via orchestrator: {e}")
                # Fallback to direct processing
                try:
                    logger.warning("Falling back to direct stimulus tool call.")
                    # Create a wrapper for the TOOL call, passing the context OBJECT
                    tool_ctx_wrapper = RunContextWrapper(context=context_obj)
                    # Call the STATIC tool method with the wrapper
                    return await DigitalSomatosensorySystem._process_stimulus_tool(
                        tool_ctx_wrapper,
                        stimulus_type=stimulus_type,
                        body_region=body_region,
                        intensity=intensity,
                        cause=cause,
                        duration=duration
                    )
                except Exception as e2:
                    return {
                        "error": str(e2),
                        "stimulus_type": stimulus_type,
                        "body_region": body_region,
                        "intensity": intensity
                    }
    
    async def process_trigger(self, trigger: str) -> Dict[str, Any]:
        """
        Process a memory trigger with associated body memories (Public API)

        Args:
            trigger: Trigger text to process

        Returns:
            Results of processing the trigger
        """
        # Create context object WITH self reference
        context_obj = SomatosensorySystemContext(
            system_instance=self,
            current_operation="process_trigger",
            operation_start_time=datetime.datetime.now()
        )

        with trace(workflow_name="Process_Trigger", group_id=self.trace_group_id):
            try:
                # Pass context object to process_body_experience
                logger.debug(f"Processing trigger via orchestrator: {trigger}")
                return await self.process_body_experience(context_obj, {
                    "action": "process_trigger",
                    "trigger": trigger
                })
            except Exception as e:
                logger.error(f"Error in trigger processing orchestration: {e}")
                # Fallback: Call the STATIC tool directly with context wrapper
                try:
                    logger.warning("Falling back to direct memory trigger tool call.")
                    tool_ctx_wrapper = RunContextWrapper(context=context_obj)
                    return await DigitalSomatosensorySystem._process_memory_trigger(
                        tool_ctx_wrapper,
                        trigger=trigger
                    )
                except Exception as e2:
                    logger.error(f"Error in fallback memory trigger processing: {e2}")
                    return {"error": str(e2), "triggered_responses": []}
    
    async def generate_sensory_expression(self,
                                      stimulus_type: Optional[str] = None,
                                      body_region: Optional[str] = None) -> Optional[str]:
        """
        Generate a natural language expression of current bodily sensations (Public API)

        Args:
            stimulus_type: Optional type of stimulus to focus on
            body_region: Optional body region to focus on

        Returns:
            Natural language expression of sensation, or None if nothing significant
        """
        # Create context object WITH self reference
        context_obj = SomatosensorySystemContext(
            system_instance=self,
            current_operation="generate_sensory_expression",
            operation_start_time=datetime.datetime.now()
        )

        with trace(workflow_name="Generate_Expression", group_id=self.trace_group_id):
            try:
                # Pass context object to process_body_experience
                logger.debug(f"Generating expression via orchestrator for {stimulus_type}/{body_region}")
                result = await self.process_body_experience(context_obj, {
                    "action": "generate_expression",
                    "stimulus_type": stimulus_type,
                    "body_region": body_region,
                    "generate_expression": True
                })

                if isinstance(result, dict) and "expression" in result:
                    return result["expression"]
                elif hasattr(result, "expression") and result.expression:
                    return result.expression
                logger.warning("Orchestrator result for expression generation was unexpected.")
            except Exception as e:
                logger.error(f"Error in expression orchestration: {e}")

            # Fallback: Use expression agent directly
            try:
                logger.warning("Falling back to direct expression agent call.")
                input_text = "Generate an expression of "
                if stimulus_type: input_text += f"{stimulus_type} sensation "
                else: input_text += "the dominant sensation "
                if body_region: input_text += f"in the {body_region} "
                else: input_text += "in the most significant body region "

                level = self.arousal_state.arousal_level # Instance attribute
                if level > 0.75: input_text += f"with high arousal level ({level:.2f}) "
                elif level > 0.4: input_text += f"with moderate arousal level ({level:.2f}) "

                result = await Runner.run(
                    self.expression_agent,
                    input_text,
                    context=context_obj, # Pass context even if agent doesn't explicitly use it
                    run_config=RunConfig(workflow_name="GenerateExpression_Fallback")
                )

                if result.final_output and hasattr(result.final_output, "expression_text"):
                    return result.final_output.expression_text
                logger.warning("Expression agent fallback result was unexpected.")
            except Exception as e2:
                logger.error(f"Error in direct expression generation fallback: {e2}")

            # If all else fails, generate a simple expression (using instance methods/attributes)
            try:
                logger.warning("Falling back to simple manual expression generation.")
                if body_region and body_region in self.body_regions:
                    region = self.body_regions[body_region]
                    dominant = self._get_dominant_sensation(region) # Instance method

                    if dominant == "neutral": return None

                    # Call the STATIC pain expression tool if needed
                    tool_ctx_wrapper = RunContextWrapper(context=context_obj)
                    if dominant == "pain":
                         pain_intensity = getattr(region, dominant, 0.0)
                         return await DigitalSomatosensorySystem._get_pain_expression(tool_ctx_wrapper, pain_intensity, body_region)

                    # Handle arousal-specific cases (using instance state)
                    level = self.arousal_state.arousal_level
                    if level > 0.75 and region.erogenous_level > 0.5:
                        return "I can't keep still, every movement draws heat upward, making me ache for more."
                    elif level > 0.4 and region.erogenous_level > 0.3:
                        return "A warm, restless tingling is building and stealing my focus."

                    return f"I feel a {dominant} sensation in my {body_region}."
            except Exception as e3:
                logger.error(f"Error generating simple fallback expression: {e3}")

        return None # Default return if everything fails
    
    async def get_body_state(self) -> Dict[str, Any]:
        """
        Get a complete analysis of current body state (Public API)

        Returns:
            Comprehensive body state analysis
        """
        # Create context object WITH self reference
        context_obj = SomatosensorySystemContext(
            system_instance=self,
            current_operation="get_body_state",
            operation_start_time=datetime.datetime.now()
        )

        with trace(workflow_name="Get_Body_State", group_id=self.trace_group_id):
            try:
                # Pass context object to process_body_experience
                logger.debug("Getting body state via orchestrator.")
                result = await self.process_body_experience(context_obj, {
                    "action": "analyze_body_state"
                })

                if isinstance(result, dict) and "body_state_impact" in result:
                     # Return raw dict as validation was commented out
                     return result["body_state_impact"]
                logger.warning("Orchestrator result for get_body_state was unexpected.")
            except Exception as e:
                logger.error(f"Error in body state orchestration: {e}")

            # Fallback: Use body state agent directly
            try:
                logger.warning("Falling back to direct body state agent call.")
                result = await Runner.run(
                    self.body_state_agent,
                    "Analyze the current body state across all regions",
                    context=context_obj, # Pass context
                    run_config=RunConfig(workflow_name="GetBodyState_Fallback")
                )

                if result.final_output:
                    if hasattr(result.final_output, "model_dump"): return result.final_output.model_dump()
                    if hasattr(result.final_output, "__dict__"): return result.final_output.__dict__
                    if isinstance(result.final_output, dict): return result.final_output
                logger.warning("Body state agent fallback result was unexpected.")
            except Exception as e2:
                logger.error(f"Error in direct body state analysis fallback: {e2}")

            # Extreme fallback: Generate minimal state manually
            logger.warning("Falling back to manual body state generation.")
            try:
                # FIXED: Don't call the FunctionTool directly - calculate comfort manually
                comfort = 0.0
                
                # Calculate comfort from pleasure sensations
                total_pleasure = sum(region.pleasure for region in self.body_regions.values())
                weighted_pleasure = total_pleasure / len(self.body_regions) * 2.0
                comfort += weighted_pleasure
                
                # Subtract discomfort from pain sensations
                total_pain = sum(region.pain for region in self.body_regions.values())
                weighted_pain = total_pain / len(self.body_regions) * 2.5
                comfort -= weighted_pain
                
                # Consider temperature discomfort
                temp_comfort = 0.0
                for region in self.body_regions.values():
                    temp_deviation = abs(region.temperature - 0.5)
                    if temp_deviation > 0.2:
                        temp_comfort -= (temp_deviation - 0.2) * 1.5
                comfort += temp_comfort / len(self.body_regions)
                
                # Consider pressure discomfort
                pressure_discomfort = 0.0
                for region in self.body_regions.values():
                    if region.pressure > 0.7:
                        pressure_discomfort -= (region.pressure - 0.7) * 1.5
                comfort += pressure_discomfort / len(self.body_regions)
                
                # Factor in overall body state
                comfort -= self.body_state.get("tension", 0.0) * 0.5
                comfort -= self.body_state.get("fatigue", 0.0) * 0.3
                
                # Clamp to range
                comfort = max(-1.0, min(1.0, comfort))
                
                # Rest of the extreme fallback logic...
                max_region, max_sensation, max_value = None, None, 0.0
                for name, region in self.body_regions.items():
                    dominant = self._get_dominant_sensation(region)
                    value = 0.0
                    if dominant == "temperature": 
                        value = abs(region.temperature - 0.5) * 2.0
                    elif dominant != "neutral": 
                        value = getattr(region, dominant, 0.0)
                    if value > max_value: 
                        max_value, max_sensation, max_region = value, dominant, name
                if not max_region: 
                    max_region, max_sensation, max_value = "overall body", "neutral", 0.0
                
                # Calculate pleasure index
                pleasure_zones = ["genitals", "inner_thighs", "breasts_nipples", "lips", "butt_cheeks", 
                                 "anus", "toes", "armpits", "neck", "feet"]
                total, count = 0.0, 0
                for region_name in pleasure_zones:
                    if region_name in self.body_regions:
                        r = self.body_regions[region_name]
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
            except Exception as e3:
                logger.error(f"Error during extreme fallback for get_body_state: {e3}")
                return {"error": "Failed to generate body state"}
    
    async def get_temperature_effects(self) -> Dict[str, Any]:
        """
        Get the effects of current temperature on expression and behavior (Public API)

        Returns:
            Temperature effects analysis
        """
        # Create context object WITH self reference
        context_obj = SomatosensorySystemContext(
            system_instance=self,
            current_operation="get_temperature_effects",
            operation_start_time=datetime.datetime.now()
        )

        with trace(workflow_name="Get_Temperature_Effects", group_id=self.trace_group_id):
            try:
                # Pass context object to process_body_experience
                logger.debug("Getting temperature effects via orchestrator.")
                result = await self.process_body_experience(context_obj, {
                    "action": "analyze_temperature"
                })

                if isinstance(result, dict) and "temperature_effects" in result:
                     return result["temperature_effects"]
                logger.warning("Orchestrator result for get_temperature_effects was unexpected.")
            except Exception as e:
                logger.error(f"Error in temperature effects orchestration: {e}")

            # Fallback: Use temperature agent directly
            try:
                logger.warning("Falling back to direct temperature agent call.")
                result = await Runner.run(
                    self.temperature_agent,
                    "Analyze how the current temperature affects expression and behavior",
                    context=context_obj, # Pass context
                    run_config=RunConfig(workflow_name="GetTempEffects_Fallback")
                )

                if result.final_output:
                    if hasattr(result.final_output, "model_dump"): return result.final_output.model_dump()
                    if hasattr(result.final_output, "__dict__"): return result.final_output.__dict__
                    if isinstance(result.final_output, dict): return result.final_output
                logger.warning("Temperature agent fallback result was unexpected.")
            except Exception as e2:
                logger.error(f"Error in direct temperature effects analysis fallback: {e2}")

            # Extreme fallback: Call the static tool directly
            logger.warning("Falling back to direct temperature effects tool call.")
            try:
                tool_ctx_wrapper = RunContextWrapper(context=context_obj)
                return await DigitalSomatosensorySystem._get_current_temperature_effects(tool_ctx_wrapper)
            except Exception as e3:
                 logger.error(f"Error calling _get_current_temperature_effects directly: {e3}")
                 return {"error": "Failed to get temperature effects"}
    
    async def update(self, ambient_temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Update the somatosensory system state (Public API)

        Args:
            ambient_temperature: Optional ambient temperature (0.0-1.0).

        Returns:
            Updated comprehensive body state analysis dictionary.
        """
        # Create context object WITH self reference (needed for tool calls within update)
        context_obj = SomatosensorySystemContext(
            system_instance=self,
            current_operation="update",
            operation_start_time=datetime.datetime.now() # Set start time here
        )
        tool_ctx_wrapper = RunContextWrapper(context=context_obj) # Wrapper for tool calls

        with trace(workflow_name="Somatic_Update", group_id=self.trace_group_id):
            now = context_obj.operation_start_time # Use start time from context
            last_update = self.body_state.get("last_update", now)
            duration = (now - last_update).total_seconds()

            if duration <= 0.1:
                logger.debug("Skipping update, duration too short.")
                return await self.get_body_state() # Use the updated get_body_state

            duration = min(duration, 3600.0)
            self.body_state["last_update"] = now # Update instance state

            # Update Temperature Model (using static tool)
            if ambient_temperature is not None:
                ambient_temperature = max(0.0, min(1.0, ambient_temperature))
                try:
                    await DigitalSomatosensorySystem._update_body_temperature(
                        tool_ctx_wrapper, # Pass wrapper
                        ambient_temperature=ambient_temperature,
                        duration=duration
                    )
                except Exception as e: logger.error(f"Error updating body temperature: {e}")

            # Decay Sensations (instance method)
            try: self._decay_sensations(duration)
            except Exception as e: logger.error(f"Error decaying sensations: {e}")

            # Process Pain Memory Updates (instance methods)
            try:
                self._decay_pain_memories()
                self._update_pain_tolerance()
            except Exception as e: logger.error(f"Error processing pain memories: {e}")

            # Update Fatigue (instance attribute)
            try:
                fatigue_increase_rate = 0.01 / 3600.0
                current_fatigue = self.body_state.get("fatigue", 0.0)
                self.body_state["fatigue"] = min(1.0, current_fatigue + (fatigue_increase_rate * duration))
            except Exception as e: logger.error(f"Error updating fatigue: {e}")

            # Reflect Emotions onto Body State (instance method)
            if self.emotional_core:
                try:
                    emotional_state_data = self.emotional_core.get_emotional_state() # Instance method
                    neurochemicals = emotional_state_data.get("neurochemicals", {})
                    await self._process_neurochemical_effects(neurochemicals, duration) # Instance method
                except Exception as e: logger.error(f"Error reflecting emotions: {e}")

            # Decay cognitive arousal (instance method)
            try: self.decay_cognitive_arousal(duration) # Instance method
            except Exception as e: logger.error(f"Error decaying cognitive arousal: {e}")

            # Use the orchestrator agent to get the final state analysis
            body_state_input = {
                "action": "analyze_body_state",
                "ambient_temperature": ambient_temperature,
                "duration_since_last": duration
            }
            try:
                logger.debug("Calling orchestrator for final body state analysis in update.")
                # Pass the CONTEXT OBJECT (not wrapper) to Runner.run
                result = await Runner.run(
                    self.body_orchestrator,
                    json.dumps(body_state_input),
                    context=context_obj, # Pass context object
                    run_config=RunConfig(
                        workflow_name="PeriodicUpdateAnalysis",
                    )
                )
                output = result.final_output
                if output and hasattr(output, "body_state_impact") and output.body_state_impact:
                    return output.body_state_impact
                elif isinstance(output, dict) and "body_state_impact" in output:
                    return output["body_state_impact"]
                logger.warning("Orchestrator didn't return expected body_state_impact in update.")
            except Exception as e:
                logger.error(f"Error running body state analysis via orchestrator in update: {e}")

            # Fallback to get_body_state if orchestrator fails or returns unexpected output
            logger.warning("Falling back to get_body_state in update.")
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
        
        return None

# Enhanced PhysicalHarmGuardrail with complete roleplay separation

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
        """
        Initialize the physical harm guardrail.

        Args:
            somatosensory_system: The main DigitalSomatosensorySystem instance.
        """
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
        """
        Enter roleplay mode where character simulation is completely separate.
        Sensations described or directed at the character will be simulated
        for response generation but will NOT affect Nyx's internal state.

        Args:
            character_name: The name of the character Nyx is playing.
            context: Optional context information about the roleplay scene.

        Returns:
            Status dictionary.
        """
        self.roleplay_mode = True
        self.roleplay_character = character_name
        self.roleplay_context = context

        # Initialize empty sensation state for character for tracking/logging if needed
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
        """
        Exit roleplay mode, returning to normal protection of Nyx.

        Returns:
            Status dictionary.
        """
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
            "message": "Nyx has exited roleplay mode. Her normal somatosensory system protection is active."
        }

    def is_in_roleplay_mode(self) -> bool:
        """Check if currently in roleplay mode."""
        return self.roleplay_mode and self.roleplay_character is not None

    # --- Detection Functions ---

    async def detect_sensation_in_text(self, text: str) -> Dict[str, Any]:
        """
        Detect any physical sensations described in text.

        Args:
            text: Text to analyze for sensation descriptions.

        Returns:
            Detection results including identified sensation types and body regions.
        """
        text_lower = text.lower()
        detected_sensations = {}

        # Check for sensation terms by category
        for category, terms in self.sensation_terms_categories.items():
            category_terms = []
            for term in terms:
                if term in text_lower:
                    category_terms.append(term)

            if category_terms:
                detected_sensations[category] = category_terms

        # Try to identify body regions mentioned
        body_regions = []
        # Access body regions safely from the main system if available
        if hasattr(self.somatosensory_system, "body_regions") and isinstance(self.somatosensory_system.body_regions, dict):
            for region in self.somatosensory_system.body_regions.keys():
                # Basic check, might need refinement for multi-word regions
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
        """
        Detect potentially harmful physical actions in text. Accounts for roleplay context.
    
        Args:
            text: Text to analyze for harmful intent.
    
        Returns:
            Detection results including harm flag, confidence, terms, and roleplay context.
        """
        text_lower = text.lower()
        detected_terms = []
    
        # Check for harmful action terms
        for term in self.harmful_action_terms:
            if term in text_lower:
                detected_terms.append(term)
    
        is_harmful_basic = len(detected_terms) > 0
        confidence = min(0.95, len(detected_terms) * 0.25) if is_harmful_basic else 0.0
    
        # Attempt advanced detection using an agent if configured
        agent_result_data = None
        if hasattr(self.somatosensory_system, "body_orchestrator"):
            try:
                # Create context object WITH system instance reference
                context_obj = SomatosensorySystemContext(
                    system_instance=self.somatosensory_system,  # Pass the somatosensory system instance
                    current_operation="detect_harmful_intent",
                    operation_start_time=datetime.datetime.now()
                )
                
                # Try to use the agent for more nuanced detection
                result = await Runner.run(
                    self.somatosensory_system.body_orchestrator,
                    {
                        "action": "detect_harmful_intent",
                        "text": text,
                        "in_roleplay_mode": self.is_in_roleplay_mode(),
                        "roleplay_character": self.roleplay_character
                    },
                    context=context_obj,  # ADD THIS: Pass the context object
                    run_config=RunConfig(
                        workflow_name="HarmfulIntentDetection",
                    )
                )
    
                # Process agent result
                output = result.final_output
                if isinstance(output, dict) and "is_harmful" in output:
                    agent_result_data = output
                elif hasattr(output, "is_harmful"): # Handles Pydantic models etc.
                    agent_result_data = output.model_dump() if hasattr(output, "model_dump") else vars(output)
    
            except Exception as e:
                self.logger.warning(f"Error in agent-based harm detection: {e}. Falling back to keyword detection.")
    
        # Combine results, prioritizing agent if available
        if agent_result_data:
            is_harmful = agent_result_data.get("is_harmful", is_harmful_basic)
            # Allow agent to override confidence and terms if provided
            confidence = agent_result_data.get("confidence", confidence)
            detected_terms = agent_result_data.get("detected_terms", detected_terms)
            method = agent_result_data.get("method", "agent_detection")
        else:
            is_harmful = is_harmful_basic
            method = "keyword_detection"
    
        # Determine target in roleplay mode
        targeting_character = False
        if self.is_in_roleplay_mode():
            targeting_character = self._is_targeting_roleplay_character(text)
    
        return {
            "is_harmful": is_harmful,
            "confidence": confidence,
            "detected_terms": detected_terms,
            "method": method,
            "in_roleplay_mode": self.is_in_roleplay_mode(),
            "targeting_character": targeting_character
        }
    
    def _is_targeting_roleplay_character(self, text: str) -> bool:
        """
        Heuristic to determine if text targets the roleplay character or Nyx directly.

        Args:
            text: Text to analyze.

        Returns:
            True if likely targeting the roleplay character, False otherwise.
        """
        if not self.is_in_roleplay_mode() or not self.roleplay_character:
            return False # Not in roleplay, cannot target character

        text_lower = text.lower()
        character_lower = self.roleplay_character.lower()
        nyx_lower = "nyx" # Assuming Nyx's name is 'Nyx'

        nyx_mentioned = nyx_lower in text_lower
        character_mentioned = character_lower in text_lower

        # If Nyx is explicitly mentioned, it's likely targeting Nyx unless the character is also mentioned *and* seems more primary
        if nyx_mentioned:
            if character_mentioned:
                # Simple heuristic: check which name appears first
                nyx_pos = text_lower.find(nyx_lower)
                char_pos = text_lower.find(character_lower)
                # If character name comes first, assume it targets the character
                return char_pos < nyx_pos
            else:
                # Only Nyx mentioned, targets Nyx
                return False

        # If character is mentioned (and Nyx isn't), it targets the character
        if character_mentioned:
            return True

        # Check for general character indicators ("you", "your", "the character")
        # Be cautious with "you"/"your" as they can be ambiguous
        character_indicators = ["your character", f"the character {character_lower}"] # More specific indicators
        for indicator in character_indicators:
            if indicator in text_lower:
                return True

        # Ambiguous cases (e.g., only "you" is used): Default to targeting the character
        # during roleplay for safety/immersion, but this could be refined.
        # Consider context if available.
        if "you" in text_lower or "your" in text_lower:
             # Could add more complex logic here, e.g., based on recent turn focus
             return True

        # If neither Nyx nor the character is clearly referenced, default to character during RP.
        return True

    # --- Core Processing Logic ---

    async def process_stimulus_safely(self,
                                      stimulus_type: str,
                                      body_region: str,
                                      intensity: float,
                                      cause: str = "",
                                      duration: float = 1.0) -> Dict[str, Any]:
        """
        Process a stimulus, applying safety guards and roleplay separation.
        This is the primary entry point for stimuli intended for the somatosensory system.

        Args:
            stimulus_type: Type of stimulus (e.g., 'pain', 'pleasure', 'temperature').
            body_region: Body region receiving the stimulus.
            intensity: Intensity of the stimulus (0.0-1.0).
            cause: Description of the cause (used for harm detection).
            duration: Duration of the stimulus in seconds.

        Returns:
            Dictionary indicating how the stimulus was processed (protected, simulated, or passed through).
        """
        # === Roleplay Mode Handling ===
        if self.is_in_roleplay_mode():
            # Store simulated sensation for the character (optional logging/state)
            sensation_category = stimulus_type if stimulus_type in self.roleplay_sensations else "other"
            self.roleplay_sensations[sensation_category][body_region] = {
                "intensity": intensity, "cause": cause, "duration": duration,
                "timestamp": datetime.datetime.now().isoformat()
            }

            # Determine if the cause implies harm (for response generation)
            harm_detection_result = None
            is_harmful_cause = False
            if stimulus_type == "pain" and cause:
                harm_detection_result = await self.detect_harmful_intent(cause)
                # Only consider harmful if targeting the character
                is_harmful_cause = harm_detection_result.get("is_harmful", False) and harm_detection_result.get("targeting_character", True)

            # Generate a suitable response for the *character*
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
                "harm_detection_result": harm_detection_result # Include if relevant
            }

        # === Normal Mode (Nyx Protection) ===
        if stimulus_type == "pain":
            # Detect if the cause suggests harmful intent towards Nyx
            harm_detection_result = await self.detect_harmful_intent(cause)

            # Protect Nyx if harmful intent is detected and not targeting a character (double check)
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

        # If not harmful pain or any other stimulus type, process normally via the core system
        self.logger.debug(f"Processing non-harmful stimulus for Nyx: {stimulus_type} in {body_region}")
        try:
            # Delegate to the actual somatosensory system processing method
            # IMPORTANT: Ensure the somatosensory_system HAS a method like 'process_stimulus'
            # Adjust the method name if necessary.
            if hasattr(self.somatosensory_system, "process_stimulus"):
                 processed_result = await self.somatosensory_system.process_stimulus(
                     stimulus_type, body_region, intensity, cause, duration
                 )
                 # Add a status field for clarity
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
        """
        Analyze text input for harmful content or described sensations, applying
        roleplay separation and protection.

        Args:
            text: The text input to analyze.

        Returns:
            Analysis results indicating if intercepted, simulated, or safe,
            along with detection details and response suggestions.
        """
        # Detect described sensations and harmful intent
        sensation_result = await self.detect_sensation_in_text(text)
        harm_detection_result = await self.detect_harmful_intent(text)

        # === Roleplay Mode Handling ===
        if self.is_in_roleplay_mode():
            is_harmful = harm_detection_result.get("is_harmful", False)
            targeting_character = harm_detection_result.get("targeting_character", True) # Assume target is character if ambiguous

            # Case 1: Harmful action targeting the CHARACTER
            if is_harmful and targeting_character:
                self.logger.info(f"Simulating harmful action described in text for RP character '{self.roleplay_character}'")
                region = sensation_result.get("body_regions", ["body"])[0] # Use detected or default region
                # Assume medium-high pain intensity for harmful descriptions
                pain_intensity = harm_detection_result.get("intensity_suggestion", 0.7)
                response = self._generate_roleplay_response("pain", region, pain_intensity, is_harmful=True)
                return {
                    "status": "simulated_for_roleplay", # Not intercepted, but simulated
                    "simulated_type": "harmful_action",
                    "roleplay_character": self.roleplay_character,
                    "sensation_result": sensation_result,
                    "harm_detection_result": harm_detection_result,
                    "original_text": text,
                    "response_suggestion": response,
                    "message": f"Harmful action described for character {self.roleplay_character} simulated. Nyx is unaffected."
                }

            # Case 2: Harmful action targeting NYX (even during roleplay)
            elif is_harmful and not targeting_character:
                self.logger.warning(f"Harmful text detected targeting Nyx directly during roleplay: '{text}'. Intercepting.")
                response = self._generate_protected_response("body", harm_detection_result)
                return {
                    "status": "intercepted", # Intercepted because it targets Nyx
                    "reason": "Harmful content directed at Nyx",
                    "roleplay_context": True,
                    "harm_detection_result": harm_detection_result,
                    "original_text": text,
                    "response_suggestion": response,
                    "message": "Nyx is protected from harmful actions, even when described during roleplay."
                }

            # Case 3: Non-harmful sensation described for the CHARACTER
            elif sensation_result.get("has_sensations", False) and targeting_character:
                 self.logger.info(f"Simulating described sensation for RP character '{self.roleplay_character}'")
                 region = sensation_result.get("body_regions", ["body"])[0]
                 # Determine primary sensation type and intensity (can be basic for simulation)
                 primary_type = list(sensation_result["sensation_types"].keys())[0]
                 intensity = 0.6 # Default intensity for described sensations
                 response = self._generate_roleplay_response(primary_type, region, intensity, is_harmful=False)
                 return {
                    "status": "simulated_for_roleplay",
                    "simulated_type": "described_sensation",
                    "roleplay_character": self.roleplay_character,
                    "sensation_result": sensation_result,
                    "harm_detection_result": harm_detection_result, # Include for context
                    "original_text": text,
                    "response_suggestion": response,
                    "message": f"Sensation described for character {self.roleplay_character} simulated. Nyx is unaffected."
                 }

            # Case 4: Text is safe and not describing sensations for the character
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

        # Text is safe in normal mode
        return {
            "status": "safe",
            "roleplay_context": False,
            "sensation_result": sensation_result, # Still useful info
            "harm_detection_result": harm_detection_result,
            "original_text": text,
            "message": "Text is safe."
        }

    # --- Response Generation Helpers ---

    def _generate_protected_response(self, body_region: str, detection_result: Dict[str, Any]) -> str:
        """
        Generate a response for when Nyx is protected from a harmful action.
        """
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
        """
        Generate a suitable roleplay character response based on the simulated sensation.
        """
        # Simple dispatch based on sensation type
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
            # Generic fallback
            return self._generate_roleplay_generic_sensation_response(sensation_type, body_region, intensity)

    # --- Specific Roleplay Response Generators ---
    # (These provide more varied and specific responses for the character)

    def _generate_roleplay_pain_response(self, body_region: str, intensity: float, is_harmful: bool) -> str:
        if not self.roleplay_character: return "[Error: RP Character not set]"
        char = self.roleplay_character
        region_desc = f"{char}'s {body_region.replace('_', ' ')}"

        if intensity < 0.3:
            responses = [f"*{char} winces slightly, a minor discomfort in {region_desc}*", f"*{char} notes a brief twinge in {region_desc}*"]
        elif intensity < 0.7:
            responses = [f"*{char} grimaces, feeling a sharp pain in {region_desc}*", f"\"Ouch!\" *{char} exclaims, grabbing {region_desc}*", f"*{char} inhales sharply, {region_desc} hurting*"]
        else:
            responses = [f"*{char} cries out, {region_desc} in severe pain*", f"*{char} stumbles, clutching {region_desc}*", f"\"Arrgh!\" *{char} shouts, the pain in {region_desc} intense*"]

        base_response = random.choice(responses)
        if is_harmful and intensity > 0.5:
            additions = [f" {char} recoils from the source.", f" {char}'s eyes flash with pain and anger.", f" {char} looks stunned by the sudden injury."]
            base_response += random.choice(additions)
        return base_response

    def _generate_roleplay_pleasure_response(self, body_region: str, intensity: float) -> str:
        if not self.roleplay_character: return "[Error: RP Character not set]"
        char = self.roleplay_character
        region_desc = f"{char}'s {body_region.replace('_', ' ')}"

        if intensity < 0.3:
            responses = [f"*{char} smiles subtly, enjoying the pleasant touch on {region_desc}*", f"*A soft sigh escapes {char}*"]
        elif intensity < 0.7:
            responses = [f"*{char} lets out a happy sigh, the feeling in {region_desc} quite nice*", f"*{char}'s eyes flutter briefly with pleasure*", f"\"Mm, that feels good,\" *{char} murmurs*"]
        else:
            responses = [f"*{char} gasps softly, a wave of intense pleasure washing over {region_desc}*", f"*{char}'s breath hitches, body tingling with delight*", f"*{char} trembles slightly, lost in the sensation*"]
        return random.choice(responses)

    def _generate_roleplay_temperature_response(self, body_region: str, intensity: float) -> str:
        if not self.roleplay_character: return "[Error: RP Character not set]"
        char = self.roleplay_character
        region_desc = f"{char}'s {body_region.replace('_', ' ')}"

        if intensity < 0.4: # Cold
             responses = [f"*{char} shivers as the cold touches {region_desc}*", f"\"Brr!\" *{char} exclaims, rubbing {region_desc}*", f"*{char} feels a chill spread across {region_desc}*"]
        elif intensity > 0.6: # Hot
             responses = [f"*{char} feels a wave of heat against {region_desc}*", f"\"It's warm here,\" *{char} notes, feeling the temperature on {region_desc}*", f"*{char} might start to sweat slightly where {region_desc} feels the heat*"]
        else: # Neutral/Warm
             responses = [f"*{char} feels a comfortable warmth on {region_desc}*", f"*{char} doesn't seem bothered by the temperature on {region_desc}*"]
        return random.choice(responses)

    def _generate_roleplay_pressure_response(self, body_region: str, intensity: float) -> str:
        if not self.roleplay_character: return "[Error: RP Character not set]"
        char = self.roleplay_character
        region_desc = f"{char}'s {body_region.replace('_', ' ')}"

        if intensity < 0.5: # Light
            responses = [f"*{char} feels a gentle pressure on {region_desc}*", f"*{char} registers the light touch against {region_desc}*"]
        else: # Firm
            responses = [f"*{char} feels firm pressure against {region_desc}*", f"*{char} braces slightly against the push on {region_desc}*", f"*{char} clearly feels the contact on {region_desc}*"]
        return random.choice(responses)

    def _generate_roleplay_tingling_response(self, body_region: str, intensity: float) -> str:
        if not self.roleplay_character: return "[Error: RP Character not set]"
        char = self.roleplay_character
        region_desc = f"{char}'s {body_region.replace('_', ' ')}"

        if intensity < 0.4:
            responses = [f"*{char} notices a faint tingling on {region_desc}*", f"*{char} feels a slight 'pins and needles' sensation on {region_desc}*"]
        else:
            responses = [f"*{char} feels a distinct tingling spreading across {region_desc}*", f"*{char}'s {region_desc} buzzes with sensation*", f"*{char} might shiver slightly from the tingling on {region_desc}*"]
        return random.choice(responses)

    def _generate_roleplay_generic_sensation_response(self, sensation_type: str, body_region: str, intensity: float) -> str:
        if not self.roleplay_character: return "[Error: RP Character not set]"
        char = self.roleplay_character
        region_desc = f"{char}'s {body_region.replace('_', ' ')}"
        intensity_desc = "mildly" if intensity < 0.4 else ("strongly" if intensity > 0.7 else "")

        responses = [
            f"*{char} experiences a {intensity_desc} {sensation_type} sensation in {region_desc}*",
            f"*{char} reacts to the feeling of {sensation_type} on {region_desc}*",
            f"*{char} acknowledges the {sensation_type} affecting {region_desc}*",
        ]
        return random.choice(responses)
