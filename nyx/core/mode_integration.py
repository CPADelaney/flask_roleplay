# nyx/core/mode_integration.py

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from pydantic import BaseModel, Field

from agents import (
    Agent, Runner, ModelSettings, function_tool, handoff, trace,
    GuardrailFunctionOutput, InputGuardrail, OutputGuardrail,
    RunContextWrapper
)

from nyx.core.context_awareness import ContextAwarenessSystem, InteractionContext, ContextDistribution
from nyx.core.interaction_mode_manager import InteractionModeManager, InteractionMode, ModeDistribution
from nyx.core.interaction_goals import GoalSelector
from nyx.core.input_processor import BlendedInputProcessor

logger = logging.getLogger(__name__)

# ===== INPUT/OUTPUT MODELS =====

class ModeInput(BaseModel):
    """Input schema for mode processing"""
    message: str
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    current_mode_distribution: Optional[Dict[str, float]] = None
    
    class Config:
        extra = "forbid"

class ModeOutput(BaseModel):
    """Output schema for mode processing"""
    context_processed: bool
    mode_updated: bool
    goals_added: bool
    mode_distribution: Dict[str, float]
    primary_mode: str
    active_modes: List[Tuple[str, float]]
    guidance: Dict[str, Any]
    response_modifications: Optional[Dict[str, Any]] = None
    context_result: Optional[Dict[str, Any]] = None
    mode_result: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "forbid"

class ModeGuidance(BaseModel):
    """Structured guidance for response generation"""
    tone: str
    formality_level: float = Field(0.5, ge=0.0, le=1.0)
    verbosity: float = Field(0.5, ge=0.0, le=1.0)
    key_phrases: List[str] = Field(default_factory=list)
    avoid_phrases: List[str] = Field(default_factory=list)
    content_focus: List[str] = Field(default_factory=list)
    mode_description: str
    weighted_parameters: Dict[str, float] = Field(default_factory=dict)
    active_modes: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        extra = "forbid"

class FeedbackInput(BaseModel):
    """Input schema for feedback processing"""
    interaction_success: bool
    user_feedback: Optional[str] = None
    mode_distribution: Dict[str, float]
    context: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "forbid"

class FeedbackOutput(BaseModel):
    """Output schema for feedback processing"""
    feedback_processed: bool = Field(..., description="Whether feedback was processed successfully")
    sentiment: float = Field(..., description="Sentiment analysis of the feedback (-1.0 to 1.0)")
    mode_adjustments: Dict[str, float] = Field(..., description="Suggested mode distribution adjustments")
    reward_value: float = Field(..., description="Calculated reward value for learning")
    feedback_summary: str = Field(..., description="Summary of the feedback analysis")
    
    class Config:
        extra = "forbid"

# ===== FUNCTION TOOL RETURN MODELS =====

class ContextProcessingResult(BaseModel):
    """Result from context processing"""
    context_distribution: Dict[str, float] = Field(default_factory=dict)
    primary_context: str = "undefined"
    context_confidence: float = 0.0
    active_contexts: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    
    class Config:
        extra = "forbid"

class ModeUpdateResult(BaseModel):
    """Result from mode distribution update"""
    mode_distribution: Dict[str, float] = Field(default_factory=dict)
    primary_mode: str = "default"
    mode_changed: bool = False
    error: Optional[str] = None
    
    class Config:
        extra = "forbid"

class GoalAdditionResult(BaseModel):
    """Result from adding goals"""
    goals_added: bool = False
    added_goal_ids: List[str] = Field(default_factory=list)
    blended_goals: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    
    class Config:
        extra = "forbid"

class ResponseGuidanceResult(BaseModel):
    """Result from getting response guidance"""
    tone: str = "balanced"
    formality_level: float = 0.5
    verbosity: float = 0.5
    key_phrases: List[str] = Field(default_factory=list)
    avoid_phrases: List[str] = Field(default_factory=list)
    content_focus: List[str] = Field(default_factory=list)
    mode_description: str = "Blended mode"
    primary_mode: str = "default"
    active_modes: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    
    class Config:
        extra = "forbid"

class FeedbackAnalysisResult(BaseModel):
    """Result from feedback analysis"""
    sentiment: float = 0.0
    reward_value: float = 0.0
    positive_indicators: int = 0
    negative_indicators: int = 0
    feedback_summary: str = "Neutral feedback"
    
    class Config:
        extra = "forbid"

class ModeAdjustmentResult(BaseModel):
    """Suggested mode adjustments"""
    adjustments: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        extra = "forbid"

class ModeDistributionInfo(BaseModel):
    """Current mode distribution information"""
    mode_distribution: Dict[str, float] = Field(default_factory=dict)
    primary_mode: Optional[str] = None
    primary_weight: float = 0.0
    active_modes: List[Tuple[str, float]] = Field(default_factory=list)
    
    class Config:
        extra = "forbid"

class ModeParameters(BaseModel):
    """Parameters for a specific mode"""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "forbid"

class ConversationStyle(BaseModel):
    """Conversation style for a mode"""
    style: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "forbid"

class BlendedGuidanceElements(BaseModel):
    """Blended guidance elements from multiple modes"""
    tone: str = "balanced"
    formality_level: float = 0.5
    verbosity: float = 0.5
    key_phrases: List[str] = Field(default_factory=list)
    avoid_phrases: List[str] = Field(default_factory=list)
    content_focus: List[str] = Field(default_factory=list)
    weighted_parameters: Dict[str, float] = Field(default_factory=dict)
    active_modes: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        extra = "forbid"

class BlendedModeOutput(BaseModel):
    """Blended output from multiple systems"""
    context_processed: bool = True
    mode_updated: bool = True
    goals_added: bool = False
    mode_distribution: Dict[str, float] = Field(default_factory=dict)
    primary_mode: str = "default"
    active_modes: List[Tuple[str, float]] = Field(default_factory=list)
    context_result: ContextProcessingResult = Field(default_factory=ContextProcessingResult)
    mode_result: ModeUpdateResult = Field(default_factory=ModeUpdateResult)
    goals_result: GoalAdditionResult = Field(default_factory=GoalAdditionResult)
    
    class Config:
        extra = "forbid"

class CoherenceCheckResult(BaseModel):
    """Result of coherence check between context and mode"""
    coherence_score: float = 1.0
    is_coherent: bool = True
    misalignments: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        extra = "forbid"

class ModeIntegrationManager:
    """
    Manages the integration of the blended interaction mode system with
    other Nyx components. Serves as a central hub for mode-related
    functionality and coordinates between systems.
    """
    
    def __init__(self, nyx_brain=None):
        """
        Initialize the integration manager
        
        Args:
            nyx_brain: Reference to the main NyxBrain instance
        """
        self.brain = nyx_brain
        
        # Core mode components
        self.context_system = None
        self.mode_manager = None
        self.goal_selector = None
        self.input_processor = None
        
        # Connected Nyx components
        self.emotional_core = None
        self.identity_evolution = None
        self.goal_manager = None
        self.reward_system = None
        self.autobiographical_narrative = None
        
        # Initialize if brain reference provided
        if self.brain:
            self.initialize_from_brain()
            
        # Agent infrastructure
        self.agents_initialized = False
        self.main_agent = None
        self.feedback_agent = None
        self.guidance_agent = None
        self.blender_agent = None
        
        # Trace ID for linking traces
        self.trace_group_id = f"nyx_mode_{asyncio.get_event_loop().time()}"
        
        logger.info("ModeIntegrationManager initialized")
    
    def initialize_from_brain(self) -> bool:
        """Initialize components from the brain reference"""
        try:
            # Get references to existing components
            self.emotional_core = getattr(self.brain, 'emotional_core', None)
            self.identity_evolution = getattr(self.brain, 'identity_evolution', None)
            self.goal_manager = getattr(self.brain, 'goal_manager', None)
            self.reward_system = getattr(self.brain, 'reward_system', None)
            self.autobiographical_narrative = getattr(self.brain, 'autobiographical_narrative', None)
            
            # Use existing context_system from brain instead of creating new one
            self.context_system = getattr(self.brain, 'context_system', None)
            
            # Use existing mode_manager if already initialized
            existing_mode_manager = getattr(self.brain, 'mode_manager', None)
            if existing_mode_manager:
                self.mode_manager = existing_mode_manager
            else:
                # Only create if not exists (shouldn't happen if brain is properly initialized)
                logger.warning("mode_manager not found in brain, creating new instance")
                self.mode_manager = InteractionModeManager(
                    context_system=self.context_system,
                    emotional_core=self.emotional_core,
                    reward_system=self.reward_system,
                    goal_manager=self.goal_manager
                )
            
            # Initialize other components that aren't in brain
            self.goal_selector = GoalSelector(
                mode_manager=self.mode_manager,
                goal_manager=self.goal_manager
            )
            
            # Use existing input processor if available
            existing_processor = getattr(self.brain, 'conditioned_input_processor', None)
            if existing_processor:
                self.input_processor = existing_processor
                logger.debug("Using existing conditioned_input_processor from brain")
            else:
                # Create new processor with brain instance
                self.input_processor = BlendedInputProcessor(
                    brain=self.brain  # Pass brain instead of individual components
                )
                logger.debug("Created new BlendedInputProcessor with brain instance")
            
            # Update brain references
            if self.brain:
                setattr(self.brain, 'goal_selector', self.goal_selector)
                setattr(self.brain, 'mode_integration', self)
            
            logger.info("ModeIntegrationManager successfully initialized from brain")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing mode integration: {e}")
            return False
        
    async def initialize_agents(self) -> bool:
        """Initialize the agent infrastructure for mode integration"""
        if self.agents_initialized:
            return True
            
        # Create feedback agent
        self.feedback_agent = Agent(
            name="Mode_Feedback_Agent",
            instructions="""
            You analyze user feedback about interaction modes and determine:
            1. Sentiment and tone of the feedback
            2. How well the current mode distribution met user expectations
            3. What adjustments should be made to the mode distribution
            4. What reward signal should be sent to the learning system
            
            Generate structured feedback that can be used to adapt the
            interaction mode system for future interactions.
            """,
            tools=[
                self._create_analyze_feedback_tool(),
                self._create_suggest_mode_adjustments_tool(),
                self._create_calculate_feedback_reward_tool()
            ],
            model="gpt-4.1-nano",
            output_type=FeedbackOutput
        )
        
        # Create mode guidance agent
        self.guidance_agent = Agent(
            name="Mode_Guidance_Agent",
            instructions="""
            You generate comprehensive guidance for responding based on
            the current blended mode distribution.
            
            Your role is to:
            1. Extract key parameters from the mode distribution
            2. Create a coherent set of guidance that blends multiple modes
            3. Provide specific phrasing suggestions and tone recommendations
            4. Ensure the guidance reflects the proportional weight of each mode
            
            Create guidance that helps generate responses that naturally blend
            elements from all active modes in the current distribution.
            """,
            tools=[
                self._create_get_mode_distribution_tool(),
                self._create_get_mode_parameters_tool(),
                self._create_get_conversation_style_tool(),
                self._create_blend_guidance_elements_tool()
            ],
            model="gpt-4.1-nano",
            output_type=ModeGuidance
        )
        
        # Create blender agent
        self.blender_agent = Agent(
            name="Mode_Blender",
            instructions="""
            You analyze and integrate information from multiple mode-related systems
            to create coherent, blended experiences.
            
            Your role is to:
            1. Blend insights from context awareness, mode management, and goal systems
            2. Ensure all active modes are properly represented in the results
            3. Maintain coherence and natural flow in the blended output
            4. Provide comprehensive guidance for response generation
            
            Create blends that feel natural and integrated, rather than
            like separate modes stitched together.
            """,
            tools=[
                self._create_blend_mode_outputs_tool(),
                self._create_check_blend_coherence_tool(),
                self._create_extract_blended_guidance_tool()
            ],
            model="gpt-4.1-nano",
            output_type=BlendedModeOutput
        )
        
        # Create input validation guardrail
        async def validate_input(ctx, agent, input_data):
            """Validate the input for mode processing"""
            try:
                # Check if input is a string (message)
                if isinstance(input_data, str):
                    # Create ModeInput object
                    return GuardrailFunctionOutput(
                        output_info={"is_valid": True, "message": input_data},
                        tripwire_triggered=False
                    )
                
                # If dict or ModeInput object, check for required field
                if isinstance(input_data, dict) and "message" not in input_data:
                    return GuardrailFunctionOutput(
                        output_info={"is_valid": False, "error": "Missing required field 'message'"},
                        tripwire_triggered=True
                    )
                
                return GuardrailFunctionOutput(
                    output_info={"is_valid": True},
                    tripwire_triggered=False
                )
            except Exception as e:
                return GuardrailFunctionOutput(
                    output_info={"is_valid": False, "error": str(e)},
                    tripwire_triggered=True
                )
                
        input_guardrail = InputGuardrail(guardrail_function=validate_input)
        
        # Create the main mode agent
        self.main_agent = Agent(
            name="Mode_Integration_Manager",
            instructions="""
            You manage the integration of multiple mode-related systems for Nyx.
            
            Your role is to:
            1. Process user input through context awareness
            2. Update the mode distribution based on context
            3. Add appropriate goals based on the mode distribution
            4. Provide guidance for response generation
            
            Coordinate the various systems to create a coherent, natural
            blended experience that proportionally represents all active modes.
            """,
            tools=[
                self._create_process_context_tool(),
                self._create_update_mode_distribution_tool(),
                self._create_add_mode_goals_tool(),
                self._create_get_response_guidance_tool()
            ],
            handoffs=[
                handoff(self.feedback_agent, 
                       tool_name_override="process_feedback",
                       tool_description_override="Process feedback about interaction"),
                
                handoff(self.guidance_agent, 
                       tool_name_override="generate_guidance",
                       tool_description_override="Generate detailed response guidance"),
                
                handoff(self.blender_agent, 
                       tool_name_override="blend_mode_systems",
                       tool_description_override="Blend outputs from multiple mode systems")
            ],
            input_guardrails=[input_guardrail],
            model="gpt-4.1-nano",
            output_type=ModeOutput
        )
        
        self.agents_initialized = True
        logger.info("Mode integration agents initialized")
        return True
    
    async def process_message(self, message: str, user_id: str = "default", additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a user message through the blended mode system
        
        Args:
            message: User message
            user_id: User ID
            additional_context: Additional context information
            
        Returns:
            Processing results with mode, context, and guidance information
        """
        # Initialize agents if needed
        if not self.agents_initialized:
            await self.initialize_agents()
            
        results = {
            "context_processed": False,
            "mode_updated": False,
            "goals_added": False
        }
        
        try:
            # Use agent-based processing if available
            if self.main_agent:
                with trace(workflow_name="Mode_Integration", group_id=self.trace_group_id):
                    # Create input
                    input_data = {
                        "message": message,
                        "user_id": user_id
                    }
                    
                    # Add additional context if provided
                    if additional_context:
                        input_data["context"] = additional_context
                    
                    # Add current mode distribution if available
                    if self.mode_manager and hasattr(self.mode_manager, 'context'):
                        try:
                            input_data["current_mode_distribution"] = self.mode_manager.context.mode_distribution.dict()
                        except:
                            pass
                    
                    # Run main agent
                    agent_result = await Runner.run(
                        self.main_agent, 
                        input_data,
                        run_config={
                            "workflow_name": "Blended_Mode_Processing",
                            "trace_metadata": {"user_id": user_id, "message_length": len(message)}
                        }
                    )
                
                # Return the structured output
                return agent_result.final_output.dict()
            
            # Fall back to original implementation if agents not available
            # 1. Process through context system
            if self.context_system:
                context_result = await self.context_system.process_message(message)
                results["context_result"] = context_result
                results["context_processed"] = True
                
                # 2. Update interaction mode based on context
                if self.mode_manager:
                    mode_result = await self.mode_manager.update_interaction_mode(context_result)
                    results["mode_result"] = mode_result
                    results["mode_updated"] = True
                    
                    # Get current mode distribution
                    mode_distribution = self.mode_manager.context.mode_distribution.dict()
                    results["mode_distribution"] = mode_distribution
                    
                    # Get active modes
                    active_modes = self.mode_manager.context.mode_distribution.active_modes
                    results["active_modes"] = active_modes
                    
                    # Get primary mode
                    primary_mode, _ = self.mode_manager.context.mode_distribution.primary_mode
                    results["primary_mode"] = primary_mode
                    
                    # 3. Add appropriate goals from all active modes
                    if self.goal_selector:
                        blended_goals = await self.goal_selector.select_goals(mode_distribution)
                        
                        if self.goal_manager:
                            # Add goals to goal manager if available
                            for goal in blended_goals:
                                await self.goal_manager.add_goal(
                                    description=goal.get("description", ""),
                                    priority=goal.get("priority", 0.5),
                                    source="mode_integration",
                                    plan=goal.get("plan", [])
                                )
                                
                        results["goals_added"] = True
                        results["blended_goals"] = blended_goals
                    
                    # 4. Get response guidance
                    guidance = await self.mode_manager.get_current_mode_guidance()
                    results["guidance"] = guidance
            
            return results
            
        except Exception as e:
            logger.error(f"Error in mode integration processing: {e}")
            return {
                "error": str(e),
                **results
            }
    
    async def process_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process feedback about an interaction
        
        Args:
            feedback: Feedback information including success and user comments
            
        Returns:
            Feedback processing results
        """
        # Initialize agents if needed
        if not self.agents_initialized:
            await self.initialize_agents()
            
        try:
            # Use the feedback agent if available
            if self.feedback_agent:
                with trace(workflow_name="Mode_Feedback", group_id=self.trace_group_id):
                    # Get current mode distribution
                    mode_distribution = {}
                    if self.mode_manager and hasattr(self.mode_manager, 'context'):
                        try:
                            mode_distribution = self.mode_manager.context.mode_distribution.dict()
                        except:
                            pass
                    
                    # Create feedback input
                    feedback_input = FeedbackInput(
                        interaction_success=feedback.get("success", True),
                        user_feedback=feedback.get("feedback", ""),
                        mode_distribution=mode_distribution,
                        context=feedback.get("context", {})
                    )
                    
                    # Process feedback
                    result = await Runner.run(
                        self.feedback_agent, 
                        feedback_input.dict(),
                        run_config={
                            "workflow_name": "Feedback_Processing",
                            "trace_metadata": {"success": feedback.get("success", True)}
                        }
                    )
                    
                    # Extract results
                    feedback_result = result.final_output
                    
                    # Apply mode adjustments if suggested
                    if (feedback_result.mode_adjustments and 
                        self.mode_manager and 
                        hasattr(self.mode_manager, 'context')):
                        
                        # Get current distribution
                        current_dist = self.mode_manager.context.mode_distribution
                        
                        # Create adjusted distribution
                        adjusted_dist = ModeDistribution(**current_dist.dict())
                        
                        # Apply adjustments
                        for mode, adjustment in feedback_result.mode_adjustments.items():
                            # Only adjust if it exists
                            if hasattr(adjusted_dist, mode):
                                # Get current value
                                current_value = getattr(adjusted_dist, mode)
                                
                                # Calculate new value (ensuring it stays in range 0-1)
                                new_value = max(0.0, min(1.0, current_value + adjustment))
                                
                                # Set new value
                                setattr(adjusted_dist, mode, new_value)
                        
                        # Normalize if needed
                        if adjusted_dist.sum_weights() > 0:
                            adjusted_dist = adjusted_dist.normalize()
                            
                        # Apply the adjustment
                        self.mode_manager.context.mode_distribution = adjusted_dist
                        
                        # Update legacy fields
                        primary_mode, primary_weight = adjusted_dist.to_enum_and_confidence()
                        self.mode_manager.context.current_mode = primary_mode
                    
                    # Apply reward signal if available
                    if self.reward_system and hasattr(self.reward_system, 'process_reward_signal'):
                        # Create reward context
                        reward_context = {
                            "feedback": feedback.get("feedback", ""),
                            "success": feedback.get("success", True),
                            "mode_distribution": mode_distribution,
                            "sentiment": feedback_result.sentiment,
                            "feedback_summary": feedback_result.feedback_summary
                        }
                        
                        # Create and process reward signal
                        from nyx.core.reward_system import RewardSignal
                        
                        reward_signal = RewardSignal(
                            value=feedback_result.reward_value,
                            source="user_feedback",
                            context=reward_context
                        )
                        
                        await self.reward_system.process_reward_signal(reward_signal)
                    
                    return feedback_result.dict()
            
            # Fall back to simple processing
            interaction_success = feedback.get("success", True)
            user_feedback = feedback.get("feedback", "")
            
            # Record feedback
            if self.mode_manager:
                await self.mode_manager.record_mode_feedback(interaction_success, user_feedback)
                
            return {
                "feedback_processed": True,
                "interaction_success": interaction_success,
                "user_feedback": user_feedback
            }
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return {
                "feedback_processed": False,
                "error": str(e)
            }
    
    async def modify_response(self, response_text: str, mode_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Modify a response to match the current blended mode
        
        Args:
            response_text: Original response text
            mode_info: Optional mode information
            
        Returns:
            Modified response text
        """
        try:
            # Get current mode distribution
            mode_distribution = {}
            
            # First try to get from provided mode_info
            if mode_info and "mode_distribution" in mode_info:
                mode_distribution = mode_info["mode_distribution"]
            # Fall back to mode manager
            elif self.mode_manager and hasattr(self.mode_manager, 'context'):
                try:
                    mode_distribution = self.mode_manager.context.mode_distribution.dict()
                except:
                    pass
            
            # Use blended input processor if available
            if self.input_processor and mode_distribution:
                # Use the processor's blended response modification
                result = await self.input_processor.modify_blended_response(response_text, mode_distribution)
                
                # Extract modified text
                if isinstance(result, dict) and "modified_text" in result:
                    return result["modified_text"]
                elif hasattr(result, "modified_text"):
                    return result.modified_text
                else:
                    logger.warning("Blended response modification result does not contain modified_text")
            
            # Fall back to mode manager if needed
            if self.mode_manager:
                # Get guidance for response modification
                guidance = await self.mode_manager.get_current_mode_guidance()
                
                # Simple modification based on primary mode
                primary_mode, _ = self.mode_manager.context.mode_distribution.primary_mode
                
                # Get conversation style for primary mode
                style = self.mode_manager.get_conversation_style(primary_mode)
                patterns = self.mode_manager.get_vocalization_patterns(primary_mode)
                
                # Simple enhancement with key phrases if available
                if patterns and "key_phrases" in patterns and patterns["key_phrases"]:
                    import random
                    key_phrases = patterns["key_phrases"]
                    # Maybe add a phrase at beginning
                    if random.random() < 0.3:  # 30% chance
                        selected_phrase = random.choice(key_phrases)
                        if not response_text.startswith(selected_phrase):
                            response_text = f"{selected_phrase} {response_text}"
            
            return response_text
                
        except Exception as e:
            logger.error(f"Error modifying response: {e}")
            return response_text  # Return original if error
    
    async def get_response_guidance(self) -> Dict[str, Any]:
        """
        Get comprehensive guidance for response generation
        
        Returns:
            Guidance parameters for current mode distribution
        """
        # Initialize agents if needed
        if not self.agents_initialized:
            await self.initialize_agents()
            
        try:
            # Use the guidance agent if available
            if self.guidance_agent:
                with trace(workflow_name="Response_Guidance", group_id=self.trace_group_id):
                    # Get current mode distribution
                    mode_distribution = {}
                    if self.mode_manager and hasattr(self.mode_manager, 'context'):
                        try:
                            mode_distribution = self.mode_manager.context.mode_distribution.dict()
                        except:
                            pass
                    
                    # Create guidance prompt
                    guidance_prompt = f"""
                    Generate comprehensive guidance for the current mode distribution:
                    
                    MODE DISTRIBUTION: {mode_distribution}
                    
                    Create guidance that proportionally reflects all active modes
                    in the distribution, with detailed parameters and suggestions.
                    """
                    
                    # Generate guidance
                    result = await Runner.run(
                        self.guidance_agent, 
                        guidance_prompt,
                        run_config={
                            "workflow_name": "Guidance_Generation",
                            "trace_metadata": {"active_modes": [m for m, w in mode_distribution.items() if w >= 0.2]}
                        }
                    )
                    
                    # Return guidance
                    return result.final_output.dict()
            
            # Fall back to mode manager
            if self.mode_manager:
                return await self.mode_manager.get_current_mode_guidance()
                
            # Default empty guidance
            return {
                "tone": "balanced",
                "formality_level": 0.5,
                "verbosity": 0.5,
                "key_phrases": [],
                "avoid_phrases": [],
                "content_focus": [],
                "mode_description": "Default mode"
            }
            
        except Exception as e:
            logger.error(f"Error getting response guidance: {e}")
            return {
                "error": str(e),
                "tone": "balanced",
                "formality_level": 0.5,
                "verbosity": 0.5
            }
    
    # Agent function tools
    def _create_process_context_tool(self):
        """Create the process context tool with proper access to self"""
        @function_tool
        async def _process_context(ctx: RunContextWrapper, message: str) -> ContextProcessingResult:
            """
            Process message through context awareness system
            
            Args:
                message: User message
                
            Returns:
                Context processing results
            """
            if not self.context_system:
                return ContextProcessingResult(error="Context system not initialized")
            
            try:
                context_result = await self.context_system.process_message(message)
                return ContextProcessingResult(
                    context_distribution=context_result.get("context_distribution", {}),
                    primary_context=context_result.get("primary_context", "undefined"),
                    context_confidence=context_result.get("overall_confidence", 0.0),
                    active_contexts=context_result.get("active_contexts", [])
                )
            except Exception as e:
                logger.error(f"Error processing context: {e}")
                return ContextProcessingResult(error=str(e))
        
        return _process_context
    
    def _create_update_mode_distribution_tool(self):
        """Create the update mode distribution tool with proper access to self"""
        @function_tool
        async def _update_mode_distribution(ctx: RunContextWrapper, context_result: ContextProcessingResult) -> ModeUpdateResult:
            """
            Update mode distribution based on context
            
            Args:
                context_result: Result from context processing
                
            Returns:
                Mode update results
            """
            if not self.mode_manager:
                return ModeUpdateResult(error="Mode manager not initialized")
            
            try:
                # Convert back to dict for compatibility
                context_dict = context_result.dict()
                mode_result = await self.mode_manager.update_interaction_mode(context_dict)
                
                return ModeUpdateResult(
                    mode_distribution=mode_result.get("mode_distribution", {}),
                    primary_mode=mode_result.get("primary_mode", "default"),
                    mode_changed=mode_result.get("mode_changed", False)
                )
            except Exception as e:
                logger.error(f"Error updating mode: {e}")
                return ModeUpdateResult(error=str(e))
        
        return _update_mode_distribution

    def _create_add_mode_goals_tool(self):
        """Create the add mode goals tool with proper access to self"""
        @function_tool
        async def _add_mode_goals(ctx: RunContextWrapper, mode_distribution: Dict[str, float]) -> GoalAdditionResult:
            """
            Add goals based on mode distribution
            
            Args:
                mode_distribution: Current mode distribution
                
            Returns:
                Results of adding goals
            """
            if not self.goal_selector:
                return GoalAdditionResult(error="Goal selector not initialized")
                
            try:
                blended_goals = await self.goal_selector.select_goals(mode_distribution)
                
                if self.goal_manager:
                    # Add goals to goal manager if available
                    added_goals = []
                    for goal in blended_goals:
                        goal_id = await self.goal_manager.add_goal(
                            description=goal.get("description", ""),
                            priority=goal.get("priority", 0.5),
                            source="mode_integration",
                            plan=goal.get("plan", [])
                        )
                        if goal_id:
                            added_goals.append(goal_id)
                            
                    return GoalAdditionResult(
                        goals_added=len(added_goals) > 0,
                        added_goal_ids=added_goals,
                        blended_goals=blended_goals
                    )
                else:
                    return GoalAdditionResult(
                        goals_added=False,
                        blended_goals=blended_goals,
                        error="Goal manager not available to add goals"
                    )
                    
            except Exception as e:
                logger.error(f"Error adding mode goals: {e}")
                return GoalAdditionResult(error=str(e))
        
        return _add_mode_goals

    def _create_get_response_guidance_tool(self):
        """Create the get response guidance tool with proper access to self"""
        @function_tool
        async def _get_response_guidance(ctx: RunContextWrapper, mode_distribution: Dict[str, float]) -> ResponseGuidanceResult:
            """
            Get guidance for response generation based on mode distribution
            
            Args:
                mode_distribution: Current mode distribution
                
            Returns:
                Guidance for response generation
            """
            if not self.mode_manager:
                return ResponseGuidanceResult(error="Mode manager not initialized")
                
            try:
                # Get guidance from mode manager
                raw_guidance = await self.mode_manager.get_current_mode_guidance()
                
                # Extract key elements for response generation
                return ResponseGuidanceResult(
                    tone=raw_guidance.get("tone", "balanced"),
                    formality_level=raw_guidance.get("formality_level", 0.5),
                    verbosity=raw_guidance.get("verbosity", 0.5),
                    key_phrases=raw_guidance.get("key_phrases", []),
                    avoid_phrases=raw_guidance.get("avoid_phrases", []),
                    content_focus=raw_guidance.get("content_focus", []),
                    mode_description=raw_guidance.get("description", "Blended mode"),
                    primary_mode=raw_guidance.get("primary_mode", "default"),
                    active_modes=[m for m, w in mode_distribution.items() if w >= 0.2]
                )
                
            except Exception as e:
                logger.error(f"Error getting response guidance: {e}")
                return ResponseGuidanceResult(error=str(e))
        
        return _get_response_guidance

    def _create_analyze_feedback_tool(self):
        """Create the analyze feedback tool with proper access to self"""
        @function_tool
        async def _analyze_feedback(ctx: RunContextWrapper, 
                                    feedback: str, 
                                    interaction_success: bool) -> FeedbackAnalysisResult:
            """
            Analyze user feedback about interaction
            
            Args:
                feedback: User feedback text
                interaction_success: Whether interaction was successful
                
            Returns:
                Analysis of feedback
            """
            # Simple sentiment and keyword analysis
            positive_indicators = ["good", "great", "like", "helpful", "excellent", "perfect"]
            negative_indicators = ["bad", "wrong", "not helpful", "didn't like", "confused", "frustrated"]
            
            sentiment = 0.0  # Neutral
            positive_matches = 0
            negative_matches = 0
            
            if feedback:
                feedback_lower = feedback.lower()
                positive_matches = sum(1 for word in positive_indicators if word in feedback_lower)
                negative_matches = sum(1 for word in negative_indicators if word in feedback_lower)
                
                if positive_matches + negative_matches > 0:
                    sentiment = (positive_matches - negative_matches) / (positive_matches + negative_matches)
            
            # Calculate reward value
            base_reward = 0.3 if interaction_success else -0.2
            sentiment_modifier = sentiment * 0.2  # Scale sentiment to +/- 0.2
            reward_value = base_reward + sentiment_modifier
            
            return FeedbackAnalysisResult(
                sentiment=sentiment,
                reward_value=reward_value,
                positive_indicators=positive_matches,
                negative_indicators=negative_matches,
                feedback_summary="Positive feedback" if sentiment > 0.3 else 
                                ("Negative feedback" if sentiment < -0.3 else "Neutral feedback")
            )
        
        return _analyze_feedback

    def _create_suggest_mode_adjustments_tool(self):
        """Create the suggest mode adjustments tool with proper access to self"""
        @function_tool
        async def _suggest_mode_adjustments(ctx: RunContextWrapper,
                                            feedback_analysis: FeedbackAnalysisResult,
                                            mode_distribution: Dict[str, float]) -> ModeAdjustmentResult:
            """
            Suggest adjustments to mode distribution based on feedback
            
            Args:
                feedback_analysis: Analysis of feedback
                mode_distribution: Current mode distribution
                
            Returns:
                Suggested mode adjustments
            """
            # Extract sentiment
            sentiment = feedback_analysis.sentiment
            
            # Initialize adjustments
            adjustments = {}
            
            # Check for active modes (weight >= 0.2)
            active_modes = {mode: weight for mode, weight in mode_distribution.items() if weight >= 0.2}
            
            # Primary mode
            primary_mode = max(mode_distribution.items(), key=lambda x: x[1])[0] if mode_distribution else None
            
            # Base adjustment scales based on sentiment
            if sentiment > 0.5:  # Very positive
                # Reinforce current distribution - slight increase to primary mode
                if primary_mode:
                    adjustments[primary_mode] = 0.05
                    
            elif sentiment > 0.1:  # Somewhat positive
                # Minimal adjustment - slight increase to primary mode
                if primary_mode:
                    adjustments[primary_mode] = 0.03
                    
            elif sentiment < -0.5:  # Very negative
                # Significant adjustment away from primary mode
                if primary_mode:
                    adjustments[primary_mode] = -0.1
                    
                    # Find a secondary mode to increase
                    secondary_modes = sorted(
                        [(m, w) for m, w in active_modes.items() if m != primary_mode],
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    if secondary_modes:
                        secondary_mode = secondary_modes[0][0]
                        adjustments[secondary_mode] = 0.1
                    else:
                        # If no secondary mode, try a default mode
                        adjustments["friendly"] = 0.1
                        
            elif sentiment < -0.1:  # Somewhat negative
                # Moderate adjustment away from primary mode
                if primary_mode:
                    adjustments[primary_mode] = -0.05
                    
                    # Small increase to second highest mode
                    secondary_modes = sorted(
                        [(m, w) for m, w in active_modes.items() if m != primary_mode],
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    if secondary_modes:
                        secondary_mode = secondary_modes[0][0]
                        adjustments[secondary_mode] = 0.05
            
            return ModeAdjustmentResult(adjustments=adjustments)
        
        return _suggest_mode_adjustments

    def _create_calculate_feedback_reward_tool(self):
        """Create the calculate feedback reward tool with proper access to self"""
        @function_tool
        async def _calculate_feedback_reward(ctx: RunContextWrapper,
                                            feedback_analysis: FeedbackAnalysisResult,
                                            interaction_success: bool) -> float:
            """
            Calculate reward value for feedback
            
            Args:
                feedback_analysis: Analysis of feedback
                interaction_success: Whether interaction was successful
                
            Returns:
                Reward value
            """
            # Extract sentiment
            sentiment = feedback_analysis.sentiment
            
            # Base reward based on success
            base_reward = 0.3 if interaction_success else -0.2
            
            # Adjust based on sentiment
            sentiment_modifier = sentiment * 0.3  # Scale sentiment impact
            
            # Calculate final reward
            reward = base_reward + sentiment_modifier
            
            # Ensure in valid range (-1.0 to 1.0)
            reward = max(-1.0, min(1.0, reward))
            
            return reward
        
        return _calculate_feedback_reward

    def _create_get_mode_distribution_tool(self):
        """Create the get mode distribution tool with proper access to self"""
        @function_tool
        async def _get_mode_distribution(ctx: RunContextWrapper) -> ModeDistributionInfo:
            """
            Get the current mode distribution
            
            Returns:
                Current mode distribution information
            """
            if not self.mode_manager or not hasattr(self.mode_manager, 'context'):
                return ModeDistributionInfo()
                
            try:
                # Get mode distribution
                mode_distribution = self.mode_manager.context.mode_distribution.dict()
                
                # Get primary mode
                primary_mode, primary_weight = self.mode_manager.context.mode_distribution.primary_mode
                
                # Get active modes
                active_modes = self.mode_manager.context.mode_distribution.active_modes
                
                return ModeDistributionInfo(
                    mode_distribution=mode_distribution,
                    primary_mode=primary_mode,
                    primary_weight=primary_weight,
                    active_modes=active_modes
                )
            except Exception as e:
                logger.error(f"Error getting mode distribution: {e}")
                return ModeDistributionInfo()
        
        return _get_mode_distribution

    def _create_get_mode_parameters_tool(self):
        """Create the get mode parameters tool with proper access to self"""
        @function_tool
        async def _get_mode_parameters(ctx: RunContextWrapper, mode: str) -> ModeParameters:
            """
            Get parameters for a specific mode
            
            Args:
                mode: Mode to get parameters for
                
            Returns:
                Mode parameters
            """
            if not self.mode_manager:
                return ModeParameters()
                
            try:
                params = self.mode_manager.get_mode_parameters(mode)
                return ModeParameters(parameters=params)
            except Exception as e:
                logger.error(f"Error getting mode parameters: {e}")
                return ModeParameters()
        
        return _get_mode_parameters

    def _create_get_conversation_style_tool(self):
        """Create the get conversation style tool with proper access to self"""
        @function_tool
        async def _get_conversation_style(ctx: RunContextWrapper, mode: str) -> ConversationStyle:
            """
            Get conversation style for a specific mode
            
            Args:
                mode: Mode to get style for
                
            Returns:
                Conversation style
            """
            if not self.mode_manager:
                return ConversationStyle()
                
            try:
                style = self.mode_manager.get_conversation_style(mode)
                return ConversationStyle(style=style)
            except Exception as e:
                logger.error(f"Error getting conversation style: {e}")
                return ConversationStyle()
        
        return _get_conversation_style

    def _create_blend_guidance_elements_tool(self):
        """Create the blend guidance elements tool with proper access to self"""
        @function_tool
        async def _blend_guidance_elements(ctx: RunContextWrapper,
                                          mode_distribution: Dict[str, float]) -> BlendedGuidanceElements:
            """
            Blend guidance elements from multiple modes
            
            Args:
                mode_distribution: Mode distribution
                
            Returns:
                Blended guidance elements
            """
            # Initialize blended elements
            blended_elements = {
                "tone": [],
                "key_phrases": [],
                "avoid_phrases": [],
                "content_focus": []
            }
            
            # Get significant modes (weight >= 0.2)
            significant_modes = {mode: weight for mode, weight in mode_distribution.items() if weight >= 0.2}
            
            # Normalize significant mode weights
            total_weight = sum(significant_modes.values())
            normalized_weights = {mode: weight/total_weight for mode, weight in significant_modes.items()} if total_weight > 0 else {}
            
            # For each significant mode
            for mode, norm_weight in normalized_weights.items():
                try:
                    # Get conversation style
                    style_tool = self._create_get_conversation_style_tool()
                    style_result = await style_tool(ctx, mode)
                    style = style_result.style
                    
                    # Extract tone
                    if "tone" in style:
                        tone_elements = [t.strip() for t in style["tone"].split(",")] if isinstance(style["tone"], str) else []
                        
                        # Number of elements to include based on weight
                        num_elements = max(1, round(len(tone_elements) * norm_weight))
                        
                        # Add top elements
                        blended_elements["tone"].extend(tone_elements[:num_elements])
                    
                    # Extract topics to emphasize
                    if "topics_to_emphasize" in style:
                        topics = style["topics_to_emphasize"]
                        topic_elements = [t.strip() for t in topics.split(",")] if isinstance(topics, str) else []
                        
                        # Number of elements to include
                        num_elements = max(1, round(len(topic_elements) * norm_weight))
                        
                        # Add top elements
                        blended_elements["content_focus"].extend(topic_elements[:num_elements])
                    
                    # Extract topics to avoid
                    if "topics_to_avoid" in style:
                        avoid_topics = style["topics_to_avoid"]
                        avoid_elements = [t.strip() for t in avoid_topics.split(",")] if isinstance(avoid_topics, str) else []
                        
                        # Number of elements to include
                        num_elements = max(1, round(len(avoid_elements) * norm_weight))
                        
                        # Add top elements
                        blended_elements["avoid_phrases"].extend(avoid_elements[:num_elements])
                    
                    # Add key phrases
                    vocalization = self.mode_manager.get_vocalization_patterns(mode)
                    if vocalization and "key_phrases" in vocalization:
                        key_phrases = vocalization["key_phrases"]
                        
                        # Number of phrases to include
                        num_phrases = max(1, round(len(key_phrases) * norm_weight))
                        
                        # Add top phrases
                        blended_elements["key_phrases"].extend(key_phrases[:num_phrases])
                        
                except Exception as e:
                    logger.warning(f"Error blending guidance elements for mode {mode}: {e}")
                    continue
            
            # Remove duplicates while preserving order
            for element_type in blended_elements:
                seen = set()
                blended_elements[element_type] = [x for x in blended_elements[element_type] if x and not (x in seen or seen.add(x))]
            
            # Create blended tone string
            tone_string = ", ".join(blended_elements["tone"]) if blended_elements["tone"] else "balanced"
            
            # Calculate weighted parameters
            params = {}
            for mode, weight in normalized_weights.items():
                try:
                    mode_params_tool = self._create_get_mode_parameters_tool()
                    mode_params_result = await mode_params_tool(ctx, mode)
                    mode_params = mode_params_result.parameters
                    
                    for param_name, param_value in mode_params.items():
                        if isinstance(param_value, (int, float)):
                            if param_name not in params:
                                params[param_name] = 0.0
                                
                            # Add weighted contribution
                            params[param_name] += param_value * weight
                except:
                    continue
            
            return BlendedGuidanceElements(
                tone=tone_string,
                formality_level=params.get("formality", 0.5),
                verbosity=params.get("depth", 0.5),
                key_phrases=blended_elements["key_phrases"],
                avoid_phrases=blended_elements["avoid_phrases"],
                content_focus=blended_elements["content_focus"],
                weighted_parameters=params,
                active_modes={mode: weight for mode, weight in normalized_weights.items()}
            )
        
        return _blend_guidance_elements

    def _create_blend_mode_outputs_tool(self):
        """Create the blend mode outputs tool with proper access to self"""
        @function_tool
        async def _blend_mode_outputs(ctx: RunContextWrapper,
                                      context_result: ContextProcessingResult,
                                      mode_result: ModeUpdateResult,
                                      goals_result: GoalAdditionResult) -> BlendedModeOutput:
            """
            Blend outputs from multiple mode systems
            
            Args:
                context_result: Result from context system
                mode_result: Result from mode manager
                goals_result: Result from goal selector
                
            Returns:
                Blended output
            """
            # Get active modes
            active_modes = [(mode, weight) for mode, weight in mode_result.mode_distribution.items() if weight >= 0.2]
            
            return BlendedModeOutput(
                context_processed=True,
                mode_updated=True,
                goals_added=goals_result.goals_added,
                mode_distribution=mode_result.mode_distribution,
                primary_mode=mode_result.primary_mode,
                active_modes=active_modes,
                context_result=context_result,
                mode_result=mode_result,
                goals_result=goals_result
            )
        
        return _blend_mode_outputs

    def _create_check_blend_coherence_tool(self):
        """Create the check blend coherence tool with proper access to self"""
        @function_tool
        async def _check_blend_coherence(ctx: RunContextWrapper,
                                         context_distribution: Dict[str, float],
                                         mode_distribution: Dict[str, float]) -> CoherenceCheckResult:
            """
            Check coherence between context and mode distributions
            
            Args:
                context_distribution: Context distribution
                mode_distribution: Mode distribution
                
            Returns:
                Coherence analysis
            """
            # Check alignment between context and mode distributions
            # There should be a direct mapping between them
            
            correlation = 0.0
            misalignments = []
            
            # Map context types to corresponding mode types
            context_to_mode = {
                "dominant": "dominant",
                "casual": "friendly",
                "intellectual": "intellectual",
                "empathic": "compassionate",
                "playful": "playful",
                "creative": "creative",
                "professional": "professional"
            }
            
            # Check correlation for each context-mode pair
            total_pairs = 0
            for context, context_weight in context_distribution.items():
                if context in context_to_mode:
                    mode = context_to_mode[context]
                    mode_weight = mode_distribution.get(mode, 0.0)
                    
                    # Calculate weight difference
                    diff = abs(context_weight - mode_weight)
                    
                    # If significant difference, record misalignment
                    if diff > 0.2 and (context_weight >= 0.2 or mode_weight >= 0.2):
                        misalignments.append({
                            "context": context,
                            "context_weight": context_weight,
                            "mode": mode,
                            "mode_weight": mode_weight,
                            "difference": diff
                        })
                    
                    # Add to correlation
                    correlation += (1.0 - diff)
                    total_pairs += 1
            
            # Calculate average correlation
            if total_pairs > 0:
                correlation /= total_pairs
            else:
                correlation = 1.0  # Default if no pairs to check
                
            return CoherenceCheckResult(
                coherence_score=correlation,
                is_coherent=correlation >= 0.7,
                misalignments=misalignments
            )
        
        return _check_blend_coherence

    def _create_extract_blended_guidance_tool(self):
        """Create the extract blended guidance tool with proper access to self"""
        @function_tool
        async def _extract_blended_guidance(ctx: RunContextWrapper, mode_distribution: Dict[str, float]) -> ModeGuidance:
            """
            Extract response guidance from blended mode distribution
            
            Args:
                mode_distribution: Mode distribution
                
            Returns:
                Blended guidance
            """
            # Get elements based on mode distribution
            blend_tool = self._create_blend_guidance_elements_tool()
            elements = await blend_tool(ctx, mode_distribution)
            
            # Create guidance object
            return ModeGuidance(
                tone=elements.tone,
                formality_level=elements.formality_level,
                verbosity=elements.verbosity,
                key_phrases=elements.key_phrases,
                avoid_phrases=elements.avoid_phrases,
                content_focus=elements.content_focus,
                mode_description=f"Blended mode: {', '.join(elements.active_modes.keys())}",
                weighted_parameters=elements.weighted_parameters,
                active_modes=elements.active_modes
            )
        
        return _extract_blended_guidance
