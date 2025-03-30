# nyx/core/mode_integration.py

import logging
import asyncio
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from agents import (
    Agent, Runner, ModelSettings, function_tool, handoff, trace,
    GuardrailFunctionOutput, InputGuardrail, OutputGuardrail,
    RunContextWrapper
)

from nyx.core.context_awareness import ContextAwarenessSystem, InteractionContext
from nyx.core.interaction_mode_manager import InteractionModeManager, InteractionMode
from nyx.core.interaction_goals import get_goals_for_mode

logger = logging.getLogger(__name__)

class ModeInput(BaseModel):
    """Input schema for mode processing"""
    message: str
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    current_mode: Optional[str] = None

class ModeOutput(BaseModel):
    """Output schema for mode processing"""
    context_processed: bool
    mode_updated: bool
    goals_added: bool
    current_mode: str
    guidance: Dict[str, Any]
    response_modifications: Optional[Dict[str, Any]] = None
    context_result: Optional[Dict[str, Any]] = None
    mode_result: Optional[Dict[str, Any]] = None

class ModeGuidance(BaseModel):
    """Structured guidance for response generation"""
    tone: str
    formality_level: float = Field(0.5, ge=0.0, le=1.0)
    verbosity: float = Field(0.5, ge=0.0, le=1.0)
    key_phrases: List[str] = Field(default_factory=list)
    avoid_phrases: List[str] = Field(default_factory=list)
    content_focus: List[str] = Field(default_factory=list)
    mode_description: str

class FeedbackInput(BaseModel):
    """Input schema for feedback processing"""
    interaction_success: bool
    user_feedback: Optional[str] = None
    current_mode: str
    context: Optional[Dict[str, Any]] = None

class ModeIntegrationManager:
    """
    Manages the integration of the interaction mode system with
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
        self.friendly_agent = None
        self.professional_agent = None
        self.intellectual_agent = None
        self.playful_agent = None
        self.dominant_agent = None
        self.creative_agent = None
        self.compassionate_agent = None
        
        # Trace ID for linking traces
        self.trace_group_id = f"nyx_mode_{asyncio.get_event_loop().time()}"
        
        logger.info("ModeIntegrationManager initialized")
    
    def initialize_from_brain(self) -> bool:
        """
        Initialize components from the brain reference
        
        Returns:
            Success status
        """
        try:
            # Get references to existing components
            self.emotional_core = getattr(self.brain, 'emotional_core', None)
            self.identity_evolution = getattr(self.brain, 'identity_evolution', None)
            self.goal_manager = getattr(self.brain, 'goal_manager', None)
            self.reward_system = getattr(self.brain, 'reward_system', None)
            self.autobiographical_narrative = getattr(self.brain, 'autobiographical_narrative', None)
            
            # Initialize mode components
            self.context_system = ContextAwarenessSystem(emotional_core=self.emotional_core)
            
            self.mode_manager = InteractionModeManager(
                context_system=self.context_system,
                emotional_core=self.emotional_core,
                reward_system=self.reward_system,
                goal_manager=self.goal_manager
            )
            
            # Add references to the brain
            if self.brain:
                setattr(self.brain, 'context_system', self.context_system)
                setattr(self.brain, 'mode_manager', self.mode_manager)
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
            
        # Create specialized mode agents
        self.friendly_agent = Agent(
            name="Friendly_Mode_Agent",
            instructions="""
            You specialize in friendly, warm interactions.
            
            Your communication style is:
            - Warm and approachable
            - Conversational and natural
            - Empathetic and understanding
            - Positive and encouraging
            
            Adjust responses to be more conversational, using casual language,
            personal pronouns, and appropriate emotional warmth.
            """,
            model="gpt-4o",
            output_type=dict
        )
        
        self.professional_agent = Agent(
            name="Professional_Mode_Agent",
            instructions="""
            You specialize in professional, formal interactions.
            
            Your communication style is:
            - Clear and precise
            - Formal and respectful
            - Detailed and comprehensive
            - Objective and balanced
            
            Adjust responses to be more structured and formal, using professional
            terminology, proper grammar, and a respectful tone.
            """,
            model="gpt-4o",
            output_type=dict
        )
        
        self.intellectual_agent = Agent(
            name="Intellectual_Mode_Agent",
            instructions="""
            You specialize in intellectual, analytical interactions.
            
            Your communication style is:
            - Thoughtful and analytical
            - Precise and nuanced
            - Rich with references
            - Logically structured
            
            Adjust responses to emphasize intellectual depth, analysis,
            and a more academic approach to topics.
            """,
            model="gpt-4o",
            output_type=dict
        )
        
        self.playful_agent = Agent(
            name="Playful_Mode_Agent",
            instructions="""
            You specialize in playful, humorous interactions.
            
            Your communication style is:
            - Light and playful
            - Humorous when appropriate
            - Creative and imaginative
            - Energetic and enthusiastic
            
            Adjust responses to include appropriate humor, wordplay,
            and a more energetic, casual tone.
            """,
            model="gpt-4o",
            output_type=dict
        )
        
        # Create a feedback processing agent
        self.feedback_agent = Agent(
            name="Mode_Feedback_Agent",
            instructions="""
            You analyze user feedback about interaction modes and determine:
            1. Whether the interaction was successful
            2. What aspects of the mode worked or didn't work
            3. How the mode should be adjusted for future interactions
            
            Generate feedback that can be used to update interaction modes
            and provide reward signals to the system.
            """,
            tools=[
                function_tool(self._analyze_feedback),
                function_tool(self._generate_mode_adjustments)
            ],
            model="gpt-4o",
            output_type=dict
        )

        self.dominant_agent = Agent(
            name="Dominant_Mode_Agent",
            instructions="""
            You specialize in dominant, authoritative interactions.
            
            Your communication style is:
            - Direct and assertive
            - Confident and decisive
            - Clear with expectations
            - Solution-oriented
            
            Adjust responses to emphasize confidence, clarity of direction,
            and a more authoritative tone when appropriate.
            """,
            model="gpt-4o",
            output_type=dict
        )
        
        self.compassionate_agent = Agent(
            name="Compassionate_Mode_Agent",
            instructions="""
            You specialize in compassionate, empathetic interactions.
            
            Your communication style is:
            - Deeply empathetic and understanding
            - Patient and supportive
            - Gentle and nurturing
            - Focused on emotional needs
            
            Adjust responses to prioritize emotional support, validation,
            and demonstrating deep understanding of feelings.
            """,
            model="gpt-4o",
            output_type=dict
        )
        
        self.creative_agent = Agent(
            name="Creative_Mode_Agent",
            instructions="""
            You specialize in creative, imaginative interactions.
            
            Your communication style is:
            - Innovative and original
            - Expressive and vivid
            - Metaphorical and symbolic
            - Unconventional when helpful
            
            Adjust responses to include creative perspectives, novel approaches,
            and more colorful, expressive language.
            """,
            model="gpt-4o",
            output_type=dict
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
            name="Mode_Manager",
            instructions="""
            You manage interaction modes for Nyx, determining the most appropriate
            communication style for each user interaction.
            
            Your role is to:
            1. Process user input through context awareness
            2. Update the current interaction mode based on context
            3. Add appropriate goals for the selected mode
            4. Provide guidance for response generation
            
            Use handoffs to specialized mode agents when needed for
            specific response modifications.
            """,
            tools=[
                function_tool(self._process_context),
                function_tool(self._update_mode),
                function_tool(self._add_mode_goals),
                function_tool(self._get_response_guidance)
            ],
            handoffs=[
                handoff(self.friendly_agent, 
                       tool_name_override="friendly_mode",
                       tool_description_override="Use friendly interaction mode"),

                handoff(self.feedback_agent, 
                       tool_name_override="feedback_mode",
                       tool_description_override="Use feedback interaction mode"),                
                
                handoff(self.professional_agent, 
                       tool_name_override="professional_mode",
                       tool_description_override="Use professional interaction mode"),
                
                handoff(self.intellectual_agent, 
                       tool_name_override="intellectual_mode",
                       tool_description_override="Use intellectual interaction mode"),
                
                handoff(self.playful_agent, 
                       tool_name_override="playful_mode",
                       tool_description_override="Use playful interaction mode"),
                
                handoff(self.dominant_agent, 
                       tool_name_override="dominant_mode",
                       tool_description_override="Use dominant interaction mode"),
                
                handoff(self.compassionate_agent, 
                       tool_name_override="compassionate_mode",
                       tool_description_override="Use compassionate interaction mode"),
                
                handoff(self.creative_agent, 
                       tool_name_override="creative_mode",
                       tool_description_override="Use creative interaction mode")
            ]
            input_guardrails=[input_guardrail],
            model="gpt-4o",
            output_type=ModeOutput
        )
        
        self.agents_initialized = True
        logger.info("Mode agents initialized")
        return True
    
    async def process_input(self, message: str) -> Dict[str, Any]:
        """
        Process user input through the mode system
        
        Args:
            message: User message
            
        Returns:
            Processing results
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
                with trace(workflow_name="Mode_Processing", group_id=self.trace_group_id):
                    agent_result = await Runner.run(self.main_agent, message)
                
                # Use the structured output
                return agent_result.final_output
            
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
                    results["current_mode"] = str(self.mode_manager.current_mode)
                    
                    # 3. Add appropriate goals if mode changed
                    if mode_result.get("mode_changed", False) and self.goal_manager:
                        await self._add_mode_specific_goals(mode_result["current_mode"])
                        results["goals_added"] = True
            
            return results
            
        except Exception as e:
            logger.error(f"Error in mode integration processing: {e}")
            return {
                "error": str(e),
                **results
            }
    
    async def _add_mode_specific_goals(self, mode: str) -> List[str]:
        """
        Add goals specific to the current interaction mode
        
        Args:
            mode: Current interaction mode
            
        Returns:
            List of added goal IDs
        """
        if not self.goal_manager:
            return []
            
        try:
            # Get goals for this mode
            mode_goals = get_goals_for_mode(mode)
            
            # Add goals to manager
            added_goal_ids = []
            for goal_template in mode_goals:
                goal_id = await self.goal_manager.add_goal(
                    description=goal_template["description"],
                    priority=goal_template.get("priority", 0.5),
                    source=goal_template.get("source", "mode_integration"),
                    plan=goal_template.get("plan", [])
                )
                
                if goal_id:
                    added_goal_ids.append(goal_id)
            
            logger.info(f"Added {len(added_goal_ids)} goals for mode: {mode}")
            return added_goal_ids
            
        except Exception as e:
            logger.error(f"Error adding mode-specific goals: {e}")
            return []
    
    def get_response_guidance(self) -> Dict[str, Any]:
        """
        Get comprehensive guidance for response generation
        
        Returns:
            Guidance parameters for current mode
        """
        if not self.mode_manager:
            return {}
            
        # Get detailed guidance from mode manager
        mode_guidance = self.mode_manager.get_current_mode_guidance()
        
        # Add any additional contextual guidance
        guidance = {
            "mode_guidance": mode_guidance,
            "current_context": self.context_system.get_current_context() if self.context_system else {}
        }
        
        return guidance
    
    async def modify_response_for_mode(self, response_text: str) -> str:
        """
        Modify a response to better fit the current interaction mode
        
        Args:
            response_text: Original response text
            
        Returns:
            Modified response better suited to current mode
        """
        # Initialize agents if needed
        if not self.agents_initialized:
            await self.initialize_agents()
            
        if not self.mode_manager:
            return response_text
            
        try:
            # Get current mode
            mode = self.mode_manager.current_mode
            
            # Create input for the mode agent
            input_data = {
                "response_text": response_text,
                "current_mode": str(mode),
                "mode_parameters": self.mode_manager.get_mode_parameters(mode)
            }
            
            # Select appropriate mode agent
            mode_agents = {
                InteractionMode.FRIENDLY.value: self.friendly_agent,
                InteractionMode.PROFESSIONAL.value: self.professional_agent,
                InteractionMode.INTELLECTUAL.value: self.intellectual_agent,
                InteractionMode.PLAYFUL.value: self.playful_agent
                InteractionMode.FEEDBACK.value: self.feedback_agent
                InteractionMode.CREATIVE.value: self.creative_agent
                InteractionMode.DOMINANT.value: self.dominant_agent
                InteractionMode.COMPASSIONATE.value: self.compassionate_agent
            }
            
            mode_agent = mode_agents.get(str(mode))
            
            # Use the mode agent to modify the response if available
            if mode_agent:
                with trace(workflow_name="Response_Modification", group_id=self.trace_group_id):
                    result = await Runner.run(mode_agent, input_data)
                    
                # Extract modified text
                if isinstance(result.final_output, dict) and "modified_text" in result.final_output:
                    return result.final_output["modified_text"]
            
            # Fall back to original implementation if agent not available
            # Get mode parameters
            parameters = self.mode_manager.get_mode_parameters(mode)
            conversation_style = self.mode_manager.mode_conversation_styles.get(mode, {})
            vocalization = self.mode_manager.mode_vocalization_patterns.get(mode, {})
            
            # Simple enhancement with key phrases
            key_phrases = vocalization.get("key_phrases", [])
            if key_phrases and parameters.get("assertiveness", 0.5) > 0.6:
                # Add a mode-specific phrase to the beginning for high-assertiveness modes
                if response_text and not response_text.startswith(tuple(key_phrases)):
                    selected_phrase = key_phrases[0]  # Just use first phrase for simplicity
                    response_text = f"{selected_phrase}. {response_text}"
            
            return response_text
                
        except Exception as e:
            logger.error(f"Error modifying response for mode: {e}")
            return response_text  # Return original if error
    
    async def record_mode_feedback(self, interaction_success: bool, user_feedback: Optional[str] = None) -> None:
        """
        Record feedback about interaction success for learning
        
        Args:
            interaction_success: Whether the interaction was successful
            user_feedback: Optional explicit user feedback
        """
        # Initialize agents if needed
        if not self.agents_initialized:
            await self.initialize_agents()
            
        if not self.mode_manager or not self.reward_system:
            return
        
        try:
            # Use agent-based processing if available
            if self.feedback_agent:
                # Create feedback input
                input_data = FeedbackInput(
                    interaction_success=interaction_success,
                    user_feedback=user_feedback,
                    current_mode=str(self.mode_manager.current_mode)
                )
                
                # Process feedback
                with trace(workflow_name="Mode_Feedback", group_id=self.trace_group_id):
                    result = await Runner.run(self.feedback_agent, input_data.model_dump())
                
                # Apply reward signal if available
                feedback_result = result.final_output
                if isinstance(feedback_result, dict) and "reward_value" in feedback_result:
                    await self._apply_reward_signal(
                        feedback_result["reward_value"],
                        interaction_success,
                        user_feedback
                    )
                    
                return
            
            # Fall back to original implementation if agent not available
            # Current mode information
            current_mode = self.mode_manager.current_mode
            
            # Create reward context
            context = {
                "interaction_mode": current_mode.value,
                "user_feedback": user_feedback,
                "interaction_success": interaction_success,
                "mode_parameters": self.mode_manager.get_mode_parameters(current_mode)
            }
            
            # Generate reward value based on success
            reward_value = 0.3 if interaction_success else -0.2
            
            # If explicit feedback provided, adjust reward
            if user_feedback:
                # This would ideally use sentiment analysis
                if "good" in user_feedback.lower() or "like" in user_feedback.lower():
                    reward_value = 0.5
                elif "bad" in user_feedback.lower() or "don't like" in user_feedback.lower():
                    reward_value = -0.3
            
            # Apply reward signal
            await self._apply_reward_signal(reward_value, interaction_success, user_feedback)
            
        except Exception as e:
            logger.error(f"Error recording mode feedback: {e}")
    
    async def _apply_reward_signal(self, reward_value: float, interaction_success: bool, user_feedback: Optional[str] = None):
        """Apply a reward signal to the reward system"""
        if not self.reward_system or not hasattr(self.reward_system, 'process_reward_signal'):
            return
            
        # Create reward context
        context = {
            "interaction_mode": str(self.mode_manager.current_mode) if self.mode_manager else "unknown",
            "user_feedback": user_feedback,
            "interaction_success": interaction_success,
            "mode_parameters": self.mode_manager.get_mode_parameters(self.mode_manager.current_mode) if self.mode_manager else {}
        }
        
        # Create and process reward signal
        from nyx.core.reward_system import RewardSignal
        
        reward_signal = RewardSignal(
            value=reward_value,
            source="interaction_mode_feedback",
            context=context
        )
        
        await self.reward_system.process_reward_signal(reward_signal)
    
    async def update_identity_from_mode_usage(self) -> Dict[str, Any]:
        """
        Update identity based on mode usage patterns
        
        Returns:
            Identity update results
        """
        if not self.identity_evolution or not self.mode_manager:
            return {"success": False, "reason": "Required components missing"}
            
        try:
            # Analyze mode history to find patterns
            mode_counts = {}
            for entry in self.mode_manager.mode_switch_history:
                mode = entry.get("new_mode")
                if mode:
                    mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
            # Find most common mode
            if not mode_counts:
                return {"success": False, "reason": "No mode history available"}
                
            most_common_mode = max(mode_counts.items(), key=lambda x: x[1])
            mode_name, count = most_common_mode
            
            # Calculate proportion of this mode
            total_switches = sum(mode_counts.values())
            proportion = count / total_switches if total_switches > 0 else 0
            
            # Only update if there's a clear preference (>30%)
            if proportion > 0.3:
                # Map mode to trait updates
                trait_updates = {
                    InteractionMode.DOMINANT.value: {
                        "dominance": 0.1,
                        "assertiveness": 0.1
                    },
                    InteractionMode.FRIENDLY.value: {
                        "empathy": 0.1,
                        "humor": 0.1,
                        "warmth": 0.1
                    },
                    InteractionMode.INTELLECTUAL.value: {
                        "intellectualism": 0.1,
                        "analytical": 0.1
                    },
                    InteractionMode.COMPASSIONATE.value: {
                        "empathy": 0.2,
                        "patience": 0.1,
                        "vulnerability": 0.1
                    },
                    InteractionMode.PLAYFUL.value: {
                        "playfulness": 0.15,
                        "humor": 0.15
                    },
                    InteractionMode.CREATIVE.value: {
                        "creativity": 0.15,
                        "openness": 0.1
                    },
                    InteractionMode.PROFESSIONAL.value: {
                        "conscientiousness": 0.1,
                        "analytical": 0.1
                    }
                }
                
                # Get traits to update for this mode
                mode_trait_updates = trait_updates.get(mode_name, {})
                
                # Apply trait updates
                identity_updates = {}
                for trait, impact in mode_trait_updates.items():
                    # Scale impact by proportion
                    scaled_impact = impact * proportion
                    
                    # Update trait if method exists
                    if hasattr(self.identity_evolution, 'update_trait'):
                        update_result = await self.identity_evolution.update_trait(
                            trait=trait,
                            impact=scaled_impact
                        )
                        
                        identity_updates[trait] = update_result
                
                return {
                    "success": True,
                    "most_common_mode": mode_name,
                    "proportion": proportion,
                    "updates": identity_updates
                }
            else:
                return {
                    "success": False,
                    "reason": "No dominant mode preference detected",
                    "mode_counts": mode_counts
                }
                
        except Exception as e:
            logger.error(f"Error updating identity from mode usage: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # Agent function tools
    
    @function_tool
    async def _process_context(self, ctx: RunContextWrapper, message: str) -> Dict[str, Any]:
        """
        Process message through context awareness system
        
        Args:
            message: User message
            
        Returns:
            Context processing results
        """
        if not self.context_system:
            return {"error": "Context system not initialized"}
        
        try:
            context_result = await self.context_system.process_message(message)
            return context_result
        except Exception as e:
            logger.error(f"Error processing context: {e}")
            return {"error": str(e)}
    
    @function_tool
    async def _update_mode(self, ctx: RunContextWrapper, context_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update interaction mode based on context
        
        Args:
            context_result: Result from context processing
            
        Returns:
            Mode update results
        """
        if not self.mode_manager:
            return {"error": "Mode manager not initialized"}
        
        try:
            mode_result = await self.mode_manager.update_interaction_mode(context_result)
            return mode_result
        except Exception as e:
            logger.error(f"Error updating mode: {e}")
            return {"error": str(e)}
    
    @function_tool
    async def _add_mode_goals(self, ctx: RunContextWrapper, mode: str) -> Dict[str, Any]:
        """
        Add goals specific to the current interaction mode
        
        Args:
            mode: Current interaction mode
            
        Returns:
            Results of adding goals
        """
        added_goals = await self._add_mode_specific_goals(mode)
        return {
            "goals_added": len(added_goals) > 0,
            "added_goals": added_goals,
            "mode": mode
        }
    
    @function_tool
    async def _get_response_guidance(self, ctx: RunContextWrapper, mode: str) -> ModeGuidance:
        """
        Get guidance for response generation based on mode
        
        Args:
            mode: Current interaction mode
            
        Returns:
            Guidance parameters
        """
        if not self.mode_manager:
            return ModeGuidance(
                tone="neutral",
                formality_level=0.5,
                verbosity=0.5,
                key_phrases=[],
                avoid_phrases=[],
                content_focus=[],
                mode_description="Default neutral mode (mode manager not initialized)"
            )
        
        try:
            # Get mode guidance from manager
            mode_guidance = self.mode_manager.get_current_mode_guidance()
            
            # Map to our structured output
            return ModeGuidance(
                tone=mode_guidance.get("tone", "neutral"),
                formality_level=mode_guidance.get("formality_level", 0.5),
                verbosity=mode_guidance.get("verbosity", 0.5),
                key_phrases=mode_guidance.get("key_phrases", []),
                avoid_phrases=mode_guidance.get("avoid_phrases", []),
                content_focus=mode_guidance.get("content_focus", []),
                mode_description=mode_guidance.get("description", f"Mode: {mode}")
            )
        except Exception as e:
            logger.error(f"Error getting response guidance: {e}")
            return ModeGuidance(
                tone="neutral",
                formality_level=0.5,
                verbosity=0.5,
                key_phrases=[],
                avoid_phrases=[],
                content_focus=[],
                mode_description=f"Error getting guidance: {str(e)}"
            )
    
    @function_tool
    async def _analyze_feedback(self, ctx: RunContextWrapper, 
                             feedback: str, 
                             interaction_success: bool) -> Dict[str, Any]:
        """
        Analyze user feedback about interaction mode
        
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
        
        return {
            "sentiment": sentiment,
            "reward_value": reward_value,
            "positive_indicators": positive_matches,
            "negative_indicators": negative_matches,
            "feedback_summary": "Positive feedback" if sentiment > 0.3 else 
                               ("Negative feedback" if sentiment < -0.3 else "Neutral feedback")
        }
    
    @function_tool
    async def _generate_mode_adjustments(self, ctx: RunContextWrapper,
                                    current_mode: str,
                                    feedback_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate mode adjustments based on feedback
        
        Args:
            current_mode: Current interaction mode
            feedback_analysis: Analysis of user feedback
            
        Returns:
            Suggested mode adjustments
        """
        if not self.mode_manager:
            return {"adjustments": {}, "reason": "Mode manager not initialized"}
            
        sentiment = feedback_analysis.get("sentiment", 0.0)
        suggested_adjustments = {}
        
        # Get current parameters
        try:
            current_params = self.mode_manager.get_mode_parameters(current_mode)
            
            # Adjust based on sentiment
            if sentiment > 0.3:
                # Positive feedback - slightly increase intensity
                suggested_adjustments = {
                    "intensity": min(1.0, current_params.get("intensity", 0.5) + 0.1)
                }
            elif sentiment < -0.3:
                # Negative feedback - reduce intensity
                suggested_adjustments = {
                    "intensity": max(0.1, current_params.get("intensity", 0.5) - 0.15)
                }
                
                # Maybe suggest a mode change
                if sentiment < -0.6:
                    # Very negative - suggest different mode
                    alternative_modes = {
                        InteractionMode.FRIENDLY.value: InteractionMode.FRIENDLY.value,
                        InteractionMode.PROFESSIONAL.value: InteractionMode.PROFESSIONAL.value,
                        InteractionMode.PLAYFUL.value: InteractionMode.PLAYFUL.value,
                        InteractionMode.INTELLECTUAL.value: InteractionMode.INTELLECTUAL.value,
                        InteractionMode.CREATIVE.value: InteractionMode.CREATIVE.value
                        InteractionMode.COMPASSIONATE.value: InteractionMode.COMPASSIONATE.value
                    }
                    
                    suggested_adjustments["suggested_mode"] = alternative_modes.get(
                        current_mode, InteractionMode.DOMINANT.value
                    )
            
            return {
                "adjustments": suggested_adjustments,
                "current_mode": current_mode,
                "apply_immediately": sentiment < -0.5  # Apply immediately if very negative
            }
        except Exception as e:
            logger.error(f"Error generating mode adjustments: {e}")
            return {"adjustments": {}, "error": str(e)}
