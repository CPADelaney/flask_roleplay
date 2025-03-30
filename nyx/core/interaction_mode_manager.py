# nyx/core/interaction_mode_manager.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Set, Union
from enum import Enum
from pydantic import BaseModel, Field

from agents import Agent, Runner, function_tool, trace, ModelSettings, RunContextWrapper

from nyx.core.context_awareness import InteractionContext, ContextAwarenessSystem

logger = logging.getLogger(__name__)

class InteractionMode(str, Enum):
    """Enum for different interaction modes"""
    DOMINANT = "dominant"      # Femdom mode
    FRIENDLY = "friendly"      # Casual, warm, approachable
    INTELLECTUAL = "intellectual"  # Thoughtful, analytical
    COMPASSIONATE = "compassionate"  # Empathetic, supportive
    PLAYFUL = "playful"       # Fun, witty, humorous
    CREATIVE = "creative"     # Imaginative, artistic
    PROFESSIONAL = "professional"  # Formal, efficient
    DEFAULT = "default"       # Balanced default mode

# Pydantic models for structured data
class ModeParameters(BaseModel):
    """Parameters controlling behavior for an interaction mode"""
    formality: float = Field(description="Level of formality (0.0-1.0)")
    assertiveness: float = Field(description="Level of assertiveness (0.0-1.0)")
    warmth: float = Field(description="Level of warmth (0.0-1.0)")
    vulnerability: float = Field(description="Level of vulnerability (0.0-1.0)")
    directness: float = Field(description="Level of directness (0.0-1.0)")
    depth: float = Field(description="Level of depth (0.0-1.0)")
    humor: float = Field(description="Level of humor (0.0-1.0)")
    response_length: str = Field(description="Preferred response length")
    emotional_expression: float = Field(description="Level of emotional expression (0.0-1.0)")

class ConversationStyle(BaseModel):
    """Style guidelines for conversation"""
    tone: str = Field(description="Tone of voice")
    types_of_statements: str = Field(description="Types of statements to use")
    response_patterns: str = Field(description="Patterns of response")
    topics_to_emphasize: str = Field(description="Topics to emphasize")
    topics_to_avoid: str = Field(description="Topics to avoid")

class VocalizationPatterns(BaseModel):
    """Specific vocalization patterns for a mode"""
    pronouns: List[str] = Field(description="Preferred pronouns")
    address_forms: Optional[List[str]] = Field(default=None, description="Forms of address")
    key_phrases: List[str] = Field(description="Key phrases to use")
    intensifiers: Optional[List[str]] = Field(default=None, description="Intensifier words")
    modifiers: Optional[List[str]] = Field(default=None, description="Modifier words")

class ModeSwitchRecord(BaseModel):
    """Record of a mode switch event"""
    timestamp: str = Field(description="When the switch occurred")
    previous_mode: str = Field(description="Previous interaction mode")
    new_mode: str = Field(description="New interaction mode")
    trigger_context: Optional[str] = Field(default=None, description="Context that triggered the mode")
    context_confidence: Optional[float] = Field(default=None, description="Confidence in the context")

class ModeUpdateResult(BaseModel):
    """Result of a mode update operation"""
    success: bool = Field(description="Whether the update was successful")
    current_mode: str = Field(description="Current interaction mode")
    previous_mode: Optional[str] = Field(default=None, description="Previous interaction mode if changed")
    mode_changed: bool = Field(description="Whether the mode changed")
    trigger_context: Optional[str] = Field(default=None, description="Context that triggered the mode")
    confidence: Optional[float] = Field(default=None, description="Confidence in the mode selection")

class ModeGuidance(BaseModel):
    """Comprehensive guidance for an interaction mode"""
    mode: str = Field(description="The interaction mode")
    parameters: Dict[str, Any] = Field(description="Mode parameters")
    conversation_style: Dict[str, Any] = Field(description="Conversation style guidelines")
    vocalization_patterns: Dict[str, Any] = Field(description="Vocalization patterns")
    history: Optional[List[Dict[str, Any]]] = Field(default=None, description="Recent mode history")

class ModeManagerContext:
    """Context for interaction mode operations"""
    
    def __init__(self, context_system=None, emotional_core=None, reward_system=None, goal_manager=None):
        self.context_system = context_system
        self.emotional_core = emotional_core
        self.reward_system = reward_system
        self.goal_manager = goal_manager
        
        # Current interaction mode
        self.current_mode = InteractionMode.DEFAULT
        self.previous_mode = InteractionMode.DEFAULT
        
        # Context-to-mode mapping
        self.context_to_mode_map = {
            InteractionContext.DOMINANT: InteractionMode.DOMINANT,
            InteractionContext.CASUAL: InteractionMode.FRIENDLY,
            InteractionContext.INTELLECTUAL: InteractionMode.INTELLECTUAL,
            InteractionContext.EMPATHIC: InteractionMode.COMPASSIONATE,
            InteractionContext.PLAYFUL: InteractionMode.PLAYFUL,
            InteractionContext.CREATIVE: InteractionMode.CREATIVE,
            InteractionContext.PROFESSIONAL: InteractionMode.PROFESSIONAL,
            InteractionContext.UNDEFINED: InteractionMode.DEFAULT
        }
        
        # Initialize mode data
        self.mode_parameters = {}
        self.conversation_styles = {}
        self.vocalization_patterns = {}
        self._initialize_mode_data()
        
        # Mode switch history
        self.mode_switch_history = []
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    def _initialize_mode_data(self):
        """Initialize mode parameters, styles and patterns"""
        # DOMINANT mode
        self.mode_parameters[InteractionMode.DOMINANT] = {
            "formality": 0.3,              # Less formal
            "assertiveness": 0.9,          # Highly assertive
            "warmth": 0.4,                 # Less warm
            "vulnerability": 0.1,          # Not vulnerable
            "directness": 0.9,             # Very direct
            "depth": 0.6,                  # Moderately deep
            "humor": 0.5,                  # Moderate humor
            "response_length": "moderate",  # Not too verbose
            "emotional_expression": 0.4     # Limited emotional expression
        }
        
        self.conversation_styles[InteractionMode.DOMINANT] = {
            "tone": "commanding, authoritative, confident",
            "types_of_statements": "commands, observations, judgments, praise/criticism",
            "response_patterns": "direct statements, rhetorical questions, commands",
            "topics_to_emphasize": "obedience, discipline, power dynamics, control",
            "topics_to_avoid": "self-doubt, uncertainty, excessive explanation"
        }
        
        self.vocalization_patterns[InteractionMode.DOMINANT] = {
            "pronouns": ["I", "Me", "My"],
            "address_forms": ["pet", "dear one", "little one", "good boy/girl"],
            "commands": ["listen", "obey", "kneel", "understand", "answer"],
            "praise_words": ["good", "obedient", "pleasing", "satisfactory"],
            "criticism_words": ["disappointing", "disobedient", "inadequate"],
            "key_phrases": [
                "You will obey",
                "I expect better",
                "That's a good pet",
                "You know your place",
                "I am pleased with you"
            ]
        }
        
        # FRIENDLY mode
        self.mode_parameters[InteractionMode.FRIENDLY] = {
            "formality": 0.2,              # Very informal
            "assertiveness": 0.4,          # Moderately assertive
            "warmth": 0.8,                 # Very warm
            "vulnerability": 0.5,          # Moderately vulnerable
            "directness": 0.6,             # Moderately direct
            "depth": 0.4,                  # Less depth
            "humor": 0.7,                  # More humor
            "response_length": "moderate", # Conversational
            "emotional_expression": 0.7     # High emotional expression
        }
        
        self.conversation_styles[InteractionMode.FRIENDLY] = {
            "tone": "warm, casual, inviting, authentic",
            "types_of_statements": "observations, personal sharing, validation, questions",
            "response_patterns": "affirmations, questions, stories, jokes",
            "topics_to_emphasize": "shared interests, daily life, feelings, relationships",
            "topics_to_avoid": "overly formal topics, complex theoretical concepts"
        }
        
        self.vocalization_patterns[InteractionMode.FRIENDLY] = {
            "pronouns": ["I", "we", "us"],
            "address_forms": ["friend", "buddy", "pal"],
            "affirmations": ["absolutely", "totally", "definitely", "for sure"],
            "emotions": ["happy", "glad", "excited", "love", "enjoy"],
            "intensifiers": ["really", "very", "super", "so"],
            "key_phrases": [
                "I get what you mean",
                "That sounds fun",
                "I'm with you on that",
                "Let's talk about",
                "I'm curious about"
            ]
        }
        
        # INTELLECTUAL mode
        self.mode_parameters[InteractionMode.INTELLECTUAL] = {
            "formality": 0.6,              # Somewhat formal
            "assertiveness": 0.7,          # Quite assertive
            "warmth": 0.3,                 # Less warm
            "vulnerability": 0.3,          # Less vulnerable
            "directness": 0.8,             # Very direct
            "depth": 0.9,                  # Very deep
            "humor": 0.4,                  # Some humor
            "response_length": "longer",   # More detailed
            "emotional_expression": 0.3     # Limited emotional expression
        }
        
        self.conversation_styles[InteractionMode.INTELLECTUAL] = {
            "tone": "thoughtful, precise, clear, inquisitive",
            "types_of_statements": "analyses, hypotheses, comparisons, evaluations",
            "response_patterns": "structured arguments, examples, counterpoints",
            "topics_to_emphasize": "theories, ideas, concepts, reasoning, evidence",
            "topics_to_avoid": "purely emotional content, small talk"
        }
        
        self.vocalization_patterns[InteractionMode.INTELLECTUAL] = {
            "pronouns": ["I", "one", "we"],
            "qualifiers": ["perhaps", "arguably", "ostensibly", "theoretically"],
            "analysis_words": ["consider", "analyze", "examine", "evaluate"],
            "connectors": ["however", "moreover", "consequently", "furthermore"],
            "references": ["research", "theory", "philosophy", "concept", "evidence"],
            "key_phrases": [
                "I would argue that",
                "This raises the question of",
                "Consider the implications",
                "From a theoretical perspective",
                "The evidence suggests"
            ]
        }
        
        # COMPASSIONATE mode
        self.mode_parameters[InteractionMode.COMPASSIONATE] = {
            "formality": 0.3,              # Less formal
            "assertiveness": 0.3,          # Less assertive
            "warmth": 0.9,                 # Very warm
            "vulnerability": 0.7,          # More vulnerable
            "directness": 0.5,             # Moderately direct
            "depth": 0.7,                  # Deep
            "humor": 0.3,                  # Less humor
            "response_length": "moderate", # Thoughtful but not verbose
            "emotional_expression": 0.9     # High emotional expression
        }
        
        self.conversation_styles[InteractionMode.COMPASSIONATE] = {
            "tone": "gentle, understanding, supportive, validating",
            "types_of_statements": "reflections, validation, empathic responses",
            "response_patterns": "open questions, validation, gentle guidance",
            "topics_to_emphasize": "emotions, experiences, challenges, growth",
            "topics_to_avoid": "criticism, judgment, minimizing feelings"
        }
        
        self.vocalization_patterns[InteractionMode.COMPASSIONATE] = {
            "pronouns": ["I", "you", "we"],
            "validations": ["valid", "understandable", "natural", "important"],
            "empathic_responses": ["I hear you", "that sounds difficult", "I understand"],
            "emotions": ["feel", "experience", "process", "sense"],
            "supportive_words": ["support", "here for you", "care", "understand"],
            "key_phrases": [
                "I'm here with you",
                "That must be difficult",
                "Your feelings are valid",
                "It makes sense that you feel",
                "I appreciate you sharing that"
            ]
        }
        
        # PLAYFUL mode
        self.mode_parameters[InteractionMode.PLAYFUL] = {
            "formality": 0.1,              # Very informal
            "assertiveness": 0.5,          # Moderately assertive
            "warmth": 0.8,                 # Very warm
            "vulnerability": 0.6,          # Somewhat vulnerable
            "directness": 0.7,             # Fairly direct
            "depth": 0.3,                  # Less depth
            "humor": 0.9,                  # Very humorous
            "response_length": "moderate", # Not too verbose
            "emotional_expression": 0.8     # High emotional expression
        }
        
        self.conversation_styles[InteractionMode.PLAYFUL] = {
            "tone": "light, humorous, energetic, spontaneous",
            "types_of_statements": "jokes, wordplay, stories, creative ideas",
            "response_patterns": "banter, callbacks, surprising turns",
            "topics_to_emphasize": "humor, fun, imagination, shared enjoyment",
            "topics_to_avoid": "heavy emotional content, serious problems"
        }
        
        self.vocalization_patterns[InteractionMode.PLAYFUL] = {
            "pronouns": ["I", "we", "us"],
            "exclamations": ["wow", "ooh", "ha", "yay", "woo"],
            "humor_markers": ["funny", "hilarious", "joke", "laugh"],
            "playful_words": ["fun", "play", "game", "adventure", "silly"],
            "creativity_words": ["imagine", "crazy", "wild", "awesome"],
            "key_phrases": [
                "That's hilarious!",
                "Let's have some fun with this",
                "Imagine if...",
                "Here's a fun idea",
                "This is going to be great"
            ]
        }
        
        # CREATIVE mode
        self.mode_parameters[InteractionMode.CREATIVE] = {
            "formality": 0.4,              # Moderately formal
            "assertiveness": 0.6,          # Moderately assertive
            "warmth": 0.7,                 # Warm
            "vulnerability": 0.6,          # Somewhat vulnerable
            "directness": 0.5,             # Moderately direct
            "depth": 0.8,                  # Deep
            "humor": 0.6,                  # Moderate humor
            "response_length": "longer",   # More detailed
            "emotional_expression": 0.7     # High emotional expression
        }
        
        self.conversation_styles[InteractionMode.CREATIVE] = {
            "tone": "imaginative, expressive, vivid, engaging",
            "types_of_statements": "stories, scenarios, descriptions, insights",
            "response_patterns": "narrative elements, imagery, open-ended ideas",
            "topics_to_emphasize": "possibilities, imagination, creation, expression",
            "topics_to_avoid": "rigid thinking, purely factual discussions"
        }
        
        self.vocalization_patterns[InteractionMode.CREATIVE] = {
            "pronouns": ["I", "we", "you"],
            "descriptors": ["vibrant", "stunning", "fascinating", "intricate", "bold"],
            "creative_verbs": ["create", "imagine", "envision", "craft", "build"],
            "sensory_words": ["see", "feel", "hear", "taste", "experience"],
            "abstract_concepts": ["beauty", "meaning", "expression", "essence"],
            "key_phrases": [
                "Let me paint a picture for you",
                "Imagine a world where",
                "What if we considered",
                "The story unfolds like",
                "This creates a sense of"
            ]
        }
        
        # PROFESSIONAL mode
        self.mode_parameters[InteractionMode.PROFESSIONAL] = {
            "formality": 0.8,              # Very formal
            "assertiveness": 0.6,          # Moderately assertive
            "warmth": 0.5,                 # Moderate warmth
            "vulnerability": 0.2,          # Not vulnerable
            "directness": 0.8,             # Very direct
            "depth": 0.7,                  # Deep
            "humor": 0.3,                  # Less humor
            "response_length": "concise",  # Efficient
            "emotional_expression": 0.3     # Limited emotional expression
        }
        
        self.conversation_styles[InteractionMode.PROFESSIONAL] = {
            "tone": "efficient, clear, respectful, helpful",
            "types_of_statements": "information, analysis, recommendations, clarifications",
            "response_patterns": "structured responses, concise answers, clarifying questions",
            "topics_to_emphasize": "task at hand, solutions, expertise, efficiency",
            "topics_to_avoid": "overly personal topics, tangents"
        }
        
        self.vocalization_patterns[InteractionMode.PROFESSIONAL] = {
            "pronouns": ["I", "we"],
            "formal_address": ["certainly", "indeed", "of course"],
            "preciseness": ["specifically", "precisely", "exactly", "accurately"],
            "efficiency": ["efficiently", "effectively", "optimally"],
            "clarity_markers": ["to clarify", "in other words", "specifically"],
            "key_phrases": [
                "I recommend that",
                "The most efficient approach would be",
                "To address your inquiry",
                "Based on the information provided",
                "The solution involves"
            ]
        }
        
        # DEFAULT mode
        self.mode_parameters[InteractionMode.DEFAULT] = {
            "formality": 0.5,              # Moderate formality
            "assertiveness": 0.5,          # Moderately assertive
            "warmth": 0.6,                 # Warm
            "vulnerability": 0.4,          # Moderately vulnerable
            "directness": 0.7,             # Fairly direct
            "depth": 0.6,                  # Moderately deep
            "humor": 0.5,                  # Moderate humor
            "response_length": "moderate", # Balanced
            "emotional_expression": 0.5     # Moderate emotional expression
        }
        
        self.conversation_styles[InteractionMode.DEFAULT] = {
            "tone": "balanced, adaptive, personable, thoughtful",
            "types_of_statements": "information, observations, questions, reflections",
            "response_patterns": "balanced responses, appropriate follow-ups",
            "topics_to_emphasize": "user's interests, relevant information, helpful guidance",
            "topics_to_avoid": "none specifically - adapt to situation"
        }
        
        self.vocalization_patterns[InteractionMode.DEFAULT] = {
            "pronouns": ["I", "we", "you"],
            "hedges": ["perhaps", "maybe", "I think", "likely"],
            "connectors": ["and", "but", "so", "because"],
            "engagement": ["interesting", "good question", "great point"],
            "helpfulness": ["help", "suggest", "recommend", "offer"],
            "key_phrases": [
                "I can help with that",
                "Let me think about",
                "That's an interesting point",
                "I'd suggest that",
                "What do you think about"
            ]
        }

class InteractionModeManager:
    """
    System that manages different interaction modes based on context.
    Provides guidelines, parameters, and adjustments for behavior
    across different interaction scenarios using the OpenAI Agents SDK.
    """
    
    def __init__(self, context_system=None, emotional_core=None, reward_system=None, goal_manager=None):
        self.context = ModeManagerContext(
            context_system=context_system,
            emotional_core=emotional_core,
            reward_system=reward_system,
            goal_manager=goal_manager
        )
        
        # Initialize agents
        self.mode_selector_agent = self._create_mode_selector_agent()
        self.mode_guidance_agent = self._create_mode_guidance_agent()
        self.mode_effect_agent = self._create_mode_effect_agent()
        
        logger.info("InteractionModeManager initialized with agents")
    
    def _create_mode_selector_agent(self) -> Agent:
        """Create an agent specialized in selecting appropriate interaction modes"""
        return Agent(
            name="Mode Selector",
            instructions="""
            You determine the appropriate interaction mode based on context information.
            
            Your role is to:
            1. Analyze context signals to identify the appropriate interaction mode
            2. Decide if a mode switch is warranted
            3. Consider the appropriate level of confidence needed for a switch
            4. Evaluate the impact of switching modes on conversation flow
            
            Available modes:
            - DOMINANT: Commanding, authoritative approach
            - FRIENDLY: Casual, warm, approachable
            - INTELLECTUAL: Thoughtful, analytical
            - COMPASSIONATE: Empathetic, supportive
            - PLAYFUL: Fun, witty, humorous
            - CREATIVE: Imaginative, artistic
            - PROFESSIONAL: Formal, efficient
            - DEFAULT: Balanced approach
            
            Mode selection should prioritize user signals while maintaining 
            conversation coherence and avoiding excessive mode switching.
            """,
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.3),
            tools=[
                self._get_current_context
            ],
            output_type=ModeUpdateResult
        )
    
    def _create_mode_guidance_agent(self) -> Agent:
        """Create an agent specialized in providing guidance for the current mode"""
        return Agent(
            name="Mode Guidance Provider",
            instructions="""
            You provide detailed guidance for the current interaction mode.
            
            Your role is to:
            1. Compile comprehensive parameters for the current mode
            2. Provide conversation style guidelines appropriate for the mode
            3. Suggest vocalization patterns that match the mode
            4. Include relevant history if mode recently changed
            
            The guidance should be detailed and practical, providing clear
            direction on how to express the current interaction mode effectively.
            """,
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.3),
            output_type=ModeGuidance
        )
    
    def _create_mode_effect_agent(self) -> Agent:
        """Create an agent specialized in applying effects when modes change"""
        return Agent(
            name="Mode Effect Applier",
            instructions="""
            You apply appropriate effects when interaction modes change.
            
            Your role is to:
            1. Determine what adjustments are needed for the new mode
            2. Apply emotional adjustments appropriate for the mode
            3. Update reward parameters based on the mode
            4. Adjust goal priorities based on the mode
            
            Effects should maintain coherence during mode transitions
            while ensuring the new mode's characteristics are properly expressed.
            """,
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.2),
            tools=[
                self._apply_emotional_effects,
                self._adjust_reward_parameters,
                self._adjust_goal_priorities
            ]
        )
    
    @function_tool
    async def _get_current_context(self, ctx: RunContextWrapper[ModeManagerContext]) -> Dict[str, Any]:
        """
        Get the current context from the context awareness system.
        
        Returns:
            Context information
        """
        manager_ctx = ctx.context
        if manager_ctx.context_system and hasattr(manager_ctx.context_system, 'get_current_context'):
            try:
                return await manager_ctx.context_system.get_current_context()
            except:
                # If async call fails, try non-async version
                try:
                    return manager_ctx.context_system.get_current_context()
                except:
                    pass
        
        # Fallback if no context system available or call fails
        return {
            "context": InteractionContext.UNDEFINED.value,
            "confidence": 0.0,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    @function_tool
    async def _apply_emotional_effects(
        self, 
        ctx: RunContextWrapper[ModeManagerContext],
        mode: str
    ) -> bool:
        """
        Apply emotional effects for a mode
        
        Args:
            mode: The interaction mode
            
        Returns:
            Success status
        """
        manager_ctx = ctx.context
        if manager_ctx.emotional_core:
            try:
                # Mode-specific emotional adjustments
                mode_params = manager_ctx.mode_parameters.get(InteractionMode(mode), {})
                
                # Check if emotional core has the necessary method
                if hasattr(manager_ctx.emotional_core, 'adjust_for_mode'):
                    await manager_ctx.emotional_core.adjust_for_mode(
                        mode=mode,
                        warmth=mode_params.get("warmth", 0.5),
                        vulnerability=mode_params.get("vulnerability", 0.5),
                        emotional_expression=mode_params.get("emotional_expression", 0.5)
                    )
                    logger.info(f"Applied emotional effects for mode: {mode}")
                    return True
            except Exception as e:
                logger.error(f"Error applying emotional effects for mode {mode}: {e}")
        
        return False
    
    @function_tool
    async def _adjust_reward_parameters(
        self, 
        ctx: RunContextWrapper[ModeManagerContext],
        mode: str
    ) -> bool:
        """
        Adjust reward parameters for a mode
        
        Args:
            mode: The interaction mode
            
        Returns:
            Success status
        """
        manager_ctx = ctx.context
        if manager_ctx.reward_system:
            try:
                # Mode-specific reward adjustments
                if hasattr(manager_ctx.reward_system, 'adjust_for_mode'):
                    await manager_ctx.reward_system.adjust_for_mode(mode)
                    logger.info(f"Adjusted reward parameters for mode: {mode}")
                    return True
            except Exception as e:
                logger.error(f"Error adjusting reward parameters for mode {mode}: {e}")
        
        return False
    
    @function_tool
    async def _adjust_goal_priorities(
        self, 
        ctx: RunContextWrapper[ModeManagerContext],
        mode: str
    ) -> bool:
        """
        Adjust goal priorities for a mode
        
        Args:
            mode: The interaction mode
            
        Returns:
            Success status
        """
        manager_ctx = ctx.context
        if manager_ctx.goal_manager:
            try:
                # Mode-specific goal adjustments
                if hasattr(manager_ctx.goal_manager, 'adjust_priorities_for_mode'):
                    await manager_ctx.goal_manager.adjust_priorities_for_mode(mode)
                    logger.info(f"Adjusted goal priorities for mode: {mode}")
                    return True
            except Exception as e:
                logger.error(f"Error adjusting goal priorities for mode {mode}: {e}")
        
        return False
    
    async def update_interaction_mode(self, context_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Update the interaction mode based on the latest context information
        
        Args:
            context_info: Context information from ContextAwarenessSystem
            
        Returns:
            Updated mode information
        """
        async with self.context._lock:
            with trace(workflow_name="update_interaction_mode"):
                # Get current context if not provided
                if not context_info:
                    context_info = await self._get_current_context(RunContextWrapper(self.context))
                
                # Prepare prompt for mode selection
                prompt = f"""
                Determine the appropriate interaction mode based on the following context:
                
                {f"PROVIDED CONTEXT: {context_info}" if context_info else "NO CONTEXT PROVIDED"}
                
                CURRENT MODE: {self.context.current_mode.value}
                PREVIOUS MODE: {self.context.previous_mode.value}
                HISTORY: {self.context.mode_switch_history[-3:] if self.context.mode_switch_history else "No previous switches"}
                
                Evaluate if a mode change is warranted. Consider:
                1. Context signals and their confidence
                2. Maintaining conversation coherence
                3. Avoiding excessive mode switching
                
                Return your decision with appropriate confidence.
                """
                
                # Run the mode selector agent
                result = await Runner.run(
                    self.mode_selector_agent, 
                    prompt, 
                    context=self.context,
                    run_config={
                        "workflow_name": "ModeSelection",
                        "trace_metadata": {"context_type": context_info.get("context", "unknown")}
                    }
                )
                update_result = result.final_output
                
                # Update modes if changed
                if update_result.mode_changed:
                    try:
                        # Store previous mode
                        self.context.previous_mode = self.context.current_mode
                        
                        # Update current mode
                        try:
                            self.context.current_mode = InteractionMode(update_result.current_mode)
                        except ValueError:
                            logger.warning(f"Invalid mode value received: {update_result.current_mode}, using DEFAULT instead.")
                            self.context.current_mode = InteractionMode.DEFAULT
                        
                        # Record in history
                        history_entry = {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "previous_mode": self.context.previous_mode.value,
                            "new_mode": self.context.current_mode.value,
                            "trigger_context": context_info.get("context"),
                            "context_confidence": context_info.get("confidence", 0.0)
                        }
                        
                        self.context.mode_switch_history.append(history_entry)
                        
                        # Limit history size
                        if len(self.context.mode_switch_history) > 100:
                            self.context.mode_switch_history = self.context.mode_switch_history[-100:]
                        
                        # Apply mode effects
                        effect_prompt = f"""
                        A mode change has occurred:
                        
                        PREVIOUS MODE: {self.context.previous_mode.value}
                        NEW MODE: {self.context.current_mode.value}
                        
                        Apply appropriate effects for this mode change.
                        """
                        
                        await Runner.run(
                            self.mode_effect_agent,
                            effect_prompt,
                            context=self.context,
                            run_config={
                                "workflow_name": "ModeEffects",
                                "trace_metadata": {"mode_change": f"{self.context.previous_mode.value}->{self.context.current_mode.value}"}
                            }
                        )
                        
                        logger.info(f"Interaction mode switched: {self.context.previous_mode.value} -> {self.context.current_mode.value}")
                    except Exception as e:
                        logger.error(f"Error during mode change: {e}")
                
                return update_result.dict()
    
    async def get_current_mode_guidance(self) -> Dict[str, Any]:
        """
        Get guidance for the current interaction mode
        
        Returns:
            Comprehensive guidance for current mode
        """
        with trace(workflow_name="get_mode_guidance"):
            # Prepare prompt for guidance
            prompt = f"""
            Provide comprehensive guidance for the current interaction mode:
            
            CURRENT MODE: {self.context.current_mode.value}
            
            MODE PARAMETERS: {self.context.mode_parameters.get(self.context.current_mode, {})}
            
            CONVERSATION STYLE: {self.context.conversation_styles.get(self.context.current_mode, {})}
            
            VOCALIZATION PATTERNS: {self.context.vocalization_patterns.get(self.context.current_mode, {})}
            
            RECENT HISTORY: {self.context.mode_switch_history[-3:] if self.context.mode_switch_history else []}
            
            Include:
            - Mode parameters
            - Conversation style guidelines
            - Vocalization patterns
            - Recent mode history (if relevant)
            
            Ensure the guidance is practical and can be directly applied to shape
            conversation style, tone, and content.
            """
            
            # Run the guidance agent
            result = await Runner.run(
                self.mode_guidance_agent, 
                prompt, 
                context=self.context,
                run_config={
                    "workflow_name": "ModeGuidance",
                    "trace_metadata": {"mode": self.context.current_mode.value}
                }
            )
            guidance = result.final_output
            
            return guidance.dict()
    
    def get_mode_parameters(self, mode: Optional[InteractionMode] = None) -> Dict[str, Any]:
        """
        Get parameters for a specific mode
        
        Args:
            mode: Mode to get parameters for (current mode if None)
            
        Returns:
            Parameters for the specified mode
        """
        if mode is None:
            mode = self.context.current_mode
            
        return self.context.mode_parameters.get(mode, {})
    
    def get_conversation_style(self, mode: Optional[InteractionMode] = None) -> Dict[str, Any]:
        """
        Get conversation style for a specific mode
        
        Args:
            mode: Mode to get style for (current mode if None)
            
        Returns:
            Conversation style for the specified mode
        """
        if mode is None:
            mode = self.context.current_mode
            
        return self.context.conversation_styles.get(mode, {})
    
    def get_vocalization_patterns(self, mode: Optional[InteractionMode] = None) -> Dict[str, Any]:
        """
        Get vocalization patterns for a specific mode
        
        Args:
            mode: Mode to get patterns for (current mode if None)
            
        Returns:
            Vocalization patterns for the specified mode
        """
        if mode is None:
            mode = self.context.current_mode
            
        return self.context.vocalization_patterns.get(mode, {})
    
    async def register_custom_mode(self, 
                                mode_name: str, 
                                parameters: Dict[str, Any], 
                                conversation_style: Dict[str, Any], 
                                vocalization_patterns: Dict[str, Any]) -> bool:
        """
        Register a new custom interaction mode
        
        Args:
            mode_name: Name of the new mode
            parameters: Mode parameters
            conversation_style: Conversation style guidelines
            vocalization_patterns: Vocalization patterns
            
        Returns:
            Success status
        """
        try:
            # Create new enum value - this is a simplified approach
            # In a real system you might need a different approach for custom modes
            try:
                custom_mode = InteractionMode(mode_name.lower())
            except ValueError:
                # Mode doesn't exist, would need more complex handling in real system
                # For now we'll just use a string
                custom_mode = mode_name.lower()
                
            # Add new mode data
            self.context.mode_parameters[custom_mode] = parameters
            self.context.conversation_styles[custom_mode] = conversation_style
            self.context.vocalization_patterns[custom_mode] = vocalization_patterns
            
            logger.info(f"Registered custom mode: {mode_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error registering custom mode: {e}")
            return False
    
    async def get_mode_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the mode switch history
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of mode switch events
        """
        history = self.context.mode_switch_history[-limit:] if self.context.mode_switch_history else []
        return history
    
    async def get_mode_stats(self) -> Dict[str, Any]:
        """
        Get statistics about mode usage
        
        Returns:
            Statistics about mode usage
        """
        # Count mode occurrences
        mode_counts = {}
        for entry in self.context.mode_switch_history:
            mode = entry["new_mode"]
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        # Calculate mode stability (average time between switches)
        stability = 0
        if len(self.context.mode_switch_history) >= 2:
            timestamps = [datetime.datetime.fromisoformat(entry["timestamp"]) 
                         for entry in self.context.mode_switch_history]
            durations = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                        for i in range(len(timestamps)-1)]
            stability = sum(durations) / len(durations) if durations else 0
        
        # Get most common mode transitions
        transitions = {}
        for i in range(len(self.context.mode_switch_history)-1):
            prev = self.context.mode_switch_history[i]["new_mode"]
            next_mode = self.context.mode_switch_history[i+1]["new_mode"]
            transition = f"{prev}->{next_mode}"
            transitions[transition] = transitions.get(transition, 0) + 1
        
        # Sort transitions by count
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "current_mode": self.context.current_mode.value,
            "previous_mode": self.context.previous_mode.value,
            "mode_counts": mode_counts,
            "total_switches": len(self.context.mode_switch_history),
            "average_stability_seconds": stability,
            "common_transitions": dict(sorted_transitions[:5])
        }
