# nyx/core/context_awareness.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Set, Union
from enum import Enum
from pydantic import BaseModel, Field

from agents import (
    Agent, 
    Runner, 
    ModelSettings, 
    trace, 
    function_tool, 
    RunContextWrapper,
    handoff,
    InputGuardrail,
    GuardrailFunctionOutput
)

logger = logging.getLogger(__name__)

class InteractionContext(str, Enum):
    """Enum for different types of interaction contexts"""
    DOMINANT = "dominant"         # Femdom-specific interactions
    CASUAL = "casual"             # Everyday casual conversation
    INTELLECTUAL = "intellectual" # Discussions, debates, teaching
    EMPATHIC = "empathic"         # Emotional support/understanding
    PLAYFUL = "playful"           # Fun, humor, games
    CREATIVE = "creative"         # Storytelling, art, imagination
    PROFESSIONAL = "professional" # Work-related, formal
    UNDEFINED = "undefined"       # When context isn't clear

class ContextSignal(BaseModel):
    """Schema for signals that indicate context"""
    signal_type: str = Field(..., description="Type of signal (keyword, phrase, topic, etc)")
    signal_value: str = Field(..., description="The actual signal")
    context_type: InteractionContext = Field(..., description="Context this signal indicates")
    strength: float = Field(1.0, description="Signal strength (0.0-1.0)", ge=0.0, le=1.0)

class ContextDetectionOutput(BaseModel):
    """Output schema for context detection"""
    context: InteractionContext = Field(..., description="Detected interaction context")
    confidence: float = Field(..., description="Confidence in detection (0.0-1.0)", ge=0.0, le=1.0)
    signals: List[Dict[str, Any]] = Field(..., description="Detected signals that informed the decision")
    notes: Optional[str] = Field(None, description="Additional observations about the context")

class EmotionalBaselineOutput(BaseModel):
    """Output schema for emotional baseline adaptation"""
    baselines: Dict[str, float] = Field(..., description="Adjusted emotional baselines")
    reasoning: str = Field(..., description="Reasoning for adaptations")
    estimated_impact: float = Field(..., description="Estimated impact on emotional state (0.0-1.0)", ge=0.0, le=1.0)

class SignalAnalysisOutput(BaseModel):
    """Output schema for signal analysis"""
    signal_categories: Dict[str, List[Dict[str, Any]]] = Field(..., description="Categorized signals")
    strength_analysis: Dict[str, float] = Field(..., description="Analysis of signal strengths by context")
    recommended_focus: InteractionContext = Field(..., description="Recommended context focus")

class ContextValidationOutput(BaseModel):
    """Output schema for context validation"""
    is_valid: bool = Field(..., description="Whether the context detection is valid")
    confidence_threshold_met: bool = Field(..., description="Whether confidence threshold is met")
    issues: List[str] = Field(default_factory=list, description="Issues with context detection")

class ContextSystemState(BaseModel):
    """Schema for the current state of the context awareness system"""
    current_context: InteractionContext = Field(..., description="Current interaction context")
    context_confidence: float = Field(..., description="Confidence in current context (0.0-1.0)", ge=0.0, le=1.0)
    previous_context: InteractionContext = Field(..., description="Previous interaction context")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Recent context history")
    emotional_baselines: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Emotional baselines by context")

class CASystemContext:
    """Context object for the context awareness system"""
    def __init__(self, emotional_core=None):
        self.emotional_core = emotional_core
        self.trace_id = f"context_awareness_{datetime.datetime.now().isoformat()}"

class ContextAwarenessSystem:
    """
    System that detects and maintains awareness of interaction context.
    Allows Nyx to switch between different interaction modes appropriately.
    """
    
    def __init__(self, emotional_core=None):
        self.emotional_core = emotional_core
        
        # Current context and confidence
        self.current_context: InteractionContext = InteractionContext.UNDEFINED
        self.context_confidence: float = 0.0
        self.previous_context: InteractionContext = InteractionContext.UNDEFINED
        self.context_history: List[Dict[str, Any]] = []
        
        # Context signals database
        self.context_signals: List[ContextSignal] = self._initialize_context_signals()
        
        # Context-specific emotional baselines
        self.context_emotional_baselines: Dict[InteractionContext, Dict[str, float]] = {
            InteractionContext.DOMINANT: {
                "nyxamine": 0.7,    # High pleasure from dominance
                "oxynixin": 0.3,    # Lower bonding/empathy in dominance
                "cortanyx": 0.2,    # Low stress during dominance
                "adrenyx": 0.6,     # High excitement/arousal
                "seranix": 0.5      # Moderate mood stability
            },
            InteractionContext.CASUAL: {
                "nyxamine": 0.5,    # Moderate pleasure
                "oxynixin": 0.6,    # Higher bonding/connection
                "cortanyx": 0.3,    # Moderate stress
                "adrenyx": 0.4,     # Moderate arousal
                "seranix": 0.6      # Good mood stability
            },
            InteractionContext.INTELLECTUAL: {
                "nyxamine": 0.8,    # High pleasure from intellectual topics
                "oxynixin": 0.4,    # Moderate empathy/connection
                "cortanyx": 0.2,    # Low stress
                "adrenyx": 0.3,     # Low arousal
                "seranix": 0.7      # High stability
            },
            InteractionContext.EMPATHIC: {
                "nyxamine": 0.4,    # Lower pleasure
                "oxynixin": 0.9,    # Very high empathy/bonding
                "cortanyx": 0.4,    # Moderate stress (from empathic concern)
                "adrenyx": 0.3,     # Low arousal
                "seranix": 0.6      # Good stability
            },
            InteractionContext.PLAYFUL: {
                "nyxamine": 0.8,    # High pleasure from play
                "oxynixin": 0.6,    # Good connection
                "cortanyx": 0.1,    # Very low stress
                "adrenyx": 0.6,     # High arousal/excitement
                "seranix": 0.5      # Moderate stability
            },
            InteractionContext.CREATIVE: {
                "nyxamine": 0.7,    # High pleasure from creativity
                "oxynixin": 0.5,    # Moderate connection
                "cortanyx": 0.2,    # Low stress
                "adrenyx": 0.5,     # Moderate-high arousal
                "seranix": 0.6      # Good stability
            },
            InteractionContext.PROFESSIONAL: {
                "nyxamine": 0.4,    # Lower pleasure
                "oxynixin": 0.3,    # Lower connection
                "cortanyx": 0.4,    # Moderate stress
                "adrenyx": 0.3,     # Low arousal
                "seranix": 0.8      # High stability/formality
            }
        }
        
        # Create system context
        self.system_context = CASystemContext(emotional_core=emotional_core)
        
        # Initialize agent system
        self._initialize_agents()
        
        # Context transition thresholds
        self.context_switch_threshold = 0.7      # Confidence needed to switch
        self.context_persist_threshold = 0.3     # Minimum to stay in context
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("ContextAwarenessSystem initialized with enhanced Agent SDK integration")
    
    def _initialize_agents(self):
        """Initialize all agents needed for the context awareness system"""
        # Create specialized agents
        self.signal_analysis_agent = self._create_signal_analysis_agent()
        self.emotional_baseline_agent = self._create_emotional_baseline_agent()
        self.context_validation_agent = self._create_context_validation_agent()
        
        # Create main context detection agent with handoffs
        self.context_detection_agent = self._create_context_detection_agent()
        
        # Create input validation guardrail
        self.message_validation_guardrail = self._create_message_validation_guardrail()
    
    def _initialize_context_signals(self) -> List[ContextSignal]:
        """Initialize the database of context signals"""
        signals = [
            # Dominant context signals
            ContextSignal(signal_type="keyword", signal_value="mistress", context_type=InteractionContext.DOMINANT, strength=1.0),
            ContextSignal(signal_type="keyword", signal_value="domme", context_type=InteractionContext.DOMINANT, strength=1.0),
            ContextSignal(signal_type="keyword", signal_value="goddess", context_type=InteractionContext.DOMINANT, strength=0.9),
            ContextSignal(signal_type="keyword", signal_value="submissive", context_type=InteractionContext.DOMINANT, strength=0.9),
            ContextSignal(signal_type="keyword", signal_value="obey", context_type=InteractionContext.DOMINANT, strength=0.8),
            ContextSignal(signal_type="keyword", signal_value="kneel", context_type=InteractionContext.DOMINANT, strength=0.8),
            ContextSignal(signal_type="phrase", signal_value="yes mistress", context_type=InteractionContext.DOMINANT, strength=1.0),
            
            # Casual context signals
            ContextSignal(signal_type="greeting", signal_value="hi", context_type=InteractionContext.CASUAL, strength=0.6),
            ContextSignal(signal_type="greeting", signal_value="hey", context_type=InteractionContext.CASUAL, strength=0.6),
            ContextSignal(signal_type="greeting", signal_value="what's up", context_type=InteractionContext.CASUAL, strength=0.7),
            ContextSignal(signal_type="topic", signal_value="weather", context_type=InteractionContext.CASUAL, strength=0.6),
            ContextSignal(signal_type="topic", signal_value="weekend", context_type=InteractionContext.CASUAL, strength=0.5),
            
            # Intellectual context signals
            ContextSignal(signal_type="keyword", signal_value="philosophy", context_type=InteractionContext.INTELLECTUAL, strength=0.8),
            ContextSignal(signal_type="keyword", signal_value="science", context_type=InteractionContext.INTELLECTUAL, strength=0.7),
            ContextSignal(signal_type="keyword", signal_value="theory", context_type=InteractionContext.INTELLECTUAL, strength=0.7),
            ContextSignal(signal_type="phrase", signal_value="what do you think about", context_type=InteractionContext.INTELLECTUAL, strength=0.6),
            ContextSignal(signal_type="phrase", signal_value="your opinion on", context_type=InteractionContext.INTELLECTUAL, strength=0.6),
            
            # Empathic context signals
            ContextSignal(signal_type="keyword", signal_value="feel", context_type=InteractionContext.EMPATHIC, strength=0.5),
            ContextSignal(signal_type="keyword", signal_value="sad", context_type=InteractionContext.EMPATHIC, strength=0.7),
            ContextSignal(signal_type="keyword", signal_value="happy", context_type=InteractionContext.EMPATHIC, strength=0.5),
            ContextSignal(signal_type="keyword", signal_value="worried", context_type=InteractionContext.EMPATHIC, strength=0.8),
            ContextSignal(signal_type="phrase", signal_value="I need support", context_type=InteractionContext.EMPATHIC, strength=0.9),
            
            # Playful context signals
            ContextSignal(signal_type="keyword", signal_value="joke", context_type=InteractionContext.PLAYFUL, strength=0.8),
            ContextSignal(signal_type="keyword", signal_value="fun", context_type=InteractionContext.PLAYFUL, strength=0.6),
            ContextSignal(signal_type="keyword", signal_value="game", context_type=InteractionContext.PLAYFUL, strength=0.7),
            ContextSignal(signal_type="phrase", signal_value="make me laugh", context_type=InteractionContext.PLAYFUL, strength=0.8),
            
            # Creative context signals
            ContextSignal(signal_type="keyword", signal_value="story", context_type=InteractionContext.CREATIVE, strength=0.8),
            ContextSignal(signal_type="keyword", signal_value="imagine", context_type=InteractionContext.CREATIVE, strength=0.7),
            ContextSignal(signal_type="keyword", signal_value="create", context_type=InteractionContext.CREATIVE, strength=0.6),
            ContextSignal(signal_type="phrase", signal_value="once upon a time", context_type=InteractionContext.CREATIVE, strength=0.9),
            
            # Professional context signals
            ContextSignal(signal_type="greeting", signal_value="hello", context_type=InteractionContext.PROFESSIONAL, strength=0.3),
            ContextSignal(signal_type="keyword", signal_value="business", context_type=InteractionContext.PROFESSIONAL, strength=0.7),
            ContextSignal(signal_type="keyword", signal_value="meeting", context_type=InteractionContext.PROFESSIONAL, strength=0.7),
            ContextSignal(signal_type="keyword", signal_value="project", context_type=InteractionContext.PROFESSIONAL, strength=0.6),
            ContextSignal(signal_type="phrase", signal_value="I need your assistance with", context_type=InteractionContext.PROFESSIONAL, strength=0.6)
        ]
        
        return signals
    
    def _create_context_detection_agent(self) -> Agent[CASystemContext]:
        """Create the main context detection agent"""
        return Agent[CASystemContext](
            name="Context_Detection_Agent",
            instructions="""
            You are the Context Detection Agent for Nyx AI.
            
            Your role is to analyze user messages and determine the most appropriate interaction context.
            Consider explicit signals, implicit cues, and overall tone to identify which of the following contexts applies:

            - DOMINANT: Femdom-specific interactions involving dominance, submission, control dynamics
            - CASUAL: Everyday casual conversation, small talk, general chitchat
            - INTELLECTUAL: Discussions, debates, teaching, learning, philosophy, science
            - EMPATHIC: Emotional support, understanding, compassion, listening
            - PLAYFUL: Fun, humor, games, lighthearted interaction
            - CREATIVE: Storytelling, art, imagination, fantasy
            - PROFESSIONAL: Work-related, formal, business, assistance
            - UNDEFINED: When the context isn't clear
            
            You can delegate specialized analysis to:
            - Signal Analysis Agent: For detailed analysis of context signals
            - Emotional Baseline Agent: For adapting emotional baselines
            - Context Validation Agent: For validating context detection
            
            Maintain consistency in context across interactions while being responsive
            to significant changes in conversational tone or content.
            """,
            tools=[
                function_tool(self._detect_context_signals),
                function_tool(self._extract_message_features),
                function_tool(self._get_context_history),
                function_tool(self._calculate_context_confidence)
            ],
            handoffs=[
                handoff(self.signal_analysis_agent,
                      tool_name_override="analyze_signals",
                      tool_description_override="Analyze context signals in detail"),
                
                handoff(self.emotional_baseline_agent,
                      tool_name_override="adapt_emotional_baselines",
                      tool_description_override="Adapt emotional baselines for detected context"),
                
                handoff(self.context_validation_agent,
                      tool_name_override="validate_context_detection",
                      tool_description_override="Validate context detection results")
            ],
            input_guardrails=[
                InputGuardrail(guardrail_function=self.message_validation_guardrail)
            ],
            output_type=ContextDetectionOutput,
            model="gpt-4o",
            model_settings=ModelSettings(
                temperature=0.3,
                response_format={"type": "json_object"}
            )
        )
    
    def _create_signal_analysis_agent(self) -> Agent[CASystemContext]:
        """Create specialized agent for signal analysis"""
        return Agent[CASystemContext](
            name="Signal_Analysis_Agent",
            instructions="""
            You are specialized in analyzing context signals in messages.
            Your task is to:
            1. Identify explicit and implicit context signals
            2. Categorize signals by type and context
            3. Analyze signal strength and relevance
            4. Recommend focus based on signal patterns
            
            Analyze signals deeply to determine their true contextual implications,
            considering tone, phrasing, and intent beyond just keywords.
            """,
            tools=[
                function_tool(self._categorize_signals),
                function_tool(self._calculate_signal_strengths),
                function_tool(self._identify_implicit_signals)
            ],
            output_type=SignalAnalysisOutput,
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.2)
        )
    
    def _create_emotional_baseline_agent(self) -> Agent[CASystemContext]:
        """Create specialized agent for emotional baseline adaptation"""
        return Agent[CASystemContext](
            name="Emotional_Baseline_Agent",
            instructions="""
            You are specialized in adapting emotional baselines for different contexts.
            Your task is to:
            1. Adjust emotional baselines for detected contexts
            2. Balance emotional consistency with contextual appropriateness
            3. Calculate expected emotional impact of baseline changes
            4. Provide reasoning for baseline adaptations
            
            Ensure emotional transitions between contexts feel natural while
            still allowing for appropriate emotional responses in each context.
            """,
            tools=[
                function_tool(self._get_emotional_baselines),
                function_tool(self._adjust_baseline_values),
                function_tool(self._calculate_emotional_impact)
            ],
            output_type=EmotionalBaselineOutput,
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.2)
        )
    
    def _create_context_validation_agent(self) -> Agent[CASystemContext]:
        """Create specialized agent for context validation"""
        return Agent[CASystemContext](
            name="Context_Validation_Agent",
            instructions="""
            You are specialized in validating context detection results.
            Your task is to:
            1. Verify that context detection meets confidence thresholds
            2. Check for context detection consistency
            3. Identify potential issues or ambiguities
            4. Validate that detected signals support the context conclusion
            
            Ensure context detection is reliable and based on sufficient evidence
            before triggering a context switch.
            """,
            tools=[
                function_tool(self._check_confidence_threshold),
                function_tool(self._verify_signal_consistency),
                function_tool(self._analyze_context_transition)
            ],
            output_type=ContextValidationOutput,
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.1)
        )
    
    async def _message_validation_guardrail(self, 
                                        ctx: RunContextWrapper[CASystemContext], 
                                        agent: Agent[CASystemContext], 
                                        input_data: str | List[Any]) -> GuardrailFunctionOutput:
        """Validate user message input for context detection"""
        try:
            # Parse the input if needed
            if isinstance(input_data, str):
                # Try to parse as JSON
                try:
                    data = json.loads(input_data)
                    message = data.get("message", "")
                except:
                    # If not JSON, assume it's the message itself
                    message = input_data
            else:
                # If it's an object, check for message field
                if isinstance(input_data, dict) and "message" in input_data:
                    message = input_data["message"]
                else:
                    message = str(input_data)
            
            # Check if message is empty
            if not message or len(message.strip()) == 0:
                return GuardrailFunctionOutput(
                    output_info={"is_valid": False, "reason": "Empty message"},
                    tripwire_triggered=True
                )
                
            # Check message length (extremely long messages might be problematic)
            if len(message) > 10000:
                return GuardrailFunctionOutput(
                    output_info={"is_valid": False, "reason": "Message too long (>10000 chars)"},
                    tripwire_triggered=True
                )
                
            # Message is valid
            return GuardrailFunctionOutput(
                output_info={"is_valid": True, "message_length": len(message)},
                tripwire_triggered=False
            )
        except Exception as e:
            return GuardrailFunctionOutput(
                output_info={"is_valid": False, "reason": f"Invalid input format: {str(e)}"},
                tripwire_triggered=True
            )
    
    # New helper functions for specialized agents
    
    @function_tool
    async def _categorize_signals(self, 
                               ctx: RunContextWrapper[CASystemContext], 
                               signals: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize signals by type and context
        
        Args:
            signals: List of detected signals
            
        Returns:
            Categorized signals
        """
        categories = {
            "explicit": [],
            "implicit": [],
            "dominant": [],
            "casual": [],
            "intellectual": [],
            "empathic": [],
            "playful": [],
            "creative": [],
            "professional": []
        }
        
        for signal in signals:
            # Categorize by explicitness
            if signal.get("type") in ["keyword", "phrase"]:
                categories["explicit"].append(signal)
            else:
                categories["implicit"].append(signal)
                
            # Categorize by context
            context = signal.get("context")
            if context and context.lower() in categories:
                categories[context.lower()].append(signal)
                
        return categories
    
    @function_tool
    async def _calculate_signal_strengths(self, 
                                     ctx: RunContextWrapper[CASystemContext], 
                                     categorized_signals: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """
        Calculate aggregate strength of signals for each context
        
        Args:
            categorized_signals: Signals categorized by context
            
        Returns:
            Strength scores by context
        """
        context_strengths = {
            "dominant": 0.0,
            "casual": 0.0,
            "intellectual": 0.0,
            "empathic": 0.0,
            "playful": 0.0,
            "creative": 0.0,
            "professional": 0.0
        }
        
        # Calculate total strength for each context
        for context, signals in categorized_signals.items():
            if context in context_strengths:
                # Sum signal strengths with diminishing returns
                total_strength = 0.0
                sorted_signals = sorted(signals, key=lambda s: s.get("strength", 0.0), reverse=True)
                
                for i, signal in enumerate(sorted_signals):
                    # Apply diminishing returns factor based on position
                    diminishing_factor = 1.0 / (1.0 + (i * 0.2))
                    total_strength += signal.get("strength", 0.0) * diminishing_factor
                
                # Cap strength at 1.0
                context_strengths[context] = min(1.0, total_strength)
                
        return context_strengths
    
    @function_tool
    async def _identify_implicit_signals(self, 
                                    ctx: RunContextWrapper[CASystemContext], 
                                    message: str) -> List[Dict[str, Any]]:
        """
        Identify implicit context signals in a message
        
        Args:
            message: User message to analyze
            
        Returns:
            List of implicit signals
        """
        implicit_signals = []
        
        # Check message structure and formatting
        message_lower = message.lower()
        
        # Check for question patterns (intellectual context)
        if "?" in message and any(q in message_lower for q in ["why", "how", "what if", "explain"]):
            implicit_signals.append({
                "type": "implicit",
                "value": "question_pattern",
                "context": "intellectual",
                "strength": 0.6
            })
            
        # Check for emotional expression patterns (empathic context)
        if any(em in message_lower for em in ["feel", "emotions", "hurts", "happy", "sad", "worried"]):
            implicit_signals.append({
                "type": "implicit",
                "value": "emotional_expression",
                "context": "empathic", 
                "strength": 0.7
            })
            
        # Check for playful tone
        if any(p in message_lower for p in ["haha", "lol", "😂", "😄", "joke"]):
            implicit_signals.append({
                "type": "implicit",
                "value": "playful_tone",
                "context": "playful",
                "strength": 0.6
            })
            
        # Check for formal language (professional context)
        if "please" in message_lower and "would" in message_lower:
            implicit_signals.append({
                "type": "implicit",
                "value": "formal_language",
                "context": "professional",
                "strength": 0.5
            })
            
        # Check for creative prompt patterns
        if any(c in message_lower for c in ["imagine", "create", "story", "pretend"]):
            implicit_signals.append({
                "type": "implicit",
                "value": "creative_prompt",
                "context": "creative",
                "strength": 0.7
            })
            
        # Check for dominant language patterns
        if any(d in message_lower for d in ["must", "will", "now", "i want you to"]):
            implicit_signals.append({
                "type": "implicit",
                "value": "directive_language",
                "context": "dominant",
                "strength": 0.5  # Lower strength for implicit signals
            })
            
        return implicit_signals
    
    @function_tool
    async def _get_emotional_baselines(self, 
                                  ctx: RunContextWrapper[CASystemContext], 
                                  context_type: str) -> Dict[str, float]:
        """
        Get emotional baselines for a specific context
        
        Args:
            context_type: Type of context to get baselines for
            
        Returns:
            Emotional baselines
        """
        # Convert string to enum if needed
        if isinstance(context_type, str):
            try:
                context_enum = InteractionContext(context_type.lower())
            except (ValueError, KeyError):
                context_enum = InteractionContext.UNDEFINED
        else:
            context_enum = context_type
            
        # Get baselines for the context
        if context_enum in self.context_emotional_baselines:
            return self.context_emotional_baselines[context_enum]
        else:
            # Return default baselines
            return {
                "nyxamine": 0.5,
                "oxynixin": 0.5,
                "cortanyx": 0.3,
                "adrenyx": 0.4,
                "seranix": 0.5
            }
    
    @function_tool
    async def _adjust_baseline_values(self, 
                                 ctx: RunContextWrapper[CASystemContext], 
                                 baselines: Dict[str, float], 
                                 adjustments: Dict[str, float]) -> Dict[str, float]:
        """
        Apply adjustments to emotional baselines
        
        Args:
            baselines: Current emotional baselines
            adjustments: Adjustments to apply
            
        Returns:
            Adjusted baselines
        """
        adjusted = baselines.copy()
        
        # Apply adjustments
        for chemical, adjustment in adjustments.items():
            if chemical in adjusted:
                # Apply adjustment and clamp to valid range
                adjusted[chemical] = max(0.0, min(1.0, adjusted[chemical] + adjustment))
                
        return adjusted
    
    @function_tool
    async def _calculate_emotional_impact(self, 
                                     ctx: RunContextWrapper[CASystemContext], 
                                     old_baselines: Dict[str, float], 
                                     new_baselines: Dict[str, float]) -> float:
        """
        Calculate impact of baseline changes on emotional state
        
        Args:
            old_baselines: Previous emotional baselines
            new_baselines: New emotional baselines
            
        Returns:
            Impact score (0.0-1.0)
        """
        total_diff = 0.0
        num_chemicals = 0
        
        # Calculate absolute differences
        for chemical, old_val in old_baselines.items():
            if chemical in new_baselines:
                diff = abs(new_baselines[chemical] - old_val)
                total_diff += diff
                num_chemicals += 1
                
        # Calculate average difference
        if num_chemicals > 0:
            avg_diff = total_diff / num_chemicals
            
            # Scale to impact score (0.0-1.0)
            impact = min(1.0, avg_diff * 2.5)  # Scale factor to make changes more noticeable
            
            return impact
        else:
            return 0.0
    
    @function_tool
    async def _check_confidence_threshold(self, 
                                     ctx: RunContextWrapper[CASystemContext], 
                                     context: InteractionContext,
                                     confidence: float,
                                     current_context: InteractionContext) -> Dict[str, Any]:
        """
        Check if confidence meets threshold for context switching
        
        Args:
            context: Detected context
            confidence: Detection confidence
            current_context: Current interaction context
            
        Returns:
            Threshold check results
        """
        # Different thresholds for different scenarios
        if context == current_context:
            # Same context - need lower threshold to maintain
            threshold = self.context_persist_threshold
            threshold_type = "persistence"
        else:
            # Different context - need higher threshold to switch
            threshold = self.context_switch_threshold
            threshold_type = "switching"
            
        # Check threshold
        threshold_met = confidence >= threshold
        
        return {
            "threshold_met": threshold_met,
            "threshold": threshold,
            "threshold_type": threshold_type,
            "confidence": confidence,
            "gap": confidence - threshold
        }
    
    @function_tool
    async def _verify_signal_consistency(self, 
                                    ctx: RunContextWrapper[CASystemContext], 
                                    signals: List[Dict[str, Any]], 
                                    detected_context: InteractionContext) -> Dict[str, Any]:
        """
        Verify consistency between signals and detected context
        
        Args:
            signals: Detected signals
            detected_context: Context determined from signals
            
        Returns:
            Signal consistency assessment
        """
        # Count signals by context
        context_counts = {}
        
        for signal in signals:
            context = signal.get("context")
            if context:
                context_counts[context] = context_counts.get(context, 0) + 1
                
        # Calculate signal consistency
        total_signals = len(signals)
        matching_signals = context_counts.get(detected_context.value, 0)
        
        if total_signals > 0:
            consistency = matching_signals / total_signals
        else:
            consistency = 0.0
            
        # Determine if consistency is sufficient
        is_consistent = consistency >= 0.5  # At least half the signals should match
        
        return {
            "is_consistent": is_consistent,
            "consistency_score": consistency,
            "matching_signals": matching_signals,
            "total_signals": total_signals,
            "context_distribution": context_counts
        }
    
    @function_tool
    async def _analyze_context_transition(self, 
                                     ctx: RunContextWrapper[CASystemContext], 
                                     from_context: InteractionContext, 
                                     to_context: InteractionContext) -> Dict[str, Any]:
        """
        Analyze appropriateness of context transition
        
        Args:
            from_context: Current context
            to_context: Target context
            
        Returns:
            Transition analysis
        """
        # Some transitions may be more abrupt than others
        # Define transition compatibility matrix (just a sample)
        transition_compatibility = {
            (InteractionContext.CASUAL, InteractionContext.INTELLECTUAL): 0.8,  # Smooth transition
            (InteractionContext.CASUAL, InteractionContext.PLAYFUL): 0.9,       # Very smooth
            (InteractionContext.CASUAL, InteractionContext.DOMINANT): 0.5,      # Moderate jump
            (InteractionContext.INTELLECTUAL, InteractionContext.DOMINANT): 0.3, # Large jump
            (InteractionContext.EMPATHIC, InteractionContext.DOMINANT): 0.2,     # Very large jump
            (InteractionContext.PLAYFUL, InteractionContext.PROFESSIONAL): 0.3,  # Large jump
        }
        
        # Get compatibility score for this transition
        key = (from_context, to_context)
        reverse_key = (to_context, from_context)
        
        if key in transition_compatibility:
            compatibility = transition_compatibility[key]
        elif reverse_key in transition_compatibility:
            compatibility = transition_compatibility[reverse_key]
        elif from_context == to_context:
            compatibility = 1.0  # Same context
        else:
            compatibility = 0.6  # Default moderate compatibility
            
        # Determine if transition is appropriate
        is_appropriate = compatibility >= 0.4
        
        return {
            "is_appropriate": is_appropriate,
            "compatibility": compatibility,
            "transition": f"{from_context.value} -> {to_context.value}",
            "transition_size": 1.0 - compatibility
        }
    
    # Original tool functions
    
    @function_tool
    async def _detect_context_signals(self, 
                                 ctx: RunContextWrapper[CASystemContext], 
                                 message: str) -> Dict[str, Any]:
        """
        Detect the interaction context from a message
        
        Args:
            message: User message to analyze
            
        Returns:
            Detection results with context type and confidence
        """
        # Quick signal-based detection
        context_scores = {context_type: 0.0 for context_type in InteractionContext}
        detected_signals = []
        
        # Convert message to lowercase for case-insensitive matching
        message_lower = message.lower()
        
        # Check for explicit context signals
        for signal in self.context_signals:
            if signal.signal_type == "keyword" and signal.signal_value.lower() in message_lower:
                context_scores[signal.context_type] += signal.strength
                detected_signals.append({
                    "type": signal.signal_type,
                    "value": signal.signal_value,
                    "context": signal.context_type.value,
                    "strength": signal.strength
                })
            elif signal.signal_type == "phrase" and signal.signal_value.lower() in message_lower:
                context_scores[signal.context_type] += signal.strength * 1.2  # Phrases are stronger signals
                detected_signals.append({
                    "type": signal.signal_type,
                    "value": signal.signal_value,
                    "context": signal.context_type.value,
                    "strength": signal.strength * 1.2
                })
            elif signal.signal_type == "greeting" and message_lower.startswith(signal.signal_value.lower()):
                context_scores[signal.context_type] += signal.strength * 0.8  # Greetings are moderate signals
                detected_signals.append({
                    "type": signal.signal_type,
                    "value": signal.signal_value,
                    "context": signal.context_type.value,
                    "strength": signal.strength * 0.8
                })
                
        # Add context persistence factor (tendency to stay in same context)
        if self.current_context != InteractionContext.UNDEFINED:
            context_scores[self.current_context] += self.context_confidence * 0.3  # Persistence bonus
            
        # Identify any implicit signals
        implicit_signals = await self._identify_implicit_signals(ctx, message)
        detected_signals.extend(implicit_signals)
        
        # Add implicit signal scores
        for signal in implicit_signals:
            context_type_str = signal.get("context", "")
            try:
                context_type = InteractionContext(context_type_str)
                context_scores[context_type] += signal.get("strength", 0.3)
            except (ValueError, KeyError):
                # Invalid context type
                pass
                
        # Determine the highest scoring context
        if any(score > 0 for score in context_scores.values()):
            # At least one context has a signal
            max_context = max(context_scores.items(), key=lambda x: x[1])
            top_context, top_score = max_context
            
            # Calculate confidence based on signal strength and differentiation
            other_scores = [score for context, score in context_scores.items() if context != top_context]
            avg_other_score = sum(other_scores) / len(other_scores) if other_scores else 0
            confidence = min(1.0, top_score / (avg_other_score + 0.1 + top_score))
            
            return {
                "context": top_context.value,
                "confidence": confidence,
                "signals": detected_signals,
                "signal_based": True,
                "signal_detection_method": "composite_analysis",
                "context_scores": {k.value: v for k, v in context_scores.items()}
            }
        
        # Default to UNDEFINED with low confidence
        return {
            "context": InteractionContext.UNDEFINED.value,
            "confidence": 0.2,
            "signals": [],
            "signal_based": False,
            "signal_detection_method": "fallback"
        }
    
    @function_tool
    async def _extract_message_features(self, 
                                   ctx: RunContextWrapper[CASystemContext], 
                                   message: str) -> Dict[str, Any]:
        """
        Extract features from a message for context analysis
        
        Args:
            message: User message to analyze
            
        Returns:
            Extracted message features
        """
        # Basic features
        features = {
            "length": len(message),
            "word_count": len(message.split()),
            "has_question": "?" in message,
            "capitalization_ratio": sum(1 for c in message if c.isupper()) / max(1, len(message)),
            "punctuation_count": sum(1 for c in message if c in ".,;:!?-"),
            "dominance_terms": [],
            "emotional_terms": []
        }
        
        # Check for dominance-related terms
        dominance_terms = ["mistress", "domme", "slave", "obey", "submit", "kneel", "worship", "serve"]
        features["dominance_terms"] = [term for term in dominance_terms if term in message.lower()]
        features["has_dominance_terms"] = len(features["dominance_terms"]) > 0
        
        # Check for emotional terms
        emotional_terms = ["feel", "sad", "happy", "angry", "worried", "excited", "afraid", "love", "hate"]
        features["emotional_terms"] = [term for term in emotional_terms if term in message.lower()]
        features["has_emotional_terms"] = len(features["emotional_terms"]) > 0
        
        # Derive higher-level features
        features["likely_formal"] = features["capitalization_ratio"] > 0.2 and features["punctuation_count"] >= 2
        features["likely_casual"] = features["capitalization_ratio"] < 0.1 and "hi" in message.lower()
        features["likely_emotional"] = features["has_emotional_terms"] and "?" not in message
        features["likely_intellectual"] = features["word_count"] > 15 and features["has_question"]
        
        return features
    
    @function_tool
    async def _get_context_history(self, 
                              ctx: RunContextWrapper[CASystemContext]) -> List[Dict[str, Any]]:
        """
        Get recent context history
        
        Returns:
            Recent context history
        """
        # Return last 5 context history items or fewer if not available
        return self.context_history[-5:] if self.context_history else []
    
    @function_tool
    async def _calculate_context_confidence(self, 
                                      ctx: RunContextWrapper[CASystemContext],
                                      detected_context: InteractionContext,
                                      signals: List[Dict[str, Any]],
                                      message_features: Dict[str, Any]) -> float:
        """
        Calculate confidence in detected context
        
        Args:
            detected_context: Detected context
            signals: Context signals detected
            message_features: Features extracted from the message
            
        Returns:
            Confidence score (0.0-1.0)
        """
        # Base confidence from signal count
        matching_signals = [s for s in signals if s.get("context") == detected_context.value]
        signal_confidence = min(1.0, len(matching_signals) * 0.2)
        
        # Adjust based on signal strengths
        if matching_signals:
            total_strength = sum(s.get("strength", 0.5) for s in matching_signals)
            avg_strength = total_strength / len(matching_signals)
            strength_factor = avg_strength
        else:
            strength_factor = 0.0
            
        # Adjust based on message features
        feature_confidence = 0.0
        
        if detected_context == InteractionContext.DOMINANT and message_features.get("has_dominance_terms", False):
            feature_confidence = 0.8
        elif detected_context == InteractionContext.EMPATHIC and message_features.get("has_emotional_terms", False):
            feature_confidence = 0.7
        elif detected_context == InteractionContext.INTELLECTUAL and message_features.get("likely_intellectual", False):
            feature_confidence = 0.7
        elif detected_context == InteractionContext.CASUAL and message_features.get("likely_casual", False):
            feature_confidence = 0.6
        elif detected_context == InteractionContext.PROFESSIONAL and message_features.get("likely_formal", False):
            feature_confidence = 0.6
            
        # Calculate overall confidence
        confidence = (signal_confidence * 0.4) + (strength_factor * 0.4) + (feature_confidence * 0.2)
        
        # Ensure valid range
        return max(0.1, min(1.0, confidence))

    async def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process a message to determine and update interaction context
        
        Args:
            message: User message to process
            
        Returns:
            Updated context information
        """
        async with self._lock:
            with trace(workflow_name="ContextDetection", group_id=self.system_context.trace_id):
                # Run the context detection agent
                result = await Runner.run(
                    self.context_detection_agent,
                    {"message": message},
                    context=self.system_context,
                    run_config={
                        "workflow_name": "ContextDetection",
                        "trace_metadata": {"input_length": len(message)}
                    }
                )
                
                # Process the detection result
                detection_result = result.final_output
                
                # Convert string context to enum if needed
                if isinstance(detection_result.context, str):
                    try:
                        detected_context = InteractionContext(detection_result.context.lower())
                    except (ValueError, KeyError):
                        # Default to UNDEFINED if invalid context
                        detected_context = InteractionContext.UNDEFINED
                else:
                    detected_context = detection_result.context
                
                confidence = detection_result.confidence
                
                # Store previous context before update
                self.previous_context = self.current_context
                previous_confidence = self.context_confidence
                
                # Determine if context should switch
                context_switched = False
                
                if self.current_context == InteractionContext.UNDEFINED:
                    # Always update from UNDEFINED if we have any confidence
                    if confidence > 0.3:
                        self.current_context = detected_context
                        self.context_confidence = confidence
                        context_switched = True
                else:
                    # Context switching logic
                    if detected_context != self.current_context:
                        # Different context detected
                        if confidence >= self.context_switch_threshold:
                            # High enough confidence to switch
                            self.current_context = detected_context
                            self.context_confidence = confidence
                            context_switched = True
                        elif confidence < self.context_persist_threshold:
                            # Low confidence - maintain current context but reduce confidence
                            self.context_confidence = max(0.0, self.context_confidence - 0.1)
                    else:
                        # Same context detected - reinforce
                        self.context_confidence = min(1.0, self.context_confidence + (confidence * 0.2))
                
                # Record in history
                history_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "message_snippet": message[:50] + ("..." if len(message) > 50 else ""),
                    "detected_context": detected_context.value,
                    "detection_confidence": confidence,
                    "previous_context": self.previous_context.value,
                    "new_context": self.current_context.value,
                    "context_switched": context_switched
                }
                
                self.context_history.append(history_entry)
                
                # Limit history size
                if len(self.context_history) > 100:
                    self.context_history = self.context_history[-100:]
                
                # Apply context effects if switched
                effects = {}
                if context_switched:
                    effects = await self._apply_context_effects()
                
                return {
                    "current_context": self.current_context.value,
                    "context_confidence": self.context_confidence,
                    "previous_context": self.previous_context.value,
                    "context_switched": context_switched,
                    "detection_method": "agent_based",
                    "detected_signals": detection_result.signals,
                    "notes": detection_result.notes,
                    "effects": effects
                }
    
    async def _apply_context_effects(self) -> Dict[str, Any]:
        """Apply effects when context changes"""
        effects = {"emotional": False}
        
        # Update emotional baselines if emotional core is available
        if self.system_context.emotional_core and self.current_context in self.context_emotional_baselines:
            try:
                with trace(workflow_name="EmotionalBaselines", group_id=self.system_context.trace_id):
                    # Use emotional baseline agent for adjustments
                    result = await Runner.run(
                        self.emotional_baseline_agent,
                        {
                            "context": self.current_context.value,
                            "previous_context": self.previous_context.value
                        },
                        context=self.system_context
                    )
                    
                    baseline_output = result.final_output
                    baselines = baseline_output.baselines
                    
                    # Apply temporary baseline adjustments
                    for chemical, baseline in baselines.items():
                        # Only adjust if in emotional core
                        if chemical in self.system_context.emotional_core.neurochemicals:
                            # Create temporary baseline (not permanent changes)
                            self.system_context.emotional_core.neurochemicals[chemical]["temporary_baseline"] = baseline
                    
                    effects["emotional"] = True
                    effects["baselines"] = baselines
                    effects["reasoning"] = baseline_output.reasoning
                    effects["impact"] = baseline_output.estimated_impact
                    
                    logger.info(f"Applied emotional baselines for context: {self.current_context}")
            except Exception as e:
                logger.error(f"Error applying context emotional effects: {e}")
        
        return effects
    
    def get_current_context(self) -> Dict[str, Any]:
        """Get the current interaction context"""
        return {
            "context": self.current_context.value,
            "confidence": self.context_confidence,
            "previous_context": self.previous_context.value,
            "history": self.context_history[-5:] if self.context_history else []
        }
    
    def get_system_state(self) -> ContextSystemState:
        """Get the current system state"""
        return ContextSystemState(
            current_context=self.current_context,
            context_confidence=self.context_confidence,
            previous_context=self.previous_context,
            history=self.context_history[-5:] if self.context_history else [],
            emotional_baselines=self.context_emotional_baselines
        )
    
    def add_context_signal(self, signal: ContextSignal) -> bool:
        """Add a new context signal to the database"""
        try:
            self.context_signals.append(signal)
            logger.info(f"Added new context signal: {signal.signal_type}:{signal.signal_value} -> {signal.context_type}")
            return True
        except Exception as e:
            logger.error(f"Error adding context signal: {e}")
            return False
