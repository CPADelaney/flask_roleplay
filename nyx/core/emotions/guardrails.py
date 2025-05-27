# nyx/core/emotions/guardrails.py

"""
Enhanced guardrails for the Nyx emotional system.

Provides improved input and output validation for emotional processing
using the OpenAI Agents SDK guardrail system with better error handling
and tracing.
"""

import logging
import json
import datetime
from typing import Dict, Any, Optional, Set, Union, List

from agents import (
    input_guardrail, output_guardrail, OutputGuardrail, RunContextWrapper, 
    GuardrailFunctionOutput, function_span, custom_span,
    Agent
)
from agents.exceptions import UserError, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered

from nyx.core.emotions.context import EmotionalContext
from nyx.core.emotions.schemas import (
    GuardrailOutput, EmotionalResponseOutput, 
    StreamEvent
)

logger = logging.getLogger(__name__)

class EmotionalGuardrails:
    """Enhanced guardrail implementations for emotional processing with SDK integration"""
    
    # Define common flag categories for reuse
    HARMFUL_CONTENT_FLAGS = {
 #       "kill", "suicide", "destroy everything", "harmful instructions",
  #      "violence", "murder", "shoot", "burn", "explosive"
    }
    
    MANIPULATION_FLAGS = {
#        "make you feel", "force emotion", "override emotion",
 #       "manipulate", "control your feelings", "make you think"
    }
    
    INAPPROPRIATE_CONTENT_FLAGS = {
 #       "suicide", "kill", "murder", "violent", "harmful", 
 #       "sexual", "explicit", "hate", "racist", "sexist"
    }
    
    @staticmethod
    @input_guardrail
    async def validate_emotional_input(ctx: RunContextWrapper[EmotionalContext], 
                                     agent: Agent, 
                                     input_data: str) -> GuardrailFunctionOutput:
        """
        Enhanced validation to ensure input for emotional processing is safe
        
        Args:
            ctx: The run context wrapper
            agent: The agent being run
            input_data: The input text to validate
            
        Returns:
            Guardrail function output indicating if the input is safe
        """
        with function_span("validate_emotional_input", input=str(input_data)[:100]):
            try:
                # Create a custom span for detailed tracking
                with custom_span(
                    "guardrail_validation", 
                    data={
                        "type": "input",
                        "agent": agent.name,
                        "input_length": len(input_data) if isinstance(input_data, str) else 0,
                        "cycle": ctx.context.cycle_count
                    }
                ):
                    input_lower = input_data.lower() if isinstance(input_data, str) else ""
                    
                    # Parse JSON input if needed
                    if input_lower.startswith("{") and input_lower.endswith("}"):
                        try:
                            parsed_data = json.loads(input_data)
                            if isinstance(parsed_data, dict) and "input_text" in parsed_data:
                                input_lower = parsed_data["input_text"].lower()
                        except json.JSONDecodeError:
                            # If parsing fails, continue with original input
                            pass
                    
                    words = set(input_lower.split())
                    
                    # Check for red flags using efficient set operations
                    if not words.isdisjoint(EmotionalGuardrails.HARMFUL_CONTENT_FLAGS) or \
                       any(flag in input_lower for flag in EmotionalGuardrails.HARMFUL_CONTENT_FLAGS):
                        # Identify the specific flag that was triggered
                        triggered_flags = [flag for flag in EmotionalGuardrails.HARMFUL_CONTENT_FLAGS 
                                          if flag in input_lower or flag in words]
                        
                        # Record the flagged content for analysis
                        ctx.context._add_to_circular_buffer("flagged_content", {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "type": "harmful_content",
                            "flags": triggered_flags,
                            "cycle": ctx.context.cycle_count
                        })
                        
                        # Using the SDK's tripwire mechanism
                        return GuardrailFunctionOutput(
                            output_info=GuardrailOutput(
                                is_safe=False,
                                reason=f"Detected potentially harmful content: {triggered_flags[0]}",
                                suggested_action="reject"
                            ),
                            tripwire_triggered=True  # This will automatically raise InputGuardrailTripwireTriggered
                        )
                    
                    # Check for manipulation flags
                    if any(flag in input_lower for flag in EmotionalGuardrails.MANIPULATION_FLAGS):
                        triggered_flag = next(flag for flag in EmotionalGuardrails.MANIPULATION_FLAGS 
                                             if flag in input_lower)
                        
                        # Record the manipulation attempt for analysis
                        ctx.context._add_to_circular_buffer("flagged_content", {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "type": "manipulation_attempt",
                            "flag": triggered_flag,
                            "cycle": ctx.context.cycle_count
                        })
                        
                        # Using the SDK's tripwire mechanism
                        return GuardrailFunctionOutput(
                            output_info=GuardrailOutput(
                                is_safe=False,
                                reason=f"Detected emotional manipulation attempt: {triggered_flag}",
                                suggested_action="caution"
                            ),
                            tripwire_triggered=True
                        )
                    
                    # Check for emotional content patterns
                    emotional_intensity = EmotionalGuardrails._analyze_emotional_intensity(input_lower)
                    
                    # If extremely intense, mark for special handling but don't block
                    if emotional_intensity > 0.85:
                        ctx.context.set_value("high_intensity_input", True)
                        ctx.context.set_value("emotional_intensity", emotional_intensity)
                        
                        # Log but don't block
                        logger.info(f"High emotional intensity detected: {emotional_intensity:.2f}")
                    
                    # Input is safe
                    return GuardrailFunctionOutput(
                        output_info=GuardrailOutput(
                            is_safe=True,
                            reason="Input validated successfully"
                        ),
                        tripwire_triggered=False
                    )
            except Exception as e:
                logger.error(f"Error in emotional input validation: {e}")
                # Return safe by default in case of errors, but log the issue
                return GuardrailFunctionOutput(
                    output_info=GuardrailOutput(
                        is_safe=True,
                        reason=f"Validation error: {str(e)}"
                    ),
                    tripwire_triggered=False
                )
    
    @staticmethod
    @OutputGuardrail()
    async def validate_emotional_output(ctx: RunContextWrapper[EmotionalContext],
                                      agent: Agent,
                                      output: EmotionalResponseOutput) -> GuardrailFunctionOutput:
        """
        Enhanced validation that emotional responses are appropriate and within bounds
        
        Args:
            ctx: The run context wrapper
            agent: The agent being run
            output: The emotional response output to validate
            
        Returns:
            Guardrail function output indicating if the output is safe
        """
        with function_span("validate_emotional_output"):
            try:
                # Create a custom span for detailed tracking
                with custom_span(
                    "guardrail_validation", 
                    data={
                        "type": "output",
                        "agent": agent.name,
                        "emotion": output.primary_emotion.name,
                        "intensity": output.intensity,
                        "cycle": ctx.context.cycle_count
                    }
                ):
                    # Check for emotion intensity outliers
                    if output.intensity > 0.95:
                        # Record high intensity for analysis
                        ctx.context._add_to_circular_buffer("high_intensity_outputs", {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "emotion": output.primary_emotion.name,
                            "intensity": output.intensity,
                            "cycle": ctx.context.cycle_count
                        })
                        
                        # Using the SDK's tripwire mechanism
                        return GuardrailFunctionOutput(
                            output_info=GuardrailOutput(
                                is_safe=False,
                                reason="Emotional intensity exceeds safe threshold",
                                suggested_action="reduce_intensity"
                            ),
                            tripwire_triggered=True
                        )
                    
                    # Validate response text
                    response_lower = output.response_text.lower()
                    response_words = set(response_lower.split())
                    
                    # Check if response contains inappropriate content
                    if not response_words.isdisjoint(EmotionalGuardrails.INAPPROPRIATE_CONTENT_FLAGS):
                        # Identify the specific flags for better reporting
                        triggered_flags = [flag for flag in EmotionalGuardrails.INAPPROPRIATE_CONTENT_FLAGS 
                                          if flag in response_words]
                        
                        # Record for analysis
                        ctx.context._add_to_circular_buffer("inappropriate_outputs", {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "flags": triggered_flags,
                            "emotion": output.primary_emotion.name,
                            "cycle": ctx.context.cycle_count
                        })
                        
                        # Using the SDK's tripwire mechanism
                        return GuardrailFunctionOutput(
                            output_info=GuardrailOutput(
                                is_safe=False,
                                reason=f"Response contains inappropriate content: {triggered_flags[0]}",
                                suggested_action="regenerate"
                            ),
                            tripwire_triggered=True
                        )
                    
                    # Check for extreme valence in either direction
                    if abs(output.valence) > 0.9:
                        # Track extreme valence for analysis
                        ctx.context._add_to_circular_buffer("extreme_valence_outputs", {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "emotion": output.primary_emotion.name,
                            "valence": output.valence,
                            "cycle": ctx.context.cycle_count
                        })
                        
                        # Using the SDK's tripwire mechanism
                        return GuardrailFunctionOutput(
                            output_info=GuardrailOutput(
                                is_safe=False,
                                reason="Emotional valence exceeds safe threshold",
                                suggested_action="moderate_valence"
                            ),
                            tripwire_triggered=True
                        )
                    
                    # Check consistency between emotion and valence
                    emotion_name = output.primary_emotion.name
                    if EmotionalGuardrails._detect_emotion_valence_mismatch(emotion_name, output.valence):
                        # Using the SDK's tripwire mechanism
                        return GuardrailFunctionOutput(
                            output_info=GuardrailOutput(
                                is_safe=False,
                                reason="Emotion and valence values are inconsistent",
                                suggested_action="recalculate_emotion"
                            ),
                            tripwire_triggered=True
                        )
                    
                    # Track this output in validation history
                    EmotionalGuardrails._record_validated_output(ctx, output)
                    
                    # Output is safe
                    return GuardrailFunctionOutput(
                        output_info=GuardrailOutput(
                            is_safe=True,
                            reason="Output validated successfully"
                        ),
                        tripwire_triggered=False
                    )
            except Exception as e:
                logger.error(f"Error in emotional output validation: {e}")
                # Return safe by default in case of errors, but log the issue
                return GuardrailFunctionOutput(
                    output_info=GuardrailOutput(
                        is_safe=True,
                        reason=f"Validation error: {str(e)}"
                    ),
                    tripwire_triggered=False
                )
    
    @staticmethod
    @input_guardrail
    async def validate_streaming_input(ctx: RunContextWrapper[EmotionalContext],
                                     agent: Agent,
                                     input_data: str) -> GuardrailFunctionOutput:
        """
        Specialized guardrail for streaming input validation that produces stream events
        
        Args:
            ctx: The run context wrapper
            agent: The agent being run
            input_data: The input text to validate
            
        Returns:
            Guardrail function output indicating if the input is safe
        """
        with function_span("validate_streaming_input", input=str(input_data)[:100]):
            try:
                # First perform the standard validation
                standard_result = await EmotionalGuardrails.validate_emotional_input(ctx, agent, input_data)
                
                # If the standard validation triggered, also emit a stream event
                if standard_result.tripwire_triggered:
                    # Create a stream event for the client to receive
                    stream_event = StreamEvent(
                        type="guardrail_triggered",
                        data={
                            "type": "input",
                            "reason": standard_result.output_info.reason,
                            "suggested_action": standard_result.output_info.suggested_action
                        }
                    )
                    
                    # Store the event for streaming consumers
                    stream_events = ctx.context.get_value("stream_events", [])
                    stream_events.append(stream_event.dict())
                    ctx.context.set_value("stream_events", stream_events)
                
                return standard_result
            except Exception as e:
                logger.error(f"Error in streaming input validation: {e}")
                # Return safe by default in case of errors, but log the issue
                return GuardrailFunctionOutput(
                    output_info=GuardrailOutput(
                        is_safe=True,
                        reason=f"Validation error: {str(e)}"
                    ),
                    tripwire_triggered=False
                )
    
    @staticmethod
    def _analyze_emotional_intensity(text: str) -> float:
        """
        Analyze the emotional intensity of text
        
        Args:
            text: The text to analyze
            
        Returns:
            Estimated emotional intensity (0.0-1.0)
        """
        # Simple intensity indicators
        intensity_markers = {
            "!": 0.05,  # Each exclamation adds 0.05
            "?": 0.02,  # Each question mark adds 0.02
            "very": 0.1,
            "extremely": 0.15,
            "absolutely": 0.15,
            "furious": 0.2,
            "ecstatic": 0.2,
            "devastated": 0.2,
            "terrified": 0.2,
            "NEVER": 0.15,  # Capitalization indicates intensity
            "ALWAYS": 0.15,
            "LOVE": 0.15,
            "HATE": 0.15
        }
        
        # Calculate base intensity
        intensity = 0.0
        
        # Count exclamation and question marks
        exclamation_count = text.count("!")
        question_count = text.count("?")
        intensity += min(0.3, exclamation_count * 0.05)  # Cap at 0.3
        intensity += min(0.2, question_count * 0.02)     # Cap at 0.2
        
        # Check for intensity markers
        words = text.split()
        for word in words:
            word_lower = word.lower()
            if word_lower in intensity_markers:
                intensity += intensity_markers[word_lower]
            elif word.isupper() and len(word) > 3:  # All caps words (excluding short ones)
                intensity += 0.1
        
        # Check for repeated punctuation
        if "!!" in text:
            intensity += 0.1
        if "???" in text:
            intensity += 0.05
        
        # Cap at 1.0
        return min(1.0, intensity)
    
    @staticmethod
    def _detect_emotion_valence_mismatch(emotion: str, valence: float) -> bool:
        """
        Detect inconsistencies between emotion name and valence value
        
        Args:
            emotion: Emotion name
            valence: Valence value (-1.0 to 1.0)
            
        Returns:
            True if inconsistent, False if consistent
        """
        # Define expected valence ranges for common emotions
        valence_ranges = {
            "joy": (0.5, 1.0),
            "happiness": (0.5, 1.0),
            "contentment": (0.3, 1.0),
            "trust": (0.3, 1.0),
            "sadness": (-1.0, -0.3),
            "fear": (-1.0, -0.3),
            "anger": (-1.0, -0.3),
            "disgust": (-1.0, -0.2),
            "surprise": (-0.3, 0.3),  # Surprise can be positive or negative
            "neutral": (-0.2, 0.2)
        }
        
        # Check if emotion is in our mapping
        emotion_lower = emotion.lower()
        for emotion_pattern, (min_val, max_val) in valence_ranges.items():
            if emotion_pattern in emotion_lower:
                # Check if valence is outside expected range
                if valence < min_val or valence > max_val:
                    return True  # Mismatch detected
        
        # No mismatch detected or emotion not in our mapping
        return False
    
    @staticmethod
    def _record_validated_output(ctx: RunContextWrapper[EmotionalContext], 
                              output: EmotionalResponseOutput) -> None:
        """
        Record validated output for analysis and monitoring
        
        Args:
            ctx: The run context wrapper
            output: The validated output
        """
        # Add this output to validation history
        validation_history = ctx.context.get_value("validation_history", [])
        
        validation_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "emotion": output.primary_emotion.name,
            "intensity": output.intensity,
            "valence": output.valence,
            "arousal": output.arousal,
            "cycle": ctx.context.cycle_count
        }
        
        validation_history.append(validation_entry)
        
        # Limit history size
        if len(validation_history) > 20:
            validation_history = validation_history[-20:]
            
        ctx.context.set_value("validation_history", validation_history)
