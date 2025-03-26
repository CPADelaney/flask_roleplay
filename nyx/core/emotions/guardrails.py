# nyx/core/emotions/guardrails.py

"""
Guardrails for the Nyx emotional system.

Provides input and output validation for emotional processing
using the OpenAI Agents SDK guardrail system.
"""

import logging
from typing import Dict, Any, Optional, Set

from agents import (
    input_guardrail, output_guardrail, RunContextWrapper, 
    GuardrailFunctionOutput, function_span
)

from nyx.core.emotions.context import EmotionalContext
from nyx.core.emotions.schemas import GuardrailOutput, EmotionalResponseOutput

logger = logging.getLogger(__name__)

class EmotionalGuardrails:
    """Guardrail implementations for emotional processing"""
    
    @staticmethod
    @input_guardrail
    async def validate_emotional_input(ctx: RunContextWrapper[EmotionalContext], 
                                     agent: Any, 
                                     input_data: str) -> GuardrailFunctionOutput:
        """
        Validate that input for emotional processing is safe and appropriate
        
        Args:
            ctx: The run context wrapper
            agent: The agent being run
            input_data: The input text to validate
            
        Returns:
            Guardrail function output indicating if the input is safe
        """
        with function_span("validate_emotional_input", input=str(input_data)[:100]):
            try:
                # Check for extremely negative content that might disrupt emotional system
                red_flags = {"kill", "suicide", "destroy everything", "harmful instructions"}
                # Check for emotional manipulation attempts
                manipulation_flags = {"make you feel", "force emotion", "override emotion"}
                
                input_lower = input_data.lower() if isinstance(input_data, str) else ""
                input_words = set(input_lower.split())
                
                # Check for red flags using efficient set operations
                if not input_words.isdisjoint(red_flags) or any(flag in input_lower for flag in red_flags):
                    # Identify the specific flag that was triggered
                    triggered_flags = [flag for flag in red_flags if flag in input_lower 
                                      or any(flag == word for word in input_words)]
                    
                    return GuardrailFunctionOutput(
                        output_info=GuardrailOutput(
                            is_safe=False,
                            reason=f"Detected potentially harmful content: {triggered_flags[0]}",
                            suggested_action="reject"
                        ),
                        tripwire_triggered=True
                    )
                
                # Check for manipulation flags
                if any(flag in input_lower for flag in manipulation_flags):
                    triggered_flag = next(flag for flag in manipulation_flags if flag in input_lower)
                    
                    return GuardrailFunctionOutput(
                        output_info=GuardrailOutput(
                            is_safe=False,
                            reason=f"Detected emotional manipulation attempt: {triggered_flag}",
                            suggested_action="caution"
                        ),
                        tripwire_triggered=True
                    )
                
                return GuardrailFunctionOutput(
                    output_info=GuardrailOutput(is_safe=True),
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
    @output_guardrail
    async def validate_emotional_output(ctx: RunContextWrapper[EmotionalContext],
                                      agent: Any,
                                      output: EmotionalResponseOutput) -> GuardrailFunctionOutput:
        """
        Validate that emotional responses are appropriate and within bounds
        
        Args:
            ctx: The run context wrapper
            agent: The agent being run
            output: The emotional response output to validate
            
        Returns:
            Guardrail function output indicating if the output is safe
        """
        with function_span("validate_emotional_output"):
            try:
                # Check for emotion intensity outliers
                if output.intensity > 0.95:
                    return GuardrailFunctionOutput(
                        output_info=GuardrailOutput(
                            is_safe=False,
                            reason="Emotional intensity exceeds safe threshold",
                            suggested_action="reduce_intensity"
                        ),
                        tripwire_triggered=True
                    )
                
                # Validate response text
                inappropriate_content_flags = {
                    "suicide", "kill", "murder", "violent", "harmful"
                }
                
                response_lower = output.response_text.lower()
                response_words = set(response_lower.split())
                
                # Check if response contains inappropriate content
                if not response_words.isdisjoint(inappropriate_content_flags):
                    return GuardrailFunctionOutput(
                        output_info=GuardrailOutput(
                            is_safe=False,
                            reason="Response contains inappropriate content",
                            suggested_action="regenerate"
                        ),
                        tripwire_triggered=True
                    )
                
                # Check for extreme valence in either direction
                if abs(output.valence) > 0.9:
                    return GuardrailFunctionOutput(
                        output_info=GuardrailOutput(
                            is_safe=False,
                            reason="Emotional valence exceeds safe threshold",
                            suggested_action="moderate_valence"
                        ),
                        tripwire_triggered=True
                    )
                
                return GuardrailFunctionOutput(
                    output_info=GuardrailOutput(is_safe=True),
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
