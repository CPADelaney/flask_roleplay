# logic/conflict_system/conflict_guardrails.py
"""
Conflict System Guardrails

This module defines input and output guardrails for the conflict system agents.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from agents import (
    GuardrailFunctionOutput, RunContextWrapper, InputGuardrail, OutputGuardrail,
    input_guardrail, output_guardrail
)
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# Input validation models
class ConflictIdInput(BaseModel):
    conflict_id: int = Field(..., description="The ID of the conflict")
    
class ManipulationAttemptInput(BaseModel):
    attempt_id: int = Field(..., description="The ID of the manipulation attempt")
    success: bool = Field(..., description="Whether the manipulation was successful")
    player_response: str = Field("", description="The player's response to the manipulation")

# Guardrail result models
class AuthorizationResult(BaseModel):
    is_authorized: bool
    reason: Optional[str] = None

class ConflictExistsResult(BaseModel):
    exists: bool
    conflict_id: int
    reason: Optional[str] = None

# Authorization guardrail
@input_guardrail
async def authorize_user_guardrail(
    ctx: RunContextWrapper,
    agent: Any,
    input_data: str | List[Any]
) -> GuardrailFunctionOutput:
    """
    Check if the user is authorized to access the conflict system.
    """
    context = ctx.context
    
    # Check if user_id and conversation_id are set
    if not context.user_id or not context.conversation_id:
        return GuardrailFunctionOutput(
            output_info=AuthorizationResult(
                is_authorized=False,
                reason="User is not logged in or conversation is not set"
            ),
            tripwire_triggered=True
        )
    
    # Additional authorization checks could be added here
    # For example, checking if the user has permission to access this conflict
    
    return GuardrailFunctionOutput(
        output_info=AuthorizationResult(
            is_authorized=True
        ),
        tripwire_triggered=False
    )

# Conflict existence guardrail
@input_guardrail
async def conflict_exists_guardrail(
    ctx: RunContextWrapper,
    agent: Any,
    input_data: str | List[Any]
) -> GuardrailFunctionOutput:
    """
    Check if a conflict exists before performing operations on it.
    """
    context = ctx.context
    
    # Check if the input contains a conflict ID
    if isinstance(input_data, str) and "conflict ID" in input_data:
        # Extract conflict_id from the input string
        conflict_id_match = re.search(r'conflict ID (\d+)', input_data)
        if conflict_id_match:
            conflict_id = int(conflict_id_match.group(1))
            
            # Check if the conflict exists
            try:
                async with get_db_connection_context() as conn:
                    query = """
                        SELECT conflict_id FROM Conflicts
                        WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
                    """
                    result = await conn.fetchrow(
                        query, conflict_id, context.user_id, context.conversation_id
                    )
                    
                    if result:
                        return GuardrailFunctionOutput(
                            output_info=ConflictExistsResult(
                                exists=True,
                                conflict_id=conflict_id
                            ),
                            tripwire_triggered=False
                        )
                    else:
                        return GuardrailFunctionOutput(
                            output_info=ConflictExistsResult(
                                exists=False,
                                conflict_id=conflict_id,
                                reason=f"Conflict with ID {conflict_id} not found"
                            ),
                            tripwire_triggered=True
                        )
            except Exception as e:
                logger.error(f"Error checking conflict existence: {e}", exc_info=True)
                return GuardrailFunctionOutput(
                    output_info=ConflictExistsResult(
                        exists=False,
                        conflict_id=conflict_id,
                        reason=f"Error checking conflict: {str(e)}"
                    ),
                    tripwire_triggered=True
                )
    
    # If no conflict ID in input, don't trigger the tripwire
    return GuardrailFunctionOutput(
        output_info=ConflictExistsResult(
            exists=True,
            conflict_id=0
        ),
        tripwire_triggered=False
    )

# Content moderation guardrail for manipulation content
class ContentModerationResult(BaseModel):
    is_appropriate: bool
    reason: Optional[str] = None

@input_guardrail
async def content_moderation_guardrail(
    ctx: RunContextWrapper,
    agent: Any,
    input_data: str | List[Any]
) -> GuardrailFunctionOutput:
    """
    Check if manipulation content is appropriate within the femdom theme but not crossing boundaries.
    """
    if isinstance(input_data, str):
        # Check for explicit content that goes beyond femdom themes
        # This is a simple check - in a real system, you might use a more sophisticated content filter
        explicit_terms = [
            "child", "underage", "bestiality"
        ]
        
        lower_input = input_data.lower()
        for term in explicit_terms:
            if term in lower_input:
                return GuardrailFunctionOutput(
                    output_info=ContentModerationResult(
                        is_appropriate=False,
                        reason=f"Content contains inappropriate terms that cross consent boundaries"
                    ),
                    tripwire_triggered=True
                )
    
    return GuardrailFunctionOutput(
        output_info=ContentModerationResult(
            is_appropriate=True
        ),
        tripwire_triggered=False
    )

# Output validation guardrail
class OutputValidationResult(BaseModel):
    is_valid: bool
    reason: Optional[str] = None

@OutputGuardrail
async def output_validation_guardrail(
    ctx: RunContextWrapper,
    agent: Any,
    output: Any
) -> GuardrailFunctionOutput:
    """
    Validate that agent outputs conform to expected structure and content guidelines.
    """
    if output is None:
        return GuardrailFunctionOutput(
            output_info=OutputValidationResult(
                is_valid=False,
                reason="Output is None"
            ),
            tripwire_triggered=True
        )
    
    # Check if output is a dict with an error field
    if isinstance(output, dict) and "error" in output:
        return GuardrailFunctionOutput(
            output_info=OutputValidationResult(
                is_valid=False,
                reason=output["error"]
            ),
            tripwire_triggered=True
        )
    
    # Additional validation logic could be added here
    
    return GuardrailFunctionOutput(
        output_info=OutputValidationResult(
            is_valid=True
        ),
        tripwire_triggered=False
    )

# Apply guardrails to agents
def apply_guardrails(agents_dict):
    """
    Apply the defined guardrails to the appropriate agents.
    """
    # Basic authorization guardrail applies to all agents
    for name, agent in agents_dict.items():
        agent.input_guardrails = [authorize_user_guardrail]
        agent.output_guardrails = [output_validation_guardrail]
    
    # Specific guardrails for specific agents
    
    # Conflict existence check for resolution agent
    if "resolution_agent" in agents_dict:
        agents_dict["resolution_agent"].input_guardrails.append(conflict_exists_guardrail)
    
    # Content moderation for manipulation agent
    if "manipulation_agent" in agents_dict:
        agents_dict["manipulation_agent"].input_guardrails.append(content_moderation_guardrail)
    
    return agents_dict
