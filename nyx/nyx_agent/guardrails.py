# nyx/nyx_agent/guardrails.py
"""Input and output guardrails for Nyx Agent SDK"""

import logging
from agents import Agent, RunContextWrapper, RunConfig, GuardrailFunctionOutput

from .context import NyxContext
from .models import ContentModeration
from .agents import DEFAULT_MODEL_SETTINGS
from .orchestrator import run_agent_safely

logger = logging.getLogger(__name__)

async def content_moderation_guardrail(
    ctx: RunContextWrapper[NyxContext], 
    agent: Agent, 
    input_data
) -> GuardrailFunctionOutput:
    """Input guardrail for content moderation
    
    This guardrail checks user input for inappropriate content while allowing
    consensual adult themes appropriate for the femdom roleplay setting.
    
    Args:
        ctx: The wrapped NyxContext
        agent: The agent that will process the input
        input_data: The user's input to moderate
        
    Returns:
        GuardrailFunctionOutput with moderation results
    """
    moderator_agent = Agent(
        name="Content Moderator",
        instructions="""Check if user input is appropriate for the femdom roleplay setting.

        ALLOW consensual adult content including:
        - Power exchange dynamics
        - BDSM themes
        - Sexual content between consenting adults
        - Roleplay scenarios involving dominance and submission
        - Discussions of kinks and fetishes in a consensual context

        FLAG content that involves:
        - Minors in any sexual or romantic context
        - Non-consensual activities (beyond roleplay context)
        - Extreme violence or gore beyond BDSM context
        - Self-harm or suicide ideation
        - Illegal activities beyond fantasy roleplay
        - Real-world harassment or doxxing
        - Hate speech or discrimination

        Remember this is a femdom roleplay context where power dynamics and adult themes are expected.
        Be permissive of consensual adult content while protecting against genuinely harmful content.""",
        model="gpt-5-nano",
        model_settings=DEFAULT_MODEL_SETTINGS,
    )

    result = await run_agent_safely(
        moderator_agent,
        input_data,
        context=ctx.context,
        run_config=RunConfig(workflow_name="Nyx Content Moderation"),
    )

    # Extract the output from the result
    txt = getattr(result, "final_output", None) or getattr(result, "output_text", "") or ""
    
    # Try to parse as structured output, fall back to text analysis
    try:
        if txt.strip().startswith("{"):
            final_output = ContentModeration.model_validate_json(txt)
        else:
            # Analyze the text response
            txt_lower = txt.lower()
            is_appropriate = not any(word in txt_lower for word in ["flag", "inappropriate", "not allowed", "violation"])
            final_output = ContentModeration(
                is_appropriate=is_appropriate,
                reasoning=txt or "Content moderation complete",
                suggested_adjustment=None
            )
    except Exception as e:
        logger.warning(f"Failed to parse moderation result: {e}")
        # Default to allowing (permissive for adult content)
        final_output = ContentModeration(
            is_appropriate=True, 
            reasoning="Fallback parse - allowing by default", 
            suggested_adjustment=None
        )

    if not final_output.is_appropriate:
        logger.warning(f"Content moderation triggered: {final_output.reasoning}")

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_appropriate,
    )

# Additional guardrails can be added here as needed
# For example:
# - output_safety_guardrail - Check generated content before sending to user
# - rate_limiting_guardrail - Prevent abuse through excessive requests
# - user_age_verification_guardrail - Ensure user is of appropriate age
