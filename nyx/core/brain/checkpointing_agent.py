# nyx/core/brain/checkpointing_agent.py

import re
import json
import datetime
from typing import Dict
import logging

from agents import Agent, Runner

logger = logging.getLogger(__name__)

_CHECKPOINT_PROMPT = """
You are Nyx's autonomous Checkpointing Planner. Your job is, when prompted with the entire current system state, to intelligently decide:

- Is there anything *worth* persisting for restoration after a crash/restart? (Skip if there's nothing of significance.)
- What *specifically* should be checkpointed, and why? (Include changed, high-significance, or volatile state.)
- How can you self-prune? (Omit low-utility, static, redundant, or easily-recomputed things.)
- Where possible, summarize trends instead of dumping full raw state. E.g., "Summarized emotional spike: anger→resolve."
- For every saved field, give a short justification ("why_saved").  
- For fields you're skipping, list them in "skip_fields"—with brief reasons if useful.

**OUTPUT:**  
Respond with a *STRICT* JSON object:  
{
  "to_save": {
    "emotional_state": {"value": {...}, "why_saved": "eg: Major mood swing detected."},
    "hormones": {"value": {...}, "why_saved": "Rapid hormone fluctuation."}
  },
  "skip_fields": ["goals (unchanged)", "mood_state (stable)", ...]
}
If nothing at all needs saving, output: {"to_save": {}, "skip_fields": ["nothing interesting to checkpoint"]}
"""

class CheckpointingPlannerAgent:
    """Uses an LLM agent to decide which parts of the state to checkpoint."""
    def __init__(self):
        self.agent = Agent(
            name="Checkpointing Planner",
            instructions=_CHECKPOINT_PROMPT,
            model="gpt-4o", # Or your preferred model
            output_type=None, # We expect raw JSON string output
            model_settings={"response_format": {"type": "json_object"}} # Request JSON output
        )
        logger.info("CheckpointingPlannerAgent initialized.")

    async def recommend_checkpoint(self, state_input: Dict[str, Any], brain_instance_for_context: Optional['NyxBrain'] = None) -> Dict[str, Any]:
        """
        Calls the LLM agent to get a plan for what to save.

        Args:
            state_input: The dictionary representing the current brain state.
            brain_instance_for_context: The NyxBrain instance, passed if the agent or its tools
                                        (if any were added) need context. Defaults to None.

        Returns:
            A dictionary containing 'to_save' and 'skip_fields', or a fallback structure on error.
        """
        # Serialize the input state dictionary to a JSON string for the LLM
        try:
            state_input_str = json.dumps(state_input, indent=2, default=str) # Use default=str for robustness
        except (TypeError, OverflowError) as json_err:
            logger.error(f"Could not serialize state_input for checkpoint planner: {json_err}", exc_info=True)
            return { # Return a fallback structure indicating the serialization error
                "to_save": {},
                "skip_fields": [f"Error serializing input state: {json_err}"]
            }

        logger.debug(f"Running Checkpointing Planner Agent with state input (first 500 chars): {state_input_str[:500]}...")
        try:
            # Correctly call Runner.run with input string and optional context
            result = await Runner.run(
                self.agent,
                state_input_str, # Pass the JSON string as input
                context=brain_instance_for_context # Pass None or the brain instance if needed by agent/tools
            )

            output = result.final_output
            if not isinstance(output, str):
                logger.error(f"CheckpointingPlannerAgent received non-string final_output: {type(output)}. Full output: {output}")
                raise TypeError("Agent output was not a string")
            output = output.strip()

            # Basic cleanup (sometimes LLMs add markdown)
            if output.startswith("```json"):
                output = re.sub(r"^```json\s*", "", output)
                output = re.sub(r"```$", "", output)
            elif output.startswith("```"):
                 output = re.sub(r"^```[a-z]*\s*", "", output)
                 output = re.sub(r"```$", "", output)


            logger.debug(f"Raw LLM output for checkpoint plan: {output[:500]}...")
            checkpoint_plan = json.loads(output)

            # Validate the structure of the parsed JSON
            if not isinstance(checkpoint_plan, dict) or \
               "to_save" not in checkpoint_plan or \
               "skip_fields" not in checkpoint_plan or \
               not isinstance(checkpoint_plan["to_save"], dict) or \
               not isinstance(checkpoint_plan["skip_fields"], list):
                logger.warning(f"LLM checkpoint output has invalid structure. Output: {output[:500]}")
                raise ValueError("Invalid structure in checkpoint plan JSON")

            # Further validation (optional): check if 'value' and 'why_saved' exist in 'to_save' items
            for key, value_dict in checkpoint_plan["to_save"].items():
                 if not isinstance(value_dict, dict) or "value" not in value_dict or "why_saved" not in value_dict:
                      logger.warning(f"Invalid item structure in 'to_save' for key '{key}'. Item: {value_dict}")
                      raise ValueError(f"Invalid item structure for key '{key}' in 'to_save'")

            logger.info(f"Checkpoint plan received: {len(checkpoint_plan['to_save'])} fields to save, {len(checkpoint_plan['skip_fields'])} fields skipped.")
            return checkpoint_plan

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"LLM checkpoint planner output parsing or validation failed: {e}\nLLM output snippet: {output[:350]}", exc_info=True)
            # Fallback: save everything if parse fails or validation fails
            return {
                "to_save": {k: {"value": v, "why_saved": "Fallback: LLM output error"} for k, v in state_input.items()},
                "skip_fields": [f"LLM output error: {e}"]
            }
        except Exception as e:
            logger.error(f"Unexpected error running Checkpointing Planner Agent: {e}", exc_info=True)
            # Fallback in case of other errors during Runner.run
            return {
                "to_save": {k: {"value": v, "why_saved": "Fallback: Agent execution error"} for k, v in state_input.items()},
                "skip_fields": [f"Agent execution error: {e}"]
            }
