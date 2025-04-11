# nyx/core/brain/checkpointing_agent.py

import re
import json
import datetime
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
    def __init__(self):
        self.agent = Agent(
            name="Checkpointing Planner",
            instructions=_CHECKPOINT_PROMPT,
            output_type=None,  # We want raw (LLM) output, not a Pydantic model
        )

    async def llm_periodic_checkpoint(state_input: dict) -> dict:
        """Wrapper function to handle periodic checkpointing via LLM."""
        agent = CheckpointingPlannerAgent()
        return await agent.recommend_checkpoint(state_input)  

    async def recommend_checkpoint(self, state_input: dict) -> dict:
        """
        Calls the LLM agent. Returns a dict with "to_save" and "skip_fields".
        """
        result = await Runner.run(self.agent, state_input)
        output = result.final_output.strip()
        # Remove code block if present
        if output.startswith("```"):
            output = re.sub(r"^```[a-z]*\s*", "", output)
            output = re.sub(r"```$", "", output)
        try:
            checkpoint_data = json.loads(output)
        except Exception as e:
            logger.warning(f"LLM checkpoint parser failed: {e}\nLLM output: {output[:350]}")
            # Fallback: save everything if parse fails
            return {
                "to_save": {k: {"value": v, "why_saved": "Fallback: LLM parser error"} for k, v in state_input.items()},
                "skip_fields": ["LLM output parse error"]
            }
        return checkpoint_data
