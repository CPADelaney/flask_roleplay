# lore/core/validation.py
"""
Contains specialized agents for validation tasks within the Canon.
"""
import logging
from agents import Agent, Runner
from typing import Dict, Any

logger = logging.getLogger(__name__)

class CanonValidationAgent:
    def __init__(self):
        self.agent = Agent(
            name="CanonDuplicateValidator",
            instructions=(
                "You are an expert data steward for a game's lore database. "
                "Your task is to determine if a new proposed entity is a semantic duplicate of an existing one. "
                "Base your final decision on the provided context. Respond with a single word: 'true' if it is a duplicate, 'false' if it is not."
            ),
            model="gpt-4.1-nano"  # Removed model_settings
        )

    async def confirm_is_duplicate_npc(self, conn, proposal: Dict[str, Any], existing_npc_id: int) -> bool:
        """
        Asks the LLM to make a final confirmation on whether two NPCs are the same person.
        """
        # Fetch the full details of the existing NPC for a rich comparison
        existing_npc = await conn.fetchrow("SELECT * FROM NPCStats WHERE npc_id = $1", existing_npc_id)
        if not existing_npc:
            return False # Can't be a duplicate if the existing one isn't found

        prompt = f"""
        I am considering creating a new NPC, but my system found a semantically similar existing NPC.
        Please determine if they are the same person.

        Proposed New NPC:
        - Name: "{proposal.get('name')}"
        - Role/Context: "{proposal.get('role')}"

        Most Similar Existing NPC:
        - ID: {existing_npc.get('npc_id')}
        - Name: "{existing_npc.get('npc_name')}"
        - Role/Context: {existing_npc.get('archetype_summary', 'No summary provided.')}

        Based on this information, are these the same individual described in two different ways?
        Answer with only the word 'true' or 'false'.
        """

        result = await Runner.run(self.agent, prompt)
        response_text = result.final_output.strip().lower()

        logger.debug(f"Validator agent response for NPC duplicate check: '{response_text}'")

        return response_text == 'true'
