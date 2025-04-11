# nyx/core/brain/nyx_distributed_checkpoint.py

import os
import json
import uuid
import datetime
import logging

# Adapt these imports to your db wrapper
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

NYX_ID = os.getenv("NYX_ID", "nyx_v1")
INSTANCE_ID = os.getenv("NYX_INSTANCE_ID", str(uuid.uuid4()))

class DistributedCheckpointMixin:
    """
    Mixin for distributed, agentic, mergeable checkpointing and restoration.
    Assumes self.gather_checkpoint_state() and self.restore_from_checkpoint(_dict_)
    are implemented in your NyxBrain class.
    """

    # ------- Saving --------
    async def save_checkpoint(self, event="periodic", merged_from=None, notes=None):
        state = await self.gather_checkpoint_state(event=event)
        checkpoint_time = datetime.datetime.utcnow()
        merged_from = merged_from or []
        notes = notes or ""

        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO nyx_brain_checkpoints (
                    nyx_id, instance_id, checkpoint_time, event, serialized_state, merged_from, notes
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            NYX_ID, INSTANCE_ID, checkpoint_time, event, json.dumps(state), merged_from, notes)
        logger.info(f"Checkpoint saved for {NYX_ID} [instance={INSTANCE_ID}, event={event}]")

    # ------- Loading --------
    async def load_latest_checkpoints(self, lookback_mins=20):
        """
        Get latest unique checkpoints for global Nyx identity within N mins.
        Returns a list of asyncpg.Record objects.
        """
        recent_since = datetime.datetime.utcnow() - datetime.timedelta(minutes=lookback_mins)
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT * FROM nyx_brain_checkpoints
                WHERE nyx_id = $1 AND checkpoint_time > $2
                ORDER BY checkpoint_time DESC
            """, NYX_ID, recent_since)
        # Take only latest entry per instance_id
        seen_by_instance = {}
        for row in rows:
            iid = row["instance_id"]
            if iid not in seen_by_instance:
                seen_by_instance[iid] = row
        return list(seen_by_instance.values())

    async def maybe_merge_checkpoints(self, checkpoints):
        """
        Returns either a dict (merged or loaded state), or None.
        If multiple divergent checkpoints, calls agentic_merge_states.
        """
        if not checkpoints:
            logger.info("No checkpoints to load.")
            return None
        if len(checkpoints) == 1:
            logger.info(f"Single checkpoint found from {checkpoints[0]['instance_id']}")
            return checkpoints[0]["serialized_state"]

        # Multiple recent checkpoints, run agentic merge
        logger.warning(f"Multiple checkpoints detected ({[c['instance_id'] for c in checkpoints]}), running agentic merge.")
        states = [c["serialized_state"] for c in checkpoints]
        merged_state, merge_notes = await self.agentic_merge_states(states)
        merged_from = [c["instance_id"] for c in checkpoints]
        await self.save_checkpoint(event="merge", merged_from=merged_from, notes=merge_notes)
        return merged_state

    async def agentic_merge_states(self, states, system_prompt=None):
        """
        Given a list of JSON state dicts, call an LLM to merge them.
        Uses OpenAI ChatCompletion as an example.
        Returns merged_state (dict), notes (str).
        """
        import openai  # Or Anthropic, etc

        # Ensure dicts
        state_jsons = [
            json.dumps(s, indent=2, sort_keys=True) if isinstance(s, dict) else str(s)
            for s in states
        ]

        merge_instructions = """
You are Nyx, an intelligent agent who has been running in parallel on multiple servers and must self-merge your split states into one. The JSON below are your brain states from these divergent runs. 
- Identify the most recent/important information from each.
- For emotions, goals, and needs: reconcile any conflicts as YOU would—if they clash, choose the one from the most eventful/important branch, or blend as you see appropriate.
- For memories/diary, merge all unique events.
- For any subtle differences (e.g. hormone levels), adopt the most "active"/alert state unless instructed otherwise.
- At the end, provide a short note explaining your merge reasoning.
Respond ONLY with a JSON object with two fields:
{
  "merged_state": { ...new state dict... },
  "merge_notes": "explain what you did and why. Note any lost/conflicting elements."
}
"""
        prompt = system_prompt + "\n" + merge_instructions if system_prompt else merge_instructions

        user_content = (
            "Merge these brain checkpoints (as JSON), resulting in the best-possible self-continuity:\n\n"
            + "\n\n".join(state_jsons)
        )

        try:
            completion = await openai.ChatCompletion.acreate(
                model="gpt-4o",  # or "gpt-4"
                messages=[
                    {"role": "system", "content": prompt.strip()},
                    {"role": "user", "content": user_content}
                ],
                temperature=0,
            )
            merged_obj = json.loads(completion.choices[0].message.content)
            logger.info("Merged state created via LLM.")
            return merged_obj["merged_state"], merged_obj.get("merge_notes", "")
        except Exception as e:
            logger.error(f"Agentic merge via OpenAI failed: {e}", exc_info=True)
            # Fallback: use most recent
            fallback = states[0] if isinstance(states[0], dict) else json.loads(states[0])
            return fallback, f"Auto-fallback to newest: {e}"

    # ------- Main restoration entrypoint --------
    async def restore_entity_from_distributed_checkpoints(self):
        """
        Call this ONCE during brain boot/init. Will perform merge if needed.
        """
        logger.info("Restoring distributed Nyx entity state...")
        recents = await self.load_latest_checkpoints()
        merged = await self.maybe_merge_checkpoints(recents)
        if not merged:
            logger.info("No prior state restored; booting fresh.")
            return False
        state = merged if isinstance(merged, dict) else json.loads(merged)
        await self.restore_from_checkpoint(state)
        logger.info("Restore complete.")
        return True
