# nyx/core/brain/nyx_distributed_checkpoint.py

import os
import json
import uuid
import datetime
import logging
import re # Import re
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING

# Adapt these imports to your db wrapper
from db.connection import get_db_connection_context

# Import the planner agent if needed within the mixin (e.g., for type hints or potential future use)
from nyx.core.brain.checkpointing_agent import CheckpointingPlannerAgent # Optional here

# Type hint for NyxBrain without causing circular import
if TYPE_CHECKING:
    from nyx.core.brain.base import NyxBrain

logger = logging.getLogger(__name__)

NYX_ID = os.getenv("NYX_ID", "nyx_v1")
INSTANCE_ID = os.getenv("NYX_INSTANCE_ID", str(uuid.uuid4()))

class DistributedCheckpointMixin:
    """
    Mixin for distributed, agentic, mergeable checkpointing and restoration.
    Assumes self.gather_checkpoint_state() and self.restore_from_checkpoint(dict)
    are implemented in your NyxBrain class. Includes methods for saving full
    and planned (LLM-driven) checkpoints.
    """

    # ------- Saving --------

    async def save_full_checkpoint(self, event="periodic", merged_from=None, notes=None):
        """Saves the *entire* gathered state. Use for manual or merge checkpoints."""
        if not hasattr(self, 'gather_checkpoint_state'):
             logger.error("Cannot save checkpoint: gather_checkpoint_state method missing.")
             return

        state = await self.gather_checkpoint_state(event=event)
        checkpoint_time = datetime.datetime.utcnow()
        merged_from_list = merged_from or []
        notes_str = notes or ""

        try:
             # Use default=str to handle non-serializable types like datetime
             state_json = json.dumps(state, default=str)
        except (TypeError, OverflowError) as json_err:
             logger.error(f"Failed to serialize full state for checkpoint event '{event}': {json_err}", exc_info=True)
             state_json = json.dumps({"error": "Failed to serialize full state", "event": event})
             notes_str = f"Serialization Error: {json_err}\n{notes_str}"

        try:
            async with get_db_connection_context() as conn:
                # Ensure 'merged_from' column accepts array type (e.g., TEXT[]) in your DB schema
                await conn.execute("""
                    INSERT INTO nyx_brain_checkpoints (
                        nyx_id, instance_id, checkpoint_time, event, serialized_state, merged_from, notes
                    ) VALUES ($1, $2, $3, $4, $5, $6::TEXT[], $7)
                """,
                NYX_ID, INSTANCE_ID, checkpoint_time, event, state_json, merged_from_list, notes_str)
            logger.info(f"Full checkpoint saved for {NYX_ID} [instance={INSTANCE_ID}, event={event}]")
        except Exception as db_err:
            logger.error(f"Database error saving full checkpoint: {db_err}", exc_info=True)


    async def save_planned_checkpoint(self, event: str, data_to_save: dict, justifications: dict, skipped: list, merged_from=None, notes: Optional[str] = None):
        """
        Saves a checkpoint containing only the data selected by the CheckpointingPlannerAgent.
        """
        checkpoint_time = datetime.datetime.utcnow()
        merged_from_list = merged_from or []
        base_notes = notes or ""

        # Structure the notes to include justifications and skipped fields
        detailed_notes = f"Event: {event}\n{base_notes}\n--- JUSTIFICATIONS ---\n"
        # Ensure justifications is a dict before iterating
        if isinstance(justifications, dict):
            for key, reason in justifications.items():
                detailed_notes += f"- {key}: {reason}\n"
        else:
             detailed_notes += "[Justifications format error]\n"

        detailed_notes += f"\n--- SKIPPED FIELDS ({len(skipped)}) ---\n"
        # Ensure skipped is a list before iterating/slicing
        if isinstance(skipped, list):
            detailed_notes += "\n".join([f"- {item}" for item in skipped[:20]]) # Limit skipped list length in notes
            if len(skipped) > 20:
                 detailed_notes += "\n- ... (more fields skipped)"
        else:
             detailed_notes += "[Skipped fields format error]"


        # Serialize only the selected data ('value' part from the plan)
        state_to_serialize = {}
        if isinstance(data_to_save, dict):
             for key, value_dict in data_to_save.items():
                  if isinstance(value_dict, dict) and "value" in value_dict:
                       state_to_serialize[key] = value_dict["value"]
                  else:
                       # Log or handle cases where the structure isn't as expected
                       logger.warning(f"Unexpected structure for key '{key}' in data_to_save during planned checkpoint. Saving raw value.")
                       state_to_serialize[key] = value_dict # Save whatever was passed

        try:
            state_json = json.dumps(state_to_serialize, default=str)
        except (TypeError, OverflowError) as json_err:
             logger.error(f"Failed to serialize PLANNED state for checkpoint event '{event}': {json_err}", exc_info=True)
             state_json = json.dumps({"error": "Failed to serialize planned state", "event": event})
             detailed_notes = f"Serialization Error: {json_err}\n{detailed_notes}"

        try:
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    INSERT INTO nyx_brain_checkpoints (
                        nyx_id, instance_id, checkpoint_time, event, serialized_state, merged_from, notes
                    ) VALUES ($1, $2, $3, $4, $5, $6::TEXT[], $7)
                """,
                NYX_ID, INSTANCE_ID, checkpoint_time, f"{event}_planned", state_json, merged_from_list, detailed_notes[:2000]) # Limit notes length
            logger.info(f"LLM-planned checkpoint saved for {NYX_ID} [instance={INSTANCE_ID}, event={event}, fields={len(state_to_serialize)}]")
        except Exception as db_err:
            logger.error(f"Database error saving planned checkpoint: {db_err}", exc_info=True)


    # ------- Loading --------
    async def load_latest_checkpoints(self, lookback_mins=20):
        """
        Get latest unique checkpoints for global Nyx identity within N mins.
        Returns a list of asyncpg.Record objects.
        """
        recent_since = datetime.datetime.utcnow() - datetime.timedelta(minutes=lookback_mins)
        try:
            async with get_db_connection_context() as conn:
                # Fetch necessary columns explicitly
                rows = await conn.fetch("""
                    SELECT instance_id, checkpoint_time, serialized_state, merged_from, event, notes
                    FROM nyx_brain_checkpoints
                    WHERE nyx_id = $1 AND checkpoint_time > $2
                    ORDER BY checkpoint_time DESC
                """, NYX_ID, recent_since)

            # Take only latest entry per instance_id
            seen_by_instance = {}
            for row in rows:
                iid = row["instance_id"]
                # Check if row has checkpoint_time before comparing
                current_time = row.get("checkpoint_time")
                existing_time = seen_by_instance.get(iid, {}).get("checkpoint_time")

                if iid not in seen_by_instance or (current_time and existing_time and current_time > existing_time):
                    seen_by_instance[iid] = row
            return list(seen_by_instance.values())
        except Exception as db_err:
            logger.error(f"Database error loading latest checkpoints: {db_err}", exc_info=True)
            return [] # Return empty list on error


    async def maybe_merge_checkpoints(self, checkpoints: List[Any]) -> Optional[dict]:
        """
        Deserializes, potentially merges checkpoints via LLM, and returns the final state dict.
        """
        if not checkpoints:
            logger.info("No checkpoints provided for potential merge.")
            return None

        # Deserialize states safely
        deserialized_states = []
        valid_checkpoints_for_merge = [] # Keep track of original records for saving merge info
        for cp_record in checkpoints:
            try:
                # Ensure cp_record is a dict-like object (like asyncpg.Record)
                if not hasattr(cp_record, 'get'):
                    logger.warning(f"Skipping invalid checkpoint record format: {type(cp_record)}")
                    continue

                state_str = cp_record.get("serialized_state")
                instance_id = cp_record.get("instance_id", "unknown")
                event_type = cp_record.get("event", "unknown")

                if not state_str:
                     logger.warning(f"Checkpoint from {instance_id} (event: {event_type}) has empty state string, skipping.")
                     continue

                state_dict = json.loads(state_str)
                if isinstance(state_dict, dict): # Ensure it's a dictionary
                    deserialized_states.append(state_dict)
                    valid_checkpoints_for_merge.append(cp_record)
                else:
                    logger.warning(f"Checkpoint from {instance_id} (event: {event_type}) deserialized but was not a dict, skipping.")
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to deserialize checkpoint from {instance_id} (event: {event_type}): {e}")

        if not deserialized_states:
             logger.error("No valid checkpoints could be deserialized for merging or loading.")
             return None

        if len(deserialized_states) == 1:
            logger.info(f"Single valid checkpoint found from {valid_checkpoints_for_merge[0]['instance_id']}. Using its state.")
            return deserialized_states[0] # Return the deserialized dict

        # --- Multiple valid checkpoints -> Agentic Merge ---
        logger.warning(f"Multiple ({len(deserialized_states)}) valid checkpoints detected ({[c['instance_id'] for c in valid_checkpoints_for_merge]}), running agentic merge.")

        # Pass the list of deserialized state dictionaries to the merge function
        merged_state_dict, merge_notes = await self.agentic_merge_states(deserialized_states)

        if merged_state_dict: # Only save if merge was successful
            logger.info("Saving the result of the agentic merge as a new checkpoint.")
            merged_from_ids = [c["instance_id"] for c in valid_checkpoints_for_merge]
            # Save the merged state as a *full* checkpoint for future reference
            await self.save_full_checkpoint(
                event="merge_result",
                merged_from=merged_from_ids,
                notes=f"Agentic Merge Result:\n{merge_notes}"
            )
            return merged_state_dict # Return the newly merged state dict
        else:
            logger.error("Agentic merge failed to produce a valid merged state. Cannot proceed with restore.")
            return None # Indicate failure

    async def agentic_merge_states(
        self,
        states: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Merge divergent Nyx state dicts with the Responses API.
        """
        try:
            from openai import error as openai_err
            client = get_openai_client()
        except Exception:
            logger.error("OpenAI not configured; fallback merge.")
            return states[0] if states else None, "Fallback: OpenAI unavailable."
    
        # ── prep prompt ───────────────────────────────────────────────────────────
        state_blobs = []
        for i, st in enumerate(states):
            try:
                state_blobs.append(json.dumps(st, indent=2, sort_keys=True, default=str))
            except Exception as e:
                logger.error("State %d serialise error: %s", i, e)
                state_blobs.append(json.dumps({"error": "serialization-failed", "idx": i}))
    
        merge_instr = system_prompt.strip() + "\n" if system_prompt else ""
        merge_instr += (
            "You are Nyx, merging divergent checkpoints. "
            "Output ONLY JSON: "
            '{"merged_state":{...},"merge_notes":"..."}'
        )
    
        user_input = (
            "Merge these JSON checkpoint states:\n\n"
            + "\n\n--- CHECKPOINT SEPARATOR ---\n\n".join(state_blobs)
        )
    
        try:
            resp = await client.responses.create(
                model="gpt-4.1-nano",
                instructions=merge_instr,
                input=user_input,
                temperature=0.1,
                max_tokens=800,
                text={"format": {"type": "json_object"}},   # force JSON mode
            )
            raw_json = json.loads(resp.output_text)
            merged = raw_json.get("merged_state")
            if not isinstance(merged, dict):
                raise ValueError("missing/invalid merged_state")
            notes = raw_json.get("merge_notes", "")
            return merged, notes
    
        except openai_err.RateLimitError as rl:
            logger.error("Rate-limited during merge: %s", rl)
            return states[0] if states else None, "Fallback (rate-limit)."
        except Exception as e:
            logger.error("Agentic merge failed: %s", e, exc_info=True)
            return states[0] if states else None, f"Fallback merge error: {e}"


    # ------- Main restoration entrypoint --------
    async def restore_entity_from_distributed_checkpoints(self):
        """
        Loads latest checkpoints, merges if necessary, and calls the instance's
        restore_from_checkpoint method with the final state dictionary.
        """
        # Ensure restore_from_checkpoint exists on the class using this mixin (NyxBrain)
        if not hasattr(self, 'restore_from_checkpoint') or not callable(self.restore_from_checkpoint):
            logger.error("restore_from_checkpoint method is not implemented in the main class. Cannot restore.")
            return False

        logger.info(f"Restoring distributed Nyx entity state for {NYX_ID}...")
        try:
            # 1. Load latest checkpoint records from DB
            recent_checkpoint_records = await self.load_latest_checkpoints()

            # 2. Deserialize states and potentially merge them
            final_state_dict = await self.maybe_merge_checkpoints(recent_checkpoint_records)

            # 3. Check if a valid state was obtained
            if not final_state_dict: # Handles no checkpoints or merge/deserialization failure
                logger.info("No valid prior state found or loaded; booting fresh.")
                # Optionally run baseline initialization if needed when booting fresh
                if hasattr(self, 'initialize_baseline_personality') and callable(self.initialize_baseline_personality):
                    logger.info("Attempting to initialize baseline personality for fresh boot.")
                    await self.initialize_baseline_personality() # Assuming this method exists on NyxBrain or relevant system
                return False # Indicate that no state was *restored*

            # 4. Call the restore method implemented on NyxBrain
            # Ensure restore_from_checkpoint accepts a dictionary
            await self.restore_from_checkpoint(final_state_dict)

            logger.info(f"Restore complete for {NYX_ID} using merged/loaded state.")
            return True # Indicate successful restoration

        except Exception as e:
            logger.critical(f"CRITICAL ERROR during restore_entity_from_distributed_checkpoints for {NYX_ID}: {e}", exc_info=True)
            return False # Indicate restoration failed
