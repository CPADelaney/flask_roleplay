import json
from typing import Any, Dict

from logic.aggregator_sdk import get_aggregated_roleplay_context
from logic.universal_updater_agent import (
    UniversalUpdaterContext,
    apply_universal_updates_async,
    convert_updates_for_database,
)
from db.connection import get_db_connection_context


class RoleplayEngine:
    """Unified orchestrator for roleplay turns.

    This engine retrieves context, generates narrative plus structured
    updates in a single LLM call, and applies those updates back to the
    database. The goal is to minimize round trips and keep canon in sync.
    """

    async def prepare_context(
        self, user_id: int, conversation_id: int, player_name: str
    ) -> Dict[str, Any]:
        """Fetch aggregated context for the current turn."""
        return await get_aggregated_roleplay_context(
            user_id, conversation_id, player_name
        )

    async def generate_turn(
        self,
        user_id: int,
        conversation_id: int,
        player_name: str,
        player_input: str,
    ) -> Dict[str, Any]:
        """Generate narrative and apply updates for a single turn."""
        context = await self.prepare_context(user_id, conversation_id, player_name)

        prompt = (
            "You are the roleplay engine. Using the provided context and "
            "player input, continue the story and specify any game state "
            "updates in JSON. Return an object with keys 'narrative' and "
            "'updates'.\n\n"
            f"Context: {json.dumps(context)}\n"
            f"Player Input: {player_input}"
        )
        from logic.gpt_utils import call_gpt_json

        result = await call_gpt_json(conversation_id, context, prompt)
        narrative = result.get("narrative", "")
        updates = result.get("updates", {})

        if updates:
            await self.apply_updates(user_id, conversation_id, updates)

        return {"narrative": narrative, "updates": updates}

    async def apply_updates(
        self, user_id: int, conversation_id: int, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply structured updates to the database."""
        ctx = UniversalUpdaterContext(user_id, conversation_id)
        await ctx.initialize()

        db_updates = convert_updates_for_database(updates)
        async with get_db_connection_context() as conn:
            return await apply_universal_updates_async(
                ctx, user_id, conversation_id, db_updates, conn
            )
