# logic/state_update_helper.py
import os
import json
import asyncpg
from datetime import datetime
import logging

async def get_previous_update(user_id: int, conversation_id: int) -> dict:
    """
    Retrieve the previously stored state update payload from the StateUpdates table.
    If none exists, return an empty dictionary.
    """
    dsn = os.getenv("DB_DSN")
    conn = await asyncpg.connect(dsn=dsn)
    try:
        row = await conn.fetchrow("""
            SELECT update_payload
            FROM StateUpdates
            WHERE user_id = $1 AND conversation_id = $2
        """, user_id, conversation_id)
        if row and row["update_payload"]:
            payload = row["update_payload"]
            # If payload is a string, convert it to a dict.
            if isinstance(payload, str):
                try:
                    return json.loads(payload)
                except Exception as e:
                    logging.error("Failed to parse stored update_payload: %s", e)
                    return {}
            else:
                return payload  # Already a dict
        else:
            return {}
    finally:
        await conn.close()

async def store_state_update(user_id: int, conversation_id: int, update_payload: dict) -> None:
    """
    Store the merged state update payload into the StateUpdates table.
    If a record already exists, update it.
    """
    dsn = os.getenv("DB_DSN")
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute("""
            INSERT INTO StateUpdates (user_id, conversation_id, update_payload)
            VALUES ($1, $2, $3)
            ON CONFLICT (user_id, conversation_id)
            DO UPDATE SET update_payload = $3, updated_at = CURRENT_TIMESTAMP
        """, user_id, conversation_id, json.dumps(update_payload))
    finally:
        await conn.close()

def merge_state_updates(old_update: dict, new_update: dict) -> dict:
    """
    Merge two state update payloads.
    
    For the inventory_updates section:
      - If new_update explicitly includes a key (even if its value is an empty list),
        then use that value.
      - Otherwise, fallback to the value from old_update.
    """
    merged = new_update.copy()
    old_inv = old_update.get("inventory_updates", {})
    new_inv = new_update.get("inventory_updates", {})

    merged_inv = {}
    for key in ["added_items", "removed_items"]:
        if key in new_inv:
            merged_inv[key] = new_inv[key]
        elif key in old_inv:
            merged_inv[key] = old_inv[key]
    if merged_inv:
        # Also ensure that the player_name is set correctly.
        merged["inventory_updates"] = {
            "player_name": new_inv.get("player_name", old_inv.get("player_name", "Chase")),
            **merged_inv
        }
    return merged
