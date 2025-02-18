# logic/state_update_helper.py
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
            return row["update_payload"]  # asyncpg returns JSONB as a dict
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
    For the inventory_updates section, if the new update's 'added_items' (or 'removed_items')
    is empty but the old update has values, preserve the old values.
    """
    merged = new_update.copy()
    old_inv = old_update.get("inventory_updates", {})
    new_inv = new_update.get("inventory_updates", {})

    if not new_inv.get("added_items") and old_inv.get("added_items"):
        merged.setdefault("inventory_updates", {})["added_items"] = old_inv["added_items"]

    if not new_inv.get("removed_items") and old_inv.get("removed_items"):
        merged.setdefault("inventory_updates", {})["removed_items"] = old_inv["removed_items"]

    # Extend merge logic for other sections if needed
    return merged
