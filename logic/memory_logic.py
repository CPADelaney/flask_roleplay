# logic/memory_logic.py

import os
import json
import random
import logging
import psycopg2
from flask import Blueprint, request, jsonify, session
from db.connection import get_db_connection
from logic.chatgpt_integration import get_openai_client

logging.basicConfig(level=logging.DEBUG)

memory_bp = Blueprint('memory_bp', __name__)

@memory_bp.route('/get_current_roleplay', methods=['GET'])
def get_current_roleplay():
    """
    Returns an array of {key, value} objects from CurrentRoleplay,
    scoped to user_id + conversation_id.
    The front-end or route call must pass ?conversation_id=XX or use session/headers.
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "No conversation_id provided"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT key, value
            FROM currentroleplay
            WHERE user_id=%s AND conversation_id=%s
            ORDER BY key
        """, (user_id, conversation_id))
        rows = cursor.fetchall()
        data = [{"key": r[0], "value": r[1]} for r in rows]
        return jsonify(data), 200
    finally:
        conn.close()

def record_npc_event(user_id, conversation_id, npc_id, event_description):
    """
    Appends a new event to the NPC's memory field for a given user_id + conversation_id.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE NPCStats
            SET memory = COALESCE(memory, '[]'::jsonb) || to_jsonb(%s::text)
            WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
            RETURNING memory
        """, (event_description, npc_id, user_id, conversation_id))
        updated_row = cursor.fetchone()
        if not updated_row:
            logging.warning(f"NPC with ID={npc_id} (user_id={user_id}, conversation_id={conversation_id}) not found.")
        else:
            logging.info(f"Updated memory for NPC {npc_id} => {updated_row[0]}")
        conn.commit()
    except psycopg2.Error as e:
        conn.rollback()
        logging.error(f"Error recording NPC event: {e}")
    finally:
        conn.close()

@memory_bp.route('/store_roleplay_segment', methods=['POST'], endpoint="store_roleplay_segment_endpoint")
def store_roleplay_segment():
    """
    Stores or updates a key-value pair in the CurrentRoleplay table,
    scoped to user_id + conversation_id.
    The payload should include:
      { "conversation_id": X, "key": "abc", "value": "..." }
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    try:
        payload = request.get_json() or {}
        conversation_id = payload.get("conversation_id")
        segment_key = payload.get("key")
        segment_value = payload.get("value")

        if not conversation_id:
            return jsonify({"error": "No conversation_id provided"}), 400
        if not segment_key or segment_value is None:
            return jsonify({"error": "Missing 'key' or 'value'"}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO currentroleplay (user_id, conversation_id, key, value)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, (user_id, conversation_id, segment_key, segment_value))
        conn.commit()
        return jsonify({"message": "Stored successfully"}), 200
    except Exception as e:
        logging.error(f"Error in store_roleplay_segment: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if 'conn' in locals():
            conn.close()

@memory_bp.route('/update_npc_memory', methods=['POST'])
def update_npc_memory():
    """
    Accepts a JSON payload containing:
      {
         "conversation_id": X,
         "npc_id": Y,
         "relationship": {
             "type": "mother",
             "target": "player",
             "target_name": "Chase"
         }
      }
    Retrieves the NPC's name along with their synthesized archetype fields,
    then calls GPT to generate a shared memory.
    The generated memory is appended to the NPC's memory field.
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    payload = request.get_json() or {}
    conversation_id = payload.get("conversation_id")
    npc_id = payload.get("npc_id")
    relationship = payload.get("relationship")
    if not conversation_id or not npc_id or not relationship:
        return jsonify({"error": "Missing conversation_id, npc_id, or relationship data"}), 400

    # Retrieve the NPC's name and synthesized archetype fields
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT npc_name, archetype_summary, archetype_extras_summary
        FROM NPCStats
        WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
    """, (npc_id, user_id, conversation_id))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": f"NPC with id {npc_id} not found"}), 404
    npc_name, archetype_summary, archetype_extras_summary = row
    conn.close()

    # Generate a shared memory using GPT, now including the NPC's background details.
    from logic.memory import get_shared_memory
    memory_text = get_shared_memory(user_id, conversation_id, relationship, npc_name, archetype_summary or "", archetype_extras_summary or "")
    try:
        record_npc_event(user_id, conversation_id, npc_id, memory_text)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"message": "NPC memory updated", "memory": memory_text}), 200


async def get_stored_setting(conn, user_id, conversation_id):
    # Retrieve the setting name and description from CurrentRoleplay.
    row = await conn.fetchrow(
        "SELECT key, value FROM CurrentRoleplay WHERE user_id=$1 AND conversation_id=$2 AND key IN ('CurrentSetting', 'EnvironmentDesc')",
        user_id, conversation_id
    )
    # If you expect both keys, you might need to run two separate queries or fetch all rows.
    # Here's one approach:
    rows = await conn.fetch(
        "SELECT key, value FROM CurrentRoleplay WHERE user_id=$1 AND conversation_id=$2 AND key IN ('CurrentSetting', 'EnvironmentDesc')",
        user_id, conversation_id
    )
    result = {r["key"]: r["value"] for r in rows}
    # Fallbacks if not found:
    result.setdefault("CurrentSetting", "Default Setting Name")
    result.setdefault("EnvironmentDesc", "Default environment description.")
    return result

def propagate_shared_memories(user_id, conversation_id, source_npc_id, source_npc_name, memories):
    """
    For each memory in 'memories':
      1) Check if it references the name of any *other* NPC in this conversation.
      2) If so, call record_npc_event(...) to add that memory to that NPC's memory as well.
    """
    if not memories:
        return  # no new memories => nothing to do

    # 1) Build a map of { npc_name_lower: npc_id }
    #    for all NPCs in this conversation.
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT npc_id, LOWER(npc_name)
        FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s
    """, (user_id, conversation_id))
    rows = cursor.fetchall()
    conn.close()

    name_to_id_map = {}
    for (other_id, other_name_lower) in rows:
        name_to_id_map[other_name_lower] = other_id

    # 2) For each memory text, see if it references another npc's name
    for mem_text in memories:
        # Let's do naive substring matching:
        mem_text_lower = mem_text.lower()

        for (other_npc_name_lower, other_npc_id) in name_to_id_map.items():
            if other_npc_id == source_npc_id:
                continue  # don't replicate to self if you don't want that

            # If the memory references that NPC's name
            # (maybe it also references the source NPC, but that's expected)
            if other_npc_name_lower in mem_text_lower:
                # We found a reference => replicate memory
                # Use your existing record_npc_event
                record_npc_event(user_id, conversation_id, other_npc_id, mem_text)

def fetch_formatted_locations(user_id, conversation_id):
    """
    Query the Locations table for the given user_id and conversation_id,
    then format each location into a bullet list string with a truncated description.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        query = """
            SELECT location_name, description
            FROM Locations
            WHERE user_id = %s AND conversation_id = %s
        """
        cursor.execute(query, (user_id, conversation_id))
        rows = cursor.fetchall()
        
        formatted = ""
        for loc in rows:
            location_name = loc[0]
            # If description exists and is longer than 80 characters, truncate it.
            if loc[1]:
                description = loc[1][:80] + "..." if len(loc[1]) > 80 else loc[1]
            else:
                description = "No description"
            formatted += f"- {location_name}: {description}\n"
        return formatted
    except Exception as e:
        logging.error(f"[fetch_formatted_locations] Error fetching locations: {e}")
        return "No location data available."
    finally:
        conn.close()


def get_shared_memory(user_id, conversation_id, relationship, npc_name, archetype_summary="", archetype_extras_summary=""):
    """
    Given a relationship dict and the NPC's name, returns a shared memory.
    It uses the stored setting from CurrentRoleplay (keys 'CurrentSetting' and 'EnvironmentDesc')
    and includes current location data from the Locations table.
    """
    # Fetch stored environment details from CurrentRoleplay.
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT key, value FROM CurrentRoleplay 
            WHERE user_id=%s AND conversation_id=%s 
              AND key IN ('CurrentSetting', 'EnvironmentDesc')
        """, (user_id, conversation_id))
        rows = cursor.fetchall()
        stored = {row[0]: row[1] for row in rows}
        mega_description = stored.get("EnvironmentDesc", "an undefined setting")
    except Exception as e:
        logging.error(f"[get_shared_memory] Error retrieving stored setting: {e}")
        mega_description = "an undefined setting"
    finally:
        conn.close()
    
    # Fetch and format current locations.
    locations_table_formatted = fetch_formatted_locations(user_id, conversation_id)
    
    target = relationship.get("target", "player")
    target_name = relationship.get("target_name", "the player")
    rel_type = relationship.get("type", "related")
    
    extra_context = ""
    if archetype_summary:
        extra_context += f"Background: {archetype_summary}. "
    if archetype_extras_summary:
        extra_context += f"Extra Details: {archetype_extras_summary}. "
    
    system_instructions = f"""
The NPC {npc_name} has a pre-existing relationship as a {rel_type} with {target_name}.
The current setting is: {mega_description}.
{extra_context}
Current established locations are:
{locations_table_formatted}

Generate three first-person memories from {npc_name}'s perspective about their relationship with {target_name}. These memories should:

1. Be 2–3 sentences each, written in {npc_name}'s authentic voice.
2. Show different stages of their relationship (early, middle, recent).
3. Each take place in a specific location chosen from the locations list above or a location that would make sense in this setting.
4. Include at least one sensory detail (sight, sound, smell, taste, touch).
5. Very subtley foreshadows the characters in the context of a future femdom dynamic.
6. Show an emotional reaction from both characters.
7. Include a small consequence or change in the relationship from each interaction.

Each memory should feel personal, impactful, concise, vivid, and naturally fit within the setting.
Memories can be either positive or negative.
To determine the nature of their relationship, consider {npc_name}'s likes, dislikes, affiliations, and schedule, and develop a plausible connection to {target_name}.
"""
    messages = [{"role": "system", "content": system_instructions}]
    
    try:
        response = get_openai_client().chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"[get_shared_memory] GPT error: {e}")
        return "Shared memory could not be generated."
