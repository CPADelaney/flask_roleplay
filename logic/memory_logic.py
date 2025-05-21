# logic/memory_logic.py

import os
import json
import random
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
# Removed: import psycopg2
import asyncpg # Added
import asyncio # Added (potentially needed for to_thread if OpenAI client is sync)
from quart import Blueprint, request, jsonify, session
from contextlib import asynccontextmanager # Added just in case, though not used directly here
from db.connection import get_db_connection_context
from logic.chatgpt_integration import get_openai_client # Ensure this provides an ASYNC client or use asyncio.to_thread

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__) # Use module-specific logger

memory_bp = Blueprint('memory_bp', __name__)

@memory_bp.route('/get_current_roleplay', methods=['GET'])
async def get_current_roleplay(): # Changed to async def
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

    try:
        async with get_db_connection_context() as conn: # Use async context manager
            rows = await conn.fetch("""
                SELECT key, value
                FROM currentroleplay
                WHERE user_id=$1 AND conversation_id=$2
                ORDER BY key
            """, user_id, conversation_id) # Use $ placeholders and conn.fetch
            data = [{"key": r['key'], "value": r['value']} for r in rows] # Access by key
            return jsonify(data), 200
    except asyncpg.PostgresError as e:
        logger.error(f"Database error in get_current_roleplay: {e}", exc_info=True)
        return jsonify({"error": "Database error"}), 500
    except ConnectionError as e:
        logger.error(f"Pool error in get_current_roleplay: {e}", exc_info=True)
        return jsonify({"error": "Could not connect to database"}), 503
    except asyncio.TimeoutError:
        logger.error("Timeout getting DB connection in get_current_roleplay", exc_info=True)
        return jsonify({"error": "Database timeout"}), 504
    except Exception as e:
        logger.error(f"Unexpected error in get_current_roleplay: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500



async def record_npc_event(user_id: int, conversation_id: int, npc_id: int, event_description: str): # Changed to async def
    """
    Appends a new event to the NPC's memory field (JSONB array) for a given user_id + conversation_id.
    """
    try:
        async with get_db_connection_context() as conn:
            # asyncpg handles JSON serialization automatically for jsonb columns usually
            # Ensure event_description is a serializable object/string
            # Using json.dumps explicitly can ensure it's valid JSON text for the || operator
            event_json_text = json.dumps(event_description)

            # Using a transaction isn't strictly necessary for a single UPDATE
            # but good practice if logic becomes more complex
            # async with conn.transaction():
            updated_row = await conn.fetchrow("""
                UPDATE NPCStats
                SET memory = COALESCE(memory, '[]'::jsonb) || $1::jsonb
                WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
                RETURNING memory
            """, event_json_text, npc_id, user_id, conversation_id) # Use $ placeholders

            if not updated_row:
                logger.warning(f"Attempted to record event for non-existent NPC ID={npc_id} (user_id={user_id}, conversation_id={conversation_id}).")
            else:
                # Be careful logging potentially large memory fields
                logger.info(f"Recorded event for NPC {npc_id}. New memory field length (if list): {len(updated_row['memory']) if isinstance(updated_row['memory'], list) else 'N/A'}")

    except asyncpg.PostgresError as e:
        logger.error(f"Database error recording NPC event for NPC {npc_id}: {e}", exc_info=True)
        # No explicit rollback needed, context manager handles connection release
    except ConnectionError as e:
        logger.error(f"Pool error recording NPC event: {e}", exc_info=True)
    except asyncio.TimeoutError:
        logger.error(f"Timeout getting DB connection for recording NPC event", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error recording NPC event: {e}", exc_info=True)

@memory_bp.route('/store_roleplay_segment', methods=['POST'], endpoint="store_roleplay_segment_endpoint")
async def store_roleplay_segment(): # Changed to async def
    """
    Stores or updates a key-value pair in the CurrentRoleplay table,
    scoped to user_id + conversation_id.
    Payload: { "conversation_id": X, "key": "abc", "value": "..." }
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    try:
        payload = request.get_json() or {}
        conversation_id = payload.get("conversation_id")
        segment_key = payload.get("key")
        segment_value = payload.get("value") # Can be any JSON-serializable type

        if not conversation_id:
            return jsonify({"error": "No conversation_id provided"}), 400
        if not segment_key or segment_value is None: # Check for None explicitly
            return jsonify({"error": "Missing 'key' or 'value'"}), 400

        async with get_db_connection_context() as conn:
            # Value can be stored directly if column type is appropriate (e.g., TEXT, JSONB)
            # If value is complex and column is TEXT, use json.dumps(segment_value)
            await conn.execute("""
                INSERT INTO currentroleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = EXCLUDED.value
            """, user_id, conversation_id, segment_key, segment_value) # Use $ placeholders
            # No commit needed for single statement outside explicit transaction

        return jsonify({"message": "Stored successfully"}), 200
    except asyncpg.PostgresError as e:
        logger.error(f"Database error in store_roleplay_segment: {e}", exc_info=True)
        return jsonify({"error": "Database error"}), 500
    except ConnectionError as e:
        logger.error(f"Pool error in store_roleplay_segment: {e}", exc_info=True)
        return jsonify({"error": "Could not connect to database"}), 503
    except asyncio.TimeoutError:
        logger.error("Timeout getting DB connection in store_roleplay_segment", exc_info=True)
        return jsonify({"error": "Database timeout"}), 504
    except Exception as e: # Catch potential JSON errors from request.get_json() too
        logger.error(f"Error in store_roleplay_segment: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred"}), 500


@memory_bp.route('/update_npc_memory', methods=['POST'])
async def update_npc_memory(): # Changed to async def
    """
    Generates and stores a shared memory for an NPC based on relationship and context.
    Payload: { "conversation_id": X, "npc_id": Y, "relationship": {...} }
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

    try:
        # Retrieve the NPC's name and synthesized archetype fields
        npc_name = None
        archetype_summary = None
        archetype_extras_summary = None
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT npc_name, archetype_summary, archetype_extras_summary
                FROM NPCStats
                WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
            """, npc_id, user_id, conversation_id) # Use $ placeholders
            if not row:
                return jsonify({"error": f"NPC with id {npc_id} not found"}), 404
            npc_name = row['npc_name']
            archetype_summary = row['archetype_summary']
            archetype_extras_summary = row['archetype_extras_summary']

        # Generate a shared memory using GPT
        # This now calls the async version of get_shared_memory
        memory_json_str = await get_shared_memory(
            user_id,
            conversation_id,
            relationship,
            npc_name,
            archetype_summary or "",
            archetype_extras_summary or ""
        )

        if not memory_json_str:
             return jsonify({"error": "Failed to generate NPC memory via AI"}), 500

        try:
             memory_data = json.loads(memory_json_str)
             memories_list = memory_data.get("memory", [])
        except json.JSONDecodeError:
             logger.error(f"Failed to decode JSON memory from GPT for NPC {npc_id}: {memory_json_str[:200]}...")
             return jsonify({"error": "AI returned invalid memory format"}), 500

        if not memories_list:
            logger.warning(f"AI returned empty memory list for NPC {npc_id}")
            return jsonify({"message": "No new memories generated", "memory": []}), 200 # Or maybe 204 No Content

        # Store each generated memory using the MemoryManager (assuming it handles individual additions)
        # Or adapt record_npc_event if you want to append the whole list structure
        memory_added_count = 0
        for mem_text in memories_list:
            if isinstance(mem_text, str) and mem_text.strip():
                 # Using MemoryManager.add_memory which is now async
                success = await MemoryManager.add_memory(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    entity_id=npc_id,
                    entity_type="npc",
                    memory_text=mem_text,
                    memory_type=MemoryType.EMOTIONAL, # Or determine type based on content?
                    significance=MemorySignificance.MEDIUM, # Or determine significance?
                    tags=["shared_memory", f"related_to:{relationship.get('target_name', 'player')}"]
                 )
                if success:
                    memory_added_count += 1
                # Optionally: Propagate these new memories if needed
                # propagate_shared_memories might need adjustment based on how memories are added now
                await propagate_shared_memories(user_id, conversation_id, npc_id, npc_name, [mem_text])

        logger.info(f"Added {memory_added_count}/{len(memories_list)} generated memories for NPC {npc_id}")
        return jsonify({"message": f"NPC memory updated with {memory_added_count} entries", "memory_preview": memories_list[0] if memories_list else None}), 200

    # Specific exceptions first
    except asyncpg.PostgresError as e:
        logger.error(f"Database error in update_npc_memory for NPC {npc_id}: {e}", exc_info=True)
        return jsonify({"error": "Database error during memory update"}), 500
    except ConnectionError as e:
        logger.error(f"Pool error in update_npc_memory: {e}", exc_info=True)
        return jsonify({"error": "Could not connect to database"}), 503
    except asyncio.TimeoutError:
        logger.error(f"Timeout getting DB connection in update_npc_memory", exc_info=True)
        return jsonify({"error": "Database timeout during memory update"}), 504
    except Exception as e: # Catch errors from get_shared_memory, json parsing, etc.
        logger.error(f"Error in update_npc_memory for NPC {npc_id}: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during memory update"}), 500



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

async def propagate_shared_memories(user_id: int, conversation_id: int, source_npc_id: int, source_npc_name: str, memories: list[str]): # Changed to async def
    """
    For each memory text, check if it mentions other NPCs in the conversation
    and add the memory to their log using MemoryManager.add_memory.
    """
    if not memories:
        return

    name_to_id_map = {}
    try:
        async with get_db_connection_context() as conn:
            # 1) Build map of { npc_name_lower: npc_id }
            rows = await conn.fetch("""
                SELECT npc_id, LOWER(npc_name) as name_lower
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2
            """, user_id, conversation_id)
            name_to_id_map = {r['name_lower']: r['npc_id'] for r in rows}

        if not name_to_id_map:
            logger.warning(f"No NPCs found to propagate memories in conv {conversation_id}")
            return

        # 2) Check each memory against other NPCs
        propagation_count = 0
        for mem_text in memories:
            mem_text_lower = mem_text.lower()
            for other_npc_name_lower, other_npc_id in name_to_id_map.items():
                # Don't replicate to self unless intended
                if other_npc_id == source_npc_id:
                    continue

                # Simple substring check (can be improved with NLP/NER)
                if other_npc_name_lower in mem_text_lower:
                    logger.info(f"Propagating memory from NPC {source_npc_id} ({source_npc_name}) to NPC {other_npc_id} ({other_npc_name_lower}) based on name match.")
                    # Add as an "overheard" or "secondhand" memory using MemoryManager
                    success = await MemoryManager.add_memory(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        entity_id=other_npc_id,
                        entity_type="npc",
                        memory_text=f"I heard something about {source_npc_name}: \"{mem_text}\"", # Reformulate as secondhand info
                        memory_type=MemoryType.OBSERVATION, # Treat as observation/info
                        significance=MemorySignificance.LOW, # Lower significance than direct experience
                        tags=["propagated", "secondhand", f"from_npc:{source_npc_id}"]
                    )
                    if success:
                         propagation_count += 1

        if propagation_count > 0:
             logger.info(f"Propagated {propagation_count} memories in conv {conversation_id}.")

    except asyncpg.PostgresError as e:
        logger.error(f"Database error during memory propagation in conv {conversation_id}: {e}", exc_info=True)
    except ConnectionError as e:
        logger.error(f"Pool error during memory propagation: {e}", exc_info=True)
    except asyncio.TimeoutError:
        logger.error(f"Timeout getting DB connection during memory propagation", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error during memory propagation: {e}", exc_info=True)

async def fetch_formatted_locations(user_id: int, conversation_id: int) -> str: # Changed to async def
    """
    Query Locations table and format results into a bulleted string.
    """
    formatted = ""
    try:
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT location_name, description
                FROM Locations
                WHERE user_id = $1 AND conversation_id = $2
            """, user_id, conversation_id)

            if not rows:
                return "No locations found.\n" # Return specific message

            for loc in rows:
                location_name = loc['location_name']
                desc = loc['description']
                if desc:
                    description = desc[:80] + "..." if len(desc) > 80 else desc
                else:
                    description = "No description"
                formatted += f"- {location_name}: {description}\n"
            return formatted if formatted else "No locations found.\n"

    except asyncpg.PostgresError as e:
        logger.error(f"[fetch_formatted_locations] DB error: {e}", exc_info=True)
        return "Error retrieving location data.\n"
    except ConnectionError as e:
        logger.error(f"[fetch_formatted_locations] Pool error: {e}", exc_info=True)
        return "Error connecting to database for locations.\n"
    except asyncio.TimeoutError:
        logger.error(f"[fetch_formatted_locations] Timeout error", exc_info=True)
        return "Timeout retrieving location data.\n"
    except Exception as e:
        logger.error(f"[fetch_formatted_locations] Unexpected error: {e}", exc_info=True)
        return "Error processing location data.\n"


async def get_shared_memory(user_id: int, conversation_id: int, relationship: dict, 
                           npc_name: str, archetype_summary: str = "", 
                           archetype_extras_summary: str = "") -> Optional[str]:
    """
    Generates shared memory text using GPT, incorporating DB lookups.
    
    Args:
        user_id: User ID for database context
        conversation_id: Conversation ID for database context
        relationship: Dictionary containing relationship details
        npc_name: Name of the NPC
        archetype_summary: Summary of NPC archetype (optional)
        archetype_extras_summary: Additional archetype details (optional)
        
    Returns:
        JSON string containing generated memories, or None if generation fails
    """
    logger.info(f"Starting get_shared_memory for NPC '{npc_name}' with relationship: {relationship}")

    mega_description = "an undefined setting"
    current_setting = "Default Setting Name"
    locations_table_formatted = "No location data available.\n"

    try:
        # Fetch stored environment details within a single context
        async with get_db_connection_context() as conn:
            logger.debug("Fetching stored environment details...")
            # Use the existing async helper function, passing the connection
            stored_settings = await get_stored_setting(conn, user_id, conversation_id)
            mega_description = stored_settings.get("EnvironmentDesc", "an undefined setting")
            current_setting = stored_settings.get("CurrentSetting", "Default Setting Name")
            logger.info(f"Retrieved environment desc (first 100): {mega_description[:100]}...")
            logger.info(f"Current setting: {current_setting}")

        # Fetch formatted locations
        logger.debug("Fetching and formatting current locations...")
        locations_table_formatted = await fetch_formatted_locations(user_id, conversation_id)
        logger.info(f"Formatted locations retrieved:\n{locations_table_formatted}")

    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logger.error(f"Database error fetching context for get_shared_memory: {db_err}", exc_info=True)
        # Continue with default descriptions

    except Exception as e:
        logger.error(f"Unexpected error fetching context in get_shared_memory: {e}", exc_info=True)
        # Continue with default descriptions

    target = relationship.get("target", "player")
    target_name = relationship.get("target_name", "the player")
    rel_type = relationship.get("type", "related")

    extra_context = ""
    if archetype_summary:
        extra_context += f"Background: {archetype_summary}. "
    if archetype_extras_summary:
        extra_context += f"Extra Details: {archetype_extras_summary}. "
    
    # Create an example of the expected output format
    example_output = {
        "memory": [
            f"I still remember the first time {target_name} and I crossed paths at the marketplace. The scent of fresh spices hung in the air as our eyes met briefly over a display of exotic fruits, creating an unexpected moment of connection in the midst of the bustling crowd.",
            f"Last month when {target_name} challenged my authority during the council meeting, I felt my temper flare dangerously. The tension in the room was palpable as our voices rose, but somehow that confrontation led to a grudging respect between us that wasn't there before.",
            f"The afternoon we spent by the lake skipping stones remains vivid in my mind. The cool mist on my face and the soft laughter we shared created a rare moment of peace between us, momentarily setting aside our complicated history."
        ]
    }
    
    # Incorporate current_setting into the system instructions
    system_instructions = f"""
# Memory Generation for {npc_name}

## Relationship Context
{npc_name} has a relationship with {target_name} that may encompass multiple roles reflecting {npc_name}'s complete character. 
These memories should authentically represent all aspects of {npc_name}'s identity—whether they involve familial bonds, professional connections, or unique dynamics based on their defining traits.

## Setting Information
- **Current World:** {mega_description}
- **Current Setting:** {current_setting}
- **Key Locations:**
{locations_table_formatted}
- **Additional Context:**
{extra_context}

## Memory Generation Guidelines

### Core Requirements:
1. Generate THREE distinct first-person memories from {npc_name}'s perspective about interactions with {target_name}.
   - If no relationships exist, create three defining life moment memories instead.
2. Each memory must be 2-3 sentences written in {npc_name}'s authentic voice.
3. Set each memory in a specific location from the provided list or another contextually appropriate location in {current_setting}.
4. Include at least one vivid sensory detail (sight, sound, smell, taste, or touch) per memory.
5. Show clear emotional responses from both {npc_name} and {target_name}.
6. Include different types of interactions:
   - One positive experience (e.g., shared laughter, mutual accomplishment)
   - One challenging interaction (e.g., conflict, disappointment)
   - One memory that adds additional dimension to their relationship

### Memory Diversity:
- Include at least one positive experience (e.g., shared laughter, mutual accomplishment, comfort)
- Include at least one challenging interaction (e.g., conflict, disappointment, consequence)
- Ensure the third memory provides additional dimension to the relationship

### Character Integration:
- Weave ALL of {npc_name}'s defining traits and archetypes into the memories organically
- If {npc_name} has multiple roles or characteristics (e.g., giantess, pirate, online friend), ensure memories reflect this complex identity rather than focusing on a single trait
- Subtly incorporate and foreshadow evolving femdom dynamics in the relationship
- Maintain timeline consistency with established relationship history

## REQUIRED OUTPUT FORMAT
Your response MUST be valid JSON with exactly this structure:
```json
{json.dumps(example_output, indent=2)}
```

⚠️ IMPORTANT: 
- DO NOT include code fences (```) in your response
- DO NOT include any explanations or extra text
- DO NOT modify the key name "memory"
- Return ONLY the JSON object
- You MUST return exactly 3 memories
"""
    messages = [{"role": "system", "content": system_instructions}]
    logger.info("Calling GPT for shared memory generation...")

    # Implement retry logic with backoff (async sleep)
    max_retries = 2
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Memory generation attempt {attempt}/{max_retries}")
            
            # Get OpenAI client
            openai_client = get_openai_client()
            
            # Call the OpenAI API asynchronously using the new Responses API
            response = await openai_client.chat.responses.create(
                model="gpt-4.1-nano", 
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            memory_output = response.choices[0].message.content.strip()
            logger.info(f"GPT response received (length: {len(memory_output)})")

            # Validate JSON structure and memory count
            if memory_output.startswith("{") and memory_output.endswith("}"):
                try:
                    memory_data = json.loads(memory_output)
                    if "memory" in memory_data and isinstance(memory_data["memory"], list):
                        memories = memory_data["memory"]
                        
                        # Ensure we have exactly 3 memories as required
                        if len(memories) < 3:
                            logger.warning(f"Only received {len(memories)} memories, expecting 3")
                            raise ValueError(f"Invalid number of memories: {len(memories)}, need exactly 3")
                        
                        # If we have more than 3, trim to exactly 3
                        if len(memories) > 3:
                            logger.warning(f"Received {len(memories)} memories, trimming to 3")
                            memory_data["memory"] = memories[:3]
                            memory_output = json.dumps(memory_data)
                            
                        # Valid JSON with exactly 3 memories, return it
                        return memory_output
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in response: {e}")
            else:
                logger.warning(f"Response doesn't look like JSON: {memory_output[:50]}...")

        except Exception as e:
            last_exception = e
            logger.error(f"Error during GPT call in get_shared_memory (attempt {attempt}): {e}", exc_info=True)

        # Wait before retrying
        if attempt < max_retries:
            wait_time = 2 ** attempt  # Exponential backoff (2, 4 seconds)
            logger.info(f"Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)

    # If all retries failed
    logger.error(f"Failed to generate memory after {max_retries} attempts. Last error: {last_exception}")
    return None


def extract_or_create_memory_fallback(text_output, npc_name, target_name):
    """
    Attempts to extract memories from malformed output or creates fallbacks.
    """
    import json, re
    
    # Try to find a JSON-like structure with regex
    json_pattern = r'\{.*"memory"\s*:\s*\[.*\].*\}'
    match = re.search(json_pattern, text_output, re.DOTALL)
    
    if match:
        try:
            # Try to parse the extracted JSON
            extracted_json = match.group(0)
            memory_data = json.loads(extracted_json)
            if "memory" in memory_data and isinstance(memory_data["memory"], list):
                if len(memory_data["memory"]) >= 1:
                    # We got at least one valid memory
                    logging.info(f"Extracted {len(memory_data['memory'])} memories from malformed response")
                    
                    # Ensure we have exactly 3 memories
                    memories = memory_data["memory"]
                    while len(memories) < 3:
                        # Add fallback memories if needed
                        index = len(memories)
                        if index == 0:
                            memories.append(f"I remember meeting {target_name} for the first time. There was something about their presence that left a lasting impression on me.")
                        elif index == 1:
                            memories.append(f"Once {target_name} and I had a disagreement that tested our relationship. Despite the tension, we found a way to resolve our differences.")
                        else:
                            memories.append(f"I cherish the quiet moments {target_name} and I have shared. Those simple times together have strengthened our bond in ways words cannot express.")
                    
                    return json.dumps({"memory": memories})
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse extracted JSON: {e}")
    
    # If extraction failed, create complete fallback
    return create_memory_fallback(npc_name, target_name)

def create_memory_fallback(npc_name, target_name):
    """
    Creates a basic set of fallback memories when all else fails.
    """
    logging.warning(f"Using fallback memory generation for {npc_name} and {target_name}")
    
    memories = [
        f"I still remember when I first met {target_name}. There was an immediate sense of connection between us that I hadn't expected, and it made a lasting impression on me.",
        f"The time {target_name} and I had that heated disagreement taught me something important. Though our voices were raised and emotions ran high, I came to respect their conviction and perspective.",
        f"One quiet evening, {target_name} and I shared a moment of unexpected understanding. Sometimes the most meaningful connections happen in the simplest moments, away from the noise of daily life."
    ]
    
    return json.dumps({"memory": memories})


import logging
import json
import random
from datetime import datetime, timedelta

class MemoryType:
    INTERACTION = "interaction"  # Direct interaction between player and NPC
    OBSERVATION = "observation"  # NPC observes player doing something
    EMOTIONAL = "emotional"      # Emotionally significant event
    TRAUMATIC = "traumatic"      # Highly negative event
    INTIMATE = "intimate"        # Deeply personal or sexual event
    
class MemorySignificance:
    LOW = 1      # Routine interaction
    MEDIUM = 3   # Notable but not remarkable
    HIGH = 5     # Important, remembered clearly
    CRITICAL = 10 # Life-changing, unforgettable

class EnhancedMemory:
    """
    Enhanced memory class that tracks emotional impact, decay rates,
    and determines when memories should be recalled.
    """
    def __init__(self, text, memory_type=MemoryType.INTERACTION, significance=MemorySignificance.MEDIUM):
        self.text = text
        self.timestamp = datetime.now().isoformat()
        self.memory_type = memory_type
        self.significance = significance
        self.recall_count = 0
        self.last_recalled = None
        self.emotional_valence = 0  # -10 to +10, negative to positive emotion
        self.tags = []
        
    def to_dict(self):
        return {
            "text": self.text,
            "timestamp": self.timestamp,
            "memory_type": self.memory_type,
            "significance": self.significance,
            "recall_count": self.recall_count,
            "last_recalled": self.last_recalled,
            "emotional_valence": self.emotional_valence,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data):
        memory = cls(data["text"], data["memory_type"], data["significance"])
        memory.timestamp = data["timestamp"]
        memory.recall_count = data["recall_count"]
        memory.last_recalled = data["last_recalled"]
        memory.emotional_valence = data["emotional_valence"]
        memory.tags = data["tags"]
        return memory

class MemoryManager:
    """
    Manages creation, storage, retrieval, and sharing of memories between NPCs.
    Implements advanced features like emotional weighting, memory decay, and
    contextual recall.
    """
    
@staticmethod
async def add_memory(user_id: int, conversation_id: int, entity_id: Union[int, str], entity_type: str,
                    memory_text: str, memory_type: str = MemoryType.INTERACTION,
                    significance: int = MemorySignificance.MEDIUM,
                    emotional_valence: int = 0, tags: Optional[list] = None) -> bool: # Changed to async def
    """Add a new memory to an entity (NPC or player) using asyncpg."""
    tags = tags or []
    memory = EnhancedMemory(memory_text, memory_type, significance)
    memory.emotional_valence = emotional_valence
    memory.tags = tags
    memory_dict = memory.to_dict()

    try:
        async with get_db_connection_context() as conn:
            # Use transaction for read-then-write safety
            async with conn.transaction():
                # 1. Get current memories
                current_memories_json = None
                if entity_type == "npc":
                    current_memories_json = await conn.fetchval("""
                        SELECT memory FROM NPCStats
                        WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3 FOR UPDATE
                    """, user_id, conversation_id, entity_id) # Use FOR UPDATE within transaction
                elif entity_type == "player": # Assume player identified by name 'entity_id'
                     current_memories_json = await conn.fetchval("""
                        SELECT memories FROM PlayerStats
                        WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3 FOR UPDATE
                    """, user_id, conversation_id, entity_id)
                else:
                    logger.error(f"Invalid entity_type '{entity_type}' in add_memory")
                    return False

                # 2. Append new memory
                memories = []
                if current_memories_json:
                    # asyncpg might return list/dict directly for jsonb, or string
                    if isinstance(current_memories_json, list):
                         memories = current_memories_json
                    elif isinstance(current_memories_json, str):
                         try:
                            memories = json.loads(current_memories_json)
                            if not isinstance(memories, list): memories = [] # Ensure it's a list
                         except json.JSONDecodeError:
                            logger.warning(f"Could not decode existing memories for {entity_type} {entity_id}, starting fresh.")
                            memories = []
                    else:
                         logger.warning(f"Unexpected type for existing memories: {type(current_memories_json)}, starting fresh.")
                         memories = []

                memories.append(memory_dict)
                updated_memories_json = json.dumps(memories) # Serialize for DB update

                # 3. Update the database
                if entity_type == "npc":
                    await conn.execute("""
                        UPDATE NPCStats SET memory = $1
                        WHERE user_id=$2 AND conversation_id=$3 AND npc_id=$4
                    """, updated_memories_json, user_id, conversation_id, entity_id)
                else: # player
                    await conn.execute("""
                        UPDATE PlayerStats SET memories = $1
                        WHERE user_id=$2 AND conversation_id=$3 AND player_name=$4
                    """, updated_memories_json, user_id, conversation_id, entity_id)

        # 4. Propagation (outside the transaction for adding the memory itself)
        # Check propagation condition AFTER successfully adding memory
        if memory_type in [MemoryType.EMOTIONAL, MemoryType.TRAUMATIC] and significance >= MemorySignificance.HIGH:
            # Call the async version of propagate
            await MemoryManager.propagate_significant_memory(user_id, conversation_id, entity_id, entity_type, memory)

        return True

    except asyncpg.PostgresError as e:
        logger.error(f"Database error adding memory for {entity_type} {entity_id}: {e}", exc_info=True)
        return False
    except ConnectionError as e:
        logger.error(f"Pool error adding memory: {e}", exc_info=True)
        return False
    except asyncio.TimeoutError:
        logger.error(f"Timeout getting DB connection for adding memory", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error adding memory: {e}", exc_info=True)
        return False

    
@staticmethod
async def propagate_significant_memory(user_id: int, conversation_id: int, source_entity_id: Union[int, str], source_entity_type: str, memory: EnhancedMemory): # Changed to async def
    """Propagate significant memories to related NPCs using asyncpg."""
    links = []
    try:
        async with get_db_connection_context() as conn:
            # Find strong social links (adapt table/column names if needed)
            links = await conn.fetch("""
                SELECT entity1_type, entity1_id, entity2_type, entity2_id, link_type, link_level
                FROM SocialLinks
                WHERE user_id=$1 AND conversation_id=$2 AND link_level >= 50
                AND ((entity1_type=$3 AND entity1_id::text=$4::text) OR (entity2_type=$3 AND entity2_id::text=$4::text))
            """, user_id, conversation_id,
                 source_entity_type, str(source_entity_id)) # Ensure IDs are compared correctly (casting might be needed if types differ)

        if not links:
             # logger.debug(f"No strong links found for {source_entity_type} {source_entity_id} to propagate memory.")
             return True # Not an error if no links exist

        propagation_tasks = []
        for link in links:
            e1_type, e1_id, e2_type, e2_id, link_type, link_level = link['entity1_type'], str(link['entity1_id']), link['entity2_type'], str(link['entity2_id']), link['link_type'], link['link_level']

            target_type, target_id = (e2_type, e2_id) if e1_type == source_entity_type and e1_id == str(source_entity_id) else (e1_type, e1_id)

            if target_type != "npc": # Don't propagate back to player this way
                continue
            if target_id == str(source_entity_id) and target_type == source_entity_type: # Skip self
                 continue

            # Create modified memory for target's perspective
            target_memory = EnhancedMemory(
                f"I heard that {memory.text}",
                memory_type=MemoryType.OBSERVATION,
                significance=max(MemorySignificance.LOW, memory.significance - 2),
                emotional_valence = int(memory.emotional_valence * 0.7), # Keep as int
                tags = memory.tags + ["secondhand", f"link:{link_type}"]
            )

            # Create a task to add the memory to the target NPC
            # Avoid awaiting each add_memory sequentially inside the loop
            propagation_tasks.append(
                MemoryManager.add_memory(
                    user_id, conversation_id,
                    int(target_id) if target_id.isdigit() else target_id, # Convert back to int if needed
                    target_type,
                    target_memory.text,
                    target_memory.memory_type,
                    target_memory.significance,
                    target_memory.emotional_valence,
                    target_memory.tags
                )
            )

        # Run all propagation additions concurrently
        if propagation_tasks:
            results = await asyncio.gather(*propagation_tasks, return_exceptions=True)
            # Log any errors from propagation attempts
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                     logger.error(f"Error propagating memory during gather task {i}: {result}", exc_info=result)
            logger.info(f"Attempted to propagate memory to {len(propagation_tasks)} linked NPCs.")

        return True # Indicate propagation attempt finished

    except asyncpg.PostgresError as e:
        logger.error(f"Database error propagating memory from {source_entity_type} {source_entity_id}: {e}", exc_info=True)
        return False
    except ConnectionError as e:
        logger.error(f"Pool error propagating memory: {e}", exc_info=True)
        return False
    except asyncio.TimeoutError:
        logger.error(f"Timeout getting DB connection for propagating memory", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error propagating memory: {e}", exc_info=True)
        return False
    
@staticmethod
async def retrieve_relevant_memories(user_id: int, conversation_id: int, entity_id: Union[int, str], entity_type: str,
                                     context: Optional[str] = None, tags: Optional[list] = None, limit: int = 5) -> list[EnhancedMemory]: # Changed to async def
    """Retrieve and score relevant memories using asyncpg, updating recall counts."""
    try:
        async with get_db_connection_context() as conn:
             # Use transaction to ensure read and update consistency for recall counts
             async with conn.transaction():
                # 1. Get all memories
                current_memories_json = None
                if entity_type == "npc":
                    current_memories_json = await conn.fetchval("""
                        SELECT memory FROM NPCStats
                        WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3 FOR UPDATE
                    """, user_id, conversation_id, entity_id)
                elif entity_type == "player":
                    current_memories_json = await conn.fetchval("""
                        SELECT memories FROM PlayerStats
                        WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3 FOR UPDATE
                    """, user_id, conversation_id, entity_id)
                else:
                     logger.error(f"Invalid entity_type '{entity_type}' in retrieve_relevant_memories")
                     return []

                if not current_memories_json:
                    return []

                memories_data = []
                if isinstance(current_memories_json, list):
                    memories_data = current_memories_json
                elif isinstance(current_memories_json, str):
                     try:
                         memories_data = json.loads(current_memories_json)
                         if not isinstance(memories_data, list): memories_data = []
                     except json.JSONDecodeError:
                          logger.warning(f"Could not decode memories for retrieve_relevant_memories {entity_type} {entity_id}")
                          memories_data = []
                else:
                     logger.warning(f"Unexpected type for memories data: {type(current_memories_json)}")
                     memories_data = []

                if not memories_data:
                     return []

                # 2. Convert to EnhancedMemory objects and score (Sync logic)
                memory_objects = [EnhancedMemory.from_dict(m) for m in memories_data]
                # (Filter by tags - sync)
                if tags:
                    memory_objects = [m for m in memory_objects if any(tag in m.tags for tag in tags)]
                # (Score memories - sync)
                scored_memories = []
                for memory in memory_objects:
                    # Scoring logic remains the same (sync)
                    score = memory.significance
                    try: # Recency bonus
                        memory_date = datetime.fromisoformat(memory.timestamp)
                        days_old = (datetime.now(memory_date.tzinfo) - memory_date).days # Timezone aware if possible
                        recency_score = max(0, 10 - days_old / 30)
                        score += recency_score
                    except (ValueError, TypeError): pass
                    if context: # Context bonus
                        context_words = set(context.lower().split())
                        memory_words = set(memory.text.lower().split())
                        common_words = context_words.intersection(memory_words)
                        context_score = len(common_words) * 0.5
                        score += context_score
                    score += abs(memory.emotional_valence) * 0.3 # Emotion bonus
                    score -= min(memory.recall_count * 0.2, 2) # Recall penalty
                    scored_memories.append((memory, score))
                # (Sort and limit - sync)
                scored_memories.sort(key=lambda x: x[1], reverse=True)
                top_memory_tuples = scored_memories[:limit]

                # 3. Update recall counts for the selected memories
                updated_memory_dict = {m.timestamp: m for m in memory_objects} # Map for quick lookup
                top_memories_list = []
                now_iso = datetime.now().isoformat()
                for memory, score in top_memory_tuples:
                    if memory.timestamp in updated_memory_dict:
                         mem_obj = updated_memory_dict[memory.timestamp]
                         mem_obj.recall_count += 1
                         mem_obj.last_recalled = now_iso
                         top_memories_list.append(mem_obj) # Add the updated object

                # 4. Convert all back to dicts for storage
                updated_memories_for_db = [m.to_dict() for m in updated_memory_dict.values()]
                updated_memories_json = json.dumps(updated_memories_for_db)

                # 5. Save updated memories back to DB
                if entity_type == "npc":
                    await conn.execute("""
                        UPDATE NPCStats SET memory = $1
                        WHERE user_id=$2 AND conversation_id=$3 AND npc_id=$4
                    """, updated_memories_json, user_id, conversation_id, entity_id)
                else: # player
                    await conn.execute("""
                        UPDATE PlayerStats SET memories = $1
                        WHERE user_id=$2 AND conversation_id=$3 AND player_name=$4
                    """, updated_memories_json, user_id, conversation_id, entity_id)

             # Transaction commits here automatically on successful exit
             return top_memories_list # Return the list of EnhancedMemory objects

    except asyncpg.PostgresError as e:
        logger.error(f"Database error retrieving memories for {entity_type} {entity_id}: {e}", exc_info=True)
        return []
    except ConnectionError as e:
        logger.error(f"Pool error retrieving memories: {e}", exc_info=True)
        return []
    except asyncio.TimeoutError:
        logger.error(f"Timeout getting DB connection for retrieving memories", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Unexpected error retrieving memories: {e}", exc_info=True)
        return []
            
@staticmethod
async def generate_flashback(user_id: int, conversation_id: int, npc_id: int, current_context: str) -> Optional[dict]: # Changed to async def
    """Generate a flashback moment for an NPC based on relevant memories."""
    try:
        # Retrieve memories using the async method
        memories = await MemoryManager.retrieve_relevant_memories(
            user_id, conversation_id, npc_id, "npc",
            context=current_context, limit=3
        )

        if not memories:
            return None

        # Select memory (sync logic)
        emotional_memories = [m for m in memories if m.memory_type in [MemoryType.EMOTIONAL, MemoryType.TRAUMATIC]]
        selected_memory = random.choice(emotional_memories) if emotional_memories else random.choice(memories)

        # Get NPC name (async DB call)
        npc_name = "the NPC" # Default
        async with get_db_connection_context() as conn:
             name_record = await conn.fetchval("""
                SELECT npc_name FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
             """, user_id, conversation_id, npc_id)
             if name_record:
                  npc_name = name_record

        # Format flashback (sync logic)
        flashback_text = f"{npc_name}'s expression shifts momentarily, a distant look crossing their face. \"This reminds me of... {selected_memory.text}\""
        if selected_memory.emotional_valence < -5:
            flashback_text += " A shadow crosses their face at the memory."
        elif selected_memory.emotional_valence > 5:
            flashback_text += " Their eyes seem to light up for a moment."

        return {
            "type": "flashback",
            "npc_id": npc_id,
            "npc_name": npc_name,
            "text": flashback_text,
            "memory": selected_memory.text # Include the raw memory text
        }

    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logger.error(f"Database error during flashback generation for NPC {npc_id}: {db_err}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error during flashback generation for NPC {npc_id}: {e}", exc_info=True)
        return None

class RevealType:
    """Types of revelations for progressive character development"""
    VERBAL_SLIP = "verbal_slip"  # Verbal slip revealing true nature
    BEHAVIOR = "behavior"        # Behavioral inconsistency
    EMOTIONAL = "emotional"      # Emotional reaction revealing depth
    PHYSICAL = "physical"        # Physical tell or appearance change
    KNOWLEDGE = "knowledge"      # Knowledge they shouldn't have
    OVERHEARD = "overheard"      # Overheard saying something revealing
    OBJECT = "object"            # Possession of revealing object
    OPINION = "opinion"          # Expressed opinion revealing true nature
    SKILL = "skill"              # Demonstration of unexpected skill
    HISTORY = "history"          # Revealed history inconsistent with persona

class RevealSeverity:
    """How significant/obvious the revelation is"""
    SUBTLE = 1     # Barely noticeable, could be dismissed
    MINOR = 2      # Noticeable but easily explained away
    MODERATE = 3   # Clearly inconsistent with presented persona
    MAJOR = 4      # Significant revelation of true nature
    COMPLETE = 5   # Mask completely drops, true nature fully revealed

class NPCMask:
    """Represents the facade an NPC presents versus their true nature"""
    def __init__(self, presented_traits=None, hidden_traits=None, reveal_history=None):
        self.presented_traits = presented_traits or {}
        self.hidden_traits = hidden_traits or {}
        self.reveal_history = reveal_history or []
        self.integrity = 100  # How intact the mask is (100 = perfect mask, 0 = completely revealed)
        
    def to_dict(self):
        return {
            "presented_traits": self.presented_traits,
            "hidden_traits": self.hidden_traits,
            "reveal_history": self.reveal_history,
            "integrity": self.integrity
        }
    
    @classmethod
    def from_dict(cls, data):
        mask = cls(data.get("presented_traits"), data.get("hidden_traits"), data.get("reveal_history"))
        mask.integrity = data.get("integrity", 100)
        return mask

class ProgressiveRevealManager:
    """
    Manages the progressive revelation of NPC true natures,
    tracking facade integrity and creating revelation events.
    """
    
    # Opposing trait pairs for mask/true nature contrast
    OPPOSING_TRAITS = {
        "kind": "cruel",
        "gentle": "harsh",
        "caring": "callous",
        "patient": "impatient",
        "humble": "arrogant",
        "honest": "deceptive",
        "selfless": "selfish",
        "supportive": "manipulative",
        "trusting": "suspicious",
        "relaxed": "controlling",
        "open": "secretive",
        "egalitarian": "domineering",
        "casual": "formal",
        "empathetic": "cold",
        "nurturing": "exploitative"
    }
    
    # Physical tells for different hidden traits
    PHYSICAL_TELLS = {
        "cruel": ["momentary smile at others' discomfort", "subtle gleam in eyes when causing pain", "fingers flexing as if eager to inflict harm"],
        "harsh": ["brief scowl before composing face", "jaw tightening when frustrated", "eyes hardening momentarily"],
        "callous": ["dismissive flick of the wrist", "eyes briefly glazing over during others' emotional moments", "impatient foot tapping"],
        "arrogant": ["subtle sneer quickly hidden", "looking down nose briefly", "momentary eye-roll"],
        "deceptive": ["micro-expression of calculation", "eyes darting briefly", "subtle change in vocal pitch"],
        "manipulative": ["predatory gaze quickly masked", "fingers steepling then separating", "momentary satisfied smirk"],
        "controlling": ["unconscious straightening of surroundings", "stiffening posture when not obeyed", "fingers drumming impatiently"],
        "domineering": ["stance widening to take up space", "chin raising imperiously before catching themselves", "hand gesture that suggests expectation of obedience"]
    }
    
    # Verbal slips for different hidden traits
    VERBAL_SLIPS = {
        "cruel": ["That will teach th-- I mean, I hope they learn from this experience", "The pain should be excru-- educational for them", "I enjoy seeing-- I mean, I hope they recover quickly"],
        "manipulative": ["Once they're under my-- I mean, once they understand my point", "They're so easy to contr-- convince", "Just as I've planned-- I mean, just as I'd hoped"],
        "domineering": ["They will obey-- I mean, they will understand", "I expect complete submis-- cooperation", "My orders are-- I mean, my suggestions are"],
        "deceptive": ["The lie is perfect-- I mean, the explanation is clear", "They never suspect that-- they seem to understand completely", "I've fabricated-- formulated a perfect response"]
    }
    
@staticmethod
async def initialize_npc_mask(user_id: int, conversation_id: int, npc_id: int, overwrite: bool = False) -> dict: # Changed to async def
    """Create initial mask for NPC using asyncpg."""
    try:
        async with get_db_connection_context() as conn:
            async with conn.transaction(): # Use transaction for consistency checks and writes
                # 1. Check if mask exists (if not overwriting)
                if not overwrite:
                    existing_mask = await conn.fetchval("""
                        SELECT 1 FROM NPCMasks
                        WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                    """, user_id, conversation_id, npc_id)
                    if existing_mask:
                        return {"message": "Mask already exists", "already_exists": True, "npc_id": npc_id}

                # 2. Get NPC data (lock row if overwriting?)
                # Add FOR UPDATE if overwrite is true and you want to prevent concurrent init?
                npc_row = await conn.fetchrow("""
                    SELECT npc_name, dominance, cruelty, personality_traits, archetype_summary
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                """, user_id, conversation_id, npc_id)

                if not npc_row:
                    return {"error": f"NPC with id {npc_id} not found", "npc_id": npc_id}

                npc_name, dominance, cruelty, personality_traits_json, archetype_summary = npc_row['npc_name'], npc_row['dominance'], npc_row['cruelty'], npc_row['personality_traits'], npc_row['archetype_summary']

            
                # 3. Generate traits (Sync logic)
                # Parse personality traits
                personality_traits = []
                if personality_traits_json:
                     # asyncpg might auto-decode JSONB, check type
                     if isinstance(personality_traits_json, list):
                          personality_traits = personality_traits_json
                     elif isinstance(personality_traits_json, str):
                         try: personality_traits = json.loads(personality_traits_json)
                         except: pass
                     if not isinstance(personality_traits, list): personality_traits = [] # Ensure list

                # Trait generation logic remains the same (sync)
                presented_traits, hidden_traits = {}, {}
                mask_depth = (dominance + cruelty) / 2
                num_masked_traits = int(mask_depth / 20) + 1 # 1-6 traits
                trait_candidates = {}
                reversed_opposites = {v: k for k, v in ProgressiveRevealManager.OPPOSING_TRAITS.items()}
                
                for trait in personality_traits:
                    trait_lower = trait.lower()
                    
                    # Find traits that have opposites in our OPPOSING_TRAITS dictionary
                    reversed_dict = {v: k for k, v in ProgressiveRevealManager.OPPOSING_TRAITS.items()}
                    
                    if trait_lower in ProgressiveRevealManager.OPPOSING_TRAITS:
                        # This is a "good" trait that could mask a "bad" one
                        opposite = ProgressiveRevealManager.OPPOSING_TRAITS[trait_lower]
                        trait_candidates[trait] = opposite
                    elif trait_lower in reversed_dict:
                        # This is already a "bad" trait, so it's part of the hidden nature
                        hidden_traits[trait] = {"intensity": random.randint(60, 90)}
                        
                        # Generate a presented trait to mask it
                        presented_traits[reversed_dict[trait_lower]] = {"confidence": random.randint(60, 90)}
                
                # If we don't have enough contrasting traits, add some
                if len(trait_candidates) < num_masked_traits:
                    additional_needed = num_masked_traits - len(trait_candidates)
                    
                    # Choose random traits from OPPOSING_TRAITS
                    available_traits = list(ProgressiveRevealManager.OPPOSING_TRAITS.keys())
                    random.shuffle(available_traits)
                    
                    for i in range(min(additional_needed, len(available_traits))):
                        trait = available_traits[i]
                        opposite = ProgressiveRevealManager.OPPOSING_TRAITS[trait]
                        
                        if trait not in trait_candidates and trait not in presented_traits:
                            trait_candidates[trait] = opposite
                
                # Select traits to mask
                masked_traits = dict(list(trait_candidates.items())[:num_masked_traits])
                
                # Add to presented and hidden traits
                for presented, hidden in masked_traits.items():
                    if presented not in presented_traits:
                        presented_traits[presented] = {"confidence": random.randint(60, 90)}
                    
                    if hidden not in hidden_traits:
                        hidden_traits[hidden] = {"intensity": random.randint(60, 90)}
                
                # 4. Create mask object (Sync)
                mask = NPCMask(presented_traits, hidden_traits, [])
                mask_json = json.dumps(mask.to_dict()) # Serialize for DB

                # 5. Store in database
                await conn.execute("""
                    INSERT INTO NPCMasks (user_id, conversation_id, npc_id, mask_data)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (user_id, conversation_id, npc_id)
                    DO UPDATE SET mask_data = EXCLUDED.mask_data
                """, user_id, conversation_id, npc_id, mask_json)

            # Transaction commits automatically
            return {
                "npc_id": npc_id,
                "npc_name": npc_name,
                "mask_created": True,
                "presented_traits": presented_traits,
                "hidden_traits": hidden_traits,
                "message": "NPC mask initialized/updated successfully"
            }

    except asyncpg.PostgresError as e:
        logger.error(f"Database error initializing NPC mask for {npc_id}: {e}", exc_info=True)
        return {"error": f"Database error initializing mask for NPC {npc_id}", "npc_id": npc_id}
    # Handle ConnectionError, TimeoutError etc. as in other functions
    except Exception as e:
        logger.error(f"Unexpected error initializing NPC mask for {npc_id}: {e}", exc_info=True)
        return {"error": f"Unexpected error initializing mask for NPC {npc_id}", "npc_id": npc_id}

@staticmethod
async def get_npc_mask(user_id: int, conversation_id: int, npc_id: int) -> dict: # Changed to async def
    """Retrieve an NPC's mask data using asyncpg, initializing if needed."""
    try:
        async with get_db_connection_context() as conn:
             # Fetch mask and NPC data in parallel? Or sequentially is fine.
             mask_data_json = await conn.fetchval("""
                  SELECT mask_data FROM NPCMasks
                  WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
             """, user_id, conversation_id, npc_id)

             if not mask_data_json:
                  # Mask doesn't exist, try to initialize it
                  logger.info(f"No mask found for NPC {npc_id}, attempting initialization...")
                  init_result = await ProgressiveRevealManager.initialize_npc_mask(user_id, conversation_id, npc_id)
                  if "error" in init_result:
                       return init_result # Return the error from initialization
                  if not init_result.get("mask_created"):
                        return {"error": f"Failed to auto-initialize mask for NPC {npc_id}", "npc_id": npc_id}

                  # Fetch the newly created mask data
                  mask_data_json = await conn.fetchval("""
                       SELECT mask_data FROM NPCMasks
                       WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                  """, user_id, conversation_id, npc_id)
                  if not mask_data_json:
                       return {"error": f"Failed to retrieve mask even after initialization for NPC {npc_id}", "npc_id": npc_id}

             # Decode JSON
             mask_data = {}
             if isinstance(mask_data_json, dict): # asyncpg might auto-decode
                  mask_data = mask_data_json
             elif isinstance(mask_data_json, str):
                  try: mask_data = json.loads(mask_data_json)
                  except: pass
             if not isinstance(mask_data, dict): mask_data = {} # Ensure dict

             # Fetch NPC basic info
             npc_row = await conn.fetchrow("""
                  SELECT npc_name, dominance, cruelty FROM NPCStats
                  WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
             """, user_id, conversation_id, npc_id)

             if not npc_row:
                  return {"error": f"NPC {npc_id} not found, cannot provide mask context", "npc_id": npc_id}

             npc_name, dominance, cruelty = npc_row['npc_name'], npc_row['dominance'], npc_row['cruelty']

             # Create mask object (Sync)
             mask = NPCMask.from_dict(mask_data)

             return {
                "npc_id": npc_id,
                "npc_name": npc_name,
                "dominance": dominance,
                "cruelty": cruelty,
                "presented_traits": mask.presented_traits,
                "hidden_traits": mask.hidden_traits,
                "integrity": mask.integrity,
                "reveal_history": mask.reveal_history
             }

    except asyncpg.PostgresError as e:
        logger.error(f"Database error getting NPC mask for {npc_id}: {e}", exc_info=True)
        return {"error": f"Database error getting mask for NPC {npc_id}", "npc_id": npc_id}
    # Handle ConnectionError, TimeoutError etc.
    except Exception as e:
        logger.error(f"Unexpected error getting NPC mask for {npc_id}: {e}", exc_info=True)
        return {"error": f"Unexpected error getting mask for NPC {npc_id}", "npc_id": npc_id}

    
@staticmethod
async def generate_mask_slippage(user_id: int, conversation_id: int, npc_id: int,
                                 trigger: Optional[str] = None,
                                 severity: Optional[int] = None,
                                 reveal_type: Optional[str] = None) -> dict: # Changed to async def
    """Generate and record a mask slippage event using asyncpg."""
    try:
        async with get_db_connection_context() as conn:
            # Use transaction for read-update-insert consistency
            async with conn.transaction():
                # 1. Get current mask data (lock the row)
                mask_data_json = await conn.fetchval("""
                    SELECT mask_data FROM NPCMasks
                    WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3 FOR UPDATE
                """, user_id, conversation_id, npc_id)

                if not mask_data_json:
                    # Optionally try to initialize mask here? Or return error.
                    return {"error": f"Cannot generate slippage, mask not found for NPC {npc_id}", "npc_id": npc_id}

                # Get NPC name for context (could be fetched outside transaction if preferred)
                npc_name = await conn.fetchval("""
                     SELECT npc_name FROM NPCStats WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                """, user_id, conversation_id, npc_id) or f"NPC {npc_id}"
                # Decode JSON and create mask object (Sync)
                mask_data = {}
                if isinstance(mask_data_json, dict): mask_data = mask_data_json
                elif isinstance(mask_data_json, str):
                     try: mask_data = json.loads(mask_data_json)
                     except: pass
                if not isinstance(mask_data, dict): mask_data = {}
                mask = NPCMask.from_dict(mask_data)

            
            # Choose a severity level if not provided
            if severity is None:
                # Higher chance of subtle reveals early, more major reveals as integrity decreases
                if integrity > 80:
                    severity_weights = [0.7, 0.2, 0.1, 0, 0]  # Mostly subtle
                elif integrity > 60:
                    severity_weights = [0.4, 0.4, 0.2, 0, 0]  # Subtle to minor
                elif integrity > 40:
                    severity_weights = [0.2, 0.3, 0.4, 0.1, 0]  # Minor to moderate
                elif integrity > 20:
                    severity_weights = [0.1, 0.2, 0.3, 0.4, 0]  # Moderate to major
                else:
                    severity_weights = [0, 0.1, 0.2, 0.4, 0.3]  # Major to complete
                
                severity_levels = [
                    RevealSeverity.SUBTLE,
                    RevealSeverity.MINOR,
                    RevealSeverity.MODERATE,
                    RevealSeverity.MAJOR,
                    RevealSeverity.COMPLETE
                ]
                
                severity = random.choices(severity_levels, weights=severity_weights, k=1)[0]
            
            # Choose a reveal type if not provided
            if reveal_type is None:
                reveal_types = [
                    RevealType.VERBAL_SLIP,
                    RevealType.BEHAVIOR,
                    RevealType.EMOTIONAL,
                    RevealType.PHYSICAL,
                    RevealType.KNOWLEDGE,
                    RevealType.OPINION
                ]
                
                # Check if we've used any types recently to avoid repetition
                recent_types = [event["type"] for event in reveal_history[-3:]]
                available_types = [t for t in reveal_types if t not in recent_types]
                
                if not available_types:
                    available_types = reveal_types
                
                reveal_type = random.choice(available_types)
            
            # Choose a hidden trait to reveal
            if hidden_traits:
                trait, trait_info = random.choice(list(hidden_traits.items()))
            else:
                # Fallback if no hidden traits defined
                trait = random.choice(["manipulative", "controlling", "domineering"])
                trait_info = {"intensity": random.randint(60, 90)}
            
            # Generate reveal description based on type and trait
            reveal_description = ""
            
            if reveal_type == RevealType.VERBAL_SLIP:
                if trait in ProgressiveRevealManager.VERBAL_SLIPS:
                    slip = random.choice(ProgressiveRevealManager.VERBAL_SLIPS[trait])
                    reveal_description = f"{npc_name} lets slip: \"{slip}\""
                else:
                    reveal_description = f"{npc_name} momentarily speaks in a {trait} tone before catching themselves."
            
            elif reveal_type == RevealType.PHYSICAL:
                if trait in ProgressiveRevealManager.PHYSICAL_TELLS:
                    tell = random.choice(ProgressiveRevealManager.PHYSICAL_TELLS[trait])
                    reveal_description = f"{npc_name} displays a {tell} before resuming their usual demeanor."
                else:
                    reveal_description = f"{npc_name}'s expression briefly shifts to something more {trait} before they compose themselves."
            
            elif reveal_type == RevealType.EMOTIONAL:
                reveal_description = f"{npc_name} has an uncharacteristic emotional reaction, revealing a {trait} side that's usually hidden."
            
            elif reveal_type == RevealType.BEHAVIOR:
                reveal_description = f"{npc_name}'s behavior momentarily shifts, showing a {trait} tendency that contradicts their usual persona."
            
            elif reveal_type == RevealType.KNOWLEDGE:
                reveal_description = f"{npc_name} reveals knowledge they shouldn't have, suggesting a {trait} side to their character."
            
            elif reveal_type == RevealType.OPINION:
                reveal_description = f"{npc_name} expresses an opinion that reveals {trait} tendencies, contrasting with their usual presented self."
            
            # If trigger provided, incorporate it
            if trigger:
                reveal_description += f" This was triggered by {trigger}."
            
            # Calculate integrity damage based on severity
            integrity_damage = {
                RevealSeverity.SUBTLE: random.randint(1, 3),
                RevealSeverity.MINOR: random.randint(3, 7),
                RevealSeverity.MODERATE: random.randint(7, 12),
                RevealSeverity.MAJOR: random.randint(12, 20),
                RevealSeverity.COMPLETE: random.randint(20, 40)
            }[severity]
            
            # Apply damage
            new_integrity = max(0, integrity - integrity_damage)
            
            # Create event record
            event = {
                "date": datetime.now().isoformat(),
                "type": reveal_type,
                "severity": severity,
                "trait_revealed": trait,
                "description": reveal_description,
                "integrity_before": integrity,
                "integrity_after": new_integrity,
                "trigger": trigger
            }
            
            # Update mask
            reveal_history.append(event)
            
            mask = NPCMask(presented_traits, hidden_traits, reveal_history)
            mask.integrity = new_integrity
            
            # Save to database
            cursor.execute("""
                UPDATE NPCMasks
                SET mask_data = %s
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
            """, (json.dumps(mask.to_dict()), user_id, conversation_id, npc_id))
            
            # Add to player journal
            cursor.execute("""
                INSERT INTO PlayerJournal (
                    user_id, conversation_id, entry_type, entry_text, timestamp
                )
                VALUES (%s, %s, 'npc_revelation', %s, CURRENT_TIMESTAMP)
            """, (
                user_id, conversation_id,
                f"Observed {npc_name} reveal: {reveal_description}"
            ))
            
            conn.commit()
            
            # If integrity falls below thresholds, trigger special events
            special_event = None
            
            if new_integrity <= 50 and integrity > 50:
                special_event = {
                    "type": "mask_threshold",
                    "threshold": 50,
                    "message": f"{npc_name}'s mask is beginning to crack significantly. Their true nature is becoming more difficult to hide."
                }
            elif new_integrity <= 20 and integrity > 20:
                special_event = {
                    "type": "mask_threshold",
                    "threshold": 20,
                    "message": f"{npc_name}'s facade is crumbling. Their true nature is now plainly visible to those paying attention."
                }
            elif new_integrity <= 0 and integrity > 0:
                special_event = {
                    "type": "mask_threshold",
                    "threshold": 0,
                    "message": f"{npc_name}'s mask has completely fallen away. They no longer attempt to hide their true nature from you."
                }
            
            return {
                "npc_id": npc_id, "npc_name": npc_name, "reveal_type": reveal_type,
                "severity": severity, "trait_revealed": trait, "description": reveal_description,
                "integrity_before": event['integrity_before'], "integrity_after": new_integrity,
                "special_event": special_event
            }

    except asyncpg.PostgresError as e:
        logger.error(f"Database error generating mask slippage for {npc_id}: {e}", exc_info=True)
        return {"error": f"Database error generating slippage for NPC {npc_id}", "npc_id": npc_id}
    # Handle ConnectionError, TimeoutError etc.
    except Exception as e:
        logger.error(f"Unexpected error generating mask slippage for {npc_id}: {e}", exc_info=True)
        return {"error": f"Unexpected error generating slippage for NPC {npc_id}", "npc_id": npc_id}
        
@staticmethod
async def check_for_automated_reveals(user_id: int, conversation_id: int) -> list[dict]: # Changed to async def
    """Check for and trigger automated reveals using asyncpg."""
    reveals = []
    try:
         async with get_db_connection_context() as conn:
              # 1. Get all NPCs with masks + relevant stats
              npc_data = await conn.fetch("""
                  SELECT m.npc_id, m.mask_data, n.npc_name, n.dominance, n.cruelty
                  FROM NPCMasks m
                  JOIN NPCStats n ON m.npc_id = n.npc_id
                      AND m.user_id = n.user_id
                      AND m.conversation_id = n.conversation_id
                  WHERE m.user_id=$1 AND m.conversation_id=$2
              """, user_id, conversation_id)

              if not npc_data:
                   return [] # No NPCs with masks found

              # 2. Get current time (optional, could be passed in)
              time_of_day = await conn.fetchval("""
                  SELECT value FROM CurrentRoleplay
                  WHERE user_id=$1 AND conversation_id=$2 AND key='TimeOfDay'
              """, user_id, conversation_id) or "Afternoon" # Default time

         # 3. Iterate through NPCs and check chance (mostly sync logic, but calls async slippage)
         reveal_chance_map = { "Morning": 0.1, "Afternoon": 0.15, "Evening": 0.2, "Night": 0.25 }
         base_chance = reveal_chance_map.get(time_of_day, 0.15)

         generation_tasks = []
         for record in npc_data:
              npc_id, mask_data_json, npc_name, dominance, cruelty = record['npc_id'], record['mask_data'], record['npc_name'], record['dominance'], record['cruelty']

              mask_data = {}
              if isinstance(mask_data_json, dict): mask_data = mask_data_json
              elif isinstance(mask_data_json, str):
                  try: mask_data = json.loads(mask_data_json)
                  except: pass
              if not isinstance(mask_data, dict): mask_data = {}
              mask = NPCMask.from_dict(mask_data)

              # Calculate chance (sync)
              modifier = (dominance + cruelty) / 200
              integrity_factor = (100 - mask.integrity) / 100
              final_chance = base_chance + (modifier * 0.2) + (integrity_factor * 0.3)

              # Roll for reveal (sync)
              if random.random() < final_chance:
                   logger.info(f"Automated reveal triggered for NPC {npc_id} ({npc_name}) with chance {final_chance:.2f}")
                   # Schedule the slippage generation (which involves DB writes)
                   generation_tasks.append(
                        ProgressiveRevealManager.generate_mask_slippage(
                             user_id, conversation_id, npc_id, trigger="automated check"
                        )
                   )

         # 4. Run all triggered reveals concurrently
         if generation_tasks:
              results = await asyncio.gather(*generation_tasks, return_exceptions=True)
              for result in results:
                   if isinstance(result, dict) and "error" not in result:
                        reveals.append(result)
                   elif isinstance(result, Exception):
                        logger.error(f"Error during automated reveal generation task: {result}", exc_info=result)
                   elif isinstance(result, dict) and "error" in result:
                       logger.error(f"Failed automated reveal generation for NPC {result.get('npc_id', '?')}: {result['error']}")

         return reveals

    except asyncpg.PostgresError as e:
        logger.error(f"Database error checking automated reveals: {e}", exc_info=True)
        return []
    # Handle ConnectionError, TimeoutError etc.
    except Exception as e:
        logger.error(f"Unexpected error checking automated reveals: {e}", exc_info=True)
        return []
    
@staticmethod
async def get_perception_difficulty(user_id: int, conversation_id: int, npc_id: int) -> dict:
    """Calculate perception difficulty against NPC mask using asyncpg."""
    try:
        async with get_db_connection_context() as conn:
             # Fetch mask, NPC stats, and player stats
             mask_data_json = await conn.fetchval("""
                  SELECT mask_data FROM NPCMasks
                  WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
             """, user_id, conversation_id, npc_id)

             if not mask_data_json:
                  return {"error": f"No mask found for NPC {npc_id}", "npc_id": npc_id}

             npc_row = await conn.fetchrow("""
                  SELECT dominance, cruelty FROM NPCStats
                  WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
             """, user_id, conversation_id, npc_id)

             if not npc_row:
                  return {"error": f"NPC {npc_id} not found", "npc_id": npc_id}
                  
             dominance, cruelty = npc_row['dominance'], npc_row['cruelty']

             # Assume player name is 'Chase' - adapt if needed
             player_row = await conn.fetchrow("""
                  SELECT mental_resilience, confidence FROM PlayerStats
                  WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
             """, user_id, conversation_id)
             
             # Default values if player not found
             mental_resilience, confidence = 50, 50
             if player_row:
                 mental_resilience = player_row['mental_resilience']
                 confidence = player_row['confidence']

             # Decode mask (Sync)
             mask_data = {}
             if isinstance(mask_data_json, dict): mask_data = mask_data_json
             elif isinstance(mask_data_json, str):
                  try: mask_data = json.loads(mask_data_json)
                  except: pass
             if not isinstance(mask_data, dict): mask_data = {}
             mask = NPCMask.from_dict(mask_data)
            
             # Calculate base difficulty based on integrity
             base_difficulty = mask.integrity / 2  # 0-50
            
             # Add difficulty based on dominance/cruelty (higher = better at deception)
             stat_factor = (dominance + cruelty) / 4  # 0-50
            
             # Calculate total difficulty
             total_difficulty = base_difficulty + stat_factor
            
             # Calculate player's perception ability
             perception_ability = (mental_resilience + confidence) / 2
            
             # Calculate final difficulty rating
             if perception_ability > 0:
                 relative_difficulty = total_difficulty / perception_ability
             else:
                 relative_difficulty = total_difficulty
            
             difficulty_rating = ""
             if relative_difficulty < 0.5:
                 difficulty_rating = "Very Easy"
             elif relative_difficulty < 0.8:
                 difficulty_rating = "Easy"
             elif relative_difficulty < 1.2:
                 difficulty_rating = "Moderate"
             elif relative_difficulty < 1.5:
                 difficulty_rating = "Difficult"
             else:
                 difficulty_rating = "Very Difficult"
            
             return {
                  "npc_id": npc_id, "mask_integrity": mask.integrity,
                  "difficulty_score": round(total_difficulty, 1),
                  "player_perception": round(perception_ability, 1),
                  "relative_difficulty": round(relative_difficulty, 2),
                  "difficulty_rating": difficulty_rating
             }

    except asyncpg.PostgresError as e:
        logger.error(f"Database error calculating perception difficulty for {npc_id}: {e}", exc_info=True)
        return {"error": f"Database error calculating difficulty for NPC {npc_id}", "npc_id": npc_id}
    # Handle ConnectionError, TimeoutError etc.
    except Exception as e:
        logger.error(f"Unexpected error calculating perception difficulty for {npc_id}: {e}", exc_info=True)
        return {"error": f"Unexpected error calculating difficulty for NPC {npc_id}", "npc_id": npc_id}
