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
    logging.info(f"Starting get_shared_memory for NPC '{npc_name}' with relationship: {relationship}")
    
    # Fetch stored environment details from CurrentRoleplay.
    logging.debug("Fetching stored environment details from CurrentRoleplay...")
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
        logging.info(f"Retrieved environment description (first 100 chars): {mega_description[:100]}...")
    except Exception as e:
        logging.error(f"Error retrieving stored setting: {e}")
        mega_description = "an undefined setting"
    finally:
        conn.close()
    
    # Fetch and format current locations.
    logging.debug("Fetching and formatting current locations...")
    locations_table_formatted = fetch_formatted_locations(user_id, conversation_id)
    logging.info(f"Formatted locations: {locations_table_formatted}")
    
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
    system_instructions = f"""
# Memory Generation for {npc_name}

## Relationship Context
{npc_name} has a relationship with {target_name} that may encompass multiple roles reflecting {npc_name}'s complete character. 
These memories should authentically represent all aspects of {npc_name}'s identity—whether they involve familial bonds, professional connections, or unique dynamics based on their defining traits.

## Setting Information
- **Current World:** {mega_description}
- **Key Locations:**
{locations_table_formatted}
- **Additional Context:**
{extra_context}

## Memory Generation Guidelines

### Core Requirements:
1. Generate THREE distinct first-person memories from {npc_name}'s perspective about interactions with {target_name}.
   - If no relationships exist, create three defining life moment memories instead.
2. Each memory must be 2-3 sentences written in {npc_name}'s authentic voice.
3. Set each memory in a specific location from the provided list or another contextually appropriate location.
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
"""
    logging.debug(f"Constructed system instructions with length: {len(system_instructions)} characters")
    
    messages = [{"role": "system", "content": system_instructions}]
    logging.info("Calling GPT for shared memory generation...")
    
    # Implement retry logic with backoff
    max_retries = 2
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"Memory generation attempt {attempt}/{max_retries}")
            response = get_openai_client().chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            # Extract and process the content
            memory_output = response.choices[0].message.content.strip()
            logging.info(f"GPT response received (length: {len(memory_output)})")
            
            # Basic validation - check if it looks like JSON
            if memory_output.startswith("{") and memory_output.endswith("}"):
                try:
                    # Parse the JSON to validate it
                    memory_data = json.loads(memory_output)
                    # Check if it has the expected structure
                    if "memory" in memory_data and isinstance(memory_data["memory"], list):
                        if len(memory_data["memory"]) >= 3:
                            return memory_output
                        else:
                            logging.warning(f"Memory list too short, only {len(memory_data['memory'])} entries")
                    else:
                        logging.warning("Response missing 'memory' array")
                except json.JSONDecodeError as e:
                    logging.error(f"Invalid JSON in response: {e}")
            else:
                logging.warning(f"Response doesn't look like JSON: {memory_output[:50]}...")
                
            # If we're here, validation failed but we have a response
            if attempt == max_retries:
                # Last attempt, try to salvage what we can
                return extract_or_create_memory_fallback(memory_output, npc_name, target_name)
                
        except Exception as e:
            logging.error(f"Error during GPT call in get_shared_memory (attempt {attempt}): {e}")
            if attempt == max_retries:
                # Generate a fallback on final attempt
                return create_memory_fallback(npc_name, target_name)
    
    # If we somehow get here, provide a basic fallback
    return create_memory_fallback(npc_name, target_name)

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
from db.connection import get_db_connection

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
    async def add_memory(user_id, conversation_id, entity_id, entity_type, 
                        memory_text, memory_type=MemoryType.INTERACTION, 
                        significance=MemorySignificance.MEDIUM, 
                        emotional_valence=0, tags=None):
        """Add a new memory to an entity (NPC or player)"""
        tags = tags or []
        
        # Create the memory object
        memory = EnhancedMemory(memory_text, memory_type, significance)
        memory.emotional_valence = emotional_valence
        memory.tags = tags
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get current memories
            if entity_type == "npc":
                cursor.execute("""
                    SELECT memory FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (user_id, conversation_id, entity_id))
            else:  # player
                cursor.execute("""
                    SELECT memories FROM PlayerStats
                    WHERE user_id=%s AND conversation_id=%s AND player_name=%s
                """, (user_id, conversation_id, entity_id))
                
            row = cursor.fetchone()
            
            memories = []
            if row and row[0]:
                if isinstance(row[0], str):
                    try:
                        memories = json.loads(row[0])
                    except json.JSONDecodeError:
                        memories = []
                else:
                    memories = row[0]
            
            # Add new memory
            memories.append(memory.to_dict())
            
            # Update the database
            if entity_type == "npc":
                cursor.execute("""
                    UPDATE NPCStats
                    SET memory = %s
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (json.dumps(memories), user_id, conversation_id, entity_id))
            else:  # player
                cursor.execute("""
                    UPDATE PlayerStats
                    SET memories = %s
                    WHERE user_id=%s AND conversation_id=%s AND player_name=%s
                """, (json.dumps(memories), user_id, conversation_id, entity_id))
                
            conn.commit()
            
            # Determine if this memory should propagate to other NPCs
            if memory_type in [MemoryType.EMOTIONAL, MemoryType.TRAUMATIC] and significance >= MemorySignificance.HIGH:
                await MemoryManager.propagate_significant_memory(user_id, conversation_id, entity_id, entity_type, memory)
                
            return True
        except Exception as e:
            conn.rollback()
            logging.error(f"Error adding memory: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    @staticmethod
    async def propagate_significant_memory(user_id, conversation_id, source_entity_id, source_entity_type, memory):
        """Propagate significant memories to related NPCs based on social links"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Find strong social links
            cursor.execute("""
                SELECT entity1_type, entity1_id, entity2_type, entity2_id, link_type, link_level
                FROM SocialLinks
                WHERE user_id=%s AND conversation_id=%s AND link_level >= 50
                AND ((entity1_type=%s AND entity1_id=%s) OR (entity2_type=%s AND entity2_id=%s))
            """, (user_id, conversation_id, 
                 source_entity_type, source_entity_id, 
                 source_entity_type, source_entity_id))
                
            links = cursor.fetchall()
            
            for link in links:
                e1_type, e1_id, e2_type, e2_id, link_type, link_level = link
                
                # Determine the target entity
                if e1_type == source_entity_type and e1_id == source_entity_id:
                    target_type = e2_type
                    target_id = e2_id
                else:
                    target_type = e1_type
                    target_id = e1_id
                
                # Skip if target is not an NPC (don't propagate to player)
                if target_type != "npc":
                    continue
                
                # Modify the memory for the target's perspective
                # This creates a "heard about" memory rather than direct experience
                target_memory = EnhancedMemory(
                    f"I heard that {memory.text}",
                    memory_type="observation",
                    significance=max(MemorySignificance.LOW, memory.significance - 2)
                )
                target_memory.emotional_valence = memory.emotional_valence * 0.7  # Reduced emotional impact
                target_memory.tags = memory.tags + ["secondhand"]
                
                # Add the modified memory to the target
                await MemoryManager.add_memory(
                    user_id, conversation_id, 
                    target_id, target_type,
                    target_memory.text,
                    target_memory.memory_type,
                    target_memory.significance,
                    target_memory.emotional_valence,
                    target_memory.tags
                )
                
            return True
        except Exception as e:
            logging.error(f"Error propagating memory: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    @staticmethod
    async def retrieve_relevant_memories(user_id, conversation_id, entity_id, entity_type, 
                                       context=None, tags=None, limit=5):
        """
        Retrieve memories relevant to the given context or tags.
        Applies weighting based on significance, recency, and emotional impact.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get all memories
            if entity_type == "npc":
                cursor.execute("""
                    SELECT memory FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (user_id, conversation_id, entity_id))
            else:  # player
                cursor.execute("""
                    SELECT memories FROM PlayerStats
                    WHERE user_id=%s AND conversation_id=%s AND player_name=%s
                """, (user_id, conversation_id, entity_id))
                
            row = cursor.fetchone()
            
            if not row or not row[0]:
                return []
                
            memories = []
            if isinstance(row[0], str):
                try:
                    memories = json.loads(row[0])
                except json.JSONDecodeError:
                    memories = []
            else:
                memories = row[0]
            
            # Convert to EnhancedMemory objects
            memory_objects = [EnhancedMemory.from_dict(m) for m in memories]
            
            # Filter by tags if provided
            if tags:
                memory_objects = [m for m in memory_objects if any(tag in m.tags for tag in tags)]
            
            # Score memories based on relevance
            scored_memories = []
            for memory in memory_objects:
                score = memory.significance  # Base score is significance
                
                # Recency bonus
                try:
                    memory_date = datetime.fromisoformat(memory.timestamp)
                    days_old = (datetime.now() - memory_date).days
                    recency_score = max(0, 10 - days_old/30)  # Higher score for more recent memories
                    score += recency_score
                except (ValueError, TypeError):
                    pass
                
                # Context relevance if context is provided
                if context:
                    context_words = context.lower().split()
                    memory_words = memory.text.lower().split()
                    common_words = set(context_words) & set(memory_words)
                    context_score = len(common_words) * 0.5
                    score += context_score
                
                # Emotional impact bonus
                emotion_score = abs(memory.emotional_valence) * 0.3
                score += emotion_score
                
                # Penalize frequently recalled memories slightly to ensure variety
                recall_penalty = min(memory.recall_count * 0.2, 2)
                score -= recall_penalty
                
                scored_memories.append((memory, score))
            
            # Sort by score and take top 'limit'
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            top_memories = [m[0] for m in scored_memories[:limit]]
            
            # Update recall count for selected memories
            for memory in top_memories:
                memory.recall_count += 1
                memory.last_recalled = datetime.now().isoformat()
            
            # Update the stored memories
            updated_memories = []
            for memory in memory_objects:
                # Check if this memory is in top_memories
                matching_memory = next((m for m in top_memories if m.timestamp == memory.timestamp), None)
                if matching_memory:
                    updated_memories.append(matching_memory.to_dict())
                else:
                    updated_memories.append(memory.to_dict())
            
            # Save back to database
            if entity_type == "npc":
                cursor.execute("""
                    UPDATE NPCStats
                    SET memory = %s
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (json.dumps(updated_memories), user_id, conversation_id, entity_id))
            else:  # player
                cursor.execute("""
                    UPDATE PlayerStats
                    SET memories = %s
                    WHERE user_id=%s AND conversation_id=%s AND player_name=%s
                """, (json.dumps(updated_memories), user_id, conversation_id, entity_id))
                
            conn.commit()
            
            return top_memories
        except Exception as e:
            conn.rollback()
            logging.error(f"Error retrieving memories: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
            
    @staticmethod
    async def generate_flashback(user_id, conversation_id, npc_id, current_context):
        """
        Generate a flashback moment for an NPC to reference a significant past memory
        that relates to the current context.
        """
        # First, retrieve relevant memories
        memories = await MemoryManager.retrieve_relevant_memories(
            user_id, conversation_id, npc_id, "npc", 
            context=current_context, limit=3
        )
        
        if not memories:
            return None
            
        # Select a memory, favoring emotional or traumatic ones
        emotional_memories = [m for m in memories if m.memory_type in [MemoryType.EMOTIONAL, MemoryType.TRAUMATIC]]
        selected_memory = random.choice(emotional_memories) if emotional_memories else random.choice(memories)
        
        # Get the NPC's name
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT npc_name FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (user_id, conversation_id, npc_id))
        row = cursor.fetchone()
        conn.close()
        
        npc_name = row[0] if row else "the NPC"
        
        # Format the flashback
        flashback_text = f"{npc_name}'s expression shifts momentarily, a distant look crossing their face. \"This reminds me of {selected_memory.text}\""
        
        if selected_memory.emotional_valence < -5:
            flashback_text += " A shadow crosses their face at the memory."
        elif selected_memory.emotional_valence > 5:
            flashback_text += " Their eyes light up at the pleasant memory."
            
        return {
            "type": "flashback",
            "npc_id": npc_id,
            "npc_name": npc_name,
            "text": flashback_text,
            "memory": selected_memory.text
        }

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
    async def initialize_npc_mask(user_id, conversation_id, npc_id, overwrite=False):
        """
        Create an initial mask for an NPC based on their attributes,
        generating contrasting presented vs. hidden traits
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # First check if mask already exists
            if not overwrite:
                cursor.execute("""
                    SELECT mask_data FROM NPCMasks
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (user_id, conversation_id, npc_id))
                
                row = cursor.fetchone()
                if row:
                    return {"message": "Mask already exists for this NPC", "already_exists": True}
            
            # Get NPC data
            cursor.execute("""
                SELECT npc_name, dominance, cruelty, personality_traits, archetype_summary
                FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
            """, (user_id, conversation_id, npc_id))
            
            row = cursor.fetchone()
            if not row:
                return {"error": f"NPC with id {npc_id} not found"}
                
            npc_name, dominance, cruelty, personality_traits_json, archetype_summary = row
            
            # Parse personality traits
            personality_traits = []
            if personality_traits_json:
                if isinstance(personality_traits_json, str):
                    try:
                        personality_traits = json.loads(personality_traits_json)
                    except json.JSONDecodeError:
                        personality_traits = []
                else:
                    personality_traits = personality_traits_json
            
            # Generate presented and hidden traits
            presented_traits = {}
            hidden_traits = {}
            
            # Use dominance and cruelty to determine mask severity
            mask_depth = (dominance + cruelty) / 2
            
            # More dominant/cruel NPCs have more to hide
            num_masked_traits = int(mask_depth / 20) + 1  # 1-5 masked traits
            
            # Generate contrasting traits based on existing personality
            trait_candidates = {}
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
            
            # Create mask object
            mask = NPCMask(presented_traits, hidden_traits, [])
            
            # Store in database
            cursor.execute("""
                INSERT INTO NPCMasks (user_id, conversation_id, npc_id, mask_data)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, conversation_id, npc_id)
                DO UPDATE SET mask_data = EXCLUDED.mask_data
            """, (user_id, conversation_id, npc_id, json.dumps(mask.to_dict())))
            
            conn.commit()
            
            return {
                "npc_id": npc_id,
                "npc_name": npc_name,
                "mask_created": True,
                "presented_traits": presented_traits,
                "hidden_traits": hidden_traits,
                "message": "NPC mask created successfully"
            }
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error initializing NPC mask: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()
    
    @staticmethod
    async def get_npc_mask(user_id, conversation_id, npc_id):
        """
        Retrieve an NPC's mask data
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT mask_data FROM NPCMasks
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
            """, (user_id, conversation_id, npc_id))
            
            row = cursor.fetchone()
            if not row:
                # Try to initialize a mask
                result = await ProgressiveRevealManager.initialize_npc_mask(user_id, conversation_id, npc_id)
                
                if "error" in result:
                    return result
                
                # Get the new mask
                cursor.execute("""
                    SELECT mask_data FROM NPCMasks
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (user_id, conversation_id, npc_id))
                
                row = cursor.fetchone()
                if not row:
                    return {"error": "Failed to create mask"}
            
            mask_data_json = row[0]
            
            mask_data = {}
            if mask_data_json:
                if isinstance(mask_data_json, str):
                    try:
                        mask_data = json.loads(mask_data_json)
                    except json.JSONDecodeError:
                        mask_data = {}
                else:
                    mask_data = mask_data_json
            
            # Get NPC basic info
            cursor.execute("""
                SELECT npc_name, dominance, cruelty
                FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
            """, (user_id, conversation_id, npc_id))
            
            npc_row = cursor.fetchone()
            if not npc_row:
                return {"error": f"NPC with id {npc_id} not found"}
                
            npc_name, dominance, cruelty = npc_row
            
            # Create mask object
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
            
        except Exception as e:
            logging.error(f"Error getting NPC mask: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()
    
    @staticmethod
    async def generate_mask_slippage(user_id, conversation_id, npc_id, trigger=None, 
                                   severity=None, reveal_type=None):
        """
        Generate a mask slippage event for an NPC based on their true nature
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get mask data
            mask_result = await ProgressiveRevealManager.get_npc_mask(user_id, conversation_id, npc_id)
            
            if "error" in mask_result:
                return mask_result
                
            npc_name = mask_result["npc_name"]
            presented_traits = mask_result["presented_traits"]
            hidden_traits = mask_result["hidden_traits"]
            integrity = mask_result["integrity"]
            reveal_history = mask_result["reveal_history"]
            
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
                "npc_id": npc_id,
                "npc_name": npc_name,
                "reveal_type": reveal_type,
                "severity": severity,
                "trait_revealed": trait,
                "description": reveal_description,
                "integrity_before": integrity,
                "integrity_after": new_integrity,
                "special_event": special_event
            }
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error generating mask slippage: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()
    
    @staticmethod
    async def check_for_automated_reveals(user_id, conversation_id):
        """
        Check for automatic reveals based on various triggers
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get all NPCs with masks
            cursor.execute("""
                SELECT m.npc_id, m.mask_data, n.npc_name, n.dominance, n.cruelty
                FROM NPCMasks m
                JOIN NPCStats n ON m.npc_id = n.npc_id 
                    AND m.user_id = n.user_id 
                    AND m.conversation_id = n.conversation_id
                WHERE m.user_id=%s AND m.conversation_id=%s
            """, (user_id, conversation_id))
            
            npc_masks = cursor.fetchall()
            
            # Get current time
            cursor.execute("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=%s AND conversation_id=%s AND key='TimeOfDay'
            """, (user_id, conversation_id))
            
            time_row = cursor.fetchone()
            time_of_day = time_row[0] if time_row else "Morning"
            
            # Each time period has a chance of reveal for each NPC
            reveal_chance = {
                "Morning": 0.1,
                "Afternoon": 0.15,
                "Evening": 0.2,
                "Night": 0.25  # Higher chance at night when guards are down
            }
            
            base_chance = reveal_chance.get(time_of_day, 0.15)
            reveals = []
            
            for npc_id, mask_data_json, npc_name, dominance, cruelty in npc_masks:
                mask_data = {}
                if mask_data_json:
                    if isinstance(mask_data_json, str):
                        try:
                            mask_data = json.loads(mask_data_json)
                        except json.JSONDecodeError:
                            mask_data = {}
                    else:
                        mask_data = mask_data_json
                
                mask = NPCMask.from_dict(mask_data)
                
                # Higher dominance/cruelty increases chance of slip
                modifier = (dominance + cruelty) / 200  # 0.0 to 1.0
                
                # Lower integrity increases chance of slip
                integrity_factor = (100 - mask.integrity) / 100  # 0.0 to 1.0
                
                # Calculate final chance
                final_chance = base_chance + (modifier * 0.2) + (integrity_factor * 0.3)
                
                # Roll for reveal
                if random.random() < final_chance:
                    # Generate a reveal
                    reveal_result = await ProgressiveRevealManager.generate_mask_slippage(
                        user_id, conversation_id, npc_id
                    )
                    
                    if "error" not in reveal_result:
                        reveals.append(reveal_result)
            
            return reveals
            
        except Exception as e:
            logging.error(f"Error checking for automated reveals: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    @staticmethod
    async def get_perception_difficulty(user_id, conversation_id, npc_id):
        """
        Calculate how difficult it is to see through an NPC's mask
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get mask data
            cursor.execute("""
                SELECT mask_data FROM NPCMasks
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
            """, (user_id, conversation_id, npc_id))
            
            row = cursor.fetchone()
            if not row:
                return {"error": f"No mask found for NPC with id {npc_id}"}
                
            mask_data_json = row[0]
            
            mask_data = {}
            if mask_data_json:
                if isinstance(mask_data_json, str):
                    try:
                        mask_data = json.loads(mask_data_json)
                    except json.JSONDecodeError:
                        mask_data = {}
                else:
                    mask_data = mask_data_json
            
            # Get NPC stats
            cursor.execute("""
                SELECT dominance, cruelty
                FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
            """, (user_id, conversation_id, npc_id))
            
            npc_row = cursor.fetchone()
            if not npc_row:
                return {"error": f"NPC with id {npc_id} not found"}
                
            dominance, cruelty = npc_row
            
            # Get player stats
            cursor.execute("""
                SELECT mental_resilience, confidence
                FROM PlayerStats
                WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
            """, (user_id, conversation_id))
            
            player_row = cursor.fetchone()
            
            if player_row:
                mental_resilience, confidence = player_row
            else:
                mental_resilience, confidence = 50, 50  # Default values
            
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
                "npc_id": npc_id,
                "mask_integrity": mask.integrity,
                "difficulty_score": total_difficulty,
                "player_perception": perception_ability,
                "relative_difficulty": relative_difficulty,
                "difficulty_rating": difficulty_rating
            }
            
        except Exception as e:
            logging.error(f"Error calculating perception difficulty: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()
