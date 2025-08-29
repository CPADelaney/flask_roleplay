# logic/memory_logic.py

import os
import json
import random
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

import asyncpg
import asyncio
from quart import Blueprint, request, jsonify, session

from db.connection import get_db_connection_context
from memory.memory_orchestrator import get_memory_orchestrator, EntityType

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

memory_bp = Blueprint("memory_bp", __name__)


async def _orch(user_id: int, conversation_id: int):
    return await get_memory_orchestrator(user_id, conversation_id)


@memory_bp.route("/get_current_roleplay", methods=["GET"])
async def get_current_roleplay():
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
        async with get_db_connection_context() as conn:
            rows = await conn.fetch(
                """
                SELECT key, value
                  FROM currentroleplay
                 WHERE user_id=$1 AND conversation_id=$2
              ORDER BY key
                """,
                user_id,
                conversation_id,
            )
            data = [{"key": r["key"], "value": r["value"]} for r in rows]
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


async def record_npc_event(
    user_id: int, conversation_id: int, npc_id: int, event_description: str
):
    """
    Appends a new NPC event as a memory via MemoryOrchestrator.
    """
    try:
        orch = await _orch(user_id, conversation_id)
        await orch.integrated_add_memory(
            entity_type=EntityType.NPC.value,
            entity_id=npc_id,
            memory_text=event_description,
            memory_kwargs={
                "significance": 3,
                "tags": ["npc_event", "roleplay"],
                "metadata": {"source": "logic.memory_logic"},
            },
        )
    except Exception as e:
        logger.error(f"record_npc_event failed: {e}", exc_info=True)


@memory_bp.route("/store_roleplay_segment", methods=["POST"], endpoint="store_roleplay_segment_endpoint")
async def store_roleplay_segment():
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
        segment_value = payload.get("value")  # Any JSON-serializable type

        if not conversation_id:
            return jsonify({"error": "No conversation_id provided"}), 400
        if not segment_key or segment_value is None:
            return jsonify({"error": "Missing 'key' or 'value'"}), 400

        async with get_db_connection_context() as conn:
            await conn.execute(
                """
                INSERT INTO currentroleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = EXCLUDED.value
                """,
                user_id,
                conversation_id,
                segment_key,
                segment_value,
            )

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
    except Exception as e:
        logger.error(f"Error in store_roleplay_segment: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred"}), 500


@memory_bp.route("/update_npc_memory", methods=["POST"])
async def update_npc_memory():
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
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                """
                SELECT npc_name, archetype_summary, archetype_extras_summary
                  FROM NPCStats
                 WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
                """,
                npc_id,
                user_id,
                conversation_id,
            )
            if not row:
                return jsonify({"error": f"NPC with id {npc_id} not found"}), 404
            npc_name = row["npc_name"]
            archetype_summary = row["archetype_summary"]
            archetype_extras_summary = row["archetype_extras_summary"]

        # Generate memory JSON (string)
        memory_json_str = await get_shared_memory(
            user_id,
            conversation_id,
            relationship,
            npc_name,
            archetype_summary or "",
            archetype_extras_summary or "",
        )
        if not memory_json_str:
            return jsonify({"error": "Failed to generate NPC memory via AI"}), 500

        try:
            memory_data = json.loads(memory_json_str)
            memories_list = memory_data.get("memory", [])
        except json.JSONDecodeError:
            logger.error(
                f"Failed to decode JSON memory from GPT for NPC {npc_id}: {memory_json_str[:200]}..."
            )
            return jsonify({"error": "AI returned invalid memory format"}), 500

        if not memories_list:
            logger.warning(f"AI returned empty memory list for NPC {npc_id}")
            return jsonify(
                {"message": "No new memories generated", "memory": []}
            ), 200

        # Store via orchestrator and propagate
        orch = await _orch(user_id, conversation_id)
        memory_added_count = 0
        for mem_text in memories_list:
            if isinstance(mem_text, str) and mem_text.strip():
                await orch.integrated_add_memory(
                    entity_type=EntityType.NPC.value,
                    entity_id=int(npc_id),
                    memory_text=mem_text,
                    memory_kwargs={
                        "significance": MemorySignificance.MEDIUM,
                        "tags": [
                            "shared_memory",
                            f"related_to:{relationship.get('target_name', 'player')}",
                        ],
                        "metadata": {"source": "update_npc_memory"},
                    },
                )
                memory_added_count += 1

        # Optional propagation to other NPCs (secondhand)
        try:
            await propagate_shared_memories(
                user_id, conversation_id, npc_id, npc_name, memories_list
            )
        except Exception:
            pass

        logger.info(
            f"Added {memory_added_count}/{len(memories_list)} generated memories for NPC {npc_id}"
        )
        return jsonify(
            {
                "message": f"NPC memory updated with {memory_added_count} entries",
                "memory_preview": memories_list[0] if memories_list else None,
            }
        ), 200

    except asyncpg.PostgresError as e:
        logger.error(
            f"Database error in update_npc_memory for NPC {npc_id}: {e}", exc_info=True
        )
        return jsonify({"error": "Database error during memory update"}), 500
    except ConnectionError as e:
        logger.error(f"Pool error in update_npc_memory: {e}", exc_info=True)
        return jsonify({"error": "Could not connect to database"}), 503
    except asyncio.TimeoutError:
        logger.error(
            f"Timeout getting DB connection in update_npc_memory", exc_info=True
        )
        return jsonify({"error": "Database timeout during memory update"}), 504
    except Exception as e:
        logger.error(
            f"Error in update_npc_memory for NPC {npc_id}: {e}", exc_info=True
        )
        return jsonify({"error": "An internal error occurred during memory update"}), 500


async def get_stored_setting(conn, user_id, conversation_id):
    """
    Retrieve the setting name and description from CurrentRoleplay (using an existing connection).
    """
    rows = await conn.fetch(
        """
        SELECT key, value
          FROM CurrentRoleplay
         WHERE user_id=$1 AND conversation_id=$2
           AND key IN ('CurrentSetting', 'EnvironmentDesc')
        """,
        user_id,
        conversation_id,
    )
    result = {r["key"]: r["value"] for r in rows}
    result.setdefault("CurrentSetting", "Default Setting Name")
    result.setdefault("EnvironmentDesc", "Default environment description.")
    return result


async def propagate_shared_memories(
    user_id: int, conversation_id: int, source_npc_id: int, source_npc_name: str, memories: List[str]
):
    """
    For each memory text, check if it mentions other NPCs in the conversation
    and add a secondhand memory to their logs via MemoryOrchestrator.
    """
    if not memories:
        return

    orch = await _orch(user_id, conversation_id)

    try:
        async with get_db_connection_context() as conn:
            # 1) Build map of { npc_name_lower: npc_id }
            rows = await conn.fetch(
                """
                SELECT npc_id, LOWER(npc_name) as name_lower
                  FROM NPCStats
                 WHERE user_id=$1 AND conversation_id=$2
                """,
                user_id,
                conversation_id,
            )
            name_to_id_map = {r["name_lower"]: r["npc_id"] for r in rows}

        if not name_to_id_map:
            logger.warning(f"No NPCs found to propagate memories in conv {conversation_id}")
            return

        # 2) Check each memory against other NPC names (simple substring)
        tasks: List[asyncio.Task] = []
        for mem_text in memories:
            mem_text_lower = mem_text.lower()
            for other_npc_name_lower, other_npc_id in name_to_id_map.items():
                if other_npc_id == source_npc_id:
                    continue
                if other_npc_name_lower in mem_text_lower:
                    secondhand_text = f"I heard something about {source_npc_name}: \"{mem_text}\""
                    tasks.append(
                        asyncio.create_task(
                            orch.integrated_add_memory(
                                entity_type=EntityType.NPC.value,
                                entity_id=int(other_npc_id),
                                memory_text=secondhand_text,
                                memory_kwargs={
                                    "significance": MemorySignificance.LOW,
                                    "tags": [
                                        "propagated",
                                        "secondhand",
                                        f"from_npc:{source_npc_id}",
                                    ],
                                    "metadata": {"source": "propagation"},
                                },
                            )
                        )
                    )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    except asyncpg.PostgresError as e:
        logger.error(
            f"Database error during memory propagation in conv {conversation_id}: {e}",
            exc_info=True,
        )
    except ConnectionError as e:
        logger.error(f"Pool error during memory propagation: {e}", exc_info=True)
    except asyncio.TimeoutError:
        logger.error("Timeout getting DB connection during memory propagation", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error during memory propagation: {e}", exc_info=True)


async def fetch_formatted_locations(user_id: int, conversation_id: int) -> str:
    """
    Query Locations table and format results into a bulleted string.
    """
    formatted = ""
    try:
        async with get_db_connection_context() as conn:
            rows = await conn.fetch(
                """
                SELECT location_name, description
                  FROM Locations
                 WHERE user_id = $1 AND conversation_id = $2
                """,
                user_id,
                conversation_id,
            )

            if not rows:
                return "No locations found.\n"

            for loc in rows:
                location_name = loc["location_name"]
                desc = loc["description"]
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


async def get_shared_memory(
    user_id: int,
    conversation_id: int,
    relationship: dict,
    npc_name: str,
    archetype_summary: str = "",
    archetype_extras_summary: str = "",
) -> Optional[str]:
    """
    Generates shared memory text using GPT, incorporating DB lookups.
    Returns JSON string with {"memory": [str, str, str]} or None on failure.
    """
    from logic.chatgpt_integration import get_openai_client

    logger.info(
        f"Starting get_shared_memory for NPC '{npc_name}' with relationship: {relationship}"
    )

    mega_description = "an undefined setting"
    current_setting = "Default Setting Name"
    locations_table_formatted = "No location data available.\n"

    try:
        # Fetch stored environment details
        async with get_db_connection_context() as conn:
            stored_settings = await get_stored_setting(conn, user_id, conversation_id)
            mega_description = stored_settings.get("EnvironmentDesc", "an undefined setting")
            current_setting = stored_settings.get("CurrentSetting", "Default Setting Name")
            logger.info(f"Retrieved environment desc (first 100): {mega_description[:100]}...")
            logger.info(f"Current setting: {current_setting}")

        # Fetch formatted locations
        locations_table_formatted = await fetch_formatted_locations(user_id, conversation_id)
        logger.info(f"Formatted locations retrieved:\n{locations_table_formatted}")

    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logger.error(f"Database error fetching context for get_shared_memory: {db_err}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error fetching context in get_shared_memory: {e}", exc_info=True)

    target_name = relationship.get("target_name", "the player")

    example_output = {
        "memory": [
            f"I still remember the first time {target_name} and I crossed paths at the marketplace. "
            "The scent of fresh spices hung in the air as our eyes met briefly over a display of exotic fruits, "
            "creating an unexpected moment of connection in the midst of the bustling crowd.",
            f"Last month when {target_name} challenged my authority during the council meeting, I felt my temper flare dangerously. "
            "The tension in the room was palpable as our voices rose, but somehow that confrontation led to a grudging respect between us.",
            "The afternoon we spent by the lake skipping stones remains vivid in my mind. "
            "The cool mist on my face and the soft laughter we shared created a rare moment of peace between us.",
        ]
    }

    # Build the instruction text
    system_instructions = f"""
# Memory Generation for {npc_name}

## Relationship Context
{npc_name} has a relationship with {target_name} that may encompass multiple roles reflecting {npc_name}'s complete character.
These memories should authentically represent all aspects of {npc_name}'s identity—whether they involve familial bonds, professional connections, or unique dynamics based on their defining traits.

## Setting Information
- Current World: {mega_description}
- Current Setting: {current_setting}
- Key Locations:
{locations_table_formatted}
- Additional Context:
{(('Background: ' + archetype_summary + '. ') if archetype_summary else '')}{(('Extra Details: ' + archetype_extras_summary + '. ') if archetype_extras_summary else '')}

## Memory Generation Guidelines
1. Generate THREE distinct first-person memories from {npc_name}'s perspective about interactions with {target_name}.
2. Each memory must be 2-3 sentences in {npc_name}'s authentic voice.
3. Set each memory in a specific location from the provided list or another contextually appropriate location in {current_setting}.
4. Include at least one vivid sensory detail per memory.
5. Include one positive, one challenging, and one additional dimension memory.

REQUIRED OUTPUT:
Return ONLY a valid JSON object exactly like:
{json.dumps(example_output, indent=2)}
"""

    logger.info("Calling GPT for shared memory generation...")

    max_retries = 2
    last_exception: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Memory generation attempt {attempt}/{max_retries}")

            client = get_openai_client()
            # Use Responses API; request JSON object
            resp = client.responses.create(
                model="gpt-5-nano",
                instructions="You generate exactly three JSON-formatted first-person memories for the NPC.",
                input=system_instructions,
                text={"format": {"type": "json_object"}},
            )

            output_text = (getattr(resp, "output_text", None) or "").strip()
            if not output_text:
                logger.warning("Empty response from model")
                raise ValueError("Empty model response")

            # Validate JSON and ensure exactly 3 memories
            try:
                data = json.loads(output_text)
            except json.JSONDecodeError:
                # Try to extract a JSON object if model wrapped it oddly
                extracted = extract_or_create_memory_fallback(output_text, npc_name, target_name)
                if extracted:
                    return extracted
                raise

            if "memory" not in data or not isinstance(data["memory"], list):
                logger.warning("Model output missing 'memory' key or not a list")
                extracted = extract_or_create_memory_fallback(output_text, npc_name, target_name)
                return extracted

            memories = data["memory"]
            if len(memories) < 3:
                logger.warning(f"Only received {len(memories)} memories, expecting 3")
                # top up via fallback
                return extract_or_create_memory_fallback(json.dumps(data), npc_name, target_name)
            if len(memories) > 3:
                data["memory"] = memories[:3]

            return json.dumps(data)

        except Exception as e:
            last_exception = e
            logger.error(
                f"Error during GPT call in get_shared_memory (attempt {attempt}): {e}",
                exc_info=True,
            )
            if attempt < max_retries:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

    logger.error(
        f"Failed to generate memory after {max_retries} attempts. Last error: {last_exception}"
    )
    return None


def extract_or_create_memory_fallback(text_output: str, npc_name: str, target_name: str) -> str:
    """
    Attempts to extract memories from malformed output or creates fallbacks.
    """
    import re

    json_pattern = r'\{.*"memory"\s*:\s*\[.*\].*\}'
    match = re.search(json_pattern, text_output, re.DOTALL)

    if match:
        try:
            extracted_json = match.group(0)
            memory_data = json.loads(extracted_json)
            if "memory" in memory_data and isinstance(memory_data["memory"], list):
                memories = memory_data["memory"]
                while len(memories) < 3:
                    idx = len(memories)
                    if idx == 0:
                        memories.append(
                            f"I remember meeting {target_name} for the first time. "
                            "There was something about their presence that left a lasting impression on me."
                        )
                    elif idx == 1:
                        memories.append(
                            f"Once {target_name} and I had a disagreement that tested our relationship. "
                            "Despite the tension, we found a way to resolve our differences."
                        )
                    else:
                        memories.append(
                            f"I cherish the quiet moments {target_name} and I have shared. "
                            "Those simple times together strengthened our bond."
                        )
                return json.dumps({"memory": memories[:3]})
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extracted JSON: {e}")

    # Complete fallback
    return create_memory_fallback(npc_name, target_name)


def create_memory_fallback(npc_name: str, target_name: str) -> str:
    """
    Creates a basic set of fallback memories when all else fails.
    """
    logger.warning(f"Using fallback memory generation for {npc_name} and {target_name}")
    memories = [
        f"I still remember when I first met {target_name}. There was an immediate sense of connection I hadn't expected.",
        f"The time {target_name} and I had that heated disagreement taught me something important about both of us.",
        f"One quiet evening, {target_name} and I shared a moment of understanding that meant more than words.",
    ]
    return json.dumps({"memory": memories})


# Lightweight DTOs for local usage (route-level)
class MemoryType:
    INTERACTION = "interaction"
    OBSERVATION = "observation"
    EMOTIONAL = "emotional"
    TRAUMATIC = "traumatic"
    INTIMATE = "intimate"


class MemorySignificance:
    LOW = 1
    MEDIUM = 3
    HIGH = 5
    CRITICAL = 10


class EnhancedMemory:
    """
    Enhanced memory class for route-level transformations only.
    """
    def __init__(self, text, memory_type=MemoryType.INTERACTION, significance=MemorySignificance.MEDIUM):
        self.text = text
        self.timestamp = datetime.now().isoformat()
        self.memory_type = memory_type
        self.significance = significance
        self.recall_count = 0
        self.last_recalled = None
        self.emotional_valence = 0  # -10 to +10
        self.tags: List[str] = []

    def to_dict(self):
        return {
            "text": self.text,
            "timestamp": self.timestamp,
            "memory_type": self.memory_type,
            "significance": self.significance,
            "recall_count": self.recall_count,
            "last_recalled": self.last_recalled,
            "emotional_valence": self.emotional_valence,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data):
        m = cls(data.get("text", ""), data.get("memory_type", MemoryType.INTERACTION), data.get("significance", MemorySignificance.MEDIUM))
        m.timestamp = data.get("timestamp", datetime.now().isoformat())
        m.recall_count = data.get("recall_count", 0)
        m.last_recalled = data.get("last_recalled")
        m.emotional_valence = data.get("emotional_valence", 0)
        m.tags = data.get("tags", []) or []
        return m


class MemoryManager:
    """
    Thin compatibility wrapper that delegates to MemoryOrchestrator.
    """

    @staticmethod
    async def add_memory(
        user_id: int,
        conversation_id: int,
        entity_id: Union[int, str],
        entity_type: str,
        memory_text: str,
        memory_type: str = MemoryType.INTERACTION,
        significance: int = MemorySignificance.MEDIUM,
        emotional_valence: int = 0,
        tags: Optional[List[str]] = None,
    ) -> bool:
        tags = tags or []
        try:
            orch = await _orch(user_id, conversation_id)
            await orch.integrated_add_memory(
                entity_type=entity_type,
                entity_id=int(entity_id) if isinstance(entity_id, str) and entity_id.isdigit() else entity_id,
                memory_text=memory_text,
                memory_kwargs={
                    "significance": significance,
                    "tags": tags + [f"type:{memory_type}"],
                    "metadata": {"emotional_valence": emotional_valence, "source": "logic.memory_logic"},
                },
            )
            return True
        except Exception as e:
            logger.error(f"Orchestrator add_memory failed for {entity_type} {entity_id}: {e}", exc_info=True)
            return False

    @staticmethod
    async def propagate_significant_memory(
        user_id: int,
        conversation_id: int,
        source_entity_id: Union[int, str],
        source_entity_type: str,
        memory: EnhancedMemory,
    ) -> bool:
        """
        Propagate significant memories to related NPCs via orchestrator.
        """
        try:
            orch = await _orch(user_id, conversation_id)

            async with get_db_connection_context() as conn:
                links = await conn.fetch(
                    """
                    SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id, dynamics
                      FROM SocialLinks
                     WHERE user_id=$1 AND conversation_id=$2
                       AND ((entity1_type=$3 AND entity1_id=$4) OR (entity2_type=$3 AND entity2_id=$4))
                    """,
                    user_id,
                    conversation_id,
                    source_entity_type,
                    int(source_entity_id),
                )

            strong = []
            for link in links:
                dyn = link["dynamics"]
                if isinstance(dyn, str):
                    try:
                        dyn = json.loads(dyn)
                    except Exception:
                        dyn = {}
                if (dyn or {}).get("trust", 0) >= 50 or (dyn or {}).get("affection", 0) >= 50:
                    strong.append(link)

            tasks: List[asyncio.Task] = []
            for link in strong:
                e1t, e1i = link["entity1_type"], str(link["entity1_id"])
                e2t, e2i = link["entity2_type"], str(link["entity2_id"])
                if e1t == source_entity_type and e1i == str(source_entity_id):
                    tgt_t, tgt_i = e2t, e2i
                else:
                    tgt_t, tgt_i = e1t, e1i

                if tgt_t != "npc" or tgt_i == str(source_entity_id):
                    continue

                secondhand_text = f"I heard that {memory.text}"
                tasks.append(
                    asyncio.create_task(
                        orch.integrated_add_memory(
                            entity_type=tgt_t,
                            entity_id=int(tgt_i) if tgt_i.isdigit() else tgt_i,
                            memory_text=secondhand_text,
                            memory_kwargs={
                                "significance": max(MemorySignificance.LOW, memory.significance - 2),
                                "tags": (memory.tags or []) + ["secondhand", "propagated"],
                                "metadata": {"source_entity": f"{source_entity_type}:{source_entity_id}"},
                            },
                        )
                    )
                )

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            return True

        except Exception as e:
            logger.error(f"Propagation via orchestrator failed: {e}", exc_info=True)
            return False

    @staticmethod
    async def retrieve_relevant_memories(
        user_id: int,
        conversation_id: int,
        entity_id: Union[int, str],
        entity_type: str,
        context: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[EnhancedMemory]:
        """
        Retrieve top memories via orchestrator.retrieve_memories.
        """
        try:
            orch = await _orch(user_id, conversation_id)
            res = await orch.retrieve_memories(
                entity_type=entity_type,
                entity_id=int(entity_id) if isinstance(entity_id, str) and entity_id.isdigit() else entity_id,
                limit=limit,
                tags=tags or None,
            )
            items = res.get("memories", []) or []

            out: List[EnhancedMemory] = []
            for m in items:
                em = EnhancedMemory(
                    text=m.get("text") or m.get("memory_text", ""),
                    memory_type=str(m.get("memory_type") or m.get("type") or MemoryType.INTERACTION),
                    significance=int(m.get("significance", MemorySignificance.MEDIUM)),
                )
                em.emotional_valence = int((m.get("metadata", {}) or {}).get("emotional_valence", 0))
                em.tags = list(m.get("tags") or [])
                em.timestamp = (m.get("timestamp") or m.get("created_at") or datetime.now().isoformat())
                out.append(em)
            return out

        except Exception as e:
            logger.error(f"Unified retrieval failed: {e}", exc_info=True)
            return []

    @staticmethod
    async def generate_flashback(
        user_id: int, conversation_id: int, npc_id: int, current_context: str
    ) -> Optional[dict]:
        """
        Use orchestrator.create_flashback for NPCs.
        """
        try:
            orch = await _orch(user_id, conversation_id)
            fb = await orch.create_flashback(
                entity_type=EntityType.NPC.value, entity_id=npc_id, trigger=current_context
            )
            if not fb or "text" not in fb:
                return None

            # Fetch name for display
            async with get_db_connection_context() as conn:
                npc_name = await conn.fetchval(
                    """
                    SELECT npc_name
                      FROM NPCStats
                     WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                    """,
                    user_id,
                    conversation_id,
                    npc_id,
                ) or "the NPC"

            return {
                "type": "flashback",
                "npc_id": npc_id,
                "npc_name": npc_name,
                "text": fb.get("text"),
                "memory": fb.get("text"),
            }

        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(
                f"Database error during flashback generation for NPC {npc_id}: {db_err}",
                exc_info=True,
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error during flashback generation for NPC {npc_id}: {e}",
                exc_info=True,
            )
            return None
