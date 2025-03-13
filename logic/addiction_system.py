# logic/addiction_system_agentic.py
"""
Comprehensive End-to-End Addiction System with an Agentic approach using OpenAI's Agents SDK.

Features:
1) An addiction manager that handles DB logic: checking, updating, and processing narrative effects.
2) Subclasses of Addiction for specialized behaviors: SocksAddiction, FeetAddiction, etc.
3) Function tools bridging your existing Python logic so the LLM-based agent can invoke them.
4) An "AddictionAgent" that uses these tools and can produce orchestrated results.
5) Example usage demonstrating how to run the agent in code.
"""

import logging
import random
import json
import re
import asyncio
from datetime import datetime

# ~~~~~~~~~ Agents SDK imports ~~~~~~~~~
from agents import (
    Agent,
    ModelSettings,
    Runner,
    function_tool,
    RunContextWrapper
)
from agents.models.openai_responses import OpenAIResponsesModel

# ~~~~~~~~~ DB & GPT placeholders ~~~~~~~~~
# Adjust to point to your actual code or database. 
from db.connection import get_db_connection
from logic.chatgpt_integration import get_openai_client

# -------------------------------------------------------------------------------
# Global Constants & Thematic Messages
# -------------------------------------------------------------------------------

ADDICTION_LEVELS = {
    0: "None",
    1: "Mild",
    2: "Moderate",
    3: "Heavy",
    4: "Extreme"
}

# Default fallback if external JSON is missing
DEFAULT_THEMATIC_MESSAGES = {
    "socks": {
        "1": "You occasionally steal glances at sumptuous stockings.",
        "2": "A subtle craving for the delicate feel of silk emerges within you.",
        "3": "The allure of sensuous socks overwhelms your thoughts.",
        "4": "Under your Mistress’s commanding presence, your obsession with exquisite socks leaves you trembling in servile adoration."
    },
    "feet": {
        "1": "Your eyes frequently wander to the graceful arch of bare feet.",
        "2": "A surge of forbidden excitement courses through you at the mere glimpse of uncovered toes.",
        "3": "Distracted by the sight of enticing feet, you find it difficult to focus on anything else.",
        "4": "In the presence of your dominant Mistress, your fixation on every tantalizing curve of feet renders you utterly submissive."
    },
    "sweat": {
        "1": "The scent of perspiration begins to evoke an unspoken thrill within you.",
        "2": "Each drop of sweat stokes a simmering desire you dare not fully acknowledge.",
        "3": "Your senses heighten as the aroma of exertion casts a spell over your inhibitions.",
        "4": "Overwhelmed by the intoxicating allure of sweat, you are compelled to seek it out under your Mistress’s relentless command."
    },
    "ass": {
        "1": "Your gaze lingers a little longer on the curves of a well-shaped rear.",
        "2": "A subtle, forbidden thrill courses through you at the sight of a pert backside.",
        "3": "You find yourself fixated on every tantalizing detail of exposed derrieres, your mind wandering into submissive fantasies.",
        "4": "Under your Mistress’s unwavering control, your obsession with perfectly sculpted rear ends drives you to desperate submission."
    },
    "scent": {
        "1": "You become acutely aware of natural pheromones and subtle scents around you.",
        "2": "Every hint of an enticing aroma sends a shiver down your spine, awakening deep desires.",
        "3": "You begin to collect memories of scents, each evoking a surge of submissive longing.",
        "4": "In the grip of your extreme addiction, the mere whiff of a scent under your Mistress’s watchful eye reduces you to euphoric submission."
    },
    "humiliation": {
        "1": "The sting of humiliation sparks a curious thrill in your submissive heart.",
        "2": "You find yourself yearning for more degrading scenarios as your pride withers under each slight.",
        "3": "Every act of public embarrassment intensifies your craving to be dominated and humiliated.",
        "4": "In the presence of your ruthless Mistress, the exquisite agony of humiliation consumes you, binding your will entirely to her desires."
    },
    "submission": {
        "1": "The taste of obedience becomes subtly intoxicating as you seek her approval in every glance.",
        "2": "Your need to surrender grows, craving the reassurance that only your Mistress can provide.",
        "3": "In every command, you find a deeper satisfaction in your subjugated state, yearning to be molded by her hand.",
        "4": "Your identity dissolves in the overwhelming tide of submission, as your Mistress’s word becomes the sole law governing your existence."
    }
}

try:
    with open("thematic_messages.json", "r") as f:
        THEMATIC_MESSAGES = json.load(f)
except Exception as e:
    logging.warning("Could not load external thematic messages; using defaults.")
    THEMATIC_MESSAGES = DEFAULT_THEMATIC_MESSAGES

# -------------------------------------------------------------------------------
# Helper Utilities for JSON
# -------------------------------------------------------------------------------

def safe_json_loads(json_str):
    """
    Safely attempt to load a JSON string.
    """
    try:
        return json.loads(json_str)
    except Exception as e:
        logging.error(f"JSON parsing error: {e}")
        return None

def extract_field_from_text(text, field_name):
    """
    Attempt to extract a field value from text using a simple regex fallback.
    """
    pattern = rf'"{field_name}"\s*:\s*"([^"]+)"'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

# -------------------------------------------------------------------------------
# Base Addiction & Subclasses
# -------------------------------------------------------------------------------

class Addiction:
    def __init__(self, level=0):
        self.level = level

    def get_messages(self):
        """Return a list of thematic messages based on current addiction level."""
        raise NotImplementedError("Subclasses must implement this method.")

    def update_level(self, progression_chance=0.2, progression_multiplier=1.0, regression_chance=0.1):
        """
        Update the addiction level with some random chance:
          - progress if random roll < progression_chance * multiplier
          - regress if random roll > 1 - regression_chance
        """
        roll = random.random()
        previous_level = self.level
        if roll < (progression_chance * progression_multiplier) and self.level < 4:
            self.level += 1
            logging.info(f"Addiction progressed from {previous_level} to {self.level}")
        elif roll > (1 - regression_chance) and self.level > 0:
            self.level -= 1
            logging.info(f"Addiction regressed from {previous_level} to {self.level}")
        return previous_level, self.level

    def apply_stat_penalties(self, user_id, conversation_id, player_name):
        """
        If addiction is Extreme (level 4), apply stat penalty to player's willpower, etc.
        """
        if self.level == 4:
            apply_stat_penalty(user_id, conversation_id, player_name, stat="willpower", penalty=5)
            logging.info("Applying extreme addiction penalty to willpower.")

class SocksAddiction(Addiction):
    def get_messages(self):
        return [THEMATIC_MESSAGES["socks"].get(str(lvl), "") for lvl in range(1, self.level + 1)]

class FeetAddiction(Addiction):
    def get_messages(self):
        return [THEMATIC_MESSAGES["feet"].get(str(lvl), "") for lvl in range(1, self.level + 1)]

class SweatAddiction(Addiction):
    def get_messages(self):
        return [THEMATIC_MESSAGES["sweat"].get(str(lvl), "") for lvl in range(1, self.level + 1)]

class AssAddiction(Addiction):
    def get_messages(self):
        return [THEMATIC_MESSAGES["ass"].get(str(lvl), "") for lvl in range(1, self.level + 1)]

class ScentAddiction(Addiction):
    def get_messages(self):
        return [THEMATIC_MESSAGES["scent"].get(str(lvl), "") for lvl in range(1, self.level + 1)]

class HumiliationAddiction(Addiction):
    def get_messages(self):
        return [THEMATIC_MESSAGES["humiliation"].get(str(lvl), "") for lvl in range(1, self.level + 1)]

class SubmissionAddiction(Addiction):
    def get_messages(self):
        return [THEMATIC_MESSAGES["submission"].get(str(lvl), "") for lvl in range(1, self.level + 1)]

# Map string -> class
ADDICTION_CLASSES = {
    "socks": SocksAddiction,
    "feet": FeetAddiction,
    "sweat": SweatAddiction,
    "ass": AssAddiction,
    "scent": ScentAddiction,
    "humiliation": HumiliationAddiction,
    "submission": SubmissionAddiction
}

def create_addiction(addiction_type, level):
    """
    Create an addiction object from its type and level.
    """
    cls = ADDICTION_CLASSES.get(addiction_type)
    if not cls:
        raise ValueError(f"Unsupported addiction type: {addiction_type}")
    return cls(level=level)

# -------------------------------------------------------------------------------
# Stat / DB Helper
# -------------------------------------------------------------------------------

def apply_stat_penalty(user_id, conversation_id, player_name, stat="willpower", penalty=5):
    """
    Example: deduct 'penalty' from the 'stat' column in 'PlayerStats'.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = f"""
            UPDATE PlayerStats
            SET {stat} = GREATEST({stat} - %s, 0)
            WHERE user_id = %s AND conversation_id = %s AND player_name = %s
        """
        cursor.execute(query, (penalty, user_id, conversation_id, player_name))
        conn.commit()
        logging.info(f"Applied {penalty} penalty to {stat} for player {player_name}.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error applying {stat} penalty: {str(e)}")
    finally:
        cursor.close()
        conn.close()

# -------------------------------------------------------------------------------
# GPT-Driven "Special Event" Generation for Extreme Addictions
# -------------------------------------------------------------------------------

async def fetch_npc_data(user_id, conversation_id, npc_id):
    """
    Retrieves NPC details from NPCStats table (sample columns).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT npc_name, archetype_summary, archetype_extras_summary, relationships,
                   personality_traits, likes, dislikes, dominance, cruelty, intensity
            FROM NPCStats
            WHERE user_id = %s AND conversation_id = %s AND npc_id = %s
        """, (user_id, conversation_id, npc_id))
        row = cursor.fetchone()
        if row:
            return {
                "npc_name": row[0],
                "archetype_summary": row[1],
                "archetype_extras_summary": row[2],
                "relationships": row[3],
                "personality_traits": row[4],
                "likes": row[5],
                "dislikes": row[6],
                "dominance": row[7],
                "cruelty": row[8],
                "intensity": row[9],
            }
        return {}
    except Exception as e:
        logging.error(f"Error fetching NPC data: {e}")
        return {}
    finally:
        cursor.close()
        conn.close()

async def gpt_generate_special_event(user_id, conversation_id, npc_data, addiction_type, addiction_level, context_desc):
    """
    Dynamically generate a 2-3 paragraph event scene for an extreme addiction,
    returned as JSON with "special_event" key. Example:
      { "special_event": "The atmosphere crackles with tension..." }
    """
    npc_name = npc_data.get("npc_name", "Unknown NPC")
    archetype_summary = npc_data.get("archetype_summary", "")
    archetype_extras = npc_data.get("archetype_extras_summary", "")
    relationships = npc_data.get("relationships", "{}")
    personality_traits = npc_data.get("personality_traits", [])
    likes = npc_data.get("likes", [])
    dislikes = npc_data.get("dislikes", [])
    dominance = npc_data.get("dominance", 50)
    cruelty = npc_data.get("cruelty", 30)
    intensity = npc_data.get("intensity", 40)

    prompt = f"""
Generate a dynamic, immersive narrative event scene for a femdom roleplaying game.
The player's addiction is at level {addiction_level} ({ADDICTION_LEVELS.get(addiction_level, "Unknown")})
to {npc_name}'s {addiction_type}, triggering a critical moment.

NPC Details:
Name: {npc_name}
Archetype summary: {archetype_summary}
Archetype extras: {archetype_extras}
Stats: Dominance {dominance}/100, Cruelty {cruelty}/100, Intensity {intensity}/100
Relationships: {relationships}
Personality Traits: {personality_traits}
Likes: {likes}
Dislikes: {dislikes}

ENVIRONMENT:
{context_desc}

GOAL:
Write 2-3 paragraphs of intense narrative. Return valid JSON with a key "special_event"
that holds the generated text. 
"""
    client = get_openai_client()
    try:
        response = client.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
        )
        event_json = response.choices[0].message.content
        data = safe_json_loads(event_json)
        if data and "special_event" in data:
            return data["special_event"]

        # fallback: naive extraction by regex
        event_description = extract_field_from_text(event_json, "special_event")
        if event_description and len(event_description) > 50:
            return event_description

    except Exception as e:
        logging.error(f"Error generating special event: {e}")

    return f"{npc_name}'s presence overwhelms you, marking a dangerous new chapter in your servitude."

# -------------------------------------------------------------------------------
# The AddictionManager: direct DB logic
# -------------------------------------------------------------------------------

class AddictionManager:
    """
    Manages creation, retrieval, update of PlayerAddictions in your DB,
    plus logic for generating narrative messages or special events.
    """

    @staticmethod
    def ensure_table_exists():
        """
        Create PlayerAddictions if it doesn't exist.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS PlayerAddictions (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    player_name VARCHAR(255) NOT NULL,
                    addiction_type VARCHAR(50) NOT NULL,
                    level INTEGER NOT NULL DEFAULT 0,
                    target_npc_id INTEGER NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, conversation_id, player_name, addiction_type, target_npc_id)
                )
            """)
            conn.commit()
        except Exception as e:
            logging.error(f"Could not create PlayerAddictions table: {e}")
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    async def check_addiction_levels(user_id, conversation_id, player_name):
        """
        Retrieve both general and NPC-specific addictions for the given player.
        Returns a dict with "addiction_levels" and "npc_specific_addictions".
        """
        AddictionManager.ensure_table_exists()
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT addiction_type, level, target_npc_id
                FROM PlayerAddictions
                WHERE user_id=%s AND conversation_id=%s AND player_name=%s
            """, (user_id, conversation_id, player_name))

            addiction_data = {}
            npc_specific = []
            for row in cursor.fetchall():
                addiction_type, level, target_npc_id = row
                if target_npc_id is None:
                    addiction_data[addiction_type] = level
                else:
                    # optionally fetch NPC name
                    cursor.execute("""
                        SELECT npc_name FROM NPCStats
                        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                    """, (user_id, conversation_id, target_npc_id))
                    r2 = cursor.fetchone()
                    npc_name = r2[0] if r2 else f"NPC#{target_npc_id}"
                    npc_specific.append({
                        "addiction_type": addiction_type,
                        "level": level,
                        "npc_id": target_npc_id,
                        "npc_name": npc_name
                    })
            return {
                "addiction_levels": addiction_data,
                "npc_specific_addictions": npc_specific,
                "has_addictions": any(lvl > 0 for lvl in addiction_data.values()) or bool(npc_specific)
            }
        except Exception as e:
            logging.error(f"Error checking addiction levels: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    async def update_addiction_level(
        user_id, conversation_id, player_name,
        addiction_type,
        progression_chance=0.2, progression_multiplier=1.0,
        regression_chance=0.1, target_npc_id=None
    ):
        """
        Update or create a new addiction record in the DB. Returns old/new levels + additional data.
        """
        AddictionManager.ensure_table_exists()
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            # fetch current level if exists
            if target_npc_id is None:
                cursor.execute("""
                    SELECT level FROM PlayerAddictions
                    WHERE user_id=%s AND conversation_id=%s AND player_name=%s
                      AND addiction_type=%s AND target_npc_id IS NULL
                """, (user_id, conversation_id, player_name, addiction_type))
            else:
                cursor.execute("""
                    SELECT level FROM PlayerAddictions
                    WHERE user_id=%s AND conversation_id=%s AND player_name=%s
                      AND addiction_type=%s AND target_npc_id=%s
                """, (user_id, conversation_id, player_name, addiction_type, target_npc_id))
            row = cursor.fetchone()
            current_level = row[0] if row else 0

            instance = create_addiction(addiction_type, current_level)
            prev_level, new_level = instance.update_level(
                progression_chance, progression_multiplier, regression_chance
            )

            # insert or update
            if target_npc_id is None:
                cursor.execute("""
                    INSERT INTO PlayerAddictions
                        (user_id, conversation_id, player_name, addiction_type, level, last_updated)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (user_id, conversation_id, player_name, addiction_type, target_npc_id)
                    DO UPDATE SET level=EXCLUDED.level, last_updated=NOW()
                """, (user_id, conversation_id, player_name, addiction_type, new_level))
            else:
                cursor.execute("""
                    INSERT INTO PlayerAddictions
                        (user_id, conversation_id, player_name, addiction_type, level, target_npc_id, last_updated)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (user_id, conversation_id, player_name, addiction_type, target_npc_id)
                    DO UPDATE SET level=EXCLUDED.level, last_updated=NOW()
                """, (user_id, conversation_id, player_name, addiction_type, new_level, target_npc_id))

            conn.commit()

            # apply penalty if extreme
            instance.apply_stat_penalties(user_id, conversation_id, player_name)

            return {
                "addiction_type": addiction_type,
                "previous_level": prev_level,
                "new_level": new_level,
                "level_name": ADDICTION_LEVELS.get(new_level, "Unknown"),
                "progressed": new_level > prev_level,
                "regressed": new_level < prev_level,
                "target_npc_id": target_npc_id
            }
        except Exception as e:
            conn.rollback()
            logging.error(f"Error updating addiction: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    async def process_addiction_effects(user_id, conversation_id, player_name, addiction_status):
        """
        Summarize or generate narrative for each addiction, including special events for level 4.
        Returns: { "effects": [...], "has_effects": bool }
        """
        effects = []

        # general addictions
        addiction_levels = addiction_status.get("addiction_levels", {})
        for addiction_type, lvl in addiction_levels.items():
            if lvl <= 0:
                continue
            a = create_addiction(addiction_type, lvl)
            messages = a.get_messages()
            # filter out empty strings
            effects.extend(msg for msg in messages if msg)

        # npc-specific
        npc_specific = addiction_status.get("npc_specific_addictions", [])
        for entry in npc_specific:
            addiction_type = entry["addiction_type"]
            npc_name = entry.get("npc_name", f"NPC#{entry['npc_id']}")
            lvl = entry["level"]
            if lvl >= 3:
                effects.append(f"You have a {ADDICTION_LEVELS[lvl]} addiction to {npc_name}'s {addiction_type}.")
                if lvl >= 4:
                    # fetch npc data, run GPT for special event
                    npc_data = await fetch_npc_data(user_id, conversation_id, entry["npc_id"])
                    context_desc = "A tense, dimly lit environment intensifies every sensation."
                    sp_event = await gpt_generate_special_event(
                        user_id, conversation_id, npc_data, addiction_type, lvl, context_desc
                    )
                    if sp_event:
                        effects.append(sp_event)

        return {
            "effects": effects,
            "has_effects": bool(effects)
        }

# -------------------------------------------------------------------------------
# Tools (function_tool) that the agent can call
# -------------------------------------------------------------------------------

@function_tool
async def check_addiction_levels_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    player_name: str
) -> dict:
    """
    Checks the player's addiction levels. 
    Returns a dict with 'addiction_levels' and 'npc_specific_addictions'.
    """
    return await AddictionManager.check_addiction_levels(user_id, conversation_id, player_name)

@function_tool
async def update_addiction_level_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    player_name: str,
    addiction_type: str,
    progression_chance: float = 0.2,
    progression_multiplier: float = 1.0,
    regression_chance: float = 0.1,
    target_npc_id: int = None
) -> dict:
    """
    Update or create an addiction entry. 
    Returns old/new levels, whether it progressed or regressed, etc.
    """
    return await AddictionManager.update_addiction_level(
        user_id, conversation_id, player_name,
        addiction_type,
        progression_chance, progression_multiplier,
        regression_chance, target_npc_id
    )

@function_tool
async def process_addiction_effects_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    player_name: str,
    addiction_status: dict
) -> dict:
    """
    Generate narrative effects for the player's addictions. 
    E.g. returning special messages or GPT events for level 4.
    """
    return await AddictionManager.process_addiction_effects(
        user_id, conversation_id, player_name, addiction_status
    )

# -------------------------------------------------------------------------------
# The Agent: "AddictionAgent"
# -------------------------------------------------------------------------------

AddictionAgent = Agent(
    name="AddictionAgent",
    instructions=(
        "You are a specialized 'Addiction Manager Agent' in a femdom roleplaying context. "
        "You have multiple function tools that let you:\n"
        " - check_addiction_levels_tool(user_id, conversation_id, player_name)\n"
        " - update_addiction_level_tool(user_id, conversation_id, player_name, addiction_type, ...)\n"
        " - process_addiction_effects_tool(user_id, conversation_id, player_name, addiction_status)\n\n"
        "When the user asks about addiction states or requests an update, you may call these tools. "
        "Return short, final JSON or text summarizing the result. Don't forget you can chain multiple tool calls."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),  # or "gpt-4o", "gpt-3.5-turbo", etc.
    model_settings=ModelSettings(temperature=0.5),
    tools=[
        check_addiction_levels_tool,
        update_addiction_level_tool,
        process_addiction_effects_tool
    ],
    output_type=None
)
