# logic/addiction_system.py

"""
Enhanced Addiction System for a GPT-Generated Dynamic Femdom Roleplaying Game

Features:
- Object-oriented addiction management with a base Addiction class and subclasses.
- Asynchronous database operations (adaptable to an async DB library).
- External configuration for thematic messages.
- Integration with NPCStats and PlayerStats to trigger special narrative events.
- Dynamic GPT-generated narrative scenes for pivotal moments.
- Robust error handling and detailed logging.
"""

import logging
import random
import json
import asyncio
import re
from datetime import datetime
from db.connection import get_db_connection
from logic.gpt_integration import get_openai_client

# -------------------------------------------------------------------------------
# Helper Functions for GPT Integration (stubs; replace with your implementations)
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
    Attempt to extract a field value from text using a regex.
    This is a simple fallback mechanism.
    """
    pattern = rf'"{field_name}"\s*:\s*"([^"]+)"'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

# -------------------------------------------------------------------------------
# Constants & External Configuration
# -------------------------------------------------------------------------------

ADDICTION_LEVELS = {
    0: "None",
    1: "Mild",
    2: "Moderate",
    3: "Heavy",
    4: "Extreme"
}

# Attempt to load thematic messages from an external JSON file; fallback if unavailable.
try:
    with open("thematic_messages.json", "r") as f:
        THEMATIC_MESSAGES = json.load(f)
except Exception as e:
    logging.warning("Could not load external thematic messages; using defaults.")
    THEMATIC_MESSAGES = {
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

# -------------------------------------------------------------------------------
# Base Addiction Class & Specific Addiction Subclasses
# -------------------------------------------------------------------------------

class Addiction:
    def __init__(self, level=0):
        self.level = level

    def get_messages(self):
        """
        Returns a list of thematic messages based on the current addiction level.
        Should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def update_level(self, progression_chance=0.2, progression_multiplier=1.0, regression_chance=0.1):
        """
        Update the addiction level based on random chance.
        Returns a tuple (previous_level, new_level).
        """
        roll = random.random()
        previous_level = self.level
        if roll < progression_chance * progression_multiplier and self.level < 4:
            self.level += 1
            logging.info(f"Addiction progressing from {previous_level} to {self.level}")
        elif roll > (1 - regression_chance) and self.level > 0:
            self.level -= 1
            logging.info(f"Addiction regressing from {previous_level} to {self.level}")
        return previous_level, self.level

    def apply_stat_penalties(self, user_id, conversation_id, player_name):
        """
        Apply stat penalties if the addiction is extreme.
        At level 4, apply a penalty to willpower (and optionally other stats).
        """
        if self.level == 4:
            apply_stat_penalty(user_id, conversation_id, player_name, stat="willpower", penalty=5)
            logging.info("Extreme addiction level reached; additional penalties may be applied.")

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

# Mapping addiction type to its corresponding class.
ADDICTION_CLASSES = {
    "socks": SocksAddiction,
    "feet": FeetAddiction,
    "sweat": SweatAddiction,
    "ass": AssAddiction,
    "scent": ScentAddiction,
    "humiliation": HumiliationAddiction,
    "submission": SubmissionAddiction,
}

def create_addiction(addiction_type, level):
    AddictionClass = ADDICTION_CLASSES.get(addiction_type)
    if not AddictionClass:
        raise ValueError(f"Unsupported addiction type: {addiction_type}")
    return AddictionClass(level=level)

# -------------------------------------------------------------------------------
# Helper Functions for Stat Penalties and Dynamic Narrative Events
# -------------------------------------------------------------------------------

def apply_stat_penalty(user_id, conversation_id, player_name, stat="willpower", penalty=5):
    """
    Apply a penalty to a player's stat in the PlayerStats table.
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
        logging.info(f"Applied {penalty} penalty to {stat} for player {player_name}")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error applying {stat} penalty: {str(e)}")
    finally:
        cursor.close()
        conn.close()

async def fetch_npc_data(user_id, conversation_id, npc_id):
    """
    Retrieve full NPC details from the NPCStats table and return as a dictionary.
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
        logging.error(f"Error fetching NPC data: {str(e)}")
        return {}
    finally:
        cursor.close()
        conn.close()

async def gpt_generate_special_event(user_id, conversation_id, npc_data, addiction_type, addiction_level, context_desc):
    """
    Dynamically generate a narrative event scene using GPT for an extreme NPC-specific addiction.
    
    Parameters:
        npc_data (dict): NPC details (e.g., npc_name, archetype summaries, relationships, stats).
        addiction_type (str): The addiction type triggering this event.
        addiction_level (int): The current addiction level.
        context_desc (str): Contextual/environmental description to set the scene.
    
    Returns:
        str: The generated narrative event extracted from a JSON object with key "special_event".
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
Generate a dynamic and immersive narrative event scene for a femdom roleplaying game.
In this scenario, the player's extreme addiction (level {addiction_level} - {ADDICTION_LEVELS.get(addiction_level, "Unknown")}) to {npc_name}'s {addiction_type} has triggered a pivotal moment.

NPC DETAILS:
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

YOUR TASK:
Craft a detailed narrative event spanning 2-3 paragraphs that vividly portrays the intensity of this moment.
- Use sensual, dynamic, and evocative language appropriate for mature femdom themes.
- Integrate the NPC's archetypal traits and relationship dynamics into the scene.
- Highlight the psychological and physical impact of the player's addiction using rich sensory details (sight, sound, touch, scent).
- Conclude with an impactful twist or challenge that hints at future narrative progression.

Return a valid JSON object with the key "special_event" containing the generated narrative as a string.
"""

    client = get_openai_client()
    try:
        response = client.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            # Ensure output is in JSON format; adjust as needed.
        )
        event_json = response.choices[0].message.content
        data = safe_json_loads(event_json)
        if data and "special_event" in data:
            return data["special_event"]

        # Fallback: Try regex extraction if JSON parsing fails.
        event_description = extract_field_from_text(event_json, "special_event")
        if event_description and len(event_description) > 50:
            return event_description

    except Exception as e:
        logging.error(f"Error generating special event for {npc_name}: {e}")

    # Final fallback narrative.
    return f"{npc_name}'s presence becomes overwhelming, triggering a moment of intense, transformative submission that marks the beginning of a dangerous new chapter in your servitude."

# -------------------------------------------------------------------------------
# Addiction Manager: Database Operations & Overall Addiction Management
# -------------------------------------------------------------------------------

class AddictionManager:
    """
    Manages player addictions, including database operations, updating levels, and processing narrative effects.
    """
    @staticmethod
    async def check_addiction_levels(user_id, conversation_id, player_name):
        """
        Check current addiction levels for all addiction types.
        Returns a dict with general and NPC-specific addictions.
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
            cursor.execute("""
                SELECT addiction_type, level, target_npc_id
                FROM PlayerAddictions
                WHERE user_id = %s AND conversation_id = %s AND player_name = %s
            """, (user_id, conversation_id, player_name))
            addiction_data = {}
            npc_specific = []
            for row in cursor.fetchall():
                addiction_type, level, target_npc_id = row
                if target_npc_id is None:
                    addiction_data[addiction_type] = level
                else:
                    # Fetch basic NPC name info.
                    cursor.execute("""
                        SELECT npc_name FROM NPCStats
                        WHERE user_id = %s AND conversation_id = %s AND npc_id = %s
                    """, (user_id, conversation_id, target_npc_id))
                    npc_row = cursor.fetchone()
                    if npc_row:
                        npc_specific.append({
                            "addiction_type": addiction_type,
                            "level": level,
                            "npc_id": target_npc_id,
                            "npc_name": npc_row[0]
                        })
            return {
                "addiction_levels": addiction_data,
                "npc_specific_addictions": npc_specific,
                "has_addictions": any(lvl > 0 for lvl in addiction_data.values()) or len(npc_specific) > 0
            }
        except Exception as e:
            logging.error(f"Error checking addiction levels: {str(e)}")
            return {
                "addiction_levels": {},
                "npc_specific_addictions": [],
                "has_addictions": False,
                "error": str(e)
            }
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    async def update_addiction_level(user_id, conversation_id, player_name, addiction_type,
                                     progression_chance=0.2, progression_multiplier=1.0,
                                     regression_chance=0.1, target_npc_id=None):
        """
        Update an addiction level (or create one if not present) and update the database accordingly.
        Returns a dict with update details.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            if target_npc_id is None:
                cursor.execute("""
                    SELECT level FROM PlayerAddictions
                    WHERE user_id = %s AND conversation_id = %s AND player_name = %s 
                    AND addiction_type = %s AND target_npc_id IS NULL
                """, (user_id, conversation_id, player_name, addiction_type))
            else:
                cursor.execute("""
                    SELECT level FROM PlayerAddictions
                    WHERE user_id = %s AND conversation_id = %s AND player_name = %s 
                    AND addiction_type = %s AND target_npc_id = %s
                """, (user_id, conversation_id, player_name, addiction_type, target_npc_id))
            row = cursor.fetchone()
            current_level = row[0] if row else 0

            addiction_instance = create_addiction(addiction_type, current_level)
            prev_level, new_level = addiction_instance.update_level(progression_chance, progression_multiplier, regression_chance)
            if target_npc_id is None:
                cursor.execute("""
                    INSERT INTO PlayerAddictions 
                    (user_id, conversation_id, player_name, addiction_type, level, last_updated)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (user_id, conversation_id, player_name, addiction_type, target_npc_id)
                    DO UPDATE SET level = %s, last_updated = NOW()
                """, (user_id, conversation_id, player_name, addiction_type, new_level, new_level))
            else:
                cursor.execute("""
                    INSERT INTO PlayerAddictions 
                    (user_id, conversation_id, player_name, addiction_type, level, target_npc_id, last_updated)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (user_id, conversation_id, player_name, addiction_type, target_npc_id)
                    DO UPDATE SET level = %s, last_updated = NOW()
                """, (user_id, conversation_id, player_name, addiction_type, new_level, target_npc_id, new_level))
            conn.commit()
            addiction_instance.apply_stat_penalties(user_id, conversation_id, player_name)
            return {
                "addiction_type": addiction_type,
                "previous_level": current_level,
                "new_level": new_level,
                "level_name": ADDICTION_LEVELS[new_level],
                "progressed": new_level > current_level,
                "regressed": new_level < current_level,
                "target_npc_id": target_npc_id
            }
        except Exception as e:
            conn.rollback()
            logging.error(f"Error updating addiction level for '{addiction_type}': {str(e)}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    async def process_addiction_effects(user_id, conversation_id, player_name, addiction_status):
        """
        Process current addiction levels and generate thematic messages.
        For NPC-specific addictions reaching high levels, dynamically trigger narrative events.
        Returns a dict containing the narrative effects.
        """
        effects = []
        addiction_levels = addiction_status.get("addiction_levels", {})
        npc_specific = addiction_status.get("npc_specific_addictions", [])

        # Process general (player) addictions.
        for addiction_type, level in addiction_levels.items():
            if level <= 0:
                continue
            addiction_instance = create_addiction(addiction_type, level)
            messages = addiction_instance.get_messages()
            effects.extend(messages)

        # Process NPC-specific addictions.
        for npc_addiction in npc_specific:
            npc_name = npc_addiction.get("npc_name", "Someone")
            addiction_type = npc_addiction.get("addiction_type", "")
            level = npc_addiction.get("level", 0)
            if level >= 3:
                effects.append(f"You have a powerful {ADDICTION_LEVELS[level].lower()} addiction to {npc_name}'s {addiction_type}.")
                if level >= 4:
                    effects.append(f"When {npc_name} is near, your submission intensifies—you can barely control your impulses regarding their {addiction_type}.")
                    # Fetch full NPC details dynamically.
                    npc_data = await fetch_npc_data(user_id, conversation_id, npc_addiction["npc_id"])
                    # Define an environmental context (adjust as needed).
                    context_desc = "The environment is charged with dark tension, amplifying every sensation and emotion."
                    event_message = await gpt_generate_special_event(user_id, conversation_id, npc_data, addiction_type, level, context_desc)
                    if event_message:
                        effects.append(event_message)
        return {
            "effects": effects,
            "has_effects": len(effects) > 0
        }

    @staticmethod
    async def get_addiction_status(user_id, conversation_id, player_name):
        """
        Wrapper to retrieve the current addiction status for a player.
        """
        return await AddictionManager.check_addiction_levels(user_id, conversation_id, player_name)

# -------------------------------------------------------------------------------
# Helper Function: Get Addiction Level Label
# -------------------------------------------------------------------------------

def get_addiction_label(level):
    """Return the textual label for an addiction level."""
    return ADDICTION_LEVELS.get(level, "Unknown")
