# logic/social_links_agentic.py
"""
Comprehensive End-to-End Social Links System with an Agentic approach using OpenAI's Agents SDK.

Features:
1) Core CRUD and advanced relationship logic for SocialLinks.
2) Relationship dynamics, Crossroads, Ritual checking, multi-dimensional relationships.
3) Function tools bridging your existing Python logic so the LLM-based agent can invoke them.
4) A "SocialLinksAgent" that uses these tools.
5) Example usage demonstrating how to run the agent in code.
"""

import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

# ~~~~~~~~~ Agents SDK imports ~~~~~~~~~
from agents import (
    Agent,
    ModelSettings,
    Runner,
    function_tool,
    RunContextWrapper
)
from agents.models.openai_responses import OpenAIResponsesModel

# ~~~~~~~~~ DB imports & any other placeholders ~~~~~~~~~
from db.connection import get_db_connection


# ~~~~~~~~~ Logging Configuration ~~~~~~~~~
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Simple Core CRUD for SocialLinks Table
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_social_link(
    user_id: int,
    conversation_id: int,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int
) -> Optional[Dict[str, Any]]:
    """
    Fetch an existing social link row if it exists for (user_id, conversation_id, e1, e2).
    Returns a dict with link_id, link_type, link_level, link_history, etc., or None if not found.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT link_id, link_type, link_level, link_history
            FROM SocialLinks
            WHERE user_id=%s AND conversation_id=%s
              AND entity1_type=%s AND entity1_id=%s
              AND entity2_type=%s AND entity2_id=%s
            """,
            (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id)
        )
        row = cursor.fetchone()
        if row:
            (link_id, link_type, link_level, link_hist) = row
            return {
                "link_id": link_id,
                "link_type": link_type,
                "link_level": link_level,
                "link_history": link_hist,
            }
        return None
    finally:
        cursor.close()
        conn.close()


def create_social_link(
    user_id: int,
    conversation_id: int,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int,
    link_type: str = "neutral",
    link_level: int = 0
) -> int:
    """
    Create a new SocialLinks row for (user_id, conversation_id, e1, e2).
    Initializes link_history as an empty array.
    If a matching row already exists, returns its link_id.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO SocialLinks (
                user_id, conversation_id,
                entity1_type, entity1_id,
                entity2_type, entity2_id,
                link_type, link_level, link_history
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, '[]')
            ON CONFLICT (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id)
            DO NOTHING
            RETURNING link_id
            """,
            (
                user_id,
                conversation_id,
                entity1_type,
                entity1_id,
                entity2_type,
                entity2_id,
                link_type,
                link_level,
            )
        )
        result = cursor.fetchone()
        if result is None:
            # If the insert did nothing because row exists, fetch existing link_id
            cursor.execute(
                """
                SELECT link_id
                FROM SocialLinks
                WHERE user_id=%s
                  AND conversation_id=%s
                  AND entity1_type=%s AND entity1_id=%s
                  AND entity2_type=%s AND entity2_id=%s
                """,
                (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id),
            )
            result = cursor.fetchone()
        link_id = result[0]
        conn.commit()
        return link_id
    except:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def update_link_type_and_level(
    user_id: int,
    conversation_id: int,
    link_id: int,
    new_type: Optional[str] = None,
    level_change: int = 0
) -> Optional[Dict[str, Any]]:
    """
    Adjust an existing link's type or level, within user_id+conversation_id+link_id scope.
    Returns a dict with new_type, new_level if updated, or None if no row found.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT link_type, link_level
            FROM SocialLinks
            WHERE link_id=%s AND user_id=%s AND conversation_id=%s
            """,
            (link_id, user_id, conversation_id)
        )
        row = cursor.fetchone()
        if not row:
            return None

        (old_type, old_level) = row
        final_type = new_type if new_type else old_type
        final_level = old_level + level_change

        cursor.execute(
            """
            UPDATE SocialLinks
            SET link_type=%s, link_level=%s
            WHERE link_id=%s AND user_id=%s AND conversation_id=%s
            """,
            (final_type, final_level, link_id, user_id, conversation_id)
        )
        conn.commit()
        return {
            "link_id": link_id,
            "new_type": final_type,
            "new_level": final_level,
        }
    except:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def add_link_event(
    user_id: int,
    conversation_id: int,
    link_id: int,
    event_text: str
):
    """
    Append a string to link_history for link_id (scoped to user_id+conversation_id).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            UPDATE SocialLinks
            SET link_history = COALESCE(link_history, '[]'::jsonb) || to_jsonb(%s)
            WHERE link_id=%s AND user_id=%s AND conversation_id=%s
            RETURNING link_history
            """,
            (event_text, link_id, user_id, conversation_id)
        )
        updated = cursor.fetchone()
        if not updated:
            logging.warning(f"No link found for link_id={link_id}, user_id={user_id}, conv={conversation_id}")
        else:
            logging.info(f"Appended event to link_history => {updated[0]}")
        conn.commit()
    except:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Relationship Dynamics, Crossroads, Rituals
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RELATIONSHIP_DYNAMICS = [
    {
        "name": "control",
        "description": "One entity exerts control over the other",
        "levels": [
            {"level": 10, "name": "Subtle Influence", "description": "Occasional suggestions that subtly guide behavior"},
            {"level": 30, "name": "Regular Direction", "description": "Frequent guidance that shapes decisions"},
            {"level": 50, "name": "Strong Authority", "description": "Explicit direction with expectation of compliance"},
            {"level": 70, "name": "Dominant Control", "description": "Commands given with assumption of obedience"},
            {"level": 90, "name": "Complete Dominance", "description": "Total control with no expectation of resistance"},
        ],
    },
    {
        "name": "manipulation",
        "description": "One entity manipulates the other through indirect means",
        "levels": [
            {"level": 10, "name": "Minor Misdirection", "description": "Occasional white lies to achieve small goals"},
            {"level": 30, "name": "Regular Deception", "description": "Consistent pattern of misleading to shape behavior"},
            {"level": 50, "name": "Calculated Manipulation", "description": "Strategic dishonesty to achieve control"},
            {"level": 70, "name": "Psychological Conditioning", "description": "Systematic reshaping of target's reality"},
            {"level": 90, "name": "Complete Gaslighting", "description": "Target's entire perception is controlled"},
        ],
    },
    {
        "name": "dependency",
        "description": "One entity becomes dependent on the other",
        "levels": [
            {"level": 10, "name": "Mild Attachment", "description": "Enjoys presence but functions fine independently"},
            {"level": 30, "name": "Regular Reliance", "description": "Seeks input for decisions and emotional support"},
            {"level": 50, "name": "Strong Dependency", "description": "Difficulty making decisions without approval"},
            {"level": 70, "name": "Heavy Dependency", "description": "Significant anxiety when separated"},
            {"level": 90, "name": "Complete Dependence", "description": "Unable to function independently"},
        ],
    },
    {
        "name": "alliance",
        "description": "Entities work together for mutual benefit",
        "levels": [
            {"level": 10, "name": "Casual Cooperation", "description": "Occasional collaboration when convenient"},
            {"level": 30, "name": "Regular Partnership", "description": "Consistent working relationship"},
            {"level": 50, "name": "Strong Alliance", "description": "Prioritize mutual support in most situations"},
            {"level": 70, "name": "Deep Partnership", "description": "Significant investment in each other's success"},
            {"level": 90, "name": "Unbreakable Alliance", "description": "Complete mutual commitment"},
        ],
    },
    {
        "name": "rivalry",
        "description": "Entities compete with each other",
        "levels": [
            {"level": 10, "name": "Mild Competition", "description": "Occasional friendly contests"},
            {"level": 30, "name": "Regular Rivalry", "description": "Consistent pattern of trying to outdo each other"},
            {"level": 50, "name": "Strong Competition", "description": "Active attempts to undermine each other"},
            {"level": 70, "name": "Intense Rivalry", "description": "Significant efforts to dominate the other"},
            {"level": 90, "name": "Bitter Enemies", "description": "Complete dedication to defeating the other"},
        ],
    },
]

RELATIONSHIP_CROSSROADS = [
    {
        "name": "Trust Test",
        "description": "A moment where trust is either solidified or broken",
        "trigger_level": 40,
        "dynamic": "control",
        "choices": [
            {
                "text": "Allow increased control",
                "effects": {"control": +15, "dependency": +10, "player_willpower": -5},
                "outcome": "You accept {npc_name}'s explanation and allow her greater involvement in your decisions. The relief in surrendering control is both unexpected and concerning.",
            },
            {
                "text": "Establish boundary",
                "effects": {"control": -10, "rivalry": +15, "player_willpower": +5},
                "outcome": "You firmly establish a boundary with {npc_name}. Her expression flickers between surprise and something colder before she composes herself.",
            },
        ],
    },
    {
        "name": "Public Display",
        "description": "A test of compliance in a public setting",
        "trigger_level": 60,
        "dynamic": "control",
        "choices": [
            {
                "text": "Comply despite discomfort",
                "effects": {"control": +20, "manipulation": +10, "player_confidence": -10},
                "outcome": "You follow {npc_name}'s instructions despite your discomfort. The approval in her eyes provides a confusing sense of validation.",
            },
            {
                "text": "Refuse publicly",
                "effects": {"control": -15, "rivalry": +20, "player_confidence": +5},
                "outcome": "You refuse {npc_name}'s request, causing a momentary tension. Later, she approaches you with a different demeanor, reassessing her approach.",
            },
        ],
    },
    {
        "name": "Manipulation Revealed",
        "description": "Player discovers evidence of manipulation",
        "trigger_level": 50,
        "dynamic": "manipulation",
        "choices": [
            {
                "text": "Confront directly",
                "effects": {"manipulation": -10, "rivalry": +15, "player_mental_resilience": +10},
                "outcome": "You confront {npc_name} about her deception. She seems genuinely caught off-guard by your assertion, quickly adapting with a new approach.",
            },
            {
                "text": "Pretend not to notice",
                "effects": {"manipulation": +15, "dependency": +5, "player_mental_resilience": -10},
                "outcome": "You keep your discovery to yourself, watching as {npc_name} continues her manipulations with growing confidence, unaware that you see through them.",
            },
        ],
    },
    {
        "name": "Support Need",
        "description": "NPC appears to need emotional support",
        "trigger_level": 30,
        "dynamic": "dependency",
        "choices": [
            {
                "text": "Provide unconditional support",
                "effects": {"dependency": +20, "manipulation": +15, "player_corruption": +10},
                "outcome": "You offer complete support to {npc_name}, prioritizing her needs above your own. Her vulnerability feels oddly calculated, but the bond strengthens.",
            },
            {
                "text": "Offer limited support",
                "effects": {"dependency": -5, "rivalry": +5, "player_corruption": -5},
                "outcome": "You offer support while maintaining some distance. {npc_name} seems disappointed but respects your boundaries, adjusting her approach accordingly.",
            },
        ],
    },
    {
        "name": "Alliance Opportunity",
        "description": "Chance to deepen alliance with significant commitment",
        "trigger_level": 40,
        "dynamic": "alliance",
        "choices": [
            {
                "text": "Commit fully to alliance",
                "effects": {"alliance": +25, "dependency": +10, "player_corruption": +5},
                "outcome": "You fully commit to your partnership with {npc_name}, integrating your goals with hers. The efficiency of your collaboration masks the gradual shift in power.",
            },
            {
                "text": "Maintain independence",
                "effects": {"alliance": -10, "manipulation": +5, "player_corruption": -5},
                "outcome": "You maintain some independence from {npc_name}'s influence. She seems to accept this with grace, though you notice new, more subtle approaches to integration.",
            },
        ],
    },
]

RELATIONSHIP_RITUALS = [
    {
        "name": "Formal Agreement",
        "description": "A formalized arrangement that defines relationship expectations",
        "trigger_level": 60,
        "dynamics": ["control", "alliance"],
        "ritual_text": (
            "{npc_name} presents you with an arrangement that feels strangely formal. "
            "'Let's be clear about our expectations,' she says with an intensity that makes it more than casual. "
            "The terms feel reasonable, almost beneficial, yet something about the ritual makes you acutely aware of a threshold being crossed."
        ),
    },
    {
        "name": "Symbolic Gift",
        "description": "A gift with deeper symbolic meaning that represents the relationship dynamic",
        "trigger_level": 50,
        "dynamics": ["control", "dependency"],
        "ritual_text": (
            "{npc_name} presents you with a gift - {gift_item}. "
            "'A small token,' she says, though her expression suggests deeper significance. "
            "As you accept it, the weight feels heavier than the object itself, as if you're accepting something beyond the physical item."
        ),
    },
    {
        "name": "Private Ceremony",
        "description": "A private ritual that solidifies the relationship's nature",
        "trigger_level": 75,
        "dynamics": ["control", "dependency", "manipulation"],
        "ritual_text": (
            "{npc_name} leads you through what she describes as 'a small tradition' - a sequence of actions and words "
            "that feels choreographed for effect. The intimacy of the moment creates a strange blend of comfort and vulnerability, "
            "like something is being sealed between you."
        ),
    },
    {
        "name": "Public Declaration",
        "description": "A public acknowledgment of the relationship's significance",
        "trigger_level": 70,
        "dynamics": ["alliance", "control"],
        "ritual_text": (
            "At the gathering, {npc_name} makes a point of publicly acknowledging your relationship in front of others. "
            "The words seem innocuous, even complimentary, but the subtext feels laden with meaning. "
            "You notice others' reactions - knowing glances, subtle nods - as if your position has been formally established."
        ),
    },
    {
        "name": "Shared Secret",
        "description": "Disclosure of sensitive information that creates mutual vulnerability",
        "trigger_level": 55,
        "dynamics": ["alliance", "manipulation"],
        "ritual_text": (
            "{npc_name} shares information with you that feels dangerously private. 'I don't tell this to just anyone,' she says "
            "with meaningful eye contact. The knowledge creates an intimacy that comes with implicit responsibility - "
            "you're now a keeper of her secrets, for better or worse."
        ),
    },
]

SYMBOLIC_GIFTS = [
    "a delicate bracelet that feels oddly like a subtle marker of ownership",
    "a key to her home that represents more than just physical access",
    "a personalized item that shows how closely she's been observing your habits",
    "a journal with the first few pages already filled in her handwriting, guiding your thoughts",
    "a piece of jewelry that others in her circle would recognize as significant",
    "a custom phone with 'helpful' modifications already installed",
    "a clothing item that subtly alters how others perceive you in her presence",
]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3) Support Functions for Relationship Dynamics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_primary_dynamic(dynamics: Dict[str, int]) -> str:
    """
    Determine the primary relationship dynamic based on highest numeric level in 'dynamics'.
    """
    if not dynamics:
        return "neutral"
    primary_dynamic = "neutral"
    max_level = 0
    for dname, lvl in dynamics.items():
        if lvl > max_level:
            max_level = lvl
            primary_dynamic = dname
    return primary_dynamic


def get_dynamic_description(dynamic_name: str, level: int) -> str:
    """
    Get the appropriate textual description for a dynamic at a specific level.
    """
    for dyn in RELATIONSHIP_DYNAMICS:
        if dyn["name"] == dynamic_name:
            for level_info in dyn["levels"]:
                if level <= level_info["level"]:
                    return f"{level_info['name']}: {level_info['description']}"
            # If no matching bracket, return the highest bracket
            highest = dyn["levels"][-1]
            return f"{highest['name']}: {highest['description']}"
    return "Unknown dynamic"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4) Crossroad Checking + Ritual Checking
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def check_for_relationship_crossroads(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    Check if any NPC relationship has reached a dynamic level that triggers a Crossroads event.
    Returns the first triggered crossroads event dict (with choices), or None if none triggered.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Gather all player-related links (assuming player is entityX_id == user_id, or you might store differently)
        cursor.execute(
            """
            SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id,
                   dynamics, experienced_crossroads
            FROM SocialLinks
            WHERE user_id=%s AND conversation_id=%s
            AND (
                (entity1_type='player' AND entity1_id=%s)
                OR (entity2_type='player' AND entity2_id=%s)
            )
            """,
            (user_id, conversation_id, user_id, user_id)
        )
        links = cursor.fetchall()

        for link in links:
            link_id, e1t, e1id, e2t, e2id, dynamics_json, crossroads_json = link

            if isinstance(dynamics_json, str):
                try:
                    dynamics = json.loads(dynamics_json)
                except:
                    dynamics = {}
            else:
                dynamics = dynamics_json or {}

            if crossroads_json:
                if isinstance(crossroads_json, str):
                    try:
                        experienced = json.loads(crossroads_json)
                    except:
                        experienced = []
                else:
                    experienced = crossroads_json
            else:
                experienced = []

            # Determine which side is NPC
            if e1t == "npc" and e2t == "player":
                npc_id = e1id
            elif e2t == "npc" and e1t == "player":
                npc_id = e2id
            else:
                continue  # Not an NPC-player link

            # Grab NPC name
            cursor.execute(
                "SELECT npc_name FROM NPCStats WHERE user_id=%s AND conversation_id=%s AND npc_id=%s",
                (user_id, conversation_id, npc_id),
            )
            npcrow = cursor.fetchone()
            if not npcrow:
                continue
            npc_name = npcrow[0]

            # Check each Crossroads
            for crossroads_def in RELATIONSHIP_CROSSROADS:
                if crossroads_def["name"] in experienced:
                    continue
                dynamic_needed = crossroads_def["dynamic"]
                trigger_level = crossroads_def["trigger_level"]
                current_level = dynamics.get(dynamic_needed, 0)

                if current_level >= trigger_level:
                    # Trigger this crossroads
                    formatted_choices = []
                    for ch in crossroads_def["choices"]:
                        fc = {
                            "text": ch["text"],
                            "effects": ch["effects"],
                            "outcome": ch["outcome"].format(npc_name=npc_name),
                        }
                        formatted_choices.append(fc)
                    return {
                        "type": "relationship_crossroads",
                        "name": crossroads_def["name"],
                        "description": crossroads_def["description"],
                        "npc_id": npc_id,
                        "npc_name": npc_name,
                        "choices": formatted_choices,
                        "link_id": link_id,
                    }
        return None

    except Exception as e:
        logging.error(f"Error checking for relationship crossroads: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


def apply_crossroads_choice(
    user_id: int,
    conversation_id: int,
    link_id: int,
    crossroads_name: str,
    choice_index: int
) -> Dict[str, Any]:
    """
    Apply the chosen effect from a Crossroads event and update the link accordingly.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Locate the crossroads definition
        cr_def = None
        for c in RELATIONSHIP_CROSSROADS:
            if c["name"] == crossroads_name:
                cr_def = c
                break
        if not cr_def:
            return {"error": f"Crossroads '{crossroads_name}' not found"}

        # Validate choice
        if choice_index < 0 or choice_index >= len(cr_def["choices"]):
            return {"error": "Invalid choice index"}

        choice = cr_def["choices"][choice_index]

        # Get link details
        cursor.execute(
            """
            SELECT entity1_type, entity1_id, entity2_type, entity2_id, dynamics, experienced_crossroads
            FROM SocialLinks
            WHERE link_id=%s AND user_id=%s AND conversation_id=%s
            """,
            (link_id, user_id, conversation_id),
        )
        row = cursor.fetchone()
        if not row:
            return {"error": "Social link not found"}

        e1t, e1id, e2t, e2id, dyn_json, crossroads_json = row

        # Identify NPC
        if e1t == "npc" and e2t == "player":
            npc_id = e1id
        elif e2t == "npc" and e1t == "player":
            npc_id = e2id
        else:
            return {"error": "No NPC in this relationship"}

        # Grab NPC name
        cursor.execute(
            "SELECT npc_name FROM NPCStats WHERE user_id=%s AND conversation_id=%s AND npc_id=%s",
            (user_id, conversation_id, npc_id),
        )
        npcrow = cursor.fetchone()
        if not npcrow:
            return {"error": "NPC not found"}
        npc_name = npcrow[0]

        # Parse dynamics
        if isinstance(dyn_json, str):
            try:
                dynamics = json.loads(dyn_json)
            except:
                dynamics = {}
        else:
            dynamics = dyn_json or {}

        # Parse experienced crossroads
        if crossroads_json:
            if isinstance(crossroads_json, str):
                try:
                    experienced = json.loads(crossroads_json)
                except:
                    experienced = []
            else:
                experienced = crossroads_json
        else:
            experienced = []

        # Apply effect to either relationship or player stats
        for dynamic_name, delta in choice["effects"].items():
            if dynamic_name.startswith("player_"):
                # Update player stat
                player_stat = dynamic_name[7:]  # remove "player_"
                cursor.execute(
                    f"""
                    UPDATE PlayerStats
                    SET {player_stat} = {player_stat} + %s
                    WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
                    """,
                    (delta, user_id, conversation_id),
                )
            else:
                # Relationship dynamic
                current_val = dynamics.get(dynamic_name, 0)
                new_val = max(0, min(100, current_val + delta))
                dynamics[dynamic_name] = new_val

        # Mark this crossroads as experienced
        experienced.append(crossroads_name)

        # Update the DB
        cursor.execute(
            """
            UPDATE SocialLinks
            SET dynamics=%s,
                experienced_crossroads=%s
            WHERE link_id=%s
            """,
            (json.dumps(dynamics), json.dumps(experienced), link_id),
        )

        # Recompute link_type + link_level based on primary dynamic
        primary = get_primary_dynamic(dynamics)
        level = dynamics.get(primary, 0)
        cursor.execute(
            """
            UPDATE SocialLinks
            SET link_type=%s, link_level=%s
            WHERE link_id=%s
            """,
            (primary, level, link_id),
        )

        # Add event to link_history
        event_text = (
            f"Crossroads '{crossroads_name}' chosen: {choice['text']}. "
            f"Outcome: {choice['outcome'].format(npc_name=npc_name)}"
        )
        cursor.execute(
            """
            UPDATE SocialLinks
            SET link_history = link_history || %s::jsonb
            WHERE link_id=%s
            """,
            (json.dumps([event_text]), link_id),
        )

        # Add to PlayerJournal if desired
        journal_entry = (
            f"Crossroads: {crossroads_name} with {npc_name}. "
            f"Choice: {choice['text']} => {choice['outcome'].format(npc_name=npc_name)}"
        )
        cursor.execute(
            """
            INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, timestamp)
            VALUES (%s, %s, 'relationship_crossroads', %s, CURRENT_TIMESTAMP)
            """,
            (user_id, conversation_id, journal_entry),
        )

        conn.commit()
        return {"success": True, "outcome_text": choice["outcome"].format(npc_name=npc_name)}
    except Exception as e:
        conn.rollback()
        logging.error(f"Error applying crossroads choice: {e}")
        return {"error": str(e)}
    finally:
        cursor.close()
        conn.close()


def check_for_relationship_ritual(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    Checks whether any relationship triggers a ritual event.
    Returns the first triggered ritual event, or None if none is triggered.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Gather all player-related links
        cursor.execute(
            """
            SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id,
                   dynamics, experienced_rituals
            FROM SocialLinks
            WHERE user_id=%s AND conversation_id=%s
            AND (
                (entity1_type='player' AND entity1_id=%s)
                OR (entity2_type='player' AND entity2_id=%s)
            )
            """,
            (user_id, conversation_id, user_id, user_id),
        )
        links = cursor.fetchall()

        for link in links:
            link_id, e1t, e1id, e2t, e2id, dyn_json, rjson = link

            if isinstance(dyn_json, str):
                try:
                    dynamics = json.loads(dyn_json)
                except:
                    dynamics = {}
            else:
                dynamics = dyn_json or {}

            if rjson:
                if isinstance(rjson, str):
                    try:
                        experienced = json.loads(rjson)
                    except:
                        experienced = []
                else:
                    experienced = rjson
            else:
                experienced = []

            # Identify NPC
            if e1t == "npc" and e2t == "player":
                npc_id = e1id
            elif e2t == "npc" and e1t == "player":
                npc_id = e2id
            else:
                continue

            # Check NPC dominance
            cursor.execute(
                "SELECT npc_name, dominance FROM NPCStats WHERE user_id=%s AND conversation_id=%s AND npc_id=%s",
                (user_id, conversation_id, npc_id),
            )
            row_npc = cursor.fetchone()
            if not row_npc:
                continue
            npc_name, npc_dom = row_npc

            # Only if dominance >= 50
            if npc_dom < 50:
                continue

            # See if any ritual is triggered
            possible_rituals = []
            for rit in RELATIONSHIP_RITUALS:
                if rit["name"] in experienced:
                    continue
                triggered = False
                for dyn_name in rit["dynamics"]:
                    current_val = dynamics.get(dyn_name, 0)
                    if current_val >= rit["trigger_level"]:
                        triggered = True
                        break
                if triggered:
                    possible_rituals.append(rit)

            if possible_rituals:
                chosen = random.choice(possible_rituals)
                ritual_txt = chosen["ritual_text"]
                if "{gift_item}" in ritual_txt:
                    gift_item = random.choice(SYMBOLIC_GIFTS)
                    ritual_txt = ritual_txt.format(npc_name=npc_name, gift_item=gift_item)
                else:
                    ritual_txt = ritual_txt.format(npc_name=npc_name)

                # Mark as experienced
                experienced.append(chosen["name"])
                cursor.execute(
                    """
                    UPDATE SocialLinks
                    SET experienced_rituals=%s
                    WHERE link_id=%s
                    """,
                    (json.dumps(experienced), link_id),
                )

                # Add history
                event_text = f"Ritual '{chosen['name']}': {ritual_txt}"
                cursor.execute(
                    """
                    UPDATE SocialLinks
                    SET link_history = link_history || %s::jsonb
                    WHERE link_id=%s
                    """,
                    (json.dumps([event_text]), link_id),
                )

                # Journal entry
                cursor.execute(
                    """
                    INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, timestamp)
                    VALUES (%s, %s, 'relationship_ritual', %s, CURRENT_TIMESTAMP)
                    """,
                    (user_id, conversation_id, f"Ritual with {npc_name}: {chosen['name']}. {ritual_txt}"),
                )

                # Increase relevant dynamics by +10
                for dyn_name in chosen["dynamics"]:
                    old_val = dynamics.get(dyn_name, 0)
                    new_val = min(100, old_val + 10)
                    dynamics[dyn_name] = new_val

                cursor.execute(
                    """
                    UPDATE SocialLinks
                    SET dynamics=%s
                    WHERE link_id=%s
                    """,
                    (json.dumps(dynamics), link_id),
                )

                # Also update PlayerStats
                cursor.execute(
                    """
                    UPDATE PlayerStats
                    SET corruption=corruption+5,
                        dependency=dependency+5
                    WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
                    """,
                    (user_id, conversation_id),
                )

                conn.commit()
                return {
                    "type": "relationship_ritual",
                    "name": chosen["name"],
                    "description": chosen["description"],
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "ritual_text": ritual_txt,
                    "link_id": link_id,
                }
        return None
    except Exception as e:
        conn.rollback()
        logging.error(f"Error checking for relationship ritual: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 5) Summaries & Helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_entity_name(
    conn,
    entity_type: str,
    entity_id: int,
    user_id: int,
    conversation_id: int
) -> str:
    """
    Get the name of an entity (NPC or player).
    """
    if entity_type == "player":
        return "Chase"
    c = conn.cursor()
    c.execute(
        """
        SELECT npc_name FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """,
        (user_id, conversation_id, entity_id),
    )
    row = c.fetchone()
    c.close()
    if row:
        return row[0]
    return "Unknown"


def get_relationship_summary(
    user_id: int,
    conversation_id: int,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int
) -> Optional[Dict[str, Any]]:
    """
    Get a summary of the relationship between two entities.
    Includes link type, level, recent events, dynamics, etc.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT link_id, link_type, link_level, dynamics, link_history,
                   experienced_crossroads, experienced_rituals
            FROM SocialLinks
            WHERE user_id=%s AND conversation_id=%s
              AND entity1_type=%s AND entity1_id=%s
              AND entity2_type=%s AND entity2_id=%s
            """,
            (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id),
        )
        row = cursor.fetchone()
        if not row:
            return None

        link_id, link_type, link_level, dyn_json, hist_json, cr_json, rit_json = row

        if isinstance(dyn_json, str):
            try:
                dynamics = json.loads(dyn_json)
            except:
                dynamics = {}
        else:
            dynamics = dyn_json or {}

        if isinstance(hist_json, str):
            try:
                history = json.loads(hist_json)
            except:
                history = []
        else:
            history = hist_json or []

        if cr_json:
            if isinstance(cr_json, str):
                try:
                    cr_list = json.loads(cr_json)
                except:
                    cr_list = []
            else:
                cr_list = cr_json
        else:
            cr_list = []

        if rit_json:
            if isinstance(rit_json, str):
                try:
                    rit_list = json.loads(rit_json)
                except:
                    rit_list = []
            else:
                rit_list = rit_json
        else:
            rit_list = []

        e1_name = get_entity_name(conn, entity1_type, entity1_id, user_id, conversation_id)
        e2_name = get_entity_name(conn, entity2_type, entity2_id, user_id, conversation_id)

        # Build dynamic descriptions
        dynamic_descriptions = []
        for dnm, lvl in dynamics.items():
            desc = get_dynamic_description(dnm, lvl)
            dynamic_descriptions.append(f"{dnm.capitalize()} {lvl}/100 => {desc}")

        summary = {
            "entity1_name": e1_name,
            "entity2_name": e2_name,
            "primary_type": link_type,
            "primary_level": link_level,
            "dynamics": dynamics,
            "dynamic_descriptions": dynamic_descriptions,
            "history": history[-5:],  # last 5 events
            "experienced_crossroads": cr_list,
            "experienced_rituals": rit_list,
        }
        return summary
    except Exception as e:
        logging.error(f"Error getting relationship summary: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 6) Example Enhanced Classes (Optional)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RelationshipDimension:
    """
    A specific dimension/aspect of a relationship between entities
    """
    def __init__(self, name: str, description: str, min_value: int = -100, max_value: int = 100):
        self.name = name
        self.description = description
        self.min_value = min_value
        self.max_value = max_value

    def get_level_description(self, value: int) -> str:
        rng = self.max_value - self.min_value
        pct = (value - self.min_value) / float(rng)
        if pct < 0.2:
            return f"Very Low {self.name}"
        elif pct < 0.4:
            return f"Low {self.name}"
        elif pct < 0.6:
            return f"Moderate {self.name}"
        elif pct < 0.8:
            return f"High {self.name}"
        return f"Very High {self.name}"


class EnhancedRelationshipManager:
    """
    Manages more complex relationships: multiple dimensions,
    potential transitions, tension, etc.
    """
    RELATIONSHIP_DIMENSIONS = {
        "control": RelationshipDimension("Control", "How much one exerts control", 0, 100),
        "trust": RelationshipDimension("Trust", "Level of trust between entities", -100, 100),
        "intimacy": RelationshipDimension("Intimacy", "Emotional/physical closeness", 0, 100),
        "respect": RelationshipDimension("Respect", "Respect between entities", -100, 100),
        "dependency": RelationshipDimension("Dependency", "How dependent one is", 0, 100),
        "fear": RelationshipDimension("Fear", "How much one fears the other", 0, 100),
        "tension": RelationshipDimension("Tension", "Current tension level", 0, 100),
        "obsession": RelationshipDimension("Obsession", "How obsessed one is", 0, 100),
        "resentment": RelationshipDimension("Resentment", "Resentment level", 0, 100),
        "manipulation": RelationshipDimension("Manipulation", "Degree of manipulation", 0, 100),
    }

    RELATIONSHIP_TYPES = {
        "dominant": {"primary_dimensions": ["control", "fear", "tension", "manipulation"]},
        "submission": {"primary_dimensions": ["control", "dependency", "fear", "respect"]},
        "rivalry": {"primary_dimensions": ["tension", "resentment", "respect", "manipulation"]},
        "alliance": {"primary_dimensions": ["trust", "respect", "manipulation"]},
        "intimate": {"primary_dimensions": ["intimacy", "dependency", "obsession", "manipulation"]},
        "familial": {"primary_dimensions": ["control", "dependency", "respect", "resentment"]},
        "mentor": {"primary_dimensions": ["control", "respect", "dependency"]},
        "enmity": {"primary_dimensions": ["fear", "resentment", "tension"]},
        "neutral": {"primary_dimensions": []},
    }

    RELATIONSHIP_TRANSITIONS = [
        {
            "name": "Submission Acceptance",
            "from_type": "dominant",
            "to_type": "submission",
            "required_dimensions": {"dependency": 70, "fear": 50, "respect": 40},
        },
        {
            "name": "Rivalry to Alliance",
            "from_type": "rivalry",
            "to_type": "alliance",
            "required_dimensions": {"trust": 50, "respect": 60, "tension": -40},
        },
        {
            "name": "Alliance to Betrayal",
            "from_type": "alliance",
            "to_type": "enmity",
            "required_dimensions": {"trust": -60, "resentment": 70, "manipulation": 80},
        },
        {
            "name": "Mentor to Intimate",
            "from_type": "mentor",
            "to_type": "intimate",
            "required_dimensions": {"intimacy": 70, "obsession": 60, "dependency": 50},
        },
        {
            "name": "Enmity to Submission",
            "from_type": "enmity",
            "to_type": "submission",
            "required_dimensions": {"fear": 80, "control": 70, "dependency": 60},
        },
    ]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 7) NPCGroup & MultiNPCInteractionManager
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class NPCGroup:
    """
    Represents a group of NPCs with shared dynamics.
    """
    def __init__(self, name: str, description: str, members=None, dynamics=None):
        self.name = name
        self.description = description
        self.members = members or []   # list of dicts: [{npc_id, npc_name, role, etc.}]
        self.dynamics = dynamics or {} # e.g. {"hierarchy": 50, "cohesion": 30, ...}
        self.creation_date = datetime.now().isoformat()
        self.last_activity = None
        self.shared_history = []       # record of group events

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "members": self.members,
            "dynamics": self.dynamics,
            "creation_date": self.creation_date,
            "last_activity": self.last_activity,
            "shared_history": self.shared_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        grp = cls(data["name"], data["description"], data.get("members", []), data.get("dynamics", {}))
        grp.creation_date = data.get("creation_date", datetime.now().isoformat())
        grp.last_activity = data.get("last_activity", None)
        grp.shared_history = data.get("shared_history", [])
        return grp


class MultiNPCInteractionManager:
    """
    Manages interactions between multiple NPCs, including group dynamics, factional behavior,
    and coordinated activities (e.g., multi-NPC scenes).
    """

    GROUP_DYNAMICS = {
        "hierarchy": {
            "description": "Formalized power structure in the group",
            "effects": "Chain of command and override authority"
        },
        "cohesion": {
            "description": "How unified the group is in goals/behavior",
            "effects": "Synergy vs. friction; influences group stability"
        },
        "secrecy": {
            "description": "How much the group hides from outsiders",
            "effects": "Information control and shared secrecy"
        },
        "territoriality": {
            "description": "Protectiveness over members/resources",
            "effects": "Reactions to perceived threats or intrusion"
        },
        "exclusivity": {
            "description": "Difficulty to join / acceptance threshold",
            "effects": "Initiation tests, membership gating"
        },
    }

    INTERACTION_STYLES = {
        "coordinated": {
            "description": "NPCs act in a coordinated, deliberate manner",
            "requirements": {"cohesion": 70},
            "dialogue_style": "NPCs build on each other's statements smoothly."
        },
        "hierarchical": {
            "description": "NPCs follow a clear status hierarchy",
            "requirements": {"hierarchy": 70},
            "dialogue_style": "Lower-status NPCs defer to higher-status NPCs."
        },
        "competitive": {
            "description": "NPCs compete for dominance or attention",
            "requirements": {"cohesion": -40, "hierarchy": -30},  # example negative threshold
            "dialogue_style": "NPCs interrupt or attempt to outdo each other."
        },
        "consensus": {
            "description": "NPCs seek group agreement before acting",
            "requirements": {"cohesion": 60, "hierarchy": -40},
            "dialogue_style": "NPCs exchange opinions politely, aiming for unity."
        },
        "protective": {
            "description": "NPCs protect and defend one target or idea",
            "requirements": {"territoriality": 70},
            "dialogue_style": "NPCs focus on ensuring safety or enforcing boundaries."
        },
        "exclusionary": {
            "description": "NPCs deliberately exclude someone (the player or another NPC)",
            "requirements": {"exclusivity": 70},
            "dialogue_style": "NPCs speak in code or inside references, ignoring outsiders."
        },
        "manipulative": {
            "description": "NPCs coordinate to manipulate a target",
            "requirements": {"cohesion": 60, "secrecy": 70},
            "dialogue_style": "NPCs set conversational traps, do good-cop/bad-cop routines."
        },
    }


    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id

    def create_npc_group(
        self, name: str, description: str, member_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Create a new NPC group in the DB, set up relationships among members if needed.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            # Validate each NPC
            members_data = []
            for npc_id in member_ids:
                cursor.execute(
                    """
                    SELECT npc_id, npc_name, dominance, cruelty
                    FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                    """,
                    (self.user_id, self.conversation_id, npc_id)
                )
                row = cursor.fetchone()
                if not row:
                    return {"error": f"NPC with ID {npc_id} not found in conversation {self.conversation_id}."}
                members_data.append({
                    "npc_id": row[0],
                    "npc_name": row[1],
                    "dominance": row[2],
                    "cruelty": row[3],
                    "role": "member",
                    "status": "active",
                    "joined_date": datetime.now().isoformat()
                })

            # Generate initial group dynamics
            dynamics = {}
            for dyn_key in self.GROUP_DYNAMICS.keys():
                dynamics[dyn_key] = random.randint(20, 80)

            # Possibly assign a leader if hierarchy is high
            if len(members_data) > 1 and dynamics["hierarchy"] > 60:
                # sort by dominance
                sorted_mem = sorted(members_data, key=lambda x: x["dominance"], reverse=True)
                # top one is leader, next few can be lieutenants
                sorted_mem[0]["role"] = "leader"
                # you can pick additional roles or none
                # re-assign
                members_data = sorted_mem

            # Create NPCGroup object
            group_obj = NPCGroup(name, description, members_data, dynamics)

            # Insert into table
            cursor.execute(
                """
                INSERT INTO NPCGroups (user_id, conversation_id, group_name, group_data)
                VALUES (%s, %s, %s, %s)
                RETURNING group_id
                """,
                (self.user_id, self.conversation_id, name, json.dumps(group_obj.to_dict()))
            )
            group_id = cursor.fetchone()[0]

            conn.commit()
            return {"success": True, "group_id": group_id, "message": f"Group '{name}' created successfully."}
        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating group: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()

    def get_npc_group(
        self, group_id: Optional[int] = None, group_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve an NPC group by group_id or group_name. 
        Returns the group data or an error if not found.
        """
        if not group_id and not group_name:
            return {"error": "Must provide either group_id or group_name."}

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            if group_id:
                cursor.execute(
                    """
                    SELECT group_id, group_name, group_data
                    FROM NPCGroups
                    WHERE user_id=%s AND conversation_id=%s AND group_id=%s
                    """,
                    (self.user_id, self.conversation_id, group_id)
                )
            else:
                cursor.execute(
                    """
                    SELECT group_id, group_name, group_data
                    FROM NPCGroups
                    WHERE user_id=%s AND conversation_id=%s AND group_name=%s
                    """,
                    (self.user_id, self.conversation_id, group_name)
                )
            row = cursor.fetchone()
            if not row:
                return {"error": f"Group not found."}

            real_group_id, real_group_name, group_data_json = row
            if isinstance(group_data_json, str):
                group_dict = json.loads(group_data_json)
            else:
                group_dict = group_data_json
            # Reconstruct NPCGroup object
            group_obj = NPCGroup.from_dict(group_dict)
            return {
                "success": True,
                "group_id": real_group_id,
                "group_name": real_group_name,
                "group_data": group_obj.to_dict(),
            }
        except Exception as e:
            logger.error(f"Error retrieving group: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()

    def update_group_dynamics(
        self, group_id: int, changes: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Apply increments/decrements to the group's dynamics. 
        Example: changes = {"cohesion": +10, "hierarchy": -5}
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            # fetch existing
            cursor.execute(
                """
                SELECT group_data
                FROM NPCGroups
                WHERE user_id=%s AND conversation_id=%s AND group_id=%s
                """,
                (self.user_id, self.conversation_id, group_id)
            )
            row = cursor.fetchone()
            if not row:
                return {"error": "Group not found."}
            group_data_json = row[0]

            if isinstance(group_data_json, str):
                group_dict = json.loads(group_data_json)
            else:
                group_dict = group_data_json

            # Update
            group_obj = NPCGroup.from_dict(group_dict)
            for dyn_key, delta in changes.items():
                if dyn_key in group_obj.dynamics:
                    current = group_obj.dynamics[dyn_key]
                    new_val = max(0, min(100, current + delta))
                    group_obj.dynamics[dyn_key] = new_val

            # record a quick log
            group_obj.shared_history.append({
                "timestamp": datetime.now().isoformat(),
                "type": "dynamics_update",
                "details": changes
            })
            group_obj.last_activity = datetime.now().isoformat()

            # store back
            updated_json = json.dumps(group_obj.to_dict())
            cursor.execute(
                """
                UPDATE NPCGroups
                SET group_data = %s
                WHERE user_id=%s AND conversation_id=%s AND group_id=%s
                """,
                (updated_json, self.user_id, self.conversation_id, group_id)
            )
            conn.commit()
            return {"success": True, "message": "Group dynamics updated."}
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating group dynamics: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2) Determining Interaction Styles & Producing Scenes
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def determine_interaction_style(self, group_obj: NPCGroup) -> str:
        """
        Based on the group's dynamics, pick an appropriate style 
        from INTERACTION_STYLES if requirements are met.
        If multiple styles qualify, pick the 'highest priority' or random among them.
        If none match, default to "neutral."
        """
        candidates = []
        for style_name, style_def in self.INTERACTION_STYLES.items():
            reqs = style_def["requirements"]
            meets_all = True
            for dyn_key, threshold in reqs.items():
                current_val = group_obj.dynamics.get(dyn_key, 0)
                # If threshold is positive, require current_val >= threshold
                # If threshold is negative, require current_val <= abs(threshold)
                if threshold >= 0 and current_val < threshold:
                    meets_all = False
                    break
                elif threshold < 0 and current_val > abs(threshold):
                    meets_all = False
                    break
            if meets_all:
                candidates.append(style_name)

        if not candidates:
            return "neutral"
        # pick first or random
        return random.choice(candidates)

    def produce_multi_npc_scene(
        self,
        group_id: int,
        topic: str = "General conversation",
        extra_context: str = ""
    ) -> Dict[str, Any]:
        """
        Creates a sample scene or conversation snippet among the NPCs in the group, 
        reflecting the chosen interaction style.
        """
        group_info = self.get_npc_group(group_id=group_id)
        if "error" in group_info:
            return {"error": group_info["error"]}

        group_data = group_info["group_data"]
        group_obj = NPCGroup.from_dict(group_data)

        # Determine style
        style = self.determine_interaction_style(group_obj)
        style_def = self.INTERACTION_STYLES.get(style, {})
        dialogue_style_desc = style_def.get("dialogue_style", "Plain interaction.")
        desc = style_def.get("description", "")

        # Build a short demonstration of how they'd talk
        # We'll just do a naive approach: list each NPC's 'line' in order
        random.shuffle(group_obj.members)
        lines = []
        for mem in group_obj.members[:3]:  # limit to 3 for brevity
            name = mem["npc_name"]
            role = mem.get("role", "member")
            line = f"{name} ({role}): " \
                   f"[Responding to {topic} with style '{style}']"
            lines.append(line)

        # optionally store a small record in the group's shared history
        group_obj.shared_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "scene_example",
            "style": style,
            "topic": topic,
            "lines": lines
        })
        group_obj.last_activity = datetime.now().isoformat()

        # persist changes
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            updated_json = json.dumps(group_obj.to_dict())
            cursor.execute(
                """
                UPDATE NPCGroups
                SET group_data = %s
                WHERE user_id=%s AND conversation_id=%s AND group_id=%s
                """,
                (updated_json, self.user_id, self.conversation_id, group_id)
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing scene in group: {e}")
        finally:
            cursor.close()
            conn.close()

        scene_text = "\n".join(lines)
        return {
            "success": True,
            "interaction_style": style,
            "style_description": desc,
            "dialogue_style": dialogue_style_desc,
            "scene_preview": scene_text
        }

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3) Additional Utility Methods
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def list_all_groups(self) -> Dict[str, Any]:
        """
        Return a list of all NPC groups for the current user/conversation.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT group_id, group_name, group_data
                FROM NPCGroups
                WHERE user_id=%s AND conversation_id=%s
                """,
                (self.user_id, self.conversation_id)
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                g_id, g_name, g_data = row
                if isinstance(g_data, str):
                    g_dict = json.loads(g_data)
                else:
                    g_dict = g_data
                results.append({
                    "group_id": g_id,
                    "group_name": g_name,
                    "data": g_dict
                })
            return {"groups": results, "count": len(results)}
        except Exception as e:
            logger.error(f"Error listing groups: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()

    def delete_group(self, group_id: int) -> Dict[str, Any]:
        """
        Deletes an NPC group from the DB.
        NOTE: Potentially also handle cleanup of relationships or references.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                DELETE FROM NPCGroups
                WHERE user_id=%s AND conversation_id=%s AND group_id=%s
                """,
                (self.user_id, self.conversation_id, group_id)
            )
            if cursor.rowcount < 1:
                return {"error": "No group found or deletion failed."}
            conn.commit()
            return {"success": True, "message": f"Group {group_id} deleted."}
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting group: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()

    async def update_npc_group_dynamics(
        self,
        user_id: int,
        conversation_id: int,
        group_id: int,
        dynamics_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update NPC group dynamics and relationships within a group.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            group_id: Group ID
            dynamics_data: Dictionary containing dynamics updates
            
        Returns:
            Dictionary of updates applied
        """
        try:
            conn = await asyncpg.connect(dsn=get_db_connection())
            try:
                updates_applied = []
                
                # Get group members
                group = await self.get_npc_group(group_id=group_id)
                
                if not group or "members" not in group:
                    return {"success": False, "error": "Group not found"}
                
                # Process each dynamic update
                for dynamic, value in dynamics_data.items():
                    if dynamic not in self.GROUP_DYNAMICS:
                        continue
                    
                    # Update the dynamic in the database
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE NPCGroups
                        SET dynamics = COALESCE(dynamics, '{}'::jsonb) || 
                                     jsonb_build_object(%s, %s)
                        WHERE group_id = %s AND user_id = %s AND conversation_id = %s
                    """, (dynamic, value, group_id, user_id, conversation_id))
                    
                    # Update relationships between group members based on dynamics
                    for member1 in group["members"]:
                        for member2 in group["members"]:
                            if member1["npc_id"] >= member2["npc_id"]:
                                continue
                            
                            # Get current relationship between members
                            member_link = await get_social_link(
                                user_id,
                                conversation_id,
                                "npc",
                                member1["npc_id"],
                                "npc",
                                member2["npc_id"]
                            )
                            
                            if not member_link:
                                continue
                            
                            # Calculate relationship changes based on dynamics
                            level_change = 0
                            new_type = None
                            
                            if dynamic == "hierarchy":
                                # Hierarchy changes affect respect and control
                                if value > 50:
                                    level_change = 5
                                else:
                                    level_change = -5
                            elif dynamic == "cohesion":
                                # Cohesion affects trust and intimacy
                                if value > 50:
                                    level_change = 3
                                else:
                                    level_change = -3
                            elif dynamic == "secrecy":
                                # Secrecy affects trust and manipulation
                                if value > 50:
                                    level_change = 2
                                else:
                                    level_change = -2
                            
                            # Update the relationship
                            update_result = await update_link_type_and_level(
                                user_id,
                                conversation_id,
                                member_link["link_id"],
                                new_type,
                                level_change
                            )
                            
                            if update_result:
                                updates_applied.append({
                                    "member1_id": member1["npc_id"],
                                    "member2_id": member2["npc_id"],
                                    "dynamic": dynamic,
                                    "changes": update_result
                                })
                                
                                # Add event to relationship history
                                event_text = f"Group dynamic '{dynamic}' changed to {value}"
                                await add_link_event(
                                    user_id,
                                    conversation_id,
                                    member_link["link_id"],
                                    event_text
                                )
                
                conn.commit()
                return {
                    "success": True,
                    "updates_applied": updates_applied
                }
                
            finally:
                await conn.close()
        except Exception as e:
            logging.error(f"Error updating NPC group dynamics: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 8) Tools (function_tool) that the agent can call
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@function_tool
async def get_social_link_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int
) -> dict:
    """
    Get an existing social link's details if it exists.
    """
    link = get_social_link(
        user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id
    )
    if link is None:
        return {"error": "No link found"}
    return link


@function_tool
async def create_social_link_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int,
    link_type: str = "neutral",
    link_level: int = 0
) -> dict:
    """
    Create a new social link, or return the existing link_id if it already exists.
    """
    link_id = create_social_link(
        user_id, conversation_id, entity1_type, entity1_id,
        entity2_type, entity2_id, link_type, link_level
    )
    return {"link_id": link_id, "message": "Link created or fetched."}


@function_tool
async def update_link_type_and_level_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    link_id: int,
    new_type: str = None,
    level_change: int = 0
) -> dict:
    """
    Update an existing link's type and/or level. 
    Returns updated info or an error if not found.
    """
    result = update_link_type_and_level(user_id, conversation_id, link_id, new_type, level_change)
    if result is None:
        return {"error": "Link not found or update failed"}
    return result


@function_tool
async def add_link_event_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    link_id: int,
    event_text: str
) -> dict:
    """
    Append an event string to a link's link_history.
    """
    add_link_event(user_id, conversation_id, link_id, event_text)
    return {"success": True, "message": "Event added to link_history"}


@function_tool
async def check_for_crossroads_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int
) -> dict:
    """
    Check if there's a relationship crossroads event triggered. 
    Returns the first triggered crossroads or None.
    """
    result = check_for_relationship_crossroads(user_id, conversation_id)
    if not result:
        return {"message": "No crossroads triggered"}
    return result


@function_tool
async def apply_crossroads_choice_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    link_id: int,
    crossroads_name: str,
    choice_index: int
) -> dict:
    """
    Apply a chosen effect from a triggered crossroads.
    """
    return apply_crossroads_choice(
        user_id, conversation_id, link_id, crossroads_name, choice_index
    )


@function_tool
async def check_for_ritual_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int
) -> dict:
    """
    Check if there's a relationship ritual event triggered.
    Returns the first triggered ritual or None.
    """
    result = check_for_relationship_ritual(user_id, conversation_id)
    if not result:
        return {"message": "No ritual triggered"}
    return result


@function_tool
async def get_relationship_summary_tool(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int
) -> dict:
    """
    Get a summary of the relationship (type, level, last 5 history entries, etc.).
    """
    summary = get_relationship_summary(
        user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id
    )
    if not summary:
        return {"error": "No relationship found"}
    return summary


@function_tool
async def update_relationships_from_conflict(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    conflict_id: int,
    resolution_data: Dict[str, Any]
) -> dict:
    """
    Update relationships based on conflict resolution outcomes.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        conflict_id: Conflict ID
        resolution_data: Resolution data including outcomes and stakeholder impacts
        
    Returns:
        Dictionary of relationship updates applied
    """
    try:
        conn = await asyncpg.connect(dsn=get_db_connection())
        try:
            updates_applied = []
            
            # Get all stakeholders involved in the conflict
            stakeholders = await get_conflict_stakeholders(
                RunContextWrapper({"user_id": user_id, "conversation_id": conversation_id}),
                conflict_id
            )
            
            # Process each stakeholder's relationship changes
            for stakeholder in stakeholders:
                npc_id = stakeholder["npc_id"]
                
                # Get current relationship with player
                player_link = await get_social_link(
                    user_id,
                    conversation_id,
                    "player",
                    0,  # Player ID
                    "npc",
                    npc_id
                )
                
                if not player_link:
                    continue
                
                # Calculate relationship changes based on resolution
                level_change = 0
                new_type = None
                
                # Apply changes based on resolution outcome
                if resolution_data.get("success", False):
                    if stakeholder.get("faction_position") == resolution_data.get("winning_faction"):
                        level_change = 10  # Positive change for aligned stakeholders
                    else:
                        level_change = -5  # Slight negative for opposing stakeholders
                else:
                    if stakeholder.get("faction_position") == resolution_data.get("winning_faction"):
                        level_change = -10  # Negative change for failed aligned stakeholders
                    else:
                        level_change = 5  # Slight positive for opposing stakeholders
                
                # Update the relationship
                update_result = await update_link_type_and_level(
                    user_id,
                    conversation_id,
                    player_link["link_id"],
                    new_type,
                    level_change
                )
                
                if update_result:
                    updates_applied.append({
                        "npc_id": npc_id,
                        "changes": update_result
                    })
                    
                    # Add event to relationship history
                    event_text = f"Relationship changed due to conflict resolution: {resolution_data.get('outcome', 'unknown')}"
                    await add_link_event(
                        user_id,
                        conversation_id,
                        player_link["link_id"],
                        event_text
                    )
            
            return {
                "success": True,
                "updates_applied": updates_applied
            }
            
        finally:
            await conn.close()
    except Exception as e:
        logging.error(f"Error updating relationships from conflict: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@function_tool
async def update_relationship_context(
    ctx: RunContextWrapper,
    user_id: int,
    conversation_id: int,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int,
    context_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update relationship context with additional information.
    
    Args:
        ctx: Run context wrapper
        user_id: User ID
        conversation_id: Conversation ID
        entity1_type: Type of first entity
        entity1_id: ID of first entity
        entity2_type: Type of second entity
        entity2_id: ID of second entity
        context_data: Additional context data to store
        
    Returns:
        Dictionary of updates applied
    """
    try:
        conn = await asyncpg.connect(dsn=get_db_connection())
        try:
            # Get current relationship
            relationship = await get_social_link(
                user_id,
                conversation_id,
                entity1_type,
                entity1_id,
                entity2_type,
                entity2_id
            )
            
            if not relationship:
                return {"success": False, "error": "Relationship not found"}
            
            # Update relationship context
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE SocialLinks
                SET context = COALESCE(context, '{}'::jsonb) || %s::jsonb
                WHERE link_id = %s AND user_id = %s AND conversation_id = %s
            """, (json.dumps(context_data), relationship["link_id"], user_id, conversation_id))
            
            # Add event to relationship history
            event_text = f"Relationship context updated: {json.dumps(context_data)}"
            await add_link_event(
                user_id,
                conversation_id,
                relationship["link_id"],
                event_text
            )
            
            conn.commit()
            return {
                "success": True,
                "link_id": relationship["link_id"],
                "context_updated": context_data
            }
            
        finally:
            await conn.close()
    except Exception as e:
        logging.error(f"Error updating relationship context: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 9) The Agent: "SocialLinksAgent"
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SocialLinksAgent = Agent(
    name="SocialLinksAgent",
    instructions=(
        "You are a specialized 'Social Links Agent' for a persona-like system. "
        "You have function tools that let you manage relationships:\n"
        " - get_social_link_tool(...)\n"
        " - create_social_link_tool(...)\n"
        " - update_link_type_and_level_tool(...)\n"
        " - add_link_event_tool(...)\n"
        " - check_for_crossroads_tool(...)\n"
        " - apply_crossroads_choice_tool(...)\n"
        " - check_for_ritual_tool(...)\n"
        " - get_relationship_summary_tool(...)\n"
        " - update_relationships_from_conflict(...)\n\n"
        "Use these tools to retrieve or update relationship data, trigger or apply crossroads, or check for rituals. "
        "Return helpful final text or JSON summarizing your result."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),  # or "gpt-4o", "gpt-3.5-turbo", etc.
    model_settings=ModelSettings(temperature=0.5),
    tools=[
        get_social_link_tool,
        create_social_link_tool,
        update_link_type_and_level_tool,
        add_link_event_tool,
        check_for_crossroads_tool,
        apply_crossroads_choice_tool,
        check_for_ritual_tool,
        get_relationship_summary_tool,
        update_relationships_from_conflict
    ],
    output_type=None
)
