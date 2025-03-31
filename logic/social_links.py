# logic/social_links_agentic.py
"""
Comprehensive End-to-End Social Links System with an Agentic approach using OpenAI's Agents SDK.
Converted to use asyncpg and connection pooling.

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
import asyncio # Added for potential use
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

import asyncpg # Added import

# ~~~~~~~~~ Agents SDK imports ~~~~~~~~~
from agents import (
    Agent,
    ModelSettings,
    Runner,
    function_tool,
    RunContextWrapper,
    AsyncOpenAI,
    OpenAIResponsesModel
)
# Note: OpenAIResponsesModel imported twice, removed duplicate
# from agents.models.openai_responses import OpenAIResponsesModel

# ~~~~~~~~~ DB imports & any other placeholders ~~~~~~~~~
from db.connection import get_db_connection_context # Use context manager
# from db.connection import get_db_connection # REMOVED

# ~~~~~~~~~ Logging Configuration ~~~~~~~~~
# Define logger at module level for consistency
logger = logging.getLogger(__name__)
# Basic config if run standalone, otherwise rely on parent application config
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Simple Core CRUD for SocialLinks Table
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def get_social_link(
    user_id: int,
    conversation_id: int,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int
) -> Optional[Dict[str, Any]]:
    """
    Fetch an existing social link row using asyncpg.
    Returns a dict with link details or None if not found.
    """
    try:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                """
                SELECT link_id, link_type, link_level, link_history, dynamics, -- Added dynamics
                       experienced_crossroads, experienced_rituals          -- Added experienced
                FROM SocialLinks
                WHERE user_id = $1 AND conversation_id = $2
                  AND entity1_type = $3 AND entity1_id = $4
                  AND entity2_type = $5 AND entity2_id = $6
                """,
                user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id
            )
            if row:
                # asyncpg might auto-parse JSONB, handle potential strings/None
                history = row['link_history'] if isinstance(row['link_history'], list) else (json.loads(row['link_history']) if isinstance(row['link_history'], str) else [])
                dynamics = row['dynamics'] if isinstance(row['dynamics'], dict) else (json.loads(row['dynamics']) if isinstance(row['dynamics'], str) else {})
                crossroads = row['experienced_crossroads'] if isinstance(row['experienced_crossroads'], list) else (json.loads(row['experienced_crossroads']) if isinstance(row['experienced_crossroads'], str) else [])
                rituals = row['experienced_rituals'] if isinstance(row['experienced_rituals'], list) else (json.loads(row['experienced_rituals']) if isinstance(row['experienced_rituals'], str) else [])

                return {
                    "link_id": row['link_id'],
                    "link_type": row['link_type'],
                    "link_level": row['link_level'],
                    "link_history": history,
                    "dynamics": dynamics,
                    "experienced_crossroads": crossroads,
                    "experienced_rituals": rituals,
                }
            return None
    except (asyncpg.PostgresError, ConnectionError) as e:
        logger.error(f"Error getting social link: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting social link: {e}", exc_info=True)
        return None


async def create_social_link(
    user_id: int,
    conversation_id: int,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int,
    link_type: str = "neutral",
    link_level: int = 0,
    initial_dynamics: Optional[Dict] = None # Allow passing initial dynamics
) -> Optional[int]:
    """
    Create a new SocialLinks row using asyncpg, handling conflicts.
    Initializes link_history, dynamics, etc.
    Returns the link_id (new or existing).
    """
    initial_dynamics_json = json.dumps(initial_dynamics or {})
    initial_history_json = '[]' # Start with empty history
    initial_experienced_json = '[]' # Start with empty experienced

    try:
        async with get_db_connection_context() as conn:
            # Use INSERT ... ON CONFLICT ... RETURNING link_id for atomicity
            link_id = await conn.fetchval(
                """
                INSERT INTO SocialLinks (
                    user_id, conversation_id,
                    entity1_type, entity1_id,
                    entity2_type, entity2_id,
                    link_type, link_level,
                    link_history, dynamics, experienced_crossroads, experienced_rituals
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11::jsonb, $12::jsonb)
                ON CONFLICT (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id)
                DO UPDATE SET link_id = EXCLUDED.link_id -- No actual update, just to get RETURNING
                RETURNING link_id;
                """,
                user_id, conversation_id,
                entity1_type, entity1_id,
                entity2_type, entity2_id,
                link_type, link_level,
                initial_history_json, initial_dynamics_json,
                initial_experienced_json, initial_experienced_json
            )
            # If ON CONFLICT occurred, the above might return NULL or the existing ID
            # depending on PG version and exact conflict target. A safer way is separate SELECT.
            if link_id is None:
                 link_id = await conn.fetchval(
                     """
                     SELECT link_id FROM SocialLinks
                     WHERE user_id = $1 AND conversation_id = $2
                     AND entity1_type = $3 AND entity1_id = $4
                     AND entity2_type = $5 AND entity2_id = $6
                     """,
                     user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id
                 )
            return link_id
    except (asyncpg.PostgresError, ConnectionError) as e:
        logger.error(f"Error creating social link: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating social link: {e}", exc_info=True)
        return None


async def update_link_type_and_level(
    user_id: int,
    conversation_id: int,
    link_id: int,
    new_type: Optional[str] = None,
    level_change: int = 0
) -> Optional[Dict[str, Any]]:
    """
    Adjust an existing link's type or level using asyncpg.
    Returns updated info or None if not found.
    """
    try:
        async with get_db_connection_context() as conn:
            # Use RETURNING to get the final values in one step
            updated_row = await conn.fetchrow(
                """
                UPDATE SocialLinks
                SET link_type = COALESCE($1, link_type),
                    link_level = link_level + $2
                WHERE link_id = $3 AND user_id = $4 AND conversation_id = $5
                RETURNING link_id, link_type, link_level;
                """,
                new_type, level_change, link_id, user_id, conversation_id
            )

            if updated_row:
                return {
                    "link_id": updated_row['link_id'],
                    "new_type": updated_row['link_type'],
                    "new_level": updated_row['link_level'],
                }
            else:
                logger.warning(f"No link found to update for link_id={link_id}, user={user_id}, conv={conversation_id}")
                return None
    except (asyncpg.PostgresError, ConnectionError) as e:
        logger.error(f"Error updating link type/level: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error updating link type/level: {e}", exc_info=True)
        return None


async def add_link_event(
    user_id: int,
    conversation_id: int,
    link_id: int,
    event_text: str
) -> bool:
    """
    Append an event string to link_history using asyncpg. Returns True on success.
    """
    try:
        async with get_db_connection_context() as conn:
            # Append the new event as a JSONB element
            result = await conn.execute(
                """
                UPDATE SocialLinks
                SET link_history = COALESCE(link_history, '[]'::jsonb) || $1::jsonb
                WHERE link_id = $2 AND user_id = $3 AND conversation_id = $4
                """,
                json.dumps(event_text), # Ensure it's a valid JSON string element
                link_id, user_id, conversation_id
            )
            # Check if any row was updated
            if result == "UPDATE 1":
                logger.info(f"Appended event to link_history for link_id={link_id}")
                return True
            else:
                logger.warning(f"No link found to add event for link_id={link_id}, user={user_id}, conv={conversation_id}")
                return False
    except (asyncpg.PostgresError, ConnectionError) as e:
        logger.error(f"Error adding link event: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error adding link event: {e}", exc_info=True)
        return False


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
    max_level = -float('inf') # Handle potential negative dynamics correctly
    for dname, lvl in dynamics.items():
        # Consider absolute value or just highest value depending on desired logic
        if lvl > max_level:
            max_level = lvl
            primary_dynamic = dname
    # If all dynamics are <= 0, might still return neutral or the least negative one
    if max_level <= 0 and dynamics:
         # Find the max (least negative) among potentially all negative values
         primary_dynamic = max(dynamics, key=dynamics.get)

    return primary_dynamic


def get_dynamic_description(dynamic_name: str, level: int) -> str:
    """
    Get the appropriate textual description for a dynamic at a specific level.
    """
    for dyn in RELATIONSHIP_DYNAMICS:
        if dyn["name"] == dynamic_name:
            # Find the highest level definition that the current level is less than or equal to
            matched_level = None
            for level_info in sorted(dyn["levels"], key=lambda x: x["level"]):
                if level <= level_info["level"]:
                    matched_level = level_info
                    break
            # If level is higher than all defined levels, use the highest definition
            if not matched_level:
                matched_level = dyn["levels"][-1] # Assumes levels are sorted or max level is last
            return f"{matched_level['name']}: {matched_level['description']}"
    return "Unknown dynamic"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4) Crossroad Checking + Ritual Checking
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def check_for_relationship_crossroads(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    Async check if any NPC relationship triggers a Crossroads event.
    """
    try:
        async with get_db_connection_context() as conn:
            # Gather all player-related links
            links = await conn.fetch(
                """
                SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id,
                       dynamics, experienced_crossroads
                FROM SocialLinks
                WHERE user_id = $1 AND conversation_id = $2
                AND (
                    (entity1_type='player' AND entity1_id = $1) -- Assuming player ID is user ID here
                    OR (entity2_type='player' AND entity2_id = $1)
                )
                """,
                user_id, conversation_id
            )

            for link_record in links:
                link_id = link_record['link_id']
                e1t = link_record['entity1_type']
                e1id = link_record['entity1_id']
                e2t = link_record['entity2_type']
                e2id = link_record['entity2_id']
                dynamics_data = link_record['dynamics'] # Already parsed by asyncpg?
                crossroads_data = link_record['experienced_crossroads']

                dynamics = dynamics_data if isinstance(dynamics_data, dict) else (json.loads(dynamics_data) if isinstance(dynamics_data, str) else {})
                experienced = crossroads_data if isinstance(crossroads_data, list) else (json.loads(crossroads_data) if isinstance(crossroads_data, str) else [])

                # Determine NPC side
                npc_id = None
                if e1t == "npc" and e2t == "player":
                    npc_id = e1id
                elif e2t == "npc" and e1t == "player":
                    npc_id = e2id

                if npc_id is None:
                    continue

                # Get NPC name (can use the same connection)
                npc_name = await conn.fetchval(
                    "SELECT npc_name FROM NPCStats WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3",
                    user_id, conversation_id, npc_id
                )
                if not npc_name:
                    logger.warning(f"NPC {npc_id} not found for crossroads check.")
                    continue

                # Check each Crossroads definition
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
            return None # No crossroads triggered

    except (asyncpg.PostgresError, ConnectionError) as e:
        logger.error(f"Error checking for relationship crossroads: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error checking crossroads: {e}", exc_info=True)
        return None


async def apply_crossroads_choice(
    user_id: int,
    conversation_id: int,
    link_id: int,
    crossroads_name: str,
    choice_index: int
) -> Dict[str, Any]:
    """
    Async apply the chosen effect from a Crossroads event.
    Uses a transaction for atomicity.
    """
    cr_def = next((c for c in RELATIONSHIP_CROSSROADS if c["name"] == crossroads_name), None)
    if not cr_def:
        return {"error": f"Crossroads '{crossroads_name}' definition not found"}
    if not 0 <= choice_index < len(cr_def["choices"]):
        return {"error": "Invalid choice index"}
    choice = cr_def["choices"][choice_index]

    try:
        async with get_db_connection_context() as conn:
            async with conn.transaction(): # Ensure all updates succeed or fail together
                # Get link details
                row = await conn.fetchrow(
                    """
                    SELECT entity1_type, entity1_id, entity2_type, entity2_id, dynamics, experienced_crossroads
                    FROM SocialLinks
                    WHERE link_id = $1 AND user_id = $2 AND conversation_id = $3
                    FOR UPDATE -- Lock the row for the transaction
                    """,
                    link_id, user_id, conversation_id
                )
                if not row:
                    # Raise error to trigger transaction rollback
                    raise ValueError("Social link not found during apply_crossroads_choice")

                e1t, e1id, e2t, e2id, dyn_data, crossroads_data = row['entity1_type'], row['entity1_id'], row['entity2_type'], row['entity2_id'], row['dynamics'], row['experienced_crossroads']

                # Identify NPC
                npc_id = e1id if e1t == "npc" else (e2id if e2t == "npc" else None)
                if npc_id is None:
                     raise ValueError("No NPC found in relationship for crossroads")

                # Get NPC name
                npc_name = await conn.fetchval(
                    "SELECT npc_name FROM NPCStats WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3",
                    user_id, conversation_id, npc_id
                )
                if not npc_name:
                     raise ValueError(f"NPC {npc_id} not found for crossroads application")

                # Parse dynamics and experienced lists
                dynamics = dyn_data if isinstance(dyn_data, dict) else (json.loads(dyn_data) if isinstance(dyn_data, str) else {})
                experienced = crossroads_data if isinstance(crossroads_data, list) else (json.loads(crossroads_data) if isinstance(crossroads_data, str) else [])

                if crossroads_name in experienced:
                    # Should ideally not happen if check_for_crossroads is used correctly, but handle defensively
                    logger.warning(f"Crossroads '{crossroads_name}' already experienced for link {link_id}. Applying effects again.")
                    # Or: return {"error": "Crossroads already experienced"}

                # --- Apply Effects ---
                player_stat_updates = {}
                for dynamic_name, delta in choice["effects"].items():
                    if dynamic_name.startswith("player_"):
                        player_stat = dynamic_name[7:]
                        player_stat_updates[player_stat] = delta
                    else:
                        current_val = dynamics.get(dynamic_name, 0)
                        new_val = max(0, min(100, current_val + delta)) # Clamp between 0-100 (adjust if needed)
                        dynamics[dynamic_name] = new_val

                # Update player stats if any
                if player_stat_updates:
                    set_clauses = []
                    params = []
                    param_idx = 1
                    for stat, delta in player_stat_updates.items():
                         # IMPORTANT: Ensure 'stat' is a valid column name to prevent SQL injection
                         # Use a whitelist or validation if necessary. Assuming fixed stats here.
                         valid_player_stats = ["corruption", "confidence", "willpower", "obedience", "dependency", "lust", "mental_resilience", "physical_endurance"]
                         if stat in valid_player_stats:
                              set_clauses.append(f"{stat} = GREATEST(0, LEAST(100, {stat} + ${param_idx}))") # Clamp 0-100
                              params.append(delta)
                              param_idx += 1
                         else:
                              logger.error(f"Invalid player stat '{stat}' in crossroads effect. Skipping.")

                    if set_clauses:
                         params.extend([user_id, conversation_id])
                         player_update_sql = f"""
                            UPDATE PlayerStats
                            SET {', '.join(set_clauses)}
                            WHERE user_id = ${param_idx} AND conversation_id = ${param_idx+1} AND player_name = 'Chase' -- Adjust player name if needed
                         """
                         await conn.execute(player_update_sql, *params)

                # Mark crossroads as experienced
                if crossroads_name not in experienced: # Avoid duplicates if re-applying
                    experienced.append(crossroads_name)

                # Recompute primary link type/level based on new dynamics
                primary_type = get_primary_dynamic(dynamics)
                primary_level = dynamics.get(primary_type, 0)

                # Update the SocialLinks row
                await conn.execute(
                    """
                    UPDATE SocialLinks
                    SET dynamics = $1,
                        experienced_crossroads = $2,
                        link_type = $3,
                        link_level = $4
                    WHERE link_id = $5
                    """,
                    json.dumps(dynamics), json.dumps(experienced), primary_type, primary_level, link_id
                )

                # Add event to link history
                event_text = (
                    f"Crossroads '{crossroads_name}' chosen: {choice['text']}. "
                    f"Outcome: {choice['outcome'].format(npc_name=npc_name)}"
                )
                # Use the already awaited add_link_event function (needs connection passed or its own context)
                # For simplicity within transaction, append directly:
                await conn.execute(
                   """
                   UPDATE SocialLinks
                   SET link_history = COALESCE(link_history, '[]'::jsonb) || $1::jsonb
                   WHERE link_id = $2
                   """,
                   json.dumps(event_text), link_id
                )


                # Add Journal Entry
                journal_entry = (
                    f"Crossroads: {crossroads_name} with {npc_name}. "
                    f"Choice: {choice['text']} => {choice['outcome'].format(npc_name=npc_name)}"
                )
                await conn.execute(
                    """
                    INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, timestamp)
                    VALUES ($1, $2, 'relationship_crossroads', $3, CURRENT_TIMESTAMP)
                    """,
                    user_id, conversation_id, journal_entry
                )

            # Transaction commits automatically if no exceptions were raised
            return {"success": True, "outcome_text": choice["outcome"].format(npc_name=npc_name)}

    except (asyncpg.PostgresError, ConnectionError, ValueError) as e:
        # ValueError raised internally on data inconsistency
        logger.error(f"Error applying crossroads choice for link {link_id}: {e}", exc_info=True)
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error applying crossroads choice for link {link_id}: {e}", exc_info=True)
        return {"error": "An unexpected error occurred."}


async def check_for_relationship_ritual(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    Async check if any relationship triggers a Ritual event. Uses a transaction.
    """
    try:
        async with get_db_connection_context() as conn:
            # Gather all player-related links
            links = await conn.fetch(
                 """
                 SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id,
                        dynamics, experienced_rituals
                 FROM SocialLinks
                 WHERE user_id = $1 AND conversation_id = $2
                 AND (
                     (entity1_type='player' AND entity1_id = $1)
                     OR (entity2_type='player' AND entity2_id = $1)
                 )
                 """,
                 user_id, conversation_id
             )

            for link_record in links:
                link_id = link_record['link_id']
                e1t, e1id, e2t, e2id = link_record['entity1_type'], link_record['entity1_id'], link_record['entity2_type'], link_record['entity2_id']
                dyn_data = link_record['dynamics']
                rit_data = link_record['experienced_rituals']

                dynamics = dyn_data if isinstance(dyn_data, dict) else (json.loads(dyn_data) if isinstance(dyn_data, str) else {})
                experienced = rit_data if isinstance(rit_data, list) else (json.loads(rit_data) if isinstance(rit_data, str) else [])

                # Identify NPC
                npc_id = e1id if e1t == "npc" else (e2id if e2t == "npc" else None)
                if npc_id is None: continue

                # Check NPC dominance
                npc_info = await conn.fetchrow(
                     "SELECT npc_name, dominance FROM NPCStats WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3",
                     user_id, conversation_id, npc_id
                )
                if not npc_info or npc_info['dominance'] < 50: continue # Trigger requires dominance >= 50
                npc_name = npc_info['npc_name']

                # Check possible rituals
                possible_rituals = []
                for rit_def in RELATIONSHIP_RITUALS:
                     if rit_def["name"] in experienced: continue
                     triggered = False
                     for dyn_name in rit_def["dynamics"]:
                          if dynamics.get(dyn_name, 0) >= rit_def["trigger_level"]:
                               triggered = True
                               break
                     if triggered:
                          possible_rituals.append(rit_def)

                if possible_rituals:
                    # Choose one and apply it within a transaction
                    chosen_ritual = random.choice(possible_rituals)

                    async with conn.transaction(): # Start transaction for applying ritual
                        ritual_txt = chosen_ritual["ritual_text"]
                        if "{gift_item}" in ritual_txt:
                             gift_item = random.choice(SYMBOLIC_GIFTS)
                             ritual_txt = ritual_txt.format(npc_name=npc_name, gift_item=gift_item)
                        else:
                             ritual_txt = ritual_txt.format(npc_name=npc_name)

                        # Mark as experienced
                        experienced.append(chosen_ritual["name"])
                        await conn.execute(
                             """
                             UPDATE SocialLinks SET experienced_rituals = $1
                             WHERE link_id = $2
                             """,
                             json.dumps(experienced), link_id
                        )

                        # Add history event
                        event_text = f"Ritual '{chosen_ritual['name']}': {ritual_txt}"
                        await conn.execute(
                             """
                             UPDATE SocialLinks SET link_history = COALESCE(link_history, '[]'::jsonb) || $1::jsonb
                             WHERE link_id = $2
                             """,
                             json.dumps(event_text), link_id
                        )

                        # Journal entry
                        journal_text = f"Ritual with {npc_name}: {chosen_ritual['name']}. {ritual_txt}"
                        await conn.execute(
                             """
                             INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, timestamp)
                             VALUES ($1, $2, 'relationship_ritual', $3, CURRENT_TIMESTAMP)
                             """,
                             user_id, conversation_id, journal_text
                        )

                        # Increase relevant dynamics by +10
                        dynamics_changed = False
                        for dyn_name in chosen_ritual["dynamics"]:
                             old_val = dynamics.get(dyn_name, 0)
                             new_val = min(100, old_val + 10) # Clamp at 100
                             if new_val != old_val:
                                 dynamics[dyn_name] = new_val
                                 dynamics_changed = True

                        if dynamics_changed:
                             await conn.execute(
                                  """
                                  UPDATE SocialLinks SET dynamics = $1 WHERE link_id = $2
                                  """,
                                  json.dumps(dynamics), link_id
                             )

                        # Update PlayerStats
                        await conn.execute(
                             """
                             UPDATE PlayerStats
                             SET corruption = LEAST(100, corruption + 5), -- Clamp 0-100
                                 dependency = LEAST(100, dependency + 5)
                             WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
                             """,
                             user_id, conversation_id
                        )

                    # If transaction succeeded, return the result
                    return {
                         "type": "relationship_ritual",
                         "name": chosen_ritual["name"],
                         "description": chosen_ritual["description"],
                         "npc_id": npc_id,
                         "npc_name": npc_name,
                         "ritual_text": ritual_txt,
                         "link_id": link_id,
                    }
            # End loop through links
            return None # No ritual triggered for any link

    except (asyncpg.PostgresError, ConnectionError) as e:
        logger.error(f"Error checking for relationship ritual: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error checking ritual: {e}", exc_info=True)
        return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 5) Summaries & Helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def get_entity_name(
    conn: asyncpg.Connection, # Expect an active connection
    entity_type: str,
    entity_id: int,
    user_id: int,
    conversation_id: int
) -> str:
    """
    Async get the name of an entity (NPC or player) using the provided connection.
    """
    # Allow player ID 0 or matching user_id for flexibility
    if entity_type == "player" and (entity_id == 0 or entity_id == user_id):
        # Fetch player name from PlayerStats instead of hardcoding 'Chase'
        player_name = await conn.fetchval(
             "SELECT player_name FROM PlayerStats WHERE user_id = $1 AND conversation_id = $2 LIMIT 1",
             user_id, conversation_id
        )
        return player_name or "Player" # Fallback if no PlayerStats row
    elif entity_type == "npc":
        npc_name = await conn.fetchval(
             """
             SELECT npc_name FROM NPCStats
             WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
             """,
             user_id, conversation_id, entity_id
        )
        return npc_name or f"Unknown NPC {entity_id}"
    else:
        return f"Unknown Entity {entity_id}"



async def get_relationship_summary(
    user_id: int,
    conversation_id: int,
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int
) -> Optional[Dict[str, Any]]:
    """
    Async get a summary of the relationship using asyncpg.
    """
    try:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                 """
                 SELECT link_id, link_type, link_level, dynamics, link_history,
                        experienced_crossroads, experienced_rituals
                 FROM SocialLinks
                 WHERE user_id = $1 AND conversation_id = $2
                   AND entity1_type = $3 AND entity1_id = $4
                   AND entity2_type = $5 AND entity2_id = $6
                 """,
                 user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id
            )
            if not row:
                return None

            link_id, link_type, link_level, dyn_data, hist_data, cr_data, rit_data = \
                row['link_id'], row['link_type'], row['link_level'], row['dynamics'], \
                row['link_history'], row['experienced_crossroads'], row['experienced_rituals']

            dynamics = dyn_data if isinstance(dyn_data, dict) else (json.loads(dyn_data) if isinstance(dyn_data, str) else {})
            history = hist_data if isinstance(hist_data, list) else (json.loads(hist_data) if isinstance(hist_data, str) else [])
            cr_list = cr_data if isinstance(cr_data, list) else (json.loads(cr_data) if isinstance(cr_data, str) else [])
            rit_list = rit_data if isinstance(rit_data, list) else (json.loads(rit_data) if isinstance(rit_data, str) else [])

            # Use the async helper with the current connection
            e1_name = await get_entity_name(conn, entity1_type, entity1_id, user_id, conversation_id)
            e2_name = await get_entity_name(conn, entity2_type, entity2_id, user_id, conversation_id)

            dynamic_descriptions = [
                 f"{dnm.capitalize()} {lvl}/100 => {get_dynamic_description(dnm, lvl)}"
                 for dnm, lvl in dynamics.items()
            ]

            return {
                "entity1_name": e1_name,
                "entity2_name": e2_name,
                "primary_type": link_type,
                "primary_level": link_level,
                "dynamics": dynamics,
                "dynamic_descriptions": dynamic_descriptions,
                "history": history[-5:], # last 5 events
                "experienced_crossroads": cr_list,
                "experienced_rituals": rit_list,
            }
    except (asyncpg.PostgresError, ConnectionError) as e:
        logger.error(f"Error getting relationship summary: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting relationship summary: {e}", exc_info=True)
        return None



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

    async def create_npc_group(
        self, name: str, description: str, member_ids: List[int]
    ) -> Dict[str, Any]:
        """ Async create a new NPC group in the DB. """
        members_data = []
        try:
            async with get_db_connection_context() as conn:
                # Validate NPCs and gather data within the connection context
                for npc_id in member_ids:
                    row = await conn.fetchrow(
                        """
                        SELECT npc_id, npc_name, dominance, cruelty
                        FROM NPCStats
                        WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                        """,
                        self.user_id, self.conversation_id, npc_id
                    )
                    if not row:
                        return {"error": f"NPC with ID {npc_id} not found."}
                    members_data.append({
                        "npc_id": row['npc_id'], "npc_name": row['npc_name'],
                        "dominance": row['dominance'], "cruelty": row['cruelty'],
                        "role": "member", "status": "active", "joined_date": datetime.now().isoformat()
                    })

                # Generate initial dynamics and roles
                dynamics = {key: random.randint(20, 80) for key in self.GROUP_DYNAMICS.keys()}
                if len(members_data) > 1 and dynamics.get("hierarchy", 0) > 60:
                    sorted_mem = sorted(members_data, key=lambda x: x.get("dominance", 0), reverse=True)
                    sorted_mem[0]["role"] = "leader"
                    members_data = sorted_mem # Use sorted list

                group_obj = NPCGroup(name, description, members_data, dynamics)
                group_data_json = json.dumps(group_obj.to_dict())

                # Insert into table
                group_id = await conn.fetchval(
                    """
                    INSERT INTO NPCGroups (user_id, conversation_id, group_name, group_data, updated_at)
                    VALUES ($1, $2, $3, $4, NOW())
                    ON CONFLICT (user_id, conversation_id, group_name) DO NOTHING -- Or DO UPDATE if needed
                    RETURNING group_id
                    """,
                    self.user_id, self.conversation_id, name, group_data_json
                )

                if group_id is None: # Handle potential conflict where nothing was returned
                     # Fetch existing group_id if ON CONFLICT DO NOTHING occurred
                     group_id = await conn.fetchval(
                          """SELECT group_id FROM NPCGroups
                             WHERE user_id=$1 AND conversation_id=$2 AND group_name=$3""",
                          self.user_id, self.conversation_id, name
                     )
                     if group_id:
                          return {"success": True, "group_id": group_id, "message": f"Group '{name}' already exists."}
                     else:
                          # This case should be rare if UNIQUE constraint exists
                          return {"error": "Failed to create or find group after conflict."}

                return {"success": True, "group_id": group_id, "message": f"Group '{name}' created."}

        except (asyncpg.PostgresError, ConnectionError) as e:
            logger.error(f"Error creating group '{name}': {e}", exc_info=True)
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error creating group '{name}': {e}", exc_info=True)
            return {"error": "An unexpected error occurred."}

    async def get_npc_group(
        self, group_id: Optional[int] = None, group_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """ Async retrieve an NPC group by ID or name. """
        if not group_id and not group_name:
            return {"error": "Must provide group_id or group_name."}
        try:
            async with get_db_connection_context() as conn:
                query = """SELECT group_id, group_name, group_data FROM NPCGroups
                           WHERE user_id = $1 AND conversation_id = $2"""
                params = [self.user_id, self.conversation_id]
                if group_id:
                    query += " AND group_id = $3"
                    params.append(group_id)
                else:
                    query += " AND group_name = $3"
                    params.append(group_name)

                row = await conn.fetchrow(query, *params)
                if not row:
                    lookup = f"ID {group_id}" if group_id else f"name '{group_name}'"
                    return {"error": f"Group with {lookup} not found."}

                real_group_id, real_group_name, group_data = row['group_id'], row['group_name'], row['group_data']
                group_dict = group_data if isinstance(group_data, dict) else (json.loads(group_data) if isinstance(group_data, str) else {})

                # Pass group_id when reconstructing
                group_obj = NPCGroup.from_dict(group_dict, group_id=real_group_id)
                return {
                    "success": True,
                    "group_id": real_group_id,
                    "group_name": real_group_name,
                    "group_data": group_obj.to_dict(), # Return the dict form
                    "group_object": group_obj # Optionally return the object too
                }
        except (asyncpg.PostgresError, ConnectionError) as e:
            logger.error(f"Error retrieving group: {e}", exc_info=True)
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error retrieving group: {e}", exc_info=True)
            return {"error": "An unexpected error occurred."}

    async def update_group_dynamics(
        self, group_id: int, changes: Dict[str, int]
    ) -> Dict[str, Any]:
        """ Async apply increments/decrements to group dynamics. """
        try:
            async with get_db_connection_context() as conn:
                 async with conn.transaction(): # Use transaction
                    # Fetch existing group data, locking the row
                    group_data = await conn.fetchval(
                        """
                        SELECT group_data FROM NPCGroups
                        WHERE user_id = $1 AND conversation_id = $2 AND group_id = $3
                        FOR UPDATE
                        """,
                        self.user_id, self.conversation_id, group_id
                    )
                    if group_data is None:
                         raise ValueError("Group not found for update.") # Raise to rollback

                    group_dict = group_data if isinstance(group_data, dict) else (json.loads(group_data) if isinstance(group_data, str) else {})
                    group_obj = NPCGroup.from_dict(group_dict, group_id=group_id)

                    dynamics_updated = False
                    for dyn_key, delta in changes.items():
                         if dyn_key in self.GROUP_DYNAMICS: # Check if it's a known dynamic
                             current = group_obj.dynamics.get(dyn_key, 0) # Default to 0 if not present
                             new_val = max(0, min(100, current + delta)) # Clamp 0-100
                             if new_val != current:
                                 group_obj.dynamics[dyn_key] = new_val
                                 dynamics_updated = True
                         else:
                             logger.warning(f"Unknown dynamic key '{dyn_key}' skipped for group {group_id}.")

                    if not dynamics_updated:
                        return {"success": True, "message": "No dynamics changed."}

                    # Record history and update timestamp
                    group_obj.shared_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "type": "dynamics_update",
                        "details": changes,
                        "new_dynamics": group_obj.dynamics # Log final state
                    })
                    group_obj.last_activity = datetime.now().isoformat()

                    # Store back
                    updated_json = json.dumps(group_obj.to_dict())
                    await conn.execute(
                        """
                        UPDATE NPCGroups SET group_data = $1, updated_at = NOW()
                        WHERE user_id = $2 AND conversation_id = $3 AND group_id = $4
                        """,
                        updated_json, self.user_id, self.conversation_id, group_id
                    )

                 # Transaction commits here
                 return {"success": True, "message": "Group dynamics updated.", "updated_dynamics": group_obj.dynamics}

        except (asyncpg.PostgresError, ConnectionError, ValueError) as e:
            logger.error(f"Error updating dynamics for group {group_id}: {e}", exc_info=True)
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error updating dynamics for group {group_id}: {e}", exc_info=True)
            return {"error": "An unexpected error occurred."}


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

    async def produce_multi_npc_scene(
        self,
        group_id: int,
        topic: str = "General conversation",
        extra_context: str = ""
    ) -> Dict[str, Any]:
        """ Async create scene snippet and update group history. """
        group_info = await self.get_npc_group(group_id=group_id) # Await the async version
        if "error" in group_info or "group_object" not in group_info:
             return {"error": group_info.get("error", "Failed to retrieve group object.")}

        group_obj: NPCGroup = group_info["group_object"]

        style = self.determine_interaction_style(group_obj)
        style_def = self.INTERACTION_STYLES.get(style, {})
        dialogue_style_desc = style_def.get("dialogue_style", "Plain interaction.")
        desc = style_def.get("description", "")

        # Build demo lines
        members_to_include = random.sample(group_obj.members, k=min(len(group_obj.members), 3))
        lines = [f"{mem['npc_name']} ({mem.get('role', 'member')}): [Responding to {topic} with style '{style}']" for mem in members_to_include]
        scene_text = "\n".join(lines)

        # Update history and timestamp
        group_obj.shared_history.append({
            "timestamp": datetime.now().isoformat(), "type": "scene_example",
            "style": style, "topic": topic, "lines": lines
        })
        group_obj.last_activity = datetime.now().isoformat()

        # Persist changes
        try:
            async with get_db_connection_context() as conn:
                updated_json = json.dumps(group_obj.to_dict())
                await conn.execute(
                    """
                    UPDATE NPCGroups SET group_data = $1, updated_at = NOW()
                    WHERE user_id = $2 AND conversation_id = $3 AND group_id = $4
                    """,
                    updated_json, self.user_id, self.conversation_id, group_id
                )
            return {
                "success": True, "interaction_style": style, "style_description": desc,
                "dialogue_style": dialogue_style_desc, "scene_preview": scene_text
            }
        except (asyncpg.PostgresError, ConnectionError) as e:
            logger.error(f"Error storing scene in group {group_id}: {e}", exc_info=True)
            # Return the scene info even if saving fails, but log error
            return {
                "success": False, "error": f"Failed to save history: {e}",
                "interaction_style": style, "style_description": desc,
                "dialogue_style": dialogue_style_desc, "scene_preview": scene_text
            }
        except Exception as e:
             logger.error(f"Unexpected error producing scene for group {group_id}: {e}", exc_info=True)
             return {"error": "An unexpected error occurred during scene production."}

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3) Additional Utility Methods
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def list_all_groups(self) -> Dict[str, Any]:
        """ Async return a list of all NPC groups. """
        results = []
        try:
            async with get_db_connection_context() as conn:
                 rows = await conn.fetch(
                      """
                      SELECT group_id, group_name, group_data FROM NPCGroups
                      WHERE user_id = $1 AND conversation_id = $2
                      ORDER BY group_name
                      """,
                      self.user_id, self.conversation_id
                 )
                 for row in rows:
                      g_data = row['group_data']
                      g_dict = g_data if isinstance(g_data, dict) else (json.loads(g_data) if isinstance(g_data, str) else {})
                      results.append({
                           "group_id": row['group_id'],
                           "group_name": row['group_name'],
                           "data": g_dict # Return stored dict form
                      })
            return {"groups": results, "count": len(results)}
        except (asyncpg.PostgresError, ConnectionError) as e:
            logger.error(f"Error listing groups: {e}", exc_info=True)
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error listing groups: {e}", exc_info=True)
            return {"error": "An unexpected error occurred."}


    async def delete_group(self, group_id: int) -> Dict[str, Any]:
        """ Async delete an NPC group. """
        try:
            async with get_db_connection_context() as conn:
                result = await conn.execute(
                     """
                     DELETE FROM NPCGroups
                     WHERE user_id = $1 AND conversation_id = $2 AND group_id = $3
                     """,
                     self.user_id, self.conversation_id, group_id
                )
                if result == "DELETE 1":
                    return {"success": True, "message": f"Group {group_id} deleted."}
                else:
                    return {"error": f"Group {group_id} not found or not deleted."}
        except (asyncpg.PostgresError, ConnectionError) as e:
            logger.error(f"Error deleting group {group_id}: {e}", exc_info=True)
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error deleting group {group_id}: {e}", exc_info=True)
            return {"error": "An unexpected error occurred."}
         
    async def update_npc_group_dynamics(
        self,
        # Note: user_id and conversation_id are already attributes of the manager instance
        group_id: int,
        dynamics_data: Dict[str, int] # Renamed from 'changes' for clarity, maps dynamic_name -> new_value
    ) -> Dict[str, Any]:
        """
        Async update NPC group dynamics and potentially related relationships.
        Uses the instance's user_id and conversation_id.
        Accepts new target values for dynamics.
        """
        updates_applied = {"group_dynamics": [], "member_relationships": []}
        try:
            async with get_db_connection_context() as conn:
                 async with conn.transaction():
                    # --- 1. Update Group Dynamics ---
                    group_data = await conn.fetchval(
                        """SELECT group_data FROM NPCGroups
                           WHERE user_id = $1 AND conversation_id = $2 AND group_id = $3 FOR UPDATE""",
                        self.user_id, self.conversation_id, group_id
                    )
                    if group_data is None: raise ValueError("Group not found.")

                    group_dict = group_data if isinstance(group_data, dict) else json.loads(group_data)
                    group_obj = NPCGroup.from_dict(group_dict, group_id=group_id)

                    changed_dynamics = {}
                    for dynamic_key, new_value in dynamics_data.items():
                         if dynamic_key in self.GROUP_DYNAMICS:
                             # Clamp value to 0-100 (or defined range)
                             clamped_value = max(0, min(100, new_value))
                             if group_obj.dynamics.get(dynamic_key) != clamped_value:
                                 group_obj.dynamics[dynamic_key] = clamped_value
                                 changed_dynamics[dynamic_key] = clamped_value
                         else:
                              logger.warning(f"Unknown dynamic '{dynamic_key}' skipped for group {group_id}")

                    if not changed_dynamics:
                         logger.info(f"No actual change in dynamics for group {group_id}. Skipping update.")
                         return {"success": True, "message": "No changes applied.", "updates_applied": updates_applied}

                    group_obj.shared_history.append({
                         "timestamp": datetime.now().isoformat(),
                         "type": "dynamics_set", # Changed type
                         "details": changed_dynamics
                    })
                    group_obj.last_activity = datetime.now().isoformat()
                    updated_json = json.dumps(group_obj.to_dict())

                    await conn.execute(
                        "UPDATE NPCGroups SET group_data = $1, updated_at = NOW() WHERE group_id = $2",
                        updated_json, group_id
                    )
                    updates_applied["group_dynamics"] = changed_dynamics
                    logger.info(f"Updated group {group_id} dynamics: {changed_dynamics}")


                    # --- 2. Optional: Update Member Relationships (Example Logic) ---
                    # This part depends heavily on how you want group dynamics to affect pairs
                    # Example: Higher cohesion slightly increases trust between members
                    if "cohesion" in changed_dynamics and group_obj.members and len(group_obj.members) > 1:
                         cohesion_level = changed_dynamics["cohesion"]
                         level_change = 0
                         if cohesion_level > 70: level_change = 3 # High cohesion -> more trust
                         elif cohesion_level < 30: level_change = -3 # Low cohesion -> less trust

                         if level_change != 0:
                             for i in range(len(group_obj.members)):
                                 for j in range(i + 1, len(group_obj.members)):
                                     m1_id = group_obj.members[i]["npc_id"]
                                     m2_id = group_obj.members[j]["npc_id"]

                                     # Need to handle link directionality or fetch regardless of order
                                     # Let's assume order doesn't matter for the relationship itself
                                     link_id = await conn.fetchval(
                                           """SELECT link_id FROM SocialLinks
                                              WHERE user_id=$1 AND conversation_id=$2
                                              AND ((entity1_type='npc' AND entity1_id=$3 AND entity2_type='npc' AND entity2_id=$4)
                                                OR (entity1_type='npc' AND entity1_id=$4 AND entity2_type='npc' AND entity2_id=$3))
                                           """, self.user_id, self.conversation_id, m1_id, m2_id
                                     )

                                     if link_id:
                                         # Example: Modify 'trust' dimension within the link's dynamics JSONB
                                         # Fetch current dynamics
                                         current_link_dynamics = await conn.fetchval(
                                               "SELECT dynamics FROM SocialLinks WHERE link_id=$1", link_id
                                         )
                                         link_dynamics = current_link_dynamics if isinstance(current_link_dynamics, dict) else json.loads(current_link_dynamics or '{}')

                                         trust_level = link_dynamics.get('trust', 0)
                                         new_trust = max(-100, min(100, trust_level + level_change)) # Clamp -100 to 100 for trust

                                         if new_trust != trust_level:
                                             link_dynamics['trust'] = new_trust
                                             # Update the link's dynamics field
                                             await conn.execute(
                                                  "UPDATE SocialLinks SET dynamics=$1 WHERE link_id=$2",
                                                  json.dumps(link_dynamics), link_id
                                             )
                                             # Add event to link history
                                             event = f"Trust changed to {new_trust} due to group cohesion update."
                                             await conn.execute(
                                                 "UPDATE SocialLinks SET link_history = COALESCE(link_history, '[]'::jsonb) || $1::jsonb WHERE link_id=$2",
                                                 json.dumps(event), link_id
                                             )
                                             updates_applied["member_relationships"].append({
                                                  "member1_id": m1_id, "member2_id": m2_id,
                                                  "change": {"trust": level_change}
                                             })


            # Transaction commits automatically
            return {"success": True, "updates_applied": updates_applied}

        except (asyncpg.PostgresError, ConnectionError, ValueError) as e:
            logger.error(f"Error updating group dynamics for {group_id}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error updating group dynamics for {group_id}: {e}", exc_info=True)
            return {"success": False, "error": "An unexpected error occurred."}


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
    # user_id: int, # Use from context
    # conversation_id: int, # Use from context
    conflict_id: int,
    resolution_data: Dict[str, Any] # e.g., {"success": True, "winning_faction": "rebels", "outcome": "Rebels won"}
) -> dict:
    """
    Update relationships based on conflict resolution outcomes. Uses user_id and conversation_id from context.
    """
    user_id = ctx.run_context.get("user_id")
    conversation_id = ctx.run_context.get("conversation_id")
    if not user_id or not conversation_id:
        return {"success": False, "error": "user_id and conversation_id required in context."}

    updates_applied = []
    try:
        # Await the assumed async function
        stakeholders = await get_conflict_stakeholders(ctx, conflict_id)
        if not stakeholders:
             return {"success": True, "message": "No stakeholders found for conflict.", "updates_applied": []}

        async with get_db_connection_context() as conn:
            # Optionally run multiple updates in a transaction for consistency
            async with conn.transaction():
                for stakeholder in stakeholders:
                    npc_id = stakeholder.get("npc_id")
                    if not npc_id: continue

                    # Get player's link with this NPC (assuming player ID is user_id)
                    # Handle both directions
                    player_link_id = await conn.fetchval(
                         """SELECT link_id FROM SocialLinks
                            WHERE user_id=$1 AND conversation_id=$2
                            AND ((entity1_type='player' AND entity1_id=$1 AND entity2_type='npc' AND entity2_id=$3)
                              OR (entity1_type='npc' AND entity1_id=$3 AND entity2_type='player' AND entity2_id=$1))
                         """, user_id, conversation_id, npc_id
                    )

                    if not player_link_id: continue

                    # --- Calculate changes based on resolution_data ---
                    level_change = 0
                    # Example logic:
                    resolution_success = resolution_data.get("success", False)
                    winning_faction = resolution_data.get("winning_faction")
                    npc_faction = stakeholder.get("faction_position") # Assuming this field exists

                    if winning_faction:
                         if npc_faction == winning_faction:
                              level_change = 10 if resolution_success else -10
                         else:
                              level_change = -5 if resolution_success else 5
                    else: # No specific winner, maybe base on success?
                         level_change = 3 if resolution_success else -3

                    # --- Apply changes (Example: Update overall level) ---
                    update_result = await update_link_type_and_level( # Calls the async version
                         user_id, conversation_id, player_link_id, None, level_change
                    )
                    # Note: update_link_type_and_level uses its own connection context.
                    # If you want this *within* the current transaction, you'd need to
                    # pass 'conn' to it or reimplement its logic here.
                    # For simplicity, we'll assume separate connection is acceptable here.

                    if update_result:
                        updates_applied.append({ "npc_id": npc_id, "changes": update_result })
                        # Add history event
                        event_text = f"Relationship changed due to conflict {conflict_id} resolution: {resolution_data.get('outcome', 'unknown')}"
                        await add_link_event(user_id, conversation_id, player_link_id, event_text) # Also uses its own context

        return {"success": True, "updates_applied": updates_applied}

    except Exception as e:
        logger.error(f"Error updating relationships from conflict {conflict_id}: {e}", exc_info=True)
        return {"success": False, "error": str(e)}




@function_tool
async def update_relationship_context(
    ctx: RunContextWrapper,
    # user_id: int, # Use from context
    # conversation_id: int, # Use from context
    entity1_type: str,
    entity1_id: int,
    entity2_type: str,
    entity2_id: int,
    context_data: Dict[str, Any] # Data to merge into the context JSONB field
) -> Dict[str, Any]:
    """
    Update relationship context (merges JSONB data). Uses user_id and conversation_id from context.
    Assumes SocialLinks table has a 'context' JSONB column.
    """
    user_id = ctx.run_context.get("user_id")
    conversation_id = ctx.run_context.get("conversation_id")
    if not user_id or not conversation_id:
        return {"success": False, "error": "user_id and conversation_id required in context."}

    if not context_data:
         return {"success": False, "error": "No context_data provided."}

    try:
        # Ensure SocialLinks has a 'context' JSONB column
        # ALTER TABLE SocialLinks ADD COLUMN IF NOT EXISTS context JSONB;

        async with get_db_connection_context() as conn:
            # Find link_id, handling both directions
            link_id = await conn.fetchval(
                 """SELECT link_id FROM SocialLinks
                    WHERE user_id=$1 AND conversation_id=$2
                    AND ((entity1_type=$3 AND entity1_id=$4 AND entity2_type=$5 AND entity2_id=$6)
                      OR (entity1_type=$5 AND entity1_id=$6 AND entity2_type=$3 AND entity2_id=$4))
                 """, user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id
            )

            if not link_id:
                 # Option: Create the link first if it doesn't exist?
                 # link_id = await create_social_link(user_id, ...)
                 # if not link_id: return {"success": False, "error": "Relationship not found and could not be created."}
                 return {"success": False, "error": "Relationship not found."}


            # Merge the new context data into the existing context JSONB
            # The || operator concatenates/merges JSONB objects
            await conn.execute(
                """
                UPDATE SocialLinks
                SET context = COALESCE(context, '{}'::jsonb) || $1::jsonb
                WHERE link_id = $2
                """,
                json.dumps(context_data), link_id
            )

            # Add event to relationship history
            event_text = f"Relationship context updated: {json.dumps(context_data)}"
            # Call the async helper function
            await add_link_event(user_id, conversation_id, link_id, event_text)

        return {"success": True, "link_id": link_id, "context_updated": context_data}

    except (asyncpg.PostgresError, ConnectionError) as e:
        logger.error(f"Error updating relationship context: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error updating relationship context: {e}", exc_info=True)
        return {"success": False, "error": "An unexpected error occurred."}
     
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 9) The Agent: "SocialLinksAgent"
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from logic.chatgpt_integration import get_agents_openai_model

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
    model = get_agents_openai_model(),  # or "gpt-4o", "gpt-3.5-turbo", etc.
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
