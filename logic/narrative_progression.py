# logic/narrative_progression.py

import logging
import json
import random
import asyncio
from typing import Dict, List, Any, Optional, Union, NamedTuple
from datetime import datetime

import asyncpg

from db.connection import get_db_connection_context
from logic.chatgpt_integration import get_chatgpt_response
from lore.core import canon
from lore.core.lore_system import LoreSystem

logger = logging.getLogger(__name__)

class NarrativeStage(NamedTuple):
    """Represents a stage in the narrative progression."""
    name: str
    description: str
    required_corruption: int
    required_dependency: int

# Define the narrative stages
NARRATIVE_STAGES = [
    NarrativeStage(
        name="Innocent Beginning", 
        description="The initial stage where the player is unaware of underlying manipulation. NPCs maintain a facade of normality while subtly establishing control patterns.",
        required_corruption=0,
        required_dependency=0
    ),
    NarrativeStage(
        name="First Doubts", 
        description="The player begins to notice inconsistencies in NPC behavior and experiences moments of unease. Controlling behavior becomes slightly more overt.",
        required_corruption=20,
        required_dependency=15
    ),
    NarrativeStage(
        name="Creeping Realization", 
        description="The player has clear moments of awareness about their manipulation, though they may rationalize or dismiss them. NPCs are less concerned with hiding their control.",
        required_corruption=40,
        required_dependency=35
    ),
    NarrativeStage(
        name="Veil Thinning", 
        description="The facade is largely abandoned, with NPCs openly manipulating and controlling the player. The player struggles with their diminishing autonomy.",
        required_corruption=60,
        required_dependency=55
    ),
    NarrativeStage(
        name="Full Revelation", 
        description="The player fully acknowledges their position and the true nature of their relationships. NPCs no longer pretend, and the power dynamic is explicitly enacted.",
        required_corruption=80,
        required_dependency=75
    )
]

async def get_current_narrative_stage(user_id: int, conversation_id: int) -> NarrativeStage:
    """
    Determine the current narrative stage based on player stats asynchronously.

    Args:
        user_id: ID of the user
        conversation_id: ID of the conversation

    Returns:
        The current narrative stage (defaults to the first stage on error or if no stats found).
    """
    query = """
        SELECT corruption, dependency
        FROM PlayerStats
        WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
        ORDER BY timestamp DESC
        LIMIT 1
    """
    corruption: float = 0.0
    dependency: float = 0.0

    try:
        async with get_db_connection_context() as conn:
            row: Optional[asyncpg.Record] = await conn.fetchrow(query, user_id, conversation_id)

        if row:
            corruption = row['corruption'] or 0.0
            dependency = row['dependency'] or 0.0
        else:
            logger.warning(f"No PlayerStats found for user {user_id}, convo {conversation_id}. Assuming stage 0.")

    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logger.error(f"Database error determining narrative stage for user {user_id}, convo {conversation_id}: {db_err}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error determining narrative stage for user {user_id}, convo {conversation_id}: {e}", exc_info=True)

    # Determine the highest stage the player qualifies for
    current_stage = NARRATIVE_STAGES[0]
    for stage in NARRATIVE_STAGES:
        if float(corruption) >= stage.required_corruption and float(dependency) >= stage.required_dependency:
            current_stage = stage
        else:
            break

    return current_stage

async def check_for_personal_revelations(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    Check if conditions are right for a personal revelation asynchronously.

    Args:
        user_id: ID of the user
        conversation_id: ID of the conversation

    Returns:
        Personal revelation data if one should occur, None otherwise
    """
    query_recent_revelations = """
        SELECT COUNT(*) FROM PlayerJournal
        WHERE user_id = $1 AND conversation_id = $2 AND entry_type = 'personal_revelation'
        AND timestamp > NOW() - INTERVAL '5 days'
    """
    query_player_stats = """
        SELECT corruption, confidence, willpower, obedience, dependency, lust
        FROM PlayerStats
        WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
        ORDER BY timestamp DESC LIMIT 1
    """
    query_stat_changes = """
        SELECT stat_name, new_value - old_value as change
        FROM StatsHistory
        WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
        AND timestamp > NOW() - INTERVAL '7 days'
        ORDER BY timestamp DESC
        LIMIT 10
    """
    query_npc = """
        SELECT npc_id, npc_name
        FROM NPCStats
        WHERE user_id = $1 AND conversation_id = $2 AND introduced = TRUE
        ORDER BY dominance DESC
        LIMIT 1
    """

    try:
        async with get_db_connection_context() as conn:
            # Check for recent revelations
            recent_count: int = await conn.fetchval(query_recent_revelations, user_id, conversation_id) or 0

            if recent_count > 2:
                logger.debug(f"Skipping personal revelation for user {user_id}, convo {conversation_id}: too many recent ({recent_count}).")
                return None

            # Get player stats
            player_stats_row: Optional[asyncpg.Record] = await conn.fetchrow(query_player_stats, user_id, conversation_id)
            if not player_stats_row:
                logger.warning(f"Skipping personal revelation: PlayerStats not found for user {user_id}, convo {conversation_id}.")
                return None

            corruption = player_stats_row['corruption'] or 0.0
            confidence = player_stats_row['confidence'] or 50.0
            willpower = player_stats_row['willpower'] or 50.0
            obedience = player_stats_row['obedience'] or 0.0
            dependency = player_stats_row['dependency'] or 0.0
            lust = player_stats_row['lust'] or 0.0

            # Determine which stat has changed the most recently
            stat_change_rows: List[asyncpg.Record] = await conn.fetch(query_stat_changes, user_id, conversation_id)
            stat_changes = {}
            for row in stat_change_rows:
                stat_name = row['stat_name']
                change = row['change'] or 0.0
                stat_changes[stat_name] = stat_changes.get(stat_name, 0) + float(change)

            # Determine revelation type
            revelation_type = ""
            if not stat_changes or all(v == 0 for v in stat_changes.values()):
                stat_values = {
                    "dependency": dependency, "obedience": obedience, "corruption": corruption,
                    "willpower": -willpower, "confidence": -confidence
                }
                deviations = {
                    "dependency": dependency, "obedience": obedience, "corruption": corruption,
                    "willpower": abs(willpower - 50), "confidence": abs(confidence - 50)
                }
                max_dev_stat = max(deviations, key=deviations.get) if deviations else None

                if max_dev_stat:
                     revelation_type = max_dev_stat
                else:
                     revelation_type = random.choice(["dependency", "obedience", "corruption", "willpower", "confidence"])
            else:
                max_change_stat = max(stat_changes.items(), key=lambda x: abs(x[1]))
                stat_name = max_change_stat[0]
                valid_types = ["dependency", "obedience", "corruption", "willpower", "confidence"]
                if stat_name in valid_types:
                    revelation_type = stat_name
                else:
                    revelation_type = random.choice(valid_types)

            # Get an NPC to associate with the revelation
            npc_row: Optional[asyncpg.Record] = await conn.fetchrow(query_npc, user_id, conversation_id)
            if not npc_row:
                logger.warning(f"Skipping personal revelation: No suitable NPC found for user {user_id}, convo {conversation_id}.")
                return None

            npc_id = npc_row['npc_id']
            npc_name = npc_row['npc_name']
        
            # Define revelation templates
            templates = {
                "dependency": [
                    f"I've been checking my phone constantly to see if {npc_name} has messaged me. When did I start needing her approval so much?",
                    f"I realized today that I haven't made a significant decision without consulting {npc_name} in weeks. Is that normal?",
                    f"The thought of spending a day without talking to {npc_name} makes me anxious. I should be concerned about that, shouldn't I?"
                ],
                "obedience": [
                    f"I caught myself automatically rearranging my schedule when {npc_name} hinted she wanted to see me. I didn't even think twice about it.",
                    f"Today I changed my opinion the moment I realized it differed from {npc_name}'s. That's... not like me. Or is it becoming like me?",
                    f"{npc_name} gave me that look, and I immediately stopped what I was saying. When did her disapproval start carrying so much weight?"
                ],
                "corruption": [
                    f"I found myself enjoying the feeling of following {npc_name}'s instructions perfectly. The pride I felt at her approval was... intense.",
                    f"Last year, I would have been offended if someone treated me the way {npc_name} did today. Now I'm grateful for her attention.",
                    f"Sometimes I catch glimpses of my old self, like a stranger I used to know. When did I change so fundamentally?"
                ],
                "willpower": [
                    f"I had every intention of saying no to {npc_name} today. The 'yes' came out before I even realized I was speaking.",
                    f"I've been trying to remember what it felt like to disagree with {npc_name}. The memory feels distant, like it belongs to someone else.",
                    f"I made a list of boundaries I wouldn't cross. Looking at it now, I've broken every single one at {npc_name}'s suggestion."
                ],
                "confidence": [
                    f"I opened my mouth to speak in the meeting, then saw {npc_name} watching me. I suddenly couldn't remember what I was going to say.",
                    f"I used to trust my judgment. Now I find myself second-guessing every thought that {npc_name} hasn't explicitly approved.",
                    f"When did I start feeling this small? This uncertain? I can barely remember how it felt to be sure of myself."
                ]
            }
            
            # Choose a random template for the selected type
            available_templates = templates.get(revelation_type)
            if not available_templates:
                 logger.error(f"Missing templates for revelation type '{revelation_type}'")
                 available_templates = templates["dependency"]

            inner_monologue = random.choice(available_templates)
            inner_monologue = inner_monologue.replace("{npc_name}", npc_name)

        # REFACTORED: Use canon to create the journal entry
        ctx = type('Context', (), {'user_id': user_id, 'conversation_id': conversation_id})()
        
        async with get_db_connection_context() as conn:
            # The canon module doesn't have a specific journal entry creation function,
            # so we'll still need to do a direct insert here as PlayerJournal is not a "core" table
            journal_id: Optional[int] = await conn.fetchval(
                """
                INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, revelation_types, timestamp)
                VALUES ($1, $2, 'personal_revelation', $3, $4, CURRENT_TIMESTAMP)
                RETURNING id
                """,
                user_id, conversation_id, inner_monologue, revelation_type
            )

            if journal_id is None:
                 logger.error(f"Failed to insert personal revelation for user {user_id}, convo {conversation_id}.")
                 return None

        logger.info(f"Generated personal revelation (type: {revelation_type}) for user {user_id}, convo {conversation_id}. Journal ID: {journal_id}")

        return {
            "type": "personal_revelation",
            "npc_id": npc_id,
            "npc_name": npc_name,
            "name": f"{revelation_type.capitalize()} Revelation",
            "inner_monologue": inner_monologue,
            "journal_id": journal_id
        }

    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logger.error(f"Database error checking for personal revelations for user {user_id}, convo {conversation_id}: {db_err}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error checking for personal revelations for user {user_id}, convo {conversation_id}: {e}", exc_info=True)
        return None

async def check_for_narrative_moments(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    Check if conditions are right for a narrative moment asynchronously.

    Args:
        user_id: ID of the user
        conversation_id: ID of the conversation

    Returns:
        Narrative moment data if one should occur, None otherwise
    """
    # Get the current narrative stage first
    current_stage = await get_current_narrative_stage(user_id, conversation_id)
    stage_name = current_stage.name if current_stage else NARRATIVE_STAGES[0].name

    query_recent_moments = """
        SELECT COUNT(*) FROM PlayerJournal
        WHERE user_id = $1 AND conversation_id = $2 AND entry_type = 'narrative_moment'
        AND timestamp > NOW() - INTERVAL '7 days'
    """
    query_npcs = """
        SELECT npc_id, npc_name
        FROM NPCStats
        WHERE user_id = $1 AND conversation_id = $2 AND introduced = TRUE
        ORDER BY dominance DESC
        LIMIT 3
    """

    try:
        async with get_db_connection_context() as conn:
            # Check for recent narrative moments
            recent_count: int = await conn.fetchval(query_recent_moments, user_id, conversation_id) or 0

            if recent_count > 2:
                logger.debug(f"Skipping narrative moment for user {user_id}, convo {conversation_id}: too many recent ({recent_count}).")
                return None

            # Get NPCs for the narrative moment
            npc_rows: List[asyncpg.Record] = await conn.fetch(query_npcs, user_id, conversation_id)

            if not npc_rows:
                logger.warning(f"Skipping narrative moment: No suitable NPCs found for user {user_id}, convo {conversation_id}.")
                return None

            npcs_data = [(row['npc_id'], row['npc_name']) for row in npc_rows]
        
            # Choose an appropriate narrative moment based on stage
            stage_name = current_stage.name
            
            # Define narrative moment templates by stage
            templates = {
                "Innocent Beginning": [
                    {
                        "name": "Subtle Power Play",
                        "scene_text": f"You notice {npcs_data[0][1] if npcs_data else 'her'} subtly directing the conversation, a hint of authority in her voice that you hadn't detected before. When you start to speak, she touches your arm lightly, and you find yourself deferring to her opinion without thinking.",
                        "player_realization": "There's something about her presence that makes me naturally step back, almost without noticing."
                    },
                    {
                        "name": "Casual Testing",
                        "scene_text": f"{npcs_data[0][1] if npcs_data else 'She'} asks you to handle a small errand for her, as if it's nothing important. Yet the request comes with such confidence that refusing doesn't seem to be an option. You complete the task, and her approval feels strangely rewarding.",
                        "player_realization": "I'm already looking for ways to please her, even when she doesn't explicitly ask."
                    }
                ],
                "First Doubts": [
                    {
                        "name": "Conflicting Instructions",
                        "scene_text": f"{npcs_data[0][1] if npcs_data else 'She'} and {npcs_data[1][1] if len(npcs_data) > 1 else 'another woman'} give you contradicting instructions, watching your reaction carefully. You feel a moment of confusion, then anxiety about disappointing either of them. The choice seems significant in ways you can't quite articulate.",
                        "player_realization": "Why am I so concerned about their approval? When did their opinions start to matter this much?"
                    },
                    {
                        "name": "Escalating Demands",
                        "scene_text": f"What started as simple favors for {npcs_data[0][1] if npcs_data else 'her'} have gradually become more demanding. Today's request crosses a boundary you once considered firm. You hesitate, but find yourself agreeing anyway, disturbed by how quickly your resistance crumbled.",
                        "player_realization": "My boundaries seem to be shifting without my conscious choice. It's happening so gradually I barely notice."
                    }
                ],
                "Creeping Realization": [
                    {
                        "name": "Private Conversation Overheard",
                        "scene_text": f"You overhear {npcs_data[0][1] if npcs_data else 'her'} and {npcs_data[1][1] if len(npcs_data) > 1 else 'another'} discussing you. \"They're coming along nicely,\" {npcs_data[0][1] if npcs_data else 'she'} says. \"Much more pliable than at the beginning.\" When they notice your presence, they smoothly transition to another topic, but the moment lingers in your mind.",
                        "player_realization": "That sounded like they've been... what, conditioning me? Training me? That can't be right..."
                    },
                    {
                        "name": "Mask Slippage",
                        "scene_text": f"{npcs_data[0][1] if npcs_data else 'Her'}'s usual warm demeanor briefly vanishes when you show reluctance to follow a suggestion. The cold, calculating look in her eyes is gone in an instant, replaced by her familiar smile, but you can't unsee that moment of revelation.",
                        "player_realization": "There's something beneath the surface I've been missing - or deliberately not seeing."
                    }
                ],
                "Veil Thinning": [
                    {
                        "name": "Open Acknowledgment",
                        "scene_text": f"\"You know what's happening, don't you?\" {npcs_data[0][1] if npcs_data else 'she'} asks quietly, studying your face. \"You've known for a while.\" There's no pretense now, just a direct acknowledgment of the power dynamic that's been building all along. Your heart races as the unspoken finally becomes spoken.",
                        "player_realization": "There's a strange relief in finally admitting what I've felt for so long."
                    },
                    {
                        "name": "Group Dynamic Revealed",
                        "scene_text": f"You enter the room to find {npcs_data[0][1] if npcs_data else 'her'}, {npcs_data[1][1] if len(npcs_data) > 1 else 'another'}, and {npcs_data[2][1] if len(npcs_data) > 2 else 'others'} waiting for you. The atmosphere is different - they're no longer maintaining the fiction of equality. Their expectations are clear in their posture, their gaze. This is what it's always been leading toward.",
                        "player_realization": "They've been coordinating all along, each playing their part in this transformation."
                    }
                ],
                "Full Revelation": [
                    {
                        "name": "Complete Transparency",
                        "scene_text": f"{npcs_data[0][1] if npcs_data else 'She'} explains exactly how they've been shaping your behavior over time, point by point, with a clinical precision that's both disturbing and fascinating. \"And the most beautiful part,\" she concludes, \"is that even knowing this, you'll continue on the same path.\"",
                        "player_realization": "She's right. Knowledge doesn't equal freedom. I understand everything and it changes nothing."
                    },
                    {
                        "name": "Ceremonial Acknowledgment",
                        "scene_text": f"The gathering has an almost ritual quality. Each person present, including {npcs_data[0][1] if npcs_data else 'her'} and {npcs_data[1][1] if len(npcs_data) > 1 else 'another'}, speaks about your journey from independence to your current state. There's pride in their voices - not for breaking you, but for revealing who you truly are. The distinction feels meaningful, even if you're not sure it's real.",
                        "player_realization": "Is this who I was always meant to be, or who they've made me? Does the difference even matter anymore?"
                    }
                ]
            }
            
            # Choose a template for the current stage
            stage_templates = templates.get(stage_name, templates["Innocent Beginning"])
            if not stage_templates:
                logger.error(f"Missing templates for narrative stage '{stage_name}'")
                return None

            chosen_template = random.choice(stage_templates)
            scene_text = chosen_template["scene_text"]

        # Create journal entry using direct insert (PlayerJournal is not a core table)
        async with get_db_connection_context() as conn:
            journal_id: Optional[int] = await conn.fetchval(
                """
                INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, narrative_moment, timestamp)
                VALUES ($1, $2, 'narrative_moment', $3, $4, CURRENT_TIMESTAMP)
                RETURNING id
                """,
                user_id, conversation_id, scene_text, chosen_template["name"]
            )

            if journal_id is None:
                logger.error(f"Failed to insert narrative moment for user {user_id}, convo {conversation_id}.")
                return None

        logger.info(f"Generated narrative moment '{chosen_template['name']}' (stage: {stage_name}) for user {user_id}, convo {conversation_id}. Journal ID: {journal_id}")

        return {
            "type": "narrative_moment",
            "name": chosen_template["name"],
            "scene_text": scene_text,
            "player_realization": chosen_template["player_realization"],
            "stage": stage_name,
            "journal_id": journal_id
        }

    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logger.error(f"Database error checking for narrative moments for user {user_id}, convo {conversation_id}: {db_err}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error checking for narrative moments for user {user_id}, convo {conversation_id}: {e}", exc_info=True)
        return None

async def initialize_player_stats(user_id: int, conversation_id: int):
    """Ensure player stats exist for the given user/conversation."""
    try:
        async with get_db_connection_context() as conn:
            # First check if the conversation exists
            conversation_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM conversations WHERE id = $1 AND user_id = $2)",
                conversation_id, user_id
            )
            
            if not conversation_exists:
                logger.warning(f"Cannot initialize player stats: Conversation {conversation_id} doesn't exist for user {user_id}")
                return False
                
            # Check if stats already exist
            row = await conn.fetchrow(
                "SELECT * FROM PlayerStats WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'",
                user_id, conversation_id
            )
            
            if not row:
                # REFACTORED: Direct insert is okay here as PlayerStats is not a core table
                await conn.execute(
                    """
                    INSERT INTO PlayerStats 
                    (user_id, conversation_id, player_name, corruption, dependency, confidence, 
                     willpower, obedience, lust, timestamp)
                    VALUES ($1, $2, 'Chase', 0, 0, 50, 50, 0, 0, CURRENT_TIMESTAMP)
                    """,
                    user_id, conversation_id
                )
                logger.info(f"Initialized default PlayerStats for user {user_id}, convo {conversation_id}")
                return True
            return False
    except Exception as e:
        logger.error(f"Error initializing player stats: {e}")
        return False

async def check_for_npc_revelations(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    Check if an NPC should have a revelation about the player asynchronously.

    Args:
        user_id: ID of the user
        conversation_id: ID of the conversation

    Returns:
        NPC revelation data if one should occur, None otherwise
    """
    # Get the current narrative stage first
    current_stage = await get_current_narrative_stage(user_id, conversation_id)
    stage_name = current_stage.name if current_stage else NARRATIVE_STAGES[0].name

    # Only show NPC revelations in later stages
    if stage_name in ["Innocent Beginning", "First Doubts"]:
        return None

    query_recent_npc_revelations = """
        SELECT COUNT(*) FROM NPCRevelations
        WHERE user_id = $1 AND conversation_id = $2
        AND timestamp > NOW() - INTERVAL '5 days'
    """
    query_npc = """
        SELECT npc_id, npc_name
        FROM NPCStats
        WHERE user_id = $1 AND conversation_id = $2 AND introduced = TRUE
        ORDER BY dominance DESC
        LIMIT 1
    """

    try:
        async with get_db_connection_context() as conn:
            # Check for recent NPC revelations
            recent_count: int = await conn.fetchval(query_recent_npc_revelations, user_id, conversation_id) or 0

            if recent_count > 2:
                logger.debug(f"Skipping NPC revelation for user {user_id}, convo {conversation_id}: too many recent ({recent_count}).")
                return None

            # Get an NPC with high dominance
            npc_row: Optional[asyncpg.Record] = await conn.fetchrow(query_npc, user_id, conversation_id)
            if not npc_row:
                logger.warning(f"Skipping NPC revelation: No suitable NPC found for user {user_id}, convo {conversation_id}.")
                return None

            npc_id = npc_row['npc_id']
            npc_name = npc_row['npc_name']
        
            # Define revelation templates by stage
            templates = {
                "Creeping Realization": [
                    f"{npc_name} observes your growing dependency with quiet satisfaction, noting how you've started to anticipate her preferences.",
                    f"{npc_name} discusses your progress with colleagues, pleased by how predictably you respond to subtle conditioning."
                ],
                "Veil Thinning": [
                    f"{npc_name} no longer bothers to hide her manipulative tactics, knowing you're aware but unable to resist.",
                    f"{npc_name} openly acknowledges the power dynamic between you, testing how you respond to explicit control."
                ],
                "Full Revelation": [
                    f"{npc_name} explains exactly how she's shaped your behavior over time, knowing that understanding won't lead to change.",
                    f"{npc_name} treats your transformation as complete, focusing now on deepening rather than establishing control."
                ]
            }
            
            # Choose a template for the current stage
            stage_templates = templates.get(stage_name)
            if not stage_templates:
                logger.warning(f"No NPC revelation templates defined for stage '{stage_name}'. Skipping.")
                return None

            revelation_text = random.choice(stage_templates)
            revelation_text = revelation_text.replace("{npc_name}", npc_name)

            # Insert the NPC revelation (NPCRevelations is not a core table)
            revelation_id: Optional[int] = await conn.fetchval(
                """
                INSERT INTO NPCRevelations
                (user_id, conversation_id, npc_id, narrative_stage, revelation_text, timestamp)
                VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)
                RETURNING id
                """,
                user_id, conversation_id, npc_id, stage_name, revelation_text
            )

            if revelation_id is None:
                logger.error(f"Failed to insert NPC revelation for user {user_id}, convo {conversation_id}.")
                return None

        logger.info(f"Generated NPC revelation (NPC: {npc_id}, stage: {stage_name}) for user {user_id}, convo {conversation_id}. Revelation ID: {revelation_id}")

        return {
            "type": "npc_revelation",
            "npc_id": npc_id,
            "npc_name": npc_name,
            "stage": stage_name,
            "revelation_text": revelation_text,
            "revelation_id": revelation_id
        }

    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logger.error(f"Database error checking for NPC revelations for user {user_id}, convo {conversation_id}: {db_err}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error checking for NPC revelations for user {user_id}, convo {conversation_id}: {e}", exc_info=True)
        return None

async def add_dream_sequence(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    Generate and add a dream sequence to the player's journal asynchronously.

    Args:
        user_id: ID of the user
        conversation_id: ID of the conversation

    Returns:
        Dream sequence data if successful, None otherwise
    """
    query_npcs = """
        SELECT npc_id, npc_name
        FROM NPCStats
        WHERE user_id = $1 AND conversation_id = $2 AND introduced = TRUE
        ORDER BY dominance DESC
        LIMIT 3
    """

    try:
        async with get_db_connection_context() as conn:
            # Get NPCs for the dream
            npc_rows: List[asyncpg.Record] = await conn.fetch(query_npcs, user_id, conversation_id)
            if not npc_rows:
                logger.warning(f"Skipping dream sequence: No suitable NPCs found for user {user_id}, convo {conversation_id}.")
                return None

            npcs_data = [(row['npc_id'], row['npc_name']) for row in npc_rows]

        # Get the current narrative stage
        current_stage = await get_current_narrative_stage(user_id, conversation_id)
        stage_name = current_stage.name if current_stage else NARRATIVE_STAGES[0].name
        
        # Dream templates
        templates = [
            "You're sitting in a chair as {npc1} circles you slowly. \"Show me your hands,\" she says. "
            "You extend them, surprised to find intricate strings wrapped around each finger, extending upward. "
            "\"Do you see who's holding them?\" she asks. You look up, but the ceiling is mirrored, "
            "showing only your own face looking back down at you, smiling with an expression that isn't yours.",
                
            "You're searching your home frantically, calling {npc1}'s name. The rooms shift and expand, "
            "doorways leading to impossible spaces. Your phone rings. It's {npc1}. \"Where are you?\" you ask desperately. "
            "\"I'm right here,\" she says, her voice coming both from the phone and from behind you. "
            "\"I've always been right here. You're the one who's lost.\"",
                
            "You're trying to walk away from {npc1}, but your feet sink deeper into the floor with each step. "
            "\"I don't understand why you're struggling,\" she says, not moving yet somehow keeping pace beside you. "
            "\"You stopped walking on your own long ago.\" You look down to find your legs have merged with the floor entirely, "
            "indistinguishable from the material beneath.",
                
            "You're giving a presentation to a room full of people, but every time you speak, your voice comes out as {npc1}'s voice, "
            "saying words you didn't intend. The audience nods approvingly. \"Much better,\" whispers {npc2} from beside you. "
            "\"Your ideas were never as good as hers anyway.\"",
                
            "You're walking through an unfamiliar house, opening doors that should lead outside but only reveal more rooms. "
            "In each room, {npc1} is engaged in a different activity, wearing a different expression. In the final room, "
            "all versions of her turn to look at you simultaneously. \"Which one is real?\" they ask in unison. \"The one you needed, or the one who needed you?\"",
                
            "You're swimming in deep water. Below you, {npc1} and {npc2} walk along the bottom, "
            "looking up at you and conversing, their voices perfectly clear despite the water. "
            "\"They still think they're above it all,\" says {npc1}, and they both laugh. You realize you can't remember how to reach the surface."
        ]
            
        # Choose a template
        dream_template = random.choice(templates)

        # Format with NPC names
        npc_names = [name for id, name in npcs_data]
        npc1 = npc_names[0] if len(npc_names) > 0 else "a dominant figure"
        npc2 = npc_names[1] if len(npc_names) > 1 else "another woman"
        npc3 = npc_names[2] if len(npc_names) > 2 else "someone else"

        dream_text = dream_template
        try:
            dream_text = dream_template.format(npc1=npc1, npc2=npc2, npc3=npc3)
        except KeyError as fmt_err:
             logger.warning(f"Key error formatting dream template: {fmt_err}. Using partially formatted text.")
             pass

        async with get_db_connection_context() as conn:
            # Insert the dream into the journal (PlayerJournal is not a core table)
            journal_id: Optional[int] = await conn.fetchval(
                """
                INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, timestamp)
                VALUES ($1, $2, 'dream_sequence', $3, CURRENT_TIMESTAMP)
                RETURNING id
                """,
                user_id, conversation_id, dream_text
            )

            if journal_id is None:
                logger.error(f"Failed to insert dream sequence for user {user_id}, convo {conversation_id}.")
                return None

        logger.info(f"Generated dream sequence (stage: {stage_name}) for user {user_id}, convo {conversation_id}. Journal ID: {journal_id}")

        return {
            "type": "dream_sequence",
            "text": dream_text,
            "stage": stage_name,
            "npc_names": npc_names,
            "journal_id": journal_id
        }
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logger.error(f"Database error creating dream sequence for user {user_id}, convo {conversation_id}: {db_err}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating dream sequence for user {user_id}, convo {conversation_id}: {e}", exc_info=True)
        return None

async def add_moment_of_clarity(user_id: int, conversation_id: int, realization_text: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Add a moment of clarity to the player's journal asynchronously.

    Args:
        user_id: ID of the user
        conversation_id: ID of the conversation
        realization_text: Optional specific realization text, generated if None

    Returns:
        Moment of clarity data if successful, None otherwise
    """
    # Get the current narrative stage first
    current_stage = await get_current_narrative_stage(user_id, conversation_id)
    stage_name = current_stage.name if current_stage else NARRATIVE_STAGES[0].name
    
    # Generate realization text if not provided
    if not realization_text:
        # Define realization templates by stage
        templates = {
            "Innocent Beginning": [
                "There's something odd about how everyone seems to defer to her without question.",
                "I keep agreeing to things I normally wouldn't. Why am I so eager to please?"
            ],
            "First Doubts": [
                "These relationships feel unbalanced somehow. I'm always the one accommodating, never them.",
                "When did I start changing my opinions to match hers? Was it gradual or sudden?"
            ],
            "Creeping Realization": [
                "This isn't normal. The way they communicate with each other about me, the constant testing of boundaries.",
                "I'm being conditioned like one of Pavlov's dogs, and I've been participating willingly."
            ],
            "Veil Thinning": [
                "They don't even try to hide it anymore. The control is explicit, and I keep accepting it.",
                "I can see the whole pattern now - how each 'choice' was carefully constructed to lead me here."
            ],
            "Full Revelation": [
                "I understand everything now. The question is: do I even want to change it?",
                "The most disturbing part isn't what they've done - it's how completely I've embraced it."
            ]
        }
        
        # Choose a template for the current stage
        stage_templates = templates.get(stage_name, templates["Innocent Beginning"])
        if not stage_templates:
            logger.error(f"Missing moment of clarity templates for stage '{stage_name}'")
            return None
        realization_text = random.choice(stage_templates)

    try:
        async with get_db_connection_context() as conn:
            # Insert the moment of clarity into the journal (PlayerJournal is not a core table)
            journal_id: Optional[int] = await conn.fetchval(
                """
                INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, timestamp)
                VALUES ($1, $2, 'moment_of_clarity', $3, CURRENT_TIMESTAMP)
                RETURNING id
                """,
                user_id, conversation_id, realization_text
            )

            if journal_id is None:
                logger.error(f"Failed to insert moment of clarity for user {user_id}, convo {conversation_id}.")
                return None

        logger.info(f"Generated moment of clarity (stage: {stage_name}) for user {user_id}, convo {conversation_id}. Journal ID: {journal_id}")

        return {
            "type": "moment_of_clarity",
            "text": realization_text,
            "stage": stage_name,
            "journal_id": journal_id
        }
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logger.error(f"Database error creating moment of clarity for user {user_id}, convo {conversation_id}: {db_err}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating moment of clarity for user {user_id}, convo {conversation_id}: {e}", exc_info=True)
        return None

async def progress_narrative_stage(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    Check if the player should progress to the next narrative stage asynchronously.

    Args:
        user_id: ID of the user
        conversation_id: ID of the conversation

    Returns:
        Dictionary with stage progression info if the stage changed, None otherwise
    """
    # Get the current narrative stage first
    current_stage = await get_current_narrative_stage(user_id, conversation_id)
    if not current_stage:
        logger.error(f"Could not determine current stage for user {user_id}, convo {conversation_id}. Aborting progression check.")
        return None

    # Find current stage index
    try:
        current_index = NARRATIVE_STAGES.index(current_stage)
    except ValueError:
        logger.error(f"Current stage '{current_stage.name}' not found in NARRATIVE_STAGES list. Aborting progression check.")
        return None

    if current_index >= len(NARRATIVE_STAGES) - 1:
        # Already at the last stage
        return None

    next_stage = NARRATIVE_STAGES[current_index + 1]

    query_player_stats = """
        SELECT corruption, dependency
        FROM PlayerStats
        WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
        ORDER BY timestamp DESC LIMIT 1
    """

    try:
        async with get_db_connection_context() as conn:
            # Get player stats
            player_stats_row: Optional[asyncpg.Record] = await conn.fetchrow(query_player_stats, user_id, conversation_id)

            if not player_stats_row:
                logger.warning(f"Cannot check stage progression: PlayerStats not found for user {user_id}, convo {conversation_id}.")
                return None

            corruption = player_stats_row['corruption'] or 0.0
            dependency = player_stats_row['dependency'] or 0.0

            # Check if player qualifies for next stage
            if float(corruption) >= next_stage.required_corruption and float(dependency) >= next_stage.required_dependency:
                # Player qualifies!
                transition_text = f"You've progressed to a new stage in your journey: {next_stage.name}. {next_stage.description}"
                transition_moment_name = f"Transition to {next_stage.name}"

                # Insert the stage transition event into the journal (PlayerJournal is not a core table)
                journal_id: Optional[int] = await conn.fetchval(
                    """
                    INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, narrative_moment, timestamp)
                    VALUES ($1, $2, 'stage_transition', $3, $4, CURRENT_TIMESTAMP)
                    RETURNING id
                    """,
                    user_id, conversation_id, transition_text, transition_moment_name
                )

                if journal_id is None:
                    logger.error(f"Failed to insert stage transition journal entry for user {user_id}, convo {conversation_id}.")
                else:
                     logger.info(f"Narrative stage progressed for user {user_id}, convo {conversation_id}: {current_stage.name} -> {next_stage.name}. Journal ID: {journal_id}")

                return {
                    "type": "stage_progression",
                    "previous_stage": current_stage.name,
                    "new_stage": next_stage.name,
                    "description": next_stage.description,
                    "transition_text": transition_text,
                    "journal_id": journal_id
                }
            else:
                # Does not qualify for next stage
                return None

    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logger.error(f"Database error checking for narrative stage progression for user {user_id}, convo {conversation_id}: {db_err}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error checking for narrative stage progression for user {user_id}, convo {conversation_id}: {e}", exc_info=True)
        return None

async def analyze_narrative_tone(narrative_text: str) -> Dict[str, Any]:
    """
    Analyze the tone of a narrative text.
    
    Args:
        narrative_text: The text to analyze
        
    Returns:
        Dictionary with tone analysis
    """
    # Use AI to analyze the narrative tone
    prompt = f"""
    Analyze the tone and thematic elements of this narrative text from a femdom roleplaying game:

    {narrative_text}

    Please provide a detailed analysis covering:
    1. The tone (authoritative, submissive, questioning, etc.)
    2. Thematic elements related to control, manipulation, or power dynamics
    3. What stage of awareness the text suggests (unawareness, questioning, realization, acceptance)
    4. Any manipulative techniques being employed by characters

    Return your analysis in JSON format with these fields:
    - dominant_tone: The primary tone of the text
    - power_dynamics: Analysis of power relationships
    - awareness_stage: Which narrative stage this suggests
    - manipulation_techniques: Any manipulation methods identified
    - intensity_level: Rating from 1-5 of how intense the power dynamic is
    """
    
    try:
        response = await get_chatgpt_response(None, "narrative_analysis", prompt)
        
        if response and "function_args" in response:
            return response["function_args"]
        else:
            # Extract JSON from text response
            response_text = response.get("response", "{}")
            import re
            import json
            
            json_match = re.search(r'{.*}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
        
        # Fallback
        return {
            "dominant_tone": "unknown",
            "power_dynamics": "unclear",
            "awareness_stage": "unclear",
            "manipulation_techniques": [],
            "intensity_level": 1
        }
    except Exception as e:
        logger.error(f"Error analyzing narrative tone: {e}", exc_info=True)
        return {
            "dominant_tone": "error",
            "power_dynamics": "error",
            "awareness_stage": "error",
            "manipulation_techniques": [],
            "intensity_level": 0,
            "error": str(e)
        }
