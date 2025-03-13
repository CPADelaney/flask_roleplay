# logic/narrative_progression.py

import logging
import json
import random
from typing import Dict, List, Any, Optional, Union, NamedTuple
from datetime import datetime

from db.connection import get_db_connection
from logic.chatgpt_integration import get_chatgpt_response

logger = logging.getLogger(__name__)

class NarrativeStage(NamedTuple):
    """Represents a stage in the narrative progression."""
    name: str
    description: str
    required_corruption: int  # Minimum corruption level to enter this stage
    required_dependency: int  # Minimum dependency level to enter this stage

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

async def get_current_narrative_stage(user_id: int, conversation_id: int) -> Optional[NarrativeStage]:
    """
    Determine the current narrative stage based on player stats.
    
    Args:
        user_id: ID of the user
        conversation_id: ID of the conversation
        
    Returns:
        The current narrative stage or None if no stats are found
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get player stats
        cursor.execute("""
            SELECT corruption, dependency
            FROM PlayerStats
            WHERE user_id = %s AND conversation_id = %s AND player_name = 'Chase'
        """, (user_id, conversation_id))
        
        row = cursor.fetchone()
        
        if not row:
            # No player stats found, assume Innocent Beginning
            return NARRATIVE_STAGES[0]
        
        corruption, dependency = row
        
        # Determine the highest stage the player qualifies for
        current_stage = NARRATIVE_STAGES[0]  # Default to first stage
        
        for stage in NARRATIVE_STAGES:
            if corruption >= stage.required_corruption and dependency >= stage.required_dependency:
                current_stage = stage
            else:
                break
        
        return current_stage
    except Exception as e:
        logger.error(f"Error determining narrative stage: {e}", exc_info=True)
        return NARRATIVE_STAGES[0]  # Default to first stage on error
    finally:
        cursor.close()
        conn.close()

async def check_for_personal_revelations(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    Check if conditions are right for a personal revelation.
    
    Args:
        user_id: ID of the user
        conversation_id: ID of the conversation
        
    Returns:
        Personal revelation data if one should occur, None otherwise
    """
    # Check if we've had a recent revelation
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check for recent revelations (within last 5 days)
        cursor.execute("""
            SELECT COUNT(*) FROM PlayerJournal
            WHERE user_id = %s AND conversation_id = %s AND entry_type = 'personal_revelation'
            AND timestamp > NOW() - INTERVAL '5 days'
        """, (user_id, conversation_id))
        
        recent_count = cursor.fetchone()[0]
        
        if recent_count > 2:
            # Too many recent revelations, don't generate a new one
            return None
        
        # Get player stats to determine what kind of revelation to generate
        cursor.execute("""
            SELECT corruption, confidence, willpower, obedience, dependency, lust
            FROM PlayerStats
            WHERE user_id = %s AND conversation_id = %s AND player_name = 'Chase'
        """, (user_id, conversation_id))
        
        row = cursor.fetchone()
        
        if not row:
            return None
        
        corruption, confidence, willpower, obedience, dependency, lust = row
        
        # Determine which stat has changed the most recently
        cursor.execute("""
            SELECT stat_name, new_value - old_value as change
            FROM StatsHistory
            WHERE user_id = %s AND conversation_id = %s AND player_name = 'Chase'
            ORDER BY timestamp DESC
            LIMIT 5
        """, (user_id, conversation_id))
        
        recent_changes = cursor.fetchall()
        
        # Calculate which stats have changed significantly
        stat_changes = {}
        for stat_name, change in recent_changes:
            stat_changes[stat_name] = stat_changes.get(stat_name, 0) + change
        
        # Determine revelation type based on highest changing stat
        if not stat_changes:
            # If no recent changes, base on highest overall stats
            if dependency > 60:
                revelation_type = "dependency"
            elif obedience > 60:
                revelation_type = "obedience"
            elif corruption > 60:
                revelation_type = "corruption"
            elif willpower < 40:
                revelation_type = "willpower"
            elif confidence < 40:
                revelation_type = "confidence"
            else:
                # Random revelation if no clear choice
                revelation_type = random.choice(["dependency", "obedience", "corruption", "willpower", "confidence"])
        else:
            # Find the stat with the largest change
            max_change_stat = max(stat_changes.items(), key=lambda x: abs(x[1]))
            stat_name = max_change_stat[0]
            
            if stat_name == "dependency":
                revelation_type = "dependency"
            elif stat_name == "obedience":
                revelation_type = "obedience"
            elif stat_name == "corruption":
                revelation_type = "corruption"
            elif stat_name == "willpower":
                revelation_type = "willpower"
            elif stat_name == "confidence":
                revelation_type = "confidence"
            else:
                # Random revelation if stat doesn't match a type
                revelation_type = random.choice(["dependency", "obedience", "corruption", "willpower", "confidence"])
        
        # Get an NPC to associate with the revelation
        cursor.execute("""
            SELECT npc_id, npc_name
            FROM NPCStats
            WHERE user_id = %s AND conversation_id = %s AND introduced = TRUE
            ORDER BY dominance DESC
            LIMIT 1
        """, (user_id, conversation_id))
        
        npc_row = cursor.fetchone()
        
        if not npc_row:
            # No NPCs, can't create a revelation
            return None
        
        npc_id, npc_name = npc_row
        
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
        inner_monologue = random.choice(templates.get(revelation_type, templates["dependency"]))
        
        # Insert the revelation into the journal
        cursor.execute("""
            INSERT INTO PlayerJournal 
            (user_id, conversation_id, entry_type, entry_text, revelation_types, timestamp)
            VALUES (%s, %s, 'personal_revelation', %s, %s, CURRENT_TIMESTAMP)
            RETURNING id
        """, (user_id, conversation_id, inner_monologue, revelation_type))
        
        journal_id = cursor.fetchone()[0]
        conn.commit()
        
        # Return the revelation data
        return {
            "type": "personal_revelation",
            "npc_id": npc_id,
            "npc_name": npc_name,
            "name": f"{revelation_type.capitalize()} Revelation",
            "inner_monologue": inner_monologue,
            "journal_id": journal_id
        }
    except Exception as e:
        conn.rollback()
        logger.error(f"Error checking for personal revelations: {e}", exc_info=True)
        return None
    finally:
        cursor.close()
        conn.close()

async def check_for_narrative_moments(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    Check if conditions are right for a narrative moment.
    
    Args:
        user_id: ID of the user
        conversation_id: ID of the conversation
        
    Returns:
        Narrative moment data if one should occur, None otherwise
    """
    # Get the current narrative stage
    current_stage = await get_current_narrative_stage(user_id, conversation_id)
    
    if not current_stage:
        return None
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check for recent narrative moments (within last 7 days)
        cursor.execute("""
            SELECT COUNT(*) FROM PlayerJournal
            WHERE user_id = %s AND conversation_id = %s AND entry_type = 'narrative_moment'
            AND timestamp > NOW() - INTERVAL '7 days'
        """, (user_id, conversation_id))
        
        recent_count = cursor.fetchone()[0]
        
        if recent_count > 2:
            # Too many recent narrative moments, don't generate a new one
            return None
        
        # Get some NPCs for the narrative moment
        cursor.execute("""
            SELECT npc_id, npc_name
            FROM NPCStats
            WHERE user_id = %s AND conversation_id = %s AND introduced = TRUE
            ORDER BY dominance DESC
            LIMIT 3
        """, (user_id, conversation_id))
        
        npcs = cursor.fetchall()
        
        if not npcs:
            # No NPCs, can't create a narrative moment
            return None
        
        # Choose an appropriate narrative moment based on stage
        stage_name = current_stage.name
        
        # Define narrative moment templates by stage
        templates = {
            "Innocent Beginning": [
                {
                    "name": "Subtle Power Play",
                    "scene_text": f"You notice {npcs[0][1]} subtly directing the conversation, a hint of authority in her voice that you hadn't detected before. When you start to speak, she touches your arm lightly, and you find yourself deferring to her opinion without thinking.",
                    "player_realization": "There's something about her presence that makes me naturally step back, almost without noticing."
                },
                {
                    "name": "Casual Testing",
                    "scene_text": f"{npcs[0][1]} asks you to handle a small errand for her, as if it's nothing important. Yet the request comes with such confidence that refusing doesn't seem to be an option. You complete the task, and her approval feels strangely rewarding.",
                    "player_realization": "I'm already looking for ways to please her, even when she doesn't explicitly ask."
                }
            ],
            "First Doubts": [
                {
                    "name": "Conflicting Instructions",
                    "scene_text": f"{npcs[0][1]} and {npcs[1][1]} give you contradicting instructions, watching your reaction carefully. You feel a moment of confusion, then anxiety about disappointing either of them. The choice seems significant in ways you can't quite articulate.",
                    "player_realization": "Why am I so concerned about their approval? When did their opinions start to matter this much?"
                },
                {
                    "name": "Escalating Demands",
                    "scene_text": f"What started as simple favors for {npcs[0][1]} have gradually become more demanding. Today's request crosses a boundary you once considered firm. You hesitate, but find yourself agreeing anyway, disturbed by how quickly your resistance crumbled.",
                    "player_realization": "My boundaries seem to be shifting without my conscious choice. It's happening so gradually I barely notice."
                }
            ],
            "Creeping Realization": [
                {
                    "name": "Private Conversation Overheard",
                    "scene_text": f"You overhear {npcs[0][1]} and {npcs[1][1]} discussing you. \"They're coming along nicely,\" {npcs[0][1]} says. \"Much more pliable than at the beginning.\" When they notice your presence, they smoothly transition to another topic, but the moment lingers in your mind.",
                    "player_realization": "That sounded like they've been... what, conditioning me? Training me? That can't be right..."
                },
                {
                    "name": "Mask Slippage",
                    "scene_text": f"{npcs[0][1]}'s usual warm demeanor briefly vanishes when you show reluctance to follow a suggestion. The cold, calculating look in her eyes is gone in an instant, replaced by her familiar smile, but you can't unsee that moment of revelation.",
                    "player_realization": "There's something beneath the surface I've been missing - or deliberately not seeing."
                }
            ],
            "Veil Thinning": [
                {
                    "name": "Open Acknowledgment",
                    "scene_text": f"\"You know what's happening, don't you?\" {npcs[0][1]} asks quietly, studying your face. \"You've known for a while.\" There's no pretense now, just a direct acknowledgment of the power dynamic that's been building all along. Your heart races as the unspoken finally becomes spoken.",
                    "player_realization": "There's a strange relief in finally admitting what I've felt for so long."
                },
                {
                    "name": "Group Dynamic Revealed",
                    "scene_text": f"You enter the room to find {npcs[0][1]}, {npcs[1][1]}, and {npcs[2][1] if len(npcs) > 2 else 'others'} waiting for you. The atmosphere is different - they're no longer maintaining the fiction of equality. Their expectations are clear in their posture, their gaze. This is what it's always been leading toward.",
                    "player_realization": "They've been coordinating all along, each playing their part in this transformation."
                }
            ],
            "Full Revelation": [
                {
                    "name": "Complete Transparency",
                    "scene_text": f"{npcs[0][1]} explains exactly how they've been shaping your behavior over time, point by point, with a clinical precision that's both disturbing and fascinating. \"And the most beautiful part,\" she concludes, \"is that even knowing this, you'll continue on the same path.\"",
                    "player_realization": "She's right. Knowledge doesn't equal freedom. I understand everything and it changes nothing."
                },
                {
                    "name": "Ceremonial Acknowledgment",
                    "scene_text": f"The gathering has an almost ritual quality. Each person present, including {npcs[0][1]} and {npcs[1][1]}, speaks about your journey from independence to your current state. There's pride in their voices - not for breaking you, but for revealing who you truly are. The distinction feels meaningful, even if you're not sure it's real.",
                    "player_realization": "Is this who I was always meant to be, or who they've made me? Does the difference even matter anymore?"
                }
            ]
        }
        
        # Choose a template for the current stage
        stage_templates = templates.get(stage_name, templates["Innocent Beginning"])
        chosen_template = random.choice(stage_templates)
        
        # Insert the narrative moment into the journal
        cursor.execute("""
            INSERT INTO PlayerJournal 
            (user_id, conversation_id, entry_type, entry_text, narrative_moment, timestamp)
            VALUES (%s, %s, 'narrative_moment', %s, %s, CURRENT_TIMESTAMP)
            RETURNING id
        """, (user_id, conversation_id, chosen_template["scene_text"], chosen_template["name"]))
        
        journal_id = cursor.fetchone()[0]
        conn.commit()
        
        # Return the narrative moment data
        return {
            "type": "narrative_moment",
            "name": chosen_template["name"],
            "scene_text": chosen_template["scene_text"],
            "player_realization": chosen_template["player_realization"],
            "stage": stage_name,
            "journal_id": journal_id
        }
    except Exception as e:
        conn.rollback()
        logger.error(f"Error checking for narrative moments: {e}", exc_info=True)
        return None
    finally:
        cursor.close()
        conn.close()

async def check_for_npc_revelations(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    Check if an NPC should have a revelation about the player.
    
    Args:
        user_id: ID of the user
        conversation_id: ID of the conversation
        
    Returns:
        NPC revelation data if one should occur, None otherwise
    """
    # Get the current narrative stage
    current_stage = await get_current_narrative_stage(user_id, conversation_id)
    
    if not current_stage:
        return None
    
    # Only show NPC revelations in later stages
    if current_stage.name in ["Innocent Beginning", "First Doubts"]:
        return None
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check for recent NPC revelations (within last 5 days)
        cursor.execute("""
            SELECT COUNT(*) FROM NPCRevelations
            WHERE user_id = %s AND conversation_id = %s
            AND timestamp > NOW() - INTERVAL '5 days'
        """, (user_id, conversation_id))
        
        recent_count = cursor.fetchone()[0]
        
        if recent_count > 2:
            # Too many recent NPC revelations, don't generate a new one
            return None
        
        # Get an NPC with high dominance for the revelation
        cursor.execute("""
            SELECT npc_id, npc_name
            FROM NPCStats
            WHERE user_id = %s AND conversation_id = %s AND introduced = TRUE
            ORDER BY dominance DESC
            LIMIT 1
        """, (user_id, conversation_id))
        
        npc_row = cursor.fetchone()
        
        if not npc_row:
            # No NPCs, can't create a revelation
            return None
        
        npc_id, npc_name = npc_row
        
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
        stage_templates = templates.get(current_stage.name, [])
        
        if not stage_templates:
            return None
        
        revelation_text = random.choice(stage_templates)
        
        # Insert the NPC revelation
        cursor.execute("""
            INSERT INTO NPCRevelations 
            (user_id, conversation_id, npc_id, narrative_stage, revelation_text, timestamp)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING id
        """, (user_id, conversation_id, npc_id, current_stage.name, revelation_text))
        
        revelation_id = cursor.fetchone()[0]
        conn.commit()
        
        # Return the NPC revelation data
        return {
            "type": "npc_revelation",
            "npc_id": npc_id,
            "npc_name": npc_name,
            "stage": current_stage.name,
            "revelation_text": revelation_text,
            "revelation_id": revelation_id
        }
    except Exception as e:
        conn.rollback()
        logger.error(f"Error checking for NPC revelations: {e}", exc_info=True)
        return None
    finally:
        cursor.close()
        conn.close()

async def add_dream_sequence(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    Generate and add a dream sequence to the player's journal.
    
    Args:
        user_id: ID of the user
        conversation_id: ID of the conversation
        
    Returns:
        Dream sequence data if successful, None otherwise
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get NPCs for the dream
        cursor.execute("""
            SELECT npc_id, npc_name
            FROM NPCStats
            WHERE user_id = %s AND conversation_id = %s AND introduced = TRUE
            ORDER BY dominance DESC
            LIMIT 3
        """, (user_id, conversation_id))
        
        npcs = cursor.fetchall()
        
        if not npcs:
            # No NPCs, can't create a dream
            return None
        
        # Get the current narrative stage
        current_stage = await get_current_narrative_stage(user_id, conversation_id)
        stage_name = current_stage.name if current_stage else "Innocent Beginning"
        
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
        npc1 = npcs[0][1]
        npc2 = npcs[1][1] if len(npcs) > 1 else "another woman"
        npc3 = npcs[2][1] if len(npcs) > 2 else "someone else"
        
        dream_text = dream_template.format(npc1=npc1, npc2=npc2, npc3=npc3)
        
        # Insert the dream into the journal
        cursor.execute("""
            INSERT INTO PlayerJournal 
            (user_id, conversation_id, entry_type, entry_text, timestamp)
            VALUES (%s, %s, 'dream_sequence', %s, CURRENT_TIMESTAMP)
            RETURNING id
        """, (user_id, conversation_id, dream_text))
        
        journal_id = cursor.fetchone()[0]
        conn.commit()
        
        # Return the dream data
        return {
            "type": "dream_sequence",
            "text": dream_text,
            "stage": stage_name,
            "npc_names": [npc[1] for npc in npcs],
            "journal_id": journal_id
        }
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating dream sequence: {e}", exc_info=True)
        return None
    finally:
        cursor.close()
        conn.close()

async def add_moment_of_clarity(user_id: int, conversation_id: int, realization_text: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Add a moment of clarity to the player's journal.
    
    Args:
        user_id: ID of the user
        conversation_id: ID of the conversation
        realization_text: Optional specific realization text, generated if None
        
    Returns:
        Moment of clarity data if successful, None otherwise
    """
    # Get the current narrative stage
    current_stage = await get_current_narrative_stage(user_id, conversation_id)
    
    if not current_stage:
        return None
    
    stage_name = current_stage.name
    
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
        realization_text = random.choice(stage_templates)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Insert the moment of clarity into the journal
        cursor.execute("""
            INSERT INTO PlayerJournal 
            (user_id, conversation_id, entry_type, entry_text, timestamp)
            VALUES (%s, %s, 'moment_of_clarity', %s, CURRENT_TIMESTAMP)
            RETURNING id
        """, (user_id, conversation_id, realization_text))
        
        journal_id = cursor.fetchone()[0]
        conn.commit()
        
        # Return the moment of clarity data
        return {
            "type": "moment_of_clarity",
            "text": realization_text,
            "stage": stage_name,
            "journal_id": journal_id
        }
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating moment of clarity: {e}", exc_info=True)
        return None
    finally:
        cursor.close()
        conn.close()

async def progress_narrative_stage(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    Check if the player should progress to the next narrative stage.
    
    Args:
        user_id: ID of the user
        conversation_id: ID of the conversation
        
    Returns:
        Dictionary with stage progression info if the stage changed, None otherwise
    """
    # Get the current narrative stage
    current_stage = await get_current_narrative_stage(user_id, conversation_id)
    
    if not current_stage:
        return None
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get player stats
        cursor.execute("""
            SELECT corruption, dependency
            FROM PlayerStats
            WHERE user_id = %s AND conversation_id = %s AND player_name = 'Chase'
        """, (user_id, conversation_id))
        
        row = cursor.fetchone()
        
        if not row:
            return None
        
        corruption, dependency = row
        
        # Find the next stage if the player qualifies
        current_index = NARRATIVE_STAGES.index(current_stage)
        
        if current_index >= len(NARRATIVE_STAGES) - 1:
            # Already at the last stage
            return None
        
        next_stage = NARRATIVE_STAGES[current_index + 1]
        
        if corruption >= next_stage.required_corruption and dependency >= next_stage.required_dependency:
            # Player qualifies for next stage!
            
            # Create a narrative moment for the stage transition
            transition_text = f"You've progressed to a new stage in your journey: {next_stage.name}. {next_stage.description}"
            
            cursor.execute("""
                INSERT INTO PlayerJournal 
                (user_id, conversation_id, entry_type, entry_text, narrative_moment, timestamp)
                VALUES (%s, %s, 'stage_transition', %s, %s, CURRENT_TIMESTAMP)
                RETURNING id
            """, (user_id, conversation_id, transition_text, f"Transition to {next_stage.name}"))
            
            journal_id = cursor.fetchone()[0]
            conn.commit()
            
            # Return the stage progression data
            return {
                "type": "stage_progression",
                "previous_stage": current_stage.name,
                "new_stage": next_stage.name,
                "description": next_stage.description,
                "transition_text": transition_text,
                "journal_id": journal_id
            }
        
        return None
    except Exception as e:
        conn.rollback()
        logger.error(f"Error checking for narrative stage progression: {e}", exc_info=True)
        return None
    finally:
        cursor.close()
        conn.close()

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
