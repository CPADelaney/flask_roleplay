# logic/narrative_progression.py

import random
import json
import logging
from datetime import datetime
from db.connection import get_db_connection

class NarrativeStage:
    """Defines a stage in the narrative progression"""
    def __init__(self, name, description, requirements, revelation_texts):
        self.name = name
        self.description = description
        self.requirements = requirements  # Dict of stat requirements
        self.revelation_texts = revelation_texts  # Potential revelation texts for this stage

# Define the stages of narrative progression
NARRATIVE_STAGES = [
    NarrativeStage(
        name="Innocent Beginning",
        description="The world appears normal, with only subtle hints of control dynamics",
        requirements={"corruption": (0, 20), "obedience": (0, 30), "dependency": (0, 20)},
        revelation_texts=[
            "You notice {npc_name} watching you with an unusual intensity when she thinks you aren't looking.",
            "Something about the way {npc_name} phrases her 'suggestions' makes you pause, though you can't quite place why.",
            "You overhear {npc_name} speaking to someone else in a tone that sounds nothing like how she speaks to you."
        ]
    ),
    NarrativeStage(
        name="First Doubts",
        description="Player begins to notice inconsistencies in seemingly innocent interactions",
        requirements={"corruption": (20, 40), "obedience": (30, 50), "dependency": (20, 40)},
        revelation_texts=[
            "You realize that {npc_name} has been subtly directing your choices for weeks, always making it seem like your idea.",
            "The praise {npc_name} gives you feels different now - less about your accomplishments and more about your compliance.",
            "You catch {npc_name} exchanging a knowing glance with {other_npc} after you agree to do something uncomfortable."
        ]
    ),
    NarrativeStage(
        name="Creeping Realization",
        description="Player recognizes manipulation but still believes they have agency",
        requirements={"corruption": (40, 60), "obedience": (50, 70), "dependency": (40, 60)},
        revelation_texts=[
            "As {npc_name} gives you instructions, you suddenly realize how naturally you've begun to follow her commands without question.",
            "You find yourself checking with {npc_name} before making decisions that used to be automatic for you.",
            "The thought of disappointing {npc_name} now causes you physical anxiety - when did that start happening?"
        ]
    ),
    NarrativeStage(
        name="Veil Thinning",
        description="Dominant characters begin dropping their pretense more frequently",
        requirements={"corruption": (60, 80), "obedience": (70, 85), "dependency": (60, 80)},
        revelation_texts=[
            "{npc_name} corrects your behavior in public with a tone that makes it clear she expects immediate compliance. You provide it without thinking.",
            "You enter a room to find {npc_name} and {other_npc} discussing you openly, only slightly altering their conversation when they notice you.",
            "'You're making progress,' {npc_name} says with a smile that suddenly seems predatory. The comment feels loaded with meaning."
        ]
    ),
    NarrativeStage(
        name="Full Revelation",
        description="The true nature of relationships is now explicit and undeniable",
        requirements={"corruption": (80, 100), "obedience": (85, 100), "dependency": (80, 100)},
        revelation_texts=[
            "{npc_name} no longer bothers hiding her control, openly giving you orders and expecting immediate obedience - which you provide without hesitation.",
            "'You've been ours for quite some time,' {npc_name} explains calmly, 'but now you're ready to acknowledge it consciously.'",
            "You realize with startling clarity that your entire social circle has been carefully curated and controlled, a web of influence with you at the center."
        ]
    )
]

# Define personal revelation types that affect the player's inner monologue
PERSONAL_REVELATION_TYPES = [
    {
        "name": "Dependency Awareness",
        "triggers": {"dependency": 50},
        "inner_monologue": [
            "I've been checking my phone constantly to see if {npc_name} has messaged me. When did I start needing her approval so much?",
            "I realized today that I haven't made a significant decision without consulting {npc_name} in weeks. Is that normal?",
            "The thought of spending a day without talking to {npc_name} makes me anxious. I should be concerned about that, shouldn't I?"
        ]
    },
    {
        "name": "Obedience Recognition",
        "triggers": {"obedience": 60},
        "inner_monologue": [
            "I caught myself automatically rearranging my schedule when {npc_name} hinted she wanted to see me. I didn't even think twice about it.",
            "Today I changed my opinion the moment I realized it differed from {npc_name}'s. That's... not like me. Or is it becoming like me?",
            "{npc_name} gave me that look, and I immediately stopped what I was saying. When did her disapproval start carrying so much weight?"
        ]
    },
    {
        "name": "Corruption Awareness",
        "triggers": {"corruption": 70},
        "inner_monologue": [
            "I found myself enjoying the feeling of following {npc_name}'s instructions perfectly. The pride I felt at her approval was... intense.",
            "Last year, I would have been offended if someone treated me the way {npc_name} did today. Now I'm grateful for her attention.",
            "Sometimes I catch glimpses of my old self, like a stranger I used to know. When did I change so fundamentally?"
        ]
    },
    {
        "name": "Willpower Deterioration",
        "triggers": {"willpower": -50},  # Negative means below this value
        "inner_monologue": [
            "I had every intention of saying no to {npc_name} today. The 'yes' came out before I even realized I was speaking.",
            "I've been trying to remember what it felt like to disagree with {npc_name}. The memory feels distant, like it belongs to someone else.",
            "I made a list of boundaries I wouldn't cross. Looking at it now, I've broken every single one at {npc_name}'s suggestion."
        ]
    },
    {
        "name": "Confidence Collapse",
        "triggers": {"confidence": -40},
        "inner_monologue": [
            "I opened my mouth to speak in the meeting, then saw {npc_name} watching me. I suddenly couldn't remember what I was going to say.",
            "I used to trust my judgment. Now I find myself second-guessing every thought that {npc_name} hasn't explicitly approved.",
            "When did I start feeling this small? This uncertain? I can barely remember how it felt to be sure of myself."
        ]
    }
]

# Key narrative moments that trigger significant story progression
NARRATIVE_MOMENTS = [
    {
        "name": "First Command",
        "description": "First time an NPC gives a direct command without disguising it",
        "requirements": {"npc_dominance": 60, "player_obedience": 40},
        "scene_text": "{npc_name} looks you directly in the eyes, tone shifting to something unmistakably commanding: \"{command_text}\" The silence that follows makes it clear this isn't a request.",
        "command_variations": [
            "Sit down.",
            "Give that to me.",
            "Wait here until I return.",
            "Tell me what you're thinking. Now.",
            "Stop what you're doing and come with me."
        ],
        "player_realization": "That wasn't a suggestion or request. That was a command... and I followed it without hesitation."
    },
    {
        "name": "Overheard Conversation",
        "description": "Player overhears NPCs discussing their manipulation",
        "requirements": {"player_corruption": 50, "npc_introduced_count": 3},
        "scene_text": "You freeze as you overhear {npc_name} speaking to {other_npc} around the corner: \"{conversation_snippet}\" They haven't noticed you yet.",
        "conversation_snippets": [
            "...more receptive to suggestion than I expected. The techniques you recommended are working perfectly.",
            "...barely questions anything anymore when I use that tone with them. It's almost too easy.",
            "...building the dependency gradually. They're already showing signs of anxiety when I'm not available.",
            "...wouldn't have believed how quickly they've adapted to the new expectations. They're practically anticipating what I want now."
        ],
        "player_realization": "They're talking about me. About... manipulating me. How long has this been happening?"
    },
    {
        "name": "Group Dynamic Revelation",
        "description": "Player realizes they're the only one being treated differently",
        "requirements": {"player_mental_resilience": -60, "social_links_count": 4},
        "scene_text": "The room shifts as {npc_name} enters. You watch in dawning realization as everyone's behavior subtly changes - postures straighten, voices lower, eyes defer... except how they interact with you. With you, there's a permissiveness, an indulgence, like you're being... handled.",
        "player_realization": "Everyone else knows something I don't. There's an understanding here, a hierarchy I'm only just beginning to see."
    },
    {
        "name": "Direct Confrontation",
        "description": "Player directly questions what's happening and receives honesty",
        "requirements": {"player_corruption": 70, "player_willpower": 40, "npc_dominance": 80},
        "scene_text": "\"What is this?\" you finally ask, frustration breaking through. \"What's been happening to me?\" {npc_name} studies you for a long moment, then smiles with unexpected openness. \"{honest_response}\"",
        "honest_responses": [
            "I was wondering when you'd notice. You're finally ready to acknowledge what you've wanted all along.",
            "We've been guiding you toward your true nature. The person you're becoming is who you were always meant to be.",
            "You've been an experiment in conditioning. And a remarkably successful one. That discomfort you feel? It's just your old self struggling against what you're becoming.",
            "You gave up your autonomy in inches, so gradually you never noticed. Now you're asking if the cage is real after you've been living in it for months."
        ],
        "player_realization": "Part of me wants to reject what she's saying... but another part recognizes the truth in her words. Have I been complicit in my own transformation?"
    },
    {
        "name": "Breaking Point Choice",
        "description": "Player faces a significant choice that will determine their path",
        "requirements": {"player_corruption": 85, "player_dependency": 80, "player_confidence": -50},
        "scene_text": "\"It's time to make a choice,\" {npc_name} says, voice gentle but unyielding. \"You can continue pretending you still have the same autonomy you did before, or you can embrace what you've become. What WE have become together.\" She extends her hand, waiting. \"{choice_text}\"",
        "choice_variations": [
            "Accept who you are now, or walk away and try to remember who you used to be.",
            "Take my hand and acknowledge what you've known for weeks, or leave and we'll see how long you last on your own.",
            "Stop fighting what you've already surrendered. Take your place willingly, or continue this exhausting resistance."
        ],
        "player_realization": "This is the moment where I decide who I truly am - or who I'm willing to become."
    }
]

def get_current_narrative_stage(user_id, conversation_id):
    """
    Determine which narrative stage the player is currently in based on their stats
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get current player stats
        cursor.execute("""
            SELECT corruption, obedience, dependency
            FROM PlayerStats
            WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
        """, (user_id, conversation_id))
        
        row = cursor.fetchone()
        if not row:
            return None
            
        corruption, obedience, dependency = row
        
        # Get current narrative stage from DB if it exists
        cursor.execute("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=%s AND conversation_id=%s AND key='CurrentNarrativeStage'
        """, (user_id, conversation_id))
        
        current_stage_row = cursor.fetchone()
        current_stage_name = current_stage_row[0] if current_stage_row else None
        
        # Find the highest stage the player meets requirements for
        appropriate_stage = None
        
        for stage in NARRATIVE_STAGES:
            # Check if player meets all requirements for this stage
            meets_requirements = True
            
            for stat, (min_val, max_val) in stage.requirements.items():
                stat_value = locals().get(stat, 0)
                if stat_value < min_val or stat_value > max_val:
                    meets_requirements = False
                    break
                    
            if meets_requirements:
                appropriate_stage = stage
        
        # If we found an appropriate stage and it's different from current, update it
        if appropriate_stage and appropriate_stage.name != current_stage_name:
            cursor.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES (%s, %s, 'CurrentNarrativeStage', %s)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = EXCLUDED.value
            """, (user_id, conversation_id, appropriate_stage.name))
            
            cursor.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES (%s, %s, 'NarrativeStageChanged', 'True')
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = EXCLUDED.value
            """, (user_id, conversation_id))
            
            conn.commit()
            
            # Also log this progression in PlayerJournal
            cursor.execute("""
                INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, timestamp)
                VALUES (%s, %s, 'narrative_progression', %s, CURRENT_TIMESTAMP)
            """, (user_id, conversation_id, f"Narrative has progressed to: {appropriate_stage.name} - {appropriate_stage.description}"))
            
            conn.commit()
        
        return appropriate_stage
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error determining narrative stage: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def check_for_personal_revelations(user_id, conversation_id):
    """
    Check if any personal revelations have been triggered by player stats
    Returns a revelation if triggered, None otherwise
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get current player stats
        cursor.execute("""
            SELECT corruption, confidence, willpower, obedience, dependency,
                   lust, mental_resilience, physical_endurance
            FROM PlayerStats
            WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
        """, (user_id, conversation_id))
        
        row = cursor.fetchone()
        if not row:
            return None
            
        player_stats = {
            "corruption": row[0],
            "confidence": row[1],
            "willpower": row[2],
            "obedience": row[3],
            "dependency": row[4],
            "lust": row[5],
            "mental_resilience": row[6],
            "physical_endurance": row[7]
        }
        
        # Get already experienced revelations
        cursor.execute("""
            SELECT revelation_types FROM PlayerJournal
            WHERE user_id=%s AND conversation_id=%s AND entry_type='personal_revelation'
        """, (user_id, conversation_id))
        
        experienced_revelations = set()
        for row in cursor.fetchall():
            if row[0]:
                experienced_revelations.add(row[0])
        
        # Check each revelation type
        eligible_revelations = []
        
        for revelation in PERSONAL_REVELATION_TYPES:
            # Skip if already experienced
            if revelation["name"] in experienced_revelations:
                continue
                
            # Check triggers
            meets_requirements = True
            
            for stat, threshold in revelation["triggers"].items():
                stat_value = player_stats.get(stat, 0)
                if threshold < 0:  # Negative threshold means "below this value"
                    if stat_value > abs(threshold):
                        meets_requirements = False
                        break
                else:  # Positive threshold means "above this value"
                    if stat_value < threshold:
                        meets_requirements = False
                        break
                        
            if meets_requirements:
                eligible_revelations.append(revelation)
        
        # If any eligible revelations, select one randomly
        if eligible_revelations:
            chosen_revelation = random.choice(eligible_revelations)
            
            # Select random NPC for the monologue
            cursor.execute("""
                SELECT npc_id, npc_name FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s AND introduced=TRUE
                ORDER BY closeness DESC LIMIT 1
            """, (user_id, conversation_id))
            
            npc_row = cursor.fetchone()
            npc_name = npc_row[1] if npc_row else "someone"
            
            # Select and format inner monologue
            inner_monologue = random.choice(chosen_revelation["inner_monologue"]).format(npc_name=npc_name)
            
            # Record this revelation
            cursor.execute("""
                INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, revelation_types, entry_text, timestamp)
                VALUES (%s, %s, 'personal_revelation', %s, %s, CURRENT_TIMESTAMP)
            """, (user_id, conversation_id, chosen_revelation["name"], inner_monologue))
            
            conn.commit()
            
            return {
                "type": "personal_revelation",
                "name": chosen_revelation["name"],
                "inner_monologue": inner_monologue
            }
        
        return None
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error checking for personal revelations: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def check_for_narrative_moments(user_id, conversation_id):
    """
    Check if any narrative moments have been triggered
    Returns a narrative moment if triggered, None otherwise
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get player stats
        cursor.execute("""
            SELECT corruption, confidence, willpower, obedience, dependency,
                   lust, mental_resilience, physical_endurance
            FROM PlayerStats
            WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
        """, (user_id, conversation_id))
        
        player_row = cursor.fetchone()
        if not player_row:
            return None
            
        player_stats = {
            "player_corruption": player_row[0],
            "player_confidence": player_row[1],
            "player_willpower": player_row[2],
            "player_obedience": player_row[3],
            "player_dependency": player_row[4],
            "player_lust": player_row[5],
            "player_mental_resilience": player_row[6],
            "player_physical_endurance": player_row[7]
        }
        
        # Get additional requirements data
        # Count introduced NPCs
        cursor.execute("""
            SELECT COUNT(*) FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND introduced=TRUE
        """, (user_id, conversation_id))
        
        npc_count_row = cursor.fetchone()
        npc_introduced_count = npc_count_row[0] if npc_count_row else 0
        
        # Count social links
        cursor.execute("""
            SELECT COUNT(*) FROM SocialLinks
            WHERE user_id=%s AND conversation_id=%s
        """, (user_id, conversation_id))
        
        social_links_row = cursor.fetchone()
        social_links_count = social_links_row[0] if social_links_row else 0
        
        # Find dominant NPC
        cursor.execute("""
            SELECT npc_id, npc_name, dominance FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND introduced=TRUE
            ORDER BY dominance DESC LIMIT 1
        """, (user_id, conversation_id))
        
        dominant_npc_row = cursor.fetchone()
        if dominant_npc_row:
            npc_dominance = dominant_npc_row[2]
        else:
            npc_dominance = 0
            
        # Combine all requirements data
        requirements_data = {
            **player_stats,
            "npc_introduced_count": npc_introduced_count,
            "social_links_count": social_links_count,
            "npc_dominance": npc_dominance
        }
        
        # Get already experienced narrative moments
        cursor.execute("""
            SELECT narrative_moment FROM PlayerJournal
            WHERE user_id=%s AND conversation_id=%s AND entry_type='narrative_moment'
        """, (user_id, conversation_id))
        
        experienced_moments = set()
        for row in cursor.fetchall():
            if row[0]:
                experienced_moments.add(row[0])
        
        # Check each narrative moment
        eligible_moments = []
        
        for moment in NARRATIVE_MOMENTS:
            # Skip if already experienced
            if moment["name"] in experienced_moments:
                continue
                
            # Check requirements
            meets_requirements = True
            
            for req_key, req_value in moment["requirements"].items():
                actual_value = requirements_data.get(req_key, 0)
                if actual_value < req_value:
                    meets_requirements = False
                    break
                    
            if meets_requirements:
                eligible_moments.append(moment)
        
        # If any eligible moments, select one randomly
        if eligible_moments:
            chosen_moment = random.choice(eligible_moments)
            
            # Format the scene text
            scene_text = chosen_moment["scene_text"]
            
            # Get NPC name
            npc_name = dominant_npc_row[1] if dominant_npc_row else "someone"
            
            # Get another NPC for scenes that need it
            other_npc = "another woman"
            cursor.execute("""
                SELECT npc_name FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s AND introduced=TRUE
                AND npc_id != %s
                ORDER BY RANDOM() LIMIT 1
            """, (user_id, conversation_id, dominant_npc_row[0] if dominant_npc_row else 0))
            
            other_npc_row = cursor.fetchone()
            if other_npc_row:
                other_npc = other_npc_row[0]
            
            # Select random variations
            if "command_variations" in chosen_moment:
                command_text = random.choice(chosen_moment["command_variations"])
                scene_text = scene_text.format(npc_name=npc_name, command_text=command_text)
            elif "conversation_snippets" in chosen_moment:
                conversation_snippet = random.choice(chosen_moment["conversation_snippets"])
                scene_text = scene_text.format(npc_name=npc_name, other_npc=other_npc, conversation_snippet=conversation_snippet)
            elif "honest_responses" in chosen_moment:
                honest_response = random.choice(chosen_moment["honest_responses"])
                scene_text = scene_text.format(npc_name=npc_name, honest_response=honest_response)
            elif "choice_variations" in chosen_moment:
                choice_text = random.choice(chosen_moment["choice_variations"])
                scene_text = scene_text.format(npc_name=npc_name, choice_text=choice_text)
            else:
                scene_text = scene_text.format(npc_name=npc_name, other_npc=other_npc)
            
            # Record this narrative moment
            cursor.execute("""
                INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, narrative_moment, entry_text, timestamp)
                VALUES (%s, %s, 'narrative_moment', %s, %s, CURRENT_TIMESTAMP)
            """, (user_id, conversation_id, chosen_moment["name"], scene_text))
            
            conn.commit()
            
            return {
                "type": "narrative_moment",
                "name": chosen_moment["name"],
                "scene_text": scene_text,
                "player_realization": chosen_moment["player_realization"]
            }
        
        return None
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error checking for narrative moments: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def check_for_npc_revelations(user_id, conversation_id):
    """
    Check if it's time for an NPC to reveal more of their true nature
    Returns a revelation if triggered, None otherwise
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get the current narrative stage
        cursor.execute("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=%s AND conversation_id=%s AND key='CurrentNarrativeStage'
        """, (user_id, conversation_id))
        
        stage_row = cursor.fetchone()
        current_stage = stage_row[0] if stage_row else "Innocent Beginning"
        
        # Find appropriate stage object
        current_stage_obj = None
        for stage in NARRATIVE_STAGES:
            if stage.name == current_stage:
                current_stage_obj = stage
                break
                
        if not current_stage_obj:
            return None
            
        # Get NPCs who might reveal something
        cursor.execute("""
            SELECT npc_id, npc_name, dominance, cruelty, introduced
            FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND introduced=TRUE
            ORDER BY dominance DESC
        """, (user_id, conversation_id))
        
        npcs = cursor.fetchall()
        
        # Get already revealed NPCs for this stage
        cursor.execute("""
            SELECT npc_id FROM NPCRevelations
            WHERE user_id=%s AND conversation_id=%s AND narrative_stage=%s
        """, (user_id, conversation_id, current_stage))
        
        revealed_npcs = set()
        for row in cursor.fetchall():
            revealed_npcs.add(row[0])
        
        # Filter NPCs who haven't revealed yet and meet dominance threshold
        eligible_npcs = []
        for npc in npcs:
            npc_id, npc_name, dominance, cruelty, introduced = npc
            
            # Skip if already revealed in this stage
            if npc_id in revealed_npcs:
                continue
                
            # Determine minimum dominance based on stage
            if current_stage == "Innocent Beginning":
                min_dominance = 30
            elif current_stage == "First Doubts":
                min_dominance = 40
            elif current_stage == "Creeping Realization":
                min_dominance = 60
            elif current_stage == "Veil Thinning":
                min_dominance = 70
            elif current_stage == "Full Revelation":
                min_dominance = 80
            else:
                min_dominance = 50
                
            if dominance >= min_dominance:
                eligible_npcs.append(npc)
        
        # If eligible NPCs, select one randomly
        if eligible_npcs:
            chosen_npc = random.choice(eligible_npcs)
            npc_id, npc_name, dominance, cruelty, introduced = chosen_npc
            
            # Get another NPC for context if available
            cursor.execute("""
                SELECT npc_name FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s AND introduced=TRUE
                AND npc_id != %s
                ORDER BY RANDOM() LIMIT 1
            """, (user_id, conversation_id, npc_id))
            
            other_npc_row = cursor.fetchone()
            other_npc = other_npc_row[0] if other_npc_row else "someone else"
            
            # Select a revelation text
            revelation_text = random.choice(current_stage_obj.revelation_texts)
            revelation_text = revelation_text.format(npc_name=npc_name, other_npc=other_npc)
            
            # Record this revelation
            cursor.execute("""
                INSERT INTO NPCRevelations (user_id, conversation_id, npc_id, narrative_stage, revelation_text, timestamp)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """, (user_id, conversation_id, npc_id, current_stage, revelation_text))
            
            # Also add to player journal
            cursor.execute("""
                INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, timestamp)
                VALUES (%s, %s, 'npc_revelation', %s, CURRENT_TIMESTAMP)
            """, (user_id, conversation_id, f"Revelation about {npc_name}: {revelation_text}"))
            
            conn.commit()
            
            return {
                "type": "npc_revelation",
                "npc_name": npc_name,
                "text": revelation_text,
                "narrative_stage": current_stage
            }
        
        return None
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error checking for NPC revelations: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def add_dream_sequence(user_id, conversation_id):
    """
    Generate a symbolic dream sequence based on player's current state
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get player stats
        cursor.execute("""
            SELECT corruption, confidence, willpower, obedience, dependency
            FROM PlayerStats
            WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
        """, (user_id, conversation_id))
        
        row = cursor.fetchone()
        if not row:
            return None
            
        corruption, confidence, willpower, obedience, dependency = row
        
        # Get primary NPCs
        cursor.execute("""
            SELECT npc_name FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND introduced=TRUE
            ORDER BY dominance DESC LIMIT 3
        """, (user_id, conversation_id))
        
        npc_names = [row[0] for row in cursor.fetchall()]
        
        # Define dream templates based on player state
        dream_templates = []
        
        # High corruption dreams
        if corruption > 70:
            dream_templates.append(
                "You're sitting in a chair as {npc1} circles you slowly. \"Show me your hands,\" she says. "
                "You extend them, surprised to find intricate strings wrapped around each finger, extending upward. "
                "\"Do you see who's holding them?\" she asks. You look up, but the ceiling is mirrored, "
                "showing only your own face looking back down at you, smiling with an expression that isn't yours."
            )
            
        # High dependency dreams
        if dependency > 70:
            dream_templates.append(
                "You're searching your home frantically, calling {npc1}'s name. The rooms shift and expand, "
                "doorways leading to impossible spaces. Your phone rings. It's {npc1}. \"Where are you?\" you ask desperately. "
                "\"I'm right here,\" she says, her voice coming both from the phone and from behind you. "
                "\"I've always been right here. You're the one who's lost.\""
            )
            
        # Low willpower dreams
        if willpower < 30:
            dream_templates.append(
                "You're trying to walk away from {npc1}, but your feet sink deeper into the floor with each step. "
                "\"I don't understand why you're struggling,\" she says, not moving yet somehow keeping pace beside you. "
                "\"You stopped walking on your own long ago.\" You look down to find your legs have merged with the floor entirely, "
                "indistinguishable from the material beneath."
            )
            
        # Low confidence dreams
        if confidence < 30:
            dream_templates.append(
                "You're giving a presentation to a room full of people, but every time you speak, your voice comes out as {npc1}'s voice, "
                "saying words you didn't intend. The audience nods approvingly. \"Much better,\" whispers {npc2} from beside you. "
                "\"Your ideas were never as good as hers anyway.\""
            )
            
        # Default dreams
        dream_templates.append(
            "You're walking through an unfamiliar house, opening doors that should lead outside but only reveal more rooms. "
            "In each room, {npc1} is engaged in a different activity, wearing a different expression. In the final room, "
            "all versions of her turn to look at you simultaneously. \"Which one is real?\" they ask in unison. \"The one you needed, or the one who needed you?\""
        )
        
        dream_templates.append(
            "You're swimming in deep water. Below you, {npc1} and {npc2} walk along the bottom, "
            "looking up at you and conversing, their voices perfectly clear despite the water. "
            "\"They still think they're above it all,\" says {npc1}, and they both laugh. You realize you can't remember how to reach the surface."
        )
        
        # Select a dream template
        dream_text = random.choice(dream_templates)
        
        # Format with NPC names
        npc1 = npc_names[0] if npc_names else "someone"
        npc2 = npc_names[1] if len(npc_names) > 1 else "another woman"
        npc3 = npc_names[2] if len(npc_names) > 2 else "a third woman"
        
        dream_text = dream_text.format(npc1=npc1, npc2=npc2, npc3=npc3)
        
        # Record this dream
        cursor.execute("""
            INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, timestamp)
            VALUES (%s, %s, 'dream_sequence', %s, CURRENT_TIMESTAMP)
        """, (user_id, conversation_id, dream_text))
        
        conn.commit()
        
        return {
            "type": "dream_sequence",
            "text": dream_text
        }
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error adding dream sequence: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def add_moment_of_clarity(user_id, conversation_id):
    """
    Generate a moment of clarity where the player briefly recognizes their changing state
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get player stats
        cursor.execute("""
            SELECT corruption, confidence, willpower, obedience, dependency,
                   lust, mental_resilience, physical_endurance
            FROM PlayerStats
            WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
        """, (user_id, conversation_id))
        
        row = cursor.fetchone()
        if not row:
            return None
            
        corruption, confidence, willpower, obedience, dependency, lust, mental_resilience, physical_endurance = row
        
        # Get stats history for comparison
        cursor.execute("""
            SELECT stat_name, old_value, new_value
            FROM StatsHistory
            WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
            ORDER BY timestamp DESC
            LIMIT 10
        """, (user_id, conversation_id))
        
        stats_history = cursor.fetchall()
        
        # Get the most significant recent change
        significant_change = None
        for stat_name, old_value, new_value in stats_history:
            if abs(new_value - old_value) >= 10:
                significant_change = (stat_name, old_value, new_value)
                break
                
        # Define clarity templates based on player state and changes
        clarity_templates = []
        
        # Template based on significant stat change
        if significant_change:
            stat_name, old_value, new_value = significant_change
            
            if stat_name == "corruption" and new_value > old_value:
                clarity_templates.append(
                    "It hits you while looking in the mirror—something in your eyes has changed. You used to be bothered by things that now feel natural, "
                    "even expected. When did that shift happen? You try to recall your thoughts from a few months ago, but they feel like they belonged to someone else."
                )
            elif stat_name == "obedience" and new_value > old_value:
                clarity_templates.append(
                    "You catch yourself automatically reorganizing your schedule after a casual comment from {npc_name}. Your hand freezes over your calendar. "
                    "When did her preferences begin to override your own plans without conscious thought? The realization is unsettling, but the discomfort fades quickly."
                )
            elif stat_name == "dependency" and new_value > old_value:
                clarity_templates.append(
                    "Your phone battery dies and a wave of anxiety washes over you. What if {npc_name} needs to reach you? What if she has expectations you're not meeting? "
                    "The panic feels disproportionate, and for a moment, you recognize how attached you've become. Is this healthy? The thought slips away as you rush to find a charger."
                )
            elif stat_name == "willpower" and old_value > new_value:
                clarity_templates.append(
                    "You remember making a promise to yourself that you've now broken. Standing firm on certain principles used to be important to you. "
                    "When did it become so easy to let {npc_name} redefine those boundaries? The thought creates a moment of alarm that quickly dissolves into rationalization."
                )
            elif stat_name == "confidence" and old_value > new_value:
                clarity_templates.append(
                    "You hesitate before expressing an opinion, instinctively wondering what {npc_name} would think. You used to speak freely without this filter. "
                    "The realization makes you briefly angry, but the feeling shifts to something more like resignation. Maybe your ideas really are better when vetted by her first."
                )
                
        # General clarity templates based on current stats
        if corruption > 60 and willpower < 40:
            clarity_templates.append(
                "While sorting through old photos, you find one from just before you met {npc_name}. Your expression, your posture—everything seems different. "
                "For a few minutes, you feel like you're looking at a stranger, someone more self-contained. The feeling is neither positive nor negative, just... distant."
            )
            
        if dependency > 70:
            clarity_templates.append(
                "She's late. Hours late. Your thoughts spiral between worry and abandonment as you check your phone for the thirtieth time. "
                "Suddenly, you see yourself from outside—pacing, checking, entirely consumed by her absence. 'This isn't normal,' a small voice observes, "
                "before being drowned out by relief as your phone finally buzzes with her message."
            )
            
        if obedience > 80:
            clarity_templates.append(
                "You're nodding along as {npc_name} makes a decision that affects your weekend. Mid-nod, a thought surfaces: 'When was the last time I made plans without consulting her first?' "
                "The question feels important, but also oddly irrelevant—as if questioning the sunrise. Of course you check with her. Why wouldn't you?"
            )
            
        # Default clarity templates
        clarity_templates.append(
            "Sometimes, late at night when you can't sleep, you try to trace the path that led you here. Each individual step made sense at the time, each concession seemed small. "
            "But looking at the total distance traveled... that's when the vertigo hits. By morning, the feeling has passed, replaced by the comfortable routine of seeking {npc_name}'s guidance."
        )
        
        clarity_templates.append(
            "A comment from an old friend—'You've changed'—lingers with you throughout the day. Not accusatory, just observational. "
            "You find yourself mentally defending the changes, listing all the ways you're better now, more fulfilled. Yet beneath the justifications lies a question: "
            "If the changes are so positive, why the need to defend them so vigorously?"
        )
        
        # Get an NPC name for formatting
        cursor.execute("""
            SELECT npc_name FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND introduced=TRUE
            ORDER BY dominance DESC LIMIT 1
        """, (user_id, conversation_id))
        
        npc_row = cursor.fetchone()
        npc_name = npc_row[0] if npc_row else "her"
        
        # Select a clarity template
        clarity_text = random.choice(clarity_templates)
        
        # Format with NPC name
        clarity_text = clarity_text.format(npc_name=npc_name)
        
        # Record this moment of clarity
        cursor.execute("""
            INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, timestamp)
            VALUES (%s, %s, 'moment_of_clarity', %s, CURRENT_TIMESTAMP)
        """, (user_id, conversation_id, clarity_text))
        
        conn.commit()
        
        return {
            "type": "moment_of_clarity",
            "text": clarity_text
        }
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error adding moment of clarity: {e}")
        return None
    finally:
        cursor.close()
        conn.close()
