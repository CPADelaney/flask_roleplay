# logic/npc_evolution.py

import random
import json
import logging
from datetime import datetime
from db.connection import get_db_connection
from logic.social_links import create_social_link, update_link_type_and_level, add_link_event

# NPC mask slippage triggers - these are moments when the NPC's true nature begins to show
MASK_SLIPPAGE_TRIGGERS = {
    "dominance": [
        {"threshold": 30, "event": "subtle_control", "memory": "I let my control slip through a bit today. Nobody seemed to notice the subtle shift in dynamic."},
        {"threshold": 50, "event": "mask_adjustment", "memory": "It's getting harder to maintain this facade. I caught myself giving commands too firmly, had to play it off as a joke."},
        {"threshold": 70, "event": "partial_revelation", "memory": "I showed a glimpse of my true self today. The flash of fear in their eyes was... intoxicating."},
        {"threshold": 90, "event": "mask_removal", "memory": "I'm barely pretending anymore. Those who understand appreciate the honesty. Those who don't will learn."}
    ],
    "cruelty": [
        {"threshold": 30, "event": "sharp_comment", "memory": "I said something cutting today and quickly covered it with a laugh. The momentary hurt in their eyes was satisfying."},
        {"threshold": 50, "event": "testing_boundaries", "memory": "I'm pushing further each time to see what I can get away with. People are so willing to excuse 'playful' cruelty."},
        {"threshold": 70, "event": "deliberate_harm", "memory": "I orchestrated a situation today that caused genuine distress. I maintained plausible deniability, of course."},
        {"threshold": 90, "event": "overt_sadism", "memory": "My reputation for 'intensity' is established enough that I barely need to hide my enjoyment of others' suffering now."}
    ],
    "intensity": [
        {"threshold": 30, "event": "piercing_gaze", "memory": "Someone commented on my intense stare today. I've learned to soften it in public, but sometimes I forget."},
        {"threshold": 50, "event": "forceful_presence", "memory": "People naturally move aside when I walk through a room now. My presence is becoming harder to disguise."},
        {"threshold": 70, "event": "commanding_aura", "memory": "I no longer need to raise my voice to be obeyed. My quiet commands carry weight that surprises even me."},
        {"threshold": 90, "event": "overwhelming_presence", "memory": "The mask has become nearly transparent. My true nature radiates from me, drawing submission from the vulnerable."}
    ]
}

# NPC daily activity definitions - focused on relationship building, social networking, and subtle manipulation
NPC_ACTIVITIES = [
    {
        "name": "casual_socializing",
        "description": "Spend time with others in seemingly harmless social activities",
        "stat_changes": {"closeness": +2},
        "relationship_effects": {"link_level": +3},
        "memory_templates": [
            "Spent time with {target_name} today. Just casual conversation, but I'm learning their vulnerabilities.",
            "Had coffee with {target_name}. These 'friendly' interactions make them lower their guard around me.",
            "Went shopping with {target_name}. Every personal detail they share is something I can use later."
        ]
    },
    {
        "name": "subtle_testing",
        "description": "Test boundaries in seemingly innocent ways",
        "stat_changes": {"dominance": +1, "cruelty": +1},
        "relationship_effects": {"link_level": +2},
        "memory_templates": [
            "Asked {target_name} for small favors today. Each compliance makes the next request easier.",
            "Made 'jokes' that pushed {target_name}'s comfort zone. Noting which ones they nervously laugh at versus challenge.",
            "Suggested activities just outside {target_name}'s comfort zone. Their hesitation before agreeing tells me volumes."
        ]
    },
    {
        "name": "vulnerability_sharing",
        "description": "Create false intimacy through strategic vulnerability",
        "stat_changes": {"closeness": +3, "trust": +2},
        "relationship_effects": {"link_level": +5},
        "memory_templates": [
            "Shared 'personal struggles' with {target_name} today. Carefully crafted to seem vulnerable while giving away nothing real.",
            "Had a 'deep conversation' with {target_name}. My fabricated vulnerabilities encourage them to reveal genuine ones.",
            "{target_name} opened up to me today after I presented a calculated show of trust. The intimacy is entirely one-sided."
        ]
    },
    {
        "name": "group_bonding",
        "description": "Build relationships with multiple people simultaneously",
        "relationship_effects": {"link_level": +2},
        "group_activity": True,
        "memory_templates": [
            "Organized a get-together with {target_names}. The dynamics between them are fascinating - and exploitable.",
            "Spent time with several people today. I'm carefully positioning myself in their social hierarchy.",
            "Group activities make it easy to observe everyone's weaknesses while maintaining my friendly persona."
        ]
    },
    {
        "name": "hidden_training",
        "description": "Subtly condition others to respond to control",
        "stat_changes": {"dominance": +2},
        "relationship_effects": {"link_level": +3, "submission_level": +2},
        "memory_templates": [
            "I've been conditioning {target_name} with subtle rewards when they comply. They don't even notice the pattern forming.",
            "Practiced my techniques on {target_name} today. A firm tone disguised as concern, touch that guides behavior - it's working.",
            "The training is progressing with {target_name}. They now unconsciously seek my approval before making decisions."
        ]
    },
    {
        "name": "alliance_building",
        "description": "Form connections with other dominant figures",
        "stat_changes": {"intensity": +1},
        "prerequisites": {"dominance": 40},
        "alliance_activity": True,
        "memory_templates": [
            "Connected with {target_name} today. We understand each other's true nature beneath our public personas.",
            "Met with {target_name} privately. We share similar interests in control, though we present differently in public.",
            "Spent time with the inner circle today. Our network grows stronger, all while maintaining perfectly respectable appearances."
        ]
    },
    {
        "name": "strategic_assistance",
        "description": "Create dependency through solving problems",
        "stat_changes": {"dominance": +1},
        "relationship_effects": {"link_level": +3, "dependency_level": +3},
        "memory_templates": [
            "Helped {target_name} with a problem today. Each time I solve something for them, their independence weakens.",
            "Offered my assistance to {target_name} again. They're beginning to reflexively turn to me instead of solving issues themselves.",
            "{target_name} thanked me effusively for my help today. They don't see how I'm positioning myself as indispensable."
        ]
    },
    {
        "name": "secret_observation",
        "description": "Gather information through covert observation",
        "stat_changes": {"cruelty": +1, "intensity": +1},
        "memory_templates": [
            "Observed {target_name} without their knowledge today. Learning their patterns, habits, vulnerabilities.",
            "Spent time watching how {target_name} interacts with others. Their public and private personas differ in interesting ways.",
            "Gathered valuable information about {target_name}'s fears and desires today. Knowledge is power."
        ]
    }
]

# Activity combinations that occur when multiple NPCs interact
NPC_GROUP_ACTIVITIES = [
    {
        "name": "private_discussion",
        "description": "NPCs privately discuss their true nature and plans",
        "required_dominance": 50,
        "stats_all": {"dominance": +1, "intensity": +1},
        "relationship_effects": {"link_level": +3, "alliance_level": +2},
        "memory_template": "Had a private conversation with {npc_names} today where we dropped our public masks for a while. Our alliance grows stronger."
    },
    {
        "name": "coordinated_manipulation",
        "description": "NPCs work together to manipulate a target",
        "required_dominance": 60,
        "required_cruelty": 40,
        "stats_all": {"dominance": +2, "cruelty": +1},
        "target_types": ["player", "npc"],
        "target_effects": {"confidence": -2, "dependency": +3},
        "memory_template": "Coordinated with {npc_names} today to subtly manipulate {target_name}. Our combined approach is proving quite effective."
    },
    {
        "name": "social_hierarchy_establishment",
        "description": "NPCs establish dominance hierarchies among themselves",
        "required_dominance": 70,
        "stats_winners": {"dominance": +3, "respect": +2},
        "stats_losers": {"respect": +3, "dependency": +2},
        "memory_template": "The hierarchy within our group became clearer today. {winner_names} demonstrated their control, while {loser_names} showed their understanding of their place."
    }
]

# Relationship stages that track the evolution of NPC-NPC and NPC-player relationships
RELATIONSHIP_STAGES = {
    "dominant": [
        {"level": 10, "name": "Initial Interest", "description": "Beginning to notice potential for control"},
        {"level": 30, "name": "Strategic Friendship", "description": "Establishing trust while assessing vulnerabilities"},
        {"level": 50, "name": "Subtle Influence", "description": "Exercising increasing control through 'guidance'"},
        {"level": 70, "name": "Open Control", "description": "Dropping pretense of equality in the relationship"},
        {"level": 90, "name": "Complete Dominance", "description": "Relationship is explicitly based on control and submission"}
    ],
    "alliance": [
        {"level": 10, "name": "Mutual Recognition", "description": "Recognizing similar controlling tendencies"},
        {"level": 30, "name": "Cautious Cooperation", "description": "Sharing limited information and techniques"},
        {"level": 50, "name": "Strategic Partnership", "description": "Actively collaborating while maintaining independence"},
        {"level": 70, "name": "Power Coalition", "description": "Forming a unified front with clear internal hierarchy"},
        {"level": 90, "name": "Dominant Cabal", "description": "Operating as a coordinated group to control others"}
    ],
    "rivalry": [
        {"level": 10, "name": "Veiled Competition", "description": "Competing subtly while maintaining cordial appearance"},
        {"level": 30, "name": "Strategic Undermining", "description": "Actively working to diminish the other's influence"},
        {"level": 50, "name": "Open Challenge", "description": "Directly competing for control and resources"},
        {"level": 70, "name": "Psychological Warfare", "description": "Actively attempting to break the other's control"},
        {"level": 90, "name": "Domination Contest", "description": "All-out struggle for supremacy"}
    ]
}

async def check_for_mask_slippage(user_id, conversation_id, npc_id):
    """
    Check if an NPC has reached thresholds where their true nature begins to show
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get NPC's current stats
        cursor.execute("""
            SELECT npc_name, dominance, cruelty, intensity, memory
            FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (user_id, conversation_id, npc_id))
        
        row = cursor.fetchone()
        if not row:
            return None
            
        npc_name, dominance, cruelty, intensity, memory_json = row
        
        # Parse memory
        if memory_json:
            if isinstance(memory_json, str):
                try:
                    memory = json.loads(memory_json)
                except:
                    memory = []
            else:
                memory = memory_json
        else:
            memory = []
            
        # Get slippage history
        cursor.execute("""
            SELECT mask_slippage_events
            FROM NPCEvolution
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (user_id, conversation_id, npc_id))
        
        row = cursor.fetchone()
        if row and row[0]:
            if isinstance(row[0], str):
                try:
                    slippage_history = json.loads(row[0])
                except:
                    slippage_history = []
            else:
                slippage_history = row[0]
        else:
            slippage_history = []
            
        # Check each stat for slippage triggers
        triggered_events = []
        
        for stat_name, thresholds in MASK_SLIPPAGE_TRIGGERS.items():
            stat_value = locals().get(stat_name, 0)  # Get the stat value from local vars
            
            for threshold in thresholds:
                event_name = threshold["event"]
                
                # Skip if this slippage has already occurred
                if any(e.get("event") == event_name for e in slippage_history):
                    continue
                    
                # Check if threshold is met
                if stat_value >= threshold["threshold"]:
                    # Record this slippage event
                    event_data = {
                        "event": event_name,
                        "stat": stat_name,
                        "threshold": threshold["threshold"],
                        "timestamp": datetime.now().isoformat()
                    }
                    triggered_events.append(event_data)
                    
                    # Add the memory
                    if "memory" in threshold:
                        memory.append(threshold["memory"])
                        
                    # Subtle physical description changes based on mask slippage
                    if stat_name == "dominance" and threshold["threshold"] >= 50:
                        cursor.execute("""
                            UPDATE NPCStats
                            SET physical_description = CONCAT(physical_description, ' In unguarded moments, her demeanor shifts subtly—a barely perceptible straightening of posture, a flash of something commanding in her eyes that quickly vanishes when noticed.')
                            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                        """, (user_id, conversation_id, npc_id))
                    
                    if stat_name == "cruelty" and threshold["threshold"] >= 50:
                        cursor.execute("""
                            UPDATE NPCStats
                            SET physical_description = CONCAT(physical_description, ' Occasionally her smile doesn't quite reach her eyes, revealing a momentary coldness before she adjusts her expression back to warmth.')
                            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                        """, (user_id, conversation_id, npc_id))
                        
                    if stat_name == "intensity" and threshold["threshold"] >= 50:
                        cursor.execute("""
                            UPDATE NPCStats
                            SET physical_description = CONCAT(physical_description, ' Sometimes when she thinks no one is watching, her gaze becomes unnervingly focused, studying others with an analytical intensity that disappears behind a pleasant mask when attention returns to her.')
                            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                        """, (user_id, conversation_id, npc_id))
        
        # Update memory
        cursor.execute("""
            UPDATE NPCStats
            SET memory = %s
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (json.dumps(memory), user_id, conversation_id, npc_id))
        
        # Update slippage history
        if triggered_events:
            slippage_history.extend(triggered_events)
            
            # Check if NPCEvolution record exists
            cursor.execute("""
                SELECT 1 FROM NPCEvolution
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
            """, (user_id, conversation_id, npc_id))
            
            if cursor.fetchone():
                cursor.execute("""
                    UPDATE NPCEvolution
                    SET mask_slippage_events = %s
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (json.dumps(slippage_history), user_id, conversation_id, npc_id))
            else:
                cursor.execute("""
                    INSERT INTO NPCEvolution
                    (user_id, conversation_id, npc_id, mask_slippage_events)
                    VALUES (%s, %s, %s, %s)
                """, (user_id, conversation_id, npc_id, json.dumps(slippage_history)))
        
        conn.commit()
        return triggered_events
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error checking mask slippage: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

async def perform_npc_daily_activity(user_id, conversation_id, npc_id, time_of_day):
    """
    Have an NPC perform activities during their daily schedule
    that develop relationships and subtly increase relevant stats
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get NPC info
        cursor.execute("""
            SELECT npc_name, dominance, cruelty, closeness, trust, respect, intensity, 
                   current_location, schedule, memory, introduced
            FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (user_id, conversation_id, npc_id))
        
        row = cursor.fetchone()
        if not row:
            return False
            
        npc_name, dominance, cruelty, closeness, trust, respect, intensity, \
        location, schedule_json, memory_json, introduced = row
        
        # Parse memory
        if memory_json:
            if isinstance(memory_json, str):
                try:
                    memory = json.loads(memory_json)
                except:
                    memory = []
            else:
                memory = memory_json
        else:
            memory = []
        
        # Parse schedule to see what they're doing
        if schedule_json:
            if isinstance(schedule_json, str):
                try:
                    schedule = json.loads(schedule_json)
                except:
                    schedule = {}
            else:
                schedule = schedule_json
        else:
            schedule = {}
        
        # Get the current day
        cursor.execute("""
            SELECT value FROM CurrentRoleplay 
            WHERE user_id=%s AND conversation_id=%s AND key='CurrentDay'
        """, (user_id, conversation_id))
        day_row = cursor.fetchone()
        current_day = int(day_row[0]) if day_row else 1
        
        # Get day name from calendar
        cursor.execute("""
            SELECT value FROM CurrentRoleplay 
            WHERE user_id=%s AND conversation_id=%s AND key='CalendarNames'
        """, (user_id, conversation_id))
        calendar_row = cursor.fetchone()
        if calendar_row and calendar_row[0]:
            try:
                calendar = json.loads(calendar_row[0])
                day_names = calendar.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                day_index = (current_day - 1) % len(day_names)
                day_name = day_names[day_index]
            except:
                day_name = "Unknown"
        else:
            day_name = "Unknown"
        
        # Check schedule for this day and time
        day_schedule = schedule.get(day_name, {})
        activity_desc = day_schedule.get(time_of_day, "")
        
        # Determine if we'll perform a social activity based on schedule
        if not activity_desc or "meets" in activity_desc.lower() or "with" in activity_desc.lower() or random.random() < 0.3:
            # Decide if we'll do a group activity with other NPCs
            do_group_activity = random.random() < 0.4  # 40% chance for group activity
            
            if do_group_activity:
                # Find other NPCs in the same location
                cursor.execute("""
                    SELECT npc_id, npc_name, dominance, cruelty
                    FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id!=%s
                    AND current_location=%s AND introduced=TRUE
                    LIMIT 3
                """, (user_id, conversation_id, npc_id, location))
                
                other_npcs = cursor.fetchall()
                
                if len(other_npcs) >= 1:
                    # We have enough NPCs for a group activity
                    all_npcs = [(npc_id, npc_name, dominance, cruelty)] + list(other_npcs)
                    npc_names = [n[1] for n in all_npcs]
                    
                    # Find eligible group activities
                    eligible_activities = []
                    for activity in NPC_GROUP_ACTIVITIES:
                        # Check if primary NPC meets requirements
                        if "required_dominance" in activity and dominance < activity["required_dominance"]:
                            continue
                        if "required_cruelty" in activity and cruelty < activity["required_cruelty"]:
                            continue
                        eligible_activities.append(activity)
                    
                    if eligible_activities:
                        # Choose a random eligible activity
                        activity = random.choice(eligible_activities)
                        
                        # Process activity effects
                        # Apply stat changes to all participating NPCs
                        if "stats_all" in activity:
                            for npc in all_npcs:
                                npc_id_update = npc[0]
                                stat_updates = []
                                for stat, change in activity["stats_all"].items():
                                    stat_updates.append(f"{stat} = {stat} + {change}")
                                
                                if stat_updates:
                                    cursor.execute(f"""
                                        UPDATE NPCStats
                                        SET {', '.join(stat_updates)}
                                        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                                    """, (user_id, conversation_id, npc_id_update))
                        
                        # For hierarchy activities, determine winners and losers
                        if "stats_winners" in activity and "stats_losers" in activity:
                            # Sort NPCs by dominance for determining hierarchy
                            sorted_npcs = sorted(all_npcs, key=lambda x: x[2], reverse=True)
                            mid_point = len(sorted_npcs) // 2
                            
                            winners = sorted_npcs[:mid_point]
                            losers = sorted_npcs[mid_point:]
                            
                            # Apply winner stats
                            for npc in winners:
                                npc_id_update = npc[0]
                                stat_updates = []
                                for stat, change in activity["stats_winners"].items():
                                    stat_updates.append(f"{stat} = {stat} + {change}")
                                
                                if stat_updates:
                                    cursor.execute(f"""
                                        UPDATE NPCStats
                                        SET {', '.join(stat_updates)}
                                        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                                    """, (user_id, conversation_id, npc_id_update))
                            
                            # Apply loser stats
                            for npc in losers:
                                npc_id_update = npc[0]
                                stat_updates = []
                                for stat, change in activity["stats_losers"].items():
                                    stat_updates.append(f"{stat} = {stat} + {change}")
                                
                                if stat_updates:
                                    cursor.execute(f"""
                                        UPDATE NPCStats
                                        SET {', '.join(stat_updates)}
                                        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                                    """, (user_id, conversation_id, npc_id_update))
                        
                        # Create or update relationships between all participants
                        if "relationship_effects" in activity:
                            for i in range(len(all_npcs)):
                                for j in range(i+1, len(all_npcs)):
                                    npc1_id = all_npcs[i][0]
                                    npc2_id = all_npcs[j][0]
                                    
                                    # Check existing relationship
                                    cursor.execute("""
                                        SELECT link_id, link_type, link_level
                                        FROM SocialLinks
                                        WHERE user_id=%s AND conversation_id=%s
                                        AND entity1_type='npc' AND entity1_id=%s
                                        AND entity2_type='npc' AND entity2_id=%s
                                    """, (user_id, conversation_id, npc1_id, npc2_id))
                                    
                                    link_row = cursor.fetchone()
                                    
                                    if link_row:
                                        # Update existing link
                                        link_id, link_type, link_level = link_row
                                        
                                        updates = []
                                        values = []
                                        
                                        # General link level
                                        if "link_level" in activity["relationship_effects"]:
                                            change = activity["relationship_effects"]["link_level"]
                                            updates.append("link_level = link_level + %s")
                                            values.append(change)
                                        
                                        # Specific relationship type levels
                                        for rel_type, change in activity["relationship_effects"].items():
                                            if rel_type.endswith("_level") and rel_type != "link_level":
                                                rel_name = rel_type.replace("_level", "")
                                                if link_type == rel_name:
                                                    updates.append("link_level = link_level + %s")
                                                    values.append(change)
                                        
                                        if updates:
                                            values.append(link_id)
                                            cursor.execute(f"""
                                                UPDATE SocialLinks
                                                SET {', '.join(updates)}
                                                WHERE link_id = %s
                                            """, values)
                                            
                                            # Add event to link history
                                            event_text = f"Participated in {activity['name']} together."
                                            cursor.execute("""
                                                UPDATE SocialLinks
                                                SET link_history = link_history || %s::jsonb
                                                WHERE link_id = %s
                                            """, (json.dumps([event_text]), link_id))
                                    else:
                                        # Create new link - determine relationship type
                                        if "alliance" in activity["name"].lower():
                                            link_type = "alliance"
                                        elif "hierarchy" in activity["name"].lower():
                                            # Determine based on dominance
                                            npc1_dom = all_npcs[i][2]
                                            npc2_dom = all_npcs[j][2]
                                            if abs(npc1_dom - npc2_dom) < 10:
                                                link_type = "rivalry"
                                            else:
                                                link_type = "alliance"
                                        else:
                                            link_type = "neutral"
                                        
                                        level = activity["relationship_effects"].get("link_level", 0)
                                        if f"{link_type}_level" in activity["relationship_effects"]:
                                            level += activity["relationship_effects"][f"{link_type}_level"]
                                        
                                        cursor.execute("""
                                            INSERT INTO SocialLinks
                                            (user_id, conversation_id, entity1_type, entity1_id, 
                                             entity2_type, entity2_id, link_type, link_level, link_history)
                                            VALUES (%s, %s, 'npc', %s, 'npc', %s, %s, %s, %s)
                                        """, (
                                            user_id, conversation_id,
                                            npc1_id, npc2_id,
                                            link_type, level, json.dumps([f"Formed relationship during {activity['name']}."])
                                        ))
                        
                        # Record memory for primary NPC
                        memory_text = activity["memory_template"]
                        winner_names = ", ".join([n[1] for n in winners]) if "stats_winners" in activity else ""
                        loser_names = ", ".join([n[1] for n in losers]) if "stats_losers" in activity else ""
                        formatted_memory = memory_text.format(
                            npc_names=", ".join(npc_names[1:]),  # Exclude self
                            winner_names=winner_names,
                            loser_names=loser_names
                        )
                        
                        memory.append(formatted_memory)
                        cursor.execute("""
                            UPDATE NPCStats
                            SET memory = %s
                            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                        """, (json.dumps(memory), user_id, conversation_id, npc_id))
                        
                        conn.commit()
                        return True
            
            # If we didn't do a group activity or couldn't find participants, do a regular activity
            # Choose a random activity from NPC_ACTIVITIES
            activity = random.choice(NPC_ACTIVITIES)
            
            # Apply stat changes to NPC
            if "stat_changes" in activity:
                stat_updates = []
                for stat, change in activity["stat_changes"].items():
                    stat_updates.append(f"{stat} = {stat} + {change}")
                
                if stat_updates:
                    cursor.execute(f"""
                        UPDATE NPCStats
                        SET {', '.join(stat_updates)}
                        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                    """, (user_id, conversation_id, npc_id))
            
            # Decide if targeting player or NPC
            target_id = None
            target_name = None
            target_type = None
            
            if random.random() < 0.3:  # 30% chance to target player
                target_id = user_id
                target_name = "Chase"
                target_type = "player"
            else:
                # Find another NPC to interact with
                cursor.execute("""
                    SELECT npc_id, npc_name FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id!=%s
                    ORDER BY RANDOM() LIMIT 1
                """, (user_id, conversation_id, npc_id))
                target_row = cursor.fetchone()
                
                if target_row:
                    target_id = target_row[0]
                    target_name = target_row[1]
                    target_type = "npc"
            
            # Create or update relationship if we have a target
            if target_id and target_name and "relationship_effects" in activity:
                # Check for existing link
                cursor.execute("""
                    SELECT link_id, link_type, link_level
                    FROM SocialLinks
                    WHERE user_id=%s AND conversation_id=%s
                    AND entity1_type='npc' AND entity1_id=%s
                    AND entity2_type=%s AND entity2_id=%s
                """, (user_id, conversation_id, npc_id, target_type, target_id))
                
                link_row = cursor.fetchone()
                
                if link_row:
                    # Update existing link
                    link_id, link_type, link_level = link_row
                    
                    updates = []
                    values = []
                    
                    # General link level
                    if "link_level" in activity["relationship_effects"]:
                        change = activity["relationship_effects"]["link_level"]
                        updates.append("link_level = link_level + %s")
                        values.append(change)
                    
                    # Specific relationship type levels
                    for rel_type, change in activity["relationship_effects"].items():
                        if rel_type.endswith("_level") and rel_type != "link_level":
                            rel_name = rel_type.replace("_level", "")
                            if link_type == rel_name:
                                updates.append("link_level = link_level + %s")
                                values.append(change)
                    
                    if updates:
                        values.append(link_id)
                        cursor.execute(f"""
                            UPDATE SocialLinks
                            SET {', '.join(updates)}
                            WHERE link_id = %s
                        """, values)
                        
                        # Add event to link history
                        event_text = f"{npc_name} performed {activity['name']} with {target_name}."
                        cursor.execute("""
                            UPDATE SocialLinks
                            SET link_history = link_history || %s::jsonb
                            WHERE link_id = %s
                        """, (json.dumps([event_text]), link_id))
                else:
                    # Create new link
                    link_type = "friendship"  # Default relationship type
                    
                    # Determine relationship type based on stats
                    if dominance >= 60:
                        if "hidden_training" in activity["name"] or "strategic_assistance" in activity["name"]:
                            link_type = "dominant"
                    
                    level = activity["relationship_effects"].get("link_level", 0)
                    if f"{link_type}_level" in activity["relationship_effects"]:
                        level += activity["relationship_effects"][f"{link_type}_level"]
                    
                    cursor.execute("""
                        INSERT INTO SocialLinks
                        (user_id, conversation_id, entity1_type, entity1_id, 
                         entity2_type, entity2_id, link_type, link_level, link_history)
                        VALUES (%s, %s, 'npc', %s, %s, %s, %s, %s, %s)
                    """, (
                        user_id, conversation_id,
                        npc_id, target_type, target_id,
                        link_type, level, json.dumps([f"Formed relationship during {activity['name']}."])
                    ))
            
            # Record memory
            if target_name and "memory_templates" in activity:
                memory_template = random.choice(activity["memory_templates"])
                formatted_memory = memory_template.format(target_name=target_name)
                
                memory.append(formatted_memory)
                cursor.execute("""
                    UPDATE NPCStats
                    SET memory = %s
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (json.dumps(memory), user_id, conversation_id, npc_id))
        
        conn.commit()
        return True
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error in NPC daily activity: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

async def process_daily_npc_activities(user_id, conversation_id, time_of_day):
    """
    Process activities for all NPCs during a specific time of day
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get all NPCs
        cursor.execute("""
            SELECT npc_id FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s
        """, (user_id, conversation_id))
        
        npc_ids = [row[0] for row in cursor.fetchall()]
        
        for npc_id in npc_ids:
            # Check for mask slippage
            await check_for_mask_slippage(user_id, conversation_id, npc_id)
            
            # Perform daily activity
            await perform_npc_daily_activity(user_id, conversation_id, npc_id, time_of_day)
        
        return True
    except Exception as e:
        logging.error(f"Error processing daily NPC activities: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

async def detect_relationship_stage_changes(user_id, conversation_id):
    """
    Detect changes in relationship stages and update memories accordingly
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get all social links
        cursor.execute("""
            SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id, 
                   link_type, link_level, relationship_stage
            FROM SocialLinks
            WHERE user_id=%s AND conversation_id=%s
        """, (user_id, conversation_id))
        
        links = cursor.fetchall()
        
        for link in links:
            link_id, e1_type, e1_id, e2_type, e2_id, link_type, link_level, current_stage = link
            
            # Skip if link type isn't in our defined stages
            if link_type not in RELATIONSHIP_STAGES:
                continue
            
            # Find the appropriate stage
            stages = RELATIONSHIP_STAGES[link_type]
            
            new_stage = None
            for stage in reversed(stages):  # Check from highest to lowest
                if link_level >= stage["level"]:
                    new_stage = stage
                    break
            
            if not new_stage:
                continue
                
            stage_name = new_stage["name"]
            
            # If stage has changed, update it
            if stage_name != current_stage:
                cursor.execute("""
                    UPDATE SocialLinks
                    SET relationship_stage = %s
                    WHERE link_id = %s
                """, (stage_name, link_id))
                
                # Add event to history
                event_text = f"Relationship evolved to stage: {stage_name} - {new_stage['description']}"
                cursor.execute("""
                    UPDATE SocialLinks
                    SET link_history = link_history || %s::jsonb
                    WHERE link_id = %s
                """, (json.dumps([event_text]), link_id))
                
                # Create memory entries for the entities
                memory_text = f"My relationship with {get_entity_name(conn, e2_type, e2_id, user_id, conversation_id)} has shifted. {new_stage['description']}."
                
                if e1_type == 'npc':
                    add_npc_memory(conn, user_id, conversation_id, e1_id, memory_text)
                
                if e2_type == 'npc':
                    reciprocal_text = f"My relationship with {get_entity_name(conn, e1_type, e1_id, user_id, conversation_id)} has changed. {get_reciprocal_description(new_stage['description'])}."
                    add_npc_memory(conn, user_id, conversation_id, e2_id, reciprocal_text)
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logging.error(f"Error detecting relationship stage changes: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def get_entity_name(conn, entity_type, entity_id, user_id, conversation_id):
    """
    Get the name of an entity (NPC or player)
    """
    if entity_type == 'player':
        return "Chase"
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT npc_name FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
    """, (user_id, conversation_id, entity_id))
    
    row = cursor.fetchone()
    cursor.close()
    
    return row[0] if row else "Unknown"

def get_reciprocal_description(description):
    """
    Generate a reciprocal description from the perspective of the other entity
    """
    # Simple replacements for now
    replacements = {
        "control": "being controlled",
        "dominance": "submission",
        "manipulating": "being influenced",
        "assessing vulnerabilities": "being evaluated",
        "control and submission": "submission and control"
    }
    
    result = description
    for original, replacement in replacements.items():
        result = result.replace(original, replacement)
    
    return result

def add_npc_memory(conn, user_id, conversation_id, npc_id, memory_text):
    """
    Add a memory entry for an NPC
    """
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT memory FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (user_id, conversation_id, npc_id))
        
        row = cursor.fetchone()
        if row and row[0]:
            if isinstance(row[0], str):
                try:
                    memory = json.loads(row[0])
                except:
                    memory = []
            else:
                memory = row[0]
        else:
            memory = []
        
        memory.append(memory_text)
        
        cursor.execute("""
            UPDATE NPCStats
            SET memory = %s
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (json.dumps(memory), user_id, conversation_id, npc_id))
        
    finally:
        cursor.close()
