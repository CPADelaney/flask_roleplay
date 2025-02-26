# logic/enhanced_social_links.py

import random
import json
import logging
from datetime import datetime
from db.connection import get_db_connection

# Define relationship dynamics that can exist between player and NPCs or between NPCs
RELATIONSHIP_DYNAMICS = [
    {
        "name": "control",
        "description": "One entity exerts control over the other",
        "levels": [
            {"level": 10, "name": "Subtle Influence", "description": "Occasional suggestions that subtly guide behavior"},
            {"level": 30, "name": "Regular Direction", "description": "Frequent guidance that shapes decisions"},
            {"level": 50, "name": "Strong Authority", "description": "Explicit direction with expectation of compliance"},
            {"level": 70, "name": "Dominant Control", "description": "Commands given with assumption of obedience"},
            {"level": 90, "name": "Complete Dominance", "description": "Total control with no expectation of resistance"}
        ]
    },
    {
        "name": "manipulation",
        "description": "One entity manipulates the other through indirect means",
        "levels": [
            {"level": 10, "name": "Minor Misdirection", "description": "Occasional white lies to achieve small goals"},
            {"level": 30, "name": "Regular Deception", "description": "Consistent pattern of misleading to shape behavior"},
            {"level": 50, "name": "Calculated Manipulation", "description": "Strategic dishonesty to achieve control"},
            {"level": 70, "name": "Psychological Conditioning", "description": "Systematic reshaping of target's reality"},
            {"level": 90, "name": "Complete Gaslighting", "description": "Target's entire perception is controlled"}
        ]
    },
    {
        "name": "dependency",
        "description": "One entity becomes dependent on the other",
        "levels": [
            {"level": 10, "name": "Mild Attachment", "description": "Enjoys presence but functions fine independently"},
            {"level": 30, "name": "Regular Reliance", "description": "Seeks input for decisions and emotional support"},
            {"level": 50, "name": "Strong Dependency", "description": "Difficulty making decisions without approval"},
            {"level": 70, "name": "Heavy Dependency", "description": "Significant anxiety when separated"},
            {"level": 90, "name": "Complete Dependence", "description": "Unable to function independently"}
        ]
    },
    {
        "name": "alliance",
        "description": "Entities work together for mutual benefit",
        "levels": [
            {"level": 10, "name": "Casual Cooperation", "description": "Occasional collaboration when convenient"},
            {"level": 30, "name": "Regular Partnership", "description": "Consistent working relationship"},
            {"level": 50, "name": "Strong Alliance", "description": "Prioritize mutual support in most situations"},
            {"level": 70, "name": "Deep Partnership", "description": "Significant investment in each other's success"},
            {"level": 90, "name": "Unbreakable Alliance", "description": "Complete mutual commitment"}
        ]
    },
    {
        "name": "rivalry",
        "description": "Entities compete with each other",
        "levels": [
            {"level": 10, "name": "Mild Competition", "description": "Occasional friendly contests"},
            {"level": 30, "name": "Regular Rivalry", "description": "Consistent pattern of trying to outdo each other"},
            {"level": 50, "name": "Strong Competition", "description": "Active attempts to undermine each other"},
            {"level": 70, "name": "Intense Rivalry", "description": "Significant efforts to dominate the other"},
            {"level": 90, "name": "Bitter Enemies", "description": "Complete dedication to defeating the other"}
        ]
    }
]

# Define relationship crossroads - significant choice points that alter relationship trajectory
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
                "outcome": "You accept {npc_name}'s explanation and allow her greater involvement in your decisions. The relief in surrendering control is both unexpected and concerning."
            },
            {
                "text": "Establish boundary",
                "effects": {"control": -10, "rivalry": +15, "player_willpower": +5},
                "outcome": "You firmly establish a boundary with {npc_name}. Her expression flickers between surprise and something colder before she composes herself."
            }
        ]
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
                "outcome": "You follow {npc_name}'s instructions despite your discomfort. The approval in her eyes provides a confusing sense of validation."
            },
            {
                "text": "Refuse publicly",
                "effects": {"control": -15, "rivalry": +20, "player_confidence": +5},
                "outcome": "You refuse {npc_name}'s request, causing a momentary tension. Later, she approaches you with a different demeanor, reassessing her approach."
            }
        ]
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
                "outcome": "You confront {npc_name} about her deception. She seems genuinely caught off-guard by your assertion, quickly adapting with a new approach."
            },
            {
                "text": "Pretend not to notice",
                "effects": {"manipulation": +15, "dependency": +5, "player_mental_resilience": -10},
                "outcome": "You keep your discovery to yourself, watching as {npc_name} continues her manipulations with growing confidence, unaware that you see through them."
            }
        ]
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
                "outcome": "You offer complete support to {npc_name}, prioritizing her needs above your own. Her vulnerability feels oddly calculated, but the bond strengthens."
            },
            {
                "text": "Offer limited support",
                "effects": {"dependency": -5, "rivalry": +5, "player_corruption": -5},
                "outcome": "You offer support while maintaining some distance. {npc_name} seems disappointed but respects your boundaries, adjusting her approach accordingly."
            }
        ]
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
                "outcome": "You fully commit to your partnership with {npc_name}, integrating your goals with hers. The efficiency of your collaboration masks the gradual shift in power."
            },
            {
                "text": "Maintain independence",
                "effects": {"alliance": -10, "manipulation": +5, "player_corruption": -5},
                "outcome": "You maintain some independence from {npc_name}'s influence. She seems to accept this with grace, though you notice new, more subtle approaches to integration."
            }
        ]
    }
]

# Define relationship rituals that mark significant milestones
RELATIONSHIP_RITUALS = [
    {
        "name": "Formal Agreement",
        "description": "A formalized arrangement that defines relationship expectations",
        "trigger_level": 60,
        "dynamics": ["control", "alliance"],
        "ritual_text": "{npc_name} presents you with an arrangement that feels strangely formal. 'Let's be clear about our expectations,' she says with an intensity that makes it more than casual. The terms feel reasonable, almost beneficial, yet something about the ritual makes you acutely aware of a threshold being crossed."
    },
    {
        "name": "Symbolic Gift",
        "description": "A gift with deeper symbolic meaning that represents the relationship dynamic",
        "trigger_level": 50,
        "dynamics": ["control", "dependency"],
        "ritual_text": "{npc_name} presents you with a gift - {gift_item}. 'A small token,' she says, though her expression suggests deeper significance. As you accept it, the weight feels heavier than the object itself, as if you're accepting something beyond the physical item."
    },
    {
        "name": "Private Ceremony",
        "description": "A private ritual that solidifies the relationship's nature",
        "trigger_level": 75,
        "dynamics": ["control", "dependency", "manipulation"],
        "ritual_text": "{npc_name} leads you through what she describes as 'a small tradition' - a sequence of actions and words that feels choreographed for effect. The intimacy of the moment creates a strange blend of comfort and vulnerability, like something is being sealed between you."
    },
    {
        "name": "Public Declaration",
        "description": "A public acknowledgment of the relationship's significance",
        "trigger_level": 70,
        "dynamics": ["alliance", "control"],
        "ritual_text": "At the gathering, {npc_name} makes a point of publicly acknowledging your relationship in front of others. The words seem innocuous, even complimentary, but the subtext feels laden with meaning. You notice others' reactions - knowing glances, subtle nods - as if your position has been formally established."
    },
    {
        "name": "Shared Secret",
        "description": "Disclosure of sensitive information that creates mutual vulnerability",
        "trigger_level": 55,
        "dynamics": ["alliance", "manipulation"],
        "ritual_text": "{npc_name} shares information with you that feels dangerously private. 'I don't tell this to just anyone,' she says with meaningful eye contact. The knowledge creates an intimacy that comes with implicit responsibility - you're now a keeper of her secrets, for better or worse."
    }
]

# Define symbolic gift items for the Symbolic Gift ritual
SYMBOLIC_GIFTS = [
    "a delicate bracelet that feels oddly like a subtle marker of ownership",
    "a key to her home that represents more than just physical access",
    "a personalized item that shows how closely she's been observing your habits",
    "a journal with the first few pages already filled in her handwriting, guiding your thoughts",
    "a piece of jewelry that others in her circle would recognize as significant",
    "a custom phone with 'helpful' modifications already installed",
    "a clothing item that subtly alters how others perceive you in her presence"
]

def get_relationship_dynamic_level(user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id, dynamic_name):
    """
    Get the current level of a specific relationship dynamic between two entities
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT dynamics FROM SocialLinks
            WHERE user_id=%s AND conversation_id=%s
            AND entity1_type=%s AND entity1_id=%s
            AND entity2_type=%s AND entity2_id=%s
        """, (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id))
        
        row = cursor.fetchone()
        if not row or not row[0]:
            return 0
            
        dynamics = {}
        if isinstance(row[0], str):
            try:
                dynamics = json.loads(row[0])
            except:
                dynamics = {}
        else:
            dynamics = row[0]
            
        return dynamics.get(dynamic_name, 0)
        
    except Exception as e:
        logging.error(f"Error getting relationship dynamic level: {e}")
        return 0
    finally:
        cursor.close()
        conn.close()

def update_relationship_dynamic(user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id, dynamic_name, change):
    """
    Update the level of a specific relationship dynamic between two entities
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # First get current dynamics
        cursor.execute("""
            SELECT link_id, dynamics FROM SocialLinks
            WHERE user_id=%s AND conversation_id=%s
            AND entity1_type=%s AND entity1_id=%s
            AND entity2_type=%s AND entity2_id=%s
        """, (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id))
        
        row = cursor.fetchone()
        if not row:
            # Create new social link if it doesn't exist
            cursor.execute("""
                INSERT INTO SocialLinks (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id, link_type, link_level, dynamics, link_history)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING link_id
            """, (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id, dynamic_name, 0, '{}', '[]'))
            
            link_id = cursor.fetchone()[0]
            dynamics = {}
        else:
            link_id, dynamics_json = row
            
            if isinstance(dynamics_json, str):
                try:
                    dynamics = json.loads(dynamics_json)
                except:
                    dynamics = {}
            else:
                dynamics = dynamics_json or {}
        
        # Update the specific dynamic
        current_level = dynamics.get(dynamic_name, 0)
        new_level = max(0, min(100, current_level + change))  # Clamp between 0-100
        dynamics[dynamic_name] = new_level
        
        # Update in database
        cursor.execute("""
            UPDATE SocialLinks
            SET dynamics = %s
            WHERE link_id = %s
        """, (json.dumps(dynamics), link_id))
        
        # If this is the primary dynamic, also update link_type and link_level
        if get_primary_dynamic(dynamics) == dynamic_name:
            cursor.execute("""
                UPDATE SocialLinks
                SET link_type = %s, link_level = %s
                WHERE link_id = %s
            """, (dynamic_name, new_level, link_id))
        
        conn.commit()
        return new_level
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error updating relationship dynamic: {e}")
        return 0
    finally:
        cursor.close()
        conn.close()

def get_primary_dynamic(dynamics):
    """
    Determine the primary relationship dynamic based on levels
    """
    if not dynamics:
        return "neutral"
        
    max_level = 0
    primary_dynamic = "neutral"
    
    for dynamic, level in dynamics.items():
        if level > max_level:
            max_level = level
            primary_dynamic = dynamic
            
    return primary_dynamic

def get_dynamic_description(dynamic_name, level):
    """
    Get the appropriate description for a dynamic at a specific level
    """
    for dynamic in RELATIONSHIP_DYNAMICS:
        if dynamic["name"] == dynamic_name:
            for level_info in dynamic["levels"]:
                if level <= level_info["level"]:
                    return f"{level_info['name']}: {level_info['description']}"
            
            # If no matching level found, return the highest
            highest_level = dynamic["levels"][-1]
            return f"{highest_level['name']}: {highest_level['description']}"
            
    return "Unknown dynamic"

def check_for_relationship_crossroads(user_id, conversation_id):
    """
    Check if any NPC relationships have reached a crossroads point
    Returns a crossroads event if triggered, None otherwise
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get social links involving the player
        cursor.execute("""
            SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id, dynamics, experienced_crossroads
            FROM SocialLinks
            WHERE user_id=%s AND conversation_id=%s
            AND ((entity1_type='player' AND entity1_id=%s) OR 
                 (entity2_type='player' AND entity2_id=%s))
        """, (user_id, conversation_id, user_id, user_id))
        
        player_links = cursor.fetchall()
        
        for link in player_links:
            link_id, e1_type, e1_id, e2_type, e2_id, dynamics_json, experienced_crossroads_json = link
            
            # Parse dynamics
            if isinstance(dynamics_json, str):
                try:
                    dynamics = json.loads(dynamics_json)
                except:
                    dynamics = {}
            else:
                dynamics = dynamics_json or {}
                
            # Parse experienced crossroads
            if experienced_crossroads_json:
                if isinstance(experienced_crossroads_json, str):
                    try:
                        experienced_crossroads = json.loads(experienced_crossroads_json)
                    except:
                        experienced_crossroads = []
                else:
                    experienced_crossroads = experienced_crossroads_json
            else:
                experienced_crossroads = []
            
            # Find NPC details
            npc_type = e1_type if e2_type == 'player' else e2_type
            npc_id = e1_id if e2_type == 'player' else e2_id
            
            if npc_type != 'npc':
                continue
                
            cursor.execute("""
                SELECT npc_name FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
            """, (user_id, conversation_id, npc_id))
            
            npc_row = cursor.fetchone()
            if not npc_row:
                continue
                
            npc_name = npc_row[0]
            
            # Check each crossroads
            for crossroads in RELATIONSHIP_CROSSROADS:
                # Skip if already experienced
                if crossroads["name"] in experienced_crossroads:
                    continue
                    
                # Check if the trigger dynamic is at appropriate level
                dynamic_level = dynamics.get(crossroads["dynamic"], 0)
                
                if dynamic_level >= crossroads["trigger_level"]:
                    # This crossroads is triggered!
                    
                    # Format choices
                    formatted_choices = []
                    for choice in crossroads["choices"]:
                        formatted_choice = {
                            "text": choice["text"],
                            "effects": choice["effects"],
                            "outcome": choice["outcome"].format(npc_name=npc_name)
                        }
                        formatted_choices.append(formatted_choice)
                    
                    crossroads_event = {
                        "type": "relationship_crossroads",
                        "name": crossroads["name"],
                        "description": crossroads["description"],
                        "npc_id": npc_id,
                        "npc_name": npc_name,
                        "choices": formatted_choices,
                        "link_id": link_id
                    }
                    
                    return crossroads_event
        
        return None
        
    except Exception as e:
        logging.error(f"Error checking for relationship crossroads: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def apply_crossroads_choice(user_id, conversation_id, link_id, crossroads_name, choice_index):
    """
    Apply the effects of a crossroads choice
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get the crossroads definition
        crossroads = None
        for cr in RELATIONSHIP_CROSSROADS:
            if cr["name"] == crossroads_name:
                crossroads = cr
                break
                
        if not crossroads:
            return {"error": f"Crossroads '{crossroads_name}' not found"}
            
        # Validate choice index
        if choice_index < 0 or choice_index >= len(crossroads["choices"]):
            return {"error": "Invalid choice index"}
            
        choice = crossroads["choices"][choice_index]
        
        # Get link details
        cursor.execute("""
            SELECT entity1_type, entity1_id, entity2_type, entity2_id, dynamics, experienced_crossroads
            FROM SocialLinks
            WHERE link_id=%s AND user_id=%s AND conversation_id=%s
        """, (link_id, user_id, conversation_id))
        
        row = cursor.fetchone()
        if not row:
            return {"error": "Social link not found"}
            
        e1_type, e1_id, e2_type, e2_id, dynamics_json, experienced_crossroads_json = row
        
        # Find NPC details
        npc_type = e1_type if e2_type == 'player' else e2_type
        npc_id = e1_id if e2_type == 'player' else e2_id
        
        if npc_type != 'npc':
            return {"error": "No NPC found in this relationship"}
            
        cursor.execute("""
            SELECT npc_name FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (user_id, conversation_id, npc_id))
        
        npc_row = cursor.fetchone()
        if not npc_row:
            return {"error": "NPC not found"}
            
        npc_name = npc_row[0]
        
        # Parse dynamics
        if isinstance(dynamics_json, str):
            try:
                dynamics = json.loads(dynamics_json)
            except:
                dynamics = {}
        else:
            dynamics = dynamics_json or {}
            
        # Parse experienced crossroads
        if experienced_crossroads_json:
            if isinstance(experienced_crossroads_json, str):
                try:
                    experienced_crossroads = json.loads(experienced_crossroads_json)
                except:
                    experienced_crossroads = []
            else:
                experienced_crossroads = experienced_crossroads_json
        else:
            experienced_crossroads = []
        
        # Apply effects to relationship dynamics
        for dynamic, change in choice["effects"].items():
            # Player stat effects
            if dynamic.startswith("player_"):
                stat_name = dynamic[7:]  # Remove "player_" prefix
                
                cursor.execute("""
                    UPDATE PlayerStats
                    SET {stat} = {stat} + %s
                    WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
                """.format(stat=stat_name), (change, user_id, conversation_id))
            else:
                # Relationship dynamic effects
                current_level = dynamics.get(dynamic, 0)
                new_level = max(0, min(100, current_level + change))  # Clamp between 0-100
                dynamics[dynamic] = new_level
        
        # Add to experienced crossroads
        experienced_crossroads.append(crossroads_name)
        
        # Update link in database
        cursor.execute("""
            UPDATE SocialLinks
            SET dynamics = %s, experienced_crossroads = %s
            WHERE link_id = %s
        """, (json.dumps(dynamics), json.dumps(experienced_crossroads), link_id))
        
        # Update primary dynamic
        primary_dynamic = get_primary_dynamic(dynamics)
        primary_level = dynamics.get(primary_dynamic, 0)
        
        cursor.execute("""
            UPDATE SocialLinks
            SET link_type = %s, link_level = %s
            WHERE link_id = %s
        """, (primary_dynamic, primary_level, link_id))
        
        # Add to link history
        event_text = f"Relationship crossroads: {crossroads_name}. Choice made: {choice['text']}. Outcome: {choice['outcome'].format(npc_name=npc_name)}"
        
        cursor.execute("""
            UPDATE SocialLinks
            SET link_history = link_history || %s::jsonb
            WHERE link_id = %s
        """, (json.dumps([event_text]), link_id))
        
        # Add journal entry
        cursor.execute("""
            INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, timestamp)
            VALUES (%s, %s, 'relationship_crossroads', %s, CURRENT_TIMESTAMP)
        """, (user_id, conversation_id, f"Relationship with {npc_name} reached a turning point: {crossroads_name}. {choice['outcome'].format(npc_name=npc_name)}"))
        
        conn.commit()
        
        return {
            "success": True,
            "outcome_text": choice["outcome"].format(npc_name=npc_name)
        }
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error applying crossroads choice: {e}")
        return {"error": str(e)}
    finally:
        cursor.close()
        conn.close()

def check_for_relationship_ritual(user_id, conversation_id):
    """
    Check if any NPC relationships have reached a point for a relationship ritual
    Returns a ritual event if triggered, None otherwise
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get social links involving the player
        cursor.execute("""
            SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id, dynamics, experienced_rituals
            FROM SocialLinks
            WHERE user_id=%s AND conversation_id=%s
            AND ((entity1_type='player' AND entity1_id=%s) OR 
                 (entity2_type='player' AND entity2_id=%s))
        """, (user_id, conversation_id, user_id, user_id))
        
        player_links = cursor.fetchall()
        
        for link in player_links:
            link_id, e1_type, e1_id, e2_type, e2_id, dynamics_json, experienced_rituals_json = link
            
            # Parse dynamics
            if isinstance(dynamics_json, str):
                try:
                    dynamics = json.loads(dynamics_json)
                except:
                    dynamics = {}
            else:
                dynamics = dynamics_json or {}
                
            # Parse experienced rituals
            if experienced_rituals_json:
                if isinstance(experienced_rituals_json, str):
                    try:
                        experienced_rituals = json.loads(experienced_rituals_json)
                    except:
                        experienced_rituals = []
                else:
                    experienced_rituals = experienced_rituals_json
            else:
                experienced_rituals = []
            
            # Find NPC details
            npc_type = e1_type if e2_type == 'player' else e2_type
            npc_id = e1_id if e2_type == 'player' else e2_id
            
            if npc_type != 'npc':
                continue
                
            cursor.execute("""
                SELECT npc_name, dominance FROM NPCStats
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
            """, (user_id, conversation_id, npc_id))
            
            npc_row = cursor.fetchone()
            if not npc_row:
                continue
                
            npc_name, npc_dominance = npc_row
            
            # Only NPCs with sufficient dominance initiate rituals
            if npc_dominance < 50:
                continue
            
            # Check each ritual
            eligible_rituals = []
            
            for ritual in RELATIONSHIP_RITUALS:
                # Skip if already experienced
                if ritual["name"] in experienced_rituals:
                    continue
                    
                # Check if any qualifying dynamics are at appropriate level
                for dynamic_name in ritual["dynamics"]:
                    dynamic_level = dynamics.get(dynamic_name, 0)
                    
                    if dynamic_level >= ritual["trigger_level"]:
                        eligible_rituals.append(ritual)
                        break
            
            if eligible_rituals:
                # Select a random eligible ritual
                chosen_ritual = random.choice(eligible_rituals)
                
                # Format ritual text
                ritual_text = chosen_ritual["ritual_text"]
                
                if "{gift_item}" in ritual_text:
                    gift_item = random.choice(SYMBOLIC_GIFTS)
                    ritual_text = ritual_text.format(npc_name=npc_name, gift_item=gift_item)
                else:
                    ritual_text = ritual_text.format(npc_name=npc_name)
                
                ritual_event = {
                    "type": "relationship_ritual",
                    "name": chosen_ritual["name"],
                    "description": chosen_ritual["description"],
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "ritual_text": ritual_text,
                    "link_id": link_id
                }
                
                # Apply effects
                experienced_rituals.append(chosen_ritual["name"])
                
                cursor.execute("""
                    UPDATE SocialLinks
                    SET experienced_rituals = %s
                    WHERE link_id = %s
                """, (json.dumps(experienced_rituals), link_id))
                
                # Add to link history
                event_text = f"Relationship ritual: {chosen_ritual['name']}. {ritual_text}"
                
                cursor.execute("""
                    UPDATE SocialLinks
                    SET link_history = link_history || %s::jsonb
                    WHERE link_id = %s
                """, (json.dumps([event_text]), link_id))
                
                # Add journal entry
                cursor.execute("""
                    INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, timestamp)
                    VALUES (%s, %s, 'relationship_ritual', %s, CURRENT_TIMESTAMP)
                """, (user_id, conversation_id, f"Ritual with {npc_name}: {chosen_ritual['name']}. {ritual_text}"))
                
                # Increase dynamic levels
                for dynamic_name in chosen_ritual["dynamics"]:
                    current_level = dynamics.get(dynamic_name, 0)
                    new_level = min(100, current_level + 10)  # +10 to each relevant dynamic
                    dynamics[dynamic_name] = new_level
                
                cursor.execute("""
                    UPDATE SocialLinks
                    SET dynamics = %s
                    WHERE link_id = %s
                """, (json.dumps(dynamics), link_id))
                
                # Also update player stats
                cursor.execute("""
                    UPDATE PlayerStats
                    SET corruption = corruption + 5,
                        dependency = dependency + 5
                    WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
                """, (user_id, conversation_id))
                
                conn.commit()
                
                return ritual_event
        
        return None
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error checking for relationship ritual: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def get_relationship_summary(user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id):
    """
    Get a summary of the relationship between two entities
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT link_id, link_type, link_level, dynamics, link_history,
                   experienced_crossroads, experienced_rituals
            FROM SocialLinks
            WHERE user_id=%s AND conversation_id=%s
            AND entity1_type=%s AND entity1_id=%s
            AND entity2_type=%s AND entity2_id=%s
        """, (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id))
        
        row = cursor.fetchone()
        if not row:
            return None
            
        link_id, link_type, link_level, dynamics_json, history_json, crossroads_json, rituals_json = row
        
        # Parse JSON fields
        dynamics = {}
        if isinstance(dynamics_json, str):
            try:
                dynamics = json.loads(dynamics_json)
            except:
                dynamics = {}
        else:
            dynamics = dynamics_json or {}
            
        history = []
        if history_json:
            if isinstance(history_json, str):
                try:
                    history = json.loads(history_json)
                except:
                    history = []
            else:
                history = history_json
        
        crossroads = []
        if crossroads_json:
            if isinstance(crossroads_json, str):
                try:
                    crossroads = json.loads(crossroads_json)
                except:
                    crossroads = []
            else:
                crossroads = crossroads_json
                
        rituals = []
        if rituals_json:
            if isinstance(rituals_json, str):
                try:
                    rituals = json.loads(rituals_json)
                except:
                    rituals = []
            else:
                rituals = rituals_json
        
        # Get entity names
        entity1_name = get_entity_name(conn, entity1_type, entity1_id, user_id, conversation_id)
        entity2_name = get_entity_name(conn, entity2_type, entity2_id, user_id, conversation_id)
        
        # Build dynamic descriptions
        dynamic_descriptions = []
        for dynamic_name, level in dynamics.items():
            for dynamic in RELATIONSHIP_DYNAMICS:
                if dynamic["name"] == dynamic_name:
                    for level_info in dynamic["levels"]:
                        if level <= level_info["level"]:
                            dynamic_descriptions.append(f"{dynamic_name.capitalize()}: {level_info['name']} ({level}/100) - {level_info['description']}")
                            break
        
        # Build summary
        summary = {
            "entity1_name": entity1_name,
            "entity2_name": entity2_name,
            "primary_type": link_type,
            "primary_level": link_level,
            "dynamics": dynamics,
            "dynamic_descriptions": dynamic_descriptions,
            "history": history[-5:],  # Last 5 events
            "experienced_crossroads": crossroads,
            "experienced_rituals": rituals
        }
        
        return summary
        
    except Exception as e:
        logging.error(f"Error getting relationship summary: {e}")
        return None
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
