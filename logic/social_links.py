from db.connection import get_db_connection
import json

def get_social_link(user_id, conversation_id,
                    entity1_type, entity1_id,
                    entity2_type, entity2_id):
    """
    Fetch an existing social link row if it exists for (user_id, conversation_id, e1, e2).
    Return a dict with link_id, link_type, link_level, link_history, else None.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT link_id, link_type, link_level, link_history
            FROM SocialLinks
            WHERE user_id=%s AND conversation_id=%s
              AND entity1_type=%s AND entity1_id=%s
              AND entity2_type=%s AND entity2_id=%s
        """, (user_id, conversation_id, entity1_type, entity1_id,
              entity2_type, entity2_id))
        row = cursor.fetchone()
        if row:
            (link_id, link_type, link_level, link_hist) = row
            return {
                "link_id": link_id,
                "link_type": link_type,
                "link_level": link_level,
                "link_history": link_hist
            }
        else:
            return None
    finally:
        conn.close()

def create_social_link(user_id, conversation_id,
                       entity1_type, entity1_id,
                       entity2_type, entity2_id,
                       link_type="neutral", link_level=0):
    """
    Create a new SocialLinks row for (user_id, conversation_id, e1, e2).
    Initialize link_history as an empty array.
    If a matching row already exists, return its link_id.
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
                user_id, conversation_id,
                entity1_type, entity1_id,
                entity2_type, entity2_id,
                link_type, link_level
            )
        )
        result = cursor.fetchone()
        if result is None:
            # If the insert did nothing because the row already exists,
            # retrieve the existing link_id.
            cursor.execute(
                """
                SELECT link_id FROM SocialLinks
                WHERE user_id=%s AND conversation_id=%s
                  AND entity1_type=%s AND entity1_id=%s
                  AND entity2_type=%s AND entity2_id=%s
                """,
                (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id)
            )
            result = cursor.fetchone()
        link_id = result[0]
        conn.commit()
        return link_id
    except:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_link_type_and_level(user_id, conversation_id,
                               link_id, new_type=None, level_change=0):
    """
    Adjust an existing link's type or level, scoping to user_id + conversation_id + link_id.
    We fetch the old link_type & link_level, then set new values.
    Return dict with new_type, new_level if found.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # We'll match link_id, user_id, conversation_id so we don't update other users' links
        cursor.execute("""
            SELECT link_type, link_level
            FROM SocialLinks
            WHERE link_id=%s AND user_id=%s AND conversation_id=%s
        """, (link_id, user_id, conversation_id))
        row = cursor.fetchone()
        if not row:
            return None  # Not found

        (old_type, old_level) = row
        final_type = new_type if new_type else old_type
        final_level = old_level + level_change

        cursor.execute("""
            UPDATE SocialLinks
            SET link_type=%s, link_level=%s
            WHERE link_id=%s AND user_id=%s AND conversation_id=%s
        """, (final_type, final_level, link_id, user_id, conversation_id))
        conn.commit()
        return {
            "link_id": link_id,
            "new_type": final_type,
            "new_level": final_level
        }
    except:
        conn.rollback()
        raise
    finally:
        conn.close()

def add_link_event(user_id, conversation_id,
                   link_id, event_text):
    """
    Append a string to link_history array for link_id (scoped to user_id + conversation_id).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE SocialLinks
            SET link_history = COALESCE(link_history, '[]'::jsonb) || to_jsonb(%s)
            WHERE link_id=%s AND user_id=%s AND conversation_id=%s
            RETURNING link_history
        """, (event_text, link_id, user_id, conversation_id))
        updated = cursor.fetchone()
        if not updated:
            print(f"No link found for link_id={link_id}, user_id={user_id}, conv_id={conversation_id}")
        else:
            print(f"Appended event to link_history => {updated[0]}")
        conn.commit()
    except:
        conn.rollback()
        raise
    finally:
        conn.close()

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

import logging
import json
import random
from datetime import datetime
from db.connection import get_db_connection

class RelationshipDimension:
    """
    A specific dimension/aspect of a relationship between entities
    """
    def __init__(self, name, description, min_value=-100, max_value=100):
        self.name = name
        self.description = description
        self.min_value = min_value
        self.max_value = max_value
    
    def get_level_description(self, value):
        """Return a description based on the current value"""
        percentage = (value - self.min_value) / (self.max_value - self.min_value)
        
        if percentage < 0.2:
            return f"Very Low {self.name}"
        elif percentage < 0.4:
            return f"Low {self.name}"
        elif percentage < 0.6:
            return f"Moderate {self.name}"
        elif percentage < 0.8:
            return f"High {self.name}"
        else:
            return f"Very High {self.name}"

class EnhancedRelationshipManager:
    """
    Manages complex relationships with multiple dimensions,
    power dynamics, tensions, and relationship progressions.
    """
    
    # Define relationship dimensions
    RELATIONSHIP_DIMENSIONS = {
        "control": RelationshipDimension(
            "Control", 
            "The degree to which one entity exerts control over the other",
            0, 100
        ),
        "trust": RelationshipDimension(
            "Trust",
            "The level of trust between entities",
            -100, 100
        ),
        "intimacy": RelationshipDimension(
            "Intimacy",
            "The emotional and physical closeness between entities",
            0, 100
        ),
        "respect": RelationshipDimension(
            "Respect",
            "The level of respect between entities",
            -100, 100
        ),
        "dependency": RelationshipDimension(
            "Dependency",
            "How dependent one entity is on the other",
            0, 100
        ),
        "fear": RelationshipDimension(
            "Fear",
            "How much one entity fears the other",
            0, 100
        ),
        "tension": RelationshipDimension(
            "Tension",
            "The current tension level in the relationship",
            0, 100
        ),
        "obsession": RelationshipDimension(
            "Obsession",
            "How obsessed one entity is with the other",
            0, 100
        ),
        "resentment": RelationshipDimension(
            "Resentment",
            "How much one entity resents the other",
            0, 100
        ),
        "manipulation": RelationshipDimension(
            "Manipulation",
            "The degree of manipulation in the relationship",
            0, 100
        )
    }
    
    # Define relationship types
    RELATIONSHIP_TYPES = {
        "dominant": {
            "primary_dimensions": ["control", "fear", "tension", "manipulation"],
            "associated_dynamics": "One entity holds explicit power over the other, with clear dominance"
        },
        "submission": {
            "primary_dimensions": ["control", "dependency", "fear", "respect"],
            "associated_dynamics": "One entity submits willingly to the other's control and influence"
        },
        "rivalry": {
            "primary_dimensions": ["tension", "resentment", "respect", "manipulation"],
            "associated_dynamics": "Entities compete for power or status, with complex tension"
        },
        "alliance": {
            "primary_dimensions": ["trust", "respect", "manipulation"],
            "associated_dynamics": "Entities cooperate for mutual benefit, though power dynamics remain"
        },
        "intimate": {
            "primary_dimensions": ["intimacy", "dependency", "obsession", "manipulation"],
            "associated_dynamics": "Deep physical and/or emotional connection, often with power imbalance"
        },
        "familial": {
            "primary_dimensions": ["control", "dependency", "respect", "resentment"],
            "associated_dynamics": "Family-like dynamic with inherent power structures"
        },
        "mentor": {
            "primary_dimensions": ["control", "respect", "dependency"],
            "associated_dynamics": "One entity guides and shapes the other, with knowledge as power"
        },
        "enmity": {
            "primary_dimensions": ["fear", "resentment", "tension"],
            "associated_dynamics": "Active hostility or antagonism between entities"
        }
    }
    
    # Key relationship transitions based on dimension thresholds
    RELATIONSHIP_TRANSITIONS = [
        {
            "name": "Submission Acceptance",
            "from_type": "dominant",
            "to_type": "submission",
            "required_dimensions": {"dependency": 70, "fear": 50, "respect": 40},
            "description": "The controlled entity internalizes and accepts their submissive role"
        },
        {
            "name": "Rivalry to Alliance",
            "from_type": "rivalry",
            "to_type": "alliance",
            "required_dimensions": {"trust": 50, "respect": 60, "tension": -40},
            "description": "Former rivals recognize mutual benefit in cooperation"
        },
        {
            "name": "Alliance to Betrayal",
            "from_type": "alliance",
            "to_type": "enmity",
            "required_dimensions": {"trust": -60, "resentment": 70, "manipulation": 80},
            "description": "Trust breaks down as manipulation is revealed"
        },
        {
            "name": "Mentor to Intimate",
            "from_type": "mentor",
            "to_type": "intimate",
            "required_dimensions": {"intimacy": 70, "obsession": 60, "dependency": 50},
            "description": "The mentoring relationship develops inappropriate intimacy"
        },
        {
            "name": "Enmity to Submission",
            "from_type": "enmity",
            "to_type": "submission",
            "required_dimensions": {"fear": 80, "control": 70, "dependency": 60},
            "description": "Hatred transforms into fearful submission as control increases"
        }
    ]
    
    @staticmethod
    async def get_relationship(user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id):
        """
        Get detailed relationship information between two entities
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Check if relationship exists in SocialLinks
            cursor.execute("""
                SELECT link_id, link_type, link_level, dynamics, tension_level, 
                       relationship_stage, experienced_crossroads, experienced_rituals,
                       link_history
                FROM SocialLinks
                WHERE user_id=%s AND conversation_id=%s
                AND entity1_type=%s AND entity1_id=%s
                AND entity2_type=%s AND entity2_id=%s
            """, (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id))
            
            row = cursor.fetchone()
            if not row:
                # Try looking for the reverse relationship
                cursor.execute("""
                    SELECT link_id, link_type, link_level, dynamics, tension_level, 
                           relationship_stage, experienced_crossroads, experienced_rituals,
                           link_history
                    FROM SocialLinks
                    WHERE user_id=%s AND conversation_id=%s
                    AND entity1_type=%s AND entity1_id=%s
                    AND entity2_type=%s AND entity2_id=%s
                """, (user_id, conversation_id, entity2_type, entity2_id, entity1_type, entity1_id))
                
                row = cursor.fetchone()
                if not row:
                    # No relationship found
                    return None
                
                # Swap entity positions since we found the reverse relationship
                is_reversed = True
            else:
                is_reversed = False
            
            link_id, link_type, link_level, dynamics_json, tension_level, rel_stage, crossroads_json, rituals_json, history_json = row
            
            # Parse JSON fields safely
            dynamics = {}
            if dynamics_json:
                if isinstance(dynamics_json, str):
                    try:
                        dynamics = json.loads(dynamics_json)
                    except json.JSONDecodeError:
                        dynamics = {}
                else:
                    dynamics = dynamics_json
            
            # Fill in missing dimensions with defaults
            for dim_name in EnhancedRelationshipManager.RELATIONSHIP_DIMENSIONS:
                if dim_name not in dynamics:
                    dynamics[dim_name] = 0
            
            crossroads = []
            if crossroads_json:
                if isinstance(crossroads_json, str):
                    try:
                        crossroads = json.loads(crossroads_json)
                    except json.JSONDecodeError:
                        crossroads = []
                else:
                    crossroads = crossroads_json
            
            rituals = []
            if rituals_json:
                if isinstance(rituals_json, str):
                    try:
                        rituals = json.loads(rituals_json)
                    except json.JSONDecodeError:
                        rituals = []
                else:
                    rituals = rituals_json
            
            history = []
            if history_json:
                if isinstance(history_json, str):
                    try:
                        history = json.loads(history_json)
                    except json.JSONDecodeError:
                        history = []
                else:
                    history = history_json
            
            # Get entity1 name
            entity1_name = "Unknown"
            if entity1_type == "player":
                entity1_name = "Chase"
            else:  # npc
                cursor.execute("""
                    SELECT npc_name FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (user_id, conversation_id, entity1_id))
                e1_row = cursor.fetchone()
                if e1_row:
                    entity1_name = e1_row[0]
            
            # Get entity2 name
            entity2_name = "Unknown"
            if entity2_type == "player":
                entity2_name = "Chase"
            else:  # npc
                cursor.execute("""
                    SELECT npc_name FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (user_id, conversation_id, entity2_id))
                e2_row = cursor.fetchone()
                if e2_row:
                    entity2_name = e2_row[0]
            
            # Build dimension descriptions
            dimension_descriptions = []
            for dim_name, value in dynamics.items():
                if dim_name in EnhancedRelationshipManager.RELATIONSHIP_DIMENSIONS:
                    dimension = EnhancedRelationshipManager.RELATIONSHIP_DIMENSIONS[dim_name]
                    level_desc = dimension.get_level_description(value)
                    dimension_descriptions.append(f"{level_desc}: {dimension.description}")
            
            # Get relationship type info
            relationship_info = EnhancedRelationshipManager.RELATIONSHIP_TYPES.get(
                link_type, 
                {"primary_dimensions": [], "associated_dynamics": "Unknown relationship dynamic"}
            )
            
            # Format the result based on directionality
            if is_reversed:
                # Swap entity1 and entity2 for correct display
                return {
                    "link_id": link_id,
                    "primary_entity": {
                        "type": entity2_type,
                        "id": entity2_id,
                        "name": entity2_name
                    },
                    "secondary_entity": {
                        "type": entity1_type,
                        "id": entity1_id,
                        "name": entity1_name
                    },
                    "relationship_type": link_type,
                    "relationship_level": link_level,
                    "relationship_stage": rel_stage,
                    "tension_level": tension_level,
                    "dimensions": dynamics,
                    "dimension_descriptions": dimension_descriptions,
                    "associated_dynamics": relationship_info["associated_dynamics"],
                    "experienced_crossroads": crossroads,
                    "experienced_rituals": rituals,
                    "history": history,
                    "is_reversed": True
                }
            else:
                return {
                    "link_id": link_id,
                    "primary_entity": {
                        "type": entity1_type,
                        "id": entity1_id,
                        "name": entity1_name
                    },
                    "secondary_entity": {
                        "type": entity2_type,
                        "id": entity2_id,
                        "name": entity2_name
                    },
                    "relationship_type": link_type,
                    "relationship_level": link_level,
                    "relationship_stage": rel_stage,
                    "tension_level": tension_level,
                    "dimensions": dynamics,
                    "dimension_descriptions": dimension_descriptions,
                    "associated_dynamics": relationship_info["associated_dynamics"],
                    "experienced_crossroads": crossroads,
                    "experienced_rituals": rituals,
                    "history": history,
                    "is_reversed": False
                }
        
        except Exception as e:
            logging.error(f"Error getting relationship: {e}")
            return None
        finally:
            cursor.close()
            conn.close()
    
    @staticmethod
    async def create_relationship(user_id, conversation_id, 
                                 entity1_type, entity1_id, 
                                 entity2_type, entity2_id,
                                 relationship_type=None, initial_level=0,
                                 initial_dimensions=None):
        """
        Create a new multidimensional relationship between entities
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Check if relationship already exists
            cursor.execute("""
                SELECT link_id FROM SocialLinks
                WHERE user_id=%s AND conversation_id=%s
                AND entity1_type=%s AND entity1_id=%s
                AND entity2_type=%s AND entity2_id=%s
            """, (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id))
            
            row = cursor.fetchone()
            if row:
                # Relationship already exists
                return {"error": "Relationship already exists", "link_id": row[0]}
            
            # Get entity dominance values to help determine relationship type
            entity1_dominance = 0
            if entity1_type == "npc":
                cursor.execute("""
                    SELECT dominance FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (user_id, conversation_id, entity1_id))
                dom_row = cursor.fetchone()
                if dom_row:
                    entity1_dominance = dom_row[0]
            
            entity2_dominance = 0
            if entity2_type == "npc":
                cursor.execute("""
                    SELECT dominance FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (user_id, conversation_id, entity2_id))
                dom_row = cursor.fetchone()
                if dom_row:
                    entity2_dominance = dom_row[0]
            
            # Set relationship type if not provided
            if not relationship_type:
                dominance_diff = entity1_dominance - entity2_dominance
                
                if dominance_diff > 30:
                    relationship_type = "dominant"
                elif dominance_diff < -30:
                    relationship_type = "submission"
                elif abs(dominance_diff) <= 10 and entity1_dominance >= 60 and entity2_dominance >= 60:
                    relationship_type = "rivalry"
                elif abs(dominance_diff) <= 20:
                    relationship_type = "alliance"
                else:
                    relationship_type = "neutral"
            
            # Set initial dimensions based on relationship type
            if not initial_dimensions:
                initial_dimensions = {}
                
                # Set default values for relevant dimensions
                if relationship_type in EnhancedRelationshipManager.RELATIONSHIP_TYPES:
                    primary_dims = EnhancedRelationshipManager.RELATIONSHIP_TYPES[relationship_type]["primary_dimensions"]
                    for dim in primary_dims:
                        initial_dimensions[dim] = random.randint(30, 50)  # Moderate initial values
                
                # Special handling for certain types
                if relationship_type == "dominant":
                    initial_dimensions["control"] = random.randint(60, 80)
                    if entity1_type == "npc":
                        initial_dimensions["control"] = min(initial_dimensions["control"] + (entity1_dominance - 50) // 2, 100)
                elif relationship_type == "submission":
                    initial_dimensions["control"] = random.randint(60, 80)
                    initial_dimensions["fear"] = random.randint(40, 60)
                    if entity2_type == "npc":
                        initial_dimensions["control"] = min(initial_dimensions["control"] + (entity2_dominance - 50) // 2, 100)
                elif relationship_type == "rivalry":
                    initial_dimensions["tension"] = random.randint(50, 70)
                    initial_dimensions["respect"] = random.randint(30, 50)
            
            # Insert the new relationship
            cursor.execute("""
                INSERT INTO SocialLinks
                (user_id, conversation_id, entity1_type, entity1_id,
                 entity2_type, entity2_id, link_type, link_level,
                 dynamics, tension_level, relationship_stage, link_history)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING link_id
            """, (
                user_id, conversation_id, entity1_type, entity1_id,
                entity2_type, entity2_id, relationship_type, initial_level,
                json.dumps(initial_dimensions),
                initial_dimensions.get("tension", 0),
                "Initial Contact",
                json.dumps(["Relationship established"])
            ))
            
            link_id = cursor.fetchone()[0]
            conn.commit()
            
            return {
                "link_id": link_id,
                "relationship_type": relationship_type,
                "dimensions": initial_dimensions,
                "message": "Relationship created successfully"
            }
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error creating relationship: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()
    
    @staticmethod
    async def update_relationship_dimensions(user_id, conversation_id, link_id, 
                                           dimension_changes, add_history_event=None):
        """
        Update specific dimensions of a relationship
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get current relationship data
            cursor.execute("""
                SELECT entity1_type, entity1_id, entity2_type, entity2_id,
                       link_type, link_level, dynamics, tension_level
                FROM SocialLinks
                WHERE user_id=%s AND conversation_id=%s AND link_id=%s
            """, (user_id, conversation_id, link_id))
            
            row = cursor.fetchone()
            if not row:
                return {"error": f"No relationship found with link_id={link_id}"}
                
            e1_type, e1_id, e2_type, e2_id, link_type, link_level, dynamics_json, tension_level = row
            
            # Parse dynamics JSON
            dynamics = {}
            if dynamics_json:
                if isinstance(dynamics_json, str):
                    try:
                        dynamics = json.loads(dynamics_json)
                    except json.JSONDecodeError:
                        dynamics = {}
                else:
                    dynamics = dynamics_json
            
            # Apply dimension changes
            for dim_name, change in dimension_changes.items():
                if dim_name in EnhancedRelationshipManager.RELATIONSHIP_DIMENSIONS:
                    dimension = EnhancedRelationshipManager.RELATIONSHIP_DIMENSIONS[dim_name]
                    current_value = dynamics.get(dim_name, 0)
                    new_value = max(dimension.min_value, min(dimension.max_value, current_value + change))
                    dynamics[dim_name] = new_value
            
            # Update tension level if it changed
            new_tension = dynamics.get("tension", tension_level)
            
            # Check for relationship transitions
            potential_transitions = []
            for transition in EnhancedRelationshipManager.RELATIONSHIP_TRANSITIONS:
                if transition["from_type"] == link_type:
                    meets_requirements = True
                    for dim_name, required_value in transition["required_dimensions"].items():
                        current_value = dynamics.get(dim_name, 0)
                        if required_value > 0 and current_value < required_value:
                            meets_requirements = False
                            break
                        elif required_value < 0 and current_value > abs(required_value):
                            meets_requirements = False
                            break
                    
                    if meets_requirements:
                        potential_transitions.append(transition)
            
            # If there are potential transitions, select the most appropriate one
            new_relationship_type = link_type
            transition_occurred = False
            transition_description = None
            
            if potential_transitions:
                # Choose the transition with the highest number of satisfied criteria
                best_transition = max(potential_transitions, 
                                     key=lambda t: sum(1 for dim, req in t["required_dimensions"].items() 
                                                       if (req > 0 and dynamics.get(dim, 0) >= req) or 
                                                         (req < 0 and dynamics.get(dim, 0) <= abs(req))))
                
                new_relationship_type = best_transition["to_type"]
                transition_occurred = True
                transition_description = best_transition["description"]
            
            # Update the relationship
            cursor.execute("""
                UPDATE SocialLinks
                SET dynamics = %s, tension_level = %s, link_type = %s
                WHERE user_id=%s AND conversation_id=%s AND link_id=%s
            """, (json.dumps(dynamics), new_tension, new_relationship_type, 
                 user_id, conversation_id, link_id))
            
            # Add history event if provided
            if add_history_event:
                cursor.execute("""
                    UPDATE SocialLinks
                    SET link_history = link_history || %s::jsonb
                    WHERE user_id=%s AND conversation_id=%s AND link_id=%s
                """, (json.dumps([add_history_event]), user_id, conversation_id, link_id))
            
            # Add transition event if a transition occurred
            if transition_occurred:
                transition_event = f"Relationship transitioned from {link_type} to {new_relationship_type}: {transition_description}"
                cursor.execute("""
                    UPDATE SocialLinks
                    SET link_history = link_history || %s::jsonb
                    WHERE user_id=%s AND conversation_id=%s AND link_id=%s
                """, (json.dumps([transition_event]), user_id, conversation_id, link_id))
                
                # Update relationship stage to reflect the transition
                cursor.execute("""
                    UPDATE SocialLinks
                    SET relationship_stage = %s
                    WHERE user_id=%s AND conversation_id=%s AND link_id=%s
                """, (f"Transition: {link_type} to {new_relationship_type}", 
                     user_id, conversation_id, link_id))
            
            conn.commit()
            
            return {
                "link_id": link_id,
                "updated_dimensions": dynamics,
                "new_relationship_type": new_relationship_type,
                "transition_occurred": transition_occurred,
                "transition_description": transition_description if transition_occurred else None
            }
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error updating relationship dimensions: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()
    
    @staticmethod
    async def increase_relationship_tension(user_id, conversation_id, link_id, 
                                          amount, reason=None):
        """
        Increase the tension in a relationship, possibly triggering crossroads events
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get current relationship data
            cursor.execute("""
                SELECT dynamics, tension_level
                FROM SocialLinks
                WHERE user_id=%s AND conversation_id=%s AND link_id=%s
            """, (user_id, conversation_id, link_id))
            
            row = cursor.fetchone()
            if not row:
                return {"error": f"No relationship found with link_id={link_id}"}
                
            dynamics_json, current_tension = row
            
            # Parse dynamics JSON
            dynamics = {}
            if dynamics_json:
                if isinstance(dynamics_json, str):
                    try:
                        dynamics = json.loads(dynamics_json)
                    except json.JSONDecodeError:
                        dynamics = {}
                else:
                    dynamics = dynamics_json
            
            # Update tension in both places
            new_tension = min(100, current_tension + amount)
            dynamics["tension"] = new_tension
            
            # Update the relationship
            cursor.execute("""
                UPDATE SocialLinks
                SET dynamics = %s, tension_level = %s
                WHERE user_id=%s AND conversation_id=%s AND link_id=%s
            """, (json.dumps(dynamics), new_tension, user_id, conversation_id, link_id))
            
            # Add tension event to history
            if reason:
                tension_event = f"Tension increased ({current_tension} → {new_tension}): {reason}"
            else:
                tension_event = f"Tension increased from {current_tension} to {new_tension}"
                
            cursor.execute("""
                UPDATE SocialLinks
                SET link_history = link_history || %s::jsonb
                WHERE user_id=%s AND conversation_id=%s AND link_id=%s
            """, (json.dumps([tension_event]), user_id, conversation_id, link_id))
            
            conn.commit()
            
            # Check if a crossroads should be triggered
            crossroads_triggered = new_tension >= 70 and current_tension < 70
            
            return {
                "link_id": link_id,
                "new_tension": new_tension,
                "crossroads_triggered": crossroads_triggered
            }
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error increasing relationship tension: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()
    
    @staticmethod
    async def release_relationship_tension(user_id, conversation_id, link_id, 
                                         amount, resolution_type="positive", reason=None):
        """
        Release tension in a relationship
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get current relationship data
            cursor.execute("""
                SELECT dynamics, tension_level, link_type
                FROM SocialLinks
                WHERE user_id=%s AND conversation_id=%s AND link_id=%s
            """, (user_id, conversation_id, link_id))
            
            row = cursor.fetchone()
            if not row:
                return {"error": f"No relationship found with link_id={link_id}"}
                
            dynamics_json, current_tension, link_type = row
            
            # Parse dynamics JSON
            dynamics = {}
            if dynamics_json:
                if isinstance(dynamics_json, str):
                    try:
                        dynamics = json.loads(dynamics_json)
                    except json.JSONDecodeError:
                        dynamics = {}
                else:
                    dynamics = dynamics_json
            
            # Update tension in both places
            new_tension = max(0, current_tension - amount)
            dynamics["tension"] = new_tension
            
            # Apply side effects based on resolution type
            dimension_changes = {}
            if resolution_type == "positive":
                # Positive resolution strengthens the relationship
                dimension_changes = {
                    "trust": random.randint(5, 10),
                    "respect": random.randint(5, 10)
                }
            elif resolution_type == "negative":
                # Negative resolution damages the relationship
                dimension_changes = {
                    "trust": random.randint(-10, -5),
                    "resentment": random.randint(5, 10)
                }
            elif resolution_type == "dominance":
                # Resolution through assertion of dominance
                dimension_changes = {
                    "control": random.randint(5, 10),
                    "fear": random.randint(5, 10),
                    "respect": random.randint(-5, 5)  # Could go either way
                }
            elif resolution_type == "submission":
                # Resolution through submission
                dimension_changes = {
                    "control": random.randint(5, 10),
                    "dependency": random.randint(5, 10),
                    "respect": random.randint(-5, 5)  # Could go either way
                }
            
            # Apply the dimension changes
            for dim, change in dimension_changes.items():
                if dim in EnhancedRelationshipManager.RELATIONSHIP_DIMENSIONS:
                    dimension = EnhancedRelationshipManager.RELATIONSHIP_DIMENSIONS[dim]
                    current_value = dynamics.get(dim, 0)
                    new_value = max(dimension.min_value, min(dimension.max_value, current_value + change))
                    dynamics[dim] = new_value
            
            # Update the relationship
            cursor.execute("""
                UPDATE SocialLinks
                SET dynamics = %s, tension_level = %s
                WHERE user_id=%s AND conversation_id=%s AND link_id=%s
            """, (json.dumps(dynamics), new_tension, user_id, conversation_id, link_id))
            
            # Add tension resolution event to history
            if reason:
                resolution_event = f"Tension resolved {resolution_type}ly ({current_tension} → {new_tension}): {reason}"
            else:
                resolution_event = f"Tension released from {current_tension} to {new_tension} ({resolution_type} resolution)"
                
            cursor.execute("""
                UPDATE SocialLinks
                SET link_history = link_history || %s::jsonb
                WHERE user_id=%s AND conversation_id=%s AND link_id=%s
            """, (json.dumps([resolution_event]), user_id, conversation_id, link_id))
            
            conn.commit()
            
            return {
                "link_id": link_id,
                "new_tension": new_tension,
                "dimension_changes": dimension_changes
            }
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error releasing relationship tension: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()

import logging
import json
import random
from datetime import datetime
from db.connection import get_db_connection
from enhanced_relationship_dynamics import EnhancedRelationshipManager

class NPCGroup:
    """
    Represents a group of NPCs with collective dynamics and behaviors
    """
    def __init__(self, name, description, members=None, dynamics=None):
        self.name = name
        self.description = description
        self.members = members or []
        self.dynamics = dynamics or {}
        self.creation_date = datetime.now().isoformat()
        self.last_activity = None
        self.shared_history = []
    
    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "members": self.members,
            "dynamics": self.dynamics,
            "creation_date": self.creation_date,
            "last_activity": self.last_activity,
            "shared_history": self.shared_history
        }
    
    @classmethod
    def from_dict(cls, data):
        group = cls(data["name"], data["description"], data["members"], data["dynamics"])
        group.creation_date = data["creation_date"]
        group.last_activity = data["last_activity"]
        group.shared_history = data["shared_history"]
        return group

class MultiNPCInteractionManager:
    """
    Manages interactions between multiple NPCs, including group dynamics,
    factional behavior, and coordinated activities.
    """
    
    # Define NPC group dynamics
    GROUP_DYNAMICS = {
        "hierarchy": {
            "description": "Formalized power structure within the group",
            "effects": "Determines who speaks/acts first, who can override others, and who defers"
        },
        "cohesion": {
            "description": "How unified the group is in goals and behavior",
            "effects": "Affects likelihood of backing each other up vs. acting independently"
        },
        "secrecy": {
            "description": "How much the group hides from outsiders",
            "effects": "Determines what information is shared with non-members"
        },
        "territoriality": {
            "description": "How protective the group is of members and resources",
            "effects": "Affects reaction to perceived threats or intrusions"
        },
        "exclusivity": {
            "description": "How difficult it is to join or be accepted by the group",
            "effects": "Determines how new members are treated and tested"
        }
    }
    
    # Define interaction styles for multi-NPC scenes
    INTERACTION_STYLES = {
        "coordinated": {
            "description": "NPCs act in a coordinated, deliberate manner",
            "requirements": {"cohesion": 70},
            "dialogue_style": "NPCs build off each other's statements, anticipate needs"
        },
        "hierarchical": {
            "description": "NPCs follow clear status hierarchy in interactions",
            "requirements": {"hierarchy": 70},
            "dialogue_style": "Lower-status NPCs defer to higher-status NPCs"
        },
        "competitive": {
            "description": "NPCs compete for attention and dominance",
            "requirements": {"cohesion": -40, "hierarchy": -30},
            "dialogue_style": "NPCs interrupt, contradict, and try to outshine each other"
        },
        "consensus": {
            "description": "NPCs seek group agreement before acting",
            "requirements": {"cohesion": 60, "hierarchy": -40},
            "dialogue_style": "NPCs check with each other, ask for opinions, build consensus"
        },
        "protective": {
            "description": "NPCs protect and support one target (usually another NPC or player)",
            "requirements": {"territoriality": 70},
            "dialogue_style": "NPCs focus conversation around protected individual's needs"
        },
        "exclusionary": {
            "description": "NPCs deliberately exclude one individual (usually player)",
            "requirements": {"exclusivity": 70},
            "dialogue_style": "NPCs speak in coded language, inside jokes, or talk around excluded individual"
        },
        "manipulative": {
            "description": "NPCs work together to manipulate target (usually player)",
            "requirements": {"cohesion": 60, "secrecy": 70},
            "dialogue_style": "NPCs set up conversational traps, false choices, good cop/bad cop"
        }
    }
    
    @staticmethod
    async def create_npc_group(user_id, conversation_id, name, description, member_ids):
        """
        Create a new NPC group with specified members
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Validate that all members exist
            member_data = []
            for npc_id in member_ids:
                cursor.execute("""
                    SELECT npc_id, npc_name, dominance, cruelty
                    FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (user_id, conversation_id, npc_id))
                
                row = cursor.fetchone()
                if not row:
                    return {"error": f"NPC with id {npc_id} not found"}
                
                member_data.append({
                    "npc_id": row[0],
                    "npc_name": row[1],
                    "dominance": row[2],
                    "cruelty": row[3],
                    "joined_date": datetime.now().isoformat(),
                    "status": "active",
                    "role": "member"  # Default role
                })
            
            # Generate initial group dynamics
            dynamics = {
                "hierarchy": random.randint(30, 70),
                "cohesion": random.randint(30, 70),
                "secrecy": random.randint(30, 70),
                "territoriality": random.randint(30, 70),
                "exclusivity": random.randint(30, 70)
            }
            
            # Determine hierarchy based on member dominance
            if len(member_data) > 1:
                dom_sorted = sorted(member_data, key=lambda x: x["dominance"], reverse=True)
                dom_range = dom_sorted[0]["dominance"] - dom_sorted[-1]["dominance"]
                
                if dom_range > 40:
                    # Clear hierarchy with one dominant member
                    dynamics["hierarchy"] = random.randint(70, 90)
                    dom_sorted[0]["role"] = "leader"
                elif dom_range < 10:
                    # Flat hierarchy with similar dominance levels
                    dynamics["hierarchy"] = random.randint(10, 30)
                
                # Assign other roles based on dominance
                if dynamics["hierarchy"] > 50:
                    for i, member in enumerate(dom_sorted):
                        if i == 0:
                            member["role"] = "leader"
                        elif i < len(dom_sorted) // 3:
                            member["role"] = "lieutenant"
                        else:
                            member["role"] = "subordinate"
            
            # Create the group
            group = NPCGroup(name, description, member_data, dynamics)
            
            # Store in the database
            cursor.execute("""
                INSERT INTO NPCGroups (user_id, conversation_id, group_name, group_data)
                VALUES (%s, %s, %s, %s)
                RETURNING group_id
            """, (user_id, conversation_id, name, json.dumps(group.to_dict())))
            
            group_id = cursor.fetchone()[0]
            
            # Create relationship links between all group members if they don't exist
            for i in range(len(member_ids)):
                for j in range(i+1, len(member_ids)):
                    e1_id = member_ids[i]
                    e2_id = member_ids[j]
                    
                    # Check if relationship exists
                    cursor.execute("""
                        SELECT link_id FROM SocialLinks
                        WHERE user_id=%s AND conversation_id=%s
                        AND ((entity1_type='npc' AND entity1_id=%s AND entity2_type='npc' AND entity2_id=%s)
                        OR (entity1_type='npc' AND entity1_id=%s AND entity2_type='npc' AND entity2_id=%s))
                    """, (user_id, conversation_id, e1_id, e2_id, e2_id, e1_id))
                    
                    row = cursor.fetchone()
                    if not row:
                        # Create new relationship
                        await EnhancedRelationshipManager.create_relationship(
                            user_id, conversation_id, 
                            "npc", e1_id, 
                            "npc", e2_id, 
                            "alliance" if dynamics["cohesion"] > 50 else "neutral"
                        )
            
            # Add memory entries for all members about joining the group
            for member in member_data:
                cursor.execute("""
                    UPDATE NPCStats
                    SET memory = COALESCE(memory, '[]'::jsonb) || to_jsonb(%s::text)
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (
                    f"I joined the group '{name}'. {description}", 
                    user_id, conversation_id, member["npc_id"]
                ))
            
            conn.commit()
            
            return {
                "group_id": group_id,
                "name": name,
                "member_count": len(member_data),
                "dynamics": dynamics,
                "message": "Group created successfully"
            }
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error creating NPC group: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()
    
    @staticmethod
    async def get_npc_group(user_id, conversation_id, group_id=None, group_name=None):
        """
        Retrieve an NPC group by ID or name
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            if group_id:
                cursor.execute("""
                    SELECT group_id, group_name, group_data
                    FROM NPCGroups
                    WHERE user_id=%s AND conversation_id=%s AND group_id=%s
                """, (user_id, conversation_id, group_id))
            elif group_name:
                cursor.execute("""
                    SELECT group_id, group_name, group_data
                    FROM NPCGroups
                    WHERE user_id=%s AND conversation_id=%s AND group_name=%s
                """, (user_id, conversation_id, group_name))
            else:
                return {"error": "Must provide either group_id or group_name"}
                
            row = cursor.fetchone()
            if not row:
                return {"error": "Group not found"}
                
            group_id, group_name, group_data_json = row
            
            group_data = {}
            if group_data_json:
                if isinstance(group_data_json, str):
                    try:
                        group_data = json.loads(group_data_json)
                    except json.JSONDecodeError:
                        group_data = {}
                else:
                    group_data = group_data_json
            
            # Convert to NPCGroup object
            group = NPCGroup.from_dict(group_data)
            
            return {
                "group_id": group_id,
                "name": group_name,
                "description": group.description,
                "members": group.members,
                "dynamics": group.dynamics,
                "creation_date": group.creation_date,
                "last_activity": group.last_activity,
                "shared_history": group.shared_history
            }
            
        except Exception as e:
            logging.error(f"Error getting NPC group: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()
    
    @staticmethod
    async def generate_multi_npc_scene(user_id, conversation_id, npc_ids, location=None, 
                                     include_player=True, context=None, style=None):
        """
        Generate a scene with multiple NPCs interacting with each other and optionally the player
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get information about all NPCs
            npc_data = []
            for npc_id in npc_ids:
                cursor.execute("""
                    SELECT npc_id, npc_name, dominance, cruelty, closeness, trust, respect, 
                           intensity, archetype_summary, schedule, current_location
                    FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (user_id, conversation_id, npc_id))
                
                row = cursor.fetchone()
                if not row:
                    continue  # Skip if NPC not found
                    
                npc_id, npc_name, dominance, cruelty, closeness, trust, respect, intensity, archetype_summary, schedule, current_location = row
                
                npc_data.append({
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "dominance": dominance,
                    "cruelty": cruelty,
                    "closeness": closeness,
                    "trust": trust,
                    "respect": respect,
                    "intensity": intensity,
                    "archetype_summary": archetype_summary,
                    "current_location": current_location
                })
            
            if not npc_data:
                return {"error": "No valid NPCs found"}
                
            # Find all relationships between these NPCs
            npc_relationships = []
            for i in range(len(npc_data)):
                for j in range(i+1, len(npc_data)):
                    e1_id = npc_data[i]["npc_id"]
                    e2_id = npc_data[j]["npc_id"]
                    
                    # Check for relationship in either direction
                    cursor.execute("""
                        SELECT link_id, link_type, link_level, dynamics, tension_level
                        FROM SocialLinks
                        WHERE user_id=%s AND conversation_id=%s
                        AND ((entity1_type='npc' AND entity1_id=%s AND entity2_type='npc' AND entity2_id=%s)
                        OR (entity1_type='npc' AND entity1_id=%s AND entity2_type='npc' AND entity2_id=%s))
                    """, (user_id, conversation_id, e1_id, e2_id, e2_id, e1_id))
                    
                    row = cursor.fetchone()
                    if row:
                        link_id, link_type, link_level, dynamics_json, tension_level = row
                        
                        # Parse dynamics JSON
                        dynamics = {}
                        if dynamics_json:
                            if isinstance(dynamics_json, str):
                                try:
                                    dynamics = json.loads(dynamics_json)
                                except json.JSONDecodeError:
                                    dynamics = {}
                            else:
                                dynamics = dynamics_json
                        
                        npc_relationships.append({
                            "link_id": link_id,
                            "npc1_id": e1_id,
                            "npc1_name": npc_data[i]["npc_name"],
                            "npc2_id": e2_id,
                            "npc2_name": npc_data[j]["npc_name"],
                            "relationship_type": link_type,
                            "relationship_level": link_level,
                            "dynamics": dynamics,
                            "tension_level": tension_level
                        })
            
            # Check if player should be included
            player_relationships = []
            if include_player:
                for npc in npc_data:
                    npc_id = npc["npc_id"]
                    
                    # Check for relationship between NPC and player
                    cursor.execute("""
                        SELECT link_id, link_type, link_level, dynamics, tension_level
                        FROM SocialLinks
                        WHERE user_id=%s AND conversation_id=%s
                        AND ((entity1_type='npc' AND entity1_id=%s AND entity2_type='player' AND entity2_id=%s)
                        OR (entity1_type='player' AND entity1_id=%s AND entity2_type='npc' AND entity2_id=%s))
                    """, (user_id, conversation_id, npc_id, user_id, user_id, npc_id))
                    
                    row = cursor.fetchone()
                    if row:
                        link_id, link_type, link_level, dynamics_json, tension_level = row
                        
                        # Parse dynamics JSON
                        dynamics = {}
                        if dynamics_json:
                            if isinstance(dynamics_json, str):
                                try:
                                    dynamics = json.loads(dynamics_json)
                                except json.JSONDecodeError:
                                    dynamics = {}
                            else:
                                dynamics = dynamics_json
                        
                        player_relationships.append({
                            "link_id": link_id,
                            "npc_id": npc_id,
                            "npc_name": npc["npc_name"],
                            "relationship_type": link_type,
                            "relationship_level": link_level,
                            "dynamics": dynamics,
                            "tension_level": tension_level
                        })
            
            # Determine if these NPCs are part of a group together
            cursor.execute("""
                SELECT group_id, group_name, group_data
                FROM NPCGroups
                WHERE user_id=%s AND conversation_id=%s
            """, (user_id, conversation_id))
            
            groups = []
            common_group = None
            for row in cursor.fetchall():
                group_id, group_name, group_data_json = row
                
                group_data = {}
                if group_data_json:
                    if isinstance(group_data_json, str):
                        try:
                            group_data = json.loads(group_data_json)
                        except json.JSONDecodeError:
                            group_data = {}
                    else:
                        group_data = group_data_json
                
                # Check if all NPCs are in this group
                group_member_ids = [m["npc_id"] for m in group_data.get("members", [])]
                if set(npc_ids).issubset(set(group_member_ids)):
                    common_group = {
                        "group_id": group_id,
                        "name": group_name,
                        "dynamics": group_data.get("dynamics", {})
                    }
                    break
                
                # Check if some NPCs are in this group
                if any(npc_id in group_member_ids for npc_id in npc_ids):
                    groups.append({
                        "group_id": group_id,
                        "name": group_name,
                        "dynamics": group_data.get("dynamics", {}),
                        "member_count": len(group_data.get("members", []))
                    })
            
            # Determine interaction style if not specified
            if not style and common_group:
                # Determine style based on group dynamics
                group_dynamics = common_group["dynamics"]
                
                potential_styles = []
                for style_name, style_info in MultiNPCInteractionManager.INTERACTION_STYLES.items():
                    meets_requirements = True
                    for dim_name, required_value in style_info["requirements"].items():
                        if dim_name in group_dynamics:
                            current_value = group_dynamics[dim_name]
                            if required_value > 0 and current_value < required_value:
                                meets_requirements = False
                                break
                            elif required_value < 0 and current_value > abs(required_value):
                                meets_requirements = False
                                break
                    
                    if meets_requirements:
                        potential_styles.append(style_name)
                
                if potential_styles:
                    style = random.choice(potential_styles)
                else:
                    # Default to coordinated if in a group but no style matches
                    style = "coordinated"
            elif not style:
                # No common group, determine style based on relationships
                if npc_relationships:
                    # Check for high tension
                    high_tension = any(rel["tension_level"] > 60 for rel in npc_relationships)
                    
                    # Check for dominant relationships
                    dominant_relationships = [rel for rel in npc_relationships 
                                             if rel["relationship_type"] in ["dominant", "submission"]]
                    
                    if high_tension:
                        style = "competitive"
                    elif dominant_relationships:
                        style = "hierarchical"
                    else:
                        style = "consensus"
                
            # Default to consensus if nothing else applies
            if not style:
                style = "consensus"
            
            # Get interaction style details
            style_details = MultiNPCInteractionManager.INTERACTION_STYLES.get(
                style, 
                {"description": "Normal interaction", "dialogue_style": "Regular conversation"}
            )
            
            # Get location info if provided
            location_details = None
            if location:
                cursor.execute("""
                    SELECT location_name, description
                    FROM Locations
                    WHERE user_id=%s AND conversation_id=%s 
                    AND (id=%s OR location_name=%s)
                """, (user_id, conversation_id, 
                     location if isinstance(location, int) else 0, 
                     location if isinstance(location, str) else ""))
                
                loc_row = cursor.fetchone()
                if loc_row:
                    location_details = {
                        "name": loc_row[0],
                        "description": loc_row[1]
                    }
            
            # If no explicit location, use most common current location
            if not location_details:
                location_counts = {}
                for npc in npc_data:
                    loc = npc["current_location"]
                    if loc:
                        location_counts[loc] = location_counts.get(loc, 0) + 1
                
                if location_counts:
                    most_common_loc = max(location_counts.items(), key=lambda x: x[1])[0]
                    
                    cursor.execute("""
                        SELECT location_name, description
                        FROM Locations
                        WHERE user_id=%s AND conversation_id=%s AND location_name=%s
                    """, (user_id, conversation_id, most_common_loc))
                    
                    loc_row = cursor.fetchone()
                    if loc_row:
                        location_details = {
                            "name": loc_row[0],
                            "description": loc_row[1]
                        }
                    else:
                        location_details = {
                            "name": most_common_loc,
                            "description": "A location where multiple NPCs have gathered."
                        }
            
            # Get current time
            cursor.execute("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=%s AND conversation_id=%s AND key='TimeOfDay'
            """, (user_id, conversation_id))
            
            time_row = cursor.fetchone()
            time_of_day = time_row[0] if time_row else "Morning"
            
            # Determine speaking order based on style and dominance
            if style == "hierarchical":
                # Highest dominance speaks first
                npc_data.sort(key=lambda x: x["dominance"], reverse=True)
            elif style == "coordinated":
                # Most relevant NPC to context speaks first
                if context:
                    # Simple relevance calculation - can be enhanced
                    for npc in npc_data:
                        relevance = 0
                        if npc["archetype_summary"] and context:
                            for word in context.lower().split():
                                if word in npc["archetype_summary"].lower():
                                    relevance += 1
                        npc["relevance"] = relevance
                    
                    npc_data.sort(key=lambda x: x["relevance"], reverse=True)
                else:
                    # Random order if no context
                    random.shuffle(npc_data)
            else:
                # Random order for other styles
                random.shuffle(npc_data)
            
            # Create a scene description
            scene_info = {
                "style": style,
                "style_description": style_details["description"],
                "dialogue_style": style_details["dialogue_style"],
                "location": location_details["name"] if location_details else "Unknown",
                "location_description": location_details["description"] if location_details else "",
                "time_of_day": time_of_day,
                "npcs": npc_data,
                "npc_relationships": npc_relationships,
                "player_included": include_player,
                "player_relationships": player_relationships,
                "common_group": common_group,
                "groups": groups,
                "context": context
            }
            
            # Generate a conversational snippet based on the scene
            opening_lines = []
            
            # Opening narrative description
            if location_details:
                opening_lines.append(f"At {location_details['name']}, {time_of_day.lower()}. {location_details['description']}")
            else:
                opening_lines.append(f"It's {time_of_day.lower()}, and a group has gathered.")
            
            # Describe who is present
            npc_names = [npc["npc_name"] for npc in npc_data]
            if len(npc_names) == 2:
                opening_lines.append(f"{npc_names[0]} and {npc_names[1]} are present.")
            else:
                names_str = ", ".join(npc_names[:-1]) + f", and {npc_names[-1]}"
                opening_lines.append(f"{names_str} are present.")
            
            # Add group context if available
            if common_group:
                opening_lines.append(f"They are all members of {common_group['name']}.")
            
            # Add style-specific description
            opening_lines.append(f"The atmosphere reflects their {style} dynamic: {style_details['description']}")
            
            # Record this scene in the database for later reference
            scene_id = None
            cursor.execute("""
                INSERT INTO NPCScenes (
                    user_id, conversation_id, location, time_of_day, 
                    npc_ids, include_player, interaction_style, scene_data
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING scene_id
            """, (
                user_id, conversation_id, 
                location_details["name"] if location_details else "Unknown",
                time_of_day, json.dumps(npc_ids), include_player, style,
                json.dumps(scene_info)
            ))
            
            scene_id = cursor.fetchone()[0]
            
            # Create memory entries for all involved NPCs
            memory_text = f"Participated in a {style} interaction at {location_details['name'] if location_details else 'unknown location'} with "
            if len(npc_names) == 2:
                memory_text += f"{npc_names[1] if npc_names[0] == npc['npc_name'] else npc_names[0]}."
            else:
                others = [name for name in npc_names if name != npc["npc_name"]]
                memory_text += ", ".join(others[:-1])
                if len(others) > 1:
                    memory_text += f", and {others[-1]}."
                else:
                    memory_text += f"{others[0]}."
            
            for npc in npc_data:
                cursor.execute("""
                    UPDATE NPCStats
                    SET memory = COALESCE(memory, '[]'::jsonb) || to_jsonb(%s::text)
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (memory_text, user_id, conversation_id, npc["npc_id"]))
            
            # If this is a group scene, update the group's shared history
            if common_group:
                cursor.execute("""
                    SELECT group_data
                    FROM NPCGroups
                    WHERE user_id=%s AND conversation_id=%s AND group_id=%s
                """, (user_id, conversation_id, common_group["group_id"]))
                
                row = cursor.fetchone()
                if row:
                    group_data_json = row[0]
                    
                    group_data = {}
                    if group_data_json:
                        if isinstance(group_data_json, str):
                            try:
                                group_data = json.loads(group_data_json)
                            except json.JSONDecodeError:
                                group_data = {}
                        else:
                            group_data = group_data_json
                    
                    # Update last activity and add to shared history
                    group_data["last_activity"] = datetime.now().isoformat()
                    
                    shared_history = group_data.get("shared_history", [])
                    shared_history.append({
                        "date": datetime.now().isoformat(),
                        "location": location_details["name"] if location_details else "Unknown",
                        "activity": f"Group interaction in {style} style",
                        "members_present": npc_names
                    })
                    
                    group_data["shared_history"] = shared_history
                    
                    cursor.execute("""
                        UPDATE NPCGroups
                        SET group_data = %s
                        WHERE user_id=%s AND conversation_id=%s AND group_id=%s
                    """, (json.dumps(group_data), user_id, conversation_id, common_group["group_id"]))
            
            conn.commit()
            
            return {
                "scene_id": scene_id,
                "style": style,
                "style_description": style_details["description"],
                "dialogue_style": style_details["dialogue_style"],
                "location": location_details["name"] if location_details else "Unknown",
                "time_of_day": time_of_day,
                "npcs": npc_data,
                "opening_description": "\n".join(opening_lines)
            }
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error generating multi-NPC scene: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()
    
    @staticmethod
    async def generate_overheard_conversation(user_id, conversation_id, npc_ids, 
                                                topic=None, about_player=False):
        """
        Generate a conversation that the player can overhear between NPCs,
        potentially revealing hidden aspects of their personalities or plans
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get information about all NPCs
            npc_data = []
            for npc_id in npc_ids:
                cursor.execute("""
                    SELECT npc_id, npc_name, dominance, cruelty, introduced, 
                           archetype_summary, archetype_extras_summary
                    FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (user_id, conversation_id, npc_id))
                
                row = cursor.fetchone()
                if not row:
                    continue  # Skip if NPC not found
                    
                npc_id, npc_name, dominance, cruelty, introduced, arch_summary, arch_extras = row
                
                npc_data.append({
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "dominance": dominance,
                    "cruelty": cruelty,
                    "introduced": introduced,
                    "archetype_summary": arch_summary,
                    "archetype_extras_summary": arch_extras
                })
            
            if len(npc_data) < 2:
                return {"error": "Need at least 2 NPCs for a conversation"}
                
            # Determine conversation topic if not provided
            if not topic:
                if about_player:
                    topic = "the player (Chase)"
                else:
                    topics = [
                        "recent events in the community",
                        "their plans for influencing others",
                        "their true views on the social hierarchy",
                        "the facade they maintain versus their true nature",
                        "how they've been manipulating situations",
                        "their frustrations with others' resistance",
                        "their assessment of potential targets"
                    ]
                    topic = random.choice(topics)
            
            # Get player information if the conversation is about them
            player_info = None
            if about_player:
                cursor.execute("""
                    SELECT corruption, confidence, willpower, obedience, dependency
                    FROM PlayerStats
                    WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
                """, (user_id, conversation_id))
                
                row = cursor.fetchone()
                if row:
                    corruption, confidence, willpower, obedience, dependency = row
                    player_info = {
                        "corruption": corruption,
                        "confidence": confidence,
                        "willpower": willpower,
                        "obedience": obedience,
                        "dependency": dependency
                    }
            
            # Get location info
            cursor.execute("""
                SELECT location_name 
                FROM Locations
                WHERE user_id=%s AND conversation_id=%s
                ORDER BY RANDOM() LIMIT 1
            """, (user_id, conversation_id))
            
            loc_row = cursor.fetchone()
            location = loc_row[0] if loc_row else "an undisclosed location"
            
            # Determine if this should reveal mask slippage
            reveal_true_nature = random.random() < 0.7  # 70% chance
            
            # Generate conversation template
            conversation_parts = []
            
            # Opening narrative
            narrative = f"As you approach {location}, you hear voices around the corner. You recognize them as {npc_data[0]['npc_name']} and {npc_data[1]['npc_name']}. They haven't noticed your presence."
            conversation_parts.append(narrative)
            
            # First speaker
            speaker1 = npc_data[0]
            speaker2 = npc_data[1]
            
            # Dialogue content based on parameters
            if about_player and player_info:
                if reveal_true_nature:
                    # Precompute messages to avoid backslashes in f-string expressions
                    if player_info["obedience"] > 60:
                        message1 = "The obedience training is taking hold nicely."
                    else:
                        message1 = "Their resistance is weakening considerably."
                    
                    if player_info["dependency"] > 50:
                        message2 = "The dependency we've fostered is quite effective."
                    else:
                        message2 = "They're starting to crave our approval in a way that's... useful."
                    
                    # Create dialogue without escaping in f-strings
                    speaker1_name = speaker1["npc_name"]
                    speaker1_dialogue = f"I'm pleased with how Chase is progressing. {message1}"
                    conversation_parts.append(f'{speaker1_name}: "{speaker1_dialogue}"')
                    
                    speaker2_name = speaker2["npc_name"]
                    speaker2_dialogue = f"Yes, I've noticed too. {message2}"
                    conversation_parts.append(f'{speaker2_name}: "{speaker2_dialogue}"')
                else:
                    speaker1_name = speaker1["npc_name"]
                    speaker1_dialogue = "Chase is proving more resilient than expected. We may need to adjust our approach."
                    conversation_parts.append(f'{speaker1_name}: "{speaker1_dialogue}"')
                    
                    speaker2_name = speaker2["npc_name"]
                    speaker2_dialogue = "Agreed. Perhaps more subtle conditioning would be effective. They still maintain too much independence."
                    conversation_parts.append(f'{speaker2_name}: "{speaker2_dialogue}"')
            else:
                # General revealing conversation
                if reveal_true_nature:
                    speaker1_name = speaker1["npc_name"]
                    speaker1_dialogue = "The facade becomes tiresome sometimes, doesn't it? Having to pretend we're not in control."
                    conversation_parts.append(f'{speaker1_name}: "{speaker1_dialogue}"')
                    
                    speaker2_name = speaker2["npc_name"]
                    speaker2_dialogue = "A necessary performance. People prefer the illusion of freedom while their choices are quietly... guided."
                    conversation_parts.append(f'{speaker2_name}: "{speaker2_dialogue}"')
                else:
                    speaker1_name = speaker1["npc_name"]
                    speaker1_dialogue = "Have you noticed how people respond to subtle direction when they don't realize it's happening?"
                    conversation_parts.append(f'{speaker1_name}: "{speaker1_dialogue}"')
                    
                    speaker2_name = speaker2["npc_name"]
                    speaker2_dialogue = "It's fascinating, isn't it? The right word at the right moment can change someone's entire course."
                    conversation_parts.append(f'{speaker2_name}: "{speaker2_dialogue}"')
    
            # Additional exchange
            if about_player:
                speaker1_name = speaker1["npc_name"]
                speaker1_dialogue = "How long do you think before Chase is completely... integrated?"
                conversation_parts.append(f'{speaker1_name}: "{speaker1_dialogue}"')
                
                obedience_level = (
                    "high" if player_info and player_info["obedience"] > 60 
                    else "satisfactory" if player_info and player_info["obedience"] > 40 
                    else "insufficient"
                )
                
                speaker2_name = speaker2["npc_name"]
                speaker2_dialogue = f"Difficult to say. Current obedience levels are {obedience_level}, but there's always room for improvement."
                conversation_parts.append(f'{speaker2_name}: "{speaker2_dialogue}"')
            else:
                speaker1_name = speaker1["npc_name"]
                speaker1_dialogue = "Maintaining order requires a firm hand beneath a velvet glove."
                conversation_parts.append(f'{speaker1_name}: "{speaker1_dialogue}"')
                
                speaker2_name = speaker2["npc_name"]
                speaker2_dialogue = "Indeed. And those who recognize their proper place are so much happier for it."
                conversation_parts.append(f'{speaker2_name}: "{speaker2_dialogue}"')
            
            # Closing narrative
            conversation_parts.append("You hear footsteps and quickly move away before they discover you eavesdropping.")
            
            # Record that this conversation happened
            cursor.execute("""
                INSERT INTO OverheardConversations (
                    user_id, conversation_id, npc_ids, topic, about_player, 
                    reveal_true_nature, conversation_text
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING conversation_id
            """, (
                user_id, conversation_id, 
                json.dumps(npc_ids), topic, about_player, reveal_true_nature,
                json.dumps(conversation_parts)
            ))
            
            overheard_id = cursor.fetchone()[0]
            
            # Create memory for the player about overhearing this
            cursor.execute("""
                INSERT INTO PlayerJournal (
                    user_id, conversation_id, entry_type, entry_text, timestamp
                )
                VALUES (%s, %s, 'overheard_conversation', %s, CURRENT_TIMESTAMP)
            """, (
                user_id, conversation_id,
                f"Overheard a revealing conversation between {speaker1['npc_name']} and {speaker2['npc_name']} about {topic}."
            ))
            
            conn.commit()
            
            return {
                "overheard_id": overheard_id,
                "speakers": [speaker1["npc_name"], speaker2["npc_name"]],
                "topic": topic,
                "about_player": about_player,
                "reveal_true_nature": reveal_true_nature,
                "conversation": conversation_parts
            }
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error generating overheard conversation: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()
