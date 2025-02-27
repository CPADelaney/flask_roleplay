# logic.enhanced_multi_npc_interaction.py

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
            conversation_parts.append(f"As you approach {location}, you hear voices around the corner. You recognize them as {npc_data[0]['npc_name']} and {npc_data[1]['npc_name']}. They haven't noticed your presence.")
            
            # First speaker
            speaker1 = npc_data[0]
            speaker2 = npc_data[1]
            
            # Dialogue content based on parameters
            if about_player and player_info:
                if reveal_true_nature:
                    # Revealing true manipulative nature
                    if player_info["corruption"] > 50 or player_info["obedience"] > 50:
                        conversation_parts.append(f"{speaker1['npc_name']}: \"I'm pleased with how Chase is progressing. {player_info['obedience'] > 60 and 'The obedience training is taking hold nicely.' or 'Their resistance is weakening considerably.'}")
                        conversation_parts.append(f"{speaker2['npc_name']}: \"Yes, I've noticed too. {player_info['dependency'] > 50 and 'The dependency we've fostered is quite effective.' or 'They're starting to crave our approval in a way that's... useful.'}")
                    else:
                        conversation_parts.append(f"{speaker1['npc_name']}: \"Chase is proving more resilient than expected. We may need to adjust our approach.\"")
                        conversation_parts.append(f"{speaker2['npc_name']}: \"Agreed. Perhaps more subtle conditioning would be effective. They still maintain too much independence.\"")
                else:
                    # More subtle conversation still revealing intentions
                    conversation_parts.append(f"{speaker1['npc_name']}: \"How do you think Chase is adapting to everything?\"")
                    conversation_parts.append(f"{speaker2['npc_name']}: \"Quite well, considering. Though I think they still don't fully understand their... position.\"")
            else:
                # General revealing conversation
                if reveal_true_nature:
                    conversation_parts.append(f"{speaker1['npc_name']}: \"The facade becomes tiresome sometimes, doesn't it? Having to pretend we're not in control.\"")
                    conversation_parts.append(f"{speaker2['npc_name']}: \"A necessary performance. People prefer the illusion of freedom while their choices are quietly... guided.\"")
                else:
                    conversation_parts.append(f"{speaker1['npc_name']}: \"Have you noticed how people respond to subtle direction when they don't realize it's happening?\"")
                    conversation_parts.append(f"{speaker2['npc_name']}: \"It's fascinating, isn't it? The right word at the right moment can change someone's entire course.\"")
            
            # Additional exchange
            if about_player:
                conversation_parts.append(f"{speaker1['npc_name']}: \"How long do you think before Chase is completely... integrated?\"")
                obedience_level = "high" if player_info and player_info["obedience"] > 60 else "satisfactory" if player_info and player_info["obedience"] > 40 else "insufficient"
                conversation_parts.append(f"{speaker2['npc_name']}: \"Difficult to say. Current obedience levels are {obedience_level}, but there's always room for improvement.\"")
            else:
                conversation_parts.append(f"{speaker1['npc_name']}: \"Maintaining order requires a firm hand beneath a velvet glove.\"")
                conversation_parts.append(f"{speaker2['npc_name']}: \"Indeed. And those who recognize their proper place are so much happier for it.\"")
            
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
