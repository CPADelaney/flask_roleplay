# logic.enhanced_relationship_dynamics.py

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
