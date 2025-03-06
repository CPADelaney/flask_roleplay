# logic/npc_agents/relationship_manager.py

"""
Manages individual NPC relationships with other entities
"""

import json
import logging
from typing import List, Dict, Any, Optional
from db.connection import get_db_connection

class NPCRelationshipManager:
    """Manages an individual NPC's relationships with other entities"""

    def __init__(self, npc_id, user_id, conversation_id):
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def update_relationships(self, context):
        """Update NPC's awareness of relationships based on current context"""
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT entity2_type, entity2_id, link_type, link_level
                FROM SocialLinks
                WHERE entity1_type = 'npc'
                  AND entity1_id = %s
                  AND user_id = %s
                  AND conversation_id = %s
            """, (self.npc_id, self.user_id, self.conversation_id))

            relationships = {}
            for row in cursor.fetchall():
                entity_type, entity_id, link_type, link_level = row

                # Get entity name if NPC
                entity_name = "Unknown"
                if entity_type == "npc":
                    cursor.execute("""
                        SELECT npc_name
                        FROM NPCStats
                        WHERE npc_id = %s
                          AND user_id = %s
                          AND conversation_id = %s
                    """, (entity_id, self.user_id, self.conversation_id))
                    name_row = cursor.fetchone()
                    if name_row:
                        entity_name = name_row[0]
                elif entity_type == "player":
                    entity_name = "Chase"

                relationships[entity_type] = {
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "link_type": link_type,
                    "link_level": link_level
                }

            return relationships
        except Exception as e:
            logging.error(f"Error updating relationships: {e}")
            return {}
        finally:
            conn.close()

    async def update_relationship_from_interaction(self, entity_type, entity_id,
                                                   player_action, npc_action):
        """
        Update relationship based on an interaction between NPC and another entity
        """
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            # Check if relationship exists
            cursor.execute("""
                SELECT link_id, link_type, link_level
                FROM SocialLinks
                WHERE entity1_type = 'npc'
                  AND entity1_id = %s
                  AND entity2_type = %s
                  AND entity2_id = %s
                  AND user_id = %s
                  AND conversation_id = %s
            """, (
                self.npc_id, entity_type, entity_id,
                self.user_id, self.conversation_id
            ))
            row = cursor.fetchone()

            if not row:
                # Create new relationship
                cursor.execute("""
                    INSERT INTO SocialLinks (
                        entity1_type, entity1_id,
                        entity2_type, entity2_id,
                        link_type, link_level,
                        user_id, conversation_id
                    ) VALUES (
                        'npc', %s,
                        %s, %s,
                        'neutral', 0,
                        %s, %s
                    )
                    RETURNING link_id
                """, (
                    self.npc_id, entity_type, entity_id,
                    self.user_id, self.conversation_id
                ))
                link_id = cursor.fetchone()[0]
                link_type = "neutral"
                link_level = 0
            else:
                link_id, link_type, link_level = row

            # Relationship changes
            level_change = 0
            type_change = None

            # Example rules
            if player_action["type"] == "talk" and npc_action["type"] == "talk":
                level_change += 1
            elif player_action["type"] == "talk" and npc_action["type"] == "leave":
                level_change -= 1
            elif player_action["type"] == "talk" and npc_action["type"] == "mock":
                level_change -= 2

            # Apply changes
            if level_change != 0:
                new_level = max(0, min(100, link_level + level_change))
                cursor.execute("""
                    UPDATE SocialLinks
                    SET link_level = %s
                    WHERE link_id = %s
                """, (new_level, link_id))

            if type_change:
                cursor.execute("""
                    UPDATE SocialLinks
                    SET link_type = %s
                    WHERE link_id = %s
                """, (type_change, link_id))

            # Add event to history
            if level_change != 0 or type_change:
                event_text = f"Interaction: Player {player_action['description']}, NPC {npc_action['description']}"
                cursor.execute("""
                    UPDATE SocialLinks
                    SET link_history = COALESCE(link_history, '[]'::jsonb) || %s::jsonb
                    WHERE link_id = %s
                """, (json.dumps([event_text]), link_id))

            conn.commit()
        except Exception as e:
            conn.rollback()
            logging.error(f"Error updating relationship from interaction: {e}")
        finally:
            conn.close()
