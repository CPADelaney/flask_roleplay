# logic/npc_agents/relationship_manager.py

"""
Manages individual NPC relationships with other entities.
"""

import json
import logging
from typing import Dict, Any, Optional

from db.connection import get_db_connection

logger = logging.getLogger(__name__)


class NPCRelationshipManager:
    """
    Manages an individual NPC's relationships with other entities
    (other NPCs or the player).
    """

    def __init__(self, npc_id: int, user_id: int, conversation_id: int):
        """
        Initialize a relationship manager for one NPC.

        Args:
            npc_id: The ID of the NPC whose relationships we manage.
            user_id: The user (player) ID.
            conversation_id: The active conversation or scene ID.
        """
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def update_relationships(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve and update the NPC's known relationships, returning a dict of them.

        The returned dictionary might look like:
        {
          "npc": {
              "entity_id": ...,
              "entity_name": ...,
              "link_type": ...,
              "link_level": ...
          },
          "player": {...}
        }

        Args:
            context: Additional info about the environment or scene (not heavily used here)

        Returns:
            A dictionary describing the NPC's relationships with each entity type.
        """
        relationships: Dict[str, Any] = {}

        with get_db_connection() as conn, conn.cursor() as cursor:
            try:
                # Query all links from NPC -> other entity
                cursor.execute("""
                    SELECT entity2_type, entity2_id, link_type, link_level
                    FROM SocialLinks
                    WHERE entity1_type = 'npc'
                      AND entity1_id = %s
                      AND user_id = %s
                      AND conversation_id = %s
                """, (self.npc_id, self.user_id, self.conversation_id))

                rows = cursor.fetchall()
                for entity_type, entity_id, link_type, link_level in rows:
                    entity_name = "Unknown"

                    if entity_type == "npc":
                        # fetch NPC name
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

                logger.debug("NPC %s relationships updated: %s", self.npc_id, relationships)
            except Exception as e:
                logger.error("Error updating relationships for NPC %s: %s", self.npc_id, e)

        return relationships

    async def update_relationship_from_interaction(
        self,
        entity_type: str,
        entity_id: int,
        player_action: Dict[str, Any],
        npc_action: Dict[str, Any]
    ) -> None:
        """
        Update the relationship based on an interaction between this NPC and another entity.

        For example, if the player 'talked' and the NPC 'mocked' in response, we might
        lower the relationship level.

        Args:
            entity_type: "npc" or "player"
            entity_id: The ID of the entity on the other side of the relationship
            player_action: A dict describing what the other entity did
            npc_action: A dict describing what the NPC did
        """
        with get_db_connection() as conn, conn.cursor() as cursor:
            try:
                # 1) Check if a social link record already exists
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

                if row:
                    link_id, link_type, link_level = row
                else:
                    # Create a new relationship if none exists
                    cursor.execute("""
                        INSERT INTO SocialLinks (
                            entity1_type, entity1_id,
                            entity2_type, entity2_id,
                            link_type, link_level,
                            user_id, conversation_id
                        )
                        VALUES (
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

                # 2) Determine how the link changes
                level_change = 0
                type_change = None

                # Example rules for adjusting the relationship
                if player_action.get("type") == "talk" and npc_action.get("type") == "talk":
                    level_change += 1
                elif player_action.get("type") == "talk" and npc_action.get("type") == "leave":
                    level_change -= 1
                elif player_action.get("type") == "talk" and npc_action.get("type") == "mock":
                    level_change -= 2

                # 3) Apply changes
                new_level = link_level
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

                # 4) Optionally add an event to the link history if changes occurred
                if level_change != 0 or type_change:
                    event_text = (
                        f"Interaction: Player {player_action.get('description','???')}, "
                        f"NPC {npc_action.get('description','???')}"
                    )
                    cursor.execute("""
                        UPDATE SocialLinks
                        SET link_history = COALESCE(link_history, '[]'::jsonb) || %s::jsonb
                        WHERE link_id = %s
                    """, (json.dumps([event_text]), link_id))

                conn.commit()
                logger.debug(
                    "Updated relationship for NPC %s -> entity (%s:%s). "
                    "Change: level=%d => %d, type_change=%s",
                    self.npc_id, entity_type, entity_id,
                    link_level, new_level, type_change
                )

            except Exception as e:
                conn.rollback()
                logger.error("Error updating relationship from interaction for NPC %s: %s", self.npc_id, e)
