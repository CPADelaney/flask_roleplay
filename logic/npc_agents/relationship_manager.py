# logic/npc_agents/relationship_manager.py

"""
Manages individual NPC relationships with other entities.
Enhanced with memory-informed relationship management.
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from db.connection import get_db_connection
from memory.wrapper import MemorySystem

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
        self._memory_system = None

    async def _get_memory_system(self):
        """Lazy-load the memory system."""
        if self._memory_system is None:
            self._memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        return self._memory_system

    async def update_relationships(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve and update the NPC's known relationships, returning a dict of them.
        Enhanced with memory-based context.

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
            context: Additional info about the environment or scene

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
                        entity_name = "Chase"  # Default player name

                    relationships[entity_type] = {
                        "entity_id": entity_id,
                        "entity_name": entity_name,
                        "link_type": link_type,
                        "link_level": link_level
                    }

                logger.debug("NPC %s relationships updated: %s", self.npc_id, relationships)
            except Exception as e:
                logger.error("Error updating relationships for NPC %s: %s", self.npc_id, e)

        # Enhance relationships with memory-based context
        memory_system = await self._get_memory_system()
        
        # For each relationship, get relevant memories and beliefs
        for entity_type, relationship in relationships.items():
            entity_id = relationship["entity_id"]
            
            # Get memories about interactions with this entity
            if entity_type == "npc":
                query = f"NPC {relationship['entity_name']}"
                topic = f"npc_{entity_id}"
            else:  # player
                query = "player"
                topic = "player"
                
            # Get recent interaction memories
            memory_result = await memory_system.recall(
                entity_type="npc",
                entity_id=self.npc_id,
                query=query,
                limit=3
            )
            
            # Get beliefs about this entity
            beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=self.npc_id,
                topic=topic
            )
            
            # Add memory context to relationship
            relationship["recent_memories"] = memory_result.get("memories", [])
            relationship["beliefs"] = beliefs

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
        Enhanced with memory-informed adjustments.

        For example, if the player 'talked' and the NPC 'mocked' in response, we might
        lower the relationship level.

        Args:
            entity_type: "npc" or "player"
            entity_id: The ID of the entity on the other side of the relationship
            player_action: A dict describing what the other entity did
            npc_action: A dict describing what the NPC did
        """
        # Get memory system for beliefs and emotional context
        memory_system = await self._get_memory_system()
        
        # Get beliefs about this entity to influence relationship changes
        beliefs = await memory_system.get_beliefs(
            entity_type="npc",
            entity_id=self.npc_id,
            topic=f"{entity_type}_{entity_id}" if entity_type == "npc" else "player"
        )
        
        # Calculate belief adjustment factor
        belief_adjustment = 0
        for belief in beliefs:
            belief_text = belief.get("belief", "").lower()
            confidence = belief.get("confidence", 0.5)
            
            # Positive beliefs make relationship changes more favorable
            if any(word in belief_text for word in ["trust", "friend", "like", "positive"]):
                belief_adjustment += confidence * 2
            # Negative beliefs make relationship changes more negative
            elif any(word in belief_text for word in ["distrust", "enemy", "dislike", "negative"]):
                belief_adjustment -= confidence * 2
        
        # Get emotional state to influence relationship changes
        emotional_state = None
        try:
            emotional_state = await memory_system.get_npc_emotion(self.npc_id)
        except Exception as e:
            logger.error(f"Error getting emotional state: {e}")
            
        # Calculate emotional adjustment factor
        emotional_adjustment = 0
        if emotional_state and "current_emotion" in emotional_state:
            current_emotion = emotional_state["current_emotion"]
            primary = current_emotion.get("primary", {})
            emotion_name = primary.get("name", "neutral")
            intensity = primary.get("intensity", 0.0)
            
            # Different emotions affect relationship changes differently
            if emotion_name == "joy":
                emotional_adjustment += intensity * 3
            elif emotion_name == "anger":
                emotional_adjustment -= intensity * 3
            elif emotion_name == "fear":
                emotional_adjustment -= intensity * 2
            elif emotion_name == "sadness":
                emotional_adjustment -= intensity * 1

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

                # Base relationship changes
                if player_action.get("type") == "talk" and npc_action.get("type") == "talk":
                    level_change += 1
                elif player_action.get("type") == "talk" and npc_action.get("type") == "leave":
                    level_change -= 1
                elif player_action.get("type") == "talk" and npc_action.get("type") == "mock":
                    level_change -= 2
                
                # Add more complex interaction rules
                if player_action.get("type") == "help":
                    level_change += 3
                elif player_action.get("type") == "gift":
                    level_change += 4
                elif player_action.get("type") == "insult":
                    level_change -= 4
                elif player_action.get("type") == "threaten":
                    level_change -= 5
                elif player_action.get("type") == "attack":
                    level_change -= 8
                
                # Apply mask slip effects if present
                if "mask_slippage" in npc_action:
                    # Mask slippages can cause more dramatic relationship changes
                    slip_severity = npc_action["mask_slippage"].get("severity", 1)
                    if slip_severity >= 3:  # Major slips have bigger impacts
                        if level_change > 0:
                            level_change = level_change * 2  # Amplify positive changes
                        else:
                            level_change = level_change * 2  # Amplify negative changes
                
                # Apply belief and emotional adjustments
                final_level_change = level_change
                if level_change > 0:
                    # For positive changes, positive beliefs/emotions amplify
                    final_level_change += belief_adjustment + emotional_adjustment
                else:
                    # For negative changes, negative beliefs/emotions amplify
                    final_level_change += belief_adjustment - emotional_adjustment
                
                # Round to integer, preventing tiny changes
                final_level_change = round(final_level_change)

                # 3) Apply changes
                new_level = link_level
                if final_level_change != 0:
                    new_level = max(0, min(100, link_level + final_level_change))
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
                
                # Determine new link type based on level
                new_link_type = link_type
                if new_level > 75:
                    new_link_type = "close"
                elif new_level > 50:
                    new_link_type = "friendly"
                elif new_level < 25:
                    new_link_type = "hostile"
                
                if new_link_type != link_type:
                    cursor.execute("""
                        UPDATE SocialLinks
                        SET link_type = %s
                        WHERE link_id = %s
                    """, (new_link_type, link_id))
                    type_change = new_link_type

                # 4) Add event to the link history
                change_factors = []
                if abs(level_change) > 0:
                    change_factors.append(f"base:{level_change:+d}")
                if abs(belief_adjustment) > 0:
                    change_factors.append(f"beliefs:{belief_adjustment:+.1f}")
                if abs(emotional_adjustment) > 0:
                    change_factors.append(f"emotions:{emotional_adjustment:+.1f}")
                
                change_description = ", ".join(change_factors)
                
                event_text = (
                    f"Interaction: {entity_type.capitalize()} {player_action.get('description','???')}, "
                    f"NPC {npc_action.get('description','???')}. "
                    f"Relationship change: {link_level} → {new_level} ({final_level_change:+d}) [Factors: {change_description}]"
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

                # 5) Create a memory of this relationship change
                memory_system = await self._get_memory_system()
                
                # Only create memories for significant changes
                if abs(final_level_change) >= 3 or type_change:
                    entity_name = "the player"
                    if entity_type == "npc":
                        # Get NPC name
                        npc_query = """
                            SELECT npc_name FROM NPCStats
                            WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                        """
                        cursor.execute(npc_query, (entity_id, self.user_id, self.conversation_id))
                        name_row = cursor.fetchone()
                        if name_row:
                            entity_name = name_row[0]
                    
                    direction = "improved" if final_level_change > 0 else "worsened"
                    if type_change:
                        memory_text = f"My relationship with {entity_name} changed to {type_change} (level {new_level})"
                    else:
                        memory_text = f"My relationship with {entity_name} {direction} (now level {new_level})"
                    
                    # Create memory with appropriate tags and importance
                    importance = "medium" if abs(final_level_change) >= 5 or type_change else "low"
                    
                    await memory_system.remember(
                        entity_type="npc",
                        entity_id=self.npc_id,
                        memory_text=memory_text,
                        importance=importance,
                        tags=["relationship_change", entity_type]
                    )
                    
                    # Update beliefs based on relationship changes
                    await self._update_beliefs_from_relationship_change(
                        entity_type, entity_id, entity_name, 
                        link_level, new_level, final_level_change
                    )

            except Exception as e:
                conn.rollback()
                logger.error("Error updating relationship from interaction for NPC %s: %s", self.npc_id, e)
    
    async def _update_beliefs_from_relationship_change(
        self,
        entity_type: str,
        entity_id: int,
        entity_name: str, 
        old_level: int,
        new_level: int,
        level_change: int
    ) -> None:
        """
        Update beliefs based on relationship changes.
        
        Args:
            entity_type: Type of entity ("npc" or "player")
            entity_id: ID of the entity
            entity_name: Name of the entity
            old_level: Previous relationship level
            new_level: New relationship level
            level_change: Amount of change
        """
        # Only update beliefs for significant changes
        if abs(level_change) < 5:
            return
            
        try:
            memory_system = await self._get_memory_system()
            
            topic = "player" if entity_type == "player" else f"npc_{entity_id}"
            
            # Get existing beliefs
            beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=self.npc_id,
                topic=topic
            )
            
            # Determine appropriate belief based on new level
            new_belief_text = ""
            confidence = 0.0
            
            if new_level > 80:
                new_belief_text = f"{entity_name} is someone I deeply trust and value"
                confidence = 0.8
            elif new_level > 60:
                new_belief_text = f"{entity_name} is a friend I can rely on"
                confidence = 0.7
            elif new_level > 40:
                new_belief_text = f"{entity_name} is generally trustworthy"
                confidence = 0.6
            elif new_level < 20:
                new_belief_text = f"{entity_name} is hostile and should be treated with caution"
                confidence = 0.7
            elif new_level < 40:
                new_belief_text = f"{entity_name} is somewhat untrustworthy"
                confidence = 0.6
            
            if new_belief_text:
                # Check if a similar belief exists
                existing_belief = None
                for belief in beliefs:
                    belief_text = belief.get("belief", "").lower()
                    
                    # Check for positive belief matches
                    if new_level > 50 and any(word in belief_text for word in ["trust", "friend", "value"]):
                        existing_belief = belief
                        break
                        
                    # Check for negative belief matches
                    if new_level < 50 and any(word in belief_text for word in ["hostile", "caution", "untrustworthy"]):
                        existing_belief = belief
                        break
                
                if existing_belief:
                    # Update existing belief
                    await memory_system.semantic_manager.update_belief_confidence(
                        belief_id=existing_belief["id"],
                        entity_type="npc",
                        entity_id=self.npc_id,
                        new_confidence=confidence,
                        reason=f"Relationship changed to {new_level}"
                    )
                else:
                    # Create new belief
                    await memory_system.create_belief(
                        entity_type="npc",
                        entity_id=self.npc_id,
                        belief_text=new_belief_text,
                        confidence=confidence
                    )
        except Exception as e:
            logger.error(f"Error updating beliefs from relationship change: {e}")

    async def get_relationship_history(
        self,
        entity_type: str,
        entity_id: int,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get the history of relationship changes with a specific entity.
        
        Args:
            entity_type: "npc" or "player"
            entity_id: The ID of the entity
            limit: Maximum number of history entries to return
            
        Returns:
            List of relationship history events
        """
        history = []
        
        with get_db_connection() as conn, conn.cursor() as cursor:
            try:
                # Get the link_id
                cursor.execute("""
                    SELECT link_id, link_history
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
                    return []
                    
                link_id, link_history = row
                
                if link_history:
                    # Parse history JSON
                    if isinstance(link_history, str):
                        try:
                            events = json.loads(link_history)
                        except json.JSONDecodeError:
                            events = []
                    else:
                        events = link_history
                        
                    # Return most recent events first
                    if isinstance(events, list):
                        history = events[-limit:]
                        history.reverse()  # Most recent first
            except Exception as e:
                logger.error(f"Error getting relationship history: {e}")
                
        return history
    
    async def get_relationship_memories(
        self,
        entity_type: str,
        entity_id: int,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get memories related to interactions with a specific entity.
        
        Args:
            entity_type: "npc" or "player"
            entity_id: The ID of the entity
            limit: Maximum number of memories to return
            
        Returns:
            List of memories about interactions with this entity
        """
        try:
            memory_system = await self._get_memory_system()
            
            # Get entity name for better query
            entity_name = "the player"
            if entity_type == "npc":
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT npc_name FROM NPCStats
                        WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                    """, (entity_id, self.user_id, self.conversation_id))
                    
                    row = cursor.fetchone()
                    if row:
                        entity_name = row[0]
            
            # Query memories about this entity
            query = entity_name
            result = await memory_system.recall(
                entity_type="npc",
                entity_id=self.npc_id,
                query=query,
                limit=limit
            )
            
            return result.get("memories", [])
            
        except Exception as e:
            logger.error(f"Error getting relationship memories: {e}")
            return []
