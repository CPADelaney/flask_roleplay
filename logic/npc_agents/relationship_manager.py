# logic/npc_agents/relationship_manager.py

"""
Manages individual NPC relationships with other entities.
Enhanced with memory-informed relationship management.
"""

import json
import logging
from typing import Dict, Any, Optional, List  # Added List here for type hints
from datetime import datetime

from db.connection import get_db_connection
from memory.wrapper import MemorySystem

logger = logging.getLogger(__name__)


# Constants for relationship thresholds
RELATIONSHIP_CLOSE = 75
RELATIONSHIP_FRIENDLY = 50
RELATIONSHIP_HOSTILE = 25

ALLY_THRESHOLD = 80
CO_CONSPIRATOR_THRESHOLD = 85
RIVAL_THRESHOLD = 20

# Constants for easy tuning of decay
DECAY_BASE = 1.0
DECAY_CLOSE = 0.5
DECAY_FRIENDLY = 0.7
DECAY_HOSTILE = 0.3

# Example dictionary for base relationship changes (if you want to centralize logic)
BASE_REL_CHANGE = {
    ("talk", "talk"): 1,
    ("talk", "leave"): -1,
    ("talk", "mock"): -2,
    ("help", None): 3,      # e.g. if NPC action is irrelevant, or you can refine
    ("gift", None): 4,
    ("insult", None): -4,
    ("threaten", None): -5,
    ("attack", None): -8,
}


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
        self._memory_system: Optional[MemorySystem] = None

    async def _get_memory_system(self) -> MemorySystem:
        """Lazy-load the memory system."""
        if self._memory_system is None:
            self._memory_system = await MemorySystem.get_instance(
                self.user_id, self.conversation_id
            )
        return self._memory_system

    def _fetch_npc_name(
        self, npc_id: int, conn, cursor
    ) -> str:
        """
        Fetch the name of an NPC from the DB, given an open connection/cursor.
        Returns 'Unknown' if not found.
        """
        try:
            cursor.execute(
                """
                SELECT npc_name
                FROM NPCStats
                WHERE npc_id = %s
                  AND user_id = %s
                  AND conversation_id = %s
                """,
                (npc_id, self.user_id, self.conversation_id),
            )
            row = cursor.fetchone()
            if row and row[0]:
                return row[0]
        except Exception as e:
            logger.error(f"Error fetching NPC name for npc_id={npc_id}: {e}")
        return "Unknown"

    async def update_relationships(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve and update the NPC's known relationships, returning a dict of them,
        enhanced with memory-based context.

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
                cursor.execute(
                    """
                    SELECT entity2_type, entity2_id, link_type, link_level
                    FROM SocialLinks
                    WHERE entity1_type = 'npc'
                      AND entity1_id = %s
                      AND user_id = %s
                      AND conversation_id = %s
                """,
                    (self.npc_id, self.user_id, self.conversation_id),
                )

                rows = cursor.fetchall()
                for entity_type, entity_id, link_type, link_level in rows:
                    # Default to "Unknown"
                    entity_name = "Unknown"

                    if entity_type == "npc":
                        entity_name = self._fetch_npc_name(entity_id, conn, cursor)
                    elif entity_type == "player":
                        # Could be dynamic, for now it's a placeholder
                        entity_name = "Chase"

                    relationships[entity_type] = {
                        "entity_id": entity_id,
                        "entity_name": entity_name,
                        "link_type": link_type,
                        "link_level": link_level,
                    }

                logger.debug(
                    "NPC %s relationships updated: %s", self.npc_id, relationships
                )
            except Exception as e:
                logger.error("Error updating relationships for NPC %s: %s", self.npc_id, e)

        # Enhance relationships with memory-based context
        memory_system = await self._get_memory_system()

        # For each relationship, get relevant memories and beliefs
        for entity_type, relationship in relationships.items():
            entity_id = relationship["entity_id"]

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
                limit=3,
            )

            # Get beliefs about this entity
            beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=self.npc_id,
                topic=topic,
            )

            # Add memory context to relationship
            relationship["recent_memories"] = memory_result.get("memories", [])
            relationship["beliefs"] = beliefs

        return relationships

    async def apply_relationship_decay(self, days_since_interaction: int = 1) -> Dict[str, Any]:
        """
        Apply natural decay to relationships based on time without interaction.

        Args:
            days_since_interaction: The number of days since the last interaction
                                    that triggers a decay.

        Returns:
            A dict with results of how many links were processed/decayed.
        """
        results = {"decayed_count": 0, "links_processed": 0}

        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT link_id, link_type, link_level, entity2_type, entity2_id, last_interaction
                FROM SocialLinks
                WHERE entity1_type = 'npc'
                  AND entity1_id = %s
                  AND user_id = %s
                  AND conversation_id = %s
            """,
                (self.npc_id, self.user_id, self.conversation_id),
            )

            links = cursor.fetchall()
            results["links_processed"] = len(links)

            for (
                link_id,
                link_type,
                link_level,
                entity2_type,
                entity2_id,
                last_interaction,
            ) in links:
                # If interaction was recent, skip
                if (
                    last_interaction
                    and (datetime.now() - last_interaction).days < days_since_interaction
                ):
                    continue

                # Basic decay logic
                if link_level > RELATIONSHIP_CLOSE:
                    decay_amount = DECAY_CLOSE
                elif link_level > RELATIONSHIP_FRIENDLY:
                    decay_amount = DECAY_FRIENDLY
                elif link_level < RELATIONSHIP_HOSTILE:
                    decay_amount = DECAY_HOSTILE
                else:
                    decay_amount = DECAY_BASE

                total_decay = decay_amount * days_since_interaction

                # Don't decay below 0
                new_level = max(0, link_level - total_decay)

                # Update only if there's a significant change
                if abs(new_level - link_level) >= 0.5:
                    # Update link_level
                    cursor.execute(
                        """
                        UPDATE SocialLinks
                        SET link_level = %s
                        WHERE link_id = %s
                    """,
                        (new_level, link_id),
                    )

                    results["decayed_count"] += 1

                    # Check if link_type needs to change
                    new_type = link_type
                    if new_level > RELATIONSHIP_CLOSE:
                        new_type = "close"
                    elif new_level > RELATIONSHIP_FRIENDLY:
                        new_type = "friendly"
                    elif new_level < RELATIONSHIP_HOSTILE:
                        new_type = "hostile"
                    else:
                        new_type = "neutral"

                    if new_type != link_type:
                        cursor.execute(
                            """
                            UPDATE SocialLinks
                            SET link_type = %s
                            WHERE link_id = %s
                        """,
                            (new_type, link_id),
                        )

        return results

    async def update_relationship_from_interaction(
        self,
        entity_type: str,
        entity_id: int,
        player_action: Dict[str, Any],
        npc_action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Update the relationship based on an interaction with enhanced
        error handling and memory integration.

        Args:
            entity_type: "npc" or "player"
            entity_id: The ID of the entity on the other side of the relationship
            player_action: A dict describing what the other entity did
            npc_action: A dict describing what the NPC did
            context: Additional context for the interaction

        Returns:
            Dictionary with update results
        """
        result = {
            "success": False,
            "link_id": None,
            "old_level": None,
            "new_level": None,
            "old_type": None,
            "new_type": None,
            "changes": {},
        }

        try:
            memory_system = await self._get_memory_system()

            # Get beliefs about this entity
            topic = f"{entity_type}_{entity_id}" if entity_type == "npc" else "player"
            beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=self.npc_id,
                topic=topic,
            )

            # Calculate belief adjustment factor
            belief_adjustment = 0.0
            for belief in beliefs:
                belief_text = belief.get("belief", "").lower()
                confidence = belief.get("confidence", 0.5)

                if any(word in belief_text for word in ["trust", "friend", "like", "positive"]):
                    belief_adjustment += confidence * 2
                elif any(
                    word in belief_text for word in ["distrust", "enemy", "dislike", "negative"]
                ):
                    belief_adjustment -= confidence * 2

            # Get emotional state
            emotional_state = None
            try:
                emotional_state = await memory_system.get_npc_emotion(self.npc_id)
            except Exception as e:
                logger.error(f"Error getting emotional state: {e}")

            # Calculate emotional adjustment factor
            emotional_adjustment = 0.0
            if emotional_state and "current_emotion" in emotional_state:
                primary = emotional_state["current_emotion"].get("primary", {})
                emotion_name = primary.get("name", "neutral")
                intensity = primary.get("intensity", 0.0)

                if emotion_name == "joy":
                    emotional_adjustment += intensity * 3
                elif emotion_name == "anger":
                    emotional_adjustment -= intensity * 3
                elif emotion_name == "fear":
                    emotional_adjustment -= intensity * 2
                elif emotion_name == "sadness":
                    emotional_adjustment -= intensity * 1

            # Grab optional context
            context_obj = context or {}
            location = context_obj.get("environment", {}).get("location", "Unknown")

            # Record all factors in a dictionary
            change_factors = {
                "belief_adjustment": belief_adjustment,
                "emotional_adjustment": emotional_adjustment,
                "location": location,
                "action_types": {
                    "player": player_action.get("type", "unknown"),
                    "npc": npc_action.get("type", "unknown"),
                },
            }

            # Start DB transaction
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    try:
                        conn.begin()

                        # 1) Check if a social link record already exists
                        cursor.execute(
                            """
                            SELECT link_id, link_type, link_level
                            FROM SocialLinks
                            WHERE entity1_type = 'npc'
                              AND entity1_id = %s
                              AND entity2_type = %s
                              AND entity2_id = %s
                              AND user_id = %s
                              AND conversation_id = %s
                            """,
                            (self.npc_id, entity_type, entity_id, self.user_id, self.conversation_id),
                        )
                        row = cursor.fetchone()

                        if row:
                            link_id, link_type, link_level = row
                            result["link_id"] = link_id
                            result["old_level"] = link_level
                            result["old_type"] = link_type
                        else:
                            # Create a new relationship if none exists
                            cursor.execute(
                                """
                                INSERT INTO SocialLinks (
                                    entity1_type, entity1_id,
                                    entity2_type, entity2_id,
                                    link_type, link_level,
                                    user_id, conversation_id,
                                    last_interaction
                                )
                                VALUES (
                                    'npc', %s,
                                    %s, %s,
                                    'neutral', 0,
                                    %s, %s,
                                    %s
                                )
                                RETURNING link_id
                                """,
                                (
                                    self.npc_id,
                                    entity_type,
                                    entity_id,
                                    self.user_id,
                                    self.conversation_id,
                                    datetime.now(),
                                ),
                            )
                            link_id = cursor.fetchone()[0]
                            result["link_id"] = link_id
                            link_type = "neutral"
                            link_level = 0
                            result["old_level"] = 0
                            result["old_type"] = "neutral"

                        # 2) Calculate level changes
                        player_type = player_action.get("type")
                        npc_type = npc_action.get("type")
                        base_key = (player_type, npc_type)
                        # Try an exact match, or fallback to (player_type, None)
                        level_change = BASE_REL_CHANGE.get(base_key) or BASE_REL_CHANGE.get(
                            (player_type, None)
                        ) or 0

                        # Amplify changes if there's a mask_slippage
                        if "mask_slippage" in npc_action:
                            slip_severity = npc_action["mask_slippage"].get("severity", 1)
                            level_change *= slip_severity if slip_severity >= 3 else 1

                        # Record base level change
                        change_factors["base_level_change"] = level_change

                        # Apply belief/emotional adjustments
                        # For positive base, we add both adjustments
                        # For negative base, we invert emotional
                        # (this is just an example; tune as desired)
                        final_level_change = level_change
                        if level_change > 0:
                            final_level_change += belief_adjustment + emotional_adjustment
                        else:
                            final_level_change += belief_adjustment - emotional_adjustment

                        final_level_change = round(final_level_change)
                        change_factors["final_level_change"] = final_level_change
                        result["changes"] = change_factors

                        # 3) Apply changes
                        new_level = link_level
                        if final_level_change != 0:
                            new_level = max(0, min(100, link_level + final_level_change))
                            cursor.execute(
                                """
                                UPDATE SocialLinks
                                SET link_level = %s
                                WHERE link_id = %s
                            """,
                                (new_level, link_id),
                            )
                            result["new_level"] = new_level

                        # Determine new link type based on level
                        new_link_type = link_type
                        if new_level > RELATIONSHIP_CLOSE:
                            new_link_type = "close"
                        elif new_level > RELATIONSHIP_FRIENDLY:
                            new_link_type = "friendly"
                        elif new_level < RELATIONSHIP_HOSTILE:
                            new_link_type = "hostile"
                        else:
                            new_link_type = "neutral"

                        if new_link_type != link_type:
                            cursor.execute(
                                """
                                UPDATE SocialLinks
                                SET link_type = %s
                                WHERE link_id = %s
                            """,
                                (new_link_type, link_id),
                            )
                            result["new_type"] = new_link_type

                        # Update last_interaction to "now"
                        cursor.execute(
                            """
                            UPDATE SocialLinks
                            SET last_interaction = %s
                            WHERE link_id = %s
                            """,
                            (datetime.now(), link_id),
                        )

                        # 4) Add event to the link history
                        change_description = []
                        if abs(level_change) > 0:
                            change_description.append(f"base:{level_change:+d}")
                        if abs(belief_adjustment) > 0:
                            change_description.append(f"beliefs:{belief_adjustment:+.1f}")
                        if abs(emotional_adjustment) > 0:
                            change_description.append(f"emotions:{emotional_adjustment:+.1f}")

                        change_str = ", ".join(change_description)
                        event_text = (
                            f"Interaction: {entity_type.capitalize()} "
                            f"{player_action.get('description','???')}, "
                            f"NPC {npc_action.get('description','???')}. "
                            f"Relationship change: {link_level} → {new_level} "
                            f"({final_level_change:+d}) [Factors: {change_str}]"
                        )

                        cursor.execute(
                            """
                            UPDATE SocialLinks
                            SET link_history = COALESCE(link_history, '[]'::jsonb) || %s::jsonb
                            WHERE link_id = %s
                        """,
                            (json.dumps([event_text]), link_id),
                        )

                        conn.commit()
                        result["success"] = True

                    except Exception as sql_error:
                        conn.rollback()
                        logger.error(f"Database error updating relationship: {sql_error}")
                        result["error"] = str(sql_error)

            # 5) Create a memory of this relationship change - outside transaction
            if result["success"]:
                old_type = result["old_type"]
                new_type = result.get("new_type", old_type)
                if abs(final_level_change) >= 3 or new_type != old_type:
                    # Determine entity name
                    entity_name = "the player"
                    if entity_type == "npc":
                        with get_db_connection() as conn, conn.cursor() as cursor:
                            entity_name = self._fetch_npc_name(entity_id, conn, cursor)

                    direction = "improved" if final_level_change > 0 else "worsened"
                    if new_type != old_type and result.get("new_level") is not None:
                        lvl = result["new_level"]
                        memory_text = f"My relationship with {entity_name} changed to {new_type} (level {lvl})"
                    else:
                        lvl = result.get("new_level", 0)
                        memory_text = (
                            f"My relationship with {entity_name} {direction} (now level {lvl})"
                        )

                    importance = (
                        "medium" if abs(final_level_change) >= 5 or new_type != old_type else "low"
                    )
                    try:
                        await memory_system.remember(
                            entity_type="npc",
                            entity_id=self.npc_id,
                            memory_text=memory_text,
                            importance=importance,
                            tags=["relationship_change", entity_type],
                        )
                    except Exception as memory_error:
                        logger.error(f"Error creating relationship memory: {memory_error}")

                    # Also update beliefs from relationship change
                    try:
                        await self._update_beliefs_from_relationship_change(
                            entity_type,
                            entity_id,
                            entity_name,
                            result["old_level"],
                            result["new_level"],
                            final_level_change,
                        )
                    except Exception as memory_error:
                        logger.error(f"Error updating beliefs from relationship change: {memory_error}")

            return result

        except Exception as e:
            logger.error(f"Error in update_relationship_from_interaction: {e}")
            result["error"] = str(e)
            return result

    async def _update_beliefs_from_relationship_change(
        self,
        entity_type: str,
        entity_id: int,
        entity_name: str,
        old_level: int,
        new_level: int,
        level_change: int,
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
                topic=topic,
            )

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

            if not new_belief_text:
                return

            # Check if a similar belief already exists
            existing_belief = None
            for belief in beliefs:
                belief_text = belief.get("belief", "").lower()
                # Basic checks for positive or negative beliefs
                if new_level > 50 and any(
                    word in belief_text for word in ["trust", "friend", "value"]
                ):
                    existing_belief = belief
                    break
                if new_level < 50 and any(
                    word in belief_text for word in ["hostile", "caution", "untrustworthy"]
                ):
                    existing_belief = belief
                    break

            if existing_belief:
                # Update existing belief's confidence
                await memory_system.semantic_manager.update_belief_confidence(
                    belief_id=existing_belief["id"],
                    entity_type="npc",
                    entity_id=self.npc_id,
                    new_confidence=confidence,
                    reason=f"Relationship changed to level {new_level}",
                )
            else:
                # Create a new belief
                await memory_system.create_belief(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    belief_text=new_belief_text,
                    confidence=confidence,
                )
        except Exception as e:
            logger.error(f"Error updating beliefs from relationship change: {e}")

    async def get_relationship_history(
        self,
        entity_type: str,
        entity_id: int,
        limit: int = 10,
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
                cursor.execute(
                    """
                    SELECT link_id, link_history
                    FROM SocialLinks
                    WHERE entity1_type = 'npc'
                      AND entity1_id = %s
                      AND entity2_type = %s
                      AND entity2_id = %s
                      AND user_id = %s
                      AND conversation_id = %s
                """,
                    (self.npc_id, entity_type, entity_id, self.user_id, self.conversation_id),
                )

                row = cursor.fetchone()
                if not row:
                    return []

                link_id, link_history = row
                if link_history:
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
                        history.reverse()
            except Exception as e:
                logger.error(f"Error getting relationship history: {e}")

        return history

    async def get_relationship_memories(
        self,
        entity_type: str,
        entity_id: int,
        limit: int = 5,
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
                    entity_name = self._fetch_npc_name(entity_id, conn, cursor)

            # Query memories about this entity
            query = entity_name
            result = await memory_system.recall(
                entity_type="npc",
                entity_id=self.npc_id,
                query=query,
                limit=limit,
            )

            return result.get("memories", [])
        except Exception as e:
            logger.error(f"Error getting relationship memories: {e}")
            return []

    async def evaluate_coalitions_and_rivalries(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Periodically run this to detect alliances (co-conspirators) or rivalry blocks
        among NPCs who share strong negative relationships with a common target 
        (often the player) or with each other.

        - Any pair of NPCs with link_level >= ALLY_THRESHOLD => "ally"
        - If link_level >= CO_CONSPIRATOR_THRESHOLD => "co_conspirator"
        - If link_level <= RIVAL_THRESHOLD => "rival"

        This extends the link_type to more explicit alliance/rival states.

        Args:
            context: Optional context about the environment or scene.

        Returns:
            A dictionary summarizing changes:
            {
              "alliances": [ (npc_id1, npc_id2, old_type, new_type), ... ],
              "rivalries":  [ (npc_id1, npc_id2, old_type, new_type), ... ]
            }
        """
        results = {
            "alliances": [],
            "rivalries": [],
            "co_conspirators": []
        }
        
        with get_db_connection() as conn, conn.cursor() as cursor:
            try:
                cursor.execute(
                    """
                    SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id, link_type, link_level
                    FROM SocialLinks
                    WHERE user_id = %s
                      AND conversation_id = %s
                      AND entity1_type = 'npc'
                      AND entity2_type = 'npc'
                    """,
                    (self.user_id, self.conversation_id)
                )
                rows = cursor.fetchall()

                for link_id, e1_type, e1_id, e2_type, e2_id, link_type, link_level in rows:
                    old_type = link_type
                    new_type = old_type

                    # Evaluate alliance / co-conspirator
                    if link_level >= CO_CONSPIRATOR_THRESHOLD:
                        new_type = "co_conspirator"
                    elif link_level >= ALLY_THRESHOLD:
                        new_type = "ally"
                    elif link_level <= RIVAL_THRESHOLD:
                        new_type = "rival"
                    else:
                        # If it doesn't meet thresholds, keep whatever it was
                        # or revert to neutral if you prefer
                        pass

                    if new_type != old_type:
                        cursor.execute(
                            """
                            UPDATE SocialLinks
                            SET link_type = %s
                            WHERE link_id = %s
                            """,
                            (new_type, link_id)
                        )
                        if new_type == "ally":
                            results["alliances"].append((e1_id, e2_id, old_type, new_type))
                        elif new_type == "co_conspirator":
                            results["co_conspirators"].append((e1_id, e2_id, old_type, new_type))
                        elif new_type == "rival":
                            results["rivalries"].append((e1_id, e2_id, old_type, new_type))

                conn.commit()

            except Exception as e:
                logger.error(f"Error in evaluate_coalitions_and_rivalries: {e}")
                conn.rollback()

        return results
