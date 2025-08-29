# logic/relationship_integration.py

"""
Relationship integration module connecting the enhanced relationship system 
with the existing game social links and group mechanics.

This refactor:
- Lazily initializes IntegratedNPCSystem (await .initialize()).
- Uses actual IntegratedNPCSystem APIs (create_direct_social_link, get_relationship,
  add_event_to_link, check_for_relationship_events, apply_crossroads_choice).
- Uses OptimizedRelationshipManager for dimension-level updates (link_id-based ops).
- Maps "tension" -> "volatility" for convenience.
- Resolves link_id to (entity1_type, entity1_id, entity2_type, entity2_id) as needed.
- Implements create_npc_group using DB, and keeps group additions/dynamics via LoreSystem.
"""

import logging
import json
import random
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

import asyncpg

from db.connection import get_db_connection_context
from lore.core.lore_system import LoreSystem

logger = logging.getLogger(__name__)

# Friendly aliases for dimensions
_DIMENSION_ALIAS = {
    "tension": "volatility"
}


class RelationshipIntegration:
    """
    Bridge class that integrates the dynamic relationship manager and the
    IntegratedNPCSystem with existing social links and group handling.
    """

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = int(user_id)
        self.conversation_id = int(conversation_id)
        self._npc_system = None
        self._rel_manager = None
        # Simple context object for LoreSystem (compatible with propose_and_enact_change)
        self.ctx = type('Context', (), {'user_id': self.user_id, 'conversation_id': self.conversation_id})()

    # ==================== Lazy loaders ====================

    async def _get_integrated(self):
        """Lazy-load and initialize IntegratedNPCSystem."""
        if self._npc_system is None:
            from logic.fully_integrated_npc_system import IntegratedNPCSystem
            self._npc_system = IntegratedNPCSystem(self.user_id, self.conversation_id)
            await self._npc_system.initialize()
        return self._npc_system

    async def _get_manager(self):
        """Lazy-load the optimized relationship manager for dimension-level mutations."""
        if self._rel_manager is None:
            from logic.dynamic_relationships import OptimizedRelationshipManager
            self._rel_manager = OptimizedRelationshipManager(self.user_id, self.conversation_id)
        return self._rel_manager

    async def _resolve_link_entities(self, link_id: int) -> Optional[Tuple[str, int, str, int]]:
        """Resolve a link_id into (entity1_type, entity1_id, entity2_type, entity2_id)."""
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT entity1_type, entity1_id, entity2_type, entity2_id
                FROM SocialLinks
                WHERE link_id=$1 AND user_id=$2 AND conversation_id=$3
            """, link_id, self.user_id, self.conversation_id)
        if not row:
            return None
        return (row["entity1_type"], row["entity1_id"], row["entity2_type"], row["entity2_id"])

    @staticmethod
    def _map_dynamic_name(name: str) -> Optional[str]:
        """Map friendly names like 'tension' to canonical dimension keys."""
        if not name:
            return None
        key = (name or "").lower()
        return _DIMENSION_ALIAS.get(key, key)

    # ==================== Relationship creation/retrieval ====================

    async def create_relationship(
        self,
        entity1_type: str,
        entity1_id: int,
        entity2_type: str,
        entity2_id: int,
        relationship_type: str = "neutral",
        initial_level: int = 0
    ) -> Dict[str, Any]:
        """
        Create a new relationship between two entities via IntegratedNPCSystem.
        """
        try:
            npc = await self._get_integrated()
            return await npc.create_direct_social_link(
                entity1_type=entity1_type,
                entity1_id=int(entity1_id),
                entity2_type=entity2_type,
                entity2_id=int(entity2_id),
                link_type=relationship_type,
                link_level=int(initial_level)
            )
        except Exception as e:
            logger.exception(f"create_relationship failed: {e}")
            return {"error": str(e)}

    async def get_relationship(
        self,
        entity1_type: str,
        entity1_id: int,
        entity2_type: str,
        entity2_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get relationship data via IntegratedNPCSystem.get_relationship."""
        try:
            npc = await self._get_integrated()
            return await npc.get_relationship(
                entity1_type, int(entity1_id), entity2_type, int(entity2_id)
            )
        except Exception as e:
            logger.exception(f"get_relationship failed: {e}")
            return None

    # ==================== Dimension-level updates (link_id-based) ====================

    async def update_dimensions(
        self,
        link_id: int,
        dimension_changes: Dict[str, Union[int, float]],
        reason: str = None
    ) -> Dict[str, Any]:
        """
        Update specific dimensions of a relationship using OptimizedRelationshipManager.
        Accepts link_id, resolves to entities, updates dimensions, clamps, and flushes.
        """
        try:
            mgr = await self._get_manager()
            ents = await self._resolve_link_entities(int(link_id))
            if not ents:
                return {"error": "link_not_found"}
            e1t, e1i, e2t, e2i = ents

            state = await mgr.get_relationship_state(e1t, e1i, e2t, e2i)
            before = dict(state.dimensions.to_dict())  # shallow copy

            for raw_key, delta in (dimension_changes or {}).items():
                key = self._map_dynamic_name(raw_key)
                if key and hasattr(state.dimensions, key):
                    cur = getattr(state.dimensions, key) or 0.0
                    setattr(state.dimensions, key, float(cur) + float(delta))

            state.dimensions.clamp()
            await mgr._queue_update(state)
            await mgr._flush_updates()

            after = state.dimensions.to_dict()
            diff = {k: (after.get(k, 0) - before.get(k, 0)) for k in set(after) | set(before)}
            return {"success": True, "diff": diff, "reason": reason}
        except Exception as e:
            logger.exception(f"update_dimensions failed for link_id={link_id}: {e}")
            return {"error": str(e)}

    async def increase_tension(self, link_id: int, amount: int, reason: str = None) -> Dict[str, Any]:
        """
        Increase relationship 'tension' (mapped to volatility).
        """
        try:
            return await self.update_dimensions(int(link_id), {"volatility": int(amount)}, reason)
        except Exception as e:
            logger.exception(f"increase_tension failed: {e}")
            return {"error": str(e)}

    async def release_tension(
        self,
        link_id: int,
        amount: int,
        resolution_type: str = "positive",
        reason: str = None
    ) -> Dict[str, Any]:
        """
        Release relationship tension: reduce unresolved_conflict and volatility (if positive resolution).
        """
        try:
            deltas = {"unresolved_conflict": -int(amount)}
            if (resolution_type or "").lower() == "positive":
                deltas["volatility"] = -int(amount // 2)
            return await self.update_dimensions(int(link_id), deltas, reason)
        except Exception as e:
            logger.exception(f"release_tension failed: {e}")
            return {"error": str(e)}

    # ==================== Events and crossroads ====================

    async def check_for_events(self) -> List[Dict[str, Any]]:
        """
        Drain pending relationship events from the global event generator.
        """
        try:
            from logic.dynamic_relationships import event_generator
            return await event_generator.drain_events(max_events=50)
        except Exception as e:
            logger.debug(f"[RelIntegration] event drain failed: {e}")
            return []

    async def apply_crossroads_choice(self, crossroads: Dict[str, Any], choice_index: int) -> Dict[str, Any]:
        """
        Apply a relationship crossroads choice using IntegratedNPCSystem.
        Accepts a dict with keys:
          entity1_type, entity1_id, entity2_type, entity2_id,
          event_type, description, options (list), expires_in
        """
        try:
            from logic.fully_integrated_npc_system import CrossroadsEvent
            npc = await self._get_integrated()

            ev = CrossroadsEvent(
                entity1_type=str(crossroads.get("entity1_type", "player")),
                entity1_id=int(crossroads.get("entity1_id", self.user_id)),
                entity2_type=str(crossroads.get("entity2_type", "npc")),
                entity2_id=int(crossroads.get("entity2_id")),
                relationship_state=None,  # state not required for apply
                event_type=str(crossroads.get("event_type", "relationship_crossroads")),
                description=str(crossroads.get("description", "")),
                options=list(crossroads.get("options", [])),
                expires_in=int(crossroads.get("expires_in", 3)),
            )
            return await npc.apply_crossroads_choice(ev, int(choice_index))
        except Exception as e:
            logger.exception(f"apply_crossroads_choice failed: {e}")
            return {"error": str(e)}

    async def add_link_event(self, link_id: int, event_text: str) -> bool:
        """
        Add an event note to a relationship's history via IntegratedNPCSystem.add_event_to_link.
        Resolves link_id to its entities and forwards.
        """
        try:
            npc = await self._get_integrated()
            ents = await self._resolve_link_entities(int(link_id))
            if not ents:
                return False
            e1t, e1i, e2t, e2i = ents
            return await npc.add_event_to_link(e1t, e1i, e2t, e2i, event_text)
        except Exception as e:
            logger.debug(f"[RelIntegration] add_link_event failed: {e}")
            return False

    # ==================== Dynamic accessors by entity pair ====================

    async def get_dynamic_level(
        self,
        entity1_type: str,
        entity1_id: int,
        entity2_type: str,
        entity2_id: int,
        dynamic_name: str
    ) -> int:
        """
        Get the current level of a relationship dynamic (dimension) by entity pair.
        """
        try:
            mgr = await self._get_manager()
            key = self._map_dynamic_name(dynamic_name)
            if not key:
                return 0
            state = await mgr.get_relationship_state(entity1_type, int(entity1_id), entity2_type, int(entity2_id))
            return int(round(getattr(state.dimensions, key, 0) or 0))
        except Exception as e:
            logger.exception(f"get_dynamic_level failed: {e}")
            return 0

    async def update_dynamic(
        self,
        entity1_type: str,
        entity1_id: int,
        entity2_type: str,
        entity2_id: int,
        dynamic_name: str,
        change: int
    ) -> int:
        """
        Update a relationship dynamic (dimension) by entity pair.
        """
        try:
            mgr = await self._get_manager()
            key = self._map_dynamic_name(dynamic_name)
            if not key:
                return 0
            state = await mgr.get_relationship_state(entity1_type, int(entity1_id), entity2_type, int(entity2_id))
            cur = getattr(state.dimensions, key, 0) or 0.0
            setattr(state.dimensions, key, float(cur) + float(change))
            state.dimensions.clamp()
            await mgr._queue_update(state)
            await mgr._flush_updates()
            return int(round(getattr(state.dimensions, key, 0)))
        except Exception as e:
            logger.exception(f"update_dynamic failed: {e}")
            return 0

    # ==================== Group mechanics ====================

    async def create_npc_group(self, name: str, description: str, member_ids: List[int]) -> Dict[str, Any]:
        """
        Create a group of NPCs in NPCGroups with initial dynamics and membership.
        """
        try:
            members: List[Dict[str, Any]] = []
            # Fetch NPC names/dominance for given IDs
            if member_ids:
                async with get_db_connection_context() as conn:
                    rows = await conn.fetch("""
                        SELECT npc_id, npc_name, dominance
                        FROM NPCStats
                        WHERE npc_id = ANY($1::int[]) AND user_id=$2 AND conversation_id=$3
                    """, member_ids, self.user_id, self.conversation_id)
                row_map = {r["npc_id"]: r for r in rows}
                for mid in member_ids:
                    r = row_map.get(mid)
                    if r:
                        members.append({
                            "npc_id": int(r["npc_id"]),
                            "npc_name": r["npc_name"],
                            "dominance": r["dominance"],
                            "joined_date": datetime.now().isoformat(),
                            "status": "active",
                            "role": "member"
                        })

            group_data = {
                "description": description or "",
                "members": members,
                "dynamics": {
                    "hierarchy": random.randint(30, 70),
                    "cohesion": random.randint(30, 70),
                    "secrecy": random.randint(30, 70),
                    "territoriality": random.randint(30, 70),
                    "exclusivity": random.randint(30, 70)
                },
                "shared_history": []
            }

            # Insert row; prefer DB for creation (LoreSystem can govern updates afterwards)
            async with get_db_connection_context() as conn:
                group_id = await conn.fetchval("""
                    INSERT INTO NPCGroups (user_id, conversation_id, group_name, group_data, created_at)
                    VALUES ($1, $2, $3, $4::jsonb, CURRENT_TIMESTAMP)
                    RETURNING group_id
                """, self.user_id, self.conversation_id, name, json.dumps(group_data))

            return {
                "group_id": int(group_id),
                "group_name": name,
                "group_data": group_data
            }
        except Exception as e:
            logger.exception(f"create_npc_group failed: {e}")
            return {"error": str(e)}

    async def add_npc_to_group(self, group_id: int, npc_id: int, role: str = "member") -> bool:
        """
        Add an NPC to an existing group (governed via LoreSystem update).
        """
        try:
            async with get_db_connection_context() as conn:
                group_row = await conn.fetchrow("""
                    SELECT group_data
                    FROM NPCGroups
                    WHERE group_id=$1 AND user_id=$2 AND conversation_id=$3
                """, int(group_id), self.user_id, self.conversation_id)

                if not group_row:
                    logger.warning(f"Group {group_id} not found for user {self.user_id}, convo {self.conversation_id}.")
                    return False

                group_data = group_row['group_data']
                if isinstance(group_data, str):
                    try:
                        group_data = json.loads(group_data)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in group_data for group {group_id}.", exc_info=True)
                        return False
                elif group_data is None:
                    group_data = {}

                # Get NPC data
                npc_row = await conn.fetchrow("""
                    SELECT npc_name, dominance
                    FROM NPCStats
                    WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
                """, int(npc_id), self.user_id, self.conversation_id)

                if not npc_row:
                    logger.warning(f"NPC {npc_id} not found for user {self.user_id}, convo {self.conversation_id}.")
                    return False

                npc_name, dominance = npc_row['npc_name'], npc_row['dominance']

                # Add member if not already present
                members = group_data.setdefault("members", [])
                if any(m.get("npc_id") == int(npc_id) for m in members):
                    logger.info(f"NPC {npc_id} already in group {group_id}. Skipping add.")
                    return True

                members.append({
                    "npc_id": int(npc_id),
                    "npc_name": npc_name,
                    "dominance": dominance,
                    "joined_date": datetime.now().isoformat(),
                    "status": "active",
                    "role": role
                })

                # Update via LoreSystem
                lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
                result = await lore_system.propose_and_enact_change(
                    ctx=self.ctx,
                    entity_type="NPCGroups",
                    entity_identifier={"group_id": int(group_id), "user_id": self.user_id, "conversation_id": self.conversation_id},
                    updates={"group_data": json.dumps(group_data)},
                    reason=f"Added NPC {npc_name} to group as {role}"
                )

                if result.get("status") == "committed":
                    logger.info(f"Successfully added NPC {npc_id} to group {group_id}.")
                    return True
                else:
                    logger.error(f"Failed to add NPC {npc_id} to group {group_id}: {result}")
                    return False

        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"Database error adding NPC {npc_id} to group {group_id}: {db_err}", exc_info=True)
            return False
        except Exception as e:
            logger.exception(f"Unexpected error adding NPC {npc_id} to group {group_id}: {e}")
            return False

    async def generate_group_dynamics(self, group_id: int) -> Dict[str, Any]:
        """
        Generate group dynamics and a few random events for a group. Also ensures
        pairwise relationships exist among group members.
        """
        try:
            async with get_db_connection_context() as conn:
                group_row = await conn.fetchrow("""
                    SELECT group_data, group_name
                    FROM NPCGroups
                    WHERE group_id=$1 AND user_id=$2 AND conversation_id=$3
                """, int(group_id), self.user_id, self.conversation_id)

                if not group_row:
                    logger.warning(f"Group {group_id} not found during dynamics generation.")
                    return {"error": "Group not found"}

                group_data = group_row['group_data']
                group_name = group_row['group_name']

                # JSON decode if needed
                if isinstance(group_data, str):
                    try:
                        group_data = json.loads(group_data)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in group_data for group {group_id} during dynamics.", exc_info=True)
                        return {"error": "Invalid group data"}
                elif group_data is None:
                    group_data = {}

                members = group_data.setdefault("members", [])
                member_ids = [int(m.get("npc_id")) for m in members if m.get("npc_id") is not None]

            relationships = []
            # Ensure relationships exist among members
            for i in range(len(member_ids)):
                for j in range(i + 1, len(member_ids)):
                    npc1_id = member_ids[i]
                    npc2_id = member_ids[j]
                    # Try to fetch existing; if none, create
                    rel = await self.get_relationship("npc", npc1_id, "npc", npc2_id)
                    if not rel:
                        rel_type = random.choice(["neutral", "alliance", "rivalry", "dominant", "submission"])
                        rel = await self.create_relationship("npc", npc1_id, "npc", npc2_id, rel_type)
                    relationships.append(rel)

            # Generate small set of group events
            events = []
            if members:
                for _ in range(min(3, max(1, len(members)))):
                    event_type = random.choice(["meeting", "conflict", "collaboration", "celebration", "crisis"])
                    if event_type == "meeting":
                        events.append("The group held a meeting to discuss their goals and plans.")
                    elif event_type == "conflict":
                        conflict_members = random.sample(members, min(2, len(members)))
                        events.append(f"{conflict_members[0].get('npc_name')} and {conflict_members[1].get('npc_name')} had a disagreement about the group's direction.")
                    elif event_type == "collaboration":
                        events.append("The group worked together on a project, strengthening their bonds.")
                    elif event_type == "celebration":
                        events.append("The group celebrated a significant achievement together.")
                    elif event_type == "crisis":
                        events.append("The group faced a crisis that tested their unity and resolve.")

            # Update basic group dynamics if missing
            dynamics = group_data.get("dynamics", {})
            if not dynamics:
                dynamics = {
                    "hierarchy": random.randint(30, 70),
                    "cohesion": random.randint(30, 70),
                    "secrecy": random.randint(30, 70),
                    "territoriality": random.randint(30, 70),
                    "exclusivity": random.randint(30, 70)
                }
            group_data["dynamics"] = dynamics
            group_data["shared_history"] = group_data.get("shared_history", []) + events

            lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
            result = await lore_system.propose_and_enact_change(
                ctx=self.ctx,
                entity_type="NPCGroups",
                entity_identifier={"group_id": int(group_id), "user_id": self.user_id, "conversation_id": self.conversation_id},
                updates={"group_data": json.dumps(group_data)},
                reason=f"Generated dynamics and events for group {group_name}"
            )

            if result.get("status") == "committed":
                logger.info(f"Generated dynamics for group {group_id} ('{group_name}').")
                return {
                    "group_id": int(group_id),
                    "group_name": group_name,
                    "dynamics": dynamics,
                    "events": events,
                    "relationships": relationships
                }
            else:
                logger.error(f"Failed to update group dynamics: {result}")
                return {"error": "Failed to update group dynamics"}

        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"Database error generating dynamics for group {group_id}: {db_err}", exc_info=True)
            return {"error": f"Database error: {db_err}"}
        except Exception as e:
            logger.exception(f"Unexpected error generating dynamics for group {group_id}: {e}")
            return {"error": f"Unexpected error: {e}"}

    # ==================== Utilities ====================

    async def get_entity_name(self, entity_type: str, entity_id: int) -> str:
        """
        Get the display name of an entity.
        """
        if (entity_type or "").lower() == "player":
            return "Player"
        try:
            async with get_db_connection_context() as conn:
                npc_name = await conn.fetchval("""
                    SELECT npc_name
                    FROM NPCStats
                    WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
                """, int(entity_id), self.user_id, self.conversation_id)
                return npc_name if npc_name else f"Unknown NPC ({entity_id})"
        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"Database error fetching name for {entity_type} {entity_id}: {db_err}", exc_info=True)
            return f"DB Error ({entity_type} {entity_id})"
        except Exception as e:
            logger.exception(f"Unexpected error fetching name for {entity_type} {entity_id}: {e}")
            return f"Error ({entity_type} {entity_id})"

    async def get_player_relationships(self) -> List[Dict[str, Any]]:
        """
        Get all relationships involving the player (list for UI).
        """
        relationships: List[Dict[str, Any]] = []
        try:
            player_id = self.user_id
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id,
                           link_type, link_level
                    FROM SocialLinks
                    WHERE user_id=$1 AND conversation_id=$2
                      AND ((entity1_type='player' AND entity1_id=$3) OR
                           (entity2_type='player' AND entity2_id=$3))
                """, self.user_id, self.conversation_id, player_id)

            # Fetch names concurrently
            tasks = []
            for row in rows:
                link_id = row['link_id']
                link_type = row['link_type']
                link_level = row['link_level']

                if row['entity1_type'] == "player":
                    npc_type, npc_id = row['entity2_type'], row['entity2_id']
                else:
                    npc_type, npc_id = row['entity1_type'], row['entity1_id']

                async def fetch_and_format(lt, ll, nt, ni, lid):
                    npc_name = await self.get_entity_name(nt, ni)
                    return {
                        "link_id": int(lid),
                        "npc_id": int(ni),
                        "npc_name": npc_name,
                        "relationship_type": lt,
                        "relationship_level": ll
                    }

                tasks.append(fetch_and_format(link_type, link_level, npc_type, npc_id, link_id))

            if tasks:
                relationships = await asyncio.gather(*tasks)

            return relationships

        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
            logger.error(f"Database error fetching player relationships for user {self.user_id}: {db_err}", exc_info=True)
            return []
        except Exception as e:
            logger.exception(f"Unexpected error fetching player relationships for user {self.user_id}: {e}")
            return []
