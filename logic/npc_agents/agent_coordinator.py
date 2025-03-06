# logic/npc_agents/agent_coordinator.py

"""
Coordinates multiple NPC agents for group interactions, with improved structure and clarity.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from db.connection import get_db_connection
from .npc_agent import NPCAgent

logger = logging.getLogger(__name__)

@dataclass
class NPCAction:
    """
    Simple data class to represent an NPC's chosen action.
    """
    type: str
    description: str
    target: Optional[str] = None            # Could be 'group', another npc_id, etc.
    target_name: Optional[str] = None
    stats_influenced: Dict[str, int] = None


class NPCAgentCoordinator:
    """Coordinates the behavior of multiple NPC agents."""

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.active_agents: Dict[int, NPCAgent] = {}  # Map of npc_id -> NPCAgent

    async def load_agents(self, npc_ids: Optional[List[int]] = None) -> List[int]:
        """
        Load specified NPC agents into memory, or load all if none specified.

        Returns:
            List of NPC IDs that were successfully loaded.
        """
        if npc_ids is None:
            logger.info("Loading all NPC agents for user=%s, conversation=%s", self.user_id, self.conversation_id)
        else:
            logger.info("Loading NPC agents: %s", npc_ids)

        query = """
            SELECT npc_id
            FROM NPCStats
            WHERE user_id=%s
              AND conversation_id=%s
        """
        params = [self.user_id, self.conversation_id]

        if npc_ids:
            query += " AND npc_id = ANY(%s)"
            params.append(npc_ids)

        loaded_ids: List[int] = []
        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

            for row in rows:
                npc_id = row[0]
                if npc_id not in self.active_agents:
                    self.active_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
                loaded_ids.append(npc_id)

        logger.info("Loaded agents: %s", loaded_ids)
        return loaded_ids

    async def make_group_decisions(
        self,
        npc_ids: List[int],
        shared_context: Dict[str, Any],
        available_actions: Optional[Dict[int, List[NPCAction]]] = None
    ) -> Dict[str, Any]:
        """
        Coordinate decision-making for a group of NPCs.

        Args:
            npc_ids: List of NPC IDs in the group
            shared_context: A dictionary describing the shared context/environment
            available_actions: Optional dict of predefined actions per NPC

        Returns:
            A dictionary representing the coordinated 'action_plan' for the entire group.
        """
        # Ensure all NPCs are loaded
        await self.load_agents(npc_ids)

        # 1) Each NPC perceives the environment (run concurrently for performance)
        perceive_tasks = []
        for npc_id in npc_ids:
            agent = self.active_agents.get(npc_id)
            if agent:
                perceive_tasks.append(agent.perceive_environment(shared_context))
            else:
                perceive_tasks.append(asyncio.sleep(0))  # Filler if agent is missing

        perceptions_list = await asyncio.gather(*perceive_tasks)
        perceptions = {npc_id: perceptions_list[i] for i, npc_id in enumerate(npc_ids)}

        # 2) Determine available actions if not provided
        if available_actions is None:
            available_actions = await self.generate_group_actions(npc_ids, perceptions)

        # 3) Each NPC decides individually (also run concurrently)
        decision_tasks = []
        for npc_id in npc_ids:
            agent = self.active_agents.get(npc_id)
            npc_actions = available_actions.get(npc_id, [])
            if agent:
                decision_tasks.append(agent.make_decision(perceptions[npc_id], npc_actions))
            else:
                decision_tasks.append(asyncio.sleep(0))

        decisions_list = await asyncio.gather(*decision_tasks)
        decisions = {npc_id: decisions_list[i] for i, npc_id in enumerate(npc_ids)}

        # 4) Resolve conflicts into a coherent plan
        action_plan = await self.resolve_decision_conflicts(decisions, npc_ids, perceptions)
        return action_plan

    async def generate_group_actions(
        self,
        npc_ids: List[int],
        perceptions: Dict[int, Any]
    ) -> Dict[int, List[NPCAction]]:
        """
        Generate possible actions for each NPC in a group context.

        Args:
            npc_ids: IDs of the NPCs
            perceptions: Each NPC's environment perception (unused in this basic example)

        Returns:
            A dict of npc_id -> list of NPCAction objects
        """
        npc_data = self._fetch_basic_npc_data(npc_ids)
        group_actions: Dict[int, List[NPCAction]] = {}

        for npc_id in npc_ids:
            if npc_id not in npc_data:
                continue

            actions = [
                NPCAction(type="talk", description="Talk to the group", target="group"),
                NPCAction(type="observe", description="Observe the group", target="group"),
                NPCAction(type="leave", description="Leave the group", target="group")
            ]

            # Possibly add actions targeting other NPCs
            dom = npc_data[npc_id]["dominance"]
            cru = npc_data[npc_id]["cruelty"]

            for other_id in npc_ids:
                if other_id != npc_id and other_id in npc_data:
                    other_name = npc_data[other_id]["npc_name"]
                    actions.append(NPCAction(
                        type="talk_to",
                        description=f"Talk to {other_name}",
                        target=str(other_id),
                        target_name=other_name
                    ))
                    if dom > 60:
                        actions.append(NPCAction(
                            type="command",
                            description=f"Command {other_name}",
                            target=str(other_id),
                            target_name=other_name
                        ))
                    if cru > 60:
                        actions.append(NPCAction(
                            type="mock",
                            description=f"Mock {other_name}",
                            target=str(other_id),
                            target_name=other_name
                        ))

            group_actions[npc_id] = actions

        return group_actions

    async def resolve_decision_conflicts(
        self,
        decisions: Dict[int, NPCAction],
        npc_ids: List[int],
        perceptions: Dict[int, Any]
    ) -> Dict[str, Any]:
        """
        Resolve conflicts between NPC decisions to form a coherent plan.

        This simplistic example sorts NPCs by dominance and tries to apply each NPC's action
        in descending order, skipping an action if the target is already 'affected'.

        Returns:
            A structured dict with group_actions and individual_actions.
        """
        npc_dominance = self._fetch_npc_dominance(npc_ids)
        # Sort by dominance descending
        sorted_npcs = sorted(npc_ids, key=lambda id_: npc_dominance.get(id_, 0), reverse=True)

        action_plan = {"group_actions": [], "individual_actions": {}}
        affected_npcs: Set[int] = set()

        for npc_id in sorted_npcs:
            action = decisions.get(npc_id)
            if not action:
                continue

            # Group actions
            if action.type in ["talk", "command"] and action.target == "group":
                action_plan["group_actions"].append({"npc_id": npc_id, "action": action.__dict__})
                affected_npcs.update(npc_ids)

            # Direct interactions
            elif action.type in ["talk_to", "command", "mock"] and action.target is not None:
                try:
                    target_id = int(action.target)  # The other NPC's ID
                except ValueError:
                    target_id = -1

                if target_id not in affected_npcs:
                    if npc_id not in action_plan["individual_actions"]:
                        action_plan["individual_actions"][npc_id] = []
                    action_plan["individual_actions"][npc_id].append(action.__dict__)
                    affected_npcs.add(target_id)

            # Other actions
            else:
                if npc_id not in action_plan["individual_actions"]:
                    action_plan["individual_actions"][npc_id] = []
                action_plan["individual_actions"][npc_id].append(action.__dict__)

        logger.info("Resolved action plan: %s", action_plan)
        return action_plan

    async def handle_player_action(
        self,
        player_action: Dict[str, Any],
        context: Dict[str, Any],
        npc_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Handle a player action directed at multiple NPCs.

        If npc_ids is None, we attempt to find them based on location or 'target_npc_id'.
        Returns a dict containing responses from each NPC.
        """
        affected_npcs = npc_ids or await self.determine_affected_npcs(player_action, context)
        if not affected_npcs:
            return {"npc_responses": []}

        await self.load_agents(affected_npcs)

        response_tasks = []
        for npc_id in affected_npcs:
            agent = self.active_agents.get(npc_id)
            if agent:
                response_tasks.append(agent.process_player_action(player_action))
            else:
                response_tasks.append(asyncio.sleep(0))

        responses = await asyncio.gather(*response_tasks)
        return {"npc_responses": responses}

    async def determine_affected_npcs(self, player_action: Dict[str, Any], context: Dict[str, Any]) -> List[int]:
        """
        Determine which NPCs are affected by a player action.

        If "target_npc_id" is in the action, return that. Otherwise, find all NPCs in the
        current location from the context.
        """
        if "target_npc_id" in player_action:
            return [player_action["target_npc_id"]]

        current_location = context.get("location", "Unknown")
        logger.debug("Determining NPCs at location=%s", current_location)

        query = """
            SELECT npc_id
            FROM NPCStats
            WHERE user_id=%s
              AND conversation_id=%s
              AND current_location=%s
        """
        params = (self.user_id, self.conversation_id, current_location)
        npc_list: List[int] = []

        with get_db_connection() as conn, conn.cursor() as cursor:
            try:
                cursor.execute(query, params)
                npc_list = [row[0] for row in cursor.fetchall()]
            except Exception as e:
                logger.error("Error determining affected NPCs: %s", e)

        return npc_list

    # ----------------------------------------------------------------
    # Internal helper methods
    # ----------------------------------------------------------------

    def _fetch_basic_npc_data(self, npc_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Fetch minimal NPC data (name, dominance, cruelty).
        """
        data_map: Dict[int, Dict[str, Any]] = {}
        if not npc_ids:
            return data_map

        query = f"""
            SELECT npc_id, npc_name, dominance, cruelty
            FROM NPCStats
            WHERE npc_id = ANY(%s)
              AND user_id = %s
              AND conversation_id = %s
        """
        params = (npc_ids, self.user_id, self.conversation_id)

        with get_db_connection() as conn, conn.cursor() as cursor:
            try:
                cursor.execute(query, params)
                for row in cursor.fetchall():
                    nid, name, dom, cru = row
                    data_map[nid] = {
                        "npc_name": name,
                        "dominance": dom,
                        "cruelty": cru
                    }
            except Exception as e:
                logger.error("Error fetching basic NPC data: %s", e)

        return data_map

    def _fetch_npc_dominance(self, npc_ids: List[int]) -> Dict[int, int]:
        """
        Fetch only dominance values for the NPCs, used for sorting, etc.
        """
        dom_map: Dict[int, int] = {}
        if not npc_ids:
            return dom_map

        query = f"""
            SELECT npc_id, dominance
            FROM NPCStats
            WHERE npc_id = ANY(%s)
              AND user_id = %s
              AND conversation_id = %s
        """
        params = (npc_ids, self.user_id, self.conversation_id)

        with get_db_connection() as conn, conn.cursor() as cursor:
            try:
                cursor.execute(query, params)
                for row in cursor.fetchall():
                    nid, dom = row
                    dom_map[nid] = dom
            except Exception as e:
                logger.error("Error fetching NPC dominance: %s", e)
        return dom_map
