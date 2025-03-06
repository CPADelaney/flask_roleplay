# logic/npc_agents/agent_coordinator.py

"""
Coordinates multiple NPC agents for group interactions
"""

import logging
from typing import List, Dict, Any, Optional
from db.connection import get_db_connection
from .npc_agent import NPCAgent

class NPCAgentCoordinator:
    """Coordinates the behavior of multiple NPC agents"""

    def __init__(self, user_id, conversation_id):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.active_agents = {}  # Map of npc_id -> NPCAgent

    async def load_agents(self, npc_ids=None):
        """Load specified NPC agents, or all if none specified"""
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            if npc_ids:
                cursor.execute("""
                    SELECT npc_id
                    FROM NPCStats
                    WHERE user_id=%s
                      AND conversation_id=%s
                      AND npc_id = ANY(%s)
                """, (self.user_id, self.conversation_id, npc_ids))
            else:
                cursor.execute("""
                    SELECT npc_id
                    FROM NPCStats
                    WHERE user_id=%s
                      AND conversation_id=%s
                """, (self.user_id, self.conversation_id))

            rows = cursor.fetchall()
            for row in rows:
                npc_id = row[0]
                if npc_id not in self.active_agents:
                    self.active_agents[npc_id] = NPCAgent(
                        npc_id, self.user_id, self.conversation_id
                    )
            return list(self.active_agents.keys())
        finally:
            cursor.close()
            conn.close()

    async def make_group_decisions(self, npc_ids, shared_context, available_actions=None):
        """Coordinate decision-making for a group of NPCs"""
        # Ensure all NPCs are loaded
        await self.load_agents(npc_ids)

        # Get each NPC's perception of the shared context
        perceptions = {}
        for npc_id in npc_ids:
            if npc_id in self.active_agents:
                perceptions[npc_id] = await self.active_agents[npc_id].perceive_environment(shared_context)

        # Determine available actions
        group_actions = available_actions or await self.generate_group_actions(npc_ids, perceptions)

        # Each NPC decides individually
        decisions = {}
        for npc_id in npc_ids:
            if npc_id in self.active_agents:
                decisions[npc_id] = await self.active_agents[npc_id].make_decision(
                    perceptions[npc_id],
                    group_actions.get(npc_id, [])
                )

        # Resolve conflicts
        action_plan = await self.resolve_decision_conflicts(decisions, npc_ids, perceptions)
        return action_plan

    async def generate_group_actions(self, npc_ids, perceptions):
        """Generate actions for each NPC in a group context"""
        group_actions = {}
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            placeholders = ", ".join(["%s"] * len(npc_ids))
            cursor.execute(f"""
                SELECT npc_id, npc_name, dominance, cruelty
                FROM NPCStats
                WHERE npc_id IN ({placeholders})
                  AND user_id = %s
                  AND conversation_id = %s
            """, npc_ids + [self.user_id, self.conversation_id])

            npc_data = {}
            for row in cursor.fetchall():
                npc_id, name, dom, cru = row
                npc_data[npc_id] = {
                    "npc_id": npc_id,
                    "npc_name": name,
                    "dominance": dom,
                    "cruelty": cru
                }

            # Generate basic actions for each NPC
            for npc_id in npc_ids:
                if npc_id in npc_data:
                    actions = [
                        {
                            "type": "talk",
                            "description": "Talk to the group",
                            "target": "group",
                            "stats_influenced": {}
                        },
                        {
                            "type": "observe",
                            "description": "Observe the group",
                            "target": "group",
                            "stats_influenced": {}
                        },
                        {
                            "type": "leave",
                            "description": "Leave the group",
                            "target": "group",
                            "stats_influenced": {}
                        }
                    ]
                    # Add actions targeting other NPCs
                    for other_id in npc_ids:
                        if other_id != npc_id and other_id in npc_data:
                            other_name = npc_data[other_id]["npc_name"]
                            actions.append({
                                "type": "talk_to",
                                "description": f"Talk to {other_name}",
                                "target": other_id,
                                "target_name": other_name,
                                "stats_influenced": {}
                            })
                            if npc_data[npc_id]["dominance"] > 60:
                                actions.append({
                                    "type": "command",
                                    "description": f"Command {other_name}",
                                    "target": other_id,
                                    "target_name": other_name,
                                    "stats_influenced": {}
                                })
                            if npc_data[npc_id]["cruelty"] > 60:
                                actions.append({
                                    "type": "mock",
                                    "description": f"Mock {other_name}",
                                    "target": other_id,
                                    "target_name": other_name,
                                    "stats_influenced": {}
                                })

                    group_actions[npc_id] = actions
            return group_actions
        except Exception as e:
            logging.error(f"Error generating group actions: {e}")
            return {}
        finally:
            conn.close()

    async def resolve_decision_conflicts(self, decisions, npc_ids, perceptions):
        """Resolve conflicts between NPC decisions to form a coherent plan"""
        npc_dominance = {}
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            placeholders = ", ".join(["%s"] * len(npc_ids))
            cursor.execute(f"""
                SELECT npc_id, dominance
                FROM NPCStats
                WHERE npc_id IN ({placeholders})
                  AND user_id = %s
                  AND conversation_id = %s
            """, npc_ids + [self.user_id, self.conversation_id])

            for row in cursor.fetchall():
                npc_id, dom = row
                npc_dominance[npc_id] = dom
        except Exception as e:
            logging.error(f"Error fetching NPC dominance: {e}")
        finally:
            conn.close()

        # Sort by dominance
        sorted_npcs = sorted(npc_ids, key=lambda npc_id: npc_dominance.get(npc_id, 0), reverse=True)

        action_plan = {
            "group_actions": [],
            "individual_actions": {}
        }
        affected_npcs = set()

        for npc_id in sorted_npcs:
            if npc_id not in decisions:
                continue
            action = decisions[npc_id]

            # Group actions
            if action["type"] in ["talk", "command"] and action["target"] == "group":
                action_plan["group_actions"].append({
                    "npc_id": npc_id,
                    "action": action
                })
                affected_npcs.update(npc_ids)
            # Direct interactions
            elif action["type"] in ["talk_to", "command", "mock"] and "target" in action:
                target_id = action["target"]
                if target_id not in affected_npcs:
                    if npc_id not in action_plan["individual_actions"]:
                        action_plan["individual_actions"][npc_id] = []
                    action_plan["individual_actions"][npc_id].append(action)
                    affected_npcs.add(target_id)
            # Other actions
            elif action["type"] in ["observe", "leave"]:
                if npc_id not in action_plan["individual_actions"]:
                    action_plan["individual_actions"][npc_id] = []
                action_plan["individual_actions"][npc_id].append(action)

        return action_plan

    async def handle_player_action(self, player_action, context, npc_ids=None):
        """Handle a player action directed at multiple NPCs"""
        # Which NPCs are affected?
        affected_npcs = npc_ids or await self.determine_affected_npcs(player_action, context)
        if not affected_npcs:
            return {"npc_responses": []}

        # Load the affected NPCs
        await self.load_agents(affected_npcs)

        # Get each NPC's response
        npc_responses = []
        for npc_id in affected_npcs:
            if npc_id in self.active_agents:
                response = await self.active_agents[npc_id].process_player_action(player_action)
                npc_responses.append(response)

        return {"npc_responses": npc_responses}

    async def determine_affected_npcs(self, player_action, context):
        """Determine which NPCs are affected by a player action"""
        if "target_npc_id" in player_action:
            return [player_action["target_npc_id"]]

        current_location = context.get("location", "Unknown")
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT npc_id
                FROM NPCStats
                WHERE user_id = %s
                  AND conversation_id = %s
                  AND current_location = %s
            """, (self.user_id, self.conversation_id, current_location))
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logging.error(f"Error determining affected NPCs: {e}")
            return []
        finally:
            conn.close()
