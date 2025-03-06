# logic/npc_agents/decision_engine.py

"""
Decision engine for NPC agents
"""

import random
import json
import logging
from typing import List, Dict, Any, Optional
from db.connection import get_db_connection

class NPCDecisionEngine:
    """Handles decision-making for an individual NPC"""

    def __init__(self, npc_id, user_id, conversation_id):
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def decide(self, perception, available_actions=None):
        """
        Make a decision based on current perceptions and available actions
        """
        # Get NPC's personality and stats
        npc_data = await self.get_npc_data()

        # Prepare decision context
        decision_context = {
            "npc": npc_data,
            "perception": perception,
            "available_actions": available_actions or await self.get_default_actions(npc_data, perception)
        }

        # Score each available action
        scored_actions = await self.score_actions(decision_context)

        # Select the highest-scoring action, with randomness to keep it fresh
        selected_action = await self.select_action(scored_actions)

        # Store this decision
        await self.store_decision(selected_action, decision_context)

        return selected_action

    async def get_npc_data(self):
        """Get NPC's current data from the database"""
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT npc_name, dominance, cruelty, closeness, trust,
                       respect, intensity, hobbies, personality_traits,
                       likes, dislikes, schedule, current_location, sex
                FROM NPCStats
                WHERE npc_id = %s
                  AND user_id = %s
                  AND conversation_id = %s
            """, (self.npc_id, self.user_id, self.conversation_id))
            row = cursor.fetchone()
            if not row:
                return {}

            (name, dom, cru, clos, tru, resp, inten,
             hobbies, traits, likes, dislikes,
             sched, loc, sex) = row

            npc_data = {
                "npc_id": self.npc_id,
                "npc_name": name,
                "dominance": dom,
                "cruelty": cru,
                "closeness": clos,
                "trust": tru,
                "respect": resp,
                "intensity": inten,
                "hobbies": hobbies if hobbies else [],
                "personality_traits": traits if traits else [],
                "likes": likes if likes else [],
                "dislikes": dislikes if dislikes else [],
                "schedule": sched if sched else {},
                "current_location": loc,
                "sex": sex
            }
            return npc_data
        except Exception as e:
            logging.error(f"Error getting NPC data: {e}")
            return {}
        finally:
            conn.close()

    async def get_default_actions(self, npc_data, perception):
        """Get default actions based on NPC's personality and context"""
        # Base set of actions
        actions = [
            {
                "type": "talk",
                "description": "Engage in friendly conversation",
                "target": "player",
                "stats_influenced": {"closeness": +2, "trust": +1}
            },
            {
                "type": "observe",
                "description": "Observe quietly",
                "target": "environment",
                "stats_influenced": {}
            },
            {
                "type": "leave",
                "description": "Exit the current location",
                "target": "location",
                "stats_influenced": {}
            }
        ]

        # Add dominance-based actions
        if npc_data.get("dominance", 0) > 60:
            actions.append({
                "type": "command",
                "description": "Give an authoritative command",
                "target": "player",
                "stats_influenced": {"dominance": +1, "trust": -1}
            })
            actions.append({
                "type": "test",
                "description": "Test player's obedience",
                "target": "player",
                "stats_influenced": {"dominance": +2, "respect": -1}
            })

        # Add cruelty-based actions
        if npc_data.get("cruelty", 0) > 60:
            actions.append({
                "type": "mock",
                "description": "Mock or belittle the player",
                "target": "player",
                "stats_influenced": {"cruelty": +1, "closeness": -2}
            })

        # Add trust-based actions
        if npc_data.get("trust", 0) > 60:
            actions.append({
                "type": "confide",
                "description": "Share a personal secret",
                "target": "player",
                "stats_influenced": {"trust": +3, "closeness": +2}
            })

        # Add respect-based actions
        if npc_data.get("respect", 0) > 60:
            actions.append({
                "type": "praise",
                "description": "Praise the player's abilities",
                "target": "player",
                "stats_influenced": {"respect": +2, "closeness": +1}
            })

        return actions

    async def score_actions(self, decision_context):
        """Score each action based on alignment with NPC's personality/goals"""
        npc = decision_context["npc"]
        perception = decision_context["perception"]
        actions = decision_context["available_actions"]

        scored_actions = []
        for action in actions:
            score = 0
            # 1. Personality alignment
            score += await self.score_personality_alignment(npc, action)
            # 2. Memory influence
            score += await self.score_memory_influence(
                perception.get("relevant_memories", []),
                action
            )
            # 3. Relationship influence
            score += await self.score_relationship_influence(
                perception.get("relationships", {}),
                action
            )

            scored_actions.append({
                "action": action,
                "score": score
            })

        # Sort by score descending
        scored_actions.sort(key=lambda x: x["score"], reverse=True)
        return scored_actions

    async def score_personality_alignment(self, npc, action):
        """Score how well an action aligns with NPC's personality"""
        score = 0
        stats_influenced = action.get("stats_influenced", {})
        for stat, change in stats_influenced.items():
            if stat in npc:
                current_value = npc[stat]
                # If change direction matches stat tendency, add points
                if change > 0 and current_value > 50:
                    score += 2
                elif change < 0 and current_value < 50:
                    score += 2

        personality_traits = npc.get("personality_traits", [])
        trait_alignments = {
            "commanding": {"command": 3, "test": 2},
            "cruel": {"mock": 3},
            "kind": {"talk": 2, "praise": 3},
            "shy": {"observe": 3, "leave": 2},
            "confident": {"talk": 2, "command": 1},
            "manipulative": {"confide": 3, "praise": 2},
            "honest": {"confide": 2},
            "suspicious": {"observe": 2}
        }
        for trait in personality_traits:
            trait_lower = trait.lower()
            if trait_lower in trait_alignments:
                for action_type, bonus in trait_alignments[trait_lower].items():
                    if action["type"] == action_type:
                        score += bonus
        return score

    async def score_memory_influence(self, memories, action):
        """Score how memories influence action choice"""
        score = 0
        for memory in memories:
            memory_text = memory.get("memory_text", "").lower()

            # Action-specific keyword boosts
            if action["type"] == "talk" and ("talked" in memory_text or "conversation" in memory_text):
                score += 1
            elif action["type"] == "command" and ("commanded" in memory_text or "ordered" in memory_text):
                score += 1
            elif action["type"] == "mock" and ("mocked" in memory_text or "laughed at" in memory_text):
                score += 1

            # Target-specific scoring
            if "target" in action:
                target = action["target"]
                if target in memory_text:
                    score += 2

            # Emotional valence scoring
            emotional_intensity = memory.get("emotional_intensity", 0)
            if emotional_intensity > 70:
                # Strong positive emotion
                if action["type"] in ["talk", "praise", "confide"]:
                    score += 2
            elif emotional_intensity < 30:
                # Strong negative emotion
                if action["type"] in ["mock", "leave", "observe"]:
                    score += 2
        return score

    async def score_relationship_influence(self, relationships, action):
        """Score how relationships influence action choice"""
        score = 0
        if "target" in action and action["target"] == "player":
            player_rel = relationships.get("player", {})

            # Relationship type
            rel_type = player_rel.get("link_type", "")
            if rel_type == "dominant" and action["type"] in ["command", "test"]:
                score += 3
            elif rel_type == "friendly" and action["type"] in ["talk", "confide"]:
                score += 3
            elif rel_type == "hostile" and action["type"] in ["mock", "leave"]:
                score += 3

            # Relationship level
            rel_level = player_rel.get("link_level", 0)
            if rel_level > 70:
                if action["type"] in ["talk", "confide", "praise"]:
                    score += 2
            elif rel_level < 30:
                if action["type"] in ["observe", "leave"]:
                    score += 2
        return score

    async def select_action(self, scored_actions, randomness=0.2):
        """Select an action with some randomness to avoid predictability"""
        if not scored_actions:
            return {"type": "idle", "description": "Do nothing"}

        # Add randomness
        for action in scored_actions:
            action["score"] += random.uniform(0, randomness * 10)

        # Re-sort
        scored_actions.sort(key=lambda x: x["score"], reverse=True)
        return scored_actions[0]["action"]

    async def store_decision(self, action, context):
        """Store this decision for future reference"""
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE NPCAgentState
                SET last_decision = %s
                WHERE npc_id = %s
                  AND user_id = %s
                  AND conversation_id = %s
            """, (
                json.dumps(action),
                self.npc_id,
                self.user_id,
                self.conversation_id
            ))

            # If no row was updated, insert a new one
            if cursor.rowcount == 0:
                cursor.execute("""
                    INSERT INTO NPCAgentState
                        (npc_id, user_id, conversation_id, last_decision)
                    VALUES (%s, %s, %s, %s)
                """, (
                    self.npc_id,
                    self.user_id,
                    self.conversation_id,
                    json.dumps(action)
                ))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logging.error(f"Error storing decision: {e}")
        finally:
            conn.close()
