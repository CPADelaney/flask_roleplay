# logic/decision_engine.py

import random
import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

import asyncpg

DB_DSN = os.getenv("DB_DSN", "postgresql://user:pass@localhost:5432/yourdb")

def _parse_json_field(field) -> list:
    """Simple helper to parse JSON fields from DB rows."""
    if field is None:
        return []
    if isinstance(field, str):
        try:
            return json.loads(field)
        except json.JSONDecodeError:
            return []
    if isinstance(field, list):
        return field
    return []

class EnhancedDecisionEngine:
    """Sophisticated decision-making engine for NPCs"""

    def __init__(self, npc_id: int, user_id: int, conversation_id: int):
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def decide(self, perception: Dict[str, Any], available_actions: Optional[List[dict]]=None) -> dict:
        """
        Evaluate the NPC's current state, context, and personality to pick an action.
        Now with more sophisticated reasoning and influence factors.
        """
        npc_data = await self.get_npc_data()
        if not npc_data:
            return {"type": "idle", "description": "Do nothing"}  # fallback

        decision_context = {
            "npc": npc_data,
            "perception": perception,
            "available_actions": available_actions or await self.get_default_actions(npc_data, perception)
        }

        scored_actions = await self.score_actions(decision_context)
        action = await self.select_action(scored_actions)
        await self.store_decision(action, decision_context)
        return action

    async def score_actions(self, decision_context: dict) -> List[dict]:
        npc = decision_context["npc"]
        perception = decision_context["perception"]
        actions = decision_context["available_actions"]

        scored_actions = []
        for action in actions:
            score = 0
            # 1. Personality alignment
            score += await self.score_personality_alignment(npc, action)
            # 2. Memory influence
            score += await self.score_memory_influence(perception.get("relevant_memories", []), action)
            # 3. Relationship influence
            score += await self.score_relationship_influence(perception.get("relationships", {}), action)
            # 4. Environmental context
            score += await self.score_environmental_context(perception.get("environment", {}), action)

            scored_actions.append({"action": action, "score": score})

        # Sort descending
        scored_actions.sort(key=lambda x: x["score"], reverse=True)
        return scored_actions

    async def score_personality_alignment(self, npc: dict, action: dict) -> float:
        """Score how well an action aligns with NPC's stats, traits, likes/dislikes."""
        score = 0.0
        stats_influenced = action.get("stats_influenced", {})

        # Stats alignment
        for stat, change in stats_influenced.items():
            if stat in npc:
                current_value = npc[stat]
                if change > 0 and current_value > 50:
                    score += 2
                elif change < 0 and current_value < 50:
                    score += 2

        # Personality traits
        personality_traits = npc.get("personality_traits", [])
        trait_alignments = {
            "commanding": {"command": 3, "test": 2, "direct": 2},
            "cruel": {"mock": 3},
            "kind": {"talk": 2, "praise": 3},
            "shy": {"observe": 3, "leave": 2},
            "confident": {"talk": 2, "command": 1},
            "manipulative": {"confide": 3, "praise": 2},
            "honest": {"confide": 2},
            "suspicious": {"observe": 2},
            "friendly": {"talk": 3, "talk_to": 2, "socialize": 3},
            "dominant": {"command": 3, "test": 3, "direct": 3},
            "submissive": {"observe": 2},
            "analytical": {"observe": 3},
            "impulsive": {"mock": 2, "leave": 1},
        }
        action_type = action["type"]
        for trait in personality_traits:
            trait_lower = trait.lower()
            if trait_lower in trait_alignments:
                action_bonus_map = trait_alignments[trait_lower]
                if action_type in action_bonus_map:
                    score += action_bonus_map[action_type]

        # Likes/dislikes
        if "target" in action and action["target"] not in ["environment", "location"]:
            likes = npc.get("likes", [])
            dislikes = npc.get("dislikes", [])
            target_name = action.get("target_name", "")
            # If the target name matches a like, prefer friendlier actions
            if any(like.lower() in target_name.lower() for like in likes):
                if action_type in ["talk", "talk_to", "praise", "confide"]:
                    score += 3
            if any(dl.lower() in target_name.lower() for dl in dislikes):
                if action_type in ["mock", "leave", "observe"]:
                    score += 3

        return score

    async def score_memory_influence(self, memories: list, action: dict) -> float:
        """Score how relevant or repeated a certain action is based on memories."""
        score = 0.0
        for memory in memories:
            mem_text = memory.get("memory_text", "").lower()
            if action["type"] == "talk" and ("talked" in mem_text or "conversation" in mem_text):
                score += 1
            elif action["type"] == "command" and ("commanded" in mem_text or "ordered" in mem_text):
                score += 1
            elif action["type"] == "mock" and ("mocked" in mem_text or "laughed at" in mem_text):
                score += 1

            if "target" in action:
                target = str(action["target"])
                target_name = action.get("target_name", "")
                if target in mem_text or target_name.lower() in mem_text:
                    score += 2

            # Emotional valence
            emotional_intensity = memory.get("emotional_intensity", 0)
            if emotional_intensity > 70:
                if action["type"] in ["talk", "praise", "confide", "talk_to"]:
                    score += 2
            elif emotional_intensity < 30:
                if action["type"] in ["mock", "leave", "observe"]:
                    score += 2

            # Simple recency
            ts = memory.get("timestamp")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    days_ago = (datetime.now() - dt).days
                    if days_ago < 1:
                        score += 1  # same day memory
                except Exception:
                    pass

        return score

    async def score_relationship_influence(self, relationships: dict, action: dict) -> float:
        score = 0.0
        if "target" in action and action["target"] not in ["environment", "location"]:
            target_rel = None
            if action["target"] == "player":
                target_rel = relationships.get("player", {})
            else:
                # If it's an NPC target, find a matching key
                for ent_type, rel_data in relationships.items():
                    if ent_type == "npc" and rel_data.get("entity_id") == action["target"]:
                        target_rel = rel_data
                        break

            if target_rel:
                rel_type = target_rel.get("link_type", "")
                rel_level = target_rel.get("link_level", 0)
                # Type
                if rel_type == "dominant" and action["type"] in ["command", "test", "direct"]:
                    score += 3
                elif rel_type == "friendly" and action["type"] in ["talk", "talk_to", "confide", "praise"]:
                    score += 3
                elif rel_type == "hostile" and action["type"] in ["mock", "leave", "observe"]:
                    score += 3
                elif rel_type == "romantic" and action["type"] in ["confide", "praise", "talk"]:
                    score += 4
                elif rel_type == "business" and action["type"] in ["talk", "command"]:
                    score += 2
                # Level
                if rel_level > 70:
                    if action["type"] in ["talk", "talk_to", "confide", "praise"]:
                        score += 2
                elif rel_level < 30:
                    if action["type"] in ["observe", "leave", "mock"]:
                        score += 2

        return score

    async def score_environmental_context(self, environment: dict, action: dict) -> float:
        score = 0.0
        location = environment.get("location", "").lower()
        time_of_day = environment.get("time_of_day", "").lower()

        # Example location-based
        if any(loc in location for loc in ["restaurant", "cafe", "dining", "bar"]):
            if action["type"] in ["talk", "socialize"]:
                score += 2
        if "bedroom" in location or "private" in location:
            if action["type"] in ["confide", "leave", "observe"]:
                score += 2
        if any(loc in location for loc in ["public", "crowded"]):
            if action["type"] in ["observe", "socialize"]:
                score += 2

        # Time-based
        if time_of_day in ["evening", "night"]:
            if action["type"] in ["confide", "leave"]:
                score += 1
        elif time_of_day in ["morning", "afternoon"]:
            if action["type"] in ["talk", "command", "socialize"]:
                score += 1

        return score

    async def select_action(self, scored_actions: List[dict], randomness: float = 0.2) -> dict:
        if not scored_actions:
            return {"type": "idle", "description": "Do nothing"}
        for sa in scored_actions:
            sa["score"] += random.uniform(0, randomness * 10)
        scored_actions.sort(key=lambda x: x["score"], reverse=True)
        return scored_actions[0]["action"]

    async def store_decision(self, action: dict, context: dict):
        """Store the final decision in NPCAgentState."""
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            row = await conn.fetchrow("""
                SELECT 1 FROM NPCAgentState
                WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
            """, self.npc_id, self.user_id, self.conversation_id)
            if row:
                await conn.execute("""
                    UPDATE NPCAgentState
                    SET last_decision=$1, last_updated=NOW()
                    WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
                """, json.dumps(action), self.npc_id, self.user_id, self.conversation_id)
            else:
                await conn.execute("""
                    INSERT INTO NPCAgentState (npc_id, user_id, conversation_id, last_decision, last_updated)
                    VALUES ($1, $2, $3, $4, NOW())
                """, self.npc_id, self.user_id, self.conversation_id, json.dumps(action))
        except Exception as e:
            logging.error(f"[EnhancedDecisionEngine] store_decision error: {e}")
        finally:
            if 'conn' in locals():
                await conn.close()

    async def get_npc_data(self) -> dict:
        """Fetch the NPC's stats and relevant JSON fields from the database."""
        try:
            conn = await asyncpg.connect(dsn=DB_DSN)
            row = await conn.fetchrow("""
                SELECT npc_name, dominance, cruelty, closeness, trust,
                       respect, intensity, hobbies, personality_traits,
                       likes, dislikes, schedule, current_location, sex
                FROM NPCStats
                WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
            """, self.npc_id, self.user_id, self.conversation_id)
            if not row:
                return {}

            return {
                "npc_id": self.npc_id,
                "npc_name": row["npc_name"],
                "dominance": row["dominance"],
                "cruelty": row["cruelty"],
                "closeness": row["closeness"],
                "trust": row["trust"],
                "respect": row["respect"],
                "intensity": row["intensity"],
                "hobbies": _parse_json_field(row["hobbies"]),
                "personality_traits": _parse_json_field(row["personality_traits"]),
                "likes": _parse_json_field(row["likes"]),
                "dislikes": _parse_json_field(row["dislikes"]),
                "schedule": _parse_json_field(row["schedule"]),
                "current_location": row["current_location"],
                "sex": row["sex"]
            }
        except Exception as e:
            logging.error(f"[EnhancedDecisionEngine] get_npc_data error: {e}")
            return {}
        finally:
            if 'conn' in locals():
                await conn.close()

    async def get_default_actions(self, npc_data: dict, perception: dict) -> List[dict]:
        """Generate a base set of actions for the NPC to consider."""
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
        if npc_data.get("cruelty", 0) > 60:
            actions.append({
                "type": "mock",
                "description": "Mock or belittle the player",
                "target": "player",
                "stats_influenced": {"cruelty": +1, "closeness": -2}
            })
        if npc_data.get("trust", 0) > 60:
            actions.append({
                "type": "confide",
                "description": "Share a personal secret",
                "target": "player",
                "stats_influenced": {"trust": +3, "closeness": +2}
            })
        if npc_data.get("respect", 0) > 60:
            actions.append({
                "type": "praise",
                "description": "Praise the player's abilities",
                "target": "player",
                "stats_influenced": {"respect": +2, "closeness": +1}
            })

        # Context-based expansions
        environment = perception.get("environment", {})
        loc_str = environment.get("location", "").lower()
        if any(loc in loc_str for loc in ["cafe", "restaurant", "bar", "party"]):
            actions.append({
                "type": "socialize",
                "description": "Engage in group conversation",
                "target": "group",
                "stats_influenced": {"closeness": +1}
            })
        # If other NPCs present
        for entity in environment.get("entities_present", []):
            if entity.get("type") == "npc":
                t_id = entity.get("id")
                t_name = entity.get("name", f"NPC_{t_id}")
                actions.append({
                    "type": "talk_to",
                    "description": f"Talk to {t_name}",
                    "target": t_id,
                    "target_name": t_name,
                    "stats_influenced": {"closeness": +1}
                })
                if npc_data.get("dominance", 0) > 70:
                    actions.append({
                        "type": "direct",
                        "description": f"Direct {t_name} to do something",
                        "target": t_id,
                        "target_name": t_name,
                        "stats_influenced": {"dominance": +1}
                    })

        return actions
