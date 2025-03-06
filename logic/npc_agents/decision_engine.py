# logic/decision_engine.py - Improved decision engine

import random
import json
import logging
import os
from typing import List, Dict, Any, Optional
import asyncpg

class EnhancedDecisionEngine:
    """Sophisticated decision-making engine for NPCs"""

    def __init__(self, npc_id, user_id, conversation_id):
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def decide(self, perception, available_actions=None):
        """
        Evaluate the NPC's current state, context, and personality to pick an action.
        Now with more sophisticated reasoning and influence factors.
        """
        # Get NPC's data
        npc_data = await self.get_npc_data()
        if not npc_data:
            # fallback
            return {"type": "idle", "description": "Do nothing"}
        
        # Prepare decision context
        decision_context = {
            "npc": npc_data,
            "perception": perception,
            "available_actions": available_actions or await self.get_default_actions(npc_data, perception)
        }
        
        # Score each available action
        scored_actions = await self.score_actions(decision_context)
        
        # Select the highest-scoring action with some randomness
        action = await self.select_action(scored_actions)
        
        # Store this decision
        await self.store_decision(action, decision_context)
        
        return action

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
            # 4. Environmental context
            score += await self.score_environmental_context(
                perception.get("environment", {}),
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
        
        # Score based on stats alignment
        for stat, change in stats_influenced.items():
            if stat in npc:
                current_value = npc[stat]
                # If change direction matches stat tendency, add points
                if change > 0 and current_value > 50:
                    score += 2
                elif change < 0 and current_value < 50:
                    score += 2

        # Score based on personality traits
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
        
        for trait in personality_traits:
            trait_lower = trait.lower()
            if trait_lower in trait_alignments:
                for action_type, bonus in trait_alignments[trait_lower].items():
                    if action["type"] == action_type:
                        score += bonus
                        
        # Score based on likes/dislikes
        if "target" in action and action["target"] != "environment" and action["target"] != "location":
            likes = npc.get("likes", [])
            dislikes = npc.get("dislikes", [])
            
            # Check if target is liked/disliked
            target_name = action.get("target_name", "")
            if any(like.lower() in target_name.lower() for like in likes):
                if action["type"] in ["talk", "talk_to", "praise", "confide"]:
                    score += 3
            if any(dislike.lower() in target_name.lower() for dislike in dislikes):
                if action["type"] in ["mock", "leave", "observe"]:
                    score += 3
                    
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
            if "target" in action and action["target"] != "environment" and action["target"] != "location":
                target = str(action["target"])
                if target in memory_text:
                    score += 2
                    
                # Target name scoring
                target_name = action.get("target_name", "")
                if target_name and target_name.lower() in memory_text:
                    score += 2

            # Emotional valence scoring
            emotional_intensity = memory.get("emotional_intensity", 0)
            if emotional_intensity > 70:
                # Strong positive emotion
                if action["type"] in ["talk", "praise", "confide", "talk_to"]:
                    score += 2
            elif emotional_intensity < 30:
                # Strong negative emotion
                if action["type"] in ["mock", "leave", "observe"]:
                    score += 2
                    
            # Recency bonus
            if memory.get("timestamp"):
                try:
                    from datetime import datetime
                    timestamp = memory["timestamp"]
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    days_ago = (datetime.now() - timestamp).days
                    if days_ago < 1:  # Same day
                        score += 1
                except Exception:
                    pass
                
        return score

    async def score_relationship_influence(self, relationships, action):
        """Score how relationships influence action choice"""
        score = 0
        if "target" in action and action["target"] != "environment" and action["target"] != "location":
            # Get target relationship
            target_rel = None
            target = action["target"]
            
            # For player targets
            if action["target"] == "player":
                target_rel = relationships.get("player", {})
            # For NPC targets, need to find matching entity
            else:
                for entity_type, rel_data in relationships.items():
                    if entity_type == "npc" and rel_data.get("entity_id") == target:
                        target_rel = rel_data
                        break
            
            if target_rel:
                # Relationship type influences
                rel_type = target_rel.get("link_type", "")
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

                # Relationship level influences
                rel_level = target_rel.get("link_level", 0)
                if rel_level > 70:
                    if action["type"] in ["talk", "talk_to", "confide", "praise"]:
                        score += 2
                elif rel_level < 30:
                    if action["type"] in ["observe", "leave", "mock"]:
                        score += 2
        
        return score
        
    async def score_environmental_context(self, environment, action):
        """Score based on the current environment/location"""
        score = 0
        location = environment.get("location", "").lower()
        time_of_day = environment.get("time_of_day", "").lower()
        
        # Location-based scoring
        if "restaurant" in location or "cafe" in location or "dining" in location:
            if action["type"] in ["talk", "socialize"]:
                score += 2
                
        if "bedroom" in location or "private" in location:
            if action["type"] in ["confide", "leave", "observe"]:
                score += 2
                
        if "public" in location or "crowded" in location:
            if action["type"] in ["observe", "socialize"]:
                score += 2
                
        # Time-based scoring
        if time_of_day in ["evening", "night"]:
            if action["type"] in ["confide", "leave"]:
                score += 1
        elif time_of_day in ["morning", "afternoon"]:
            if action["type"] in ["talk", "command", "socialize"]:
                score += 1
                
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
        dsn = os.getenv("DB_DSN")
        conn = await asyncpg.connect(dsn=dsn)
        try:
            # Check if record exists
            row = await conn.fetchrow("""
                SELECT 1 FROM NPCAgentState
                WHERE npc_id = $1
                  AND user_id = $2
                  AND conversation_id = $3
            """, self.npc_id, self.user_id, self.conversation_id)
            
            if row:
                # Update existing record
                await conn.execute("""
                    UPDATE NPCAgentState
                    SET last_decision = $1,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE npc_id = $2
                      AND user_id = $3
                      AND conversation_id = $4
                """, json.dumps(action), self.npc_id, self.user_id, self.conversation_id)
            else:
                # Insert new record
                await conn.execute("""
                    INSERT INTO NPCAgentState
                        (npc_id, user_id, conversation_id, last_decision, last_updated)
                    VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                """, self.npc_id, self.user_id, self.conversation_id, json.dumps(action))
                
        except Exception as e:
            logging.error(f"Error storing decision: {e}")
        finally:
            await conn.close()

    async def get_npc_data(self):
        """Get NPC's current data from the database"""
        dsn = os.getenv("DB_DSN")
        conn = await asyncpg.connect(dsn=dsn)
        try:
            row = await conn.fetchrow("""
                SELECT npc_name, dominance, cruelty, closeness, trust,
                       respect, intensity, hobbies, personality_traits,
                       likes, dislikes, schedule, current_location, sex
                FROM NPCStats
                WHERE npc_id = $1
                  AND user_id = $2
                  AND conversation_id = $3
            """, self.npc_id, self.user_id, self.conversation_id)
            
            if not row:
                return {}

            # For JSON columns, handle both string and object cases
            def parse_json_field(field):
                if field is None:
                    return []
                if isinstance(field, str):
                    try:
                        return json.loads(field)
                    except json.JSONDecodeError:
                        return []
                return field

            npc_data = {
                "npc_id": self.npc_id,
                "npc_name": row["npc_name"],
                "dominance": row["dominance"],
                "cruelty": row["cruelty"],
                "closeness": row["closeness"],
                "trust": row["trust"],
                "respect": row["respect"],
                "intensity": row["intensity"],
                "hobbies": parse_json_field(row["hobbies"]),
                "personality_traits": parse_json_field(row["personality_traits"]),
                "likes": parse_json_field(row["likes"]),
                "dislikes": parse_json_field(row["dislikes"]),
                "schedule": parse_json_field(row["schedule"]),
                "current_location": row["current_location"],
                "sex": row["sex"]
            }
            return npc_data
        except Exception as e:
            logging.error(f"Error getting NPC data: {e}")
            return {}
        finally:
            await conn.close()

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
            
        # Add context-specific actions based on perception
        environment = perception.get("environment", {})
        location = environment.get("location", "")
        entities_present = environment.get("entities_present", [])
        
        # If in a social location, add social actions
        if any(loc in location.lower() for loc in ["cafe", "restaurant", "bar", "party", "gathering"]):
            actions.append({
                "type": "socialize",
                "description": "Engage in group conversation",
                "target": "group",
                "stats_influenced": {"closeness": +1}
            })
            
        # If other NPCs present, add NPC-targeted actions
        npc_targets = [e for e in entities_present if e.get("type") == "npc"]
        for target in npc_targets:
            target_id = target.get("id")
            target_name = target.get("name", f"NPC-{target_id}")
            
            actions.append({
                "type": "talk_to",
                "description": f"Talk to {target_name}",
                "target": target_id,
                "target_name": target_name,
                "stats_influenced": {"closeness": +1}
            })
            
            # Add dominance-based interactions with other NPCs
            if npc_data.get("dominance", 0) > 70:
                actions.append({
                    "type": "direct",
                    "description": f"Direct {target_name} to do something",
                    "target": target_id,
                    "target_name": target_name,
                    "stats_influenced": {"dominance": +1}
                })

        return actions
