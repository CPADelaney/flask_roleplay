# logic/npc_agents/decision_engine.py

import asyncio
import json
import logging
import random
from datetime import datetime
from typing import Dict, Any, List, Optional

from db.connection import get_db_connection

logger = logging.getLogger(__name__)

class NPCDecisionEngine:
    """
    Decision-making engine for NPCs with enhanced memory and emotional influences.
    Uses a trait-based system combined with memories and emotional state to make decisions.
    """

    def __init__(self, npc_id: int, user_id: int, conversation_id: int):
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def decide(self, perception: Dict[str, Any], available_actions: Optional[List[dict]] = None) -> dict:
        """
        Evaluate the NPC's current state, context, and personality to pick an action.
        Now incorporates memories, emotions, and beliefs in decision-making.
        
        Args:
            perception: NPC's perception of the environment, including memories
            available_actions: Actions the NPC could take, or None to generate options
            
        Returns:
            The chosen action
        """
        # Get NPC data
        npc_data = await self.get_npc_data()
        if not npc_data:
            return {"type": "idle", "description": "Do nothing"}  # fallback

        # Extract relevant perception elements
        memories = perception.get("relevant_memories", [])
        relationships = perception.get("relationships", {})
        environment = perception.get("environment", {})
        emotional_state = perception.get("emotional_state", {})
        
        # Get or generate available actions
        if available_actions is None:
            available_actions = await self.get_default_actions(npc_data, perception)
        
        # Score actions based on traits, memories, emotional state
        scored_actions = await self.score_actions_with_memory(
            npc_data, 
            perception,
            available_actions, 
            emotional_state
        )
        
        # Select an action based on scores (with some randomness)
        chosen_action = await self.select_action(scored_actions)
        
        # Store the decision
        await self.store_decision(chosen_action, perception)
        
        return chosen_action

    async def get_npc_data(self) -> Dict[str, Any]:
        """Get NPC stats and traits from the database."""
        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
                SELECT npc_name, dominance, cruelty, closeness, trust, respect, intensity,
                       hobbies, personality_traits, likes, dislikes, schedule, current_location, sex
                FROM NPCStats
                WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
            """, (self.npc_id, self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            if not row:
                return {}
                
            # Parse JSON fields
            hobbies = self._parse_json_field(row[7])
            personality_traits = self._parse_json_field(row[8])
            likes = self._parse_json_field(row[9])
            dislikes = self._parse_json_field(row[10])
            schedule = self._parse_json_field(row[11])
            
            return {
                "npc_id": self.npc_id,
                "npc_name": row[0],
                "dominance": row[1],
                "cruelty": row[2],
                "closeness": row[3],
                "trust": row[4],
                "respect": row[5],
                "intensity": row[6],
                "hobbies": hobbies,
                "personality_traits": personality_traits,
                "likes": likes,
                "dislikes": dislikes,
                "schedule": schedule,
                "current_location": row[12],
                "sex": row[13]
            }

    def _parse_json_field(self, field) -> List[Any]:
        """Helper to parse JSON fields from DB rows."""
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

    async def get_default_actions(self, npc_data: Dict[str, Any], perception: Dict[str, Any]) -> List[dict]:
        """
        Generate a base set of actions for the NPC to consider.
        Now incorporates emotional context and memories.
        """
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

        # Get emotional state to influence available actions
        emotional_state = perception.get("emotional_state", {})
        current_emotion = emotional_state.get("current_emotion", {})
        
        if current_emotion:
            primary_emotion = current_emotion.get("primary", {}).get("name", "neutral")
            intensity = current_emotion.get("primary", {}).get("intensity", 0.0)
            
            # Add emotion-specific actions for strong emotions
            if intensity > 0.7:
                if primary_emotion == "anger":
                    actions.append({
                        "type": "express_anger",
                        "description": "Express anger forcefully",
                        "target": "player",
                        "stats_influenced": {"dominance": +2, "closeness": -3}
                    })
                elif primary_emotion == "fear":
                    actions.append({
                        "type": "act_defensive",
                        "description": "Act defensively and guarded",
                        "target": "environment",
                        "stats_influenced": {"trust": -2}
                    })
                elif primary_emotion == "joy":
                    actions.append({
                        "type": "celebrate",
                        "description": "Share happiness enthusiastically",
                        "target": "player",
                        "stats_influenced": {"closeness": +3}
                    })
        
        # Context-based expansions from environment
        environment = perception.get("environment", {})
        loc_str = environment.get("location", "").lower()
        
        if any(loc in loc_str for loc in ["cafe", "restaurant", "bar", "party"]):
            actions.append({
                "type": "socialize",
                "description": "Engage in group conversation",
                "target": "group",
                "stats_influenced": {"closeness": +1}
            })
            
        # Check if there are other NPCs present
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
        
        # Add memory-based actions
        memory_based_actions = await self._generate_memory_based_actions(perception)
        actions.extend(memory_based_actions)
        
        return actions

    async def _generate_memory_based_actions(self, perception: Dict[str, Any]) -> List[dict]:
        """Generate actions based on relevant memories."""
        actions = []
        
        # Get relevant memories
        memories = perception.get("relevant_memories", [])
        
        # Track what topics we've seen in memories
        memory_topics = set()
        
        for memory in memories:
            memory_text = memory.get("text", "")
            
            # Look for potential topics in the memory
            for topic_indicator in ["about", "mentioned", "discussed", "talked about", "interested in"]:
                if topic_indicator in memory_text.lower():
                    # Extract potential topic
                    parts = memory_text.lower().split(topic_indicator, 1)
                    if len(parts) > 1:
                        topic_part = parts[1].strip()
                        words = topic_part.split()
                        if words:
                            # Take first few words as topic
                            topic = " ".join(words[:min(3, len(words))])
                            
                            # Clean up topic (remove punctuation at end)
                            topic = topic.rstrip(".,:;!?")
                            
                            if len(topic) > 3 and topic not in memory_topics:
                                memory_topics.add(topic)
                                
                                # Add action to discuss this topic
                                actions.append({
                                    "type": "discuss_topic",
                                    "description": f"Discuss the topic of {topic}",
                                    "target": "player",
                                    "topic": topic,
                                    "stats_influenced": {"closeness": +1}
                                })
            
            # Check for references to past interactions
            if "last time" in memory_text.lower() or "previously" in memory_text.lower():
                actions.append({
                    "type": "reference_past",
                    "description": "Reference a past interaction",
                    "target": "player",
                    "memory_id": memory.get("id"),
                    "stats_influenced": {"trust": +1}
                })
        
        # Add actions based on schema interpretations
        for memory in memories:
            interpretation = memory.get("schema_interpretation")
            if interpretation:
                # Extract key concepts from interpretation
                interpretation = interpretation.lower()
                
                # Look for relationship patterns
                if "relationship" in interpretation:
                    actions.append({
                        "type": "deepen_relationship",
                        "description": "Build on established relationship pattern",
                        "target": "player",
                        "based_on": memory.get("id"),
                        "stats_influenced": {"closeness": +2, "trust": +1}
                    })
                
                # Look for teaching/mentoring patterns
                if "teach" in interpretation or "mentor" in interpretation or "guide" in interpretation:
                    actions.append({
                        "type": "teach",
                        "description": "Offer guidance or mentorship",
                        "target": "player",
                        "based_on": memory.get("id"),
                        "stats_influenced": {"respect": +2}
                    })
                
                # Look for control/dominance patterns
                if "control" in interpretation or "influence" in interpretation or "manipulate" in interpretation:
                    actions.append({
                        "type": "control",
                        "description": "Exert control over the situation",
                        "target": "player",
                        "based_on": memory.get("id"),
                        "stats_influenced": {"dominance": +2}
                    })
        
        return actions

    async def score_actions_with_memory(self, 
                                       npc_data: Dict[str, Any], 
                                       perception: Dict[str, Any],
                                       actions: List[Dict[str, Any]],
                                       emotional_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Score actions with enhanced memory and emotion-based reasoning.
        
        Args:
            npc_data: NPC's stats and traits
            perception: NPC's perception including memories
            actions: Available actions to score
            emotional_state: NPC's current emotional state
            
        Returns:
            List of actions with scores
        """
        scored_actions = []
        
        # Get current emotion data
        current_emotion = emotional_state.get("current_emotion", {})
        primary_emotion = current_emotion.get("primary", {}).get("name", "neutral")
        emotion_intensity = current_emotion.get("primary", {}).get("intensity", 0.5)
        
        # Get relevant memories
        memories = perception.get("relevant_memories", [])
        
        # Get mask integrity if available
        mask_integrity = 100
        if "mask" in perception:
            mask_integrity = perception["mask"].get("integrity", 100)
        
        for action in actions:
            # Start with base score of 0
            score = 0.0
            
            # 1. Personality alignment
            score += await self._score_personality_alignment(npc_data, action)
            
            # 2. Memory influence
            score += await self._score_memory_influence(memories, action)
            
            # 3. Relationship influence
            score += await self._score_relationship_influence(
                perception.get("relationships", {}), 
                action
            )
            
            # 4. Environmental context
            score += await self._score_environmental_context(
                perception.get("environment", {}), 
                action
            )
            
            # 5. Emotional state influence
            score += await self._score_emotional_influence(
                primary_emotion,
                emotion_intensity,
                action
            )
            
            # 6. Mask influence (true nature shows more as mask deteriorates)
            score += await self._score_mask_influence(
                mask_integrity,
                npc_data,
                action
            )
            
            scored_actions.append({
                "action": action, 
                "score": score,
                "reasoning": {
                    "personality_alignment": await self._score_personality_alignment(npc_data, action),
                    "memory_influence": await self._score_memory_influence(memories, action),
                    "relationship_influence": await self._score_relationship_influence(perception.get("relationships", {}), action),
                    "environmental_context": await self._score_environmental_context(perception.get("environment", {}), action),
                    "emotional_influence": await self._score_emotional_influence(primary_emotion, emotion_intensity, action),
                    "mask_influence": await self._score_mask_influence(mask_integrity, npc_data, action)
                }
            })
        
        # Sort by score (descending)
        scored_actions.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_actions

    async def _score_personality_alignment(self, npc: Dict[str, Any], action: Dict[str, Any]) -> float:
        """Score how well an action aligns with NPC's stats and traits."""
        score = 0.0
        
        # Score based on stat alignment
        stats_influenced = action.get("stats_influenced", {})
        for stat, change in stats_influenced.items():
            if stat in npc:
                current_value = npc[stat]
                if change > 0 and current_value > 50:
                    score += 2
                elif change < 0 and current_value < 50:
                    score += 2
        
        # Score based on personality traits
        personality_traits = npc.get("personality_traits", [])
        trait_alignments = {
            "commanding": {"command": 3, "test": 2, "direct": 2, "control": 3},
            "cruel": {"mock": 3, "express_anger": 2},
            "kind": {"talk": 2, "praise": 3, "teach": 2},
            "shy": {"observe": 3, "leave": 2},
            "confident": {"talk": 2, "command": 1, "deepen_relationship": 2},
            "manipulative": {"confide": 3, "praise": 2, "control": 3},
            "honest": {"confide": 2},
            "suspicious": {"observe": 2, "act_defensive": 3},
            "friendly": {"talk": 3, "talk_to": 2, "socialize": 3, "celebrate": 3},
            "dominant": {"command": 3, "test": 3, "direct": 3, "control": 4},
            "submissive": {"observe": 2},
            "analytical": {"observe": 3},
            "impulsive": {"mock": 2, "leave": 1, "express_anger": 3},
            "patient": {"teach": 3, "observe": 2},
            "protective": {"direct": 2, "command": 1, "act_defensive": 2},
            "sadistic": {"mock": 4, "test": 3, "express_anger": 3},
            "playful": {"celebrate": 3, "discuss_topic": 2, "socialize": 2}
        }
        
        action_type = action["type"]
        for trait in personality_traits:
            trait_lower = trait.lower()
            if trait_lower in trait_alignments:
                action_bonus_map = trait_alignments[trait_lower]
                if action_type in action_bonus_map:
                    score += action_bonus_map[action_type]
        
        # Score based on likes/dislikes
        if "target" in action and action["target"] not in ["environment", "location"]:
            likes = npc.get("likes", [])
            dislikes = npc.get("dislikes", [])
            target_name = action.get("target_name", "")
            
            # Check if target is liked
            if any(like.lower() in target_name.lower() for like in likes):
                if action_type in ["talk", "talk_to", "praise", "confide"]:
                    score += 3
            
            # Check if target is disliked
            if any(dl.lower() in target_name.lower() for dl in dislikes):
                if action_type in ["mock", "leave", "observe"]:
                    score += 3
            
            # Check for topic-based likes/dislikes
            if "topic" in action:
                topic = action["topic"].lower()
                if any(like.lower() in topic for like in likes):
                    score += 4
                if any(dl.lower() in topic for dl in dislikes):
                    score -= 4
        
        return score

    async def _score_memory_influence(self, memories: List[Dict[str, Any]], action: Dict[str, Any]) -> float:
        """Score how memories influence action preference."""
        score = 0.0
        
        for memory in memories:
            memory_text = memory.get("text", "").lower()
            memory_id = memory.get("id")
            
            # Direct reference to memory
            if action.get("memory_id") == memory_id:
                score += 5
            
            # Check for action type in memory
            if action["type"] == "talk" and ("talked" in memory_text or "conversation" in memory_text):
                score += 1
            elif action["type"] == "command" and ("commanded" in memory_text or "ordered" in memory_text):
                score += 1
            elif action["type"] == "mock" and ("mocked" in memory_text or "laughed at" in memory_text):
                score += 1
            
            # Check if action target appears in memory
            if "target" in action:
                target = str(action["target"])
                target_name = action.get("target_name", "")
                if target in memory_text or target_name.lower() in memory_text:
                    score += 2
            
            # Check for topic matches
            if "topic" in action and action["topic"].lower() in memory_text:
                score += 3
            
            # Emotional valence influence
            emotional_intensity = memory.get("emotional_intensity", 0)
            if emotional_intensity > 70:
                if action["type"] in ["talk", "praise", "confide", "talk_to"]:
                    score += 2
            elif emotional_intensity < 30:
                if action["type"] in ["mock", "leave", "observe"]:
                    score += 2
            
            # Schema interpretation influence
            interpretation = memory.get("schema_interpretation")
            if interpretation:
                # Check if interpretation suggests this action type
                if "controlling" in interpretation and action["type"] in ["command", "test", "direct", "control"]:
                    score += 3
                elif "caring" in interpretation and action["type"] in ["talk", "praise", "teach"]:
                    score += 3
                elif "intimate" in interpretation and action["type"] in ["confide", "deepen_relationship"]:
                    score += 3
        
        return score

    async def _score_relationship_influence(self, relationships: Dict[str, Any], action: Dict[str, Any]) -> float:
        """Score based on NPC's relationships."""
        score = 0.0
        
        if "target" in action and action["target"] not in ["environment", "location"]:
            target_rel = None
            
            # Check for player relationship
            if action["target"] == "player":
                target_rel = relationships.get("player", {})
            else:
                # Check for NPC relationship
                for ent_type, rel_data in relationships.items():
                    if ent_type == "npc" and rel_data.get("entity_id") == action["target"]:
                        target_rel = rel_data
                        break
            
            if target_rel:
                rel_type = target_rel.get("link_type", "")
                rel_level = target_rel.get("link_level", 0)
                
                # Score based on relationship type
                if rel_type == "dominant" and action["type"] in ["command", "test", "direct", "control"]:
                    score += 3
                elif rel_type == "friendly" and action["type"] in ["talk", "talk_to", "confide", "praise"]:
                    score += 3
                elif rel_type == "hostile" and action["type"] in ["mock", "leave", "observe", "express_anger"]:
                    score += 3
                elif rel_type == "romantic" and action["type"] in ["confide", "praise", "talk", "deepen_relationship"]:
                    score += 4
                elif rel_type == "business" and action["type"] in ["talk", "command"]:
                    score += 2
                
                # Score based on relationship level
                if rel_level > 70:
                    if action["type"] in ["talk", "talk_to", "confide", "praise", "deepen_relationship"]:
                        score += 2
                elif rel_level < 30:
                    if action["type"] in ["observe", "leave", "mock"]:
                        score += 2
        
        return score

    async def _score_environmental_context(self, environment: Dict[str, Any], action: Dict[str, Any]) -> float:
        """Score based on environmental context."""
        score = 0.0
        
        location = environment.get("location", "").lower()
        time_of_day = environment.get("time_of_day", "").lower()
        
        # Location-based scoring
        if any(loc in location for loc in ["restaurant", "cafe", "dining", "bar"]):
            if action["type"] in ["talk", "socialize"]:
                score += 2
        
        if "bedroom" in location or "private" in location:
            if action["type"] in ["confide", "leave", "observe"]:
                score += 2
        
        if any(loc in location for loc in ["public", "crowded"]):
            if action["type"] in ["observe", "socialize"]:
                score += 2
            elif action["type"] in ["confide", "express_anger"]:
                score -= 2
        
        # Time-based scoring
        if time_of_day in ["evening", "night"]:
            if action["type"] in ["confide", "leave"]:
                score += 1
        elif time_of_day in ["morning", "afternoon"]:
            if action["type"] in ["talk", "command", "socialize"]:
                score += 1
        
        return score

    async def _score_emotional_influence(self, emotion: str, intensity: float, action: Dict[str, Any]) -> float:
        """Score based on NPC's current emotional state."""
        score = 0.0
        
        # No significant emotional influence if intensity is low
        if intensity < 0.4:
            return 0.0
        
        # Different emotions favor different action types
        emotion_action_affinities = {
            "anger": {
                "express_anger": 4, "command": 2, "mock": 3, "test": 2, "leave": 1,
                "praise": -3, "confide": -2, "socialize": -1
            },
            "fear": {
                "act_defensive": 4, "observe": 3, "leave": 2,
                "command": -2, "confide": -3, "socialize": -2
            },
            "joy": {
                "celebrate": 4, "talk": 3, "praise": 3, "socialize": 3, "confide": 2,
                "mock": -3, "leave": -2, "act_defensive": -1
            },
            "sadness": {
                "observe": 3, "leave": 2, "confide": 1,
                "celebrate": -3, "socialize": -2, "talk": -1
            },
            "disgust": {
                "mock": 3, "leave": 2, "act_defensive": 1,
                "praise": -3, "confide": -2, "talk": -1
            },
            "surprise": {
                "observe": 2, "talk": 1, 
                "leave": -1
            },
            "trust": {
                "confide": 3, "praise": 2, "talk": 2, "deepen_relationship": 3,
                "mock": -3, "act_defensive": -2
            },
            "anticipation": {
                "talk": 2, "observe": 1, "discuss_topic": 3
            }
        }
        
        # Get action type affinities for this emotion
        action_affinities = emotion_action_affinities.get(emotion, {})
        
        # Apply affinity score scaled by emotional intensity
        action_type = action["type"]
        if action_type in action_affinities:
            affinity = action_affinities[action_type]
            score += affinity * intensity
        
        return score

    async def _score_mask_influence(self, mask_integrity: float, npc_data: Dict[str, Any], action: Dict[str, Any]) -> float:
        """
        Score based on mask integrity - as mask deteriorates, true nature shows more.
        This creates a tension between presented and hidden traits.
        """
        score = 0.0
        
        # No mask influence if mask is fully intact
        if mask_integrity >= 95:
            return 0.0
        
        # Calculate how much true nature is showing through
        true_nature_factor = (100 - mask_integrity) / 100
        
        # Actions aligned with dominance
        dominance = npc_data.get("dominance", 50)
        if dominance > 60:
            dom_actions = ["command", "test", "direct", "control", "express_anger"]
            if action["type"] in dom_actions:
                score += (dominance - 60) / 40 * true_nature_factor * 5
        
        # Actions aligned with cruelty
        cruelty = npc_data.get("cruelty", 50)
        if cruelty > 60:
            cruel_actions = ["mock", "test", "express_anger"]
            if action["type"] in cruel_actions:
                score += (cruelty - 60) / 40 * true_nature_factor * 5
        
        return score

    async def select_action(self, scored_actions: List[Dict[str, Any]], randomness: float = 0.2) -> Dict[str, Any]:
        """
        Select an action from scored actions, with some randomness.
        
        Args:
            scored_actions: List of actions with scores
            randomness: How much randomness to apply (0.0-1.0)
            
        Returns:
            The selected action
        """
        if not scored_actions:
            return {"type": "idle", "description": "Do nothing"}
        
        # Add randomness to scores
        for sa in scored_actions:
            sa["score"] += random.uniform(0, randomness * 10)
        
        # Re-sort with randomness applied
        scored_actions.sort(key=lambda x: x["score"], reverse=True)
        
        # Select top action
        selected_action = scored_actions[0]["action"]
        
        # Add reasoning metadata to the action
        selected_action["decision_factors"] = scored_actions[0]["reasoning"]
        
        return selected_action

    async def store_decision(self, action: Dict[str, Any], context: Dict[str, Any]):
        """Store the final decision in NPCAgentState."""
        with get_db_connection() as conn, conn.cursor() as cursor:
            # Check if state record exists
            cursor.execute("""
                SELECT 1 FROM NPCAgentState
                WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
            """, (self.npc_id, self.user_id, self.conversation_id))
            
            exists = cursor.fetchone() is not None
            
            # Remove decision_factors from stored action to keep it clean
            action_copy = action.copy()
            if "decision_factors" in action_copy:
                del action_copy["decision_factors"]
            
            if exists:
                # Update existing record
                cursor.execute("""
                    UPDATE NPCAgentState
                    SET last_decision=%s, last_updated=NOW()
                    WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
                """, (json.dumps(action_copy), self.npc_id, self.user_id, self.conversation_id))
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO NPCAgentState 
                    (npc_id, user_id, conversation_id, last_decision, last_updated)
                    VALUES (%s, %s, %s, %s, NOW())
                """, (self.npc_id, self.user_id, self.conversation_id, json.dumps(action_copy)))
            
            conn.commit()
