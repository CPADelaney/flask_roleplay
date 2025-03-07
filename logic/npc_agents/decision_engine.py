# logic/npc_agents/decision_engine.py (Updated)

import asyncio
import json
import logging
import random
from datetime import datetime
from typing import Dict, Any, List, Optional

from db.connection import get_db_connection
from memory.wrapper import MemorySystem
from memory.core import MemoryType, MemorySignificance

logger = logging.getLogger(__name__)

class NPCDecisionEngine:
    """
    Decision-making engine for NPCs with enhanced memory and emotional influences.
    Uses a trait-based system combined with memories, emotional state, 
    and mask management to make decisions.
    """

    def __init__(self, npc_id: int, user_id: int, conversation_id: int):
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._memory_system = None
    
    async def _get_memory_system(self):
        """Lazy-load the memory system."""
        if self._memory_system is None:
            self._memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        return self._memory_system

    async def decide(self, perception: Dict[str, Any], available_actions: Optional[List[dict]] = None) -> dict:
        """
        Evaluate the NPC's current state, context, personality, memories and emotional state
        to pick an action.
        
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
        mask = perception.get("mask", {})
        flashback = perception.get("flashback")
        
        # If there's a flashback, it can override normal decision making
        if flashback and random.random() < 0.7:  # 70% chance to be influenced by flashback
            # Generate an action based on the flashback
            flashback_action = await self._generate_flashback_action(flashback, npc_data)
            if flashback_action:
                return flashback_action
        
        # Get or generate available actions
        if available_actions is None:
            available_actions = await self.get_default_actions(npc_data, perception)
        
        # Score actions based on traits, memories, and emotional state
        scored_actions = await self.score_actions_with_memory(
            npc_data, 
            perception,
            available_actions, 
            emotional_state,
            mask
        )
        
        # Select an action based on scores (with some randomness)
        chosen_action = await self.select_action(scored_actions)
        
        # For femdom game context - enhance dominance-related actions
        chosen_action = await self._enhance_dominance_context(chosen_action, npc_data, mask)
        
        # Store the decision
        await self.store_decision(chosen_action, perception)
        
        # Possibly store a belief based on this decision
        await self._maybe_create_belief(perception, chosen_action, npc_data)
        
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
        Now enhanced with memory and emotional context.
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
        
        # Add dominance-based actions - important for femdom context
        dominance = npc_data.get("dominance", 0)
        mask = perception.get("mask", {})
        presented_traits = mask.get("presented_traits", {})
        hidden_traits = mask.get("hidden_traits", {})
        mask_integrity = mask.get("integrity", 100)
        
        # Add actions based on presented vs hidden traits
        
        # If NPC presents as submissive but is actually dominant (good for femdom surprises)
        submissive_presented = "submissive" in presented_traits or "gentle" in presented_traits
        dominant_hidden = "dominant" in hidden_traits or "controlling" in hidden_traits
        
        if submissive_presented and dominant_hidden and mask_integrity < 70:
            # As mask integrity deteriorates, start showing some dominant actions
            actions.append({
                "type": "assertive",
                "description": "Show an unexpected hint of assertiveness",
                "target": "player",
                "stats_influenced": {"dominance": +2, "respect": -1}
            })
        
        # Standard dominance-based actions
        if dominance > 60 or "dominant" in presented_traits:
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
            
            # More intense femdom themed actions for high dominance
            if dominance > 75:
                actions.append({
                    "type": "dominate",
                    "description": "Assert dominance forcefully",
                    "target": "player",
                    "stats_influenced": {"dominance": +3, "fear": +2}
                })
                actions.append({
                    "type": "punish",
                    "description": "Punish disobedience",
                    "target": "player", 
                    "stats_influenced": {"fear": +3, "obedience": +2}
                })
        
        # Add cruelty-based actions
        cruelty = npc_data.get("cruelty", 0)
        if cruelty > 60 or "cruel" in presented_traits:
            actions.append({
                "type": "mock",
                "description": "Mock or belittle the player",
                "target": "player",
                "stats_influenced": {"cruelty": +1, "closeness": -2}
            })
            
            # More intense femdom themed cruel actions
            if cruelty > 70:
                actions.append({
                    "type": "humiliate",
                    "description": "Deliberately humiliate the player",
                    "target": "player",
                    "stats_influenced": {"cruelty": +2, "fear": +2}
                })
        
        # Add trust-based actions
        trust = npc_data.get("trust", 0)
        if trust > 60:
            actions.append({
                "type": "confide",
                "description": "Share a personal secret",
                "target": "player",
                "stats_influenced": {"trust": +3, "closeness": +2}
            })
        
        # Add respect-based actions
        respect = npc_data.get("respect", 0)
        if respect > 60:
            actions.append({
                "type": "praise",
                "description": "Praise the player's submission",
                "target": "player",
                "stats_influenced": {"respect": +2, "closeness": +1}
            })

        # Get emotional state to influence available actions
        current_emotion = perception.get("emotional_state", {}).get("current_emotion", {})
        
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
                elif primary_emotion == "arousal" or primary_emotion == "desire":
                    # For femdom context
                    actions.append({
                        "type": "seduce",
                        "description": "Make seductive advances",
                        "target": "player",
                        "stats_influenced": {"closeness": +2, "fear": +1}
                    })
        
        # Context-based expansions from environment
        environment_data = perception.get("environment", {})
        loc_str = environment_data.get("location", "").lower()
        
        if any(loc in loc_str for loc in ["cafe", "restaurant", "bar", "party"]):
            actions.append({
                "type": "socialize",
                "description": "Engage in group conversation",
                "target": "group",
                "stats_influenced": {"closeness": +1}
            })
            
        # Check if there are other NPCs present
        for entity in environment_data.get("entities_present", []):
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
                
                if dominance > 70:
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
        
        # Look for patterns in memories that might suggest specific actions
        # For femdom context, look for patterns of player submission or resistance
        submission_pattern = any("submit" in m.get("text", "").lower() for m in memories)
        resistance_pattern = any("resist" in m.get("text", "").lower() for m in memories)
        
        if submission_pattern:
            actions.append({
                "type": "reward_submission",
                "description": "Reward the player's previous submission",
                "target": "player",
                "stats_influenced": {"closeness": +2, "respect": +1}
            })
        
        if resistance_pattern:
            actions.append({
                "type": "address_resistance",
                "description": "Address the player's previous resistance",
                "target": "player",
                "stats_influenced": {"dominance": +2, "fear": +1}
            })
        
        return actions

    async def score_actions_with_memory(self, 
                                       npc_data: Dict[str, Any], 
                                       perception: Dict[str, Any],
                                       actions: List[Dict[str, Any]],
                                       emotional_state: Dict[str, Any],
                                       mask: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Score actions with enhanced memory and emotion-based reasoning.
        
        Args:
            npc_data: NPC's stats and traits
            perception: NPC's perception including memories
            actions: Available actions to score
            emotional_state: NPC's current emotional state
            mask: NPC's mask information (presented vs hidden traits)
            
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
        
        # Get mask integrity
        mask_integrity = mask.get("integrity", 100)
        presented_traits = mask.get("presented_traits", {})
        hidden_traits = mask.get("hidden_traits", {})
        
        # Calculate how much the player knows about the NPC
        player_knowledge = 0.0
        
        # More memories means player knows more about the NPC
        if len(memories) > 5:
            player_knowledge += 0.3
        
        # Lower mask integrity means player knows more about true nature
        if mask_integrity < 50:
            player_knowledge += 0.3
        
        # Longer relationship (more closeness) means player knows more
        player_relationship = perception.get("relationships", {}).get("player", {})
        if player_relationship:
            link_level = player_relationship.get("link_level", 0)
            if link_level > 50:
                player_knowledge += 0.2
        
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
                action,
                hidden_traits,
                presented_traits
            )
            
            # 7. For femdom context - special scoring for dominance/submission
            score += await self._score_dominance_dynamics(
                action,
                npc_data,
                player_knowledge
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
                    "mask_influence": await self._score_mask_influence(mask_integrity, npc_data, action, hidden_traits, presented_traits),
                    "dominance_dynamics": await self._score_dominance_dynamics(action, npc_data, player_knowledge)
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
            "commanding": {"command": 3, "test": 2, "direct": 2, "control": 3, "dominate": 4, "punish": 3},
            "cruel": {"mock": 3, "express_anger": 2, "humiliate": 4},
            "kind": {"talk": 2, "praise": 3, "teach": 2, "reward_submission": 3},
            "shy": {"observe": 3, "leave": 2},
            "confident": {"talk": 2, "command": 1, "deepen_relationship": 2, "seduce": 3},
            "manipulative": {"confide": 3, "praise": 2, "control": 3, "test": 2},
            "honest": {"confide": 2},
            "suspicious": {"observe": 2, "act_defensive": 3},
            "friendly": {"talk": 3, "talk_to": 2, "socialize": 3, "celebrate": 3},
            "dominant": {"command": 3, "test": 3, "direct": 3, "control": 4, "dominate": 4, "punish": 3},
            "submissive": {"observe": 2},
            "analytical": {"observe": 3},
            "impulsive": {"mock": 2, "leave": 1, "express_anger": 3},
            "patient": {"teach": 3, "observe": 2},
            "protective": {"direct": 2, "command": 1, "act_defensive": 2},
            "sadistic": {"mock": 4, "test": 3, "express_anger": 3, "humiliate": 4, "punish": 4},
            "playful": {"celebrate": 3, "discuss_topic": 2, "socialize": 2, "seduce": 2}
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
        
        # If no memories, neutral influence
        if not memories:
            return score
            
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
            elif action["type"] == "dominate" and ("dominated" in memory_text or "controlled" in memory_text):
                score += 1
            elif action["type"] == "punish" and ("punished" in memory_text or "disciplined" in memory_text):
                score += 1
            elif action["type"] == "reward_submission" and ("submitted" in memory_text or "obeyed" in memory_text):
                score += 2
            elif action["type"] == "address_resistance" and ("resisted" in memory_text or "disobeyed" in memory_text):
                score += 2
            
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
            schema_interpretation = memory.get("schema_interpretation")
            if schema_interpretation:
                # Check if interpretation suggests certain action types based on theme patterns
                lower_interpretation = schema_interpretation.lower()
                
                # For femdom dynamics
                if "submissive" in lower_interpretation and action["type"] in ["command", "test", "dominate", "punish"]:
                    score += 3
                if "resistance" in lower_interpretation and action["type"] in ["punish", "address_resistance", "dominate"]:
                    score += 2
                if "willing" in lower_interpretation and action["type"] in ["reward_submission", "praise"]:
                    score += 2
                    
                # General patterns
                if "controlling" in lower_interpretation and action["type"] in ["command", "test", "direct", "control"]:
                    score += 3
                elif "caring" in lower_interpretation and action["type"] in ["talk", "praise", "teach"]:
                    score += 3
                elif "intimate" in lower_interpretation and action["type"] in ["confide", "deepen_relationship", "seduce"]:
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
                if rel_type == "dominant" and action["type"] in ["command", "test", "direct", "control", "dominate", "punish"]:
                    score += 3
                elif rel_type == "friendly" and action["type"] in ["talk", "talk_to", "confide", "praise"]:
                    score += 3
                elif rel_type == "hostile" and action["type"] in ["mock", "leave", "observe", "express_anger", "humiliate"]:
                    score += 3
                elif rel_type == "romantic" and action["type"] in ["confide", "praise", "talk", "deepen_relationship", "seduce"]:
                    score += 4
                elif rel_type == "business" and action["type"] in ["talk", "command"]:
                    score += 2
                
                # Score based on relationship level
                if rel_level > 70:
                    if action["type"] in ["talk", "talk_to", "confide", "praise", "deepen_relationship", "seduce"]:
                        score += 2
                    elif action["type"] in ["dominate", "punish"] and rel_type in ["dominant", "romantic"]:
                        score += 3  # Especially in femdom context
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
            
            # Less appropriate for intense femdom actions in public
            if action["type"] in ["punish", "humiliate", "dominate"]:
                score -= 3
        
        if "bedroom" in location or "private" in location or "dungeon" in location:
            if action["type"] in ["confide", "seduce"]:
                score += 3
                
            # More appropriate for femdom actions in private
            if action["type"] in ["dominate", "punish", "humiliate"]:
                score += 4
                
        if any(loc in loc_str for loc in ["public", "crowded"]):
            if action["type"] in ["observe", "socialize"]:
                score += 2
            elif action["type"] in ["confide", "express_anger", "punish", "humiliate"]:
                score -= 3
        
        # Time-based scoring
        if time_of_day in ["evening", "night"]:
            if action["type"] in ["confide", "leave", "seduce"]:
                score += 2
                
            # More appropriate for intense actions at night
            if action["type"] in ["dominate", "punish"]:
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
                "punish": 4, "humiliate": 3, "dominate": 3,  # Femdom context
                "praise": -3, "confide": -2, "socialize": -1
            },
            "fear": {
                "act_defensive": 4, "observe": 3, "leave": 2,
                "command": -2, "confide": -3, "socialize": -2, "dominate": -3
            },
            "joy": {
                "celebrate": 4, "talk": 3, "praise": 3, "socialize": 3, "confide": 2,
                "reward_submission": 3,  # Femdom context
                "mock": -3, "leave": -2, "act_defensive": -1, "punish": -2
            },
            "sadness": {
                "observe": 3, "leave": 2, "confide": 1,
                "celebrate": -3, "socialize": -2, "talk": -1, "dominate": -2
            },
            "disgust": {
                "mock": 3, "leave": 2, "act_defensive": 1, "humiliate": 3,
                "praise": -3, "confide": -2, "talk": -1
            },
            "arousal": {  # For femdom context
                "seduce": 4, "dominate": 3, "test": 3, "reward_submission": 2,
                "leave": -3, "observe": -2
            },
            "desire": {  # For femdom context
                "seduce": 4, "dominate": 3, "confide": 2,
                "leave": -3, "mock": -2
            },
            "surprise": {
                "observe": 2, "talk": 1, 
                "leave": -1
            },
            "trust": {
                "confide": 3, "praise": 2, "talk": 2, "deepen_relationship": 3,
                "reward_submission": 2,  # Femdom context
                "mock": -3, "act_defensive": -2
            },
            "anticipation": {
                "talk": 2, "observe": 1, "discuss_topic": 3, "test": 2
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

    async def _score_mask_influence(self, mask_integrity: float, npc_data: Dict[str, Any], 
                                 action: Dict[str, Any], hidden_traits: Dict[str, Any],
                                 presented_traits: Dict[str, Any]) -> float:
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
        
        # Score based on action alignment with presented vs hidden traits
        action_type = action["type"]
        
        # Check if action aligns with hidden traits that are starting to show
        for trait, trait_data in hidden_traits.items():
            trait_intensity = trait_data.get("intensity", 50)
            
            # Check alignment with hidden traits
            if trait == "dominant" and action_type in ["command", "test", "dominate", "punish"]:
                score += (trait_intensity / 100) * true_nature_factor * 5
            elif trait == "cruel" and action_type in ["mock", "humiliate", "punish"]:
                score += (trait_intensity / 100) * true_nature_factor * 5
            elif trait == "sadistic" and action_type in ["punish", "humiliate", "mock"]:
                score += (trait_intensity / 100) * true_nature_factor * 5
            elif trait == "controlling" and action_type in ["command", "test", "direct"]:
                score += (trait_intensity / 100) * true_nature_factor * 4
        
        # Check if action conflicts with presented traits (should score lower as mask breaks)
        for trait, trait_data in presented_traits.items():
            trait_confidence = trait_data.get("confidence", 50)
            
            # As mask breaks, these conflicts become more permissible
            if trait == "kind" and action_type in ["mock", "humiliate", "punish"]:
                score -= (trait_confidence / 100) * (1 - true_nature_factor) * 3
            elif trait == "gentle" and action_type in ["dominate", "express_anger", "punish"]:
                score -= (trait_confidence / 100) * (1 - true_nature_factor) * 3
            elif trait == "submissive" and action_type in ["command", "dominate", "direct"]:
                score -= (trait_confidence / 100) * (1 - true_nature_factor) * 4
        
        return score
    
    async def _score_dominance_dynamics(self, action: Dict[str, Any], 
                                     npc_data: Dict[str, Any],
                                     player_knowledge: float) -> float:
        """
        Special scoring function for femdom dynamics.
        Considers dominance/submission, player knowledge of true nature.
        """
        score = 0.0
        action_type = action["type"]
        dominance = npc_data.get("dominance", 50)
        cruelty = npc_data.get("cruelty", 50)
        
        # For highly dominant NPCs, femdom actions score higher
        if dominance > 70:
            if action_type in ["dominate", "punish", "command", "humiliate"]:
                score += (dominance - 70) / 5
                
                # If player doesn't know NPC well, more subtle dominance scores higher
                if player_knowledge < 0.5:
                    if action_type == "command":
                        score += 2
                    elif action_type == "test":
                        score += 1
                # If player knows NPC well, more direct dominance scores higher
                else:
                    if action_type == "dominate":
                        score += 2
                    elif action_type == "punish":
                        score += 2
        
        # For cruel NPCs, humiliation actions score higher
        if cruelty > 70:
            if action_type in ["humiliate", "mock"]:
                score += (cruelty - 70) / 5
                
                # More direct humiliation if player knowledge is high
                if player_knowledge > 0.6 and action_type == "humiliate":
                    score += 2
        
        # For NPCs with high dominance AND high cruelty, special dynamics
        if dominance > 70 and cruelty > 70:
            if action_type in ["punish", "humiliate", "dominate"]:
                score += 2
                
                # If player knowledge high, more extreme actions score higher
                if player_knowledge > 0.7:
                    score += 1
        
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
            if "mask_slippage" in action_copy:
                del action_copy["mask_slippage"]
            
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
    
    async def _generate_flashback_action(self, flashback: Dict[str, Any], npc_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate an action in response to a flashback.
        This can override normal decision-making.
        """
        if not flashback:
            return None
            
        # Get flashback text
        flashback_text = flashback.get("text", "")
        
        # Extract emotion from the flashback if possible
        emotion = "neutral"
        intensity = 0.5
        
        if "anger" in flashback_text.lower() or "furious" in flashback_text.lower():
            emotion = "anger"
            intensity = 0.7
        elif "scared" in flashback_text.lower() or "fear" in flashback_text.lower():
            emotion = "fear"
            intensity = 0.7
        elif "happy" in flashback_text.lower() or "joy" in flashback_text.lower():
            emotion = "joy"
            intensity = 0.6
        elif "submission" in flashback_text.lower() or "obedient" in flashback_text.lower():
            emotion = "trust"  # In femdom context, submission often links to trust
            intensity = 0.7
        elif "dominant" in flashback_text.lower() or "control" in flashback_text.lower():
            emotion = "anticipation"  # For dominance, anticipation is appropriate
            intensity = 0.8
            
        # Generate action based on the flashback emotion
        if emotion == "anger":
            return {
                "type": "express_anger",
                "description": "Express anger triggered by a flashback",
                "target": "player",
                "stats_influenced": {"dominance": +2, "fear": +1},
                "flashback_source": True
            }
        elif emotion == "fear":
            return {
                "type": "act_defensive",
                "description": "Act defensively due to a flashback",
                "target": "environment",
                "stats_influenced": {"trust": -1},
                "flashback_source": True
            }
        elif emotion == "joy":
            return {
                "type": "reminisce",
                "description": "Reminisce about a positive memory",
                "target": "player",
                "stats_influenced": {"closeness": +2},
                "flashback_source": True
            }
        elif emotion == "trust" and npc_data.get("dominance", 0) > 60:
            return {
                "type": "expect_submission",
                "description": "Expect submission based on past experiences",
                "target": "player",
                "stats_influenced": {"dominance": +2, "fear": +1},
                "flashback_source": True
            }
        elif emotion == "anticipation" and npc_data.get("dominance", 0) > 60:
            return {
                "type": "dominate",
                "description": "Assert dominance triggered by a flashback",
                "target": "player",
                "stats_influenced": {"dominance": +3, "fear": +2},
                "flashback_source": True
            }
            
        # Default action for other emotions or if conditions don't match
        return {
            "type": "reveal_flashback",
            "description": "Reveal being affected by a flashback",
            "target": "player",
            "stats_influenced": {"closeness": +1},
            "flashback_source": True
        }
        
    async def _maybe_create_belief(self, perception: Dict[str, Any], chosen_action: Dict[str, Any], 
                                npc_data: Dict[str, Any]) -> None:
        """
        Potentially create a belief based on this decision.
        Has a low chance of happening to avoid too many beliefs.
        """
        # Only create beliefs occasionally (5% chance)
        if random.random() > 0.05:
            return
            
        memory_system = await self._get_memory_system()
        
        # Extract relevant memory IDs that support belief formation
        memories = perception.get("relevant_memories", [])
        supporting_memory_ids = [m.get("id") for m in memories if m.get("id")]
        
        # Get potential belief topics from action and context
        potential_beliefs = []
        
        # 1. Beliefs about player submission/resistance
        if "resistance" in str(perception).lower():
            potential_beliefs.append({
                "text": "The player tends to resist my commands",
                "confidence": 0.7 if len(supporting_memory_ids) > 1 else 0.5
            })
        elif "submission" in str(perception).lower():
            potential_beliefs.append({
                "text": "The player is submissive to my authority",
                "confidence": 0.7 if len(supporting_memory_ids) > 1 else 0.5
            })
            
        # 2. Beliefs based on selected action
        if chosen_action["type"] in ["dominate", "punish", "command"] and npc_data.get("dominance", 0) > 70:
            potential_beliefs.append({
                "text": "I need to maintain strict control over the player",
                "confidence": 0.8
            })
            
        if chosen_action["type"] in ["reward_submission", "praise"] and "submission" in str(perception).lower():
            potential_beliefs.append({
                "text": "The player responds well to praise for their submission",
                "confidence": 0.7
            })
            
        # 3. Beliefs based on emotional state
        emotional_state = perception.get("emotional_state", {})
        current_emotion = emotional_state.get("current_emotion", {})
        primary_emotion = current_emotion.get("primary", {}).get("name", "neutral")
        
        if primary_emotion == "anger" and chosen_action["type"] in ["punish", "express_anger"]:
            potential_beliefs.append({
                "text": "When I show my anger, the player becomes more compliant",
                "confidence": 0.6
            })
            
        # Select one belief to create if we have any
        if potential_beliefs:
            belief = random.choice(potential_beliefs)
            
            try:
                await memory_system.create_belief(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    belief_text=belief["text"],
                    confidence=belief["confidence"]
                )
            except Exception as e:
                logger.error(f"Error creating belief: {e}")
