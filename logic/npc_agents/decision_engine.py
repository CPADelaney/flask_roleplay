# logic/npc_agents/decision_engine.py

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

    @classmethod
    async def create(cls, npc_id: int, user_id: int, conversation_id: int) -> "NPCDecisionEngine":
        """
        Recommended async factory method to ensure goals are initialized before usage.
        Example usage:
            engine = await NPCDecisionEngine.create(npc_id, user_id, conversation_id)
            chosen_action = await engine.decide(...)
        """
        self = cls(npc_id, user_id, conversation_id, initialize_goals=False)
        await self._initialize_goals()  # ensure we initialize before returning
        return self

    def __init__(self, npc_id: int, user_id: int, conversation_id: int, initialize_goals: bool = True):
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._memory_system: Optional[MemorySystem] = None
        self.long_term_goals: List[Dict[str, Any]] = []
        self._decision_log: List[Dict[str, Any]] = []

        # If someone constructs this without the factory, they can still get goals in the background
        if initialize_goals:
            # This runs in the background and might race with usage:
            asyncio.create_task(self._initialize_goals())

    async def _initialize_goals(self) -> None:
        """Initialize goals after getting NPC data."""
        try:
            npc_data = await self.get_npc_data()
            if npc_data:
                await self.initialize_long_term_goals(npc_data)
        except Exception as e:
            logger.error(f"Error initializing goals: {e}")

    async def _get_memory_system(self) -> MemorySystem:
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
        flashback = perception.get("flashback", None)

        year, month, day, time_of_day = None, None, None, None
        # Grab time context from DB
        try:
            def fetch_time_context():
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT key, value 
                        FROM CurrentRoleplay 
                        WHERE key IN ('CurrentYear', 'CurrentMonth', 'CurrentDay', 'TimeOfDay')
                          AND user_id = %s 
                          AND conversation_id = %s
                        """,
                        (self.user_id, self.conversation_id),
                    )
                    return cursor.fetchall()

            rows = await asyncio.to_thread(fetch_time_context)
            for key, value in rows:
                if key == "CurrentYear":
                    year = value
                elif key == "CurrentMonth":
                    month = value
                elif key == "CurrentDay":
                    day = value
                elif key == "TimeOfDay":
                    time_of_day = value
        except Exception as e:
            logger.error(f"Error getting time context: {e}")

        # Add time context to perception
        perception["time_context"] = {
            "year": year,
            "month": month,
            "day": day,
            "time_of_day": time_of_day
        }

        # Get narrative context
        try:
            def fetch_narrative_context():
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT key, value 
                        FROM CurrentRoleplay 
                        WHERE key IN ('CurrentPlotStage', 'CurrentTension')
                          AND user_id = %s 
                          AND conversation_id = %s
                        """,
                        (self.user_id, self.conversation_id),
                    )
                    return cursor.fetchall()

            rows = await asyncio.to_thread(fetch_narrative_context)
            narrative_context = {}
            for key, value in rows:
                narrative_context[key] = value

            perception["narrative_context"] = narrative_context
        except Exception as e:
            logger.error(f"Error getting narrative context: {e}")

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

        # Simulate action outcome to update goal progress
        action_outcome = {
            "result": "success" if random.random() > 0.2 else "failure",
            "emotional_impact": random.randint(-5, 5)
        }

        # Update goal progress based on the action and outcome
        await self._update_goal_progress(chosen_action, action_outcome)

        # Possibly store a belief based on this decision
        await self._maybe_create_belief(perception, chosen_action, npc_data)

        return chosen_action

    async def get_npc_data(self) -> Dict[str, Any]:
        """Get NPC stats and traits from the database."""
        def _fetch():
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT npc_name, dominance, cruelty, closeness, trust, respect, intensity,
                           hobbies, personality_traits, likes, dislikes, schedule, current_location, sex
                    FROM NPCStats
                    WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
                    """,
                    (self.npc_id, self.user_id, self.conversation_id),
                )
                return cursor.fetchone()

        row = await asyncio.to_thread(_fetch)
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

        # If NPC presents as submissive but is actually dominant
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

        # Cruelty-based actions
        cruelty = npc_data.get("cruelty", 0)
        if cruelty > 60 or "cruel" in presented_traits:
            actions.append({
                "type": "mock",
                "description": "Mock or belittle the player",
                "target": "player",
                "stats_influenced": {"cruelty": +1, "closeness": -2}
            })

            if cruelty > 70:
                actions.append({
                    "type": "humiliate",
                    "description": "Deliberately humiliate the player",
                    "target": "player",
                    "stats_influenced": {"cruelty": +2, "fear": +2}
                })

        # Trust-based actions
        trust = npc_data.get("trust", 0)
        if trust > 60:
            actions.append({
                "type": "confide",
                "description": "Share a personal secret",
                "target": "player",
                "stats_influenced": {"trust": +3, "closeness": +2}
            })

        # Respect-based actions
        respect = npc_data.get("respect", 0)
        if respect > 60:
            actions.append({
                "type": "praise",
                "description": "Praise the player's submission",
                "target": "player",
                "stats_influenced": {"respect": +2, "closeness": +1}
            })

        # Add emotion-specific actions
        current_emotion = perception.get("emotional_state", {}).get("current_emotion", {})
        if current_emotion:
            primary_emotion = current_emotion.get("primary", {}).get("name", "neutral")
            intensity = current_emotion.get("primary", {}).get("intensity", 0.0)

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
                elif primary_emotion in ["arousal", "desire"]:
                    actions.append({
                        "type": "seduce",
                        "description": "Make seductive advances",
                        "target": "player",
                        "stats_influenced": {"closeness": +2, "fear": +1}
                    })

        # Context-based expansions
        environment_data = perception.get("environment", {})
        loc_str = environment_data.get("location", "").lower()
        if any(loc in loc_str for loc in ["cafe", "restaurant", "bar", "party"]):
            actions.append({
                "type": "socialize",
                "description": "Engage in group conversation",
                "target": "group",
                "stats_influenced": {"closeness": +1}
            })

        # Check for other NPCs
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
        memories = perception.get("relevant_memories", [])

        memory_topics = set()
        for memory in memories:
            memory_text = memory.get("text", "")
            # Simple topic extraction
            for topic_indicator in ["about", "mentioned", "discussed", "talked about", "interested in"]:
                if topic_indicator in memory_text.lower():
                    parts = memory_text.lower().split(topic_indicator, 1)
                    if len(parts) > 1:
                        topic_part = parts[1].strip()
                        words = topic_part.split()
                        if words:
                            topic = " ".join(words[:3])
                            topic = topic.rstrip(".,:;!?")
                            if len(topic) > 3 and topic not in memory_topics:
                                memory_topics.add(topic)
                                actions.append({
                                    "type": "discuss_topic",
                                    "description": f"Discuss the topic of {topic}",
                                    "target": "player",
                                    "topic": topic,
                                    "stats_influenced": {"closeness": +1}
                                })

            # references to past interactions
            if "last time" in memory_text.lower() or "previously" in memory_text.lower():
                actions.append({
                    "type": "reference_past",
                    "description": "Reference a past interaction",
                    "target": "player",
                    "memory_id": memory.get("id"),
                    "stats_influenced": {"trust": +1}
                })

        # Look for patterns in memories that might suggest specific actions
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

    async def score_actions_with_memory(
        self,
        npc_data: Dict[str, Any],
        perception: Dict[str, Any],
        actions: List[Dict[str, Any]],
        emotional_state: Dict[str, Any],
        mask: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Score actions with enhanced memory and emotion-based reasoning.
        """
        scored_actions = []

        # Possibly apply memory biases
        memories = perception.get("relevant_memories", [])
        memories = await self._apply_memory_biases(memories)

        # Emotional data
        current_emotion = emotional_state.get("current_emotion", {})
        primary_emotion = current_emotion.get("primary", {}).get("name", "neutral")
        emotion_intensity = current_emotion.get("primary", {}).get("intensity", 0.5)

        # Mask data
        mask_integrity = mask.get("integrity", 100)
        presented_traits = mask.get("presented_traits", {})
        hidden_traits = mask.get("hidden_traits", {})

        # Calculate how much the player knows
        player_knowledge = self._calculate_player_knowledge(perception, mask_integrity)

        # Check for flashback or trauma triggers
        flashback = perception.get("flashback")
        traumatic_trigger = perception.get("traumatic_trigger")

        # Score each action
        for action in actions:
            score = 0.0
            scoring_factors = {}

            # 1. Personality alignment
            personality_score = await self._score_personality_alignment(
                npc_data, action, mask_integrity, hidden_traits, presented_traits
            )
            score += personality_score
            scoring_factors["personality_alignment"] = personality_score

            # 2. Memory influence
            memory_score = await self._score_memory_influence(memories, action)
            score += memory_score
            scoring_factors["memory_influence"] = memory_score

            # 3. Relationship influence
            relationship_score = await self._score_relationship_influence(
                perception.get("relationships", {}), 
                action
            )
            score += relationship_score
            scoring_factors["relationship_influence"] = relationship_score

            # 4. Environmental context
            environment_score = await self._score_environmental_context(
                perception.get("environment", {}), 
                action
            )
            score += environment_score
            scoring_factors["environmental_context"] = environment_score

            # 5. Emotional state influence
            emotional_score = await self._score_emotional_influence(
                primary_emotion,
                emotion_intensity,
                action
            )
            score += emotional_score
            scoring_factors["emotional_influence"] = emotional_score

            # 6. Mask influence
            mask_score = await self._score_mask_influence(
                mask_integrity,
                npc_data,
                action,
                hidden_traits,
                presented_traits
            )
            score += mask_score
            scoring_factors["mask_influence"] = mask_score

            # 7. Trauma influence
            trauma_score = 0.0
            if flashback or traumatic_trigger:
                trauma_score = await self._score_trauma_influence(action, flashback, traumatic_trigger)
                score += trauma_score
            scoring_factors["trauma_influence"] = trauma_score

            # 8. Belief influence
            belief_score = await self._score_belief_influence(
                perception.get("beliefs", []),
                action
            )
            score += belief_score
            scoring_factors["belief_influence"] = belief_score

            # 9. Decision history influence
            if hasattr(self, 'decision_history'):
                history_score = await self._score_decision_history(action)
                score += history_score
                scoring_factors["decision_history"] = history_score
            else:
                scoring_factors["decision_history"] = 0.0

            # 10. Player knowledge influence
            player_knowledge_score = await self._score_player_knowledge_influence(
                action,
                player_knowledge,
                hidden_traits
            )
            score += player_knowledge_score
            scoring_factors["player_knowledge"] = player_knowledge_score

            # 11. Time context
            time_context = perception.get("time_context", {})
            if time_context:
                time_context_score = await self._score_time_context_influence(time_context, action)
                score += time_context_score
                scoring_factors["time_context_influence"] = time_context_score

            scored_actions.append({
                "action": action,
                "score": score,
                "reasoning": scoring_factors
            })

        # Sort by score descending
        scored_actions.sort(key=lambda x: x["score"], reverse=True)

        # Log top 3 for debugging
        self._log_decision_reasoning(scored_actions[:3])

        return scored_actions

    async def _apply_memory_biases(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply recency, emotional, and personality bias to memories, 
        then sort them by relevance_score or similar metric.
        """
        if not memories:
            return memories

        # 1. Recency bias
        memories = await self.apply_recency_bias(memories)

        # 2. Emotional bias
        memories = await self.apply_emotional_bias(memories)

        # 3. Personality bias
        memories = await self.apply_personality_bias(memories)

        # Sort by adjusted "relevance_score"
        memories.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return memories

    async def apply_recency_bias(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Example stub: More recent memories get a small boost in relevance_score.
        Suppose each memory has 'timestamp' we can parse or compare.
        """
        # We'll just do a naive approach:
        for m in memories:
            # if memory has a 'timestamp' key, boost it if it's more recent
            # For demo, let's do random:
            current_relevance = m.get("relevance_score", 1.0)
            # random recency factor
            recency_factor = random.uniform(1.0, 1.2)  # e.g. slight boost
            m["relevance_score"] = current_relevance * recency_factor
        return memories

    async def apply_emotional_bias(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Example stub: If memory is highly emotional, boost relevance more.
        Suppose each memory has 'emotional_intensity' from 0 to 1.
        """
        for m in memories:
            current_relevance = m.get("relevance_score", 1.0)
            emotional_intensity = m.get("emotional_intensity", 0.0)  # 0-1 range
            # Add a factor: if emotional_intensity > 0.5, we bump it significantly
            boost = 1.0 + (emotional_intensity * 0.5)  # up to +0.5 multiplier
            m["relevance_score"] = current_relevance * boost
        return memories

    async def apply_personality_bias(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Example stub: If the NPC is 'dominant' or 'cruel', they recall certain triggers more strongly.
        We'll just do a minimal example.
        """
        # Suppose we have a dictionary of "trait" -> "keyword" -> "weight boost"
        # In real code, you'd read from NPCStats or pass it in.
        personality_keywords = {
            "dominant": ["obey", "command", "test"],
            "cruel": ["hurt", "humiliate", "punish"]
        }
        # For demo, just pick one trait
        # You might pass in npc_data or something else for your real logic

        for m in memories:
            current_relevance = m.get("relevance_score", 1.0)
            text_lower = m.get("text", "").lower()

            # naive example: if any of the keywords appear, we boost
            for trait, keywords in personality_keywords.items():
                for kw in keywords:
                    if kw in text_lower:
                        # small boost
                        m["relevance_score"] = current_relevance * 1.1
        return memories

    async def _score_personality_alignment(
        self,
        npc_data: Dict[str, Any],
        action: Dict[str, Any],
        mask_integrity: float,
        hidden_traits: Dict[str, Any],
        presented_traits: Dict[str, Any]
    ) -> float:
        """Score how well an action aligns with NPC's personality, accounting for mask."""
        score = 0.0
        action_type = action.get("type", "")

        true_nature_weight = (100 - mask_integrity) / 100  # 0-1 range
        presented_weight = mask_integrity / 100  # 0-1 range

        dominance = npc_data.get("dominance", 50)
        cruelty = npc_data.get("cruelty", 50)

        # Hidden traits alignment
        if "dominant" in hidden_traits and action_type in ["command", "dominate", "test", "punish"]:
            score += 3.0 * true_nature_weight
        if "cruel" in hidden_traits and action_type in ["mock", "humiliate", "punish"]:
            score += 3.0 * true_nature_weight
        if "sadistic" in hidden_traits and action_type in ["punish", "humiliate", "mock"]:
            score += 3.0 * true_nature_weight

        # Presented traits alignment
        if "kind" in presented_traits and action_type in ["praise", "support", "help"]:
            score += 2.0 * presented_weight
        if "gentle" in presented_traits and action_type in ["talk", "observe", "support"]:
            score += 2.0 * presented_weight

        # Base stats
        if dominance > 70 and action_type in ["command", "dominate", "test"]:
            score += 2.0
        if dominance < 30 and action_type in ["observe", "wait", "leave"]:
            score += 1.5
        if cruelty > 70 and action_type in ["mock", "humiliate", "punish"]:
            score += 2.0

        return score

    async def _score_memory_influence(self, memories: List[Dict[str, Any]], action: Dict[str, Any]) -> float:
        """Score how memories influence action preference with weighting."""
        score = 0.0
        if not memories:
            return score

        # Track references
        affected_by_memories = []
        action_type = action.get("type", "")

        for i, memory in enumerate(memories):
            memory_text = memory.get("text", "").lower()
            memory_id = memory.get("id")
            emotional_intensity = memory.get("emotional_intensity", 0) / 100.0
            relevance = memory.get("relevance_score", 1.0)

            # If action directly references memory
            if action.get("memory_id") == memory_id:
                memory_score = 5 * relevance
                score += memory_score
                affected_by_memories.append({"id": memory_id, "influence": memory_score})

            # Check content
            if action_type in memory_text:
                memory_score = 2 * relevance
                score += memory_score
                affected_by_memories.append({"id": memory_id, "influence": memory_score})

            if "target" in action:
                target_str = str(action["target"])
                target_name = action.get("target_name", "")
                if target_str in memory_text or target_name.lower() in memory_text:
                    memory_score = 3 * relevance
                    score += memory_score
                    affected_by_memories.append({"id": memory_id, "influence": memory_score})

            # Check memory emotion alignment
            memory_emotion = memory.get("primary_emotion", "").lower()
            emotion_aligned_actions = {
                "anger": ["express_anger", "mock", "attack", "challenge"],
                "fear": ["leave", "observe", "act_defensive"],
                "joy": ["celebrate", "praise", "talk", "confide"],
                "sadness": ["observe", "leave"],
                "disgust": ["mock", "leave", "observe"],
            }
            if memory_emotion in emotion_aligned_actions:
                if action_type in emotion_aligned_actions[memory_emotion]:
                    emotion_score = 2 * relevance * emotional_intensity
                    score += emotion_score
                    affected_by_memories.append({"id": memory_id, "influence_type": "emotion", "influence": emotion_score})

        if affected_by_memories:
            if "decision_metadata" not in action:
                action["decision_metadata"] = {}
            action["decision_metadata"]["memory_influences"] = affected_by_memories

        return score

    async def _score_relationship_influence(
        self,
        relationships: Dict[str, Any],
        action: Dict[str, Any]
    ) -> float:
        """
        Score how the NPC's known relationships (friendship, trust, rivalry, etc.)
        push them toward or away from the chosen action.
        """
        score = 0.0
        action_type = action.get("type", "")
        # Example: if the relationship with the player is high trust,
        # we might reduce the likelihood of "punish" or "mock".
        player_rel = relationships.get("player", {})
        link_level = player_rel.get("link_level", 0)  # 0-100 perhaps

        if action_type in ["punish", "mock", "humiliate"]:
            # penalize these actions if link_level is high
            if link_level > 60:
                score -= 3.0
        elif action_type in ["confide", "talk", "praise"]:
            # encourage these actions if link_level is high
            if link_level > 60:
                score += 2.0

        return score

    async def _score_environmental_context(
        self,
        environment: Dict[str, Any],
        action: Dict[str, Any]
    ) -> float:
        """
        Score how well the action fits the environment (e.g. library is quiet, bar is rowdy, etc.).
        """
        score = 0.0
        loc_str = environment.get("location", "").lower()
        action_type = action.get("type", "")

        # Example: if location is 'library' or 'church', reduce score for loud or violent actions
        if "library" in loc_str or "church" in loc_str:
            if action_type in ["express_anger", "mock", "shout", "celebrate", "dominate"]:
                score -= 2.0

        # Another example: if location is 'bar' or 'party', more social
        if "bar" in loc_str or "party" in loc_str:
            if action_type in ["socialize", "talk", "celebrate"]:
                score += 2.0

        return score

    async def _score_emotional_influence(self, emotion: str, intensity: float, action: Dict[str, Any]) -> float:
        """
        Score based on NPC's current emotional state with psychological realism.
        """
        score = 0.0
        if intensity < 0.4:
            return 0.0

        emotion_action_affinities = {
            "anger": {
                "express_anger": 4, "command": 2, "mock": 3, "test": 2, "leave": 1,
                "punish": 4, "humiliate": 3, "dominate": 3,
                "praise": -3, "confide": -2, "socialize": -1
            },
            "fear": {
                "act_defensive": 4, "observe": 3, "leave": 2,
                "command": -2, "confide": -3, "socialize": -2, "dominate": -3
            },
            "joy": {
                "celebrate": 4, "talk": 3, "praise": 3, "socialize": 3, "confide": 2,
                "reward_submission": 3,
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
            "trust": {
                "confide": 3, "praise": 2, "talk": 2,
                "reward_submission": 2,
                "mock": -3, "act_defensive": -2
            },
            # You can add more emotions if needed
        }

        action_type = action.get("type", "")
        action_affinities = emotion_action_affinities.get(emotion, {})
        if action_type in action_affinities:
            affinity = action_affinities[action_type]
            score += affinity * intensity

        if "decision_metadata" not in action:
            action["decision_metadata"] = {}
        action["decision_metadata"]["emotional_influence"] = {
            "emotion": emotion,
            "intensity": intensity,
            "affinity": action_affinities.get(action_type, 0),
            "score": score
        }

        return score

    async def _score_mask_influence(
        self,
        mask_integrity: float,
        npc_data: Dict[str, Any],
        action: Dict[str, Any],
        hidden_traits: Dict[str, Any],
        presented_traits: Dict[str, Any]
    ) -> float:
        """
        Score based on mask integrity - as mask deteriorates, true nature shows more.
        """
        score = 0.0
        if mask_integrity >= 95:
            return 0.0

        true_nature_factor = (100 - mask_integrity) / 100
        mask_influences = []
        action_type = action.get("type", "")

        # Hidden traits
        for trait, trait_data in hidden_traits.items():
            trait_intensity = trait_data.get("intensity", 50) / 100
            trait_weight = trait_intensity * true_nature_factor
            trait_score = 0

            if trait == "dominant" and action_type in ["command", "test", "dominate", "punish"]:
                trait_score = 5 * trait_weight
            elif trait == "cruel" and action_type in ["mock", "humiliate", "punish"]:
                trait_score = 5 * trait_weight
            elif trait == "sadistic" and action_type in ["punish", "humiliate", "mock"]:
                trait_score = 5 * trait_weight

            if trait_score != 0:
                score += trait_score
                mask_influences.append({"trait": trait, "type": "hidden", "score": trait_score})

        # Presented traits conflicts
        for trait, trait_data in presented_traits.items():
            trait_confidence = trait_data.get("confidence", 50) / 100
            mask_factor = 1.0 - true_nature_factor
            trait_score = 0

            if trait == "kind" and action_type in ["mock", "humiliate", "punish"]:
                trait_score = -3 * trait_confidence * mask_factor
            elif trait == "gentle" and action_type in ["dominate", "express_anger", "punish"]:
                trait_score = -3 * trait_confidence * mask_factor
            elif trait == "submissive" and action_type in ["command", "dominate", "direct"]:
                trait_score = -4 * trait_confidence * mask_factor
            elif trait == "honest" and action_type in ["deceive", "manipulate", "lie"]:
                trait_score = -4 * trait_confidence * mask_factor

            if trait_score != 0:
                score += trait_score
                mask_influences.append({"trait": trait, "type": "presented", "score": trait_score})

        if mask_influences:
            if "decision_metadata" not in action:
                action["decision_metadata"] = {}
            action["decision_metadata"]["mask_influence"] = {
                "integrity": mask_integrity,
                "true_nature_factor": true_nature_factor,
                "trait_influences": mask_influences
            }

        return score

    async def _score_trauma_influence(
        self,
        action: Dict[str, Any],
        flashback: Optional[Dict[str, Any]],
        traumatic_trigger: Optional[Dict[str, Any]]
    ) -> float:
        """
        Score actions based on flashbacks or traumatic triggers.
        """
        score = 0.0
        if not flashback and not traumatic_trigger:
            return score

        action_type = action.get("type", "")
        trauma_action_map = {
            "traumatic_response": 5.0,
            "act_defensive": 4.0,
            "leave": 3.5,
            "express_anger": 3.0,
            "observe": 2.5,
        }

        if action_type in trauma_action_map:
            base_score = trauma_action_map[action_type]
            if flashback and not traumatic_trigger:
                score += base_score * 0.7
            elif traumatic_trigger:
                score += base_score
                # Check the trigger's response_type
                response_type = traumatic_trigger.get("response_type")
                if response_type == "fight" and action_type in ["express_anger", "challenge"]:
                    score += 2.0
                elif response_type == "flight" and action_type == "leave":
                    score += 2.0
                elif response_type == "freeze" and action_type == "observe":
                    score += 2.0

        # Penalize vulnerability actions
        vulnerability_actions = ["confide", "praise", "talk", "socialize"]
        if action_type in vulnerability_actions:
            if traumatic_trigger:
                score -= 3.0
            elif flashback:
                score -= 2.0

        if flashback or traumatic_trigger:
            if "decision_metadata" not in action:
                action["decision_metadata"] = {}
            action["decision_metadata"]["trauma_influence"] = {
                "has_flashback": bool(flashback),
                "has_trigger": bool(traumatic_trigger),
                "score": score
            }

        return score

    async def _score_belief_influence(self, beliefs: List[Dict[str, Any]], action: Dict[str, Any]) -> float:
        """Score actions based on how well they align with NPC's beliefs."""
        score = 0.0
        if not beliefs:
            return score

        action_type = action.get("type", "")
        target = action.get("target", "")
        belief_influences = []

        for belief in beliefs:
            belief_text = belief.get("belief", "").lower()
            confidence = belief.get("confidence", 0.5)
            if confidence < 0.3:
                continue

            relevance = 0.0
            align_score = 0.0

            if (target == "player" or target == "group"):
                if any(word in belief_text for word in ["trust", "friend", "ally", "like"]):
                    if action_type in ["talk", "praise", "confide", "support"]:
                        relevance = 0.8
                        align_score = 3.0
                    elif action_type in ["mock", "challenge", "leave", "punish"]:
                        relevance = 0.7
                        align_score = -3.0
                elif any(word in belief_text for word in ["threat", "danger", "distrust", "wary"]):
                    if action_type in ["observe", "act_defensive", "leave"]:
                        relevance = 0.9
                        align_score = 4.0
                    elif action_type in ["confide", "praise", "support"]:
                        relevance = 0.8
                        align_score = -4.0
                elif any(word in belief_text for word in ["submit", "obey", "follow"]):
                    if action_type in ["command", "test", "dominate"]:
                        relevance = 0.9
                        align_score = 3.5
                    elif action_type in ["observe", "act_defensive"]:
                        relevance = 0.5
                        align_score = -2.0
                elif any(word in belief_text for word in ["rebel", "defy", "disobey"]):
                    if action_type in ["punish", "test", "command"]:
                        relevance = 0.8
                        align_score = 3.0
                    elif action_type in ["praise", "reward"]:
                        relevance = 0.6
                        align_score = -2.5

                if action_type in belief_text:
                    relevance = 0.9
                    align_score = 4.0

                if relevance > 0:
                    belief_score = align_score * confidence * relevance
                    score += belief_score
                    belief_influences.append({
                        "text": belief_text[:50] + "..." if len(belief_text) > 50 else belief_text,
                        "confidence": confidence,
                        "relevance": relevance,
                        "align_score": align_score,
                        "final_score": belief_score
                    })

        if belief_influences:
            if "decision_metadata" not in action:
                action["decision_metadata"] = {}
            action["decision_metadata"]["belief_influences"] = belief_influences

        return score

    async def _score_decision_history(self, action: Dict[str, Any]) -> float:
        """
        Score actions based on decision history for psychological continuity.
        This example requires self.decision_history to exist (which is not always set).
        """
        score = 0.0
        if not hasattr(self, 'decision_history') or not self.decision_history:
            return score

        action_type = action["type"]
        recent_actions = [d["action"]["type"] for d in self.decision_history[-3:]]
        action_counts = {}

        for i, a_type in enumerate(recent_actions):
            weight = 1.0 - (i * 0.2)
            action_counts[a_type] = action_counts.get(a_type, 0) + weight

        if action_type in action_counts:
            consistency_score = action_counts[action_type] * 1.5
            if len(recent_actions) >= 2 and recent_actions[0] == recent_actions[1] == action_type:
                consistency_score -= 3.0
            score += consistency_score

        if len(recent_actions) >= 2:
            if len(set(recent_actions)) == len(recent_actions) and action_type not in recent_actions:
                score -= 1.0

        if "decision_metadata" not in action:
            action["decision_metadata"] = {}
        action["decision_metadata"]["history_influence"] = {
            "recent_actions": recent_actions,
            "action_counts": action_counts,
            "score": score
        }
        return score

    async def _score_player_knowledge_influence(
        self,
        action: Dict[str, Any],
        player_knowledge: float,
        hidden_traits: Dict[str, Any]
    ) -> float:
        """
        Score actions based on how much the player knows about the NPC's true nature.
        """
        score = 0.0
        if player_knowledge < 0.3:
            return score

        action_type = action["type"]

        if player_knowledge > 0.7:
            if "dominant" in hidden_traits and action_type in ["command", "dominate", "punish"]:
                score += 2.0
            elif "cruel" in hidden_traits and action_type in ["mock", "humiliate"]:
                score += 2.0
            elif "submissive" in hidden_traits and action_type in ["observe", "act_defensive"]:
                score += 2.0
        elif player_knowledge > 0.4:
            if random.random() < 0.5:
                if "dominant" in hidden_traits and action_type in ["command", "direct"]:
                    score += 1.5
                elif "cruel" in hidden_traits and action_type == "mock":
                    score += 1.5
            else:
                if "dominant" in hidden_traits and action_type in ["talk", "observe"]:
                    score += 1.0

        if "decision_metadata" not in action:
            action["decision_metadata"] = {}
        action["decision_metadata"]["player_knowledge_influence"] = {
            "knowledge_level": player_knowledge,
            "score": score
        }
        return score

    def _calculate_player_knowledge(self, perception: Dict[str, Any], mask_integrity: float) -> float:
        """
        Calculate how much the player knows about the NPC's true nature.
        Based on relationship level, memory count, and mask integrity.
        """
        player_knowledge = 0.0
        if mask_integrity < 50:
            player_knowledge += 0.4
        elif mask_integrity < 75:
            player_knowledge += 0.2

        memories = perception.get("relevant_memories", [])
        if len(memories) > 7:
            player_knowledge += 0.3
        elif len(memories) > 3:
            player_knowledge += 0.15

        relationships = perception.get("relationships", {})
        player_rel = relationships.get("player", {})
        link_level = player_rel.get("link_level", 0)
        if link_level > 70:
            player_knowledge += 0.3
        elif link_level > 40:
            player_knowledge += 0.15

        return min(1.0, max(0.0, player_knowledge))

    def _log_decision_reasoning(self, top_actions: List[Dict[str, Any]]) -> None:
        """
        Record decision reasoning for introspection and debugging.
        Keeps the last 20 decisions in self._decision_log.
        """
        if not top_actions:
            return

        now = datetime.now()
        reasoning_entry = {
            "timestamp": now.isoformat(),
            "top_actions": [],
            "chosen_action": top_actions[0]["action"]["type"]
        }

        for i, action_data in enumerate(top_actions[:3]):
            action = action_data["action"]
            score = action_data["score"]
            reasoning = action_data.get("reasoning", {})
            action_entry = {
                "rank": i + 1,
                "type": action.get("type", "unknown"),
                "description": action.get("description", ""),
                "score": score,
                "reasoning_factors": reasoning
            }
            reasoning_entry["top_actions"].append(action_entry)

        self._decision_log.append(reasoning_entry)
        if len(self._decision_log) > 20:
            self._decision_log = self._decision_log[-20:]

        logger.debug(f"NPC {self.npc_id} decision reasoning recorded")

    async def select_action(self, scored_actions: List[Dict[str, Any]], randomness: float = 0.2) -> Dict[str, Any]:
        """
        Select an action from scored actions, with some randomness and pattern breaking.
        """
        if not scored_actions:
            return {"type": "idle", "description": "Do nothing"}

        # Check if we want to prioritize any action with a high "weight"
        weight_based_selection = any(
            ("weight" in sa["action"] and sa["action"]["weight"] > 2.0)
            for sa in scored_actions
        )
        if weight_based_selection:
            weighted_actions = [(sa["action"], sa["action"].get("weight", 1.0)) for sa in scored_actions]
            weighted_actions.sort(key=lambda x: x[1], reverse=True)
            selected_action = weighted_actions[0][0]
        else:
            # Normal score-based selection with randomness
            for sa in scored_actions:
                sa["score"] += random.uniform(0, randomness * 10)
            scored_actions.sort(key=lambda x: x["score"], reverse=True)
            selected_action = scored_actions[0]["action"]

        # Add decision metadata
        if len(scored_actions) > 1:
            if "decision_metadata" not in selected_action:
                selected_action["decision_metadata"] = {}
            selected_action["decision_metadata"]["alternative_actions"] = [
                {"type": sa["action"]["type"], "score": sa["score"]}
                for sa in scored_actions[1:3]
            ]
        if "reasoning" in scored_actions[0]:
            if "decision_metadata" not in selected_action:
                selected_action["decision_metadata"] = {}
            selected_action["decision_metadata"]["reasoning"] = scored_actions[0]["reasoning"]

        return selected_action

    async def store_decision(self, action: Dict[str, Any], context: Dict[str, Any]):
        """Store the final decision in NPCAgentState."""
        def _store():
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 1 FROM NPCAgentState
                    WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
                    """,
                    (self.npc_id, self.user_id, self.conversation_id),
                )
                exists = cursor.fetchone() is not None

                action_copy = action.copy()
                # Clean up any ephemeral keys you don't want to store
                for ephemeral_key in ["decision_factors", "mask_slippage"]:
                    if ephemeral_key in action_copy:
                        del action_copy[ephemeral_key]

                if exists:
                    cursor.execute(
                        """
                        UPDATE NPCAgentState
                        SET last_decision=%s, last_updated=NOW()
                        WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
                        """,
                        (json.dumps(action_copy), self.npc_id, self.user_id, self.conversation_id),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO NPCAgentState 
                        (npc_id, user_id, conversation_id, last_decision, last_updated)
                        VALUES (%s, %s, %s, %s, NOW())
                        """,
                        (self.npc_id, self.user_id, self.conversation_id, json.dumps(action_copy)),
                    )
                conn.commit()

        await asyncio.to_thread(_store)

    async def _generate_flashback_action(self, flashback: Dict[str, Any], npc_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate an action in response to a flashback."""
        if not flashback:
            return None
        flashback_text = flashback.get("text", "")
        emotion = "neutral"
        intensity = 0.5

        if any(word in flashback_text.lower() for word in ["anger", "furious"]):
            emotion = "anger"
            intensity = 0.7
        elif any(word in flashback_text.lower() for word in ["scared", "fear"]):
            emotion = "fear"
            intensity = 0.7
        elif any(word in flashback_text.lower() for word in ["happy", "joy"]):
            emotion = "joy"
            intensity = 0.6
        elif any(word in flashback_text.lower() for word in ["submission", "obedient"]):
            emotion = "trust"
            intensity = 0.7
        elif any(word in flashback_text.lower() for word in ["dominant", "control"]):
            emotion = "anticipation"
            intensity = 0.8

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

        # default
        return {
            "type": "reveal_flashback",
            "description": "Reveal being affected by a flashback",
            "target": "player",
            "stats_influenced": {"closeness": +1},
            "flashback_source": True
        }

    async def _maybe_create_belief(self, perception: Dict[str, Any], chosen_action: Dict[str, Any], npc_data: Dict[str, Any]) -> None:
        """Potentially create a belief based on this decision."""
        if random.random() > 0.05:
            return

        memory_system = await self._get_memory_system()
        memories = perception.get("relevant_memories", [])
        supporting_memory_ids = [m.get("id") for m in memories if m.get("id")]

        potential_beliefs = []
        # 1) Beliefs about player submission/resistance
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

        # 2) Based on chosen action
        if chosen_action.get("type") in ["dominate", "punish", "command"] and npc_data.get("dominance", 0) > 70:
            potential_beliefs.append({
                "text": "I need to maintain strict control over the player",
                "confidence": 0.8
            })
        if chosen_action.get("type") in ["reward_submission", "praise"] and "submission" in str(perception).lower():
            potential_beliefs.append({
                "text": "The player responds well to praise for their submission",
                "confidence": 0.7
            })

        # 3) Based on emotional state
        emotional_state = perception.get("emotional_state", {})
        current_emotion = emotional_state.get("current_emotion", {})
        primary_emotion = current_emotion.get("primary", {}).get("name", "neutral")
        if primary_emotion == "anger" and chosen_action.get("type") in ["punish", "express_anger"]:
            potential_beliefs.append({
                "text": "When I show my anger, the player becomes more compliant",
                "confidence": 0.6
            })

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

    async def initialize_long_term_goals(self, npc_data: Dict[str, Any]) -> None:
        """Initialize NPC's long-term goals based on personality and archetype."""
        self.long_term_goals = []

        dominance = npc_data.get("dominance", 50)
        cruelty = npc_data.get("cruelty", 50)
        traits = npc_data.get("personality_traits", [])
        archetypes = npc_data.get("archetypes", [])

        # Create goals based on dominance
        if dominance > 75:
            self.long_term_goals.append({
                "type": "dominance",
                "description": "Assert complete control over submissives",
                "importance": 0.9,
                "progress": 0,
                "target_entity": "player"
            })
        elif dominance > 60:
            self.long_term_goals.append({
                "type": "dominance",
                "description": "Establish authority in social hierarchy",
                "importance": 0.8,
                "progress": 0,
                "target_entity": None
            })
        elif dominance < 30:
            self.long_term_goals.append({
                "type": "submission",
                "description": "Find strong dominant to serve",
                "importance": 0.8,
                "progress": 0,
                "target_entity": None
            })

        # Create goals based on cruelty
        if cruelty > 70:
            self.long_term_goals.append({
                "type": "sadism",
                "description": "Break down resistances through humiliation",
                "importance": 0.85,
                "progress": 0,
                "target_entity": "player"
            })
        elif cruelty < 30 and dominance > 60:
            self.long_term_goals.append({
                "type": "guidance",
                "description": "Guide submissives to growth through gentle dominance",
                "importance": 0.75,
                "progress": 0,
                "target_entity": "player"
            })

        # Personality traits
        if "ambitious" in traits:
            self.long_term_goals.append({
                "type": "power",
                "description": "Increase social influence and control",
                "importance": 0.85,
                "progress": 0,
                "target_entity": None
            })
        if "protective" in traits:
            self.long_term_goals.append({
                "type": "protection",
                "description": "Ensure the safety and well-being of those in care",
                "importance": 0.8,
                "progress": 0,
                "target_entity": "player" if dominance > 50 else None
            })

        # Archetypes
        if "mentor" in archetypes:
            self.long_term_goals.append({
                "type": "development",
                "description": "Guide the development of the player's submissive nature",
                "importance": 0.9,
                "progress": 0,
                "target_entity": "player"
            })
        elif "seductress" in archetypes:
            self.long_term_goals.append({
                "type": "seduction",
                "description": "Gradually increase player's dependency and devotion",
                "importance": 0.9,
                "progress": 0,
                "target_entity": "player"
            })

    async def _update_goal_progress(self, action: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        """Update progress on long-term goals based on action outcomes."""
        if not self.long_term_goals:
            return

        success = "success" in outcome.get("result", "").lower()
        emotional_impact = outcome.get("emotional_impact", 0)

        for i, goal in enumerate(self.long_term_goals):
            goal_type = goal.get("type", "")
            target_entity = goal.get("target_entity")
            current_progress = goal.get("progress", 0)

            # Skip if mismatch target
            if (target_entity
                and action.get("target") != target_entity
                and not (target_entity == "player" and action.get("target") == "group")):
                continue

            progress_update = 0
            if goal_type == "dominance":
                if action["type"] in ["command", "dominate", "test"]:
                    if success:
                        progress_update = 5
                    else:
                        progress_update = -2
            elif goal_type == "submission":
                if action["type"] in ["observe", "obey", "assist"]:
                    progress_update = 3 if success else 1
            elif goal_type == "sadism":
                if action["type"] in ["punish", "humiliate", "mock"]:
                    progress_update = 2 if success else 0
            elif goal_type == "seduction":
                if action["type"] in ["seduce", "flirt"]:
                    if emotional_impact > 0:
                        progress_update = emotional_impact

            if progress_update != 0:
                self.long_term_goals[i]["progress"] = max(0, min(100, current_progress + progress_update))

    async def _score_time_context_influence(self, time_context: Dict[str, Any], action: Dict[str, Any]) -> float:
        """Score actions based on time-of-day appropriateness."""
        score = 0.0
        time_of_day = time_context.get("time_of_day", "").lower()
        action_type = action.get("type", "")

        if time_of_day == "morning":
            if action_type in ["talk", "observe", "socialize"]:
                score += 1.0
            elif action_type in ["sleep", "seduce", "dominate"]:
                score -= 1.0
        elif time_of_day == "afternoon":
            if action_type in ["talk", "socialize", "command", "test"]:
                score += 1.0
        elif time_of_day == "evening":
            if action_type in ["talk", "socialize", "seduce", "flirt"]:
                score += 1.5
        elif time_of_day == "night":
            if action_type in ["seduce", "dominate", "sleep"]:
                score += 2.0
            elif action_type in ["talk", "socialize"]:
                score -= 0.5
        return score

    async def _enhance_dominance_context(self, action: Dict[str, Any], npc_data: Dict[str, Any], mask: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optionally add extra flavor or modifications for actions that are dominance-related.
        For example, intensify the description or stats if the NPC is strongly dominant.
        """
        dominance = npc_data.get("dominance", 0)
        if dominance > 80 and action.get("type") in ["command", "dominate", "punish"]:
            # Add a little more flavor
            action["description"] = "With overwhelming confidence, " + action.get("description", "")
            stats_influenced = action.get("stats_influenced", {})
            stats_influenced["dominance"] = stats_influenced.get("dominance", 0) + 1
            action["stats_influenced"] = stats_influenced
        return action
