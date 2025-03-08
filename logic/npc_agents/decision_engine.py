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
        self.long_term_goals = []  # Initialize empty list
        
        # Initialize long-term goals asynchronously
        asyncio.create_task(self._initialize_goals())
        
    async def _initialize_goals(self):
        """Initialize goals after getting NPC data."""
        try:
            npc_data = await self.get_npc_data()
            if npc_data:
                await self.initialize_long_term_goals(npc_data)
        except Exception as e:
            logging.error(f"Error initializing goals: {e}")
        
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

        year, month, day, time_of_day = None, None, None, None
        try:
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute("""
                    SELECT value FROM CurrentRoleplay 
                    WHERE key IN ('CurrentYear', 'CurrentMonth', 'CurrentDay', 'TimeOfDay')
                    AND user_id = %s AND conversation_id = %s
                """, (self.user_id, self.conversation_id))
                
                for row in cursor.fetchall():
                    key, value = row
                    if key == "CurrentYear": year = value
                    elif key == "CurrentMonth": month = value
                    elif key == "CurrentDay": day = value
                    elif key == "TimeOfDay": time_of_day = value
        except Exception as e:
            logging.error(f"Error getting time context: {e}")
        
        # Add time context to perception
        perception["time_context"] = {
            "year": year,
            "month": month,
            "day": day,
            "time_of_day": time_of_day
        }
        
        # Get narrative context
        try:
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute("""
                    SELECT value FROM CurrentRoleplay 
                    WHERE key IN ('CurrentPlotStage', 'CurrentTension')
                    AND user_id = %s AND conversation_id = %s
                """, (self.user_id, self.conversation_id))
                
                narrative_context = {}
                for row in cursor.fetchall():
                    key, value = row
                    narrative_context[key] = value
                    
                perception["narrative_context"] = narrative_context
        except Exception as e:
            logging.error(f"Error getting narrative context: {e}")
        
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

    # Add a reusable memory context method in NPCDecisionEngine
    async def _get_enhanced_memory_context(self, base_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create an enhanced context with memory-related information."""
        memory_system = await self._get_memory_system()
        
        # Get emotional state
        emotional_state = await memory_system.get_npc_emotion(self.npc_id)
        
        # Get beliefs related to the context
        beliefs = await memory_system.get_beliefs(
            entity_type="npc",
            entity_id=self.npc_id,
            topic=base_context.get("topic", "general")
        )
        
        # Get flashback potential
        flashback = None
        if random.random() < 0.1:  # 10% chance
            flashback = await memory_system.npc_flashback(
                npc_id=self.npc_id,
                context=str(base_context)
            )
        
        # Create enhanced context
        enhanced_context = base_context.copy()
        enhanced_context.update({
            "emotional_state": emotional_state,
            "beliefs": beliefs,
            "flashback": flashback,
            "memory_enhanced": True
        })
        
        return enhanced_context

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

    # Add to decision_engine.py
    
    async def _apply_memory_biases(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply various psychological biases to memories to create more realistic recall.
        Includes recency bias, emotional bias, and personality-specific memory biases.
        
        Args:
            memories: List of memories to process
            
        Returns:
            List of memories with bias-adjusted relevance scores
        """
        if not memories:
            return []
        
        # Apply recency bias - more recent memories have higher relevance
        memories = await self.apply_recency_bias(memories)
        
        # Apply emotional bias - emotionally charged memories have higher relevance
        memories = await self.apply_emotional_bias(memories)
        
        # Apply personality-based bias - varies based on NPC personality traits
        memories = await self.apply_personality_bias(memories)
        
        # Sort by adjusted relevance score
        memories.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return memories
    
    def _record_decision_reasoning(self, top_actions: List[Dict[str, Any]]) -> None:
        """
        Record decision reasoning for introspection, debugging, and memory formation.
        Tracks why certain actions were preferred over others.
        
        Args:
            top_actions: List of top-ranked actions with scores and reasoning
        """
        if not hasattr(self, '_decision_log'):
            self._decision_log = []
        
        now = datetime.now()
        
        # Create reasoning entry
        reasoning_entry = {
            "timestamp": now.isoformat(),
            "top_actions": []
        }
        
        # Log detailed reasoning for top actions
        for i, action_data in enumerate(top_actions[:3]):  # Log top 3 actions
            action = action_data.get("action", {})
            score = action_data.get("score", 0)
            reasoning = action_data.get("reasoning", {})
            
            action_entry = {
                "rank": i + 1,
                "type": action.get("type", "unknown"),
                "description": action.get("description", ""),
                "score": score,
                "reasoning_factors": reasoning
            }
            
            reasoning_entry["top_actions"].append(action_entry)
        
        # Add chosen action details
        if top_actions:
            chosen_action = top_actions[0].get("action", {})
            reasoning_entry["chosen_action"] = {
                "type": chosen_action.get("type", "unknown"),
                "description": chosen_action.get("description", ""),
            }
        
        # Add to decision log, keeping maximum history
        self._decision_log.append(reasoning_entry)
        if len(self._decision_log) > 20:  # Keep last 20 decisions
            self._decision_log = self._decision_log[-20:]
        
        # Log for debugging
        logger.debug(f"NPC {self.npc_id} decision reasoning recorded")
    
    # Fix incomplete string in traumatic response
    async def _create_trauma_response_action(self, trauma_trigger: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a psychologically realistic action in response to a traumatic trigger.
        Features differentiated responses based on trauma type and emotional response.
        
        Args:
            trauma_trigger: Information about the traumatic trigger
            
        Returns:
            Trauma response action or None
        """
        try:
            # Get emotional response from trigger
            emotional_response = trauma_trigger.get("emotional_response", {})
            primary_emotion = emotional_response.get("primary_emotion", "fear")
            intensity = emotional_response.get("intensity", 0.5)
            trigger_text = trauma_trigger.get("trigger_text", "")
            
            # Different responses based on trauma type and primary emotion
            if primary_emotion == "fear":
                # Fear typically triggers fight-flight-freeze responses
                # Determine which based on NPC personality
                
                # Default to freeze response (most common)
                response_type = "freeze"
                
                # Check for fight vs flight personality factors
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT dominance, cruelty 
                        FROM NPCStats
                        WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                    """, (self.npc_id, self.user_id, self.conversation_id))
                    
                    row = cursor.fetchone()
                    if row:
                        dominance, cruelty = row
                        
                        # High dominance/cruelty NPCs tend to fight
                        if dominance > 70 or cruelty > 70:
                            response_type = "fight"
                        # Low dominance NPCs tend to flee
                        elif dominance < 30:
                            response_type = "flight"
                
                # Create appropriate response based on type
                if response_type == "fight":
                    return {
                        "type": "traumatic_response",
                        "description": "react aggressively to a triggering memory",
                        "target": "group",
                        "weight": 2.0 * intensity,
                        "stats_influenced": {"trust": -10, "fear": +5},
                        "trauma_trigger": trigger_text
                    }
                elif response_type == "flight":
                    return {
                        "type": "traumatic_response",
                        "description": "try to escape from a triggering situation",
                        "target": "location",
                        "weight": 2.0 * intensity,
                        "stats_influenced": {"trust": -5},
                        "trauma_trigger": trigger_text
                    }
                else:  # freeze
                    return {
                        "type": "traumatic_response",
                        "description": "freeze in response to a triggering memory",
                        "target": "self",
                        "weight": 2.0 * intensity,
                        "stats_influenced": {"trust": -5},
                        "trauma_trigger": trigger_text
                    }
                    
            elif primary_emotion == "anger":
                # Anger typically leads to confrontational responses
                return {
                    "type": "traumatic_response",
                    "description": "respond with anger to a triggering situation",
                    "target": "group",
                    "weight": 1.8 * intensity,
                    "stats_influenced": {"trust": -5, "respect": -5},
                    "trauma_trigger": trigger_text
                }
            elif primary_emotion == "sadness":
                # Sadness typically leads to withdrawal
                return {
                    "type": "traumatic_response",
                    "description": "become visibly downcast due to a painful memory",
                    "target": "self",
                    "weight": 1.7 * intensity,
                    "stats_influenced": {"closeness": +2},  # Can create sympathy
                    "trauma_trigger": trigger_text
                }
            else:
                # Generic response for other emotions
                return {
                    "type": "traumatic_response",
                    "description": f"respond emotionally to a trigger related to {trigger_text}",
                    "target": "self",
                    "weight": 1.5 * intensity,
                    "stats_influenced": {},
                    "trauma_trigger": trigger_text
                }
                
        except Exception as e:
            logger.error(f"Error creating trauma response: {e}")
            return None
    
    async def initialize_long_term_goals(self, npc_data: Dict[str, Any]) -> None:
        """Initialize NPC's long-term goals based on personality and archetype."""
        self.long_term_goals = []
        
        # Extract personality factors
        dominance = npc_data.get("dominance", 50)
        cruelty = npc_data.get("cruelty", 50)
        traits = npc_data.get("personality_traits", [])
        archetypes = npc_data.get("archetypes", [])
        
        # Default importance levels
        default_importance = 0.7
        
        # Create goals based on dominance level (femdom-specific)
        if dominance > 75:
            self.long_term_goals.append({
                "type": "dominance",
                "description": "Assert complete control over submissives",
                "importance": 0.9,
                "progress": 0,
                "target_entity": "player"  # Usually focused on player
            })
        elif dominance > 60:
            self.long_term_goals.append({
                "type": "dominance",
                "description": "Establish authority in social hierarchy",
                "importance": 0.8,
                "progress": 0,
                "target_entity": None  # General goal
            })
        elif dominance < 30:
            self.long_term_goals.append({
                "type": "submission",
                "description": "Find strong dominant to serve",
                "importance": 0.8,
                "progress": 0,
                "target_entity": None
            })
        
        # Create goals based on cruelty (femdom-specific)
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
        
        # Create goals based on personality traits
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
    
        # Create archetype-specific goals
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

    # Add this method to NPCDecisionEngine class
    async def _score_personality_alignment(self, npc_data: Dict[str, Any], action: Dict[str, Any], 
                                         mask_integrity: float, hidden_traits: Dict[str, Any], 
                                         presented_traits: Dict[str, Any]) -> float:
        """Score how well an action aligns with NPC's personality, accounting for mask."""
        score = 0.0
        action_type = action.get("type", "")
        
        # Calculate how much true vs presented personality influences
        true_nature_weight = (100 - mask_integrity) / 100  # 0-1 range
        presented_weight = mask_integrity / 100  # 0-1 range
        
        # Extract basic stats
        dominance = npc_data.get("dominance", 50)
        cruelty = npc_data.get("cruelty", 50)
        
        # Score alignment with true/hidden nature
        if "dominant" in hidden_traits and action_type in ["command", "dominate", "test", "punish"]:
            score += 3.0 * true_nature_weight
        if "cruel" in hidden_traits and action_type in ["mock", "humiliate", "punish"]:
            score += 3.0 * true_nature_weight
        if "sadistic" in hidden_traits and action_type in ["punish", "humiliate"]:
            score += 3.0 * true_nature_weight
        
        # Score alignment with presented/masked nature
        if "kind" in presented_traits and action_type in ["praise", "support", "help"]:
            score += 2.0 * presented_weight
        if "gentle" in presented_traits and action_type in ["talk", "observe", "support"]:
            score += 2.0 * presented_weight
        
        # Score based on base stats (especially important for femdom game)
        if dominance > 70 and action_type in ["command", "dominate", "test"]:
            score += 2.0
        if dominance < 30 and action_type in ["observe", "wait", "leave"]:
            score += 1.5
        if cruelty > 70 and action_type in ["mock", "humiliate", "punish"]:
            score += 2.0
        
        return score
    
    async def _score_long_term_goals(self, action: Dict[str, Any]) -> float:
        """Score actions based on how they advance long-term goals."""
        if not hasattr(self, 'long_term_goals') or not self.long_term_goals:
            return 0.0
                
        total_score = 0.0
        
        for goal in self.long_term_goals:
            goal_type = goal.get("type", "")
            importance = goal.get("importance", 0.7)
            target_entity = goal.get("target_entity")
            
            # Skip if action target doesn't match goal target
            if (target_entity and 
                action.get("target") != target_entity and 
                not (target_entity == "player" and action.get("target") == "group")):
                continue
            
            # Score based on goal type and action alignment
            if goal_type == "dominance":
                if action.get("type") in ["command", "dominate", "test", "punish"]:
                    total_score += 2.0 * importance
                elif action.get("type") in ["praise", "reward_submission"]:
                    total_score += 1.5 * importance
                    
            elif goal_type == "submission":
                if action["type"] in ["observe", "obey", "assist"]:
                    total_score += 2.0 * importance
                    
            elif goal_type == "sadism":
                if action["type"] in ["punish", "humiliate", "mock"]:
                    total_score += 2.0 * importance
                    
            elif goal_type == "guidance":
                if action["type"] in ["teach", "praise", "correct"]:
                    total_score += 2.0 * importance
                    
            elif goal_type == "seduction":
                if action["type"] in ["seduce", "flirt", "intimate_touch"]:
                    total_score += 2.0 * importance
        
        return total_score
    
    async def _update_goal_progress(self, action: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        """Update progress on long-term goals based on action outcomes."""
        if not hasattr(self, 'long_term_goals') or not self.long_term_goals:
            return
            
        # Get outcome data
        success = "success" in outcome.get("result", "").lower()
        emotional_impact = outcome.get("emotional_impact", 0)
        
        for i, goal in enumerate(self.long_term_goals):
            goal_type = goal.get("type", "")
            target_entity = goal.get("target_entity")
            current_progress = goal.get("progress", 0)
            
            # Skip if action target doesn't match goal target
            if (target_entity and 
                action.get("target") != target_entity and 
                not (target_entity == "player" and action.get("target") == "group")):
                continue
                
            # Calculate progress update based on action type and success
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
                    
            elif goal_type == "seduction":
                if action["type"] in ["seduce", "flirt"]:
                    # Emotional impact is key for seduction
                    if emotional_impact > 0:
                        progress_update = emotional_impact
            
            # Update progress (capped at 0-100)
            if progress_update != 0:
                self.long_term_goals[i]["progress"] = max(0, min(100, current_progress + progress_update))

    async def _score_time_context_influence(self, time_context: Dict[str, Any], action: Dict[str, Any]) -> float:
        """Score actions based on time of day appropriateness."""
        score = 0.0
        
        time_of_day = time_context.get("time_of_day", "").lower()
        action_type = action.get("type", "")
        
        # Time-appropriate actions score higher
        if time_of_day == "morning":
            if action_type in ["talk", "observe", "socialize"]:
                score += 1.0  # Good morning activities
            elif action_type in ["sleep", "seduce", "dominate"]:
                score -= 1.0  # Less appropriate for morning
                
        elif time_of_day == "afternoon":
            if action_type in ["talk", "socialize", "command", "test"]:
                score += 1.0  # Good afternoon activities
                
        elif time_of_day == "evening":
            if action_type in ["talk", "socialize", "seduce", "flirt"]:
                score += 1.5  # Good evening activities
                
        elif time_of_day == "night":
            if action_type in ["seduce", "dominate", "sleep"]:
                score += 2.0  # Good night activities
            elif action_type in ["talk", "socialize"]:
                score -= 0.5  # Less appropriate late at night
        
        return score

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
        
        # Get emotion data for psychological realism
        current_emotion = emotional_state.get("current_emotion", {})
        primary_emotion = current_emotion.get("primary", {}).get("name", "neutral")
        emotion_intensity = current_emotion.get("primary", {}).get("intensity", 0.5)
        
        # Get relevant memories with recency weighting
        memories = perception.get("relevant_memories", [])
        
        # Get mask integrity - lower integrity means true nature shows more
        mask_integrity = mask.get("integrity", 100)
        presented_traits = mask.get("presented_traits", {})
        hidden_traits = mask.get("hidden_traits", {})
        
        # Calculate how much the player knows about the NPC (influences mask behavior)
        player_knowledge = self._calculate_player_knowledge(perception, mask_integrity)
        
        # Get flashback if present
        flashback = perception.get("flashback")
        traumatic_trigger = perception.get("traumatic_trigger")
        
        # Score each action
        for action in actions:
            # Start with base score of 0
            score = 0.0
            scoring_factors = {}
            
            # 1. Personality alignment - how well action aligns with NPC traits
            personality_score = await self._score_personality_alignment(npc_data, action, mask_integrity, hidden_traits, presented_traits)
            score += personality_score
            scoring_factors["personality_alignment"] = personality_score
            
            # 2. Memory influence - how memories affect this action preference
            memory_score = await self._score_memory_influence(memories, action)
            score += memory_score
            scoring_factors["memory_influence"] = memory_score
            
            # 3. Relationship influence - how relationships affect action
            relationship_score = await self._score_relationship_influence(
                perception.get("relationships", {}), 
                action
            )
            score += relationship_score
            scoring_factors["relationship_influence"] = relationship_score
            
            # 4. Environmental context - how environment affects action choice
            environment_score = await self._score_environmental_context(
                perception.get("environment", {}), 
                action
            )
            score += environment_score
            scoring_factors["environmental_context"] = environment_score
            
            # 5. Emotional state influence - how emotions affect decisions
            emotional_score = await self._score_emotional_influence(
                primary_emotion,
                emotion_intensity,
                action
            )
            score += emotional_score
            scoring_factors["emotional_influence"] = emotional_score
            
            # 6. Mask influence - true nature shows more as mask deteriorates
            mask_score = await self._score_mask_influence(
                mask_integrity,
                npc_data,
                action,
                hidden_traits,
                presented_traits
            )
            score += mask_score
            scoring_factors["mask_influence"] = mask_score
            
            # 7. Flashback and trauma influence
            trauma_score = 0.0
            if flashback or traumatic_trigger:
                trauma_score = await self._score_trauma_influence(
                    action,
                    flashback,
                    traumatic_trigger
                )
                score += trauma_score
            scoring_factors["trauma_influence"] = trauma_score
            
            # 8. Belief influence - how beliefs affect decisions
            belief_score = await self._score_belief_influence(
                perception.get("beliefs", []),
                action
            )
            score += belief_score
            scoring_factors["belief_influence"] = belief_score
            
            # 9. Decision history - consider past decisions for continuity
            if hasattr(self, 'decision_history'):
                history_score = await self._score_decision_history(action)
                score += history_score
                scoring_factors["decision_history"] = history_score
            else:
                scoring_factors["decision_history"] = 0.0
            
            # 10. Player knowledge influence - if player knows NPC well, behave differently
            player_knowledge_score = await self._score_player_knowledge_influence(
                action,
                player_knowledge,
                hidden_traits
            )
            score += player_knowledge_score
            scoring_factors["player_knowledge"] = player_knowledge_score
            
            # Add to scored actions
            scored_actions.append({
                "action": action, 
                "score": score,
                "reasoning": scoring_factors
            })

            time_context = perception.get("time_context", {})
            if time_context:
                time_context_score = await self._score_time_context_influence(time_context, action)
                score += time_context_score
                scoring_factors["time_context_influence"] = time_context_score                                        
        
        # Sort by score (descending)
        scored_actions.sort(key=lambda x: x["score"], reverse=True)
        
        # Add thought process tracking for internal reasoning
        self._record_decision_reasoning(scored_actions[:3])
        
        return scored_actions

    async def _score_memory_influence(self, memories: List[Dict[str, Any]], action: Dict[str, Any]) -> float:
        """Score how memories influence action preference with enhanced weighting."""
        score = 0.0
        
        # If no memories, neutral influence
        if not memories:
            return score
        
        # Track which memories affected this decision for later reflection
        affected_by_memories = []
        
        # Weight by recency and emotional intensity
        for i, memory in enumerate(memories):
            memory_text = memory.get("text", "").lower()
            memory_id = memory.get("id")
            emotional_intensity = memory.get("emotional_intensity", 0) / 100.0  # Normalize to 0-1
            recency_weight = 1.0 - (i * 0.15)  # More recent memories get more weight
            significance = memory.get("significance", 3) / 10.0  # Normalize to 0-1
            
            # Calculate base memory importance 
            memory_importance = (recency_weight + emotional_intensity + significance) / 3
            
            # Direct reference to memory
            if action.get("memory_id") == memory_id:
                memory_score = 5 * memory_importance
                score += memory_score
                affected_by_memories.append({"id": memory_id, "influence": memory_score})
                
            # Check for content relevance to the action
            action_type = action["type"]
            if action_type in memory_text:
                memory_score = 2 * memory_importance
                score += memory_score
                affected_by_memories.append({"id": memory_id, "influence": memory_score})
                
            # Check target relevance
            if "target" in action:
                target = str(action.get("target", ""))
                target_name = action.get("target_name", "")
                if target in memory_text or (target_name and target_name.lower() in memory_text):
                    memory_score = 3 * memory_importance
                    score += memory_score
                    affected_by_memories.append({"id": memory_id, "influence": memory_score})
                    
            # Check emotion relevance (actions aligned with memory emotions score higher)
            memory_emotion = memory.get("primary_emotion", "").lower()
            if memory_emotion:
                emotion_aligned_actions = {
                    "anger": ["express_anger", "mock", "attack", "challenge"],
                    "fear": ["leave", "observe", "act_defensive"],
                    "joy": ["celebrate", "praise", "talk", "confide"],
                    "sadness": ["observe", "leave"],
                    "disgust": ["mock", "leave", "observe"]
                }
                
                if action_type in emotion_aligned_actions.get(memory_emotion, []):
                    emotion_score = 2 * memory_importance * emotional_intensity
                    score += emotion_score
                    affected_by_memories.append({"id": memory_id, "influence_type": "emotion", "influence": emotion_score})
                    
            # Check schema interpretations
            schema = memory.get("schema_interpretation", "")
            if schema:
                schema_lower = schema.lower()
                
                # Schema patterns that influence actions
                if "betrayal" in schema_lower and action_type in ["observe", "leave", "act_defensive"]:
                    schema_score = 3 * memory_importance
                    score += schema_score
                    affected_by_memories.append({"id": memory_id, "influence_type": "schema", "influence": schema_score})
                    
                elif "trust" in schema_lower and action_type in ["confide", "praise", "talk"]:
                    schema_score = 3 * memory_importance
                    score += schema_score
                    affected_by_memories.append({"id": memory_id, "influence_type": "schema", "influence": schema_score})
                    
                elif "dominance" in schema_lower and action_type in ["command", "test", "dominate"]:
                    schema_score = 3 * memory_importance
                    score += schema_score
                    affected_by_memories.append({"id": memory_id, "influence_type": "schema", "influence": schema_score})
        
        # Store memory influence metadata in the action for future reflection
        if affected_by_memories:
            if "decision_metadata" not in action:
                action["decision_metadata"] = {}
            action["decision_metadata"]["memory_influences"] = affected_by_memories
        
        return score

    async def _score_emotional_influence(self, emotion: str, intensity: float, action: Dict[str, Any]) -> float:
        """
        Score based on NPC's current emotional state with psychological realism.
        Different emotions favor different action types with intensity scaling.
        """
        score = 0.0
        
        # No significant emotional influence if intensity is low
        if intensity < 0.4:
            return 0.0
        
        # Different emotions favor different action types
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
            "surprise": {
                "observe": 3, "act_defensive": 2, "leave": 1,
                "confide": -2, "celebrate": -1
            },
            "trust": {
                "confide": 3, "praise": 2, "talk": 2, "deepen_relationship": 3,
                "reward_submission": 2,
                "mock": -3, "act_defensive": -2
            },
            "anticipation": {
                "talk": 2, "observe": 1, "discuss_topic": 3, "test": 2
            },
            "shame": {
                "leave": 4, "observe": 2, "act_defensive": 1,
                "celebrate": -4, "socialize": -3, "dominate": -3, "command": -2
            },
            "satisfaction": {
                "praise": 3, "talk": 2, "reward_submission": 3, "confide": 2,
                "mock": -2, "punish": -2
            },
            "curiosity": {
                "observe": 4, "talk": 3, "discuss_topic": 4, "question": 3,
                "leave": -2
            },
            "determination": {
                "command": 3, "test": 3, "dominate": 2, "challenge": 3,
                "leave": -2, "observe": -1
            }
        }
        
        # Get action type affinities for this emotion
        action_affinities = emotion_action_affinities.get(emotion, {})
        
        # Apply affinity score scaled by emotional intensity
        action_type = action["type"]
        if action_type in action_affinities:
            affinity = action_affinities[action_type]
            score += affinity * intensity
        
        # Add decision metadata for reflection
        if "decision_metadata" not in action:
            action["decision_metadata"] = {}
        action["decision_metadata"]["emotional_influence"] = {
            "emotion": emotion,
            "intensity": intensity,
            "affinity": action_affinities.get(action_type, 0),
            "score": score
        }
        
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
        
        # Track mask influence details
        mask_influences = []
        
        # Score based on action alignment with presented vs hidden traits
        action_type = action["type"]
        
        # Check if action aligns with hidden traits that are starting to show
        for trait, trait_data in hidden_traits.items():
            trait_intensity = trait_data.get("intensity", 50) / 100  # Normalize to 0-1
            
            # Calculate weight based on trait intensity and mask integrity
            trait_weight = trait_intensity * true_nature_factor
            
            # Check alignment with hidden traits
            trait_score = 0
            
            if trait == "dominant" and action_type in ["command", "test", "dominate", "punish"]:
                trait_score = 5 * trait_weight
            elif trait == "cruel" and action_type in ["mock", "humiliate", "punish"]:
                trait_score = 5 * trait_weight
            elif trait == "sadistic" and action_type in ["punish", "humiliate", "mock"]:
                trait_score = 5 * trait_weight
            elif trait == "controlling" and action_type in ["command", "test", "direct"]:
                trait_score = 4 * trait_weight
            elif trait == "manipulative" and action_type in ["confide", "praise", "direct"]:
                trait_score = 4 * trait_weight
            elif trait == "lustful" and action_type in ["seduce", "flirt", "observe"]:
                trait_score = 4 * trait_weight
            elif trait == "aggressive" and action_type in ["mock", "challenge", "express_anger"]:
                trait_score = 4 * trait_weight
                
            if trait_score != 0:
                score += trait_score
                mask_influences.append({
                    "trait": trait,
                    "type": "hidden",
                    "score": trait_score
                })
        
        # Check if action conflicts with presented traits (should score lower as mask breaks)
        for trait, trait_data in presented_traits.items():
            trait_confidence = trait_data.get("confidence", 50) / 100  # Normalize to 0-1
            
            # As mask breaks, these conflicts become more permissible
            mask_factor = 1.0 - true_nature_factor
            
            # Check conflicts with presented traits
            trait_score = 0
            
            if trait == "kind" and action_type in ["mock", "humiliate", "punish"]:
                trait_score = -3 * trait_confidence * mask_factor
            elif trait == "gentle" and action_type in ["dominate", "express_anger", "punish"]:
                trait_score = -3 * trait_confidence * mask_factor
            elif trait == "submissive" and action_type in ["command", "dominate", "direct"]:
                trait_score = -4 * trait_confidence * mask_factor
            elif trait == "honest" and action_type in ["deceive", "manipulate", "lie"]:
                trait_score = -4 * trait_confidence * mask_factor
            elif trait == "patient" and action_type in ["express_anger", "mock", "punish"]:
                trait_score = -3 * trait_confidence * mask_factor
                
            if trait_score != 0:
                score += trait_score
                mask_influences.append({
                    "trait": trait,
                    "type": "presented",
                    "score": trait_score
                })
        
        # Record mask influence details
        if mask_influences and "decision_metadata" not in action:
            action["decision_metadata"] = {}
        
        if mask_influences:
            action["decision_metadata"]["mask_influence"] = {
                "integrity": mask_integrity,
                "true_nature_factor": true_nature_factor,
                "trait_influences": mask_influences
            }
        
        return score

    async def _score_trauma_influence(self, action: Dict[str, Any], 
                                    flashback: Optional[Dict[str, Any]], 
                                    traumatic_trigger: Optional[Dict[str, Any]]) -> float:
        """
        Score actions based on flashbacks or traumatic triggers.
        Traumatic experiences strongly influence behavior when triggered.
        """
        score = 0.0
        
        if not flashback and not traumatic_trigger:
            return score
            
        # Different action types for trauma responses
        trauma_action_map = {
            "traumatic_response": 5.0,  # Dedicated trauma response
            "act_defensive": 4.0,       # Defensive behavior
            "leave": 3.5,               # Flight response
            "express_anger": 3.0,       # Fight response
            "observe": 2.5,             # Freeze response
            "emotional_outburst": 3.0,  # Emotional expression of trauma
        }
        
        # Avoidance actions that trauma typically promotes
        avoidance_actions = ["leave", "observe", "act_defensive"]
        
        # Determine action relevance to trauma
        action_type = action["type"]
        
        # Direct trauma response scores highest
        if action_type in trauma_action_map:
            base_score = trauma_action_map[action_type]
            
            # Flashbacks have less influence than active triggers
            if flashback and not traumatic_trigger:
                score += base_score * 0.7  # 70% influence for flashbacks
            elif traumatic_trigger:
                score += base_score  # 100% influence for active triggers
                
                # If the trigger has a specific response type (fight/flight/freeze)
                response_type = traumatic_trigger.get("response_type")
                if response_type == "fight" and action_type in ["express_anger", "challenge"]:
                    score += 2.0
                elif response_type == "flight" and action_type == "leave":
                    score += 2.0
                elif response_type == "freeze" and action_type == "observe":
                    score += 2.0
        
        # Penalize actions that expose vulnerability during trauma
        vulnerability_actions = ["confide", "praise", "talk", "socialize"]
        if action_type in vulnerability_actions:
            if traumatic_trigger:
                score -= 3.0
            elif flashback:
                score -= 2.0
        
        # Add trauma influence metadata
        if flashback or traumatic_trigger:
            if "decision_metadata" not in action:
                action["decision_metadata"] = {}
                
            action["decision_metadata"]["trauma_influence"] = {
                "has_flashback": flashback is not None,
                "has_trigger": traumatic_trigger is not None,
                "score": score
            }
        
        return score

    async def _score_belief_influence(self, beliefs: List[Dict[str, Any]], action: Dict[str, Any]) -> float:
        """
        Score actions based on how well they align with NPC's beliefs.
        Beliefs strongly influence decision-making based on confidence.
        """
        score = 0.0
        
        if not beliefs:
            return score
            
        action_type = action["type"]
        target = action.get("target", "")
        
        belief_influences = []
        
        for belief in beliefs:
            belief_text = belief.get("belief", "").lower()
            confidence = belief.get("confidence", 0.5)
            
            # Skip low-confidence beliefs
            if confidence < 0.3:
                continue
                
            # Calculate belief relevance to this action
            relevance = 0.0
            align_score = 0.0
            
            # Player-related beliefs
            if target == "player" or target == "group":
                # Trust/friendship beliefs
                if any(word in belief_text for word in ["trust", "friend", "ally", "like"]):
                    if action_type in ["talk", "praise", "confide", "support"]:
                        relevance = 0.8
                        align_score = 3.0
                    elif action_type in ["mock", "challenge", "leave", "punish"]:
                        relevance = 0.7
                        align_score = -3.0
                
                # Distrust/danger beliefs
                elif any(word in belief_text for word in ["threat", "danger", "distrust", "wary"]):
                    if action_type in ["observe", "act_defensive", "leave"]:
                        relevance = 0.9
                        align_score = 4.0
                    elif action_type in ["confide", "praise", "support"]:
                        relevance = 0.8
                        align_score = -4.0
                
                # Submission beliefs
                elif any(word in belief_text for word in ["submit", "obey", "follow"]):
                    if action_type in ["command", "test", "dominate"]:
                        relevance = 0.9
                        align_score = 3.5
                    elif action_type in ["observe", "act_defensive"]:
                        relevance = 0.5
                        align_score = -2.0
                
                # Rebellion beliefs
                elif any(word in belief_text for word in ["rebel", "defy", "disobey"]):
                    if action_type in ["punish", "test", "command"]:
                        relevance = 0.8
                        align_score = 3.0
                    elif action_type in ["praise", "reward"]:
                        relevance = 0.6
                        align_score = -2.5
                
                # Look for direct action mentions in beliefs
                elif action_type in belief_text:
                    relevance = 0.9
                    align_score = 4.0
                
                # Apply the influence scaled by confidence
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
        
        # Add belief influence metadata
        if belief_influences:
            if "decision_metadata" not in action:
                action["decision_metadata"] = {}
                
            action["decision_metadata"]["belief_influences"] = belief_influences
        
        return score

    async def _score_decision_history(self, action: Dict[str, Any]) -> float:
        """
        Score actions based on decision history for psychological continuity.
        This creates more consistent behavior over time.
        """
        score = 0.0
        
        if not hasattr(self, 'decision_history') or not self.decision_history:
            return score
            
        action_type = action["type"]
        
        # Track patterns in decision history
        # Recent actions are weighted more heavily
        recent_actions = [d["action"]["type"] for d in self.decision_history[-3:]]
        action_counts = {}
        
        for i, a_type in enumerate(recent_actions):
            # More recent actions get more weight
            weight = 1.0 - (i * 0.2)  # 1.0, 0.8, 0.6...
            
            if a_type not in action_counts:
                action_counts[a_type] = 0
                
            action_counts[a_type] += weight
        
        # Psychological continuity - slight preference for consistent behavior
        if action_type in action_counts:
            # Prefer some consistency, but avoid excessive repetition
            consistency_score = action_counts[action_type] * 1.5
            
            # But avoid doing the exact same thing more than twice in a row
            if len(recent_actions) >= 2 and recent_actions[0] == recent_actions[1] == action_type:
                # Penalize doing the same thing three times in a row
                consistency_score -= 3.0
                
            score += consistency_score
        
        # Conversely, don't change behavior too rapidly
        if len(recent_actions) >= 2:
            if len(set(recent_actions)) == len(recent_actions) and action_type not in recent_actions:
                # Penalize constantly changing behavior
                score -= 1.0
        
        # Add history influence metadata
        if "decision_metadata" not in action:
            action["decision_metadata"] = {}
            
        action["decision_metadata"]["history_influence"] = {
            "recent_actions": recent_actions,
            "action_counts": action_counts,
            "score": score
        }
        
        return score

    async def _score_player_knowledge_influence(self, action: Dict[str, Any], 
                                              player_knowledge: float,
                                              hidden_traits: Dict[str, Any]) -> float:
        """
        Score actions based on how much the player knows about the NPC.
        NPCs behave differently when they know they're "seen through".
        """
        score = 0.0
        
        # No influence if player knowledge is low
        if player_knowledge < 0.3:
            return score
            
        action_type = action["type"]
        
        # For high player knowledge, the NPC may be more direct
        if player_knowledge > 0.7:
            # When player knows a lot, less need to maintain the mask
            if "dominant" in hidden_traits and action_type in ["command", "dominate", "punish"]:
                score += 2.0
            elif "cruel" in hidden_traits and action_type in ["mock", "humiliate"]:
                score += 2.0
            elif "submissive" in hidden_traits and action_type in ["observe", "act_defensive"]:
                score += 2.0
                
        # For medium player knowledge, mixed behaviors
        elif player_knowledge > 0.4:
            # Sometimes show true nature, sometimes maintain mask
            if random.random() < 0.5:
                # Show glimpses of true nature
                if "dominant" in hidden_traits and action_type in ["command", "direct"]:
                    score += 1.5
                elif "cruel" in hidden_traits and action_type in ["mock"]:
                    score += 1.5
            else:
                # Try to maintain facade
                if "dominant" in hidden_traits and action_type in ["talk", "observe"]:
                    score += 1.0
        
        # Add knowledge influence metadata
        if player_knowledge > 0.3:
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
        
        # Base knowledge from mask integrity
        if mask_integrity < 50:
            player_knowledge += 0.4  # If mask is damaged, player likely knows more
        elif mask_integrity < 75:
            player_knowledge += 0.2
            
        # Knowledge from memories
        memories = perception.get("relevant_memories", [])
        if len(memories) > 7:
            player_knowledge += 0.3  # Many relevant memories means player knows NPC well
        elif len(memories) > 3:
            player_knowledge += 0.15
            
        # Knowledge from relationship
        relationships = perception.get("relationships", {})
        player_rel = relationships.get("player", {})
        
        if player_rel:
            link_level = player_rel.get("link_level", 0)
            if link_level > 70:
                player_knowledge += 0.3  # Close relationship = more knowledge
            elif link_level > 40:
                player_knowledge += 0.15
        
        # Cap at 0.0-1.0 range
        return min(1.0, max(0.0, player_knowledge))

    def _record_decision_reasoning(self, top_actions: List[Dict[str, Any]]) -> None:
        """Record decision reasoning for introspection and debugging."""
        if not hasattr(self, '_decision_log'):
            self._decision_log = []
            
        decision_entry = {
            "timestamp": datetime.now().isoformat(),
            "top_actions": [{
                "type": a["action"]["type"],
                "score": a["score"],
                "factors": a["reasoning"]
            } for a in top_actions],
            "chosen_action": top_actions[0]["action"]["type"] if top_actions else None
        }
        
        # Add to history, keeping last 20 entries
        self._decision_log.append(decision_entry)
        if len(self._decision_log) > 20:
            self._decision_log = self._decision_log[-20:]

    async def select_action(self, scored_actions: List[Dict[str, Any]], randomness: float = 0.2) -> Dict[str, Any]:
        """
        Select an action from scored actions, with some randomness and pattern breaking.
        
        Args:
            scored_actions: List of actions with scores
            randomness: How much randomness to apply (0.0-1.0)
            
        Returns:
            The selected action
        """
        if not scored_actions:
            return {"type": "idle", "description": "Do nothing"}
        
        # Check for action weights overriding scores
        weight_based_selection = False
        for sa in scored_actions:
            action = sa["action"]
            if "weight" in action and action["weight"] > 2.0:
                # High weight actions get priority
                weight_based_selection = True
                
        if weight_based_selection:
            # Use weights rather than scores
            weighted_actions = [(sa["action"], sa["action"].get("weight", 1.0)) for sa in scored_actions]
            
            # Sort by weight, highest first
            weighted_actions.sort(key=lambda x: x[1], reverse=True)
            
            # Select top weighted action
            selected_action = weighted_actions[0][0]
        else:
            # Normal score-based selection with randomness
            
            # Add randomness to scores
            for sa in scored_actions:
                sa["score"] += random.uniform(0, randomness * 10)
            
            # Re-sort with randomness applied
            scored_actions.sort(key=lambda x: x["score"], reverse=True)
            
            # Select top action
            selected_action = scored_actions[0]["action"]
        
        # Add decision metadata to action for introspection
        if len(scored_actions) > 1:
            if "decision_metadata" not in selected_action:
                selected_action["decision_metadata"] = {}
                
            selected_action["decision_metadata"]["alternative_actions"] = [
                {"type": sa["action"]["type"], "score": sa["score"]} for sa in scored_actions[1:3]
            ]
        
        # Add reasoning if available
        if len(scored_actions) > 0 and "reasoning" in scored_actions[0]:
            if "decision_metadata" not in selected_action:
                selected_action["decision_metadata"] = {}
                
            selected_action["decision_metadata"]["reasoning"] = scored_actions[0]["reasoning"]
        
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
