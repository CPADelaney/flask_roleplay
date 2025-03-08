# logic/npc_agents/npc_agent.py

"""
Core NPC agent class that manages individual NPC behavior with memory capabilities.
"""

import logging
import json
import asyncio
import random
from datetime import datetime
from typing import List, Dict, Any, Optional

from db.connection import get_db_connection
from .decision_engine import NPCDecisionEngine
from .environment_perception import (
    fetch_environment_data,
    is_significant_action,
    execute_npc_action
)

# Memory system imports
from memory.wrapper import MemorySystem
from memory.core import Memory, MemoryType, MemorySignificance
from memory.masks import ProgressiveRevealManager

logger = logging.getLogger(__name__)


class NPCAgent:
    """
    Independent AI agent controlling a single NPC's behavior.
    Enhanced with memory capabilities.

    Responsibilities:
    - Perceive environment (with memory-informed context)
    - Make decisions based on personality, current context, and memory
    - Execute chosen actions
    - Form and utilize memories with advanced cognitive features
    - Manage mask (presented vs. true personality)
    - Process emotional states and reactions
    """

    def __init__(self, npc_id: int, user_id: int, conversation_id: int):
        """
        Initialize an NPCAgent for a single NPC.

        Args:
            npc_id: The ID of the NPC
            user_id: The player or user ID
            conversation_id: The conversation/scene ID
        """
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.decision_engine = NPCDecisionEngine(npc_id, user_id, conversation_id)
        
        # Lazy-loaded memory components
        self._memory_system = None
        self._mask_manager = None
        self.current_emotional_state = None
        self.last_perception = None

        # Enhanced cache management
        self._cache = {
            'perception': {},
            'memories': {},
            'relationships': {},
            'emotional_state': None,
            'mask': None
        }
        self._cache_timestamps = {
            'perception': None,
            'memories': {},
            'relationships': None,
            'emotional_state': None,
            'mask': None
        }
        self._cache_ttls = {
            'perception': timedelta(minutes=5),
            'memories': timedelta(minutes=10),
            'relationships': timedelta(minutes=15),
            'emotional_state': timedelta(minutes=2),
            'mask': timedelta(minutes=5)
        }
        
        self.perf_metrics = {
            'perception_time': [],
            'decision_time': [],
            'action_time': [],
            'memory_retrieval_time': [],
            'last_reported': datetime.now()
        }
        self._setup_performance_reporting()    

    def invalidate_cache(self, cache_key=None):
        """
        Invalidate specific cache entries or all if key is None.
        
        Args:
            cache_key: Specific cache to invalidate, or None for all
        """
        if cache_key is None:
            # Invalidate all caches
            self._cache = {
                'perception': {},
                'memories': {},
                'relationships': {},
                'emotional_state': None,
                'mask': None
            }
            self._cache_timestamps = {
                'perception': None,
                'memories': {},
                'relationships': None,
                'emotional_state': None,
                'mask': None
            }
            logger.debug(f"Invalidated all caches for NPC {self.npc_id}")
        elif cache_key in self._cache:
            # Invalidate specific cache
            if isinstance(self._cache[cache_key], dict):
                self._cache[cache_key] = {}
            else:
                self._cache[cache_key] = None
                
            self._cache_timestamps[cache_key] = None
            logger.debug(f"Invalidated {cache_key} cache for NPC {self.npc_id}")

    def invalidate_memory_cache(self, memory_query=None):
        """
        Invalidate memory cache, either completely or for a specific query.
        
        Args:
            memory_query: Query string to invalidate, or None for all
        """
        if memory_query is None:
            # Invalidate all memory caches
            self._cache['memories'] = {}
            self._cache_timestamps['memories'] = {}
            logger.debug(f"Invalidated all memory caches for NPC {self.npc_id}")
        elif memory_query in self._cache['memories']:
            # Invalidate specific memory query
            del self._cache['memories'][memory_query]
            del self._cache_timestamps['memories'][memory_query]
            logger.debug(f"Invalidated memory cache for query '{memory_query}'")
    
    def is_cache_valid(self, cache_key, sub_key=None):
        """
        Check if a cache entry is still valid based on TTL.
        
        Args:
            cache_key: Main cache key
            sub_key: Optional sub-key for dict caches
            
        Returns:
            True if cache is valid, False otherwise
        """
        now = datetime.now()
        timestamp = None
        
        if cache_key not in self._cache_ttls:
            return False
            
        if sub_key is not None:
            # Check sub-key in dict cache
            if cache_key not in self._cache_timestamps or not isinstance(self._cache_timestamps[cache_key], dict):
                return False
                
            timestamp = self._cache_timestamps[cache_key].get(sub_key)
        else:
            # Check main cache key
            timestamp = self._cache_timestamps.get(cache_key)
            
        if timestamp is None:
            return False
            
        # Check if timestamp + TTL > now (cache still valid)
        return timestamp + self._cache_ttls[cache_key] > now
    
    async def _get_memory_system(self):
        """Lazy-load the memory system."""
        if self._memory_system is None:
            self._memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        return self._memory_system
    
    async def _get_mask_manager(self):
        """Lazy-load the mask manager."""
        if self._mask_manager is None:
            self._mask_manager = ProgressiveRevealManager(self.user_id, self.conversation_id)
        return self._mask_manager

    def _setup_performance_reporting(self):
        """Set up periodic performance reporting."""
        async def report_metrics():
            while True:
                await asyncio.sleep(600)  # Report every 10 minutes
                metrics_dict = {}
                
                # Calculate averages for each metric
                for metric, values in self.perf_metrics.items():
                    if metric != 'last_reported' and values:
                        metrics_dict[f'avg_{metric}'] = sum(values) / len(values)
                        # Keep only last 100 values to avoid memory growth
                        self.perf_metrics[metric] = values[-100:]
                
                if metrics_dict:
                    logger.info(f"NPC {self.npc_id} performance: {metrics_dict}")
                self.perf_metrics['last_reported'] = datetime.now()
        
        # Start the task without waiting for it
        asyncio.create_task(report_metrics())

    async def perceive_environment(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Measure performance of perceive_environment."""
        start_time = datetime.now()
        
        try:
            result = await self._perceive_environment_impl(current_context)
            
            # Record performance metric
            elapsed = (datetime.now() - start_time).total_seconds()
            self.perf_metrics['perception_time'].append(elapsed)
            
            # Log slow operations
            if elapsed > 0.5:  # Log if taking more than 500ms
                logger.warning(f"Slow perception for NPC {self.npc_id}: {elapsed:.2f}s")
            
            return result
        except Exception as e:
            # Record failure in metrics
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"Perception error for NPC {self.npc_id} after {elapsed:.2f}s: {e}")
            raise
    
    async def _perceive_environment_impl(self, current_context: Dict[str, Any]) -> Dict[str, Any]:

        Args:
            current_context: Dictionary that may contain location/time or relevant info

        Returns:
            A dictionary containing:
              - environment (location, time_of_day, etc.)
              - relevant_memories
              - relationships
              - emotional_state
              - mask
              - timestamp
              - beliefs
              - flashback (if triggered)

        # Fetch basic environment data
        environment_data = await fetch_environment_data(
            self.user_id,
            self.conversation_id,
            current_context
        )

        context_key = str(hash(json.dumps(current_context, sort_keys=True, default=str)))
        
        # Check if we have a valid cached perception
        if 'perception' in self._cache and context_key in self._cache['perception'] and self.is_cache_valid('perception', context_key):
            self.last_perception = self._cache['perception'][context_key]
            return self.last_perception

        # Enhance perception with memory system
        memory_system = await self._get_memory_system()
        
        # Get text description from context for better memory recall
        context_description = current_context.get("description", "")
        if "text" in current_context:
            context_description += " " + current_context["text"]
            
        context_for_recall = {
            "text": context_description,
            "location": environment_data.get("location", "Unknown"),
            "time_of_day": environment_data.get("time_of_day", "Unknown"),
            "entities_present": [e.get("name", "") for e in environment_data.get("entities_present", [])]
        }
        
        # Retrieve relevant memories based on current context
        memory_result = await memory_system.recall(
            entity_type="npc",
            entity_id=self.npc_id,
            context=context_for_recall,
            limit=7  # More memories for richer context
        )
        relevant_memories = memory_result.get("memories", [])
        
        # Check for flashback potential
        flashback = None
        if random.random() < 0.15:  # 15% chance of flashback
            flashback = await memory_system.npc_flashback(
                npc_id=self.npc_id, 
                context=context_description
            )
        
        # Get current emotional state
        emotional_state = await memory_system.get_npc_emotion(self.npc_id)
        self.current_emotional_state = emotional_state
        
        # Check for traumatic triggers
        traumatic_trigger = None
        if context_description:
            trigger_result = await memory_system.emotional_manager.process_traumatic_triggers(
                entity_type="npc",
                entity_id=self.npc_id,
                text=context_description
            )
            
            if trigger_result and trigger_result.get("triggered", False):
                traumatic_trigger = trigger_result
        
        # Get mask information (true vs. presented personality)
        mask_info = await memory_system.get_npc_mask(self.npc_id)
        
        # Get relationship data
        relationship_data = await self._fetch_relationships()
        
        # Get beliefs that might influence perception
        beliefs = await memory_system.get_beliefs(
            entity_type="npc",
            entity_id=self.npc_id,
            topic="player"  # Focus on player-related beliefs
        )
        
        # Combine into a single perception dictionary
        perception = {
            "environment": environment_data,
            "relevant_memories": relevant_memories,
            "flashback": flashback,
            "relationships": relationship_data,
            "emotional_state": emotional_state,
            "traumatic_trigger": traumatic_trigger,
            "mask": mask_info,
            "beliefs": beliefs,
            "timestamp": datetime.now().isoformat()
        }

        if 'perception' not in self._cache:
            self._cache['perception'] = {}
        self._cache['perception'][context_key] = perception
        
        if 'perception' not in self._cache_timestamps:
            self._cache_timestamps['perception'] = {}
        self._cache_timestamps['perception'][context_key] = datetime.now()
        
        self.last_perception = perception
        return perception

    async def make_decision(self,
                           perception: Optional[Dict[str, Any]] = None,
                           available_actions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Decide which action to take based on current perception and available actions.
        Enhanced with memory-driven decision making.

        Args:
            perception: The NPC's current perception dictionary
            available_actions: A list of possible actions the NPC could choose from

        Returns:
            A dictionary describing the chosen action (type, description, target, etc.)
        """
        if perception is None:
            # If no provided perception, use the last known or fetch a fresh one
            if self.last_perception is None:
                logger.debug("No prior perception found, fetching fresh environment data.")
                perception = await self.perceive_environment({})
            else:
                perception = self.last_perception

        # Get memory system
        memory_system = await self._get_memory_system()
        
        # Get beliefs that should influence decisions
        beliefs = perception.get("beliefs", [])
        
        # Apply belief weights to available actions
        if available_actions and beliefs:
            weighted_actions = await self._apply_belief_weights_to_actions(available_actions, beliefs)
            available_actions = weighted_actions
            
        # Check if there's an active traumatic trigger that should influence decision
        traumatic_trigger = perception.get("traumatic_trigger")
        if traumatic_trigger and traumatic_trigger.get("triggered", False):
            # Modify available actions or create a trauma-response action
            trauma_response = self._create_trauma_response_action(traumatic_trigger)
            if trauma_response:
                # Add trauma response as a high-priority action
                if available_actions:
                    available_actions.append(trauma_response)
                else:
                    available_actions = [trauma_response]
        
        # Apply emotional state influences to decision-making
        emotional_state = perception.get("emotional_state", {})
        current_emotion = emotional_state.get("current_emotion", {})
        
        if current_emotion:
            primary_emotion = current_emotion.get("primary", {})
            emotion_name = primary_emotion.get("name", "neutral")
            intensity = primary_emotion.get("intensity", 0.0)
            
            # For intense emotions, they might override normal decision-making
            if intensity > 0.7 and emotion_name in ["anger", "fear"]:
                emotional_action = self._create_emotional_response_action(emotion_name, intensity)
                if emotional_action:
                    if available_actions:
                        available_actions.insert(0, emotional_action)
                    else:
                        available_actions = [emotional_action]

        # Pass the enriched perception with memories to the decision engine
        chosen_action = await self.decision_engine.decide(perception, available_actions)
        logger.debug("NPCAgent %s decided on action: %s", self.npc_id, chosen_action)
        
        # Check if this decision should trigger a mask slippage
        should_slip = False
        mask_info = perception.get("mask", {})
        
        # Calculate chance of mask slippage based on:
        # - NPC stats (higher dominance/cruelty means less control)
        # - Mask integrity (lower integrity = higher chance of slippage)
        # - Current emotional intensity
        
        # Get basic stats
        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
                SELECT dominance, cruelty
                FROM NPCStats
                WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
            """, (self.npc_id, self.user_id, self.conversation_id))
            row = cursor.fetchone()
            if row:
                dominance, cruelty = row
                
                # Higher dominance/cruelty with lower mask integrity increases slip chance
                mask_integrity = mask_info.get("integrity", 100)
                slip_chance = (dominance + cruelty) / 200  # 0.0 to 1.0
                slip_chance *= (100 - mask_integrity) / 100  # Factor in mask integrity
                
                # Also factor in emotional intensity
                emotion_intensity = current_emotion.get("primary", {}).get("intensity", 0.0)
                slip_chance *= (1 + emotion_intensity)
                
                # Decision to slip
                should_slip = random.random() < slip_chance
                
                if should_slip:
                    # Generate mask slippage
                    trigger = f"deciding to {chosen_action.get('description', 'act')}"
                    
                    # Emotional triggers are more likely to cause slips
                    if emotion_name in ["anger", "fear", "excitement"]:
                        trigger = f"feeling {emotion_name} while " + trigger
                        
                    slip_result = await memory_system.reveal_npc_trait(
                        npc_id=self.npc_id,
                        trigger=trigger
                    )
                    
                    # Add mask slippage information to action
                    chosen_action["mask_slippage"] = slip_result
        
        return chosen_action
    
    async def _apply_belief_weights_to_actions(self, 
                                             available_actions: List[Dict[str, Any]], 
                                             beliefs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply belief weights to actions to influence decision-making."""
        for action in available_actions:
            action_type = action.get("type", "")
            target = action.get("target", "")
            
            # Default weight starts at 1.0
            belief_weight = 1.0
            
            for belief in beliefs:
                belief_text = belief.get("belief", "").lower()
                confidence = belief.get("confidence", 0.5)
                
                # Skip low-confidence beliefs
                if confidence < 0.3:
                    continue
                
                # Apply belief weights based on action type and target
                if target == "player" or target == "group":
                    # Positive beliefs about player increase talk/observe weights
                    if ("trust" in belief_text or "like" in belief_text) and action_type in ["talk", "observe"]:
                        belief_weight += confidence * 0.5
                    
                    # Negative beliefs increase leave/observe weights
                    elif ("distrust" in belief_text or "fear" in belief_text) and action_type in ["leave", "observe"]:
                        belief_weight += confidence * 0.5
                        
                    # Negative beliefs decrease talk weights
                    elif ("distrust" in belief_text or "fear" in belief_text) and action_type == "talk":
                        belief_weight -= confidence * 0.3
                
                # For specific NPC targets (by ID)
                elif target.isdigit():
                    target_id = int(target)
                    if f"npc_{target_id}" in belief_text:
                        if "friend" in belief_text and action_type in ["talk_to", "observe"]:
                            belief_weight += confidence * 0.5
                        elif "enemy" in belief_text and action_type in ["mock", "command"]:
                            belief_weight += confidence * 0.4
            
            # Apply the calculated weight to the action
            if "weight" in action:
                action["weight"] *= belief_weight
            else:
                action["weight"] = belief_weight
                
        return available_actions
    
    def _create_trauma_response_action(self, trauma_trigger: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create an action in response to a traumatic trigger."""
        # Get the emotional response from the trigger
        emotional_response = trauma_trigger.get("emotional_response", {})
        primary_emotion = emotional_response.get("primary_emotion", "fear")
        intensity = emotional_response.get("intensity", 0.5)
        
        # Different responses based on the primary emotion
        if primary_emotion == "fear":
            return {
                "type": "traumatic_response",
                "description": "react with fear to a triggering memory",
                "target": "group",
                "weight": 2.0 * intensity,  # High priority
                "stats_influenced": {"trust": -10}
            }
        elif primary_emotion == "anger":
            return {
                "type": "traumatic_response",
                "description": "respond with anger to a triggering situation",
                "target": "group",
                "weight": 1.8 * intensity,
                "stats_influenced": {"trust": -5, "respect": -5}
            }
        else:
            return {
                "type": "traumatic_response",
                "description": f"respond emotionally to a triggering memory",
                "target": "group",
                "weight": 1.5 * intensity,
                "stats_influenced": {}
            }
    
    def _create_emotional_response_action(self, emotion: str, intensity: float) -> Optional[Dict[str, Any]]:
        """Create an action based on a strong emotional state."""
        if emotion == "anger":
            return {
                "type": "emotional_outburst",
                "description": "express anger",
                "target": "group",
                "weight": 1.7 * intensity,
                "stats_influenced": {"respect": -10}
            }
        elif emotion == "fear":
            return {
                "type": "emotional_response",
                "description": "show fear and defensiveness",
                "target": "group",
                "weight": 1.6 * intensity,
                "stats_influenced": {"trust": -5}
            }
        elif emotion == "joy":
            return {
                "type": "emotional_expression",
                "description": "express happiness and excitement",
                "target": "group",
                "weight": 1.2 * intensity,
                "stats_influenced": {"trust": 5}
            }
        return None

    async def execute_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the chosen action in the game world.
        Records the action in memory.

        Args:
            action: Dictionary describing the action
            context: Additional contextual information for the execution

        Returns:
            A dictionary describing the result (e.g. outcome, emotional impact).
        """
        # Execute the action using existing code
        result = await execute_npc_action(
            self.npc_id,
            self.user_id,
            self.conversation_id,
            action,
            context
        )
        logger.debug("NPCAgent %s executed action '%s', got result: %s", self.npc_id, action, result)

        # Record the action in memory if significant
        if is_significant_action(action, result):
            memory_system = await self._get_memory_system()
            
            # Format memory text
            memory_text = f"I {action.get('description', 'did something')} which resulted in {result.get('outcome', 'something happening')}"
            
            # Get emotional analysis for more accurate emotional tagging
            emotion_analysis = await memory_system.emotional_manager.analyze_emotional_content(memory_text)
            
            # Record in memory system with emotional context
            memory_result = await memory_system.emotional_manager.add_emotional_memory(
                entity_type="npc",
                entity_id=self.npc_id,
                memory_text=memory_text,
                primary_emotion=emotion_analysis.get("primary_emotion", "neutral"),
                emotion_intensity=emotion_analysis.get("intensity", 0.5),
                secondary_emotions=emotion_analysis.get("secondary_emotions", {}),
                significance=MemorySignificance.MEDIUM,  # Default importance
                tags=["action", action.get("type", "unknown")]
            )
            
            # Update emotional state based on the action's impact
            emotional_impact = result.get("emotional_impact", 0)
            if abs(emotional_impact) >= 2:
                # Map emotional impact to an emotion
                if emotional_impact > 2:
                    emotion = "joy"
                    intensity = min(1.0, abs(emotional_impact) / 5.0)
                elif emotional_impact > 0:
                    emotion = "satisfaction"
                    intensity = min(0.7, abs(emotional_impact) / 5.0)
                elif emotional_impact < -2:
                    emotion = "anger"
                    intensity = min(1.0, abs(emotional_impact) / 5.0)
                else:
                    emotion = "sadness"
                    intensity = min(0.7, abs(emotional_impact) / 5.0)
                
                # Update emotional state
                await memory_system.update_npc_emotion(
                    npc_id=self.npc_id,
                    emotion=emotion,
                    intensity=intensity
                )

        return result

    async def process_player_action(self, player_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a player action and generate an NPC response.
        Enhanced with memory of the interaction.

        Steps:
          1) Update the NPC's perception based on the player action context
          2) Remember the player's action
          3) Decide on a response action
          4) Execute that action
          5) Update relationships and record a memory of the interaction

        Args:
            player_action: A dict describing the player's action

        Returns:
            A dict: { "npc_id":..., "action":..., "result":... }
        """
        # Incorporate the player's action into the environment context
        context = {
            "player_action": player_action,
            "text": player_action.get("description", ""),
            "description": f"Player {player_action.get('description', 'did something')}"
        }

        # Step 1: Refresh or update our environment perception
        perception = await self.perceive_environment(context)
        
        # Step 2: Remember the player's action
        memory_system = await self._get_memory_system()
        
        # Format memory text for what player did
        player_memory_text = f"The player {player_action.get('description', 'did something')}"
        
        # Get emotional analysis of the player's action
        emotion_analysis = await memory_system.emotional_manager.analyze_emotional_content(
            player_memory_text, 
            context=str(player_action)
        )
        
        # Record the player's action as an emotional memory
        memory_kwargs = {
            "tags": ["player_action", player_action.get("type", "unknown")]
        }
        
        # Create an emotional memory with proper emotional context
        memory_result = await memory_system.emotional_manager.add_emotional_memory(
            entity_type="npc",
            entity_id=self.npc_id,
            memory_text=player_memory_text,
            primary_emotion=emotion_analysis.get("primary_emotion", "neutral"),
            emotion_intensity=emotion_analysis.get("intensity", 0.5),
            secondary_emotions=emotion_analysis.get("secondary_emotions", {}),
            significance=MemorySignificance.MEDIUM,  # Default importance
            tags=memory_kwargs["tags"]
        )
        
        memory_id = memory_result.get("memory_id")
        
        # Apply schemas to the memory to improve future recall and interpretation
        if memory_id:
            await memory_system.integrated.apply_schema_to_memory(
                memory_id=memory_id,
                entity_type="npc",
                entity_id=self.npc_id,
                auto_detect=True
            )
        
        # Check if the player action triggers specific emotions
        player_action_type = player_action.get("type", "").lower()
        if player_action_type == "insult":
            # Insult might trigger anger or sadness
            if perception.get("mask", {}).get("integrity", 100) < 50:
                # Low mask integrity - more likely to show anger
                await memory_system.update_npc_emotion(self.npc_id, "anger", 0.7)
            else:
                # High mask integrity - more likely to mask true feelings
                await memory_system.update_npc_emotion(self.npc_id, "sadness", 0.5)
        elif player_action_type == "command" or player_action_type == "dominate":
            # Check if NPC has dominant traits hidden under the mask
            hidden_traits = perception.get("mask", {}).get("hidden_traits", {})
            if "dominant" in hidden_traits or "controlling" in hidden_traits:
                # Dominant NPC being dominated - might trigger anger or resentment
                await memory_system.update_npc_emotion(self.npc_id, "anger", 0.6)
                
                # Higher chance of mask slippage
                if random.random() < 0.3:
                    await memory_system.reveal_npc_trait(
                        npc_id=self.npc_id,
                        trigger=f"being commanded by player to {player_action.get('description', 'do something')}"
                    )
        
        # Step 3: Decide how to respond
        response_action = await self.make_decision(perception)

        # Step 4: Execute the chosen response
        result = await self.execute_action(response_action, context)

        # Step 5: Remember the interaction
        interaction_memory_text = (
            f"When the player {player_action.get('description','did something')}, "
            f"I responded by {response_action.get('description','doing something')}"
        )
        
        # Get emotional analysis of the interaction
        interaction_emotion_analysis = await memory_system.emotional_manager.analyze_emotional_content(
            interaction_memory_text
        )
        
        # Create emotional memory of the interaction
        await memory_system.emotional_manager.add_emotional_memory(
            entity_type="npc",
            entity_id=self.npc_id,
            memory_text=interaction_memory_text,
            primary_emotion=interaction_emotion_analysis.get("primary_emotion", "neutral"),
            emotion_intensity=interaction_emotion_analysis.get("intensity", 0.5),
            secondary_emotions=interaction_emotion_analysis.get("secondary_emotions", {}),
            significance=MemorySignificance.MEDIUM,
            tags=["interaction", "player", player_action.get("type", "unknown")]
        )
        
        # Step 6: Update beliefs based on interaction
        await self._update_beliefs_from_interaction(player_action, response_action, result)
        
        # Step 7: Update relationships if relevant
        if player_action.get("type") == "talk":
            # Potentially update relationship here based on emotional outcome
            pass

        logger.debug("NPCAgent %s processed player action '%s': result=%s", 
                    self.npc_id, player_action, result)

        return {
            "npc_id": self.npc_id,
            "action": response_action,
            "result": result
        }
    
    async def _update_beliefs_from_interaction(self, 
                                            player_action: Dict[str, Any], 
                                            npc_action: Dict[str, Any],
                                            result: Dict[str, Any]) -> None:
        """Update beliefs based on an interaction with the player."""
        memory_system = await self._get_memory_system()
        
        # Get current beliefs about the player
        current_beliefs = await memory_system.get_beliefs(
            entity_type="npc",
            entity_id=self.npc_id,
            topic="player"
        )
        
        # Determine if we need to create or update beliefs
        player_action_type = player_action.get("type", "")
        outcome = result.get("outcome", "")
        emotional_impact = result.get("emotional_impact", 0)
        
        # Example belief updates based on action types
        if player_action_type == "help" and emotional_impact > 0:
            # Check if we already have a belief about player helpfulness
            helpful_belief = next((b for b in current_beliefs if "helpful" in b.get("belief", "").lower()), None)
            
            if helpful_belief:
                # Update confidence in existing belief
                await memory_system.semantic_manager.update_belief_confidence(
                    belief_id=helpful_belief["id"],
                    entity_type="npc",
                    entity_id=self.npc_id,
                    new_confidence=min(1.0, helpful_belief["confidence"] + 0.1),
                    reason=f"Player helped again with {player_action.get('description', 'something')}"
                )
            else:
                # Create new belief
                await memory_system.create_belief(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    belief_text=f"The player is helpful and supportive",
                    confidence=0.6
                )
        
        elif player_action_type in ["attack", "threaten", "insult"] and emotional_impact < 0:
            # Check if we already have a belief about player being dangerous
            dangerous_belief = next((b for b in current_beliefs if "dangerous" in b.get("belief", "").lower() 
                                   or "threat" in b.get("belief", "").lower()), None)
            
            if dangerous_belief:
                # Update confidence in existing belief
                await memory_system.semantic_manager.update_belief_confidence(
                    belief_id=dangerous_belief["id"],
                    entity_type="npc",
                    entity_id=self.npc_id,
                    new_confidence=min(1.0, dangerous_belief["confidence"] + 0.15),
                    reason=f"Player was hostile again with {player_action.get('description', 'something')}"
                )
            else:
                # Create new belief
                await memory_system.create_belief(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    belief_text=f"The player is potentially dangerous and should be treated carefully",
                    confidence=0.55
                )

    async def perform_scheduled_activity(self) -> Optional[Dict[str, Any]]:
        """
        Perform the activity scheduled for this NPC at the current time of day.
        Enhanced with memory of routines.

        Returns:
            A dict like {"npc_id":..., "action":..., "result":...}
            or None if no scheduled activity found or error occurs.
        """
        from db.connection import get_db_connection
        try:
            with get_db_connection() as conn, conn.cursor() as cursor:
                # 1) Load timeOfDay & CurrentDay
                time_of_day = self._fetch_current_time_of_day(cursor)
                day_name = self._fetch_current_day_name(cursor)

                # 2) Retrieve NPC's schedule
                sched = self._fetch_npc_schedule(cursor)
                if not sched or day_name not in sched or time_of_day not in sched[day_name]:
                    logger.debug("No schedule found for NPC %s on day='%s' time='%s'.", 
                                self.npc_id, day_name, time_of_day)
                    return None

                activity_desc = sched[day_name][time_of_day]
                action = {
                    "type": "scheduled",
                    "description": activity_desc,
                    "target": "environment",
                    "stats_influenced": {}
                }

            # 3) Execute
            context = {
                "day": day_name,
                "time": time_of_day,
                "location": "scheduled_location"
            }
            result = await self.execute_action(action, context)

            # 4) Record a memory for routine using memory system
            memory_system = await self._get_memory_system()
            
            memory_text = f"I did '{activity_desc}' as scheduled for {day_name} {time_of_day}"
            
            # Analyze emotional content of the routine
            emotion_analysis = await memory_system.emotional_manager.analyze_emotional_content(memory_text)
            
            # Create emotional memory of the routine
            memory_result = await memory_system.emotional_manager.add_emotional_memory(
                entity_type="npc",
                entity_id=self.npc_id,
                memory_text=memory_text,
                primary_emotion=emotion_analysis.get("primary_emotion", "neutral"),
                emotion_intensity=emotion_analysis.get("intensity", 0.2),  # Usually low intensity for routines
                secondary_emotions=emotion_analysis.get("secondary_emotions", {}),
                significance=MemorySignificance.LOW,  # Low significance for routine activities
                tags=["routine", "scheduled", day_name.lower(), time_of_day.lower()]
            )
            
            memory_id = memory_result.get("memory_id")
            
            # Apply schemas to help categorize the routine
            if memory_id:
                await memory_system.integrated.apply_schema_to_memory(
                    memory_id=memory_id,
                    entity_type="npc",
                    entity_id=self.npc_id,
                    auto_detect=True
                )

            logger.debug("NPCAgent %s performed scheduled activity: %s => %s", 
                        self.npc_id, activity_desc, result)
            
            return {
                "npc_id": self.npc_id,
                "action": action,
                "result": result
            }
        except Exception as e:
            logger.error("Error in perform_scheduled_activity for NPC %s: %s", self.npc_id, e)
            return None
            
    async def run_memory_maintenance(self) -> Dict[str, Any]:
        """
        Run periodic maintenance tasks on the NPC's memory system.
        
        Returns:
            Results of maintenance operations
        """
        try:
            memory_system = await self._get_memory_system()
            
            # Run comprehensive memory maintenance
            return await memory_system.integrated.run_memory_maintenance(
                entity_type="npc",
                entity_id=self.npc_id,
                maintenance_options={
                    "core_maintenance": True,
                    "schema_maintenance": True,
                    "emotional_decay": True,
                    "background_reconsolidation": True,
                    "interference_processing": True,
                    "mask_checks": True
                }
            )
        except Exception as e:
            logger.error(f"Error running memory maintenance for NPC {self.npc_id}: {e}")
            return {"error": str(e)}

    async def get_beliefs_about_player(self) -> List[Dict[str, Any]]:
        """
        Get the NPC's beliefs about the player based on past interactions.
        Useful for generating dialog and response planning.
        
        Returns:
            List of beliefs with confidence levels
        """
        try:
            memory_system = await self._get_memory_system()
            beliefs = await memory_system.get_beliefs(
                entity_type="npc", 
                entity_id=self.npc_id,
                topic="player"
            )
            return beliefs
        except Exception as e:
            logger.error(f"Error getting beliefs about player for NPC {self.npc_id}: {e}")
            return []
            
    async def _fetch_relationships(self) -> Dict[str, Any]:
        """Get NPC's relationships with other entities."""
        relationships = {}
        
        with get_db_connection() as conn, conn.cursor() as cursor:
            # Query all links from NPC to other entities
            cursor.execute("""
                SELECT entity2_type, entity2_id, link_type, link_level
                FROM SocialLinks
                WHERE entity1_type = 'npc'
                  AND entity1_id = %s
                  AND user_id = %s
                  AND conversation_id = %s
            """, (self.npc_id, self.user_id, self.conversation_id))
            
            rows = cursor.fetchall()
            for entity_type, entity_id, link_type, link_level in rows:
                entity_name = "Unknown"
                
                if entity_type == "npc":
                    # Fetch NPC name
                    cursor.execute("""
                        SELECT npc_name
                        FROM NPCStats
                        WHERE npc_id = %s
                          AND user_id = %s
                          AND conversation_id = %s
                    """, (entity_id, self.user_id, self.conversation_id))
                    name_row = cursor.fetchone()
                    if name_row:
                        entity_name = name_row[0]
                elif entity_type == "player":
                    entity_name = "Player"
                
                relationships[entity_type] = {
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "link_type": link_type,
                    "link_level": link_level
                }
        
        return relationships

    # ------------------------------------------------------------------
    # Internal helper methods for scheduling/time (unchanged from original)
    # ------------------------------------------------------------------

    def _fetch_current_time_of_day(self, cursor) -> str:
        """
        Helper to fetch the current time of day (e.g. 'Morning') from DB.
        Fallback is 'Morning' if unavailable.
        """
        cursor.execute("""
            SELECT value
            FROM CurrentRoleplay
            WHERE user_id=%s
              AND conversation_id=%s
              AND key='TimeOfDay'
        """, (self.user_id, self.conversation_id))
        row = cursor.fetchone()
        return row[0] if row else "Morning"

    def _fetch_current_day_name(self, cursor) -> str:
        """
        Helper to find the current day name from DB, e.g. 'Monday'.
        Fallback is 'Monday' if not found.
        """
        # 1) fetch numeric current_day
        cursor.execute("""
            SELECT value
            FROM CurrentRoleplay
            WHERE user_id=%s
              AND conversation_id=%s
              AND key='CurrentDay'
        """, (self.user_id, self.conversation_id))
        row = cursor.fetchone()
        current_day_num = int(row[0]) if (row and str(row[0]).isdigit()) else 1

        # 2) fetch day names
        cursor.execute("""
            SELECT value
            FROM CurrentRoleplay
            WHERE user_id=%s
              AND conversation_id=%s
              AND key='CalendarNames'
        """, (self.user_id, self.conversation_id))
        row2 = cursor.fetchone()
        if row2 and row2[0]:
            try:
                calendar_data = json.loads(row2[0])
                day_names = calendar_data.get("days", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
            except Exception:
                day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        else:
            day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

        day_index = (current_day_num - 1) % len(day_names)
        return day_names[day_index]

    def _fetch_npc_schedule(self, cursor) -> Optional[Dict[str, Dict[str, str]]]:
        """
        Helper to load the schedule dict from NPCStats.schedule
        e.g. { "Monday": {"Morning":"desc", ...}, ... }
        """
        cursor.execute("""
            SELECT schedule
            FROM NPCStats
            WHERE npc_id=%s
              AND user_id=%s
              AND conversation_id=%s
        """, (self.npc_id, self.user_id, self.conversation_id))
        sched_row = cursor.fetchone()
        if not sched_row or not sched_row[0]:
            return None

        try:
            # schedule might be a string (JSON) or JSONB
            if isinstance(sched_row[0], str):
                return json.loads(sched_row[0])
            return sched_row[0]
        except Exception:
            logger.error("Invalid schedule data for NPC %s", self.npc_id)
            return None
