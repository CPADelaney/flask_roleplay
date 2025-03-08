# logic/npc_agents/npc_agent.py

"""
Core NPC agent class that manages individual NPC behavior with memory capabilities.
"""

import logging
import json
import asyncio
import random
from datetime import datetime, timedelta
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

    # Add this method:
    def _update_cache(self, cache_key, sub_key=None, value=None):
        """Update the cache with new value."""
        with self.lock:
            if sub_key is not None:
                if cache_key not in self._cache:
                    self._cache[cache_key] = {}
                self._cache[cache_key][sub_key] = value
                self._cache_timestamps[cache_key][sub_key] = datetime.now()
            else:
                self._cache[cache_key] = value
                self._cache_timestamps[cache_key] = datetime.now()

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
        """
        Enhanced environment perception with performance monitoring and emotional context.
        Integrates memories, emotional state, and mask information to create a rich perception.
        """
        start_time = datetime.now()
        
        try:
            # Fetch basic environment data
            environment_data = await fetch_environment_data(
                self.user_id,
                self.conversation_id,
                current_context
            )

            context_key = str(hash(json.dumps(current_context, sort_keys=True, default=str)))
            
            # Check if we have a valid cached perception (improved caching)
            if self.is_cache_valid('perception', context_key):
                self.last_perception = self._cache['perception'][context_key]
                self.perf_metrics['cache_hits'] = self.perf_metrics.get('cache_hits', 0) + 1
                return self.last_perception

            # Get memory system for enhanced perception
            memory_system = await self._get_memory_system()
            
            # Extract context description for memory retrieval
            context_description = current_context.get("description", "")
            if "text" in current_context:
                context_description += " " + current_context["text"]
                
            # Create enhanced context with time and location awareness
            context_for_recall = {
                "text": context_description,
                "location": environment_data.get("location", "Unknown"),
                "time_of_day": environment_data.get("time_of_day", "Unknown")
            }
            
            # Add entities present for better social context
            if "entities_present" in environment_data:
                context_for_recall["entities_present"] = [
                    e.get("name", "") for e in environment_data.get("entities_present", [])
                ]
            
            # Get current emotional state to influence memory retrieval (mood-congruent recall)
            emotional_state = await memory_system.get_npc_emotion(self.npc_id)
            self.current_emotional_state = emotional_state
            
            # Enhance memory retrieval with emotional state
            if emotional_state and "current_emotion" in emotional_state:
                current_emotion = emotional_state["current_emotion"]
                primary = current_emotion.get("primary", {})
                emotion_name = primary.get("name", "neutral")
                intensity = primary.get("intensity", 0.0)
                
                # Strong emotions bias memory retrieval (psychological realism)
                if intensity > 0.6:
                    context_for_recall["emotional_state"] = {
                        "primary_emotion": emotion_name,
                        "intensity": intensity
                    }
            
            # Retrieve relevant memories using enhanced context
            memory_result = await memory_system.recall(
                entity_type="npc",
                entity_id=self.npc_id,
                context=context_for_recall,
                limit=7  # More memories for richer context
            )
            relevant_memories = memory_result.get("memories", [])

            base_limit = 5  # Standard memory limit
            adaptive_limit = base_limit
            
            # Adjust based on context keywords
            context_importance = 0
            keywords = {
                "high": ["critical", "emergency", "dangerous", "threat", "crucial", "sex", "intimate"],
                "medium": ["important", "significant", "unusual", "strange", "unexpected"]
            }
            
            for word in keywords["high"]:
                if word in context_description.lower():
                    context_importance += 2
                    
            for word in keywords["medium"]:
                if word in context_description.lower():
                    context_importance += 1
            
            # Adjust recall limit based on context importance
            if context_importance >= 3:
                adaptive_limit = base_limit + 5  # Much more memories for critical contexts
            elif context_importance >= 1:
                adaptive_limit = base_limit + 2  # More memories for important contexts
            
            # Get memories with adaptive limit
            memory_result = await memory_system.recall(
                entity_type="npc",
                entity_id=self.npc_id,
                context=context_for_recall,
                limit=adaptive_limit
            )
            
            # Check for traumatic triggers in current context
            traumatic_trigger = None
            if context_description:
                trigger_result = await memory_system.emotional_manager.process_traumatic_triggers(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    text=context_description
                )
                
                if trigger_result and trigger_result.get("triggered", False):
                    traumatic_trigger = trigger_result
                    
                    # Traumatic triggers may update emotional state
                    if "emotional_response" in trigger_result:
                        response = trigger_result["emotional_response"]
                        # Update emotional state based on trigger
                        await memory_system.update_npc_emotion(
                            npc_id=self.npc_id,
                            emotion=response.get("primary_emotion", "fear"),
                            intensity=response.get("intensity", 0.7)
                        )
                        # Refresh emotional state
                        emotional_state = await memory_system.get_npc_emotion(self.npc_id)
                        self.current_emotional_state = emotional_state
            
            # Check for flashback potential (more likely with traumatic triggers)
            flashback = None
            flashback_chance = 0.15  # Base chance
            if traumatic_trigger:
                flashback_chance = 0.5  # Higher chance with triggers
                
            if random.random() < flashback_chance:
                flashback = await memory_system.npc_flashback(
                    npc_id=self.npc_id, 
                    context=context_description
                )
            
            # Get mask information with consistency checks
            mask_info = await memory_system.get_npc_mask(self.npc_id)
            
            # Ensure mask has integrity property
            if mask_info and "integrity" not in mask_info:
                mask_info["integrity"] = 100  # Default to perfect mask
            
            # Get relationship data with enhanced memory context
            relationship_data = await self._fetch_relationships_with_memory()
            
            # Get beliefs that might influence perception, ordered by confidence
            beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=self.npc_id,
                topic="player"  # Focus on player-related beliefs
            )
            
            # Order beliefs by confidence
            if beliefs:
                beliefs = sorted(beliefs, key=lambda x: x.get("confidence", 0), reverse=True)
            
            # Combine into enhanced perception dictionary
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
            
            # Calculate perception complexity metrics for monitoring
            perception_complexity = {
                "memory_count": len(relevant_memories),
                "relationship_count": len(relationship_data),
                "has_flashback": flashback is not None,
                "has_traumatic_trigger": traumatic_trigger is not None,
                "belief_count": len(beliefs)
            }
            perception["complexity_metrics"] = perception_complexity

            # Update perception cache
            self._update_cache('perception', context_key, perception)
            
            # Store as last perception
            self.last_perception = perception
            
            # Record performance metric
            elapsed = (datetime.now() - start_time).total_seconds()
            self.perf_metrics['perception_time'].append(elapsed)
            self.perf_metrics['perception_complexity'] = perception_complexity
            
            # Log slow operations
            if elapsed > 0.5:  # Log if taking more than 500ms
                logger.warning(f"Slow perception for NPC {self.npc_id}: {elapsed:.2f}s")
            
            return perception
            
        except Exception as e:
            # Record failure in metrics
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"Perception error for NPC {self.npc_id} after {elapsed:.2f}s: {e}")
            
            # Return minimal fallback perception
            return {
                "environment": environment_data if 'environment_data' in locals() else {},
                "relevant_memories": [],
                "relationships": {},
                "emotional_state": {"current_emotion": {"primary": {"name": "neutral", "intensity": 0.0}}},
                "mask": {"integrity": 100},
                "beliefs": [],
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def _fetch_relationships_with_memory(self) -> Dict[str, Any]:
        """Get NPC's relationships enhanced with memory references."""
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
        
        # Enhance relationships with memory references (if memory system available)
        try:
            memory_system = await self._get_memory_system()
            
            for entity_type, rel_data in relationships.items():
                entity_id = rel_data["entity_id"]
                entity_name = rel_data["entity_name"]
                
                # Get significant memories about this entity
                query = entity_name
                
                memory_result = await memory_system.recall(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    query=query,
                    limit=3
                )
                
                # Add memory context to relationship
                rel_data["memory_context"] = memory_result.get("memories", [])
        except Exception as e:
            logger.warning(f"Could not enhance relationships with memories: {e}")
        
        return relationships
    
    async def make_decision(self,
                           perception: Optional[Dict[str, Any]] = None,
                           available_actions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Decide on action with enhanced memory-driven decision making and mask system.
        Features psychological realism through emotional state, beliefs, and mask dynamics.
        """
        start_time = datetime.now()
        
        try:
            if perception is None:
                # Use the last perception or fetch a fresh one
                if self.last_perception is None:
                    perception = await self.perceive_environment({})
                else:
                    perception = self.last_perception

            # Get memory system
            memory_system = await self._get_memory_system()
            
            # Get beliefs to influence decisions
            beliefs = perception.get("beliefs", [])
            
            # Apply belief weights to available actions
            if available_actions and beliefs:
                weighted_actions = await self._apply_belief_weights_to_actions(available_actions, beliefs)
                available_actions = weighted_actions
                
            # Check for traumatic triggers that should influence decisions
            traumatic_trigger = perception.get("traumatic_trigger")
            if traumatic_trigger and traumatic_trigger.get("triggered", False):
                # Create a trauma-response action
                trauma_response = self._create_trauma_response_action(traumatic_trigger)
                if trauma_response:
                    # Add as high-priority action
                    if available_actions:
                        available_actions.insert(0, trauma_response)
                    else:
                        available_actions = [trauma_response]
            
            # Apply emotional state influences with psychological realism
            emotional_state = perception.get("emotional_state", {})
            current_emotion = emotional_state.get("current_emotion", {})
            
            if current_emotion:
                primary_emotion = current_emotion.get("primary", {})
                emotion_name = primary_emotion.get("name", "neutral")
                intensity = primary_emotion.get("intensity", 0.0)
                
                # Strong emotions can override normal decision-making
                if intensity > 0.7:
                    # Different emotions create different action biases
                    if emotion_name == "anger":
                        emotional_action = {
                            "type": "emotional_outburst",
                            "description": "express anger forcefully",
                            "target": "group",
                            "weight": 1.7 * intensity,
                            "stats_influenced": {"respect": -10}
                        }
                        if available_actions:
                            available_actions.insert(0, emotional_action)
                        else:
                            available_actions = [emotional_action]
                            
                    elif emotion_name == "fear":
                        # Fear typically leads to defensive/avoidant behavior
                        fear_action = {
                            "type": "emotional_response",
                            "description": "show fear and defensiveness",
                            "target": "group",
                            "weight": 1.6 * intensity,
                            "stats_influenced": {"trust": -5}
                        }
                        if available_actions:
                            available_actions.insert(0, fear_action)
                        else:
                            available_actions = [fear_action]
                            
                    elif emotion_name == "joy":
                        # Joy leads to expressive behavior
                        joy_action = {
                            "type": "emotional_expression",
                            "description": "express happiness enthusiastically",
                            "target": "group",
                            "weight": 1.2 * intensity,
                            "stats_influenced": {"trust": 5}
                        }
                        if available_actions:
                            available_actions.insert(0, joy_action)
                        else:
                            available_actions = [joy_action]

            # Pass the enhanced perception to the decision engine
            chosen_action = await self.decision_engine.decide(perception, available_actions)
            logger.debug("NPCAgent %s decided on action: %s", self.npc_id, chosen_action)
            
            # Check for mask slippage based on complex factors
            should_slip = await self._should_mask_slip(perception, chosen_action)
            
            if should_slip:
                # Generate mask slippage
                slip_trigger = f"deciding to {chosen_action.get('description', 'act')}"
                
                # Emotional triggers are more likely to cause slips
                if current_emotion and intensity > 0.5:
                    slip_trigger = f"feeling {emotion_name} while " + slip_trigger
                    
                slip_result = await memory_system.reveal_npc_trait(
                    npc_id=self.npc_id,
                    trigger=slip_trigger
                )
                
                # Add mask slippage information to action
                chosen_action["mask_slippage"] = slip_result
                
                # Create memory of the slippage
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    memory_text=f"My mask slipped while I was {chosen_action.get('description', 'doing something')}",
                    importance="medium",
                    tags=["mask_slip", "self_awareness"]
                )
            
            # Remember this decision for future reference (recent decision history)
            self._record_decision_history(chosen_action)
            
            # Record performance data
            elapsed = (datetime.now() - start_time).total_seconds()
            self.perf_metrics['decision_time'].append(elapsed)
            
            return chosen_action
            
        except Exception as e:
            # Record failure and return safe fallback
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"Decision error for NPC {self.npc_id} after {elapsed:.2f}s: {e}")
            
            # Return a safe fallback action
            return {"type": "observe", "description": "observe quietly", "target": "environment"}

    async def _evolve_personality_traits(self) -> Dict[str, Any]:
        """Gradually evolve personality traits based on experiences and interactions."""
        results = {
            "traits_modified": 0,
            "new_traits": 0,
            "removed_traits": 0
        }
        
        # Get current traits
        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
                SELECT personality_traits, dominance, cruelty
                FROM NPCStats
                WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
            """, (self.npc_id, self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            if not row:
                return {"error": "NPC not found"}
                
            traits_json, dominance, cruelty = row
            
            # Parse traits
            if isinstance(traits_json, str):
                try:
                    traits = json.loads(traits_json)
                except:
                    traits = []
            else:
                traits = traits_json or []
        
        # Get memory system for analyzing experiences
        memory_system = await self._get_memory_system()
        
        # Analyze recent significant memories for trait influences
        memory_result = await memory_system.recall(
            entity_type="npc",
            entity_id=self.npc_id,
            query="significant experience",
            limit=10
        )
        
        memories = memory_result.get("memories", [])
        
        # Count trait-influencing experiences
        trait_influences = {}
        for memory in memories:
            text = memory.get("text", "").lower()
            
            # Check for positive experiences with dominance
            if any(word in text for word in ["commanded", "controlled", "dominated", "power"]):
                trait_influences["dominant"] = trait_influences.get("dominant", 0) + 1
                
            # Check for submission experiences
            if any(word in text for word in ["obeyed", "submitted", "followed", "complied"]):
                trait_influences["submissive"] = trait_influences.get("submissive", 0) + 1
                
            # Check for cruel experiences
            if any(word in text for word in ["mocked", "humiliated", "hurt", "cruel"]):
                trait_influences["cruel"] = trait_influences.get("cruel", 0) + 1
                
            # Check for kind experiences
            if any(word in text for word in ["helped", "supported", "kind", "gentle"]):
                trait_influences["kind"] = trait_influences.get("kind", 0) + 1
        
        # Modify traits based on influences
        modified_traits = traits.copy()
        
        # Remove conflicting traits if strong influence in opposite direction
        if "dominant" in trait_influences and trait_influences["dominant"] >= 3 and "submissive" in modified_traits:
            modified_traits.remove("submissive")
            results["removed_traits"] += 1
            
        if "submissive" in trait_influences and trait_influences["submissive"] >= 3 and "dominant" in modified_traits:
            modified_traits.remove("dominant")
            results["removed_traits"] += 1
            
        if "cruel" in trait_influences and trait_influences["cruel"] >= 3 and "kind" in modified_traits:
            modified_traits.remove("kind")
            results["removed_traits"] += 1
            
        if "kind" in trait_influences and trait_influences["kind"] >= 3 and "cruel" in modified_traits:
            modified_traits.remove("cruel")
            results["removed_traits"] += 1
        
        # Add new traits if strong influence and not already present
        for trait, count in trait_influences.items():
            if count >= 3 and trait not in modified_traits:
                modified_traits.append(trait)
                results["new_traits"] += 1
        
        # Update dominance and cruelty stats based on experiences
        new_dominance = dominance
        new_cruelty = cruelty
        
        if "dominant" in trait_influences:
            new_dominance = min(100, dominance + trait_influences["dominant"])
            results["traits_modified"] += 1
            
        if "submissive" in trait_influences:
            new_dominance = max(0, dominance - trait_influences["submissive"])
            results["traits_modified"] += 1
            
        if "cruel" in trait_influences:
            new_cruelty = min(100, cruelty + trait_influences["cruel"])
            results["traits_modified"] += 1
            
        if "kind" in trait_influences:
            new_cruelty = max(0, cruelty - trait_influences["kind"])
            results["traits_modified"] += 1
        
        # Save changes if any were made
        if (modified_traits != traits or 
            new_dominance != dominance or 
            new_cruelty != cruelty):
            
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE NPCStats
                    SET personality_traits = %s,
                        dominance = %s,
                        cruelty = %s
                    WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                """, (json.dumps(modified_traits), new_dominance, new_cruelty, 
                      self.npc_id, self.user_id, self.conversation_id))
        
        return results
    
    async def _should_mask_slip(self, perception, chosen_action):
        """
        Determine if mask should slip based on complex psychological factors.
        Enhanced with consistent behavior tracking for mask evolution.
        """
        # Get mask info
        mask_info = perception.get("mask", {})
        mask_integrity = mask_info.get("integrity", 100)
        
        # Base probability based on mask integrity
        base_probability = (100 - mask_integrity) / 200  # 0-0.5 range
        
        # No chance if mask already broken or perfectly intact
        if mask_integrity <= 0 or mask_integrity >= 100:
            return False
        
        # Get NPC stats for personality factors
        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
                SELECT dominance, cruelty, self_control
                FROM NPCStats
                WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
            """, (self.npc_id, self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            if not row:
                return False
                
            dominance, cruelty = row[0], row[1]
            # Default self_control if not found
            self_control = row[2] if len(row) > 2 else 50
        
        # Get consistent behavior patterns (NEW)
        behavior_patterns = await self._analyze_recent_behavior_patterns()
        
        # Adjust slip chance based on behavioral consistency
        consistency_modifier = 0.0
        if behavior_patterns.get("true_nature_acting", 0) > 3:
            # If consistently acting according to true nature, mask starts to align
            consistency_modifier -= 0.2  # Reduces slip chance
        elif behavior_patterns.get("mask_reinforcing", 0) > 3:
            # If actively reinforcing mask, less likely to slip
            consistency_modifier -= 0.3
        
        # Personality modifiers
        personality_factor = 0.0
        
        # Higher dominance/cruelty increase slip chance
        personality_factor += (dominance / 200)  # 0-0.5 range 
        personality_factor += (cruelty / 200)    # 0-0.5 range
        
        # Self-control decreases slip chance
        personality_factor -= (self_control / 200)  # 0-0.5 range
        
        # Emotional state modifiers
        emotional_state = perception.get("emotional_state", {})
        emotion_factor = 0.0
        
        if emotional_state and "current_emotion" in emotional_state:
            current_emotion = emotional_state["current_emotion"]
            emotion_name = current_emotion.get("primary", {}).get("name", "neutral")
            intensity = current_emotion.get("primary", {}).get("intensity", 0.0)
            
            # Strong emotions increase slip chance
            if intensity > 0.5:
                # Different emotions have different effects
                if emotion_name == "anger":
                    emotion_factor += intensity * 0.4  # Anger causes more slips
                elif emotion_name == "fear":
                    emotion_factor += intensity * 0.3  # Fear causes some slips
                elif emotion_name == "joy":
                    emotion_factor += intensity * 0.1  # Joy causes few slips
        
        # Action-specific modifiers
        action_factor = 0.0
        action_type = chosen_action.get("type", "")
        
        # Certain action types are more likely to cause slips
        if action_type in ["dominate", "command", "punish", "mock"]:
            action_factor += 0.2
        elif action_type in ["emotional_outburst", "express_anger"]:
            action_factor += 0.3
        elif action_type in ["mask_reinforcement", "self_control"]:
            action_factor -= 0.4  # Actively maintaining mask
        
        # Context modifiers
        context_factor = 0.0
        
        # Traumatic triggers increase slip chance significantly
        if perception.get("traumatic_trigger"):
            context_factor += 0.4
        
        # Flashbacks increase slip chance
        if perception.get("flashback"):
            context_factor += 0.3
        
        # Calculate final probability
        total_probability = (
            base_probability + 
            personality_factor + 
            emotion_factor + 
            action_factor +
            context_factor +
            consistency_modifier  # NEW - behavior consistency impact
        )
        
        # Cap probability at 95%
        final_probability = min(0.95, max(0, total_probability))
        
        # Make the roll
        return random.random() < final_probability
    
    async def _analyze_recent_behavior_patterns(self):
        """
        NEW: Analyze patterns in recent behavior to determine consistency 
        with true nature vs. mask presentation.
        """
        patterns = {
            "true_nature_acting": 0,  # Acts according to true nature
            "mask_reinforcing": 0,    # Actively reinforces mask
            "mixed_signals": 0        # Inconsistent behavior
        }
        
        # Check past decisions
        if hasattr(self, 'decision_history') and len(self.decision_history) >= 5:
            # Get mask info to determine true nature vs. presented traits
            memory_system = await self._get_memory_system()
            mask_info = await memory_system.get_npc_mask(self.npc_id)
            
            if not mask_info:
                return patterns
                
            hidden_traits = mask_info.get("hidden_traits", {})
            presented_traits = mask_info.get("presented_traits", {})
            
            # Analyze last 5 decisions
            recent_actions = [d["action"] for d in self.decision_history[-5:]]
            
            for action in recent_actions:
                action_type = action.get("type", "")
                
                # Check if action aligns with hidden (true) traits
                true_nature_alignment = 0
                if "dominant" in hidden_traits and action_type in ["command", "dominate", "test"]:
                    true_nature_alignment += 1
                if "cruel" in hidden_traits and action_type in ["mock", "humiliate", "punish"]:
                    true_nature_alignment += 1
                if "sadistic" in hidden_traits and action_type in ["punish", "humiliate"]:
                    true_nature_alignment += 1
                    
                # Check if action aligns with presented traits
                mask_alignment = 0
                if "kind" in presented_traits and action_type in ["praise", "support", "help"]:
                    mask_alignment += 1
                if "gentle" in presented_traits and action_type in ["talk", "observe", "support"]:
                    mask_alignment += 1
                if "submissive" in presented_traits and action_type in ["observe", "wait", "obey"]:
                    mask_alignment += 1
                    
                # Determine primary alignment of this action
                if true_nature_alignment > mask_alignment:
                    patterns["true_nature_acting"] += 1
                elif mask_alignment > true_nature_alignment:
                    patterns["mask_reinforcing"] += 1
                elif mask_alignment > 0 and true_nature_alignment > 0:
                    patterns["mixed_signals"] += 1
        
        return patterns
    
    async def apply_mask_evolution(self):
        """
        NEW: Gradually evolve mask based on consistent behavior.
        A mask that is consistently maintained or consistently broken will
        eventually align with behavior.
        """
        # Get memory system
        memory_system = await self._get_memory_system()
        
        # Get current mask info
        mask_info = await memory_system.get_npc_mask(self.npc_id)
        if not mask_info:
            return {"status": "no_mask"}
            
        # Get current integrity
        integrity = mask_info.get("integrity", 100)
        
        # Analyze recent behavior patterns
        patterns = await self._analyze_recent_behavior_patterns()
        
        # Calculate adjustment based on patterns
        adjustment = 0
        
        # If consistently acting according to true nature, mask weakens
        if patterns["true_nature_acting"] >= 4:
            adjustment -= 5
        elif patterns["true_nature_acting"] >= 2:
            adjustment -= 2
            
        # If actively reinforcing mask, integrity improves
        if patterns["mask_reinforcing"] >= 4:
            adjustment += 3
        elif patterns["mask_reinforcing"] >= 2:
            adjustment += 1
            
        # Mixed signals create strain
        if patterns["mixed_signals"] >= 3:
            adjustment -= 2
            
        # No change if behavior isn't consistent
        if adjustment == 0:
            return {"status": "no_change"}
            
        # Apply the adjustment
        new_integrity = max(0, min(100, integrity + adjustment))
        
        # If significant change, update mask
        if abs(new_integrity - integrity) >= 2:
            trigger = "consistent behavior patterns" if adjustment < 0 else "active mask reinforcement"
            severity = min(3, abs(adjustment) // 2)  # 1-3 severity based on adjustment size
            
            # Use existing mask system to apply the change
            result = await memory_system.reveal_npc_trait(
                npc_id=self.npc_id,
                trigger=trigger,
                severity=severity if adjustment < 0 else 0  # Only positive severity for slips
            )
            
            return {
                "status": "updated",
                "old_integrity": integrity,
                "new_integrity": new_integrity,
                "adjustment": adjustment,
                "result": result
            }
            
        return {"status": "no_significant_change"}
    
    def _record_decision_history(self, action):
        """Record the decision in history for pattern analysis."""
        if not hasattr(self, 'decision_history'):
            self.decision_history = []
            
        # Add this decision to history
        self.decision_history.append({
            "action": action,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only the last 10 decisions
        if len(self.decision_history) > 10:
            self.decision_history = self.decision_history[-10:]

    async def _check_for_mask_reinforcement_behaviors(self) -> float:
        """
        Check for behaviors that would help reinforce an NPC's mask.
        Returns a reinforcement score (0 = none, higher = more reinforcement).
        """
        reinforcement_score = 0.0
        
        # Check recent actions from decision history
        if hasattr(self, 'decision_history') and self.decision_history:
            # Get most recent actions (up to 5)
            recent_actions = self.decision_history[-5:]
            
            for decision in recent_actions:
                action = decision.get("action", {})
                action_type = action.get("type", "")
                
                # Actions that reinforce mask
                if action_type in ["observe", "talk"]:
                    reinforcement_score += 0.2  # Mild reinforcement
                elif action_type in ["leave", "act_defensive"]:
                    reinforcement_score += 0.3  # Moderate reinforcement
                elif action_type == "mask_reinforcement":
                    reinforcement_score += 1.0  # Strong reinforcement
        
        # Check if NPC is alone (easier to reinforce mask when alone)
        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
                SELECT current_location 
                FROM NPCStats 
                WHERE npc_id = %s
            """, (self.npc_id,))
            location = cursor.fetchone()[0] if cursor.rowcount > 0 else None
            
            if location:
                # Check if others are present
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM NPCStats 
                    WHERE current_location = %s AND npc_id != %s
                """, (location, self.npc_id))
                other_count = cursor.fetchone()[0] if cursor.rowcount > 0 else 0
                
                if other_count == 0:
                    # Alone - easier to reinforce
                    reinforcement_score += 0.5
        
        return reinforcement_score
    
    async def _apply_belief_weights_to_actions(self, 
                                              available_actions: List[Dict[str, Any]], 
                                              beliefs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply belief weights to actions to influence decision-making with psychological realism.
        Beliefs about player or environment influence action selection.
        """
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
                
                # Relevance score - how relevant is this belief to this action?
                relevance = 0.0
                
                # Check action type keywords in belief
                if action_type in belief_text:
                    relevance += 0.5
                
                # Check target keywords in belief
                if target in belief_text:
                    relevance += 0.5
                
                # Specific belief-action mappings
                if target == "player" or target == "group":
                    # Trust beliefs
                    if "trust" in belief_text or "like" in belief_text or "friend" in belief_text:
                        if action_type in ["talk", "confide", "praise"]:
                            relevance += 0.8
                        elif action_type in ["mock", "leave", "attack"]:
                            relevance -= 0.8
                    
                    # Distrust beliefs
                    if "distrust" in belief_text or "fear" in belief_text or "threat" in belief_text:
                        if action_type in ["leave", "observe", "act_defensive"]:
                            relevance += 0.8
                        elif action_type in ["talk", "confide", "praise"]:
                            relevance -= 0.6
                    
                    # Dominance/submission beliefs
                    if "submissive" in belief_text or "obedient" in belief_text:
                        if action_type in ["command", "dominate", "test"]:
                            relevance += 0.9
                    
                    # Threat beliefs
                    if "dangerous" in belief_text or "threat" in belief_text:
                        if action_type in ["leave", "act_defensive"]:
                            relevance += 1.0
                        elif action_type in ["confide", "praise"]:
                            relevance -= 0.9
                
                # Apply relevance and confidence to weight
                belief_weight += relevance * confidence
            
            # Apply the calculated weight to the action
            if "weight" in action:
                action["weight"] *= belief_weight
            else:
                action["weight"] = belief_weight
                
            # Add belief information for introspection
            if "decision_metadata" not in action:
                action["decision_metadata"] = {}
            action["decision_metadata"]["belief_weight"] = belief_weight
            
        return available_actions

    def _create_trauma_response_action(self, trauma_trigger: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a psychologically realistic action in response to a traumatic trigger.
        Features differentiated responses based on trauma type and emotional response.
        """
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
                "description": f"respond emotionally to a triggering memory",
                "target": "self",
                "weight": 1.5 * intensity,
                "stats_influenced": {},
                "trauma_trigger": trigger_text
            }

    async def process_player_action(self, player_action: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a player's action with enhanced memory formation, psychological realism,
        and better mask integration. Creates more nuanced reactions and memories.
        """
        # Start performance timer
        start_time = datetime.now()
        
        try:
            # Create context dictionary if not provided
            context_obj = context or {}
            
            # Incorporate the player's action into the environment context
            perception_context = {
                "player_action": player_action,
                "text": player_action.get("description", ""),
                "description": f"Player {player_action.get('description', 'did something')}"
            }
            
            # Merge with provided context
            perception_context.update(context_obj)

            # Step 1: Refresh perception with player action context
            perception = await self.perceive_environment(perception_context)
            
            # Step 2: Remember the player's action with emotional context
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
            
            # Create memory with emotional context
            memory_result = await memory_system.emotional_manager.add_emotional_memory(
                entity_type="npc",
                entity_id=self.npc_id,
                memory_text=player_memory_text,
                primary_emotion=emotion_analysis.get("primary_emotion", "neutral"),
                emotion_intensity=emotion_analysis.get("intensity", 0.5),
                secondary_emotions=emotion_analysis.get("secondary_emotions", {}),
                significance=MemorySignificance.MEDIUM,
                tags=memory_kwargs["tags"]
            )
            
            memory_id = memory_result.get("memory_id")
            
            # Process schema for the memory
            if memory_id:
                await memory_system.integrated.apply_schema_to_memory(
                    memory_id=memory_id,
                    entity_type="npc",
                    entity_id=self.npc_id,
                    auto_detect=True
                )
            
            # Step 3: Update emotional state based on player action
            # More nuanced emotional response based on action type and NPC state
            player_action_type = player_action.get("type", "").lower()
            
            # Get mask for psychological realism in emotional responses
            mask_info = perception.get("mask", {})
            mask_integrity = mask_info.get("integrity", 100)
            hidden_traits = mask_info.get("hidden_traits", {})
            presented_traits = mask_info.get("presented_traits", {})
            
            # Different action types trigger different emotional responses
            if player_action_type in ["attack", "threaten", "insult"]:
                # Negative actions typically trigger fear or anger
                if mask_integrity < 50 and "dominant" in hidden_traits:
                    # Low mask integrity - dominant NPCs show anger when attacked
                    await memory_system.update_npc_emotion(self.npc_id, "anger", 0.8)
                else:
                    # Default response is fear
                    await memory_system.update_npc_emotion(self.npc_id, "fear", 0.7)
                    
            elif player_action_type in ["praise", "help", "gift"]:
                # Positive actions typically trigger joy or gratitude
                if mask_integrity < 50 and "suspicious" in hidden_traits:
                    # Suspicious NPCs might feel uncertainty even with praise
                    await memory_system.update_npc_emotion(self.npc_id, "uncertainty", 0.5)
                else:
                    # Default response is joy
                    await memory_system.update_npc_emotion(self.npc_id, "joy", 0.6)
                    
            elif player_action_type in ["command", "dominate"]:
                # Commands trigger compliance or defiance based on personality
                if mask_integrity < 50 and ("dominant" in hidden_traits or "controlling" in hidden_traits):
                    # Dominant NPCs resent being commanded
                    await memory_system.update_npc_emotion(self.npc_id, "anger", 0.7)
                    
                    # Higher chance of mask slippage when commanded
                    if random.random() < 0.3:
                        await memory_system.reveal_npc_trait(
                            npc_id=self.npc_id,
                            trigger=f"being commanded by player to {player_action.get('description', 'do something')}"
                        )
                else:
                    # More submissive NPCs feel intimidation
                    await memory_system.update_npc_emotion(self.npc_id, "submission", 0.6)
            
            # Step 4: Decide how to respond - decisions influenced by emotional state
            response_action = await self.make_decision(perception)

            # Step 5: Execute the chosen response
            result = await self.execute_action(response_action, perception_context)

            # Step 6: Remember the interaction with detailed context
            interaction_memory_text = (
                f"When the player {player_action.get('description','did something')}, "
                f"I responded by {response_action.get('description','doing something')}"
            )
            
            # Get emotional analysis of the interaction
            interaction_emotion_analysis = await memory_system.emotional_manager.analyze_emotional_content(
                interaction_memory_text
            )
            
            # Create memory of the interaction
            interaction_memory = await memory_system.emotional_manager.add_emotional_memory(
                entity_type="npc",
                entity_id=self.npc_id,
                memory_text=interaction_memory_text,
                primary_emotion=interaction_emotion_analysis.get("primary_emotion", "neutral"),
                emotion_intensity=interaction_emotion_analysis.get("intensity", 0.5),
                secondary_emotions=interaction_emotion_analysis.get("secondary_emotions", {}),
                significance=MemorySignificance.MEDIUM,
                tags=["interaction", "player", player_action.get("type", "unknown")]
            )
            
            # Step 7: Update beliefs based on interaction with more nuance
            await self._update_beliefs_from_interaction(player_action, response_action, result)
            
            # Step 8: Update relationships based on interaction outcome
            await self._update_relationship_from_interaction(
                "player", 
                self.user_id, 
                player_action, 
                response_action, 
                result
            )

            logger.debug("NPCAgent %s processed player action '%s': result=%s", 
                        self.npc_id, player_action, result)
            
            # Record performance data
            elapsed = (datetime.now() - start_time).total_seconds()
            self.perf_metrics['action_processing_time'] = elapsed

            return {
                "npc_id": self.npc_id,
                "action": response_action,
                "result": result,
                "processing_time_ms": int(elapsed * 1000)
            }
            
        except Exception as e:
            # Record failure and return safe fallback
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error processing player action for NPC {self.npc_id} after {elapsed:.2f}s: {e}")
            
            # Return minimal response with error info
            return {
                "npc_id": self.npc_id,
                "action": {"type": "error", "description": "had an internal error"},
                "result": {"outcome": "NPC seems confused", "emotional_impact": -1},
                "error": str(e)
            }
    
    async def _update_beliefs_from_interaction(self, 
                                            player_action: Dict[str, Any], 
                                            npc_action: Dict[str, Any],
                                            result: Dict[str, Any]) -> None:
        """
        Update beliefs based on interaction with more nuanced psychological modeling.
        Features consistency checking, context awareness, and personality influence.
        """
        memory_system = await self._get_memory_system()
        
        # Get current beliefs about the player
        current_beliefs = await memory_system.get_beliefs(
            entity_type="npc",
            entity_id=self.npc_id,
            topic="player"
        )
        
        # Determine interaction significance
        player_action_type = player_action.get("type", "")
        outcome = result.get("outcome", "")
        emotional_impact = result.get("emotional_impact", 0)
        
        # Categories of player actions that influence beliefs
        positive_actions = ["help", "gift", "praise", "support", "protect"]
        negative_actions = ["attack", "threaten", "mock", "betray", "insult"]
        neutral_actions = ["talk", "observe", "wait", "leave"]
        submission_actions = ["obey", "submit", "comply"]
        defiance_actions = ["defy", "resist", "disobey"]
        
        # Determine action category
        action_category = "neutral"
        if player_action_type in positive_actions:
            action_category = "positive"
        elif player_action_type in negative_actions:
            action_category = "negative"
        elif player_action_type in submission_actions:
            action_category = "submission"
        elif player_action_type in defiance_actions:
            action_category = "defiance"
        
        # Low impact for neutral actions
        if action_category == "neutral" and abs(emotional_impact) < 3:
            return  # Skip belief update for low-impact neutral actions
        
        # Get personality traits for belief formation influence
        personality_traits = {}
        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
                SELECT dominance, cruelty, personality_traits
                FROM NPCStats
                WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
            """, (self.npc_id, self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            if row:
                dominance, cruelty = row[0], row[1]
                personality_traits = row[2] if len(row) > 2 and row[2] else {}
                
                if isinstance(personality_traits, str):
                    try:
                        personality_traits = json.loads(personality_traits)
                    except:
                        personality_traits = {}
        
        # Default belief patterns
        belief_patterns = {
            "positive": {
                "text": "The player is helpful and supportive.",
                "confidence_change": 0.1,
                "new_confidence": 0.65
            },
            "negative": {
                "text": "The player is a potential threat.",
                "confidence_change": 0.15,
                "new_confidence": 0.7
            },
            "submission": {
                "text": "The player will follow my commands.",
                "confidence_change": 0.1,
                "new_confidence": 0.6
            },
            "defiance": {
                "text": "The player is rebellious and defiant.",
                "confidence_change": 0.12,
                "new_confidence": 0.65
            }
        }
        
        # Modify belief patterns based on personality traits
        if "suspicious" in personality_traits:
            # Suspicious NPCs have lower confidence in positive beliefs
            belief_patterns["positive"]["confidence_change"] = 0.05
            belief_patterns["positive"]["new_confidence"] = 0.5
            
        if "trusting" in personality_traits:
            # Trusting NPCs form positive beliefs more readily
            belief_patterns["positive"]["confidence_change"] = 0.15
            belief_patterns["positive"]["new_confidence"] = 0.75
            
        if "dominant" in personality_traits and dominance > 70:
            # Highly dominant NPCs react strongly to defiance
            belief_patterns["defiance"]["confidence_change"] = 0.2
            belief_patterns["defiance"]["new_confidence"] = 0.8
        
        # Get the appropriate belief pattern
        pattern = belief_patterns.get(action_category)
        if not pattern:
            return
            
        belief_text = pattern["text"]
        confidence_change = pattern["confidence_change"]
        new_confidence = pattern["new_confidence"]
        
        # Check for existing similar beliefs
        existing_belief = None
        for belief in current_beliefs:
            belief_text_lower = belief.get("belief", "").lower()
            pattern_text_lower = belief_text.lower()
            
            # Simple text similarity check - share at least 2 significant words
            if len(set(belief_text_lower.split()) & set(pattern_text_lower.split())) >= 2:
                existing_belief = belief
                break
        
        # Apply emotional impact modifier
        if abs(emotional_impact) > 3:
            confidence_change *= 1.5
            
        if existing_belief:
            # Update existing belief confidence
            old_confidence = existing_belief.get("confidence", 0.5)
            adjusted_confidence = min(0.95, old_confidence + confidence_change)
            
            await memory_system.semantic_manager.update_belief_confidence(
                belief_id=existing_belief["id"],
                entity_type="npc",
                entity_id=self.npc_id,
                new_confidence=adjusted_confidence,
                reason=f"Based on player's {player_action_type} action"
            )
        else:
            # Create new belief
            await memory_system.create_belief(
                entity_type="npc",
                entity_id=self.npc_id,
                belief_text=belief_text,
                confidence=new_confidence
            )
        
        # For significant interactions, potentially create a counter-belief
        # This models cognitive dissonance and ambivalence in beliefs
        if (action_category in ["positive", "negative"] and 
            abs(emotional_impact) > 4 and 
            random.random() < 0.3):  # 30% chance of counter-belief
            
            counter_category = "negative" if action_category == "positive" else "positive"
            counter_pattern = belief_patterns.get(counter_category)
            
            if counter_pattern:
                # Create counter-belief with lower confidence
                counter_confidence = counter_pattern["new_confidence"] * 0.7
                
                await memory_system.create_belief(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    belief_text=f"Despite appearances, {counter_pattern['text'].lower()}",
                    confidence=counter_confidence
                )
    
    async def _update_relationship_from_interaction(
        self,
        entity_type: str,
        entity_id: int,
        player_action: Dict[str, Any],
        npc_response: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
        """
        Update relationship based on interaction with psychological depth.
        Features emotional impact, belief consistency, and nuanced changes.
        """
        try:
            # Get relationship manager
            from .relationship_manager import NPCRelationshipManager
            relationship_manager = NPCRelationshipManager(self.npc_id, self.user_id, self.conversation_id)
            
            # Enhance context with memory and emotional state
            memory_system = await self._get_memory_system()
            
            # Get emotional state
            emotional_state = await memory_system.get_npc_emotion(self.npc_id)
            
            # Get beliefs about entity
            topic = "player" if entity_type == "player" else f"npc_{entity_id}"
            beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=self.npc_id,
                topic=topic
            )
            
            # Get recent memories about this entity
            memory_result = await memory_system.recall(
                entity_type="npc",
                entity_id=self.npc_id,
                query=entity_type,
                limit=3
            )
            
            # Prepare enhanced context
            enhanced_context = {
                "emotional_state": emotional_state,
                "beliefs": beliefs,
                "recent_memories": memory_result.get("memories", []),
                "result": result
            }
            
            # Call relationship manager to update the relationship
            await relationship_manager.update_relationship_from_interaction(
                entity_type,
                entity_id,
                player_action,
                npc_response,
                enhanced_context
            )
            
        except Exception as e:
            logger.error(f"Error updating relationship: {e}")

    async def run_memory_maintenance(self) -> Dict[str, Any]:
        """
        Run memory maintenance with sophisticated lifecycle management.
        Features consolidation, emotional decay, archiving, and schema maintenance.
        """
        try:
            memory_system = await self._get_memory_system()
            
            # Run comprehensive memory maintenance
            maintenance_result = await memory_system.integrated.run_memory_maintenance(
                entity_type="npc",
                entity_id=self.npc_id,
                maintenance_options={
                    "core_maintenance": True,         # Basic cleanup and integrity checks
                    "schema_maintenance": True,       # Update and apply schemas
                    "emotional_decay": True,          # Natural emotional fading
                    "memory_consolidation": True,     # Combine similar memories
                    "background_reconsolidation": True,  # Subtle memory alterations
                    "interference_processing": True,  # Handle conflicting memories
                    "belief_consistency": True,       # Check beliefs against memories
                    "mask_checks": True               # Update mask based on memories
                }
            )
            
            # Additional maintenance: check for contradictory beliefs
            belief_result = await self._reconcile_contradictory_beliefs()
            
            # Add belief reconciliation to results
            maintenance_result["belief_reconciliation"] = belief_result
            
            # Check if time-related maintenance is needed (less frequent)
            # Use a random chance to avoid all NPCs doing this at once
            if random.random() < 0.2:  # 20% chance each maintenance cycle
                await self._run_time_based_maintenance()
                maintenance_result["time_based_maintenance"] = True
            
            return maintenance_result
            
        except Exception as e:
            logger.error(f"Error running memory maintenance for NPC {self.npc_id}: {e}")
            return {"error": str(e)}
    
    async def _reconcile_contradictory_beliefs(self) -> Dict[str, Any]:
        """
        Find and reconcile contradictory beliefs with cognitive consistency principles.
        Models human cognitive dissonance resolution.
        """
        result = {
            "contradictions_found": 0,
            "beliefs_modified": 0,
            "beliefs_removed": 0
        }
        
        try:
            memory_system = await self._get_memory_system()
            
            # Get all beliefs
            all_beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=self.npc_id
            )
            
            # Group beliefs by topic
            beliefs_by_topic = {}
            for belief in all_beliefs:
                topic = belief.get("topic", "general")
                if topic not in beliefs_by_topic:
                    beliefs_by_topic[topic] = []
                beliefs_by_topic[topic].append(belief)
            
            # Check each topic for contradictions
            for topic, beliefs in beliefs_by_topic.items():
                # Skip topics with only one belief
                if len(beliefs) < 2:
                    continue
                
                # Compare each belief pair for contradictions
                contradictory_pairs = []
                
                for i in range(len(beliefs)):
                    for j in range(i+1, len(beliefs)):
                        belief1 = beliefs[i]
                        belief2 = beliefs[j]
                        
                        # Simple contradiction detection based on negation terms
                        text1 = belief1.get("belief", "").lower()
                        text2 = belief2.get("belief", "").lower()
                        
                        negation_terms = ["not", "isn't", "doesn't", "won't", "can't", "never"]
                        
                        has_contradiction = False
                        # Check if one belief contains negation of concepts in the other
                        words1 = set(text1.split())
                        words2 = set(text2.split())
                        
                        # If same core words but one has negation
                        common_words = words1.intersection(words2)
                        if len(common_words) >= 2:
                            # Check if one contains negation but not the other
                            has_negation1 = any(term in text1 for term in negation_terms)
                            has_negation2 = any(term in text2 for term in negation_terms)
                            
                            if has_negation1 != has_negation2:
                                has_contradiction = True
                        
                        # Direct opposites in belief sentiment
                        sentiment_pairs = [
                            ("good", "bad"), ("like", "dislike"), ("trust", "distrust"),
                            ("friend", "enemy"), ("safe", "dangerous"), ("honest", "dishonest")
                        ]
                        
                        for pos, neg in sentiment_pairs:
                            if (pos in text1 and neg in text2) or (neg in text1 and pos in text2):
                                has_contradiction = True
                                break
                        
                        if has_contradiction:
                            contradictory_pairs.append((belief1, belief2))
                            result["contradictions_found"] += 1
                
                # Resolve each contradiction
                for belief1, belief2 in contradictory_pairs:
                    # Resolution strategy: keep the belief with higher confidence
                    confidence1 = belief1.get("confidence", 0.5)
                    confidence2 = belief2.get("confidence", 0.5)
                    
                    if confidence1 >= confidence2:
                        # Keep belief1, remove belief2
                        await memory_system.remove_belief(
                            entity_type="npc",
                            entity_id=self.npc_id,
                            belief_id=belief2["id"]
                        )
                        result["beliefs_removed"] += 1
                    else:
                        # Keep belief2, remove belief1
                        await memory_system.remove_belief(
                            entity_type="npc",
                            entity_id=self.npc_id,
                            belief_id=belief1["id"]
                        )
                        result["beliefs_removed"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error reconciling contradictory beliefs: {e}")
            return {"error": str(e)}
    
    async def _run_time_based_maintenance(self) -> None:
        """
        Run maintenance tasks that depend on the passage of time.
        Models natural forgetting and memory evolution.
        """
        try:
            memory_system = await self._get_memory_system()
            
            # 1. Archive very old memories (3+ months)
            await memory_system.update_memory_status(
                entity_type="npc",
                entity_id=self.npc_id,
                criteria={"age_days": 90, "max_significance": 3},
                new_status="archived"
            )
            
            # 2. Consolidate repetitive old memories (1+ month)
            await memory_system.consolidate_memories(
                entity_type="npc",
                entity_id=self.npc_id,
                criteria={"age_days": 30, "pattern_threshold": 0.7}
            )
            
            # 3. Apply gradual decay to emotional intensity
            await memory_system.apply_memory_decay(
                entity_type="npc",
                entity_id=self.npc_id,
                decay_rate=0.05  # 5% decay
            )
            
            # 4. Update mask integrity based on long-term behavior patterns
            mask_manager = await self._get_mask_manager()
            
            # Check for long-term behavior trends
            behavior_trends = await memory_system.get_behavior_trends(
                entity_type="npc",
                entity_id=self.npc_id,
                timeframe_days=30
            )

            mask_manager = await self._get_mask_manager()
            reinforcement_score = await self._check_for_mask_reinforcement_behaviors()
            
            if reinforcement_score > 0:
                # Calculate recovery amount based on reinforcement behaviors
                recovery_amount = min(5, reinforcement_score * 2)
                
                await mask_manager.adjust_mask_integrity(
                    npc_id=self.npc_id,
                    adjustment=recovery_amount,
                    reason="Mask reinforcement behaviors"
                )
                
                # Create memory of the reinforcement
                memory_system = await self._get_memory_system()
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    memory_text=f"I spent time reinforcing my mask to hide my true nature",
                    importance="medium",
                    tags=["mask_reinforcement", "self_improvement"]
                )

            await _evolve_personality_traits
            
            # Adjust mask based on behavior consistency with presented traits
            if behavior_trends:
                true_nature_behaviors = behavior_trends.get("true_nature_consistent", 0)
                mask_behaviors = behavior_trends.get("mask_consistent", 0)
                
                # Calculate behavior ratio
                total_behaviors = true_nature_behaviors + mask_behaviors
                if total_behaviors > 0:
                    true_nature_ratio = true_nature_behaviors / total_behaviors
                    
                    # High true nature ratio = mask integrity decreases
                    if true_nature_ratio > 0.7:
                        await mask_manager.adjust_mask_integrity(
                            npc_id=self.npc_id,
                            adjustment=-5,  # Decrease integrity by 5%
                            reason="Consistent true nature behaviors"
                        )
                    # High mask ratio = mask integrity slowly recovers
                    elif true_nature_ratio < 0.3:
                        await mask_manager.adjust_mask_integrity(
                            npc_id=self.npc_id,
                            adjustment=3,  # Increase integrity by 3%
                            reason="Consistent mask-aligned behaviors"
                        )
            
        except Exception as e:
            logger.error(f"Error running time-based maintenance: {e}")
