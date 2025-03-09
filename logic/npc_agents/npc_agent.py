# logic/npc_agents/npc_agent.py

"""
Core NPC agent class that manages individual NPC behavior with memory capabilities.
"""

import logging
import json
import asyncio
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union

from db.connection import get_db_connection  # Assuming this is synchronous.
from .decision_engine import NPCDecisionEngine
from .environment_perception import (
    fetch_environment_data,
    is_significant_action,
    execute_npc_action
)

# Memory system imports
from memory.wrapper import MemorySystem
from memory.core import MemorySignificance
from memory.masks import ProgressiveRevealManager

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------
# Additional classes referenced in npc_agent.py (assumed to be in memory/wrapper.py)
# --------------------------------------------------------------------------------

class SchemaManager:
    """Schema manager handles the creation and application of memory schemas."""

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def apply_schema_to_memory(
        self,
        memory_id: int,
        entity_type: str,
        entity_id: int,
        auto_detect: bool = False
    ) -> Dict[str, Any]:
        """
        Apply a schema to a specific memory.

        Args:
            memory_id: ID of the memory
            entity_type: Type of entity that owns the memory
            entity_id: ID of the entity
            auto_detect: Whether to auto-detect appropriate schemas

        Returns:
            Result of the schema application.
        """
        # Implementation would connect to the database and apply schema logic.
        result = {
            "memory_id": memory_id,
            "schemas_applied": [],
            "auto_detected": auto_detect
        }
        logger.debug(
            f"Applied schema to memory {memory_id} for {entity_type}:{entity_id}"
        )
        return result


class IntegratedMemoryManager:
    """Integrated memory manager for comprehensive memory operations."""

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def run_memory_maintenance(
        self,
        entity_type: str,
        entity_id: int,
        maintenance_options: Dict[str, bool] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive memory maintenance for an entity.

        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            maintenance_options: Options for what maintenance to perform

        Returns:
            Results of maintenance operations.
        """
        options = maintenance_options or {
            "core_maintenance": True,
            "schema_maintenance": True,
            "emotional_decay": True,
            "memory_consolidation": True,
            "background_reconsolidation": True,
            "interference_processing": True,
            "belief_consistency": True,
            "mask_checks": True
        }

        result = {
            "memories_processed": 0,
            "memories_archived": 0,
            "memories_consolidated": 0,
            "schemas_updated": 0,
            "beliefs_checked": 0
        }

        logger.debug(f"Running memory maintenance for {entity_type}:{entity_id}")
        return result


# Add these methods to the MemorySystem class in memory/wrapper.py
async def get_behavior_trends(
    self,
    entity_type: str,
    entity_id: int,
    timeframe_days: int = 30
) -> Dict[str, int]:
    """
    Analyze trends in entity behavior based on memories.

    Args:
        entity_type: Type of entity
        entity_id: ID of the entity
        timeframe_days: Number of days to analyze

    Returns:
        Dictionary with behavior trend counts.
    """
    trends = {
        "true_nature_consistent": 0,
        "mask_consistent": 0,
        "emotional_outbursts": 0,
        "submissive_behaviors": 0,
        "dominant_behaviors": 0,
        "cruel_actions": 0,
        "kind_actions": 0
    }

    # Query for recent memories
    memory_result = await self.recall(
        entity_type=entity_type,
        entity_id=entity_id,
        query="",
        limit=50,
        context={"max_age_days": timeframe_days}
    )

    memories = memory_result.get("memories", [])

    # Analyze memories for behavior patterns
    for memory in memories:
        text = memory.get("text", "").lower()
        tags = memory.get("tags", [])

        # Check for dominant behaviors
        if any(word in text for word in ["command", "order", "dominate", "control"]):
            trends["dominant_behaviors"] += 1
            if "dominance_dynamic" in tags or "true_nature" in tags:
                trends["true_nature_consistent"] += 1
            elif "mask_reinforcing" in tags:
                trends["mask_consistent"] += 1

        # Check for submissive behaviors
        if any(word in text for word in ["obey", "submit", "follow", "comply"]):
            trends["submissive_behaviors"] += 1
            if "submission" in tags or "true_nature" in tags:
                trends["true_nature_consistent"] += 1
            elif "mask_reinforcing" in tags:
                trends["mask_consistent"] += 1

        # Check for cruel actions
        if any(word in text for word in ["mock", "humiliate", "hurt", "cruel"]):
            trends["cruel_actions"] += 1
            if "cruelty" in tags or "true_nature" in tags:
                trends["true_nature_consistent"] += 1
            elif "mask_reinforcing" in tags:
                trends["mask_consistent"] += 1

        # Check for kind actions
        if any(word in text for word in ["help", "support", "kind", "nice"]):
            trends["kind_actions"] += 1
            if "kindness" in tags or "true_nature" in tags:
                trends["true_nature_consistent"] += 1
            elif "mask_reinforcing" in tags:
                trends["mask_consistent"] += 1

        # Check for emotional outbursts
        if "emotional_outburst" in tags:
            trends["emotional_outbursts"] += 1

    return trends


# ---------------------------------------------------------
# The NPCAgent class with memory, async usage, and caching.
# ---------------------------------------------------------

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

        # Thread-safety lock for cache operations
        self.lock = asyncio.Lock()

        # Lazy-loaded memory components
        self._memory_system: Optional[MemorySystem] = None
        self._mask_manager: Optional[ProgressiveRevealManager] = None

        self.current_emotional_state: Optional[Dict[str, Any]] = None
        self.last_perception: Optional[Dict[str, Any]] = None
        self.decision_history: List[Dict[str, Any]] = []

        # Enhanced cache management
        self._cache: Dict[str, Union[Dict[str, Any], Any]] = {
            'perception': {},
            'memories': {},
            'relationships': {},
            'emotional_state': None,
            'mask': None
        }
        self._cache_timestamps: Dict[str, Union[Dict[str, datetime], datetime, None]] = {
            'perception': {},
            'memories': {},
            'relationships': None,
            'emotional_state': None,
            'mask': None
        }
        self._cache_ttls: Dict[str, timedelta] = {
            'perception': timedelta(minutes=5),
            'memories': timedelta(minutes=10),
            'relationships': timedelta(minutes=15),
            'emotional_state': timedelta(minutes=2),
            'mask': timedelta(minutes=5)
        }

        # Performance metrics tracking
        self.perf_metrics: Dict[str, Any] = {
            'perception_time': [],
            'decision_time': [],
            'action_time': [],
            'memory_retrieval_time': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'last_reported': datetime.now()
        }

        # Set up periodic performance reporting
        self._setup_performance_reporting()

    async def _get_memory_system(self) -> MemorySystem:
        """
        Lazy-load the memory system.
        """
        if self._memory_system is None:
            self._memory_system = await MemorySystem.get_instance(
                self.user_id,
                self.conversation_id
            )
        return self._memory_system

    async def _get_mask_manager(self) -> ProgressiveRevealManager:
        """
        Lazy-load the mask manager.
        """
        if self._mask_manager is None:
            self._mask_manager = ProgressiveRevealManager(
                self.user_id,
                self.conversation_id
            )
        return self._mask_manager

    def _setup_performance_reporting(self) -> None:
        """
        Set up periodic performance reporting in the background.
        """
        async def report_metrics():
            while True:
                await asyncio.sleep(600)  # Report every 10 minutes
                metrics_dict = {}

                # Calculate averages for each metric list
                for metric, values in self.perf_metrics.items():
                    if metric not in ['last_reported', 'cache_hits', 'cache_misses'] and isinstance(values, list):
                        if values:
                            avg_val = sum(values) / len(values)
                            metrics_dict[f'avg_{metric}'] = avg_val
                            # Keep only last 100 values to avoid unbounded memory growth
                            self.perf_metrics[metric] = values[-100:]

                # Add cache hit rate
                total_cache_ops = (
                    self.perf_metrics.get('cache_hits', 0)
                    + self.perf_metrics.get('cache_misses', 0)
                )
                if total_cache_ops > 0:
                    metrics_dict['cache_hit_rate'] = (
                        self.perf_metrics.get('cache_hits', 0) / total_cache_ops
                    )

                if metrics_dict:
                    logger.info(f"NPC {self.npc_id} performance: {metrics_dict}")

                self.perf_metrics['last_reported'] = datetime.now()

        # Start the background reporting task
        asyncio.create_task(report_metrics())

    async def invalidate_cache(self, cache_key: Optional[str] = None) -> None:
        """
        Invalidate specific cache entries or all if key is None, in a thread-safe way.

        Args:
            cache_key: Specific cache to invalidate, or None for all.
        """
        async with self.lock:
            if cache_key is None:
                self._cache = {
                    'perception': {},
                    'memories': {},
                    'relationships': {},
                    'emotional_state': None,
                    'mask': None
                }
                self._cache_timestamps = {
                    'perception': {},
                    'memories': {},
                    'relationships': None,
                    'emotional_state': None,
                    'mask': None
                }
                logger.debug(f"Invalidated all caches for NPC {self.npc_id}")
            elif cache_key in self._cache:
                # Invalidate the specific cache
                if isinstance(self._cache[cache_key], dict):
                    self._cache[cache_key] = {}
                    if isinstance(self._cache_timestamps[cache_key], dict):
                        self._cache_timestamps[cache_key] = {}
                else:
                    self._cache[cache_key] = None
                    self._cache_timestamps[cache_key] = None

                logger.debug(f"Invalidated {cache_key} cache for NPC {self.npc_id}")

    async def invalidate_memory_cache(self, memory_query: Optional[str] = None) -> None:
        """
        Invalidate memory cache, either completely or for a specific query.

        Args:
            memory_query: Query string to invalidate, or None for all.
        """
        async with self.lock:
            if memory_query is None:
                self._cache['memories'] = {}
                self._cache_timestamps['memories'] = {}
                logger.debug(f"Invalidated all memory caches for NPC {self.npc_id}")
            elif (
                isinstance(self._cache['memories'], dict)
                and memory_query in self._cache['memories']
            ):
                del self._cache['memories'][memory_query]
                if (
                    isinstance(self._cache_timestamps['memories'], dict)
                    and memory_query in self._cache_timestamps['memories']
                ):
                    del self._cache_timestamps['memories'][memory_query]
                logger.debug(
                    f"Invalidated memory cache for query '{memory_query}' in NPC {self.npc_id}"
                )

    async def _update_cache(
        self,
        cache_key: str,
        sub_key: Optional[str] = None,
        value: Any = None
    ) -> None:
        """
        Update agent cache with new values, thread-safe.

        Args:
            cache_key: Main cache category
            sub_key: Optional sub-key for dict caches
            value: New value to store
        """
        async with self.lock:
            now = datetime.now()

            if sub_key is not None:
                # Initialize structures if needed
                if cache_key not in self._cache:
                    self._cache[cache_key] = {}
                if cache_key not in self._cache_timestamps:
                    self._cache_timestamps[cache_key] = {}

                self._cache[cache_key][sub_key] = value
                self._cache_timestamps[cache_key][sub_key] = now
            else:
                self._cache[cache_key] = value
                self._cache_timestamps[cache_key] = now

            if cache_key in ["perception", "emotional_state", "mask"]:
                logger.debug(f"Updated {cache_key} cache for NPC {self.npc_id}")

    def is_cache_valid(self, cache_key: str, sub_key: Optional[str] = None) -> bool:
        """
        Check if a cache entry is still valid based on TTL.

        Args:
            cache_key: Main cache key
            sub_key: Optional sub-key for dict caches

        Returns:
            True if cache entry is valid, False otherwise.
        """
        now = datetime.now()
        ttl = self._cache_ttls.get(cache_key, None)

        if ttl is None:
            return False

        timestamp = None
        if sub_key is not None:
            if (
                not isinstance(self._cache_timestamps.get(cache_key), dict)
                or sub_key not in self._cache_timestamps[cache_key]
            ):
                return False
            timestamp = self._cache_timestamps[cache_key].get(sub_key)
        else:
            timestamp = self._cache_timestamps.get(cache_key)

        if timestamp is None:
            return False

        return timestamp + ttl > now

    async def perceive_environment(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced environment perception with performance monitoring and emotional context.

        Args:
            current_context: The current context information.

        Returns:
            Dictionary containing the NPC's perception.
        """
        perf_start = time.perf_counter()
        environment_data = {}

        try:
            # Fetch basic environment data
            environment_data = await fetch_environment_data(
                self.user_id,
                self.conversation_id,
                current_context
            )

            # Create a unique cache key based on the context
            context_key = str(hash(json.dumps(current_context, sort_keys=True, default=str)))

            # Check cache validity
            if self.is_cache_valid('perception', context_key):
                self.last_perception = self._cache['perception'][context_key]
                self.perf_metrics['cache_hits'] += 1
                return self.last_perception
            else:
                self.perf_metrics['cache_misses'] += 1

            # Get memory system to retrieve relevant memories
            memory_system = await self._get_memory_system()

            # Construct a text description for memory retrieval
            context_description = current_context.get("description", "")
            if "text" in current_context:
                context_description += " " + current_context["text"]

            context_for_recall = {
                "text": context_description,
                "location": environment_data.get("location", "Unknown"),
                "time_of_day": environment_data.get("time_of_day", "Unknown")
            }

            # Add entities present for richer social context
            if "entities_present" in environment_data:
                context_for_recall["entities_present"] = [
                    e.get("name", "")
                    for e in environment_data.get("entities_present", [])
                ]

            # Retrieve current emotional state
            emotional_state = await memory_system.get_npc_emotion(self.npc_id)
            self.current_emotional_state = emotional_state

            # If there's a strong emotion, incorporate into context
            if emotional_state and "current_emotion" in emotional_state:
                current_emotion = emotional_state["current_emotion"]
                emotion_name = current_emotion.get("primary", {}).get("name", "neutral")
                intensity = current_emotion.get("primary", {}).get("intensity", 0.0)

                if intensity > 0.6:
                    context_for_recall["emotional_state"] = {
                        "primary_emotion": emotion_name,
                        "intensity": intensity
                    }

            # Determine an adaptive memory limit based on context importance
            base_limit = 5
            adaptive_limit = base_limit
            context_importance = 0

            keywords_high = ["critical", "emergency", "dangerous", "threat", "crucial", "sex", "intimate"]
            keywords_medium = ["important", "significant", "unusual", "strange", "unexpected"]

            lowered_desc = context_description.lower()
            for w in keywords_high:
                if w in lowered_desc:
                    context_importance += 2
            for w in keywords_medium:
                if w in lowered_desc:
                    context_importance += 1

            if context_importance >= 3:
                adaptive_limit = base_limit + 5
            elif context_importance >= 1:
                adaptive_limit = base_limit + 2

            # Retrieve relevant memories
            memory_result = await memory_system.recall(
                entity_type="npc",
                entity_id=self.npc_id,
                context=context_for_recall,
                limit=adaptive_limit
            )
            relevant_memories = memory_result.get("memories", [])

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
                    # Possibly update emotional state
                    if "emotional_response" in trigger_result:
                        response = trigger_result["emotional_response"]
                        await memory_system.update_npc_emotion(
                            npc_id=self.npc_id,
                            emotion=response.get("primary_emotion", "fear"),
                            intensity=response.get("intensity", 0.7)
                        )
                        emotional_state = await memory_system.get_npc_emotion(self.npc_id)
                        self.current_emotional_state = emotional_state

            # Check for flashback possibility
            flashback = None
            flashback_chance = 0.15
            if traumatic_trigger:
                flashback_chance = 0.5

            if random.random() < flashback_chance:
                flashback = await memory_system.npc_flashback(
                    npc_id=self.npc_id,
                    context=context_description
                )

            # Retrieve mask info
            mask_info = await memory_system.get_npc_mask(self.npc_id)
            if mask_info and "integrity" not in mask_info:
                mask_info["integrity"] = 100

            # Retrieve relationship data
            relationship_data = await self._fetch_relationships_with_memory()

            # Retrieve beliefs
            beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=self.npc_id,
                topic="player"  # Just focusing on player-related
            )
            if beliefs:
                beliefs = sorted(beliefs, key=lambda x: x.get("confidence", 0), reverse=True)

            # Build final perception
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

            perception_complexity = {
                "memory_count": len(relevant_memories),
                "relationship_count": len(relationship_data),
                "has_flashback": flashback is not None,
                "has_traumatic_trigger": traumatic_trigger is not None,
                "belief_count": len(beliefs)
            }
            perception["complexity_metrics"] = perception_complexity

            # Cache the perception
            await self._update_cache('perception', context_key, perception)
            self.last_perception = perception

            # Record performance
            elapsed = time.perf_counter() - perf_start
            self.perf_metrics['perception_time'].append(elapsed)
            self.perf_metrics['perception_complexity'] = perception_complexity

            # Warn if slow
            if elapsed > 0.5:
                logger.warning(f"Slow perception for NPC {self.npc_id}: {elapsed:.2f}s")

            return perception

        except Exception as e:
            elapsed = time.perf_counter() - perf_start
            logger.error(
                f"Perception error for NPC {self.npc_id} after {elapsed:.2f}s: {e}"
            )
            return {
                "environment": environment_data,
                "relevant_memories": [],
                "relationships": {},
                "emotional_state": {
                    "current_emotion": {
                        "primary": {"name": "neutral", "intensity": 0.0}
                    }
                },
                "mask": {"integrity": 100},
                "beliefs": [],
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    async def _fetch_relationships_with_memory(self) -> Dict[str, Any]:
        """
        Get the NPC's relationships, enriched with memory references.

        Returns:
            Dictionary of relationships with memory context.
        """
        relationships = {}
        try:
            # NOTE: This is a synchronous DB call. Consider using async if available.
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT entity2_type, entity2_id, link_type, link_level
                    FROM SocialLinks
                    WHERE entity1_type = 'npc'
                      AND entity1_id = %s
                      AND user_id = %s
                      AND conversation_id = %s
                    """,
                    (self.npc_id, self.user_id, self.conversation_id)
                )
                rows = cursor.fetchall()
                npc_entity_ids = []

                for entity_type, entity_id, _, _ in rows:
                    if entity_type == "npc":
                        npc_entity_ids.append(entity_id)

                npc_names = {}
                if npc_entity_ids:
                    cursor.execute(
                        """
                        SELECT npc_id, npc_name
                        FROM NPCStats
                        WHERE npc_id = ANY(%s)
                          AND user_id = %s
                          AND conversation_id = %s
                        """,
                        (npc_entity_ids, self.user_id, self.conversation_id)
                    )
                    for npc_id, npc_name in cursor.fetchall():
                        npc_names[npc_id] = npc_name

                for entity_type, entity_id, link_type, link_level in rows:
                    entity_name = "Unknown"
                    if entity_type == "npc":
                        entity_name = npc_names.get(entity_id, f"NPC_{entity_id}")
                    elif entity_type == "player":
                        entity_name = "Player"

                    relationships[entity_type] = {
                        "entity_id": entity_id,
                        "entity_name": entity_name,
                        "link_type": link_type,
                        "link_level": link_level
                    }

            memory_system = await self._get_memory_system()

            for etype, rel_data in relationships.items():
                e_id = rel_data["entity_id"]
                e_name = rel_data["entity_name"]
                query = e_name

                memory_result = await memory_system.recall(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    query=query,
                    limit=3
                )
                rel_data["memory_context"] = memory_result.get("memories", [])

            return relationships

        except Exception as e:
            logger.error(f"Error fetching relationships with memory: {e}")
            return {}

    async def make_decision(
        self,
        perception: Optional[Dict[str, Any]] = None,
        available_actions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Decide on an action with memory-driven decision making and mask system.

        Args:
            perception: The NPC's environment perception (optional)
            available_actions: List of available actions (optional)

        Returns:
            The chosen action as a dictionary.
        """
        perf_start = time.perf_counter()

        try:
            if perception is None:
                if self.last_perception is None:
                    perception = await self.perceive_environment({})
                else:
                    perception = self.last_perception

            memory_system = await self._get_memory_system()

            # Apply belief weights
            beliefs = perception.get("beliefs", [])
            if available_actions and beliefs:
                weighted_actions = await self._apply_belief_weights_to_actions(
                    available_actions,
                    beliefs
                )
                available_actions = weighted_actions

            # Check traumatic triggers
            traumatic_trigger = perception.get("traumatic_trigger")
            if traumatic_trigger and traumatic_trigger.get("triggered", False):
                trauma_response = self._create_trauma_response_action(traumatic_trigger)
                if trauma_response:
                    if available_actions:
                        available_actions.insert(0, trauma_response)
                    else:
                        available_actions = [trauma_response]

            # Apply emotional influences
            emotional_state = perception.get("emotional_state", {})
            current_emotion = emotional_state.get("current_emotion", {})
            if current_emotion:
                primary_emotion = current_emotion.get("primary", {})
                emotion_name = primary_emotion.get("name", "neutral")
                intensity = primary_emotion.get("intensity", 0.0)

                if intensity > 0.7:
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

            chosen_action = await self.decision_engine.decide(perception, available_actions)
            logger.debug(f"NPCAgent {self.npc_id} decided on action: {chosen_action}")

            # Check mask slippage
            should_slip = await self._should_mask_slip(perception, chosen_action)
            if should_slip:
                slip_trigger = f"deciding to {chosen_action.get('description', 'act')}"
                if current_emotion and current_emotion.get("primary", {}).get("intensity", 0.0) > 0.5:
                    slip_trigger = f"feeling {emotion_name} while " + slip_trigger

                slip_result = await memory_system.reveal_npc_trait(
                    npc_id=self.npc_id,
                    trigger=slip_trigger
                )
                chosen_action["mask_slippage"] = slip_result

                await memory_system.remember(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    memory_text=(
                        f"My mask slipped while I was "
                        f"{chosen_action.get('description', 'doing something')}"
                    ),
                    importance="medium",
                    tags=["mask_slip", "self_awareness"]
                )

            # Record decision history
            self._record_decision_history(chosen_action)

            elapsed = time.perf_counter() - perf_start
            self.perf_metrics['decision_time'].append(elapsed)
            return chosen_action

        except Exception as e:
            elapsed = time.perf_counter() - perf_start
            logger.error(f"Decision error for NPC {self.npc_id} after {elapsed:.2f}s: {e}")
            return {"type": "observe", "description": "observe quietly", "target": "environment"}

    async def execute_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the chosen action.

        Args:
            action: The action dictionary
            context: Additional context information

        Returns:
            Result of executing the action.
        """
        perf_start = time.perf_counter()

        try:
            result = await execute_npc_action(
                self.npc_id,
                self.user_id,
                self.conversation_id,
                action,
                context
            )

            elapsed = time.perf_counter() - perf_start
            self.perf_metrics['action_time'].append(elapsed)

            # Create memory of significant actions
            if is_significant_action(action, result):
                memory_system = await self._get_memory_system()
                memory_text = (
                    f"I {action.get('description', 'did something')} "
                    f"with result: {result.get('outcome', 'unknown')}"
                )
                await memory_system.emotional_manager.analyze_emotional_content(memory_text)

                await memory_system.remember(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    memory_text=memory_text,
                    importance="medium",
                    emotional=True,
                    tags=["action", action.get("type", "unknown")]
                )

            return result

        except Exception as e:
            elapsed = time.perf_counter() - perf_start
            logger.error(f"Error executing action for NPC {self.npc_id} after {elapsed:.2f}s: {e}")
            return {
                "outcome": (
                    f"NPC attempted to {action.get('description', 'do something')} "
                    f"but encountered an error"
                ),
                "emotional_impact": 0,
                "target_reactions": [],
                "error": str(e)
            }

    async def process_player_action(
        self,
        player_action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a player's action with memory formation and psychological realism.

        Args:
            player_action: The player's action
            context: Additional context information (optional)

        Returns:
            The NPC's response to the player action.
        """
        perf_start = time.perf_counter()

        try:
            context_obj = context or {}
            perception_context = {
                "player_action": player_action,
                "text": player_action.get("description", ""),
                "description": f"Player {player_action.get('description', 'did something')}"
            }
            perception_context.update(context_obj)

            perception = await self.perceive_environment(perception_context)
            memory_system = await self._get_memory_system()

            # Player memory text
            player_memory_text = f"The player {player_action.get('description', 'did something')}"
            emotion_analysis = await memory_system.emotional_manager.analyze_emotional_content(
                player_memory_text,
                context=str(player_action)
            )
            mem_result = await memory_system.emotional_manager.add_emotional_memory(
                entity_type="npc",
                entity_id=self.npc_id,
                memory_text=player_memory_text,
                primary_emotion=emotion_analysis.get("primary_emotion", "neutral"),
                emotion_intensity=emotion_analysis.get("intensity", 0.5),
                secondary_emotions=emotion_analysis.get("secondary_emotions", {}),
                significance=MemorySignificance.MEDIUM,
                tags=["player_action", player_action.get("type", "unknown")]
            )

            memory_id = mem_result.get("memory_id")
            if memory_id:
                await memory_system.integrated.apply_schema_to_memory(
                    memory_id=memory_id,
                    entity_type="npc",
                    entity_id=self.npc_id,
                    auto_detect=True
                )

            # Update emotional state based on player action
            player_action_type = player_action.get("type", "").lower()
            mask_info = perception.get("mask", {})
            mask_integrity = mask_info.get("integrity", 100)
            hidden_traits = mask_info.get("hidden_traits", {})

            if player_action_type in ["attack", "threaten", "insult"]:
                if mask_integrity < 50 and "dominant" in hidden_traits:
                    await memory_system.update_npc_emotion(self.npc_id, "anger", 0.8)
                else:
                    await memory_system.update_npc_emotion(self.npc_id, "fear", 0.7)
            elif player_action_type in ["praise", "help", "gift"]:
                if mask_integrity < 50 and "suspicious" in hidden_traits:
                    await memory_system.update_npc_emotion(self.npc_id, "uncertainty", 0.5)
                else:
                    await memory_system.update_npc_emotion(self.npc_id, "joy", 0.6)
            elif player_action_type in ["command", "dominate"]:
                if mask_integrity < 50 and ("dominant" in hidden_traits or "controlling" in hidden_traits):
                    await memory_system.update_npc_emotion(self.npc_id, "anger", 0.7)
                    if random.random() < 0.3:
                        await memory_system.reveal_npc_trait(
                            npc_id=self.npc_id,
                            trigger=(
                                f"being commanded by player to {player_action.get('description', 'do something')}"
                            )
                        )
                else:
                    await memory_system.update_npc_emotion(self.npc_id, "submission", 0.6)

            await self.invalidate_cache("emotional_state")

            # Decide how to respond
            response_action = await self.make_decision(perception)
            result = await self.execute_action(response_action, perception_context)

            # Create memory of the interaction
            interaction_memory_text = (
                f"When the player {player_action.get('description','did something')}, "
                f"I responded by {response_action.get('description','doing something')}"
            )
            interaction_emotion_analysis = await memory_system.emotional_manager.analyze_emotional_content(
                interaction_memory_text
            )
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

            # Update beliefs
            await self._update_beliefs_from_interaction(player_action, response_action, result)

            # Update relationships
            await self._update_relationship_from_interaction(
                "player",
                self.user_id,
                player_action,
                response_action,
                result
            )

            elapsed = time.perf_counter() - perf_start
            self.perf_metrics['action_processing_time'] = elapsed

            logger.debug(
                f"NPCAgent {self.npc_id} processed player action '{player_action}': result={result}"
            )

            return {
                "npc_id": self.npc_id,
                "action": response_action,
                "result": result,
                "processing_time_ms": int(elapsed * 1000)
            }

        except Exception as e:
            elapsed = time.perf_counter() - perf_start
            logger.error(f"Error processing player action for NPC {self.npc_id} after {elapsed:.2f}s: {e}")
            return {
                "npc_id": self.npc_id,
                "action": {"type": "error", "description": "had an internal error"},
                "result": {"outcome": "NPC seems confused", "emotional_impact": -1},
                "error": str(e)
            }

    async def _update_beliefs_from_interaction(
        self,
        player_action: Dict[str, Any],
        npc_action: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
        """
        Update beliefs based on an interaction with the player.

        Args:
            player_action: The player's action
            npc_action: The NPC's response action
            result: Result of the NPC's action
        """
        try:
            memory_system = await self._get_memory_system()
            current_beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=self.npc_id,
                topic="player"
            )

            player_action_type = player_action.get("type", "")
            outcome = result.get("outcome", "")
            emotional_impact = result.get("emotional_impact", 0)

            positive_actions = ["help", "gift", "praise", "support", "protect"]
            negative_actions = ["attack", "threaten", "mock", "betray", "insult"]
            submission_actions = ["obey", "submit", "comply"]
            defiance_actions = ["defy", "resist", "disobey"]

            action_category = "neutral"
            if player_action_type in positive_actions:
                action_category = "positive"
            elif player_action_type in negative_actions:
                action_category = "negative"
            elif player_action_type in submission_actions:
                action_category = "submission"
            elif player_action_type in defiance_actions:
                action_category = "defiance"

            if action_category == "neutral" and abs(emotional_impact) < 3:
                return  # Skip updating beliefs if it's low-impact neutral

            # Retrieve personality traits for more nuance
            dominance, cruelty = 50, 50
            personality_traits = {}
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT dominance, cruelty, personality_traits
                    FROM NPCStats
                    WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                    """,
                    (self.npc_id, self.user_id, self.conversation_id)
                )
                row = cursor.fetchone()
                if row:
                    dominance, cruelty, traits_data = row[0], row[1], row[2]
                    if isinstance(traits_data, str):
                        try:
                            personality_traits = json.loads(traits_data)
                        except json.JSONDecodeError:
                            personality_traits = {}
                    else:
                        personality_traits = traits_data or {}

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

            # Trait-based modifications
            if "suspicious" in personality_traits:
                belief_patterns["positive"]["confidence_change"] = 0.05
                belief_patterns["positive"]["new_confidence"] = 0.5
            if "trusting" in personality_traits:
                belief_patterns["positive"]["confidence_change"] = 0.15
                belief_patterns["positive"]["new_confidence"] = 0.75
            if dominance and dominance > 70:
                belief_patterns["defiance"]["confidence_change"] = 0.2
                belief_patterns["defiance"]["new_confidence"] = 0.8

            pattern = belief_patterns.get(action_category)
            if not pattern:
                return

            belief_text = pattern["text"]
            confidence_change = pattern["confidence_change"]
            new_confidence = pattern["new_confidence"]

            if abs(emotional_impact) > 3:
                confidence_change *= 1.5

            existing_belief = None
            for b in current_beliefs:
                b_text = b.get("belief", "").lower()
                p_text = belief_text.lower()
                if len(set(b_text.split()) & set(p_text.split())) >= 2:
                    existing_belief = b
                    break

            if existing_belief:
                old_conf = existing_belief.get("confidence", 0.5)
                adjusted_conf = min(0.95, old_conf + confidence_change)
                await memory_system.semantic_manager.update_belief_confidence(
                    belief_id=existing_belief["id"],
                    entity_type="npc",
                    entity_id=self.npc_id,
                    new_confidence=adjusted_conf,
                    reason=f"Based on player's {player_action_type} action"
                )
            else:
                await memory_system.create_belief(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    belief_text=belief_text,
                    confidence=new_confidence
                )

            # Possibly form a counter-belief
            if (
                action_category in ["positive", "negative"]
                and abs(emotional_impact) > 4
                and random.random() < 0.3
            ):
                counter_cat = "negative" if action_category == "positive" else "positive"
                counter_pattern = belief_patterns.get(counter_cat)
                if counter_pattern:
                    counter_conf = counter_pattern["new_confidence"] * 0.7
                    await memory_system.create_belief(
                        entity_type="npc",
                        entity_id=self.npc_id,
                        belief_text=(
                            f"Despite appearances, "
                            f"{counter_pattern['text'].lower()}"
                        ),
                        confidence=counter_conf
                    )

            await self.invalidate_cache("beliefs")

        except Exception as e:
            logger.error(f"Error updating beliefs from interaction: {e}")

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

        Args:
            entity_type: Type of entity ("npc" or "player")
            entity_id: ID of the entity
            player_action: The player's action
            npc_response: The NPC's response action
            result: Result of the NPC's action
        """
        try:
            from .relationship_manager import NPCRelationshipManager
            relationship_manager = NPCRelationshipManager(
                self.npc_id,
                self.user_id,
                self.conversation_id
            )
            memory_system = await self._get_memory_system()
            emotional_state = await memory_system.get_npc_emotion(self.npc_id)
            topic = "player" if entity_type == "player" else f"npc_{entity_id}"

            beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=self.npc_id,
                topic=topic
            )
            recall_result = await memory_system.recall(
                entity_type="npc",
                entity_id=self.npc_id,
                query=entity_type,
                limit=3
            )

            enhanced_context = {
                "emotional_state": emotional_state,
                "beliefs": beliefs,
                "recent_memories": recall_result.get("memories", []),
                "result": result
            }

            await relationship_manager.update_relationship_from_interaction(
                entity_type,
                entity_id,
                player_action,
                npc_response,
                enhanced_context
            )
            await self.invalidate_cache("relationships")

        except Exception as e:
            logger.error(f"Error updating relationship from interaction: {e}")

    async def _should_mask_slip(
        self,
        perception: Dict[str, Any],
        chosen_action: Dict[str, Any]
    ) -> bool:
        """
        Determine if the mask should slip based on psychological factors.

        Args:
            perception: The NPC's perception
            chosen_action: The NPC's chosen action

        Returns:
            True if the mask should slip, else False.
        """
        try:
            mask_info = perception.get("mask", {})
            mask_integrity = mask_info.get("integrity", 100)
            if mask_integrity <= 0 or mask_integrity >= 100:
                return False

            base_probability = (100 - mask_integrity) / 200
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT dominance, cruelty, self_control
                    FROM NPCStats
                    WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                    """,
                    (self.npc_id, self.user_id, self.conversation_id)
                )
                row = cursor.fetchone()
                if not row:
                    return False

                dominance, cruelty, self_control = row
                if self_control is None:
                    self_control = 50

            behavior_patterns = await self._analyze_recent_behavior_patterns()

            consistency_modifier = 0.0
            if behavior_patterns.get("true_nature_acting", 0) > 3:
                consistency_modifier -= 0.2
            elif behavior_patterns.get("mask_reinforcing", 0) > 3:
                consistency_modifier -= 0.3

            personality_factor = 0.0
            if dominance:
                personality_factor += (dominance / 200)
            if cruelty:
                personality_factor += (cruelty / 200)
            if self_control:
                personality_factor -= (self_control / 200)

            emotional_state = perception.get("emotional_state", {})
            emotion_factor = 0.0
            if emotional_state and "current_emotion" in emotional_state:
                curr_emotion = emotional_state["current_emotion"]
                ename = curr_emotion.get("primary", {}).get("name", "neutral")
                intensity = curr_emotion.get("primary", {}).get("intensity", 0.0)
                if intensity > 0.5:
                    if ename == "anger":
                        emotion_factor += intensity * 0.4
                    elif ename == "fear":
                        emotion_factor += intensity * 0.3
                    elif ename == "joy":
                        emotion_factor += intensity * 0.1

            action_factor = 0.0
            action_type = chosen_action.get("type", "")
            if action_type in ["dominate", "command", "punish", "mock"]:
                action_factor += 0.2
            elif action_type in ["emotional_outburst", "express_anger"]:
                action_factor += 0.3
            elif action_type in ["mask_reinforcement", "self_control"]:
                action_factor -= 0.4

            context_factor = 0.0
            if perception.get("traumatic_trigger"):
                context_factor += 0.4
            if perception.get("flashback"):
                context_factor += 0.3

            total_probability = (
                base_probability
                + personality_factor
                + emotion_factor
                + action_factor
                + context_factor
                + consistency_modifier
            )
            final_probability = min(0.95, max(0, total_probability))
            return random.random() < final_probability

        except Exception as e:
            logger.error(f"Error in _should_mask_slip: {e}")
            return False

    async def _analyze_recent_behavior_patterns(self) -> Dict[str, int]:
        """
        Analyze patterns in recent behavior for mask slip logic.

        Returns:
            Dictionary with pattern counts.
        """
        patterns = {
            "true_nature_acting": 0,
            "mask_reinforcing": 0,
            "mixed_signals": 0
        }
        try:
            if len(self.decision_history) >= 5:
                memory_system = await self._get_memory_system()
                mask_info = await memory_system.get_npc_mask(self.npc_id)
                if not mask_info:
                    return patterns

                hidden_traits = mask_info.get("hidden_traits", {})
                presented_traits = mask_info.get("presented_traits", {})
                recent_actions = [d["action"] for d in self.decision_history[-5:]]

                for action in recent_actions:
                    a_type = action.get("type", "")
                    true_nature_alignment = 0
                    if "dominant" in hidden_traits and a_type in ["command", "dominate", "test"]:
                        true_nature_alignment += 1
                    if "cruel" in hidden_traits and a_type in ["mock", "humiliate", "punish"]:
                        true_nature_alignment += 1
                    if "sadistic" in hidden_traits and a_type in ["punish", "humiliate"]:
                        true_nature_alignment += 1

                    mask_alignment = 0
                    if "kind" in presented_traits and a_type in ["praise", "support", "help"]:
                        mask_alignment += 1
                    if "gentle" in presented_traits and a_type in ["talk", "observe", "support"]:
                        mask_alignment += 1
                    if "submissive" in presented_traits and a_type in ["observe", "wait", "obey"]:
                        mask_alignment += 1

                    if true_nature_alignment > mask_alignment:
                        patterns["true_nature_acting"] += 1
                    elif mask_alignment > true_nature_alignment:
                        patterns["mask_reinforcing"] += 1
                    elif mask_alignment > 0 and true_nature_alignment > 0:
                        patterns["mixed_signals"] += 1

            return patterns

        except Exception as e:
            logger.error(f"Error analyzing behavior patterns: {e}")
            return patterns

    def _record_decision_history(self, action: Dict[str, Any]) -> None:
        """
        Record a decision in the agent's decision history for pattern analysis.
    
        Args:
            action: The action the NPC decided on.
        """
        if not hasattr(self, 'decision_history'):
            self.decision_history = []
            
        self.decision_history.append({
            "action": action,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit size to prevent memory leaks
        if len(self.decision_history) > 20:  # Choose appropriate limit
            self.decision_history = self.decision_history[-20:]

    def _create_trauma_response_action(
        self,
        trauma_trigger: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Create a psychologically realistic action in response to a traumatic trigger.

        Args:
            trauma_trigger: Information about the traumatic trigger.

        Returns:
            A trauma-response action dictionary or None.
        """
        try:
            emotional_response = trauma_trigger.get("emotional_response", {})
            primary_emotion = emotional_response.get("primary_emotion", "fear")
            intensity = emotional_response.get("intensity", 0.5)
            trigger_text = trauma_trigger.get("trigger_text", "")

            if primary_emotion == "fear":
                response_type = "freeze"
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT dominance, cruelty
                        FROM NPCStats
                        WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                        """,
                        (self.npc_id, self.user_id, self.conversation_id)
                    )
                    row = cursor.fetchone()
                    if row:
                        dom, cru = row
                        if dom > 70 or cru > 70:
                            response_type = "fight"
                        elif dom < 30:
                            response_type = "flight"

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
                else:
                    return {
                        "type": "traumatic_response",
                        "description": "freeze in response to a triggering memory",
                        "target": "self",
                        "weight": 2.0 * intensity,
                        "stats_influenced": {"trust": -5},
                        "trauma_trigger": trigger_text
                    }
            elif primary_emotion == "anger":
                return {
                    "type": "traumatic_response",
                    "description": "respond with anger to a triggering situation",
                    "target": "group",
                    "weight": 1.8 * intensity,
                    "stats_influenced": {"trust": -5, "respect": -5},
                    "trauma_trigger": trigger_text
                }
            elif primary_emotion == "sadness":
                return {
                    "type": "traumatic_response",
                    "description": "become visibly downcast due to a painful memory",
                    "target": "self",
                    "weight": 1.7 * intensity,
                    "stats_influenced": {"closeness": +2},
                    "trauma_trigger": trigger_text
                }
            else:
                return {
                    "type": "traumatic_response",
                    "description": "respond emotionally to a triggering memory",
                    "target": "self",
                    "weight": 1.5 * intensity,
                    "stats_influenced": {},
                    "trauma_trigger": trigger_text
                }

        except Exception as e:
            logger.error(f"Error creating trauma response: {e}")
            return None

    async def _apply_belief_weights_to_actions(
        self,
        available_actions: List[Dict[str, Any]],
        beliefs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply belief weights to actions for more realistic decision-making.

        Args:
            available_actions: List of available actions (dicts)
            beliefs: List of beliefs

        Returns:
            Actions with updated 'weight' field.
        """
        try:
            for action in available_actions:
                action_type = action.get("type", "")
                target = action.get("target", "")
                belief_weight = 1.0

                for belief in beliefs:
                    b_text = belief.get("belief", "").lower()
                    confidence = belief.get("confidence", 0.5)
                    if confidence < 0.3:
                        continue

                    relevance = 0.0
                    if action_type in b_text:
                        relevance += 0.5
                    if target in b_text:
                        relevance += 0.5

                    if target in ["player", "group"]:
                        if any(word in b_text for word in ["trust", "like", "friend"]):
                            if action_type in ["talk", "confide", "praise"]:
                                relevance += 0.8
                            elif action_type in ["mock", "leave", "attack"]:
                                relevance -= 0.8
                        if any(word in b_text for word in ["distrust", "fear", "threat"]):
                            if action_type in ["leave", "observe", "act_defensive"]:
                                relevance += 0.8
                            elif action_type in ["talk", "confide", "praise"]:
                                relevance -= 0.6
                        if any(word in b_text for word in ["submissive", "obedient"]):
                            if action_type in ["command", "dominate", "test"]:
                                relevance += 0.9
                        if any(word in b_text for word in ["dangerous", "threat"]):
                            if action_type in ["leave", "act_defensive"]:
                                relevance += 1.0
                            elif action_type in ["confide", "praise"]:
                                relevance -= 0.9

                    belief_weight += relevance * confidence

                if "weight" in action:
                    action["weight"] *= belief_weight
                else:
                    action["weight"] = belief_weight

                if "decision_metadata" not in action:
                    action["decision_metadata"] = {}
                action["decision_metadata"]["belief_weight"] = belief_weight

            return available_actions

        except Exception as e:
            logger.error(f"Error applying belief weights to actions: {e}")
            return available_actions

    async def run_memory_maintenance(self) -> Dict[str, Any]:
        """
        Run memory maintenance for this NPC with various lifecycle steps.

        Returns:
            Results of maintenance operations.
        """
        try:
            memory_system = await self._get_memory_system()
            maintenance_result = await memory_system.integrated.run_memory_maintenance(
                entity_type="npc",
                entity_id=self.npc_id,
                maintenance_options={
                    "core_maintenance": True,
                    "schema_maintenance": True,
                    "emotional_decay": True,
                    "memory_consolidation": True,
                    "background_reconsolidation": True,
                    "interference_processing": True,
                    "belief_consistency": True,
                    "mask_checks": True
                }
            )
            belief_result = await self._reconcile_contradictory_beliefs()
            maintenance_result["belief_reconciliation"] = belief_result

            if random.random() < 0.2:
                await self._run_time_based_maintenance()
                maintenance_result["time_based_maintenance"] = True

            await self.invalidate_cache()
            return maintenance_result

        except Exception as e:
            logger.error(f"Error running memory maintenance for NPC {self.npc_id}: {e}")
            return {"error": str(e)}

    async def _reconcile_contradictory_beliefs(self) -> Dict[str, Any]:
        """
        Find and reconcile contradictory beliefs via cognitive consistency.

        Returns:
            Dict describing results of belief reconciliation.
        """
        result = {
            "contradictions_found": 0,
            "beliefs_modified": 0,
            "beliefs_removed": 0
        }
        try:
            memory_system = await self._get_memory_system()
            all_beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=self.npc_id
            )
            beliefs_by_topic = {}
            for b in all_beliefs:
                t = b.get("topic", "general")
                if t not in beliefs_by_topic:
                    beliefs_by_topic[t] = []
                beliefs_by_topic[t].append(b)

            for topic, beliefs in beliefs_by_topic.items():
                if len(beliefs) < 2:
                    continue
                contradictory_pairs = []
                for i in range(len(beliefs)):
                    for j in range(i + 1, len(beliefs)):
                        b1 = beliefs[i]
                        b2 = beliefs[j]
                        text1 = b1.get("belief", "").lower()
                        text2 = b2.get("belief", "").lower()

                        negation_terms = ["not", "isn't", "doesn't", "won't", "can't", "never"]
                        has_contradiction = False
                        words1 = set(text1.split())
                        words2 = set(text2.split())
                        common_words = words1.intersection(words2)
                        if len(common_words) >= 2:
                            has_negation1 = any(term in text1 for term in negation_terms)
                            has_negation2 = any(term in text2 for term in negation_terms)
                            if has_negation1 != has_negation2:
                                has_contradiction = True

                        sentiment_pairs = [
                            ("good", "bad"),
                            ("like", "dislike"),
                            ("trust", "distrust"),
                            ("friend", "enemy"),
                            ("safe", "dangerous"),
                            ("honest", "dishonest")
                        ]
                        for pos, neg in sentiment_pairs:
                            if (pos in text1 and neg in text2) or (neg in text1 and pos in text2):
                                has_contradiction = True
                                break

                        if has_contradiction:
                            contradictory_pairs.append((b1, b2))
                            result["contradictions_found"] += 1

                for b1, b2 in contradictory_pairs:
                    c1 = b1.get("confidence", 0.5)
                    c2 = b2.get("confidence", 0.5)
                    if c1 >= c2:
                        await memory_system.remove_belief(
                            entity_type="npc",
                            entity_id=self.npc_id,
                            belief_id=b2["id"]
                        )
                        result["beliefs_removed"] += 1
                    else:
                        await memory_system.remove_belief(
                            entity_type="npc",
                            entity_id=self.npc_id,
                            belief_id=b1["id"]
                        )
                        result["beliefs_removed"] += 1

            return result

        except Exception as e:
            logger.error(f"Error reconciling contradictory beliefs: {e}")
            return {"error": str(e)}

    async def _run_time_based_maintenance(self) -> None:
        """
        Run time-based maintenance tasks (e.g. archiving old memories, decay, etc.).
        """
        try:
            memory_system = await self._get_memory_system()
            await memory_system.update_memory_status(
                entity_type="npc",
                entity_id=self.npc_id,
                criteria={"age_days": 90, "max_significance": 3},
                new_status="archived"
            )
            await memory_system.consolidate_memories(
                entity_type="npc",
                entity_id=self.npc_id,
                criteria={"age_days": 30, "pattern_threshold": 0.7}
            )
            await memory_system.apply_memory_decay(
                entity_type="npc",
                entity_id=self.npc_id,
                decay_rate=0.05
            )

            mask_manager = await self._get_mask_manager()
            behavior_trends = await memory_system.get_behavior_trends(
                entity_type="npc",
                entity_id=self.npc_id,
                timeframe_days=30
            )
            reinforcement_score = await self._check_for_mask_reinforcement_behaviors()
            if reinforcement_score > 0:
                recovery_amount = min(5, reinforcement_score * 2)
                await mask_manager.adjust_mask_integrity(
                    npc_id=self.npc_id,
                    adjustment=recovery_amount,
                    reason="Mask reinforcement behaviors"
                )
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    memory_text=(
                        "I spent time reinforcing my mask to hide my true nature"
                    ),
                    importance="medium",
                    tags=["mask_reinforcement", "self_improvement"]
                )

            await self._evolve_personality_traits()
            if behavior_trends:
                tn = behavior_trends.get("true_nature_consistent", 0)
                mk = behavior_trends.get("mask_consistent", 0)
                total = tn + mk
                if total > 0:
                    ratio = tn / total
                    if ratio > 0.7:
                        await mask_manager.adjust_mask_integrity(
                            npc_id=self.npc_id,
                            adjustment=-5,
                            reason="Consistent true nature behaviors"
                        )
                    elif ratio < 0.3:
                        await mask_manager.adjust_mask_integrity(
                            npc_id=self.npc_id,
                            adjustment=3,
                            reason="Consistent mask-aligned behaviors"
                        )

        except Exception as e:
            logger.error(f"Error running time-based maintenance: {e}")

    async def _check_for_mask_reinforcement_behaviors(self) -> float:
        """
        Check for behaviors that would reinforce an NPC's mask.

        Returns:
            Reinforcement score (float).
        """
        try:
            reinforcement_score = 0.0
            if len(self.decision_history) > 0:
                recent_actions = self.decision_history[-5:]
                for decision in recent_actions:
                    action = decision.get("action", {})
                    a_type = action.get("type", "")
                    if a_type in ["observe", "talk"]:
                        reinforcement_score += 0.2
                    elif a_type in ["leave", "act_defensive"]:
                        reinforcement_score += 0.3
                    elif a_type == "mask_reinforcement":
                        reinforcement_score += 1.0

            # NOTE: Synchronous DB call
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT current_location
                    FROM NPCStats
                    WHERE npc_id = %s
                    """,
                    (self.npc_id,)
                )
                row = cursor.fetchone()
                if row and row[0]:
                    location = row[0]
                    cursor.execute(
                        """
                        SELECT COUNT(*)
                        FROM NPCStats
                        WHERE current_location = %s AND npc_id != %s
                        """,
                        (location, self.npc_id)
                    )
                    row_count = cursor.fetchone()
                    if row_count and row_count[0] == 0:
                        reinforcement_score += 0.5

            return reinforcement_score

        except Exception as e:
            logger.error(f"Error checking mask reinforcement behaviors: {e}")
            return 0.0

    async def _evolve_personality_traits(self) -> Dict[str, Any]:
        """
        Gradually evolve personality traits based on recent experiences.

        Returns:
            Results describing trait changes.
        """
        results = {
            "traits_modified": 0,
            "new_traits": 0,
            "removed_traits": 0
        }
        try:
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT personality_traits, dominance, cruelty
                    FROM NPCStats
                    WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                    """,
                    (self.npc_id, self.user_id, self.conversation_id)
                )
                row = cursor.fetchone()
                if not row:
                    return {"error": "NPC not found"}
                traits_json, dominance, cruelty = row
                if isinstance(traits_json, str):
                    try:
                        traits = json.loads(traits_json)
                    except json.JSONDecodeError:
                        traits = []
                else:
                    traits = traits_json or []

            memory_system = await self._get_memory_system()
            memory_result = await memory_system.recall(
                entity_type="npc",
                entity_id=self.npc_id,
                query="significant experience",
                limit=10
            )
            memories = memory_result.get("memories", [])
            trait_influences = {}
            for mem in memories:
                txt = mem.get("text", "").lower()
                if any(word in txt for word in ["commanded", "controlled", "dominated", "power"]):
                    trait_influences["dominant"] = trait_influences.get("dominant", 0) + 1
                if any(word in txt for word in ["obeyed", "submitted", "followed", "complied"]):
                    trait_influences["submissive"] = trait_influences.get("submissive", 0) + 1
                if any(word in txt for word in ["mocked", "humiliated", "hurt", "cruel"]):
                    trait_influences["cruel"] = trait_influences.get("cruel", 0) + 1
                if any(word in txt for word in ["helped", "supported", "kind", "gentle"]):
                    trait_influences["kind"] = trait_influences.get("kind", 0) + 1

            modified_traits = traits.copy()
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

            for trait, count in trait_influences.items():
                if count >= 3 and trait not in modified_traits:
                    modified_traits.append(trait)
                    results["new_traits"] += 1

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

            if (
                modified_traits != traits
                or new_dominance != dominance
                or new_cruelty != cruelty
            ):
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE NPCStats
                        SET personality_traits = %s,
                            dominance = %s,
                            cruelty = %s
                        WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                        """,
                        (
                            json.dumps(modified_traits),
                            new_dominance,
                            new_cruelty,
                            self.npc_id,
                            self.user_id,
                            self.conversation_id
                        )
                    )

            return results

        except Exception as e:
            logger.error(f"Error evolving personality traits: {e}")
            return {"error": str(e)}
