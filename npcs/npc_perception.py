# npcs/npc_perception.py

"""
Environment perception tools for NPC agents.
Refactored and aligned with the dynamic relationship system.
"""

import logging
import asyncio
import random
import time
import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from agents import function_tool, trace  # kept for compatibility; function_span is used
from agents.tracing import function_span
from db.connection import get_db_connection_context
from memory.wrapper import MemorySystem
from logic.game_time_helper import get_game_time_string, GameTimeContext

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Pydantic models for structured data
# -------------------------------------------------------

class EnvironmentData(BaseModel):
    """Data about the environment an NPC is in."""
    location: str = "Unknown"
    time_of_day: str = "Unknown"
    day_name: Optional[str] = None     # e.g., "Moonday" (semantic day-of-week)
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None          # day of month (numeric)
    timestamp: Optional[str] = None
    entities_present: List[Dict[str, Any]] = Field(default_factory=list)
    description: Optional[str] = None

class ActionSignificance(BaseModel):
    """Evaluation of an action's significance."""
    is_significant: bool
    level: int
    reason: str

class ActionResult(BaseModel):
    """Result of executing an NPC action."""
    outcome: str
    emotional_impact: int = 0
    target_reactions: List[str] = Field(default_factory=list)
    success: bool = True
    error: Optional[str] = None

class PerceptionContext(BaseModel):
    """Context information for perception."""
    location: Optional[str] = None
    time_of_day: Optional[str] = None
    description: Optional[str] = None
    player_action: Optional[Dict[str, Any]] = None
    entities_present: List[Dict[str, Any]] = Field(default_factory=list)
    query_text: Optional[str] = None

class PerceptionResult(BaseModel):
    """Result of environment perception."""
    environment: Dict[str, Any] = Field(default_factory=dict)
    relevant_memories: List[Dict[str, Any]] = Field(default_factory=list)
    relationships: Dict[str, Any] = Field(default_factory=dict)
    emotional_state: Dict[str, Any] = Field(default_factory=dict)
    flashback: Optional[Dict[str, Any]] = None
    traumatic_trigger: Optional[Dict[str, Any]] = None
    mask: Dict[str, Any] = Field(default_factory=dict)
    beliefs: List[Dict[str, Any]] = Field(default_factory=list)
    time_context: Dict[str, Any] = Field(default_factory=dict)
    narrative_context: Dict[str, Any] = Field(default_factory=dict)
    cache_hit: bool = False
    processing_time: float = 0.0
    error: Optional[str] = None  # added to avoid constructor errors on failure path

# -------------------------------------------------------
# Main EnvironmentPerception class
# -------------------------------------------------------

class EnvironmentPerception:
    """
    Provides environment perception tools for NPC agents.
    Integrates environment data with memory retrieval.
    Uses OptimizedRelationshipManager for relationship reads.
    """

    def __init__(self, npc_id: int, user_id: int, conversation_id: int):
        """
        Initialize environment perception for an NPC.

        Args:
            npc_id: ID of the NPC
            user_id: ID of the user/player
            conversation_id: ID of the conversation
        """
        self.npc_id = int(npc_id)
        self.user_id = int(user_id)
        self.conversation_id = int(conversation_id)
        self._memory_system: Optional[MemorySystem] = None
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamp: Dict[str, float] = {}
        self._cache_ttl = 300  # 5 minutes in seconds

    async def get_memory_system(self) -> MemorySystem:
        """Lazy-load the memory system."""
        if self._memory_system is None:
            self._memory_system = await MemorySystem.get_instance(
                self.user_id,
                self.conversation_id
            )
        return self._memory_system

    def _is_cache_valid(self, key: str) -> bool:
        """Check if a cache entry is valid."""
        ts = self._cache_timestamp.get(key)
        return key in self._cache and ts is not None and (time.time() - ts) < self._cache_ttl

    def _update_cache(self, key: str, value: Any) -> None:
        """Update a cache entry."""
        self._cache[key] = value
        self._cache_timestamp[key] = time.time()

    def _invalidate_cache(self, key: Optional[str] = None) -> None:
        """Invalidate cache entries."""
        if key is None:
            self._cache.clear()
            self._cache_timestamp.clear()
        else:
            self._cache.pop(key, None)
            self._cache_timestamp.pop(key, None)

    def _stable_cache_key(self, context: Dict[str, Any], detail_level: str) -> str:
        """Build a stable cache key for perception context."""
        try:
            if isinstance(context, PerceptionContext):
                ctx_dict = context.dict()
            else:
                ctx_dict = dict(context or {})
            return f"perception:{detail_level}:{json.dumps(ctx_dict, sort_keys=True, default=str)}"
        except Exception:
            return f"perception:{detail_level}:{hash(str(context))}"

    async def perceive_environment(
        self,
        context: Dict[str, Any],
        detail_level: str = "auto"  # "low", "standard", "high", "auto"
    ) -> PerceptionResult:
        """
        Perceive environment with adaptive detail levels based on system load.

        Args:
            context: Context information (or PerceptionContext)
            detail_level: Level of detail to retrieve

        Returns:
            PerceptionResult object
        """
        perf_start = time.time()

        with function_span("perceive_environment"):
            # Process detail level
            if detail_level == "auto":
                try:
                    import psutil
                    system_load = psutil.cpu_percent()
                    mem_usage = psutil.virtual_memory().percent

                    if system_load > 80 or mem_usage > 85:
                        detail_level = "low"
                    elif system_load > 60 or mem_usage > 70:
                        detail_level = "standard"
                    else:
                        detail_level = "high"
                except Exception:
                    detail_level = "standard"

            # Normalize context
            perception_context = (
                PerceptionContext(**context) if not isinstance(context, PerceptionContext) else context
            )

            # Cache check (use stable key)
            cache_key = self._stable_cache_key(perception_context.dict(), detail_level)
            if self._is_cache_valid(cache_key):
                cached = self._cache[cache_key]
                try:
                    result = PerceptionResult(**cached)
                except Exception:
                    # if cache got corrupted, drop it
                    self._invalidate_cache(cache_key)
                    result = None
                else:
                    result.processing_time = time.time() - perf_start
                    result.cache_hit = True
                    return result

            # Set memory/feature limits based on detail level
            if detail_level == "low":
                memory_limit = 3
                include_flashbacks = False
                include_beliefs = False
            elif detail_level == "high":
                memory_limit = 8
                include_flashbacks = True
                include_beliefs = True
            else:
                memory_limit = 5
                include_flashbacks = random.random() < 0.3
                include_beliefs = True

            try:
                # Get environment data
                environment_data = await self.fetch_environment_data(perception_context)

                # Get memory system
                memory_system = await self.get_memory_system()

                # Construct query for memory retrieval
                query_text = perception_context.query_text or perception_context.description or ""
                if perception_context.player_action and isinstance(perception_context.player_action, dict):
                    desc = perception_context.player_action.get("description")
                    if desc:
                        query_text = f"{query_text} {desc}".strip()

                # Build context for recall
                context_for_recall = {
                    "text": query_text,
                    "location": environment_data.get("location", "Unknown"),
                    "time_of_day": environment_data.get("time_of_day", "Unknown")
                }
                if "entities_present" in environment_data:
                    context_for_recall["entities_present"] = [
                        (e.get("name") or f"{e.get('type','entity')}_{e.get('id','?')}") for e in environment_data.get("entities_present", [])
                    ]

                # Emotional state
                emotional_state = await memory_system.get_npc_emotion(self.npc_id)
                if emotional_state and "current_emotion" in emotional_state:
                    current_emotion = emotional_state["current_emotion"]
                    # Handle different structures
                    if isinstance(current_emotion.get("primary", {}), dict):
                        emotion_name = current_emotion.get("primary", {}).get("name", "neutral")
                        intensity = current_emotion.get("primary", {}).get("intensity", 0.0)
                    else:
                        emotion_name = current_emotion.get("primary", "neutral")
                        intensity = current_emotion.get("intensity", 0.0)

                    if isinstance(intensity, (int, float)) and intensity > 0.6:
                        context_for_recall["emotional_state"] = {
                            "primary_emotion": emotion_name,
                            "intensity": float(intensity)
                        }

                # Adaptive memory limit based on context importance
                base_limit = memory_limit
                adaptive_limit = base_limit
                if query_text:
                    importance = self._calculate_context_importance(query_text)
                    if importance >= 3:
                        adaptive_limit = base_limit + 5
                    elif importance >= 1:
                        adaptive_limit = base_limit + 2

                # Retrieve memories
                relevant_memories: List[Dict[str, Any]] = []
                try:
                    memory_result = await memory_system.recall(
                        entity_type="npc",
                        entity_id=self.npc_id,
                        context=context_for_recall,
                        limit=adaptive_limit
                    )
                    relevant_memories = memory_result.get("memories", []) or []
                except Exception as me:
                    logger.debug(f"[Perception] memory recall failed for NPC {self.npc_id}: {me}")

                # Traumatic triggers
                traumatic_trigger: Optional[Dict[str, Any]] = None
                if query_text:
                    try:
                        trigger_result = await memory_system.emotional_manager.process_traumatic_triggers(
                            entity_type="npc",
                            entity_id=self.npc_id,
                            text=query_text
                        )
                        if trigger_result and trigger_result.get("triggered"):
                            traumatic_trigger = trigger_result
                            # Update emotional state if triggered
                            if "emotional_response" in trigger_result:
                                response = trigger_result["emotional_response"]
                                await memory_system.update_npc_emotion(
                                    npc_id=self.npc_id,
                                    emotion=response.get("primary_emotion", "fear"),
                                    intensity=response.get("intensity", 0.7)
                                )
                                emotional_state = await memory_system.get_npc_emotion(self.npc_id)
                    except Exception as te:
                        logger.debug(f"[Perception] trigger processing failed for NPC {self.npc_id}: {te}")

                # Flashback
                flashback: Optional[Dict[str, Any]] = None
                flashback_chance = 0.15 if not traumatic_trigger else 0.5
                if include_flashbacks and random.random() < flashback_chance:
                    try:
                        flashback = await memory_system.npc_flashback(
                            npc_id=self.npc_id,
                            context=query_text or ""
                        )
                    except Exception as fe:
                        logger.debug(f"[Perception] flashback failed for NPC {self.npc_id}: {fe}")

                # Mask info
                mask_info: Dict[str, Any] = {}
                try:
                    mask_info = await memory_system.get_npc_mask(self.npc_id) or {}
                    if "integrity" not in mask_info:
                        mask_info["integrity"] = 100
                except Exception as me:
                    mask_info = {"integrity": 100}
                    logger.debug(f"[Perception] mask fetch failed for NPC {self.npc_id}: {me}")

                # Relationships (use dynamic relationship manager)
                relationship_data = await self.fetch_relationships(environment_data)

                # Beliefs (optional based on detail level)
                beliefs: List[Dict[str, Any]] = []
                if include_beliefs:
                    try:
                        beliefs = await memory_system.get_beliefs(
                            entity_type="npc",
                            entity_id=self.npc_id,
                            topic="player"
                        ) or []
                        beliefs.sort(key=lambda x: x.get("confidence", 0), reverse=True)
                    except Exception as be:
                        logger.debug(f"[Perception] beliefs fetch failed for NPC {self.npc_id}: {be}")

                # Time context
                time_context = await self.fetch_time_context()

                # Narrative context
                narrative_context = await self.fetch_narrative_context()

                # Build result
                result = PerceptionResult(
                    environment=environment_data,
                    relevant_memories=relevant_memories,
                    flashback=flashback,
                    relationships=relationship_data,
                    emotional_state=emotional_state or {},
                    traumatic_trigger=traumatic_trigger,
                    mask=mask_info or {"integrity": 100},
                    beliefs=beliefs or [],
                    time_context=time_context,
                    narrative_context=narrative_context,
                    processing_time=time.time() - perf_start,
                    cache_hit=False
                )

                # Cache the result
                try:
                    self._update_cache(cache_key, result.dict())
                except Exception:
                    # Best-effort caching
                    pass

                return result

            except Exception as e:
                logger.error(f"[Perception] error in environment perception for NPC {self.npc_id}: {e}")
                # Return minimal perception on error
                elapsed = time.time() - perf_start
                env = locals().get("environment_data", {})
                return PerceptionResult(
                    environment=env if isinstance(env, dict) else {},
                    emotional_state={
                        "current_emotion": {
                            "primary": {"name": "neutral", "intensity": 0.0}
                        }
                    },
                    mask={"integrity": 100},
                    processing_time=elapsed,
                    cache_hit=False,
                    error=str(e)
                )

    def _calculate_context_importance(self, context_text: str) -> int:
        """
        Calculate the importance of a context based on keywords.

        Args:
            context_text: Text to analyze

        Returns:
            Importance score (0-5)
        """
        importance = 0
        lowered = (context_text or "").lower()

        # High importance keywords
        high_keywords = [
            "critical", "emergency", "dangerous", "threat", "crucial",
            "sex", "intimate", "fight", "attack", "betrayal"
        ]

        # Medium importance keywords
        medium_keywords = [
            "important", "significant", "unusual", "strange",
            "unexpected", "surprise", "secret", "revealed"
        ]

        for word in high_keywords:
            if word in lowered:
                importance += 2

        for word in medium_keywords:
            if word in lowered:
                importance += 1

        return min(5, importance)  # Cap at 5

    async def fetch_environment_data(
        self,
        context: PerceptionContext
    ) -> Dict[str, Any]:
        """
        Fetch environment data from database based on context.

        Args:
            context: Context information

        Returns:
            Dictionary with environment data
        """
        with function_span("fetch_environment_data"):
            try:
                async with GameTimeContext(self.user_id, self.conversation_id) as game_time:
                    environment_data: Dict[str, Any] = {
                        "location": "Unknown",
                        "time_of_day": game_time.time_of_day,
                        "timestamp": await game_time.to_string(),
                        "year": game_time.year,
                        "month": game_time.month,
                        "day": game_time.day,
                        "day_name": None,  # Will populate if calendar available
                        "entities_present": []
                    }

                # Populate day_name from calendar if available
                try:
                    from logic.calendar import load_calendar_names
                    calendar_data = await load_calendar_names(self.user_id, self.conversation_id)
                    day_names = calendar_data.get("days", [])
                    if day_names and environment_data.get("day"):
                        # Calculate which day of week it is
                        day_index = (int(environment_data["day"]) - 1) % len(day_names)
                        environment_data["day_name"] = day_names[day_index]
                except Exception as e:
                    logger.debug(f"[Perception] Could not load calendar day names: {e}")

                # Extract location from context if available
                if context.location:
                    environment_data["location"] = context.location

                # Fetch location from database if not in context
                if environment_data["location"] == "Unknown":
                    async with get_db_connection_context() as conn:
                        row = await conn.fetchrow(
                            """
                            SELECT value FROM CurrentRoleplay
                            WHERE key = 'CurrentLocation' AND user_id = $1 AND conversation_id = $2
                            """,
                            self.user_id, self.conversation_id
                        )
                        if row and row[0]:
                            environment_data["location"] = row[0]

                # Extract time of day from context if available
                if context.time_of_day:
                    environment_data["time_of_day"] = context.time_of_day

                # Fetch time of day from database if not in context
                if environment_data["time_of_day"] == "Unknown":
                    async with get_db_connection_context() as conn:
                        row = await conn.fetchrow(
                            """
                            SELECT value FROM CurrentRoleplay
                            WHERE key = 'TimeOfDay' AND user_id = $1 AND conversation_id = $2
                            """,
                            self.user_id, self.conversation_id
                        )
                        if row and row[0]:
                            environment_data["time_of_day"] = row[0]

                # Use entities from context if available
                if context.entities_present:
                    environment_data["entities_present"] = context.entities_present
                else:
                    # Fetch entities present in location
                    async with get_db_connection_context() as conn:
                        # Fetch NPCs
                        rows = await conn.fetch(
                            """
                            SELECT npc_id, npc_name FROM NPCStats
                            WHERE current_location = $1 AND user_id = $2 AND conversation_id = $3
                            """,
                            environment_data["location"], self.user_id, self.conversation_id
                        )

                        entities: List[Dict[str, Any]] = []
                        for row in rows:
                            npc_id = row["npc_id"]
                            if npc_id != self.npc_id:  # Don't include self
                                entities.append({
                                    "type": "npc",
                                    "id": npc_id,
                                    "name": row["npc_name"]
                                })

                        # Add player entity (use user_id for identity in environment;
                        # the dynamic relationship manager uses entity id=1 internally)
                        entities.append({
                            "type": "player",
                            "id": self.user_id,
                            "name": "Player"
                        })

                        environment_data["entities_present"] = entities

                # Add description if available
                if context.description:
                    environment_data["description"] = context.description

                return environment_data

            except Exception as e:
                logger.error(f"[Perception] Error fetching environment data: {e}")
                # Add timestamp for observability in error path
                ts = await get_game_time_string(self.user_id, self.conversation_id)
                return {
                    "location": "Unknown",
                    "time_of_day": "Unknown",
                    "entities_present": [],
                    "timestamp": ts,
                    "error": str(e),
                }

    async def fetch_time_context(self) -> Dict[str, Any]:
        """
        Fetch the current time context (year, month, day, time of day) from the database.

        Returns:
            Dictionary with time context
        """
        with function_span("fetch_time_context"):
            time_context = {
                "year": None,
                "month": None,
                "day": None,
                "time_of_day": None
            }

            try:
                async with get_db_connection_context() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT key, value
                        FROM CurrentRoleplay
                        WHERE key IN ('CurrentYear', 'CurrentMonth', 'CurrentDay', 'TimeOfDay')
                          AND user_id = $1
                          AND conversation_id = $2
                        """,
                        self.user_id, self.conversation_id
                    )

                    for row in rows:
                        key = row["key"]
                        value = row["value"]
                        if key == "CurrentYear":
                            time_context["year"] = value
                        elif key == "CurrentMonth":
                            time_context["month"] = value
                        elif key == "CurrentDay":
                            time_context["day"] = value
                        elif key == "TimeOfDay":
                            time_context["time_of_day"] = value

                    return time_context

            except Exception as e:
                logger.error(f"[Perception] Error fetching time context: {e}")
                return time_context

    async def fetch_narrative_context(self) -> Dict[str, Any]:
        """
        Fetch the current narrative context (plot stage, tension) from the database.

        Returns:
            Dictionary with narrative context
        """
        with function_span("fetch_narrative_context"):
            narrative_context: Dict[str, Any] = {}

            try:
                async with get_db_connection_context() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT key, value
                        FROM CurrentRoleplay
                        WHERE key IN ('CurrentPlotStage', 'CurrentTension')
                          AND user_id = $1
                          AND conversation_id = $2
                        """,
                        self.user_id, self.conversation_id
                    )

                    for row in rows:
                        narrative_context[row["key"]] = row["value"]

                    return narrative_context

            except Exception as e:
                logger.error(f"[Perception] Error fetching narrative context: {e}")
                return narrative_context

    async def fetch_relationships(self, environment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch relationships with entities in the environment using the dynamic relationship system.

        Returns:
            Dictionary keyed by "type_id" -> relationship payload
            Example key: "player_1", "npc_12"
        """
        with function_span("fetch_relationships"):
            relationships: Dict[str, Any] = {}

            # Lazy import to avoid heavy import at module load
            try:
                from logic.dynamic_relationships import OptimizedRelationshipManager
                dyn = OptimizedRelationshipManager(self.user_id, self.conversation_id)
            except Exception as e:
                logger.debug(f"[Perception] dynamic_relationships unavailable: {e}")
                return relationships

            try:
                memory_system = await self.get_memory_system()
            except Exception as e:
                logger.debug(f"[Perception] memory system unavailable: {e}")
                memory_system = None

            entities = environment_data.get("entities_present", []) or []
            for entity in entities:
                try:
                    entity_type = entity.get("type")
                    entity_id = entity.get("id")
                    entity_name = entity.get("name") or f"{entity_type}_{entity_id}"
                    if not entity_type or entity_id is None:
                        continue

                    # Determine pair and fetch dynamic state
                    if entity_type == "player":
                        # Convention: dynamic system uses player id = 1
                        state = await dyn.get_relationship_state("player", 1, "npc", self.npc_id)
                    elif entity_type == "npc":
                        state = await dyn.get_relationship_state("npc", self.npc_id, "npc", int(entity_id))
                    else:
                        # Unsupported entity type for relationship system
                        continue

                    dims = state.dimensions

                    # trust [-100..100] => [0..100]
                    trust_norm = (float(dims.trust) + 100.0) / 2.0
                    intimacy = max(0.0, float(dims.intimacy))
                    affection_pos = max(0.0, float(dims.affection))
                    frequency = max(0.0, float(dims.frequency))
                    closeness = (intimacy + affection_pos + frequency) / 3.0

                    payload = {
                        "entity_id": entity_id,
                        "entity_name": entity_name,
                        "dimensions": dims.to_dict(),
                        "patterns": list(state.history.active_patterns),
                        "archetypes": list(state.active_archetypes),
                        "trust_norm": round(trust_norm, 1),
                        "closeness": round(closeness, 1),
                        "link_id": state.link_id,
                        "version": state.version
                    }

                    # Add small memory context if memory system available (best-effort)
                    if memory_system is not None:
                        try:
                            mem_res = await memory_system.recall(
                                entity_type="npc",
                                entity_id=self.npc_id,
                                query=entity_name,
                                limit=3
                            )
                            payload["memory_context"] = mem_res.get("memories", []) or []
                        except Exception as me:
                            payload["memory_context"] = []
                            logger.debug(f"[Perception] memory context failed for {entity_name}: {me}")

                    # Use stable key to avoid collisions with multiple NPCs
                    key = f"{entity_type}_{entity_id}"
                    relationships[key] = payload

                except Exception as re:
                    logger.debug(f"[Perception] relationship fetch failed for entity {entity}: {re}")
                    continue

            return relationships

    async def evaluate_action_significance(
        self,
        action: Dict[str, Any],
        result: Dict[str, Any]
    ) -> ActionSignificance:
        """
        Determine if an action is significant enough to create a memory.

        Args:
            action: Action data
            result: Result data

        Returns:
            ActionSignificance object
        """
        with function_span("evaluate_action_significance"):
            # Actions that are always significant
            always_significant = {
                "talk", "command", "mock", "confide", "praise",
                "emotional_outburst", "mask_slip"
            }

            if (action or {}).get("type") in always_significant:
                return ActionSignificance(
                    is_significant=True,
                    level=2,  # Medium significance
                    reason=f"Action type '{action.get('type')}' is inherently significant"
                )

            # Check for surprising outcomes
            outcome = (result or {}).get("outcome", "") or ""
            lowered_outcome = outcome.lower()
            if any(term in lowered_outcome for term in ["surprising", "unexpected", "shocked", "unusual"]):
                return ActionSignificance(
                    is_significant=True,
                    level=2,
                    reason="Outcome was surprising or unexpected"
                )

            # Check for emotional impact
            try:
                emotional_impact = int((result or {}).get("emotional_impact", 0))
            except Exception:
                emotional_impact = 0

            if abs(emotional_impact) >= 3:
                return ActionSignificance(
                    is_significant=True,
                    level=3,  # High significance
                    reason=f"Strong emotional impact ({emotional_impact})"
                )
            elif abs(emotional_impact) >= 1:
                return ActionSignificance(
                    is_significant=True,
                    level=1,  # Low significance
                    reason=f"Moderate emotional impact ({emotional_impact})"
                )

            # Default - not significant enough
            return ActionSignificance(
                is_significant=False,
                level=0,
                reason="Action was routine with minimal impact"
            )

# -------------------------------------------------------
# Helper functions for environment perception
# -------------------------------------------------------

def get_entity_name(entity: Dict[str, Any]) -> str:
    """
    Get a display name for an entity.

    Args:
        entity: Entity data

    Returns:
        Entity display name
    """
    entity_type = entity.get("type", "unknown")
    entity_id = entity.get("id", 0)
    entity_name = entity.get("name", f"{entity_type}_{entity_id}")
    return entity_name

async def get_npc_location(npc_id: int, user_id: int, conversation_id: int) -> str:
    """
    Get an NPC's current location.

    Args:
        npc_id: NPC ID
        user_id: User ID
        conversation_id: Conversation ID

    Returns:
        Location name
    """
    try:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                """
                SELECT current_location
                FROM NPCStats
                WHERE npc_id = $1
                  AND user_id = $2
                  AND conversation_id = $3
                """,
                int(npc_id), int(user_id), int(conversation_id)
            )
            return row["current_location"] if row and row["current_location"] else "Unknown"
    except Exception as e:
        logger.error(f"[Perception] Error getting NPC location: {e}")
        return "Unknown"
