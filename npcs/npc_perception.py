# npcs/npc_perception.py

"""
Environment perception tools for NPC agents.
Refactored from the original environment_perception.py.
"""

import logging
import asyncio
import random
import time
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field

from agents import function_tool, trace
from agents.tracing import custom_span, function_span
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

# -------------------------------------------------------
# Main EnvironmentPerception class
# -------------------------------------------------------

class EnvironmentPerception:
    """
    Provides environment perception tools for NPC agents.
    Integrates environment data with memory retrieval.
    """
    
    def __init__(self, npc_id: int, user_id: int, conversation_id: int):
        """
        Initialize environment perception for an NPC.
        
        Args:
            npc_id: ID of the NPC
            user_id: ID of the user/player
            conversation_id: ID of the conversation
        """
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._memory_system = None
        self._cache = {}
        self._cache_timestamp = {}
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
        if key not in self._cache or key not in self._cache_timestamp:
            return False
        
        now = time.time()
        return now - self._cache_timestamp[key] < self._cache_ttl
    
    def _update_cache(self, key: str, value: Any) -> None:
        """Update a cache entry."""
        self._cache[key] = value
        self._cache_timestamp[key] = time.time()
    
    def _invalidate_cache(self, key: Optional[str] = None) -> None:
        """Invalidate cache entries."""
        if key is None:
            self._cache.clear()
            self._cache_timestamp.clear()
        elif key in self._cache:
            del self._cache[key]
            if key in self._cache_timestamp:
                del self._cache_timestamp[key]
    
    async def perceive_environment(
        self, 
        context: Dict[str, Any],
        detail_level: str = "auto"  # "low", "standard", "high", "auto"
    ) -> PerceptionResult:
        """
        Perceive environment with adaptive detail levels based on system load.
        
        Args:
            context: Context information
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
                except ImportError:
                    detail_level = "standard"
            
            # Create a cache key based on context
            cache_key = f"perception_{hash(str(context))}"
            
            # Check cache
            if self._is_cache_valid(cache_key):
                result = self._cache[cache_key]
                elapsed = time.time() - perf_start
                
                # Convert to PerceptionResult if it's a dict
                if isinstance(result, dict):
                    result = PerceptionResult(**result)
                
                # Update processing time and cache hit
                result.processing_time = elapsed
                result.cache_hit = True
                
                return result
            
            # Set memory limits based on detail level
            if detail_level == "low":
                memory_limit = 3
                include_flashbacks = False
            elif detail_level == "high":
                memory_limit = 8
                include_flashbacks = True
            else:
                memory_limit = 5
                include_flashbacks = random.random() < 0.3
            
            try:
                # Extract context information
                perception_context = PerceptionContext(**context) if not isinstance(context, PerceptionContext) else context
                
                # Get environment data
                environment_data = await self.fetch_environment_data(perception_context)
                
                # Get memory system
                memory_system = await self.get_memory_system()
                
                # Construct query for memory retrieval
                query_text = perception_context.query_text or perception_context.description or ""
                if hasattr(perception_context, "player_action") and perception_context.player_action:
                    player_action = perception_context.player_action
                    if "description" in player_action:
                        query_text += f" {player_action['description']}"
                
                # Get location and time context
                context_for_recall = {
                    "text": query_text,
                    "location": environment_data.get("location", "Unknown"),
                    "time_of_day": environment_data.get("time_of_day", "Unknown")
                }
                
                # Add entities present to context
                if "entities_present" in environment_data:
                    context_for_recall["entities_present"] = [
                        e.get("name", "") for e in environment_data.get("entities_present", [])
                    ]
                
                # Get emotional state
                emotional_state = await memory_system.get_npc_emotion(self.npc_id)
                
                # Include emotional state in context if strong
                if emotional_state and "current_emotion" in emotional_state:
                    current_emotion = emotional_state["current_emotion"]
                    
                    # Handle different data structures
                    if isinstance(current_emotion.get("primary", {}), dict):
                        emotion_name = current_emotion.get("primary", {}).get("name", "neutral")
                        intensity = current_emotion.get("primary", {}).get("intensity", 0.0)
                    else:
                        emotion_name = current_emotion.get("primary", "neutral")
                        intensity = current_emotion.get("intensity", 0.0)
                    
                    if intensity > 0.6:
                        context_for_recall["emotional_state"] = {
                            "primary_emotion": emotion_name,
                            "intensity": intensity
                        }
                
                # Determine adaptive memory limit based on context importance
                base_limit = memory_limit
                adaptive_limit = base_limit
                
                if query_text:
                    context_importance = self._calculate_context_importance(query_text)
                    
                    if context_importance >= 3:
                        adaptive_limit = base_limit + 5
                    elif context_importance >= 1:
                        adaptive_limit = base_limit + 2
                
                # Retrieve memories
                memory_result = await memory_system.recall(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    context=context_for_recall,
                    limit=adaptive_limit
                )
                relevant_memories = memory_result.get("memories", [])
                
                # Check for traumatic triggers
                traumatic_trigger = None
                if query_text:
                    trigger_result = await memory_system.emotional_manager.process_traumatic_triggers(
                        entity_type="npc",
                        entity_id=self.npc_id,
                        text=query_text
                    )
                    if trigger_result and trigger_result.get("triggered", False):
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
                
                # Check for flashback
                flashback = None
                flashback_chance = 0.15
                if traumatic_trigger:
                    flashback_chance = 0.5
                    
                if include_flashbacks and random.random() < flashback_chance:
                    flashback = await memory_system.npc_flashback(
                        npc_id=self.npc_id,
                        context=query_text
                    )
                
                # Get mask info
                mask_info = await memory_system.get_npc_mask(self.npc_id)
                if mask_info and "integrity" not in mask_info:
                    mask_info["integrity"] = 100
                
                # Get relationships
                relationship_data = await self.fetch_relationships(environment_data)
                
                # Get beliefs
                beliefs = await memory_system.get_beliefs(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    topic="player"
                )
                if beliefs:
                    beliefs = sorted(beliefs, key=lambda x: x.get("confidence", 0), reverse=True)
                
                # Get time context
                time_context = await self.fetch_time_context()
                
                # Get narrative context
                narrative_context = await self.fetch_narrative_context()
                
                # Build final perception result
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
                self._update_cache(cache_key, result.dict())
                
                return result
                
            except Exception as e:
                logger.error(f"Error in environment perception: {e}")
                
                # Return minimal perception on error
                elapsed = time.time() - perf_start
                return PerceptionResult(
                    environment=environment_data if 'environment_data' in locals() else {},
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
        lowered = context_text.lower()
        
        # High importance keywords
        high_keywords = ["critical", "emergency", "dangerous", "threat", "crucial", 
                         "sex", "intimate", "fight", "attack", "betrayal"]
        
        # Medium importance keywords
        medium_keywords = ["important", "significant", "unusual", "strange", 
                           "unexpected", "surprise", "secret", "revealed"]
        
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
                    environment_data = {
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
                    if day_names and environment_data["day"]:
                        # Calculate which day of week it is
                        day_index = (environment_data["day"] - 1) % len(day_names)
                        environment_data["day_name"] = day_names[day_index]
                except Exception as e:
                    logger.debug(f"Could not load calendar day names: {e}")
    
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
                        if row:
                            environment_data["location"] = row[0]
                
                # Extract time of day from context if available
                if context.time_of_day:
                    environment_data["time_of_day"] = context.time_of_day
                
                # Fetch time of day from database if not in context (fixed duplicate SELECT)
                if environment_data["time_of_day"] == "Unknown":
                    async with get_db_connection_context() as conn:
                        row = await conn.fetchrow(
                            """
                            SELECT value FROM CurrentRoleplay
                            WHERE key = 'TimeOfDay' AND user_id = $1 AND conversation_id = $2
                            """,
                            self.user_id, self.conversation_id
                        )
                        if row:
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
                        
                        entities = []
                        for row in rows:
                            if row["npc_id"] != self.npc_id:  # Don't include self
                                entities.append({
                                    "type": "npc",
                                    "id": row["npc_id"],
                                    "name": row["npc_name"]
                                })
                        
                        # Add player entity
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
                logger.error(f"Error fetching environment data: {e}")
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
                logger.error(f"Error fetching time context: {e}")
                return time_context
    
    async def fetch_narrative_context(self) -> Dict[str, Any]:
        """
        Fetch the current narrative context (plot stage, tension) from the database.
        
        Returns:
            Dictionary with narrative context
        """
        with function_span("fetch_narrative_context"):
            narrative_context = {}
            
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
                logger.error(f"Error fetching narrative context: {e}")
                return narrative_context
    
    async def fetch_relationships(self, environment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch relationships with entities in the environment.
        
        Args:
            environment_data: Environment data with entities_present
            
        Returns:
            Dictionary with relationship data
        """
        with function_span("fetch_relationships"):
            relationships = {}
            
            try:
                # Get memory system
                memory_system = await self.get_memory_system()
                
                # Process each entity in the environment
                for entity in environment_data.get("entities_present", []):
                    entity_type = entity.get("type")
                    entity_id = entity.get("id")
                    entity_name = entity.get("name")
                    
                    if not entity_type or not entity_id:
                        continue
                    
                    # Get relationship from database
                    async with get_db_connection_context() as conn:
                        row = await conn.fetchrow(
                            """
                            SELECT link_type, link_level
                            FROM SocialLinks
                            WHERE user_id = $1
                              AND conversation_id = $2
                              AND entity1_type = 'npc'
                              AND entity1_id = $3
                              AND entity2_type = $4
                              AND entity2_id = $5
                            """,
                            self.user_id, self.conversation_id, self.npc_id, entity_type, entity_id
                        )
                    
                    # Default relationship
                    relationship = {
                        "entity_id": entity_id,
                        "entity_name": entity_name,
                        "link_type": "neutral",
                        "link_level": 50
                    }
                    
                    if row:
                        relationship["link_type"] = row["link_type"]
                        relationship["link_level"] = row["link_level"]
                    
                    # Get memories about this entity
                    memory_result = await memory_system.recall(
                        entity_type="npc",
                        entity_id=self.npc_id,
                        query=entity_name,
                        limit=3
                    )
                    relationship["memory_context"] = memory_result.get("memories", [])
                    
                    # Add to relationships
                    relationships[entity_type] = relationship
                
                return relationships
                
            except Exception as e:
                logger.error(f"Error fetching relationships: {e}")
                return {}
    
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
            
            if action.get("type") in always_significant:
                return ActionSignificance(
                    is_significant=True,
                    level=2,  # Medium significance
                    reason=f"Action type '{action.get('type')}' is inherently significant"
                )
            
            # Check for surprising outcomes
            outcome = result.get("outcome", "")
            if any(term in outcome.lower() for term in ["surprising", "unexpected", "shocked", "unusual"]):
                return ActionSignificance(
                    is_significant=True,
                    level=2,
                    reason="Outcome was surprising or unexpected"
                )
            
            # Check for emotional impact
            emotional_impact = result.get("emotional_impact", 0)
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
                npc_id, user_id, conversation_id
            )
            return row["current_location"] if row else "Unknown"
    except Exception as e:
        logger.error(f"Error getting NPC location: {e}")
        return "Unknown"
