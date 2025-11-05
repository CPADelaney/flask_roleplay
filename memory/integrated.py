# memory/integrated.py

import logging
import random
import json
import asyncio
import numpy as np
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from .core import (
    Memory,
    MemoryType,
    MemorySignificance,
    UnifiedMemoryManager,
    normalize_memory_type,
)
from .schemas import MemorySchemaManager
from .emotional import EmotionalMemoryManager
from .interference import MemoryInterferenceManager
from .flashbacks import FlashbackManager
from .semantic import SemanticMemoryManager
from .reconsolidation import ReconsolidationManager
from .masks import ProgressiveRevealManager
from db.connection import get_db_connection_context
from utils.caching import get, set, delete

# Import shutdown protection
try:
    from db.connection import is_shutting_down, track_operation
    HAS_SHUTDOWN_PROTECTION = True
except ImportError:
    HAS_SHUTDOWN_PROTECTION = False
    
    def is_shutting_down():
        return False
    
    async def track_operation(coro):
        return await coro


# ---------------------------------------------------------------------------
# Timeout knobs for non-critical updates (fail-soft in foreground paths)
# ---------------------------------------------------------------------------
def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


EMO_UPDATE_TIMEOUT = float(os.getenv("MEM_EMO_UPDATE_TIMEOUT", "3.0"))
EMO_UPDATE_SOFT_FAIL = _env_bool("MEM_EMO_UPDATE_SOFT_FAIL", True)


logger = logging.getLogger("memory_integrated")

# Decorator for shutdown protection
def protect_from_shutdown(return_value=None):
    """Decorator to protect methods from shutdown errors."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if is_shutting_down():
                logger.debug(f"Skipping {func.__name__} - worker shutting down")
                return return_value
            
            try:
                if HAS_SHUTDOWN_PROTECTION:
                    return await track_operation(func(*args, **kwargs))
                else:
                    return await func(*args, **kwargs)
            except ConnectionError as e:
                if "shutting down" in str(e).lower():
                    logger.info(f"Gracefully skipping {func.__name__} during shutdown")
                    return return_value
                raise
        return wrapper
    return decorator


class IntegratedMemorySystem:
    """
    Integrated memory system with shutdown protection.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the integrated memory system with all subsystems."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize all memory subsystems
        self.core_manager = UnifiedMemoryManager(
            entity_type="integrated", 
            entity_id=0,
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        self.schema_manager = MemorySchemaManager(user_id, conversation_id)
        self.emotional_manager = EmotionalMemoryManager(user_id, conversation_id)
        self.interference_manager = MemoryInterferenceManager(user_id, conversation_id)
        self.flashback_manager = FlashbackManager(user_id, conversation_id)
        self.semantic_manager = SemanticMemoryManager(user_id, conversation_id)
        self.reconsolidation_manager = ReconsolidationManager(user_id, conversation_id)
        self.mask_manager = ProgressiveRevealManager(user_id, conversation_id)
        
        # Track memory events
        self.events = []
    
    def _safe_bool(self, value: Any) -> bool:
        """Safely convert a value to boolean, handling numpy arrays."""
        if value is None:
            return False
            
        if isinstance(value, (np.ndarray, list)):
            try:
                return bool(np.any(value))
            except:
                return bool(value) if value else False
        elif hasattr(value, '__array__'):
            try:
                return bool(np.any(value))
            except:
                return bool(value) if value else False
        else:
            return bool(value)
    
    def _safe_get(self, dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Safely get a value from a dictionary, handling array-like values."""
        if not isinstance(dictionary, dict):
            return default
            
        value = dictionary.get(key, default)
        
        if isinstance(default, bool) and value is not None:
            return self._safe_bool(value)
        
        return value
    
    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert a value to float, handling numpy arrays."""
        try:
            if value is None:
                return default
                
            if isinstance(value, (np.ndarray, list)):
                if len(value) > 0:
                    return float(np.mean(value))
                return default
            elif hasattr(value, '__array__'):
                return float(np.mean(value))
            else:
                return float(value)
        except (TypeError, ValueError):
            return default
    
    def _safe_comparison(self, value1: Any, operator: str, value2: Any) -> bool:
        """Safely compare two values, handling arrays."""
        if operator in ['<', '>', '<=', '>=']:
            v1 = self._safe_float(value1, 0.0)
            v2 = self._safe_float(value2, 0.0)
            
            if operator == '<':
                return v1 < v2
            elif operator == '>':
                return v1 > v2
            elif operator == '<=':
                return v1 <= v2
            elif operator == '>=':
                return v1 >= v2
        
        if operator == '==':
            return self._safe_bool(value1 == value2)
        elif operator == '!=':
            return self._safe_bool(value1 != value2)
        
        return False
    
    @protect_from_shutdown(return_value={"error": "Worker shutting down"})
    async def add_memory(self,
                      entity_type: str,
                      entity_id: int,
                      memory_text: str,
                      memory_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add a memory with integrated processing across all memory systems."""
        memory_kwargs = memory_kwargs or {}
        results = {"memory_text": memory_text}
        
        try:
            # 1. Analyze emotional content if not provided
            has_emotional_params = (
                "emotional_intensity" in memory_kwargs or 
                "primary_emotion" in memory_kwargs
            )
            
            if not has_emotional_params:
                emotion_analysis = await self.emotional_manager.analyze_emotional_content(memory_text)
                results["emotion_analysis"] = emotion_analysis
                
                primary_emotion = emotion_analysis.get("primary_emotion", "neutral")
                emotion_intensity = self._safe_float(
                    emotion_analysis.get("intensity", 0.5), 
                    default=0.5
                )
                
                memory_result = await self.emotional_manager.add_emotional_memory(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    memory_text=memory_text,
                    primary_emotion=primary_emotion,
                    emotion_intensity=emotion_intensity,
                    secondary_emotions=emotion_analysis.get("secondary_emotions", {}),
                    significance=memory_kwargs.get("significance", MemorySignificance.MEDIUM),
                    tags=memory_kwargs.get("tags", [])
                )
                
                memory_id = memory_result["memory_id"]
                results["memory_id"] = memory_id
                
            else:
                # Direct core memory creation
                emotional_intensity = self._safe_float(
                    memory_kwargs.get("emotional_intensity", 50),
                    default=50
                )
                
                normalized_type = normalize_memory_type(
                    memory_kwargs.get("memory_type")
                )

                memory = Memory(
                    text=memory_text,
                    memory_type=normalized_type,
                    significance=memory_kwargs.get("significance", MemorySignificance.MEDIUM),
                    emotional_intensity=emotional_intensity,
                    tags=memory_kwargs.get("tags", []),
                    metadata=memory_kwargs.get("metadata", {}),
                    timestamp=datetime.now()
                )
                
                core_manager = UnifiedMemoryManager(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    user_id=self.user_id,
                    conversation_id=self.conversation_id
                )
                
                memory_id = await core_manager.add_memory(memory)
                results["memory_id"] = memory_id
            
            # 2. Apply schema processing if requested
            if self._safe_get(memory_kwargs, "apply_schemas", True):
                schema_result = await self.schema_manager.apply_schema_to_memory(
                    memory_id=memory_id,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    auto_detect=True
                )
                results["schema_processing"] = schema_result
            
            # 3. Check for memory interference
            if self._safe_get(memory_kwargs, "check_interference", True):
                interference_result = await self.interference_manager.detect_memory_interference(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    memory_id=memory_id
                )
                results["interference_check"] = interference_result
                
                high_interference = self._safe_get(
                    interference_result, 
                    "high_interference_risk", 
                    False
                )
                
                if high_interference:
                    retroactive = interference_result.get("retroactive_interference", [])
                    proactive = interference_result.get("proactive_interference", [])
                    
                    all_interference = retroactive + proactive
                    if all_interference:
                        all_interference.sort(
                            key=lambda x: self._safe_float(x.get("similarity", 0)), 
                            reverse=True
                        )
                        interfering_memory_id = all_interference[0].get("memory_id") if all_interference else None
                        
                        if interfering_memory_id and random.random() < 0.3:
                            blend_result = await self.interference_manager.generate_blended_memory(
                                entity_type=entity_type,
                                entity_id=entity_id,
                                memory1_id=memory_id,
                                memory2_id=interfering_memory_id
                            )
                            results["memory_blend_created"] = blend_result
            
            # 4. Generate semantic memory if significant
            significance = memory_kwargs.get("significance", MemorySignificance.MEDIUM)
            if significance >= MemorySignificance.HIGH:
                semantic_result = await self.semantic_manager.generate_semantic_memory(
                    source_memory_id=memory_id,
                    entity_type=entity_type,
                    entity_id=entity_id
                )
                results["semantic_memory"] = semantic_result
            
            # 5. Apply mask processing for NPCs
            check_mask = self._safe_get(memory_kwargs, "check_mask_impact", True)
            if entity_type == "npc" and check_mask:
                mask_result = await self.mask_manager.get_npc_mask(entity_id)
                has_mask = mask_result and "error" not in mask_result
                integrity = self._safe_float(mask_result.get("integrity", 0)) if has_mask else 0
                
                if has_mask and self._safe_comparison(integrity, '>', 20):
                    if random.random() < 0.1:
                        slippage_result = await self.mask_manager.generate_mask_slippage(
                            npc_id=entity_id,
                            trigger=f"the memory: {memory_text[:50]}..."
                        )
                        results["mask_slippage"] = slippage_result
            
            # 6. Update entity emotional state
            emotion_update = {}
            try:
                # Prevent a slow DB acquire / pgvector registration from blocking the hot path
                emotion_update = await asyncio.wait_for(
                    self.update_emotional_state_from_memory(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        memory_id=memory_id,
                        memory_text=memory_text
                    ),
                    timeout=EMO_UPDATE_TIMEOUT,
                )
            except (asyncio.TimeoutError, ConnectionError) as e:
                if EMO_UPDATE_SOFT_FAIL:
                    logger.warning(
                        "Emotional state update skipped (timeout=%ss): %s",
                        EMO_UPDATE_TIMEOUT,
                        e,
                    )
                    emotion_update = {
                        "skipped": True,
                        "reason": "timeout",
                        "timeout_s": EMO_UPDATE_TIMEOUT,
                    }
                else:
                    raise
            results["emotional_state_update"] = emotion_update
            
            # Record event
            self._record_memory_event("add", entity_type, entity_id, memory_id, results)
            
            return results
            
        except ConnectionError as e:
            if "shutting down" in str(e).lower():
                logger.info(f"Memory add gracefully skipped during shutdown")
                return {"error": "Worker shutting down", "memory_text": memory_text}
            raise
        except Exception as e:
            logger.error(f"Error in integrated memory add: {e}", exc_info=True)
            results["error"] = str(e)
            return results
    
    @protect_from_shutdown(return_value={"memories": [], "error": "Worker shutting down"})
    async def retrieve_memories(self,
                             entity_type: str,
                             entity_id: int,
                             query: str = None,
                             current_context: Dict[str, Any] = None,
                             retrieval_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Retrieve memories with integrated processing."""
        retrieval_kwargs = retrieval_kwargs or {}
        current_context = current_context or {}
        results = {"query": query}
        
        try:
            # Different retrieval strategies based on emotional state
            emotional_state = None
            if self._safe_get(retrieval_kwargs, "use_emotional_context", True):
                emotional_state = await self.emotional_manager.get_entity_emotional_state(
                    entity_type=entity_type,
                    entity_id=entity_id
                )
                
                # Check for traumatic triggers
                if emotional_state and "text" in current_context:
                    trigger_text = current_context["text"]
                    trigger_result = await self.emotional_manager.process_traumatic_triggers(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        text=trigger_text
                    )
                    
                    triggered = self._safe_get(trigger_result, "triggered", False) if trigger_result else False
                    
                    if triggered:
                        results["trauma_triggered"] = trigger_result
                        trauma_memories = trigger_result.get("triggered_memories", [])
                        if trauma_memories:
                            results["traumatic_recall"] = True
                            results["memories"] = trauma_memories
                            
                            trauma_memory_ids = []
                            for m in trauma_memories:
                                if isinstance(m, dict) and "id" in m:
                                    trauma_memory_ids.append(m["id"])
                                elif hasattr(m, 'id'):
                                    trauma_memory_ids.append(m.id)
                            
                            self._record_memory_event("traumatic_recall", entity_type, entity_id, 
                                                     trauma_memory_ids, 
                                                     trigger_result)
                            return results
                
                # Mood-congruent recall
                current_emotion = emotional_state.get("current_emotion", {}) if emotional_state else {}
                emotion_intensity = self._safe_float(current_emotion.get("intensity", 0), 0)
                
                if self._safe_comparison(emotion_intensity, '>', 0.6):
                    mood_memories = await self.emotional_manager.retrieve_mood_congruent_memories(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        current_mood=current_emotion,
                        limit=retrieval_kwargs.get("limit", 5)
                    )
                    
                    if mood_memories and isinstance(mood_memories, list):
                        results["mood_congruent_recall"] = True
                        results["current_emotion"] = current_emotion
                        results["memories"] = mood_memories
                        
                        mood_memory_ids = []
                        for m in mood_memories:
                            if isinstance(m, dict) and "id" in m:
                                mood_memory_ids.append(m["id"])
                            elif hasattr(m, 'id'):
                                mood_memory_ids.append(m.id)
                        
                        self._record_memory_event("mood_congruent_recall", entity_type, entity_id,
                                                 mood_memory_ids,
                                                 {"current_emotion": current_emotion})
                        
                        force_mood = self._safe_get(retrieval_kwargs, "force_mood_congruent", False)
                        if force_mood or self._safe_comparison(emotion_intensity, '>', 0.8):
                            return results
            
            # Standard memory retrieval
            memory_manager = UnifiedMemoryManager(
                entity_type=entity_type,
                entity_id=entity_id,
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            memory_types = retrieval_kwargs.get("memory_types", None)
            tags = retrieval_kwargs.get("tags", None)
            limit = retrieval_kwargs.get("limit", 5)
            min_significance = retrieval_kwargs.get("min_significance", 0)
            
            # Memory competition simulation
            if self._safe_get(retrieval_kwargs, "simulate_competition", True) and query:
                competition_result = await self.interference_manager.simulate_memory_competition(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    query=query,
                    competition_count=limit
                )
                
                if competition_result:
                    results["memory_competition"] = True
                    results["winner"] = competition_result.get("winner")
                    results["competing_memories"] = competition_result.get("competing_memories", [])
                    
                    if "memory_blend" in competition_result:
                        results["memory_blend"] = competition_result["memory_blend"]
                    
                    winner_memory = competition_result.get("winner", {})
                    winner_id = winner_memory.get("id") if winner_memory else None
                    self._record_memory_event("memory_competition", entity_type, entity_id,
                                             winner_id,
                                             competition_result)
                    
                    if self._safe_get(retrieval_kwargs, "competition_only", False):
                        return results
            
            # Standard retrieval
            memories = await memory_manager.retrieve_memories(
                query=query,
                memory_types=memory_types,
                tags=tags,
                min_significance=min_significance,
                limit=limit
            )
            
            if not memories:
                memories = []
            
            # Get memory details
            detailed_memories = []
            for memory in memories:
                if not memory:
                    continue
                    
                # Schema interpretations
                interpretation = None
                if self._safe_get(retrieval_kwargs, "include_schema_interpretation", True):
                    try:
                        interpretation_result = await self.schema_manager.interpret_memory_with_schemas(
                            memory_id=getattr(memory, 'id', None),
                            entity_type=entity_type,
                            entity_id=entity_id
                        )
                        
                        if interpretation_result and "interpretation" in interpretation_result:
                            interpretation = interpretation_result["interpretation"]
                    except Exception as e:
                        logger.error(f"Error getting schema interpretation: {e}")
                
                # Reconsolidate on recall
                if self._safe_get(retrieval_kwargs, "reconsolidate_on_recall", True):
                    try:
                        recon_context = {}
                        if emotional_state and "current_emotion" in emotional_state:
                            recon_context["emotional_context"] = emotional_state["current_emotion"]
                        
                        await self.reconsolidation_manager.reconsolidate_memory(
                            memory_id=getattr(memory, 'id', None),
                            entity_type=entity_type,
                            entity_id=entity_id,
                            emotional_context=recon_context,
                            recall_context=str(current_context),
                            alteration_strength=0.05
                        )
                    except Exception as e:
                        logger.error(f"Error in memory reconsolidation: {e}")
                
                memory_dict = {
                    "id": getattr(memory, 'id', None),
                    "text": getattr(memory, 'text', ''),
                    "type": getattr(memory, 'memory_type', 'unknown'),
                    "significance": getattr(memory, 'significance', 0),
                    "emotional_intensity": self._safe_float(getattr(memory, 'emotional_intensity', 0)),
                    "timestamp": memory.timestamp.isoformat() if hasattr(memory, 'timestamp') and memory.timestamp else None,
                    "tags": getattr(memory, 'tags', []),
                    "schema_interpretation": interpretation
                }
                detailed_memories.append(memory_dict)
            
            # Random flashback opportunity
            if self._safe_get(retrieval_kwargs, "allow_flashbacks", True) and random.random() < 0.15:
                try:
                    flashback = await self.flashback_manager.generate_flashback(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        current_context=current_context.get("text", query or "")
                    )
                    
                    if flashback:
                        results["flashback"] = flashback
                except Exception as e:
                    logger.error(f"Error generating flashback: {e}")
            
            # Intrusive memory opportunity
            if self._safe_get(retrieval_kwargs, "allow_intrusive", True) and random.random() < 0.1:
                try:
                    intrusion = await self.interference_manager.generate_intrusive_memory(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        context=current_context.get("text", query or "")
                    )
                    
                    intrusion_generated = self._safe_get(intrusion, "intrusion_generated", False) if intrusion else False
                    if intrusion_generated:
                        results["intrusive_memory"] = intrusion
                except Exception as e:
                    logger.error(f"Error generating intrusive memory: {e}")
            
            results["memories"] = detailed_memories
            
            memory_ids = [m.id if hasattr(m, 'id') else m.get('id', None) for m in memories if m]
            memory_ids = [mid for mid in memory_ids if mid is not None]
            self._record_memory_event("recall", entity_type, entity_id,
                                     memory_ids,
                                     {"query": query, "count": len(memories)})
            
            return results
            
        except ConnectionError as e:
            if "shutting down" in str(e).lower():
                logger.info(f"Memory retrieval gracefully skipped during shutdown")
                return {"memories": [], "error": "Worker shutting down"}
            raise
        except Exception as e:
            import traceback
            logger.error(f"Error in integrated memory retrieval: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            results["error"] = str(e)
            return results
        
    @protect_from_shutdown(return_value={"error": "Worker shutting down"})
    async def update_emotional_state_from_memory(self,
                                             entity_type: str,
                                             entity_id: int,
                                             memory_id: int,
                                             memory_text: str) -> Dict[str, Any]:
        """Update entity's emotional state based on a new memory."""
        try:
            emotion_analysis = await self.emotional_manager.analyze_emotional_content(memory_text)
            
            current_emotion = {
                "primary_emotion": emotion_analysis.get("primary_emotion", "neutral"),
                "intensity": self._safe_float(emotion_analysis.get("intensity", 0.3), 0.3),
                "secondary_emotions": emotion_analysis.get("secondary_emotions", {}),
                "valence": self._safe_float(emotion_analysis.get("valence", 0.0), 0.0),
                "arousal": self._safe_float(emotion_analysis.get("arousal", 0.0), 0.0)
            }
            
            valence = self._safe_float(emotion_analysis.get("valence", 0), 0.0)
            intensity = self._safe_float(emotion_analysis.get("intensity", 0), 0.0)
            
            is_traumatic = (
                self._safe_comparison(valence, '<', -0.7) and 
                self._safe_comparison(intensity, '>', 0.7)
            )
            
            if is_traumatic:
                trauma_event = {
                    "memory_id": memory_id,
                    "memory_text": memory_text,
                    "emotion": current_emotion["primary_emotion"],
                    "intensity": current_emotion["intensity"],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                trauma_event = None
            
            updated_state = await self.emotional_manager.update_entity_emotional_state(
                entity_type=entity_type,
                entity_id=entity_id,
                current_emotion=current_emotion,
                trauma_event=trauma_event
            )
            
            return {
                "updated_emotional_state": True,
                "primary_emotion": current_emotion["primary_emotion"],
                "intensity": current_emotion["intensity"],
                "is_traumatic": is_traumatic
            }
            
        except ConnectionError as e:
            if "shutting down" in str(e).lower():
                logger.info(f"Emotional state update gracefully skipped during shutdown")
                return {"error": "Worker shutting down"}
            raise
        except Exception as e:
            logger.error(f"Error updating emotional state: {e}", exc_info=True)
            return {"error": str(e)}
    
    @protect_from_shutdown(return_value={"schema_detected": False, "error": "Worker shutting down"})
    async def generate_schemas_from_memories(self,
                                        entity_type: str,
                                        entity_id: int,
                                        tag_filter: List[str] = None) -> Dict[str, Any]:
        """Generate schemas by analyzing memory patterns."""
        try:
            result = await self.schema_manager.detect_schema_from_memories(
                entity_type=entity_type,
                entity_id=entity_id,
                tags=tag_filter
            )
            
            schema_detected = self._safe_get(result, "schema_detected", False)
            
            if schema_detected:
                return {
                    "schema_detected": True,
                    "schema_id": result.get("schema_id"),
                    "schema_name": result.get("schema_name"),
                    "already_exists": result.get("schema_already_exists", False)
                }
            else:
                return {
                    "schema_detected": False,
                    "reason": result.get("reason", "No pattern detected")
                }
                
        except Exception as e:
            logger.error(f"Error generating schemas: {e}", exc_info=True)
            return {"error": str(e)}
    
    @protect_from_shutdown(return_value={"error": "Worker shutting down"})
    async def analyze_entity_memories(self,
                                 entity_type: str,
                                 entity_id: int) -> Dict[str, Any]:
        """Perform a comprehensive analysis of an entity's memories."""
        try:
            memory_manager = UnifiedMemoryManager(
                entity_type=entity_type,
                entity_id=entity_id,
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            memories = await memory_manager.retrieve_memories(limit=100)
            
            memory_types = {}
            memory_tags = {}
            total_memories = len(memories)
            avg_significance = 0
            avg_emotional_intensity = 0
            
            if total_memories > 0:
                for memory in memories:
                    memory_type = memory.memory_type
                    if memory_type not in memory_types:
                        memory_types[memory_type] = 0
                    memory_types[memory_type] += 1
                    
                    for tag in memory.tags:
                        if tag not in memory_tags:
                            memory_tags[tag] = 0
                        memory_tags[tag] += 1
                    
                    avg_significance += self._safe_float(memory.significance)
                    avg_emotional_intensity += self._safe_float(memory.emotional_intensity)
                
                avg_significance /= total_memories
                avg_emotional_intensity /= total_memories
            
            emotional_state = await self.emotional_manager.get_entity_emotional_state(
                entity_type=entity_type,
                entity_id=entity_id
            )
            
            schemas = await self._get_entity_schemas(entity_type, entity_id)
            
            false_memory_status = await self.interference_manager.get_false_memory_status(
                entity_type=entity_type,
                entity_id=entity_id
            )
            
            analysis = {
                "memory_count": total_memories,
                "memory_types": memory_types,
                "common_tags": dict(sorted(memory_tags.items(), key=lambda x: x[1], reverse=True)[:10]),
                "avg_significance": avg_significance,
                "avg_emotional_intensity": avg_emotional_intensity,
                "emotional_state": emotional_state,
                "schemas": schemas,
                "false_memory_info": false_memory_status
            }
            
            if entity_type == "npc":
                mask_info = await self.mask_manager.get_npc_mask(entity_id)
                if mask_info and "error" not in mask_info:
                    analysis["mask"] = {
                        "integrity": self._safe_float(mask_info.get("integrity", 100)),
                        "presented_traits": mask_info.get("presented_traits", {}),
                        "hidden_traits": mask_info.get("hidden_traits", {})
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing entity memories: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def _get_entity_schemas(self, entity_type: str, entity_id: int) -> List[Dict[str, Any]]:
        """Get schemas associated with an entity."""
        if is_shutting_down():
            return []
            
        schemas = []
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT id, schema_name, category, schema_data
                    FROM MemorySchemas
                    WHERE entity_type = $1
                      AND entity_id = $2
                      AND user_id = $3
                      AND conversation_id = $4
                """, entity_type, entity_id, self.user_id, self.conversation_id)
                
                for row in rows:
                    schema_data = row["schema_data"] if isinstance(row["schema_data"], dict) else json.loads(row["schema_data"])
                    
                    schemas.append({
                        "id": row["id"],
                        "name": row["schema_name"],
                        "category": row["category"],
                        "description": schema_data.get("description", ""),
                        "confidence": self._safe_float(schema_data.get("confidence", 0.5))
                    })
        except ConnectionError as e:
            if "shutting down" in str(e).lower():
                logger.info("Skipping schema fetch during shutdown")
                return []
            raise
                
        return schemas
    
    @protect_from_shutdown(return_value={})
    async def run_memory_maintenance(self,
                                  entity_type: str,
                                  entity_id: int,
                                  maintenance_options: Dict[str, bool] = None) -> Dict[str, Any]:
        """Run maintenance tasks on entity's memory system."""
        options = maintenance_options or {
            "core_maintenance": True,
            "schema_maintenance": True,
            "emotional_decay": True,
            "background_reconsolidation": True,
            "interference_processing": True,
            "mask_checks": entity_type == "npc"
        }
        
        results = {}
        
        try:
            if self._safe_get(options, "core_maintenance", True):
                memory_manager = UnifiedMemoryManager(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    user_id=self.user_id,
                    conversation_id=self.conversation_id
                )
                
                maintenance_result = await memory_manager.perform_maintenance()
                results["core_maintenance"] = maintenance_result
            
            if self._safe_get(options, "schema_maintenance", True):
                schema_result = await self.schema_manager.run_schema_maintenance(
                    entity_type=entity_type,
                    entity_id=entity_id
                )
                results["schema_maintenance"] = schema_result
            
            if self._safe_get(options, "emotional_decay", True):
                emotional_result = await self.emotional_manager.emotional_decay_maintenance(
                    entity_type=entity_type,
                    entity_id=entity_id
                )
                results["emotional_decay"] = emotional_result
            
            if self._safe_get(options, "background_reconsolidation", True):
                recon_result = await self.reconsolidation_manager.check_memories_for_reconsolidation(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    max_memories=3
                )
                
                results["reconsolidation"] = {
                    "memories_reconsolidated": len(recon_result) if isinstance(recon_result, list) else 0
                }
            
            if self._safe_get(options, "interference_processing", True):
                interference_result = await self.interference_manager.run_interference_maintenance(
                    entity_type=entity_type,
                    entity_id=entity_id
                )
                results["interference_processing"] = interference_result
            
            if self._safe_get(options, "mask_checks", False) and entity_type == "npc":
                mask_result = await self.mask_manager.check_for_automated_reveals(self.user_id, self.conversation_id)
                
                results["mask_checks"] = {
                    "reveals_generated": len(mask_result) if isinstance(mask_result, list) else 0
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in memory maintenance: {e}", exc_info=True)
            results["error"] = str(e)
            return results
    
    def _record_memory_event(self, event_type: str, entity_type: str, entity_id: int, 
                           memory_ids: Union[int, List[int]], details: Dict[str, Any] = None):
        """Record a memory event for tracking and analysis."""
        if isinstance(memory_ids, int):
            memory_ids = [memory_ids]
            
        event = {
            "event_type": event_type,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "memory_ids": memory_ids,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        self.events.append(event)
        
        if len(self.events) > 100:
            self.events = self.events[-100:]
    
    async def initialize_tables(self):
        """Initialize all tables required by the memory system."""
        if is_shutting_down():
            logger.info("Skipping table initialization - worker shutting down")
            return
            
        from .core import UnifiedMemoryManager
        from .schemas import create_schema_tables
        from .flashbacks import create_flashback_tables
        from .emotional import create_emotional_tables
        from .reconsolidation import create_reconsolidation_tables
        from .semantic import create_semantic_tables
        
        await UnifiedMemoryManager.create_tables()
        await create_schema_tables()
        await create_flashback_tables()
        await create_emotional_tables()
        await create_reconsolidation_tables()
        await create_semantic_tables()
        
        logger.info("All memory system tables initialized")

async def init_memory_system(user_id: int, conversation_id: int) -> IntegratedMemorySystem:
    """Initialize the integrated memory system."""
    memory_system = IntegratedMemorySystem(user_id, conversation_id)
    await memory_system.initialize_tables()
    return memory_system
