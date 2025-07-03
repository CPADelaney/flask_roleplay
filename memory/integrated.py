# memory/integrated.py

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from .core import Memory, MemoryType, MemorySignificance, UnifiedMemoryManager
from .schemas import MemorySchemaManager
from .emotional import EmotionalMemoryManager
from .interference import MemoryInterferenceManager
from .flashbacks import FlashbackManager
from .semantic import SemanticMemoryManager
from .reconsolidation import ReconsolidationManager
from .masks import ProgressiveRevealManager
from db.connection import get_db_connection_context
from utils.caching import get, set, delete

logger = logging.getLogger("memory_integrated")

class IntegratedMemorySystem:
    """
    Integrated memory system that brings together all specialized memory components.
    This serves as the main entry point for memory operations, routing to the appropriate
    subsystem based on the operation type.
    
    Features:
    - Centralized API for all memory operations
    - Automated background maintenance
    - Event tracking for memory interactions
    - Memory statistics and health monitoring
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the integrated memory system with all subsystems.
        
        Args:
            user_id: ID of the user
            conversation_id: ID of the conversation
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize all memory subsystems
        self.core_manager = UnifiedMemoryManager(
            entity_type="integrated", 
            entity_id=0,  # Placeholder, will be overridden in methods
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
    
    async def add_memory(self,
                      entity_type: str,
                      entity_id: int,
                      memory_text: str,
                      memory_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Add a memory with integrated processing across all memory systems.
        
        Args:
            entity_type: Type of entity (e.g., "npc", "player")
            entity_id: ID of the entity
            memory_text: The memory text
            memory_kwargs: Additional memory parameters
            
        Returns:
            Comprehensive results from all memory processing
        """
        memory_kwargs = memory_kwargs or {}
        results = {"memory_text": memory_text}
        
        try:
            # 1. First, analyze emotional content if not provided
            if "emotional_intensity" not in memory_kwargs and "primary_emotion" not in memory_kwargs:
                emotion_analysis = await self.emotional_manager.analyze_emotional_content(memory_text)
                
                results["emotion_analysis"] = emotion_analysis
                
                # Extract emotional parameters for the memory
                primary_emotion = emotion_analysis.get("primary_emotion", "neutral")
                emotion_intensity = emotion_analysis.get("intensity", 0.5)
                
                # Create the memory with emotional context
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
                # Direct core memory creation if emotional parameters provided
                memory = Memory(
                    text=memory_text,
                    memory_type=memory_kwargs.get("memory_type", MemoryType.OBSERVATION),
                    significance=memory_kwargs.get("significance", MemorySignificance.MEDIUM),
                    emotional_intensity=memory_kwargs.get("emotional_intensity", 50),
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
            if memory_kwargs.get("apply_schemas", True):
                schema_result = await self.schema_manager.apply_schema_to_memory(
                    memory_id=memory_id,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    auto_detect=True
                )
                
                results["schema_processing"] = schema_result
            
            # 3. Check for memory interference
            if memory_kwargs.get("check_interference", True):
                interference_result = await self.interference_manager.detect_memory_interference(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    memory_id=memory_id
                )
                
                results["interference_check"] = interference_result
                
                # If high interference, potentially create a blended memory
                if interference_result.get("high_interference_risk", False):
                    # Get the highest interfering memory
                    retroactive = interference_result.get("retroactive_interference", [])
                    proactive = interference_result.get("proactive_interference", [])
                    
                    all_interference = retroactive + proactive
                    if all_interference:
                        # Sort by similarity
                        all_interference.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                        interfering_memory_id = all_interference[0].get("memory_id")
                        
                        # Create a blended memory with 30% probability
                        if interfering_memory_id and random.random() < 0.3:
                            blend_result = await self.interference_manager.generate_blended_memory(
                                entity_type=entity_type,
                                entity_id=entity_id,
                                memory1_id=memory_id,
                                memory2_id=interfering_memory_id
                            )
                            
                            results["memory_blend_created"] = blend_result
            
            # 4. Generate a semantic memory if significant
            significance = memory_kwargs.get("significance", MemorySignificance.MEDIUM)
            if significance >= MemorySignificance.HIGH:
                semantic_result = await self.semantic_manager.generate_semantic_memory(
                    source_memory_id=memory_id,
                    entity_type=entity_type,
                    entity_id=entity_id
                )
                
                results["semantic_memory"] = semantic_result
            
            # 5. Apply any mask processing for NPCs
            if entity_type == "npc" and memory_kwargs.get("check_mask_impact", True):
                # Get mask data
                mask_result = await self.mask_manager.get_npc_mask(entity_id)
                
                # If there's a mask with decent integrity left, check for triggers
                if mask_result and "error" not in mask_result and mask_result.get("integrity", 0) > 20:
                    # 10% chance to generate a slippage event
                    if random.random() < 0.1:
                        slippage_result = await self.mask_manager.generate_mask_slippage(
                            npc_id=entity_id,
                            trigger=f"the memory: {memory_text[:50]}..."
                        )
                        
                        results["mask_slippage"] = slippage_result
            
            # 6. Update entity emotional state
            emotion_update = await self.update_emotional_state_from_memory(
                entity_type=entity_type,
                entity_id=entity_id,
                memory_id=memory_id,
                memory_text=memory_text
            )
            
            results["emotional_state_update"] = emotion_update
            
            # Record this memory event
            self._record_memory_event("add", entity_type, entity_id, memory_id, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in integrated memory add: {e}")
            results["error"] = str(e)
            return results
    
    async def retrieve_memories(self,
                             entity_type: str,
                             entity_id: int,
                             query: str = None,
                             current_context: Dict[str, Any] = None,
                             retrieval_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Retrieve memories with integrated processing including schema application,
        emotional context, and interference effects.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            query: Query text to search for
            current_context: Current environmental and emotional context
            retrieval_kwargs: Additional retrieval parameters
            
        Returns:
            Retrieved memories with processing results
        """
        retrieval_kwargs = retrieval_kwargs or {}
        current_context = current_context or {}
        results = {"query": query}
        
        try:
            # Different retrieval strategies based on current emotional state
            emotional_state = None
            if retrieval_kwargs.get("use_emotional_context", True):
                emotional_state = await self.emotional_manager.get_entity_emotional_state(
                    entity_type=entity_type,
                    entity_id=entity_id
                )
                
                # Check for traumatic triggers first - only if emotional_state exists
                if emotional_state and "text" in current_context:
                    trigger_text = current_context["text"]
                    trigger_result = await self.emotional_manager.process_traumatic_triggers(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        text=trigger_text
                    )
                    
                    if trigger_result.get("triggered", False):
                        results["trauma_triggered"] = trigger_result
                        # If trauma triggered, we prioritize these memories
                        trauma_memories = trigger_result.get("triggered_memories", [])
                        if trauma_memories:
                            results["traumatic_recall"] = True
                            results["memories"] = trauma_memories
                            
                            # Record this memory event
                            self._record_memory_event("traumatic_recall", entity_type, entity_id, 
                                                     [m["id"] for m in trauma_memories], 
                                                     trigger_result)
                            
                            # Early return with trauma memories
                            return results
                
                # If in a strong emotional state, use mood-congruent recall
                current_emotion = emotional_state.get("current_emotion", {}) if emotional_state else {}
                if current_emotion.get("intensity", 0) > 0.6:
                    mood_memories = await self.emotional_manager.retrieve_mood_congruent_memories(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        current_mood=current_emotion,
                        limit=retrieval_kwargs.get("limit", 5)
                    )
                    
                    if mood_memories:
                        results["mood_congruent_recall"] = True
                        results["current_emotion"] = current_emotion
                        results["memories"] = mood_memories
                        
                        # Record this memory event
                        self._record_memory_event("mood_congruent_recall", entity_type, entity_id,
                                                 [m["id"] for m in mood_memories],
                                                 {"current_emotion": current_emotion})
                        
                        # Only use mood-congruent recall if specifically requested or emotion is very strong
                        if retrieval_kwargs.get("force_mood_congruent", False) or current_emotion.get("intensity", 0) > 0.8:
                            return results
            
            # Standard memory retrieval if no emotional override
            memory_manager = UnifiedMemoryManager(
                entity_type=entity_type,
                entity_id=entity_id,
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            # Prepare retrieval parameters
            memory_types = retrieval_kwargs.get("memory_types", None)
            tags = retrieval_kwargs.get("tags", None)
            limit = retrieval_kwargs.get("limit", 5)
            min_significance = retrieval_kwargs.get("min_significance", 0)
            
            # If we want realistic recall, use memory competition
            if retrieval_kwargs.get("simulate_competition", True) and query:
                competition_result = await self.interference_manager.simulate_memory_competition(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    query=query,
                    competition_count=limit
                )
                
                results["memory_competition"] = True
                results["winner"] = competition_result.get("winner")
                results["competing_memories"] = competition_result.get("competing_memories", [])
                
                # Possibly include a memory blend
                if "memory_blend" in competition_result:
                    results["memory_blend"] = competition_result["memory_blend"]
                
                # Record this memory event
                self._record_memory_event("memory_competition", entity_type, entity_id,
                                         competition_result.get("winner", {}).get("id"),
                                         competition_result)
                
                # If only wanting the strongest memory, return just the winner
                if retrieval_kwargs.get("competition_only", False):
                    return results
            
            # Standard memory retrieval
            memories = await memory_manager.retrieve_memories(
                query=query,
                memory_types=memory_types,
                tags=tags,
                min_significance=min_significance,
                limit=limit
            )
            
            # Get memory details
            detailed_memories = []
            for memory in memories:
                # Get schema interpretations if available
                interpretation = None
                if retrieval_kwargs.get("include_schema_interpretation", True):
                    try:
                        interpretation_result = await self.schema_manager.interpret_memory_with_schemas(
                            memory_id=memory.id,
                            entity_type=entity_type,
                            entity_id=entity_id
                        )
                        
                        if "interpretation" in interpretation_result:
                            interpretation = interpretation_result["interpretation"]
                    except Exception as e:
                        logger.error(f"Error getting schema interpretation: {e}")
                
                # Reconsolidate the memory upon recall with subtle changes
                if retrieval_kwargs.get("reconsolidate_on_recall", True):
                    try:
                        # Pass emotional context if available
                        recon_context = {}
                        if emotional_state and "current_emotion" in emotional_state:
                            recon_context["emotional_context"] = emotional_state["current_emotion"]
                        
                        # Subtle reconsolidation
                        await self.reconsolidation_manager.reconsolidate_memory(
                            memory_id=memory.id,
                            entity_type=entity_type,
                            entity_id=entity_id,
                            emotional_context=recon_context,
                            recall_context=str(current_context),
                            alteration_strength=0.05  # Very subtle
                        )
                    except Exception as e:
                        logger.error(f"Error in memory reconsolidation: {e}")
                
                # Add to detailed memories
                detailed_memories.append({
                    "id": memory.id,
                    "text": memory.text,
                    "type": memory.memory_type,
                    "significance": memory.significance,
                    "emotional_intensity": memory.emotional_intensity,
                    "timestamp": memory.timestamp.isoformat() if memory.timestamp else None,
                    "tags": memory.tags,
                    "schema_interpretation": interpretation
                })
            
            # Check for random flashback opportunity
            if retrieval_kwargs.get("allow_flashbacks", True) and random.random() < 0.15:  # 15% chance
                flashback = await self.flashback_manager.generate_flashback(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    current_context=current_context.get("text", query or "")
                )
                
                if flashback:
                    results["flashback"] = flashback
            
            # Check for intrusive memory opportunity
            if retrieval_kwargs.get("allow_intrusive", True) and random.random() < 0.1:  # 10% chance
                intrusion = await self.interference_manager.generate_intrusive_memory(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    context=current_context.get("text", query or "")
                )
                
                if intrusion and intrusion.get("intrusion_generated", False):
                    results["intrusive_memory"] = intrusion
            
            results["memories"] = detailed_memories
            
            # Record this memory event
            self._record_memory_event("recall", entity_type, entity_id,
                                     [m["id"] for m in memories],
                                     {"query": query, "count": len(memories)})
            
            return results
            
        except Exception as e:
            logger.error(f"Error in integrated memory retrieval: {e}")
            results["error"] = str(e)
            return results
    
    async def update_emotional_state_from_memory(self,
                                             entity_type: str,
                                             entity_id: int,
                                             memory_id: int,
                                             memory_text: str) -> Dict[str, Any]:
        """
        Update entity's emotional state based on a new memory.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            memory_id: ID of the memory
            memory_text: Text of the memory
            
        Returns:
            Updated emotional state
        """
        try:
            # Analyze the emotional content of the memory
            emotion_analysis = await self.emotional_manager.analyze_emotional_content(memory_text)
            
            # Prepare emotional state update
            current_emotion = {
                "primary_emotion": emotion_analysis.get("primary_emotion", "neutral"),
                "intensity": emotion_analysis.get("intensity", 0.3),
                "secondary_emotions": emotion_analysis.get("secondary_emotions", {}),
                "valence": emotion_analysis.get("valence", 0.0),
                "arousal": emotion_analysis.get("arousal", 0.0)
            }
            
            # Check if this is potentially a traumatic event
            is_traumatic = False
            if emotion_analysis.get("valence", 0) < -0.7 and emotion_analysis.get("intensity", 0) > 0.7:
                is_traumatic = True
                trauma_event = {
                    "memory_id": memory_id,
                    "memory_text": memory_text,
                    "emotion": current_emotion["primary_emotion"],
                    "intensity": current_emotion["intensity"],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                trauma_event = None
            
            # Update the emotional state
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
            
        except Exception as e:
            logger.error(f"Error updating emotional state: {e}")
            return {"error": str(e)}
    
    async def generate_schemas_from_memories(self,
                                        entity_type: str,
                                        entity_id: int,
                                        tag_filter: List[str] = None) -> Dict[str, Any]:
        """
        Generate schemas by analyzing memory patterns.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            tag_filter: Optional filter for memory tags
            
        Returns:
            Generated schemas information
        """
        try:
            # Detect schemas from memories
            result = await self.schema_manager.detect_schema_from_memories(
                entity_type=entity_type,
                entity_id=entity_id,
                tags=tag_filter
            )
            
            if result.get("schema_detected", False):
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
            logger.error(f"Error generating schemas: {e}")
            return {"error": str(e)}
    
    async def analyze_entity_memories(self,
                                 entity_type: str,
                                 entity_id: int) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of an entity's memories.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            
        Returns:
            Memory analysis results
        """
        try:
            # Get core memory stats
            memory_manager = UnifiedMemoryManager(
                entity_type=entity_type,
                entity_id=entity_id,
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            # Get all memories
            memories = await memory_manager.retrieve_memories(limit=100)
            
            # Extract memory stats
            memory_types = {}
            memory_tags = {}
            total_memories = len(memories)
            avg_significance = 0
            avg_emotional_intensity = 0
            
            if total_memories > 0:
                for memory in memories:
                    # Count by type
                    memory_type = memory.memory_type
                    if memory_type not in memory_types:
                        memory_types[memory_type] = 0
                    memory_types[memory_type] += 1
                    
                    # Count by tag
                    for tag in memory.tags:
                        if tag not in memory_tags:
                            memory_tags[tag] = 0
                        memory_tags[tag] += 1
                    
                    # Accumulate for averages
                    avg_significance += memory.significance
                    avg_emotional_intensity += memory.emotional_intensity
                
                # Calculate averages
                avg_significance /= total_memories
                avg_emotional_intensity /= total_memories
            
            # Get emotional state
            emotional_state = await self.emotional_manager.get_entity_emotional_state(
                entity_type=entity_type,
                entity_id=entity_id
            )
            
            # Get schemas
            schemas = await self._get_entity_schemas(entity_type, entity_id)
            
            # Get false memory status
            false_memory_status = await self.interference_manager.get_false_memory_status(
                entity_type=entity_type,
                entity_id=entity_id
            )
            
            # Compile the analysis
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
            
            # If this is an NPC, include mask information
            if entity_type == "npc":
                mask_info = await self.mask_manager.get_npc_mask(entity_id)
                if mask_info and "error" not in mask_info:
                    analysis["mask"] = {
                        "integrity": mask_info.get("integrity", 100),
                        "presented_traits": mask_info.get("presented_traits", {}),
                        "hidden_traits": mask_info.get("hidden_traits", {})
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing entity memories: {e}")
            return {"error": str(e)}
    
    async def _get_entity_schemas(self, entity_type: str, entity_id: int) -> List[Dict[str, Any]]:
        """
        Get schemas associated with an entity.
        """
        schemas = []
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
                    "confidence": schema_data.get("confidence", 0.5)
                })
                
        return schemas
    
    async def run_memory_maintenance(self,
                                  entity_type: str,
                                  entity_id: int,
                                  maintenance_options: Dict[str, bool] = None) -> Dict[str, Any]:
        """
        Run maintenance tasks on entity's memory system.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            maintenance_options: Options for what maintenance to perform
            
        Returns:
            Maintenance results
        """
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
            # Core memory maintenance
            if options.get("core_maintenance", True):
                memory_manager = UnifiedMemoryManager(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    user_id=self.user_id,
                    conversation_id=self.conversation_id
                )
                
                maintenance_result = await memory_manager.perform_maintenance()
                results["core_maintenance"] = maintenance_result
            
            # Schema maintenance
            if options.get("schema_maintenance", True):
                schema_result = await self.schema_manager.run_schema_maintenance(
                    entity_type=entity_type,
                    entity_id=entity_id
                )
                
                results["schema_maintenance"] = schema_result
            
            # Emotional decay
            if options.get("emotional_decay", True):
                emotional_result = await self.emotional_manager.emotional_decay_maintenance(
                    entity_type=entity_type,
                    entity_id=entity_id
                )
                
                results["emotional_decay"] = emotional_result
            
            # Background reconsolidation
            if options.get("background_reconsolidation", True):
                recon_result = await self.reconsolidation_manager.check_memories_for_reconsolidation(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    max_memories=3  # Limit to a few per maintenance cycle
                )
                
                results["reconsolidation"] = {
                    "memories_reconsolidated": len(recon_result) if isinstance(recon_result, list) else 0
                }
            
            # Interference processing
            if options.get("interference_processing", True):
                interference_result = await self.interference_manager.run_interference_maintenance(
                    entity_type=entity_type,
                    entity_id=entity_id
                )
                
                results["interference_processing"] = interference_result
            
            # Mask checks for NPCs
            if options.get("mask_checks", False) and entity_type == "npc":
                mask_result = await self.mask_manager.check_for_automated_reveals(self.user_id, self.conversation_id)
                
                results["mask_checks"] = {
                    "reveals_generated": len(mask_result)
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in memory maintenance: {e}")
            results["error"] = str(e)
            return results
    
    def _record_memory_event(self, event_type: str, entity_type: str, entity_id: int, 
                           memory_ids: Union[int, List[int]], details: Dict[str, Any] = None):
        """
        Record a memory event for tracking and analysis.
        """
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
        
        # Limit event history
        if len(self.events) > 100:
            self.events = self.events[-100:]
    
    async def initialize_tables(self):
        """
        Initialize all tables required by the memory system.
        """
        from .core import UnifiedMemoryManager
        from .schemas import create_schema_tables
        from .flashbacks import create_flashback_tables
        from .emotional import create_emotional_tables
        from .reconsolidation import create_reconsolidation_tables
        from .semantic import create_semantic_tables
        
        # Create core memory tables
        await UnifiedMemoryManager.create_tables()
        
        # Create schema tables
        await create_schema_tables()
        
        # Create flashback tables
        await create_flashback_tables()
        
        # Create emotional tables
        await create_emotional_tables()
        
        # Create reconsolidation tables
        await create_reconsolidation_tables()
        
        # Create semantic tables
        await create_semantic_tables()
        
        logger.info("All memory system tables initialized")

# Initialization function
async def init_memory_system(user_id: int, conversation_id: int) -> IntegratedMemorySystem:
    """
    Initialize the integrated memory system.
    """
    memory_system = IntegratedMemorySystem(user_id, conversation_id)
    await memory_system.initialize_tables()
    return memory_system
