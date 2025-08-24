# nyx/nyx_agent/context.py
"""NyxContext and state management for Nyx Agent SDK with full NPC and Memory integration"""

import json
import time
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union

from db.connection import get_db_connection_context
from nyx.user_model_sdk import UserModelManager
from nyx.nyx_task_integration import NyxTaskIntegration
from nyx.response_filter import ResponseFilter
from nyx.core.emotions.emotional_core import EmotionalCore
from nyx.performance_monitor import PerformanceMonitor
from nyx.core.sync.strategy_controller import get_active_strategies

# Import NPC orchestrator and related types
from npcs.npc_orchestrator import NPCOrchestrator, NPCSnapshot, NPCStatus

# Import Memory orchestrator and related types
from memory.memory_orchestrator import (
    MemoryOrchestrator, 
    EntityType,
    get_memory_orchestrator
)

# Import Conflict synthesizer and related types
from logic.conflict_system.conflict_synthesizer import (
    ConflictSynthesizer,
    get_synthesizer as get_conflict_synthesizer,
    ConflictContext,
    SubsystemType,
    EventType,
    SystemEvent
)

from .config import Config
from .utils import (
    safe_psutil, safe_process_metric, get_process_info, 
    bytes_to_mb, _prune_list, _calculate_variance
)

try:
    from story_agent.world_simulation_models import (
        CompleteWorldState, WorldState, WorldMood, TimeOfDay,
        ActivityType, PowerDynamicType, PowerExchange,
        WorldTension, RelationshipDynamics, NPCRoutine,
        CurrentTimeData, VitalsData, AddictionCravingData,
        DreamData, RevelationData, ChoiceData, ChoiceProcessingResult,
    )
    from story_agent.world_director_agent import (
        CompleteWorldDirector, WorldDirector,
        CompleteWorldDirectorContext, WorldDirectorContext,
    )
    WORLD_SIMULATION_AVAILABLE = True
except ImportError:
    logger.warning("World simulation models not available - slice-of-life features disabled")
    WORLD_SIMULATION_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class NyxContext:
    # ────────── REQUIRED (no defaults) ──────────
    user_id: int
    conversation_id: int

    # ────────── SUB-SYSTEM HANDLES ──────────
    memory_orchestrator: Optional[MemoryOrchestrator] = None  # Full memory system
    user_model:         Optional[UserModelManager]  = None
    task_integration:   Optional[NyxTaskIntegration] = None
    response_filter:    Optional[ResponseFilter]    = None
    emotional_core:     Optional[EmotionalCore]     = None
    performance_monitor: Optional[PerformanceMonitor] = None
    belief_system:      Optional[Any]               = None
    world_director:     Optional[Any]               = None
    slice_of_life_narrator: Optional[Any]           = None
    
    # ────────── ORCHESTRATOR INTEGRATION ──────────
    npc_orchestrator:   Optional[NPCOrchestrator]   = None
    conflict_synthesizer: Optional[ConflictSynthesizer] = None
    
    # Legacy memory compatibility (will use orchestrator internally)
    memory_system:      Optional[Any]               = None  # Legacy interface
    
    # ────────── MUTABLE STATE BUCKETS ──────────
    current_context:     Dict[str, Any]                = field(default_factory=dict)
    scenario_state:      Dict[str, Any]                = field(default_factory=dict)
    relationship_states: Dict[str, Dict[str, Any]]     = field(default_factory=dict)
    active_tasks:        List[Dict[str, Any]]          = field(default_factory=list)
    current_world_state: Optional[Any]                = None
    daily_routine_tracker: Optional[Dict[str, Any]]   = None
    emergent_narratives: List[Dict[str, Any]]        = field(default_factory=list)
    npc_autonomy_states: Dict[int, Dict[str, Any]]   = field(default_factory=dict)
    
    # ────────── NPC-SPECIFIC STATE ──────────
    active_npc_ids:      Set[int]                    = field(default_factory=set)
    npc_snapshots:       Dict[int, NPCSnapshot]      = field(default_factory=dict)
    npc_interaction_history: List[Dict[str, Any]]    = field(default_factory=list)
    current_scene_npcs:  List[int]                   = field(default_factory=list)
    npc_narrative_context: Dict[str, Any]            = field(default_factory=dict)
    
    # ────────── MEMORY-SPECIFIC STATE ──────────
    memory_context:      Dict[str, Any]              = field(default_factory=dict)
    recent_memories:     Dict[str, List[Dict]]       = field(default_factory=dict)  # entity_key -> memories
    active_schemas:      List[Dict[str, Any]]        = field(default_factory=list)
    memory_narratives:   List[Dict[str, Any]]        = field(default_factory=list)
    memory_predictions:  List[Dict[str, Any]]        = field(default_factory=list)
    belief_systems:      Dict[str, List[Dict]]       = field(default_factory=dict)  # entity_key -> beliefs
    emotional_memories:  Dict[str, Dict]             = field(default_factory=dict)  # entity_key -> emotional state
    
    # ────────── CONFLICT-SPECIFIC STATE ──────────
    active_conflicts:    Dict[int, Dict[str, Any]]   = field(default_factory=dict)  # conflict_id -> state
    conflict_tensions:   Dict[str, float]            = field(default_factory=dict)  # entity_pair -> tension
    conflict_events:     List[Dict[str, Any]]        = field(default_factory=list)  # Recent conflict events
    conflict_choices:    List[Dict[str, Any]]        = field(default_factory=list)  # Available choices
    conflict_context:    Optional[ConflictContext]   = None
    scene_conflicts:     List[int]                   = field(default_factory=list)  # Conflicts in current scene
    conflict_subsystems: Set[str]                    = field(default_factory=set)   # Active subsystems

    # ────────── PERFORMANCE & EMOTION ──────────
    performance_metrics: Dict[str, Any] = field(default_factory=lambda: {
        "total_actions": 0, "successful_actions": 0, "failed_actions": 0,
        "response_times": [], "memory_usage": 0, "cpu_usage": 0,
        "error_rates": {"total": 0, "recovered": 0, "unrecovered": 0},
        "npc_interactions": 0, "npc_decisions": 0, "npc_memories_created": 0,
        "conflicts_created": 0, "conflicts_resolved": 0, "conflicts_updated": 0,
        "conflict_scenes_processed": 0, "tension_calculations": 0
    })
    emotional_state: Dict[str, float] = field(default_factory=lambda: {
        "valence": 0.0, "arousal": 0.5, "dominance": 0.7
    })

    # ────────── LEARNING & ADAPTATION ──────────
    learned_patterns:      Dict[str, Any]           = field(default_factory=dict)
    strategy_effectiveness: Dict[str, Any]          = field(default_factory=dict)
    adaptation_history:    List[Dict[str, Any]]     = field(default_factory=list)
    learning_metrics:      Dict[str, Any]           = field(default_factory=lambda: {
        "pattern_recognition_rate": 0.0,
        "strategy_improvement_rate": 0.0,
        "adaptation_success_rate": 0.0,
        "npc_behavior_prediction_accuracy": 0.0
    })

    # ────────── ERROR LOGGING ──────────
    error_log: List[Dict[str, Any]] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=dict)
    
    # ────────── FEATURE FLAGS ──────────
    _tables_available: Dict[str, bool] = field(default_factory=dict)
    enable_npc_canon: bool = True  # Enable canon integration for NPCs

    # ────────── TASK SCHEDULING ──────────
    last_task_runs: Dict[str, datetime] = field(default_factory=dict)
    task_intervals: Dict[str, float]    = field(default_factory=lambda: {
        "memory_reflection": 300, "relationship_update": 600,
        "scenario_check": 60, "performance_check": 300,
        "task_generation": 300, "learning_save": 900, 
        "performance_save": 600,
        "scenario_heartbeat": 3600,
        "npc_perception_update": 30,  # Update NPC perceptions every 30s
        "npc_decision_cycle": 60,      # NPC decision making every minute
        "npc_learning_cycle": 900,     # NPC learning every 15 minutes
        "npc_scheming_check": 600,     # Check for scheming every 10 minutes
        "memory_maintenance": 1800,    # Memory maintenance every 30 minutes
        "memory_consolidation": 3600,  # Memory consolidation every hour
        "belief_update": 600,          # Belief system updates every 10 minutes
        "canon_sync": 7200,            # Canon sync every 2 hours
        "conflict_health_check": 300,  # Conflict system health check every 5 minutes
        "tension_calculation": 120,    # Recalculate tensions every 2 minutes
        "conflict_scene_check": 60,    # Check for scene conflicts every minute
        "conflict_resolution_check": 600  # Check for resolution opportunities every 10 minutes
    })

    # ────────── PRIVATE CACHES (init=False) ──────────
    _strategy_cache:             Optional[tuple] = field(init=False, default=None)
    _strategy_cache_ttl:         float = field(init=False, default=300.0)
    _strategy_cache_lock:        asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    _cpu_usage_cache:            Optional[float] = field(init=False, default=None)
    _cpu_usage_last_update:      float = field(init=False, default=0.0)
    _cpu_usage_update_interval:  float = field(init=False, default=10.0)
    
    async def initialize(self):
        """Initialize all systems including NPC and Memory orchestrators"""
        # Initialize Memory Orchestrator first (other systems may depend on it)
        try:
            self.memory_orchestrator = await get_memory_orchestrator(
                self.user_id, 
                self.conversation_id
            )
            # Legacy compatibility wrapper
            self.memory_system = self.memory_orchestrator
            
            logger.info(f"Memory Orchestrator initialized for user {self.user_id}, conversation {self.conversation_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Memory Orchestrator: {e}", exc_info=True)
            self.memory_orchestrator = None
            self.memory_system = None
        
        # Initialize other core systems
        self.user_model = await UserModelManager.get_instance(self.user_id, self.conversation_id)
        self.task_integration = await NyxTaskIntegration.get_instance(self.user_id, self.conversation_id)
        self.response_filter = ResponseFilter(self.user_id, self.conversation_id)
        self.performance_monitor = PerformanceMonitor.get_instance(self.user_id, self.conversation_id)
        
        # Initialize emotional core if available
        try:
            self.emotional_core = EmotionalCore()
        except Exception as e:
            logger.warning(f"EmotionalCore not available: {e}", exc_info=True)
        
        # Initialize belief system if available
        try:
            from nyx.nyx_belief_system import BeliefSystem
            self.belief_system = BeliefSystem(self.user_id, self.conversation_id)
        except ImportError as e:
            logger.warning(f"BeliefSystem module not available: {e}")
        except Exception as e:
            logger.warning(f"BeliefSystem initialization failed: {e}", exc_info=True)

        # Initialize NPC orchestrator
        try:
            self.npc_orchestrator = NPCOrchestrator(
                self.user_id, 
                self.conversation_id, 
                enable_canon=self.enable_npc_canon
            )
            await self.npc_orchestrator.initialize()
            
            # Load active NPCs
            self.active_npc_ids = self.npc_orchestrator._active_npcs.copy()
            
            logger.info(f"NPC Orchestrator initialized with {len(self.active_npc_ids)} active NPCs")
        except Exception as e:
            logger.error(f"Failed to initialize NPC orchestrator: {e}", exc_info=True)
            self.npc_orchestrator = None

        # Initialize Conflict Synthesizer
        try:
            self.conflict_synthesizer = await get_conflict_synthesizer(
                self.user_id,
                self.conversation_id
            )
            
            # Get initial system state
            conflict_state = await self.conflict_synthesizer.get_system_state()
            
            # Load active conflicts
            for conflict_id in conflict_state.get('active_conflicts', []):
                if isinstance(conflict_id, int):
                    state = await self.conflict_synthesizer.get_conflict_state(conflict_id)
                    self.active_conflicts[conflict_id] = state
            
            # Track active subsystems
            self.conflict_subsystems = set(conflict_state.get('subsystems', {}).keys())
            
            logger.info(f"Conflict Synthesizer initialized with {len(self.active_conflicts)} active conflicts")
        except Exception as e:
            logger.error(f"Failed to initialize Conflict Synthesizer: {e}", exc_info=True)
            self.conflict_synthesizer = None

        # Initialize world systems
        try:
            from story_agent.world_director_agent import CompleteWorldDirector
            from story_agent.slice_of_life_narrator import SliceOfLifeNarrator

            self.world_director = CompleteWorldDirector(self.user_id, self.conversation_id)
            await self.world_director.initialize()

            self.slice_of_life_narrator = SliceOfLifeNarrator(self.user_id, self.conversation_id)
            await self.slice_of_life_narrator.initialize()

            self.current_world_state = self.world_director.context.current_world_state
        except Exception as e:
            logger.warning(f"World systems initialization failed: {e}", exc_info=True)

        # Initialize CPU usage monitoring
        try:
            self._cpu_usage_cache = safe_psutil('cpu_percent', interval=0.1, default=0.0)
        except Exception as e:
            logger.debug(f"Failed to initialize CPU monitoring: {e}")
            self._cpu_usage_cache = 0.0
        
        # Load existing state from database
        await self._load_state()
        
        # Load NPC context for current scene
        await self._load_npc_context()
    
    async def _load_npc_context(self):
        """Load NPC context for the current scene with enhanced memory integration"""
        if not self.npc_orchestrator:
            return
        
        try:
            # Get current location from context or world state
            current_location = None
            if self.current_context.get("location"):
                current_location = self.current_context["location"]
            elif self.current_world_state and hasattr(self.current_world_state, 'player_location'):
                current_location = self.current_world_state.player_location
            
            # Get NPCs at current location if known
            if current_location:
                npcs_at_location = await self.npc_orchestrator.get_npcs_at_location(current_location)
                self.current_scene_npcs = [npc.npc_id for npc in npcs_at_location]
                
                # Load snapshots for scene NPCs
                for npc in npcs_at_location:
                    self.npc_snapshots[npc.npc_id] = npc
            
            # Get comprehensive narrative context
            self.npc_narrative_context = await self.npc_orchestrator.get_narrative_context(
                focus_npc_ids=self.current_scene_npcs if self.current_scene_npcs else None,
                location=current_location,
                include_relationships=True,
                include_beliefs=True,
                include_decision_patterns=True,
                include_learning_data=True,
                include_nyx_governance=True
            )
            
            # Enhance with Memory Orchestrator data if available
            if self.memory_orchestrator:
                for npc_id in self.current_scene_npcs:
                    entity_key = f"npc_{npc_id}"
                    
                    # Get rich memories from orchestrator
                    npc_memories = await self.memory_orchestrator.retrieve_memories(
                        entity_type=EntityType.NPC,
                        entity_id=npc_id,
                        limit=5,
                        use_llm_analysis=True
                    )
                    
                    if npc_memories.get("memories"):
                        self.recent_memories[entity_key] = npc_memories["memories"]
                        
                        # Add analysis to narrative context
                        if entity_key in self.npc_narrative_context.get("npcs", {}):
                            self.npc_narrative_context["npcs"][entity_key]["memory_analysis"] = npc_memories.get("analysis", {})
                    
                    # Get beliefs from orchestrator
                    npc_beliefs = await self.memory_orchestrator.get_beliefs(
                        entity_type="npc",
                        entity_id=npc_id,
                        min_confidence=0.3
                    )
                    
                    if npc_beliefs:
                        self.belief_systems[entity_key] = npc_beliefs
                    
                    # Get emotional state from memory system
                    emotional_state = await self.memory_orchestrator.get_emotional_state(
                        entity_type="npc",
                        entity_id=npc_id
                    )
                    
                    if emotional_state:
                        self.emotional_memories[entity_key] = emotional_state
                
                # Get memory-based narrative predictions
                self.memory_predictions = await self.memory_orchestrator.analyze_memory_patterns(
                    topic="current_scene"
                )
            
            logger.info(f"Loaded NPC context: {len(self.current_scene_npcs)} NPCs in scene with enhanced memory")
            
        except Exception as e:
            logger.error(f"Failed to load NPC context: {e}", exc_info=True)
    
    async def update_conflict_context(self, location: Optional[str] = None, scene_type: Optional[str] = None):
        """Update conflict context for a new scene or interaction"""
        if not self.conflict_synthesizer:
            return
        
        try:
            # Build new conflict context
            self.conflict_context = ConflictContext(
                scene_type=scene_type or self.current_context.get('scene_type', 'interaction'),
                location=location or self.current_context.get('location'),
                location_id=self.current_context.get('location_id'),
                participants=self.current_scene_npcs,
                present_npcs=self.current_scene_npcs,
                npcs=self.current_scene_npcs,
                recent_events=[e['type'] for e in self.conflict_events[-5:]],
                timestamp=datetime.now().isoformat()
            )
            
            # Process scene for conflicts
            scene_result = await self.process_scene_conflicts({
                'scene_type': scene_type,
                'location': location
            })
            
            # Update scene conflicts based on results
            if scene_result.get('conflicts_detected'):
                self.scene_conflicts = scene_result['conflicts_detected']
            
            # Update choices
            if scene_result.get('choices'):
                self.conflict_choices = scene_result['choices']
            
            # Update metrics
            self.performance_metrics["conflict_scenes_processed"] += 1
            
            logger.info(f"Updated conflict context: {len(self.scene_conflicts)} conflicts in scene")
            
        except Exception as e:
            logger.error(f"Failed to update conflict context: {e}", exc_info=True)
        """Update NPC context for a new scene or interaction"""
        if not self.npc_orchestrator:
            return
        
        try:
            # Update location-based NPCs
            if location:
                npcs_at_location = await self.npc_orchestrator.get_npcs_at_location(location)
                self.current_scene_npcs = [npc.npc_id for npc in npcs_at_location]
                
                # Update snapshots
                for npc in npcs_at_location:
                    self.npc_snapshots[npc.npc_id] = npc
            
            # Add specific NPCs if provided
            if npc_ids:
                for npc_id in npc_ids:
                    if npc_id not in self.current_scene_npcs:
                        self.current_scene_npcs.append(npc_id)
                    
                    # Get snapshot if not already loaded
                    if npc_id not in self.npc_snapshots:
                        snapshot = await self.npc_orchestrator.get_npc_snapshot(npc_id)
                        self.npc_snapshots[npc_id] = snapshot
            
            # Update narrative context
            self.npc_narrative_context = await self.npc_orchestrator.get_narrative_context(
                focus_npc_ids=self.current_scene_npcs,
                location=location,
                include_relationships=True,
                include_beliefs=True
            )
            
        except Exception as e:
            logger.error(f"Failed to update NPC context: {e}", exc_info=True)
    
    async def process_npc_interaction(
        self, 
        npc_id: int, 
        interaction_type: str, 
        player_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process an interaction with an NPC"""
        if not self.npc_orchestrator:
            return {"error": "NPC system not initialized"}
        
        try:
            # Add current emotional state to context
            if context is None:
                context = {}
            context["nyx_emotional_state"] = self.emotional_state.copy()
            context["current_scene_npcs"] = self.current_scene_npcs
            
            # Process interaction through orchestrator
            result = await self.npc_orchestrator.process_player_interaction(
                npc_id=npc_id,
                player_input=player_input,
                interaction_type=interaction_type,
                context=context
            )
            
            # Update interaction history
            self.npc_interaction_history.append({
                "timestamp": time.time(),
                "npc_id": npc_id,
                "interaction_type": interaction_type,
                "player_input": player_input[:100],  # Truncate for storage
                "result": result.get("outcome", "unknown")
            })
            
            # Keep history bounded
            if len(self.npc_interaction_history) > 100:
                self.npc_interaction_history = self.npc_interaction_history[-50:]
            
            # Update metrics
            self.performance_metrics["npc_interactions"] += 1
            
            # Refresh NPC snapshot
            self.npc_snapshots[npc_id] = await self.npc_orchestrator.get_npc_snapshot(npc_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process NPC interaction: {e}", exc_info=True)
            self.log_error(e, {"npc_id": npc_id, "interaction_type": interaction_type})
            return {"error": str(e)}
    
    async def trigger_npc_decisions(self, perception_context: Optional[Dict[str, Any]] = None):
        """Trigger decision-making for NPCs in the current scene"""
        if not self.npc_orchestrator or not self.current_scene_npcs:
            return []
        
        if not self.should_run_task("npc_decision_cycle"):
            return []
        
        decisions = []
        try:
            for npc_id in self.current_scene_npcs:
                decision = await self.npc_orchestrator.make_npc_decision(
                    npc_id=npc_id,
                    perception=perception_context
                )
                decisions.append({
                    "npc_id": npc_id,
                    "decision": decision
                })
                
                # Update metrics
                self.performance_metrics["npc_decisions"] += 1
            
            self.record_task_run("npc_decision_cycle")
            
        except Exception as e:
            logger.error(f"Failed to trigger NPC decisions: {e}", exc_info=True)
            self.log_error(e, {"perception_context": perception_context})
        
        return decisions
    
    async def update_npc_perceptions(self):
        """Update perception for all NPCs in scene"""
        if not self.npc_orchestrator or not self.current_scene_npcs:
            return
        
        if not self.should_run_task("npc_perception_update"):
            return
        
        try:
            for npc_id in self.current_scene_npcs:
                perception = await self.npc_orchestrator.get_npc_perception(npc_id)
                
                # Store in autonomy states
                if npc_id not in self.npc_autonomy_states:
                    self.npc_autonomy_states[npc_id] = {}
                self.npc_autonomy_states[npc_id]["perception"] = perception
                self.npc_autonomy_states[npc_id]["last_update"] = time.time()
            
            self.record_task_run("npc_perception_update")
            
        except Exception as e:
            logger.error(f"Failed to update NPC perceptions: {e}", exc_info=True)
            self.log_error(e, {})
    
    async def create_npc_memory(
        self, 
        npc_id: int, 
        memory_text: str, 
        memory_type: str = "observation",
        significance: int = 3,
        emotional_valence: int = 0
    ):
        """Create a memory for an NPC using the Memory Orchestrator"""
        if not self.memory_orchestrator:
            # Fallback to NPC orchestrator if memory orchestrator not available
            if self.npc_orchestrator:
                return await self.npc_orchestrator.add_memory_for_npc(
                    npc_id=npc_id,
                    memory_text=memory_text,
                    memory_type=memory_type,
                    significance=significance,
                    emotional_valence=emotional_valence,
                    tags=["nyx_interaction"],
                    use_nyx_governance=True
                )
            return None
        
        try:
            # Use Memory Orchestrator for rich memory creation
            result = await self.memory_orchestrator.store_memory(
                entity_type=EntityType.NPC,
                entity_id=npc_id,
                memory_text=memory_text,
                importance=self._significance_to_importance(significance),
                emotional=abs(emotional_valence) > 0,
                tags=["nyx_interaction", memory_type],
                metadata={
                    "memory_type": memory_type,
                    "emotional_valence": emotional_valence,
                    "created_by": "nyx"
                },
                use_governance=True,
                check_canon_consistency=self.enable_npc_canon
            )
            
            # Update metrics
            self.performance_metrics["npc_memories_created"] += 1
            
            # Cache the memory
            entity_key = f"npc_{npc_id}"
            if entity_key not in self.recent_memories:
                self.recent_memories[entity_key] = []
            self.recent_memories[entity_key].append({
                "memory_id": result.get("memory_id"),
                "text": memory_text,
                "timestamp": datetime.now().isoformat()
            })
            # Keep cache bounded
            if len(self.recent_memories[entity_key]) > 10:
                self.recent_memories[entity_key] = self.recent_memories[entity_key][-10:]
            
            return result.get("memory_id")
            
        except Exception as e:
            logger.error(f"Failed to create NPC memory via orchestrator: {e}", exc_info=True)
            self.log_error(e, {"npc_id": npc_id, "memory_text": memory_text})
            return None
    
    def _significance_to_importance(self, significance: int) -> str:
        """Convert numeric significance to importance string"""
        if significance <= 2:
            return "trivial"
        elif significance <= 4:
            return "low"
        elif significance <= 6:
            return "medium"
        elif significance <= 8:
            return "high"
        else:
            return "critical"
    
    # ────────── MEMORY ORCHESTRATOR OPERATIONS ──────────
    
    async def store_memory(
        self,
        entity_type: Union[str, EntityType],
        entity_id: int,
        memory_text: str,
        importance: str = "medium",
        tags: List[str] = None,
        emotional: bool = True,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Store a memory using the Memory Orchestrator"""
        if not self.memory_orchestrator:
            return {"error": "Memory orchestrator not initialized"}
        
        try:
            result = await self.memory_orchestrator.store_memory(
                entity_type=entity_type,
                entity_id=entity_id,
                memory_text=memory_text,
                importance=importance,
                emotional=emotional,
                tags=tags or [],
                metadata=metadata,
                use_governance=True,
                check_canon_consistency=self.enable_npc_canon
            )
            
            # Update cache
            entity_key = f"{entity_type}_{entity_id}"
            if entity_key not in self.recent_memories:
                self.recent_memories[entity_key] = []
            self.recent_memories[entity_key].append({
                "memory_id": result.get("memory_id"),
                "text": memory_text,
                "importance": importance,
                "timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}", exc_info=True)
            self.log_error(e, {"entity_type": entity_type, "entity_id": entity_id})
            return {"error": str(e)}
    
    async def retrieve_memories(
        self,
        entity_type: Union[str, EntityType],
        entity_id: int,
        query: str = None,
        limit: int = 5,
        use_llm_analysis: bool = True
    ) -> Dict[str, Any]:
        """Retrieve memories using the Memory Orchestrator"""
        if not self.memory_orchestrator:
            return {"error": "Memory orchestrator not initialized", "memories": []}
        
        try:
            result = await self.memory_orchestrator.retrieve_memories(
                entity_type=entity_type,
                entity_id=entity_id,
                query=query,
                context=self.current_context,
                limit=limit,
                include_analysis=True,
                use_governance=True,
                use_llm_analysis=use_llm_analysis
            )
            
            # Update cache
            entity_key = f"{entity_type}_{entity_id}"
            if result.get("memories"):
                self.recent_memories[entity_key] = [
                    {
                        "memory_id": m.get("id") or m.get("memory_id"),
                        "text": m.get("text") or m.get("memory_text"),
                        "importance": m.get("importance") or "medium",
                        "timestamp": m.get("timestamp") or datetime.now().isoformat()
                    }
                    for m in result["memories"][:10]
                ]
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}", exc_info=True)
            self.log_error(e, {"entity_type": entity_type, "entity_id": entity_id})
            return {"error": str(e), "memories": []}
    
    async def get_narrative_memory_context(
        self,
        include_npcs: bool = True,
        include_player: bool = True,
        include_canon: bool = True,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Get comprehensive narrative context from both memory and NPC systems"""
        if not self.memory_orchestrator:
            return {"error": "Memory orchestrator not initialized"}
        
        try:
            # Get memory-based narrative context
            if include_canon:
                memory_context = await self.memory_orchestrator.get_canon_aware_narrative_context(
                    include_canon=True,
                    time_window_hours=time_window_hours
                )
            else:
                memory_context = await self.memory_orchestrator.get_narrative_context(
                    time_window=timedelta(hours=time_window_hours),
                    include_predictions=True
                )
            
            # Enhance with NPC context if available
            if include_npcs and self.npc_orchestrator:
                npc_context = self.get_npc_context_for_response()
                memory_context["npcs"] = npc_context
            
            # Add player memories
            if include_player:
                player_memories = await self.retrieve_memories(
                    entity_type=EntityType.PLAYER,
                    entity_id=self.user_id,
                    limit=5
                )
                memory_context["player"] = {
                    "recent_memories": player_memories.get("memories", []),
                    "analysis": player_memories.get("analysis", {})
                }
            
            # Cache the context
            self.memory_context = memory_context
            
            return memory_context
            
        except Exception as e:
            logger.error(f"Failed to get narrative memory context: {e}", exc_info=True)
            self.log_error(e, {})
            return {"error": str(e)}
    
    async def create_belief(
        self,
        entity_type: str,
        entity_id: int,
        belief_text: str,
        confidence: float = 0.5
    ) -> Dict[str, Any]:
        """Create a belief for an entity"""
        if not self.memory_orchestrator:
            return {"error": "Memory orchestrator not initialized"}
        
        try:
            result = await self.memory_orchestrator.create_belief(
                entity_type=entity_type,
                entity_id=entity_id,
                belief_text=belief_text,
                confidence=confidence,
                use_governance=True
            )
            
            # Update cache
            entity_key = f"{entity_type}_{entity_id}"
            if entity_key not in self.belief_systems:
                self.belief_systems[entity_key] = []
            self.belief_systems[entity_key].append({
                "belief": belief_text,
                "confidence": confidence,
                "created_at": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create belief: {e}", exc_info=True)
            self.log_error(e, {"entity_type": entity_type, "entity_id": entity_id})
            return {"error": str(e)}
    
    async def get_beliefs(
        self,
        entity_type: str,
        entity_id: int,
        topic: str = None,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Get beliefs for an entity"""
        if not self.memory_orchestrator:
            return []
        
        try:
            beliefs = await self.memory_orchestrator.get_beliefs(
                entity_type=entity_type,
                entity_id=entity_id,
                topic=topic,
                min_confidence=min_confidence,
                use_governance=True
            )
            
            # Update cache
            entity_key = f"{entity_type}_{entity_id}"
            if beliefs:
                self.belief_systems[entity_key] = beliefs[:10]
            
            return beliefs
            
        except Exception as e:
            logger.error(f"Failed to get beliefs: {e}", exc_info=True)
            self.log_error(e, {"entity_type": entity_type, "entity_id": entity_id})
            return []
    
    async def trigger_flashback(
        self,
        entity_type: str,
        entity_id: int,
        context: str
    ) -> Optional[Dict[str, Any]]:
        """Generate a flashback for an entity"""
        if not self.memory_orchestrator:
            return None
        
        try:
            return await self.memory_orchestrator.generate_flashback(
                entity_type=entity_type,
                entity_id=entity_id,
                context=context
            )
        except Exception as e:
            logger.error(f"Failed to generate flashback: {e}", exc_info=True)
            return None
    
    async def analyze_memory_patterns(
        self,
        entity_type: str = None,
        entity_id: int = None,
        topic: str = None
    ) -> Dict[str, Any]:
        """Analyze patterns in memories"""
        if not self.memory_orchestrator:
            return {"error": "Memory orchestrator not initialized"}
        
        try:
            patterns = await self.memory_orchestrator.analyze_memory_patterns(
                entity_type=entity_type,
                entity_id=entity_id,
                topic=topic
            )
            
            # Cache predictions if available
            if patterns.get("predictions"):
                self.memory_predictions = patterns["predictions"]
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to analyze memory patterns: {e}", exc_info=True)
            self.log_error(e, {"entity_type": entity_type, "entity_id": entity_id})
            return {"error": str(e)}
    
    async def update_emotional_memory(
        self,
        entity_type: str,
        entity_id: int,
        emotion: str,
        intensity: float = 0.5
    ) -> Dict[str, Any]:
        """Update emotional state in memory system"""
        if not self.memory_orchestrator:
            return {"error": "Memory orchestrator not initialized"}
        
        try:
            result = await self.memory_orchestrator.update_emotional_state(
                entity_type=entity_type,
                entity_id=entity_id,
                emotion=emotion,
                intensity=intensity
            )
            
            # Update cache
            entity_key = f"{entity_type}_{entity_id}"
            self.emotional_memories[entity_key] = {
                "emotion": emotion,
                "intensity": intensity,
                "updated_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to update emotional memory: {e}", exc_info=True)
            self.log_error(e, {"entity_type": entity_type, "entity_id": entity_id})
            return {"error": str(e)}
    
    async def run_memory_maintenance(self, operations: List[str] = None) -> Dict[str, Any]:
        """Run memory maintenance operations"""
        if not self.memory_orchestrator:
            return {"error": "Memory orchestrator not initialized"}
        
        if not self.should_run_task("memory_maintenance"):
            return {"skipped": "Too soon since last maintenance"}
        
        try:
            result = await self.memory_orchestrator.run_maintenance(
                operations=operations,
                use_governance=True
            )
            
            self.record_task_run("memory_maintenance")
            return result
            
        except Exception as e:
            logger.error(f"Failed to run memory maintenance: {e}", exc_info=True)
            self.log_error(e, {})
            return {"error": str(e)}
    
    async def enrich_context_with_memories(self) -> Dict[str, Any]:
        """Enrich current context with relevant memories"""
        if not self.memory_orchestrator:
            return self.current_context
        
        try:
            user_input = self.current_context.get("user_input", "")
            enriched = await self.memory_orchestrator.enrich_context(
                user_input=user_input,
                context=self.current_context
            )
            
            # Merge enriched context
            self.current_context.update(enriched)
            
            return self.current_context
            
        except Exception as e:
            logger.error(f"Failed to enrich context with memories: {e}", exc_info=True)
            self.log_error(e, {})
            return self.current_context
    
    # ────────── CONFLICT SYNTHESIZER OPERATIONS ──────────
    
    async def create_conflict(
        self,
        conflict_type: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a new conflict using the Conflict Synthesizer"""
        if not self.conflict_synthesizer:
            return {"error": "Conflict synthesizer not initialized"}
        
        try:
            # Enhance context with current scene data
            enhanced_context = context or {}
            enhanced_context.update({
                'location': self.current_context.get('location'),
                'present_npcs': self.current_scene_npcs,
                'participants': self.current_scene_npcs[:],
                'recent_events': self.conflict_events[-5:] if self.conflict_events else [],
                'active_conflicts': list(self.active_conflicts.keys())
            })
            
            # Create conflict through synthesizer
            result = await self.conflict_synthesizer.create_conflict(
                conflict_type=conflict_type,
                context=enhanced_context
            )
            
            # Update local state
            conflict_id = result.get('conflict_id')
            if conflict_id:
                self.active_conflicts[conflict_id] = result
                self.scene_conflicts.append(conflict_id)
                
                # Log as event
                self.conflict_events.append({
                    'type': 'conflict_created',
                    'conflict_id': conflict_id,
                    'conflict_type': conflict_type,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Keep events bounded
                if len(self.conflict_events) > 50:
                    self.conflict_events = self.conflict_events[-50:]
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create conflict: {e}", exc_info=True)
            self.log_error(e, {"conflict_type": conflict_type, "context": context})
            return {"error": str(e)}
    
    async def update_conflict(
        self,
        conflict_id: int,
        update_type: str,
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update an existing conflict"""
        if not self.conflict_synthesizer:
            return {"error": "Conflict synthesizer not initialized"}
        
        try:
            result = await self.conflict_synthesizer.update_conflict(
                conflict_id=conflict_id,
                update_type=update_type,
                update_data=update_data
            )
            
            # Update local state
            if conflict_id in self.active_conflicts:
                self.active_conflicts[conflict_id].update(result)
            
            # Log as event
            self.conflict_events.append({
                'type': 'conflict_updated',
                'conflict_id': conflict_id,
                'update_type': update_type,
                'timestamp': datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to update conflict: {e}", exc_info=True)
            self.log_error(e, {"conflict_id": conflict_id, "update_type": update_type})
            return {"error": str(e)}
    
    async def resolve_conflict(
        self,
        conflict_id: int,
        resolution_type: str,
        resolution_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Resolve a conflict"""
        if not self.conflict_synthesizer:
            return {"error": "Conflict synthesizer not initialized"}
        
        try:
            result = await self.conflict_synthesizer.resolve_conflict(
                conflict_id=conflict_id,
                resolution_type=resolution_type,
                resolution_context=resolution_context or {}
            )
            
            # Update local state
            if conflict_id in self.active_conflicts:
                del self.active_conflicts[conflict_id]
            if conflict_id in self.scene_conflicts:
                self.scene_conflicts.remove(conflict_id)
            
            # Log as event
            self.conflict_events.append({
                'type': 'conflict_resolved',
                'conflict_id': conflict_id,
                'resolution_type': resolution_type,
                'outcome': result.get('outcome'),
                'timestamp': datetime.now().isoformat()
            })
            
            # Store any new conflicts created
            new_conflicts = result.get('new_conflicts_created', [])
            for new_id in new_conflicts:
                state = await self.conflict_synthesizer.get_conflict_state(new_id)
                self.active_conflicts[new_id] = state
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to resolve conflict: {e}", exc_info=True)
            self.log_error(e, {"conflict_id": conflict_id, "resolution_type": resolution_type})
            return {"error": str(e)}
    
    async def process_scene_conflicts(
        self,
        scene_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process conflicts for the current scene"""
        if not self.conflict_synthesizer:
            return {"error": "Conflict synthesizer not initialized"}
        
        try:
            # Build scene context
            scene_data = scene_context or {}
            scene_data.update({
                'scene_type': self.current_context.get('scene_type', 'interaction'),
                'characters_present': self.current_scene_npcs,
                'location_id': self.current_context.get('location_id'),
                'active_conflicts': self.scene_conflicts,
                'npcs': self.current_scene_npcs,
                'timestamp': datetime.now().isoformat()
            })
            
            # Process through synthesizer
            result = await self.conflict_synthesizer.process_scene(scene_data)
            
            # Update choices if available
            if result.get('choices'):
                self.conflict_choices = result['choices']
            
            # Update scene conflicts
            if result.get('conflicts_detected'):
                for conflict_id in result['conflicts_detected']:
                    if conflict_id not in self.scene_conflicts:
                        self.scene_conflicts.append(conflict_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process scene conflicts: {e}", exc_info=True)
            self.log_error(e, {"scene_context": scene_context})
            return {"error": str(e)}
    
    async def get_conflict_system_health(self) -> Dict[str, Any]:
        """Get health status of the conflict system"""
        if not self.conflict_synthesizer:
            return {"healthy": False, "error": "Conflict synthesizer not initialized"}
        
        if not self.should_run_task("conflict_health_check"):
            # Return cached state
            return {
                "healthy": True,
                "cached": True,
                "active_conflicts": len(self.active_conflicts),
                "subsystems": len(self.conflict_subsystems)
            }
        
        try:
            state = await self.conflict_synthesizer.get_system_state()
            
            self.record_task_run("conflict_health_check")
            
            return {
                "healthy": state.get('metrics', {}).get('system_health', 0) > 0.5,
                "metrics": state.get('metrics', {}),
                "active_conflicts": len(state.get('active_conflicts', [])),
                "subsystem_health": state.get('health', {}),
                "complexity_score": state.get('metrics', {}).get('complexity_score', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get conflict system health: {e}", exc_info=True)
            return {"healthy": False, "error": str(e)}
    
    async def calculate_conflict_tensions(self) -> Dict[str, float]:
        """Calculate current conflict tensions between entities"""
        if not self.conflict_synthesizer:
            return {}
        
        if not self.should_run_task("tension_calculation"):
            return self.conflict_tensions
        
        try:
            # Get tension data from conflict system
            tensions = {}
            
            for conflict_id, conflict_state in self.active_conflicts.items():
                # Extract stakeholders and their tensions
                stakeholders = conflict_state.get('subsystem_data', {}).get('stakeholder', {}).get('stakeholders', [])
                
                for i, stakeholder1 in enumerate(stakeholders):
                    for stakeholder2 in stakeholders[i+1:]:
                        key = f"{stakeholder1}_{stakeholder2}"
                        # Aggregate tension from multiple conflicts
                        current_tension = tensions.get(key, 0.0)
                        conflict_tension = conflict_state.get('subsystem_data', {}).get('tension', {}).get('level', 0.5)
                        tensions[key] = min(1.0, current_tension + conflict_tension * 0.5)
            
            self.conflict_tensions = tensions
            self.record_task_run("tension_calculation")
            
            return tensions
            
        except Exception as e:
            logger.error(f"Failed to calculate tensions: {e}", exc_info=True)
            self.log_error(e, {})
            return self.conflict_tensions
    
    async def check_conflict_resolution_opportunities(self) -> List[Dict[str, Any]]:
        """Check for opportunities to resolve conflicts"""
        if not self.conflict_synthesizer:
            return []
        
        if not self.should_run_task("conflict_resolution_check"):
            return []
        
        try:
            opportunities = []
            
            for conflict_id, conflict_state in self.active_conflicts.items():
                # Check various resolution conditions
                phase = conflict_state.get('subsystem_data', {}).get('flow', {}).get('phase', '')
                intensity = conflict_state.get('subsystem_data', {}).get('tension', {}).get('level', 0.5)
                
                # Low intensity conflicts can be resolved
                if intensity < 0.3:
                    opportunities.append({
                        'conflict_id': conflict_id,
                        'resolution_type': 'peaceful',
                        'reason': 'Low tension allows peaceful resolution',
                        'priority': 0.7
                    })
                
                # Climax phase conflicts should resolve
                if 'climax' in phase.lower() or 'resolution' in phase.lower():
                    opportunities.append({
                        'conflict_id': conflict_id,
                        'resolution_type': 'narrative',
                        'reason': 'Conflict has reached natural resolution point',
                        'priority': 0.9
                    })
                
                # Check for victory conditions
                victory_data = conflict_state.get('subsystem_data', {}).get('victory', {})
                if victory_data.get('victory_achieved'):
                    opportunities.append({
                        'conflict_id': conflict_id,
                        'resolution_type': 'victory',
                        'reason': 'Victory conditions met',
                        'priority': 1.0
                    })
            
            self.record_task_run("conflict_resolution_check")
            
            # Sort by priority
            opportunities.sort(key=lambda x: x.get('priority', 0), reverse=True)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Failed to check resolution opportunities: {e}", exc_info=True)
            self.log_error(e, {})
            return []
    
    def get_conflict_context_for_response(self) -> Dict[str, Any]:
        """Get conflict context formatted for response generation"""
        context = {
            "active_conflicts": [],
            "tensions": self.conflict_tensions,
            "available_choices": self.conflict_choices,
            "recent_events": self.conflict_events[-5:],
            "scene_conflicts": [],
            "resolution_opportunities": []
        }
        
        # Format active conflicts
        for conflict_id, state in self.active_conflicts.items():
            context["active_conflicts"].append({
                "id": conflict_id,
                "type": state.get('conflict_type', 'unknown'),
                "phase": state.get('subsystem_data', {}).get('flow', {}).get('phase', 'unknown'),
                "intensity": state.get('subsystem_data', {}).get('tension', {}).get('level', 0.5),
                "stakeholders": state.get('subsystem_data', {}).get('stakeholder', {}).get('stakeholders', [])
            })
        
        # Format scene conflicts
        for conflict_id in self.scene_conflicts:
            if conflict_id in self.active_conflicts:
                state = self.active_conflicts[conflict_id]
                context["scene_conflicts"].append({
                    "id": conflict_id,
                    "manifestations": state.get('manifestations', []),
                    "atmosphere": state.get('atmospheric_elements', [])
                })
        
        return context
    
    async def check_npc_scheming(self):
        """Check for scheming behavior in NPCs"""
        if not self.npc_orchestrator or not self.current_scene_npcs:
            return {}
        
        if not self.should_run_task("npc_scheming_check"):
            return {}
        
        try:
            scheming_results = {}
            for npc_id in self.current_scene_npcs:
                result = await self.npc_orchestrator.evaluate_npc_scheming(
                    npc_id=npc_id,
                    with_user_model=True
                )
                scheming_results[npc_id] = result
            
            self.record_task_run("npc_scheming_check")
            return scheming_results
            
        except Exception as e:
            logger.error(f"Failed to check NPC scheming: {e}", exc_info=True)
            self.log_error(e, {})
            return {}
    
    async def run_npc_learning_cycle(self):
        """Run learning cycle for NPCs"""
        if not self.npc_orchestrator or not self.current_scene_npcs:
            return
        
        if not self.should_run_task("npc_learning_cycle"):
            return
        
        try:
            for npc_id in self.current_scene_npcs:
                result = await self.npc_orchestrator.process_npc_learning_cycle(npc_id)
                
                # Update learning metrics based on NPC learning
                if result.get("memory_learning", {}).get("patterns_found"):
                    self.learning_metrics["npc_behavior_prediction_accuracy"] = (
                        self.learning_metrics.get("npc_behavior_prediction_accuracy", 0) * 0.9 +
                        0.1 * (1.0 if result["memory_learning"]["patterns_found"] else 0.0)
                    )
            
            self.record_task_run("npc_learning_cycle")
            
        except Exception as e:
            logger.error(f"Failed to run NPC learning cycle: {e}", exc_info=True)
            self.log_error(e, {})
    
    def get_npc_context_for_response(self) -> Dict[str, Any]:
        """Get NPC context formatted for response generation with memory integration"""
        context = {
            "npcs_present": [],
            "npc_states": {},
            "relationships": {},
            "recent_interactions": [],
            "group_dynamics": {},
            "active_narratives": [],
            "npc_memories": {},
            "npc_beliefs": {},
            "emotional_states": {},
            "memory_predictions": self.memory_predictions if hasattr(self, 'memory_predictions') else []
        }
        
        # Add NPC snapshots
        for npc_id, snapshot in self.npc_snapshots.items():
            if npc_id in self.current_scene_npcs:
                entity_key = f"npc_{npc_id}"
                
                context["npcs_present"].append({
                    "id": npc_id,
                    "name": snapshot.name,
                    "status": snapshot.status.value if hasattr(snapshot.status, 'value') else str(snapshot.status),
                    "activity": snapshot.current_activity,
                    "location": snapshot.location
                })
                
                context["npc_states"][npc_id] = {
                    "emotional": snapshot.emotional_state,
                    "mask_integrity": snapshot.mask_integrity,
                    "stats": snapshot.stats,
                    "scheming_level": snapshot.scheming_level,
                    "current_goals": snapshot.current_goals or []
                }
                
                context["relationships"][npc_id] = snapshot.active_relationships
                
                # Add memory data if available
                if entity_key in self.recent_memories:
                    context["npc_memories"][npc_id] = self.recent_memories[entity_key]
                
                # Add beliefs if available
                if entity_key in self.belief_systems:
                    context["npc_beliefs"][npc_id] = self.belief_systems[entity_key]
                
                # Add emotional memory state if available
                if entity_key in self.emotional_memories:
                    context["emotional_states"][npc_id] = self.emotional_memories[entity_key]
        
        # Add recent interactions
        context["recent_interactions"] = self.npc_interaction_history[-5:]
        
        # Add narrative context if available
        if self.npc_narrative_context:
            context["group_dynamics"] = self.npc_narrative_context.get("group_dynamics", {})
            context["active_narratives"] = self.npc_narrative_context.get("narrative_threads", [])
        
        # Add memory context if available
        if self.memory_context:
            context["memory_narrative"] = self.memory_context.get("narrative_threads", [])
            context["memory_insights"] = self.memory_context.get("analysis", {})
        
        return context
    
    # ────────── EXISTING METHODS (unchanged) ──────────
    
    async def get_active_strategies_cached(self):
        """Get active strategies with caching and lock to prevent thundering herd"""
        current_time = time.time()
        
        # Check cache without lock first
        if self._strategy_cache:
            cache_time, strategies = self._strategy_cache
            if current_time - cache_time < self._strategy_cache_ttl:
                return strategies
        
        # Need to refresh - use lock to prevent multiple refreshes
        async with self._strategy_cache_lock:
            # Double-check cache inside lock
            if self._strategy_cache:
                cache_time, strategies = self._strategy_cache
                if current_time - cache_time < self._strategy_cache_ttl:
                    return strategies
            
            # Fetch new strategies using its own connection
            async with get_db_connection_context() as conn:
                strategies = await get_active_strategies(conn)
            
            # Update cache
            self._strategy_cache = (current_time, strategies)
            return strategies
    
    async def _load_state(self):
        """Load existing state from database"""
        # Use context manager to get a connection
        async with get_db_connection_context() as conn:
            # Load emotional state
            row = await conn.fetchrow("""
                SELECT emotional_state FROM NyxAgentState
                WHERE user_id = $1 AND conversation_id = $2
            """, self.user_id, self.conversation_id)
            
            if row and row["emotional_state"]:
                state = json.loads(row["emotional_state"])
                self.emotional_state.update(state)
            
            # Load scenario state if exists and table is available
            if self._tables_available.get("scenario_states", True):
                try:
                    scenario_row = await conn.fetchrow("""
                        SELECT state_data FROM scenario_states
                        WHERE user_id = $1 AND conversation_id = $2
                        ORDER BY created_at DESC LIMIT 1
                    """, self.user_id, self.conversation_id)
                    
                    if scenario_row and scenario_row["state_data"]:
                        self.scenario_state = json.loads(scenario_row["state_data"])
                except Exception as e:
                    # Table might not exist yet
                    if "does not exist" in str(e) or "no such table" in str(e).lower():
                        logger.info("scenario_states table not found - migrations may need to be run")
                        self._tables_available["scenario_states"] = False
                    else:
                        logger.debug(f"Could not load scenario state: {e}")
    
    def update_performance(self, metric: str, value: Any):
        """Update performance metrics"""
        if metric in self.performance_metrics:
            if isinstance(self.performance_metrics[metric], list):
                self.performance_metrics[metric].append(value)
                # Keep only last entries based on config
                if len(self.performance_metrics[metric]) > Config.MAX_RESPONSE_TIMES:
                    self.performance_metrics[metric] = self.performance_metrics[metric][-Config.MAX_RESPONSE_TIMES:]
            else:
                self.performance_metrics[metric] = value
    
    def should_run_task(self, task_id: str) -> bool:
        """Check if enough time has passed to run task again"""
        if task_id not in self.last_task_runs:
            return True
        
        time_since_run = (datetime.now(timezone.utc) - self.last_task_runs[task_id]).total_seconds()
        return time_since_run >= self.task_intervals.get(task_id, 300)
    
    def record_task_run(self, task_id: str):
        """Record that a task has been run"""
        self.last_task_runs[task_id] = datetime.now(timezone.utc)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log an error with context and aggregate by type"""
        error_type = type(error).__name__
        error_entry = {
            "timestamp": time.time(),
            "error": str(error),
            "type": error_type,
            "context": context or {}
        }
        self.error_log.append(error_entry)
        
        # Track error counts by type
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Update error metrics
        self.performance_metrics["error_rates"]["total"] += 1
        
        # Keep error log bounded
        if len(self.error_log) > Config.MAX_ERROR_LOG_ENTRIES * 2:
            _prune_list(self.error_log, Config.MAX_ERROR_LOG_ENTRIES)
            
        # Log warning if we see repeated errors
        if self.error_counts[error_type] > 10:
            logger.warning(f"Repeated error type {error_type}: {self.error_counts[error_type]} occurrences")
    
    async def learn_from_interaction(self, action: str, outcome: str, success: bool):
        """Learn from an interaction outcome"""
        # Update patterns
        pattern_key = f"{action}_{outcome}"
        if pattern_key not in self.learned_patterns:
            self.learned_patterns[pattern_key] = {
                "occurrences": 0,
                "successes": 0,
                "last_seen": time.time()
            }
        
        pattern = self.learned_patterns[pattern_key]
        pattern["occurrences"] += 1
        if success:
            pattern["successes"] += 1
        pattern["last_seen"] = time.time()
        pattern["success_rate"] = pattern["successes"] / pattern["occurrences"]
        
        # Update adaptation history with emotional state snapshot
        self.adaptation_history.append({
            "timestamp": time.time(),
            "action": action,
            "outcome": outcome,
            "success": success,
            "emotional_state": self.emotional_state.copy()
        })
        
        # Keep adaptation history bounded
        max_history = Config.MAX_ADAPTATION_HISTORY if success else Config.MAX_ADAPTATION_HISTORY // 2
        if len(self.adaptation_history) > max_history * 2:
            self.adaptation_history = self.adaptation_history[-max_history:]
        
        # Prune old patterns (older than 24 hours)
        current_time = time.time()
        self.learned_patterns = {
            k: v for k, v in self.learned_patterns.items()
            if current_time - v.get("last_seen", 0) < 86400
        }
        
        # Update learning metrics
        self._update_learning_metrics()
    
    def should_generate_task(self) -> bool:
        """Determine if we should generate a creative task"""
        context = self.current_context
        
        if not context.get("active_npc_id"):
            return False
            
        scenario_type = context.get("scenario_type", "").lower()
        task_scenarios = ["training", "challenge", "service", "discipline"]
        if not any(t in scenario_type for t in task_scenarios):
            return False
            
        npc_relationship = context.get("npc_relationship_level", 0)
        if npc_relationship < Config.MIN_NPC_RELATIONSHIP_FOR_TASK:
            return False
            
        # Check task timing
        if not self.should_run_task("task_generation"):
            return False
            
        return True
    
    def should_recommend_activities(self) -> bool:
        """Determine if we should recommend activities"""
        context = self.current_context
        
        if not context.get("present_npc_ids"):
            return False
            
        scenario_type = context.get("scenario_type", "").lower()
        if "task" in scenario_type or "challenge" in scenario_type:
            return False
            
        user_input = context.get("user_input", "").lower()
        suggestion_triggers = ["what should", "what can", "what to do", "suggestions", "ideas"]
        if any(trigger in user_input for trigger in suggestion_triggers):
            return True
            
        if context.get("is_scene_transition") or context.get("is_activity_completed"):
            return True
            
        return False
    
    async def handle_high_memory_usage(self):
        """Handle high memory usage by cleaning up"""
        # Trim memory system cache if available
        if hasattr(self.memory_system, 'trim_cache'):
            await self.memory_system.trim_cache()
        
        # Clear old patterns
        self.learned_patterns = dict(list(self.learned_patterns.items())[-Config.MAX_LEARNED_PATTERNS:])
        
        # Clear old history
        self.adaptation_history = self.adaptation_history[-Config.MAX_ADAPTATION_HISTORY:]
        self.error_log = self.error_log[-Config.MAX_ERROR_LOG_ENTRIES:]
        self.npc_interaction_history = self.npc_interaction_history[-50:]
        self.conflict_events = self.conflict_events[-50:]
        
        # Clear old NPC snapshots not in current scene
        self.npc_snapshots = {
            npc_id: snapshot 
            for npc_id, snapshot in self.npc_snapshots.items()
            if npc_id in self.current_scene_npcs
        }
        
        # Clear inactive conflicts
        if len(self.active_conflicts) > 10:
            # Keep only the 10 most recent conflicts
            conflict_ids = sorted(self.active_conflicts.keys())[-10:]
            self.active_conflicts = {
                cid: self.active_conflicts[cid] 
                for cid in conflict_ids
            }
        
        # Clear old conflict choices
        self.conflict_choices = self.conflict_choices[-10:] if self.conflict_choices else []
        
        # Clear performance metrics history
        if "response_times" in self.performance_metrics:
            self.performance_metrics["response_times"] = self.performance_metrics["response_times"][-Config.MAX_RESPONSE_TIMES:]
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Performed memory cleanup")
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage with caching"""
        try:
            current_time = time.time()
            # Check if we need to update the cache
            if (self._cpu_usage_cache is None or 
                current_time - self._cpu_usage_last_update >= self._cpu_usage_update_interval):
                # Update the cache using safe wrapper
                new_value = safe_psutil('cpu_percent', interval=0.1, default=0.0)
                if new_value is not None:
                    self._cpu_usage_cache = new_value
                    self._cpu_usage_last_update = current_time
            
            return self._cpu_usage_cache or 0.0
        except Exception as e:
            logger.debug(f"Failed to get CPU usage: {e}")
            return 0.0

    def db_connection_ctx(self):
        """Get a database connection context manager"""
        return get_db_connection_context()
    
    # Legacy compatibility
    async def get_db_connection(self):
        """DEPRECATED: Use db_connection_ctx() instead"""
        logger.warning("get_db_connection is deprecated, use db_connection_ctx() instead")
        return self.db_connection_ctx()

    async def close_db_connection(self, conn=None):
        """No-op compatibility wrapper"""
        if conn is not None:
            await conn.__aexit__(None, None, None)
    
    def _update_learning_metrics(self):
        """Update learning-related metrics"""
        if self.learned_patterns:
            successful_patterns = sum(1 for p in self.learned_patterns.values() 
                                    if p.get("success_rate", 0) > 0.6)
            self.learning_metrics["pattern_recognition_rate"] = (
                successful_patterns / len(self.learned_patterns)
            )
        
        if self.adaptation_history:
            recent = self.adaptation_history[-Config.MAX_ADAPTATION_HISTORY:]
            successes = sum(1 for a in recent if a["success"])
            self.learning_metrics["adaptation_success_rate"] = successes / len(recent)
        
        # Update NPC behavior prediction based on interaction history
        if self.npc_interaction_history:
            recent_interactions = self.npc_interaction_history[-20:]
            successful_outcomes = sum(1 for i in recent_interactions 
                                     if i.get("result") in ["success", "positive"])
            if recent_interactions:
                self.learning_metrics["npc_behavior_prediction_accuracy"] = (
                    successful_outcomes / len(recent_interactions)
                )
