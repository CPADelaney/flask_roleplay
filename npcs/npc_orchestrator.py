# npcs/npc_orchestrator.py

"""
NPC System Orchestrator with Built-in Canon Integration
Complete replacement for the original orchestrator with all functionality preserved
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Import all NPC subsystems
from npcs.npc_agent import NPCAgent
from npcs.npc_agent_system import NPCAgentSystem
from npcs.npc_coordinator import NPCAgentCoordinator
from npcs.npc_handler import NPCHandler
from npcs.npc_memory import NPCMemoryManager
from npcs.npc_perception import EnvironmentPerception, PerceptionContext
from npcs.npc_relationship import NPCRelationshipManager

# Import Nyx integration
from npcs.nyx_integration import (
    NyxNPCBridge,
    NPCMemoryAccess,
    enhance_npc_with_memory_access,
    create_npc_with_memory_governance
)

# Import preset NPC handler
from npcs.preset_npc_handler import PresetNPCHandler

# Import belief systems
from npcs.belief_system_integration import NPCBeliefSystemIntegration, enhance_npc_with_belief_system
from npcs.npc_belief_formation import NPCBeliefFormation

# Import lore systems
from npcs.lore_context_manager import (
    LoreContextManager,
    LoreImpactAnalyzer,
    LorePropagationSystem,
    LoreContextCache
)

# Import behavior and decision systems
from npcs.npc_behavior import BehaviorEvolution, NPCBehavior
from npcs.npc_decisions import NPCDecisionEngine, DecisionContext

# Import creation and template systems
from npcs.new_npc_creation import NPCCreationHandler
from npcs.dynamic_templates import (
    get_mask_slippage_triggers,
    get_relationship_stages,
    generate_relationship_memory,
    get_reciprocal_label,
    get_calendar_months,
    get_calendar_day_names,
    generate_core_beliefs,
    get_semantic_seed_topics,
    get_trauma_keywords,
    get_alternative_names
)

# Import learning systems
from npcs.npc_learning_adaptation import NPCLearningManager, NPCLearningAdaptation

# Import related systems
from memory.wrapper import MemorySystem
from lore.core.lore_system import LoreSystem
from logic.dynamic_relationships import OptimizedRelationshipManager
from logic.npc_narrative_progression import (
    get_npc_narrative_stage,
    progress_npc_narrative_stage,
    NPCNarrativeStage
)

# Import canon systems
from lore.core import canon
from lore.core.context import CanonicalContext
from lore.core.validation import CanonValidationAgent

from db.connection import get_db_connection_context
from agents import RunContextWrapper, Runner

logger = logging.getLogger(__name__)


class NPCStatus(Enum):
    """NPC status states"""
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    IN_CONVERSATION = "in_conversation"
    IN_CONFLICT = "in_conflict"
    TRAVELING = "traveling"
    WORKING = "working"
    RESTING = "resting"
    UNCONSCIOUS = "unconscious"
    DECISION_MAKING = "decision_making"
    SCHEMING = "scheming"
    UNDER_NYX_DIRECTIVE = "under_nyx_directive"


@dataclass
class NPCSnapshot:
    """Snapshot of an NPC's current state for narrative purposes"""
    npc_id: int
    name: str
    location: str
    status: NPCStatus
    current_activity: Optional[str]
    emotional_state: Dict[str, Any]
    recent_memories: List[Dict[str, Any]]
    active_relationships: Dict[str, Any]
    current_beliefs: List[Dict[str, Any]]
    narrative_stage: Optional[str]
    mask_integrity: float
    stats: Dict[str, float]
    schedule: Optional[Dict[str, Any]]
    scheming_level: int = 0
    paranoia_level: int = 0
    current_goals: List[Dict[str, Any]] = None
    decision_history: List[Dict[str, Any]] = None
    nyx_directives: List[Dict[str, Any]] = None
    special_mechanics: Dict[str, Any] = None
    canonical_events: List[Dict[str, Any]] = None  # Added for canon
    
    def to_narrative_context(self) -> Dict[str, Any]:
        """Convert snapshot to narrative-ready context"""
        return {
            "character": {
                "id": self.npc_id,
                "name": self.name,
                "location": self.location,
                "status": self.status.value,
                "activity": self.current_activity or "idle"
            },
            "psychological": {
                "emotional_state": self.emotional_state,
                "mask_integrity": self.mask_integrity,
                "recent_thoughts": [m.get("memory_text") for m in self.recent_memories[:3]],
                "scheming_level": self.scheming_level,
                "paranoia_level": self.paranoia_level
            },
            "social": {
                "relationships": self.active_relationships,
                "narrative_stage": self.narrative_stage
            },
            "traits": self.stats,
            "goals": self.current_goals or [],
            "decision_patterns": self.decision_history or [],
            "governance": {
                "nyx_directives": self.nyx_directives or [],
                "special_mechanics": self.special_mechanics or {}
            },
            "canon": {
                "recent_events": self.canonical_events or []
            }
        }


@dataclass
class CanonCheckResult:
    """Result of a canon consistency check"""
    is_consistent: bool
    conflicts: List[Dict[str, Any]]
    warnings: List[str]
    suggestions: List[Dict[str, Any]]
    canon_references: List[Dict[str, Any]]


class NPCOrchestrator:
    """
    Master orchestrator for all NPC operations with built-in canon integration.
    Single access point for narrative generation and game systems.
    """
    
    def __init__(self, user_id: int, conversation_id: int, enable_canon: bool = True):
        """
        Initialize the orchestrator with all subsystems.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            enable_canon: Whether to enable canon integration (default: True)
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Canon configuration
        self.enable_canon = enable_canon
        self.auto_canonize = True
        self.check_canon_consistency = True
        self.canon_significance_threshold = 4  # Minimum significance to canonize
        
        # Core systems (lazy-loaded)
        self._agent_system: Optional[NPCAgentSystem] = None
        self._coordinator: Optional[NPCAgentCoordinator] = None
        self._handler: Optional[NPCHandler] = None
        self._creation_handler: Optional[NPCCreationHandler] = None
        self._preset_handler: Optional[PresetNPCHandler] = None
        self._memory_system: Optional[MemorySystem] = None
        self._lore_system: Optional[LoreSystem] = None
        self._relationship_manager: Optional[OptimizedRelationshipManager] = None
        
        # Canon integration
        self._validation_agent: Optional[CanonValidationAgent] = None
        self._canon_cache: Dict[str, Tuple[Any, datetime]] = {}
        self._canon_cache_ttl = timedelta(minutes=5)
        
        # Nyx integration
        self._nyx_bridge: Optional[NyxNPCBridge] = None
        self._nyx_memory_access: Dict[int, NPCMemoryAccess] = {}
        
        # Belief systems
        self._belief_integration: Optional[NPCBeliefSystemIntegration] = None
        self._belief_formations: Dict[int, NPCBeliefFormation] = {}
        
        # Lore context systems
        self._lore_context_manager: Optional[LoreContextManager] = None
        self._lore_impact_analyzer: Optional[LoreImpactAnalyzer] = None
        self._lore_propagation_system: Optional[LorePropagationSystem] = None
        
        # Behavior and decision systems
        self._behavior_evolution: Optional[BehaviorEvolution] = None
        self._npc_behaviors: Dict[int, NPCBehavior] = {}
        self._decision_engines: Dict[int, NPCDecisionEngine] = {}
        
        # Learning systems
        self._learning_manager: Optional[NPCLearningManager] = None
        self._learning_adaptations: Dict[int, NPCLearningAdaptation] = {}
        
        # Memory managers for individual NPCs
        self._memory_managers: Dict[int, NPCMemoryManager] = {}
        
        # Perception systems for individual NPCs
        self._perception_systems: Dict[int, EnvironmentPerception] = {}
        
        # Relationship managers for individual NPCs
        self._relationship_managers: Dict[int, NPCRelationshipManager] = {}
        
        # Caches
        self._npc_cache: Dict[int, NPCAgent] = {}
        self._snapshot_cache: Dict[int, Tuple[NPCSnapshot, datetime]] = {}
        self._lore_context_cache: Optional[LoreContextCache] = None
        self._snapshot_ttl = timedelta(minutes=5)
        
        # Tracking
        self._active_npcs: Set[int] = set()
        self._npc_status: Dict[int, NPCStatus] = {}
    
    # ==================== CANON HELPERS ====================
    
    def _get_canonical_context(self) -> CanonicalContext:
        """Get canonical context for operations"""
        return CanonicalContext(
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
    
    async def _get_validation_agent(self) -> CanonValidationAgent:
        """Get or create validation agent for canon checking"""
        if self._validation_agent is None:
            self._validation_agent = CanonValidationAgent()
        return self._validation_agent
    
    async def _check_npc_canon_consistency(
        self,
        npc_id: int,
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> CanonCheckResult:
        """Check if an NPC action is consistent with established canon."""
        if not self.enable_canon or not self.check_canon_consistency:
            return CanonCheckResult(True, [], [], [], [])
        
        conflicts = []
        warnings = []
        suggestions = []
        canon_references = []
        
        async with get_db_connection_context() as conn:
            # Get NPC's canonical data
            npc_data = await conn.fetchrow("""
                SELECT npc_name, role, personality_traits, current_location,
                       backstory, goals, special_mechanics, mask_integrity,
                       dominance, cruelty, betrayal_planning, scheming_level
                FROM NPCStats
                WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
            """, npc_id, self.user_id, self.conversation_id)
            
            if not npc_data:
                return CanonCheckResult(
                    is_consistent=False,
                    conflicts=[{"type": "npc_not_found", "npc_id": npc_id}],
                    warnings=["NPC not found in canon"],
                    suggestions=[{"action": "create_npc", "npc_id": npc_id}],
                    canon_references=[]
                )
            
            # Check location consistency
            if "location" in action and action["location"] != npc_data["current_location"]:
                location_exists = await conn.fetchrow("""
                    SELECT id FROM Locations
                    WHERE location_name = $1 AND user_id = $2 AND conversation_id = $3
                """, action["location"], self.user_id, self.conversation_id)
                
                if not location_exists:
                    conflicts.append({
                        "type": "unknown_location",
                        "location": action["location"]
                    })
                    suggestions.append({
                        "action": "create_location",
                        "location": action["location"]
                    })
            
            # Check personality consistency
            if "behavior" in action:
                personality_traits = npc_data.get("personality_traits", [])
                validation_agent = await self._get_validation_agent()
                
                prompt = f"""
                NPC {npc_data['npc_name']} has these personality traits: {personality_traits}
                Current mask integrity: {npc_data.get('mask_integrity', 100)}%
                
                They want to: {action['behavior']}
                
                Is this behavior consistent with their personality?
                Reply with JSON: {{"consistent": true/false, "reason": "explanation"}}
                """
                
                result = await Runner.run(validation_agent.agent, prompt)
                try:
                    check = json.loads(result.final_output)
                    if not check["consistent"]:
                        warnings.append(f"Behavior inconsistency: {check['reason']}")
                except:
                    pass
        
        return CanonCheckResult(
            is_consistent=len(conflicts) == 0,
            conflicts=conflicts,
            warnings=warnings,
            suggestions=suggestions,
            canon_references=canon_references
        )
    
    async def _canonize_npc_action(
        self,
        npc_id: int,
        action: Dict[str, Any],
        result: Dict[str, Any],
        significance: int = 5
    ) -> Dict[str, Any]:
        """Make an NPC action canonical by updating all relevant canon systems."""
        if not self.enable_canon or not self.auto_canonize:
            return {"skipped": True}
        
        if significance < self.canon_significance_threshold:
            return {"skipped": True, "reason": "below significance threshold"}
        
        ctx = self._get_canonical_context()
        canonized = {
            "success": True,
            "updates": [],
            "events_logged": []
        }
        
        async with get_db_connection_context() as conn:
            # Get NPC name
            npc_data = await conn.fetchrow("""
                SELECT npc_name FROM NPCStats
                WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
            """, npc_id, self.user_id, self.conversation_id)
            
            if not npc_data:
                return {"success": False, "error": "NPC not found"}
            
            npc_name = npc_data["npc_name"]
            
            # Update location if changed
            if "new_location" in result:
                location_id = await canon.find_or_create_location(
                    ctx, conn,
                    location_name=result["new_location"],
                    location_type="settlement"
                )
                
                await conn.execute("""
                    UPDATE NPCStats
                    SET current_location = $1
                    WHERE npc_id = $2 AND user_id = $3 AND conversation_id = $4
                """, result["new_location"], npc_id, self.user_id, self.conversation_id)
                
                await canon.log_canonical_event(
                    ctx, conn,
                    f"{npc_name} moved to {result['new_location']}",
                    tags=["npc", "movement", "location"],
                    significance=significance
                )
                canonized["events_logged"].append(f"Movement to {result['new_location']}")
            
            # Canonize significant statements
            if "statement" in action and significance >= 6:
                await canon.log_canonical_event(
                    ctx, conn,
                    f'{npc_name} declared: "{action["statement"][:200]}"',
                    tags=["npc", "declaration", "dialogue"],
                    significance=significance
                )
                canonized["events_logged"].append("Important declaration")
            
            # Update relationships if affected
            if "relationship_changes" in result:
                for rel_change in result["relationship_changes"]:
                    link_id = await canon.find_or_create_social_link(
                        ctx, conn,
                        entity1_type="npc",
                        entity1_id=npc_id,
                        entity2_type=rel_change.get("target_type", "npc"),
                        entity2_id=rel_change["target_id"],
                        link_type=rel_change.get("new_type", "neutral")
                    )
                    
                    canonized["updates"].append({
                        "type": "relationship",
                        "link_id": link_id
                    })
        
        return canonized
    
    # ==================== INITIALIZATION ====================
    
    async def initialize(self) -> None:
        """Initialize all core systems."""
        logger.info(f"Initializing NPC Orchestrator for user {self.user_id}, conversation {self.conversation_id}")
        logger.info(f"Canon integration: {'ENABLED' if self.enable_canon else 'DISABLED'}")
        
        # Initialize agent system (manages all NPCs)
        self._agent_system = NPCAgentSystem(self.user_id, self.conversation_id, None)
        await self._agent_system.initialize_agents()
        
        # Initialize memory and lore systems
        self._memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        self._lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
        
        # Initialize behavior evolution
        self._behavior_evolution = BehaviorEvolution(self.user_id, self.conversation_id)
        
        # Initialize lore context cache
        self._lore_context_cache = LoreContextCache(ttl_seconds=300)
        
        # Initialize Nyx bridge
        self._nyx_bridge = NyxNPCBridge(self.user_id, self.conversation_id)
        
        # Load active NPCs
        await self._load_active_npcs()
    
    async def _load_active_npcs(self) -> None:
        """Load all active NPCs from the database."""
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT npc_id, npc_name, current_location, scheming_level, 
                       betrayal_planning, personality_traits
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2
                AND introduced = TRUE
            """, self.user_id, self.conversation_id)
            
            for row in rows:
                npc_id = row['npc_id']
                self._active_npcs.add(npc_id)
                
                # Set initial status based on scheming level
                scheming_level = row.get('scheming_level', 0) or 0
                if scheming_level >= 5:
                    self._npc_status[npc_id] = NPCStatus.SCHEMING
                else:
                    self._npc_status[npc_id] = NPCStatus.IDLE
    
    # ==================== LAZY LOADERS ====================
    
    async def _get_agent_system(self) -> NPCAgentSystem:
        """Get or create agent system."""
        if self._agent_system is None:
            self._agent_system = NPCAgentSystem(self.user_id, self.conversation_id, None)
            await self._agent_system.initialize_agents()
        return self._agent_system
    
    async def _get_coordinator(self) -> NPCAgentCoordinator:
        """Get or create coordinator."""
        if self._coordinator is None:
            self._coordinator = NPCAgentCoordinator(self.user_id, self.conversation_id)
        return self._coordinator
    
    async def _get_handler(self) -> NPCHandler:
        """Get or create handler."""
        if self._handler is None:
            self._handler = NPCHandler(self.user_id, self.conversation_id)
        return self._handler
    
    async def _get_creation_handler(self) -> NPCCreationHandler:
        """Get or create creation handler."""
        if self._creation_handler is None:
            self._creation_handler = NPCCreationHandler()
        return self._creation_handler
    
    async def _get_preset_handler(self) -> PresetNPCHandler:
        """Get or create preset handler."""
        if self._preset_handler is None:
            self._preset_handler = PresetNPCHandler()
        return self._preset_handler
    
    async def _get_nyx_bridge(self) -> NyxNPCBridge:
        """Get or create Nyx bridge."""
        if self._nyx_bridge is None:
            self._nyx_bridge = NyxNPCBridge(self.user_id, self.conversation_id)
        return self._nyx_bridge
    
    async def _get_nyx_memory_access(self, npc_id: int) -> NPCMemoryAccess:
        """Get or create Nyx memory access for an NPC."""
        if npc_id not in self._nyx_memory_access:
            self._nyx_memory_access[npc_id] = NPCMemoryAccess(
                npc_id, self.user_id, self.conversation_id
            )
        return self._nyx_memory_access[npc_id]
    
    async def _get_memory_manager(self, npc_id: int) -> NPCMemoryManager:
        """Get or create memory manager for a specific NPC."""
        if npc_id not in self._memory_managers:
            self._memory_managers[npc_id] = NPCMemoryManager(
                npc_id, self.user_id, self.conversation_id
            )
        return self._memory_managers[npc_id]
    
    async def _get_perception_system(self, npc_id: int) -> EnvironmentPerception:
        """Get or create perception system for a specific NPC."""
        if npc_id not in self._perception_systems:
            self._perception_systems[npc_id] = EnvironmentPerception(
                npc_id, self.user_id, self.conversation_id
            )
        return self._perception_systems[npc_id]
    
    async def _get_relationship_manager(self, npc_id: int) -> NPCRelationshipManager:
        """Get or create relationship manager for a specific NPC."""
        if npc_id not in self._relationship_managers:
            self._relationship_managers[npc_id] = NPCRelationshipManager(
                npc_id, self.user_id, self.conversation_id
            )
        return self._relationship_managers[npc_id]
    
    async def _get_belief_integration(self) -> NPCBeliefSystemIntegration:
        """Get or create belief integration."""
        if self._belief_integration is None:
            self._belief_integration = NPCBeliefSystemIntegration(self.user_id, self.conversation_id)
            await self._belief_integration.initialize()
        return self._belief_integration
    
    async def _get_belief_formation(self, npc_id: int) -> NPCBeliefFormation:
        """Get or create belief formation for a specific NPC."""
        if npc_id not in self._belief_formations:
            self._belief_formations[npc_id] = NPCBeliefFormation(
                self.user_id, self.conversation_id, npc_id
            )
            await self._belief_formations[npc_id].initialize()
        return self._belief_formations[npc_id]
    
    async def _get_lore_context_manager(self) -> LoreContextManager:
        """Get or create lore context manager."""
        if self._lore_context_manager is None:
            self._lore_context_manager = LoreContextManager(self.user_id, self.conversation_id)
        return self._lore_context_manager
    
    async def _get_lore_impact_analyzer(self) -> LoreImpactAnalyzer:
        """Get or create lore impact analyzer."""
        if self._lore_impact_analyzer is None:
            self._lore_impact_analyzer = LoreImpactAnalyzer(self.user_id, self.conversation_id)
        return self._lore_impact_analyzer
    
    async def _get_lore_propagation_system(self) -> LorePropagationSystem:
        """Get or create lore propagation system."""
        if self._lore_propagation_system is None:
            self._lore_propagation_system = LorePropagationSystem(self.user_id, self.conversation_id)
        return self._lore_propagation_system
    
    async def _get_behavior_evolution(self) -> BehaviorEvolution:
        """Get or create behavior evolution system."""
        if self._behavior_evolution is None:
            self._behavior_evolution = BehaviorEvolution(self.user_id, self.conversation_id)
        return self._behavior_evolution
    
    async def _get_npc_behavior(self, npc_id: int) -> NPCBehavior:
        """Get or create NPC behavior for a specific NPC."""
        if npc_id not in self._npc_behaviors:
            self._npc_behaviors[npc_id] = NPCBehavior(npc_id)
        return self._npc_behaviors[npc_id]
    
    async def _get_decision_engine(self, npc_id: int) -> NPCDecisionEngine:
        """Get or create decision engine for a specific NPC."""
        if npc_id not in self._decision_engines:
            self._decision_engines[npc_id] = await NPCDecisionEngine.create(
                npc_id, self.user_id, self.conversation_id
            )
        return self._decision_engines[npc_id]
    
    async def _get_learning_manager(self) -> NPCLearningManager:
        """Get or create learning manager."""
        if self._learning_manager is None:
            self._learning_manager = NPCLearningManager(self.user_id, self.conversation_id)
            await self._learning_manager.initialize()
        return self._learning_manager
    
    async def _get_learning_adaptation(self, npc_id: int) -> NPCLearningAdaptation:
        """Get or create learning adaptation for a specific NPC."""
        if npc_id not in self._learning_adaptations:
            learning_manager = await self._get_learning_manager()
            self._learning_adaptations[npc_id] = learning_manager.get_learning_system_for_npc(npc_id)
            await self._learning_adaptations[npc_id].initialize()
        return self._learning_adaptations[npc_id]
    
    # ==================== NPC CREATION & MANAGEMENT ====================
    
    async def create_npc(
        self,
        environment_desc: Optional[str] = None,
        archetype_names: Optional[List[str]] = None,
        specific_traits: Optional[Dict[str, Any]] = None,
        use_dynamic_templates: bool = True
    ) -> Dict[str, Any]:
        """Create a new NPC with optional dynamic template generation and canon checking."""
        
        # Check if similar NPC already exists in canon
        if self.enable_canon and specific_traits and "name" in specific_traits:
            ctx = self._get_canonical_context()
            async with get_db_connection_context() as conn:
                # Use canon's semantic matching to check for duplicates
                existing_id = await conn.fetchrow("""
                    SELECT npc_id FROM NPCStats
                    WHERE LOWER(npc_name) = LOWER($1)
                    AND user_id = $2 AND conversation_id = $3
                """, specific_traits["name"], self.user_id, self.conversation_id)
                
                if existing_id:
                    return {
                        "success": False,
                        "reason": "NPC with this name already exists",
                        "existing_npc_id": existing_id["npc_id"]
                    }
        
        # Generate dynamic templates if requested
        if use_dynamic_templates and environment_desc:
            if not specific_traits:
                specific_traits = {}
            
            archetype_summary = " ".join(archetype_names) if archetype_names else "generic character"
            personality_traits = specific_traits.get("personality_traits", [])
            personality_traits_str = ", ".join(personality_traits) if personality_traits else "neutral"
            
            # Generate dynamic beliefs
            beliefs = await generate_core_beliefs(
                archetype_summary, 
                personality_traits_str,
                environment_desc,
                n=5
            )
            specific_traits["core_beliefs"] = beliefs
            
            # Generate semantic topics for knowledge graph
            topics = await get_semantic_seed_topics(archetype_summary, environment_desc)
            specific_traits["knowledge_topics"] = topics
        
        handler = await self._get_creation_handler()
        result = await handler.create_npc(
            environment_desc=environment_desc,
            archetype_names=archetype_names,
            specific_traits=specific_traits,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        if "npc_id" in result:
            npc_id = result["npc_id"]
            self._active_npcs.add(npc_id)
            self._npc_status[npc_id] = NPCStatus.IDLE
            
            # Initialize belief system for new NPC
            belief_formation = await self._get_belief_formation(npc_id)
            
            # Form initial beliefs if we generated them
            if "core_beliefs" in specific_traits:
                for belief_text in specific_traits["core_beliefs"]:
                    ctx = RunContextWrapper(context={
                        'user_id': self.user_id,
                        'conversation_id': self.conversation_id,
                        'npc_id': npc_id
                    })
                    await belief_formation.form_subjective_belief_from_observation(
                        ctx, belief_text, factuality=0.9
                    )
            
            # Canonize creation if enabled
            if self.enable_canon and self.auto_canonize:
                ctx = self._get_canonical_context()
                async with get_db_connection_context() as conn:
                    await canon.log_canonical_event(
                        ctx, conn,
                        f"New NPC created: {specific_traits.get('name', 'Unknown')} ({', '.join(archetype_names or ['no archetype'])})",
                        tags=["npc", "creation", "canonical"],
                        significance=8
                    )
        
        return result
    
    async def create_preset_npc(
        self,
        npc_data: Dict[str, Any],
        story_context: Dict[str, Any]
    ) -> int:
        """Create a rich preset NPC with full features and canon integration."""
        
        # Check for duplicates if canon enabled
        if self.enable_canon and "name" in npc_data:
            ctx = self._get_canonical_context()
            async with get_db_connection_context() as conn:
                existing = await conn.fetchrow("""
                    SELECT npc_id FROM NPCStats
                    WHERE LOWER(npc_name) = LOWER($1)
                    AND user_id = $2 AND conversation_id = $3
                """, npc_data["name"], self.user_id, self.conversation_id)
                
                if existing:
                    logger.warning(f"NPC '{npc_data['name']}' already exists")
                    return existing["npc_id"]
        
        handler = await self._get_preset_handler()
        
        ctx = RunContextWrapper(context={
            'user_id': self.user_id,
            'conversation_id': self.conversation_id
        })
        
        npc_id = await handler.create_detailed_npc(ctx, npc_data, story_context)
        
        # Add to tracking
        self._active_npcs.add(npc_id)
        self._npc_status[npc_id] = NPCStatus.IDLE
        
        # Canonize if enabled
        if self.enable_canon and self.auto_canonize:
            ctx = self._get_canonical_context()
            async with get_db_connection_context() as conn:
                await canon.log_canonical_event(
                    ctx, conn,
                    f"Preset NPC created: {npc_data.get('name', 'Unknown')} - {npc_data.get('role', 'no role')}",
                    tags=["npc", "preset", "creation", "canonical"],
                    significance=8
                )
        
        # Invalidate cache
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        
        return npc_id
    
    async def spawn_multiple_npcs(
        self,
        count: int = 3,
        environment_desc: Optional[str] = None,
        use_dynamic_names: bool = True
    ) -> List[int]:
        """Spawn multiple NPCs at once with optional dynamic name generation."""
        # Generate alternative names if requested
        if use_dynamic_names and environment_desc:
            names = await get_alternative_names("mixed", environment_desc, n=count)
        else:
            names = None
        
        handler = await self._get_creation_handler()
        npc_ids = await handler.spawn_multiple_npcs(
            count=count,
            environment_desc=environment_desc,
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            names=names  # Pass generated names if available
        )
        
        for npc_id in npc_ids:
            self._active_npcs.add(npc_id)
            self._npc_status[npc_id] = NPCStatus.IDLE
            
        return npc_ids
    
    # ==================== NYX INTEGRATION ====================
    
    async def issue_nyx_directive_to_npc(
        self,
        npc_id: int,
        directive_type: str,
        directive_data: Dict[str, Any],
        priority: str = "HIGH",
        duration_minutes: int = 30
    ) -> Dict[str, Any]:
        """Issue a Nyx directive to an NPC."""
        bridge = await self._get_nyx_bridge()
        
        from nyx.nyx_governance import DirectiveType, DirectivePriority
        
        # Convert string types to enums
        directive_type_enum = DirectiveType[directive_type.upper()]
        priority_enum = DirectivePriority[priority.upper()]
        
        directive_id = await bridge.governor.issue_directive(
            npc_id=npc_id,
            directive_type=directive_type_enum,
            directive_data=directive_data,
            priority=priority_enum,
            duration_minutes=duration_minutes
        )
        
        # Update NPC status
        if directive_id > 0:
            self._npc_status[npc_id] = NPCStatus.UNDER_NYX_DIRECTIVE
        
        # Invalidate cache
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        
        return {
            "directive_id": directive_id,
            "status": "issued" if directive_id > 0 else "failed",
            "npc_id": npc_id
        }
    
    async def issue_nyx_scene_directives(
        self,
        scene_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Issue Nyx directives to all NPCs in a scene."""
        bridge = await self._get_nyx_bridge()
        result = await bridge.issue_scene_directives(scene_plan)
        
        # Update status for affected NPCs
        for npc_id, directive_result in result.get("results", {}).items():
            if directive_result.get("status") == "issued":
                self._npc_status[npc_id] = NPCStatus.UNDER_NYX_DIRECTIVE
        
        # Invalidate caches
        for npc_id in result.get("results", {}).keys():
            if npc_id in self._snapshot_cache:
                del self._snapshot_cache[npc_id]
        
        return result
    
    async def process_nyx_governed_event(
        self,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process an event through Nyx governance."""
        bridge = await self._get_nyx_bridge()
        result = await bridge.process_group_event(event_type, event_data)
        
        # Invalidate caches for affected NPCs
        for npc_id in result.get("affected_npcs", []):
            if npc_id in self._snapshot_cache:
                del self._snapshot_cache[npc_id]
        
        return result
    
    # ==================== MEMORY OPERATIONS (CANON-ENHANCED) ====================
    
    async def add_memory_for_npc(
        self,
        npc_id: int,
        memory_text: str,
        memory_type: str = "observation",
        significance: int = 3,
        emotional_valence: int = 0,
        tags: Optional[List[str]] = None,
        feminine_context: bool = False,
        use_nyx_governance: bool = True
    ) -> Optional[int]:
        """Add a memory for an NPC with canon consistency checking."""
        
        # Check memory consistency with canon if enabled
        if self.enable_canon and self.check_canon_consistency:
            async with get_db_connection_context() as conn:
                # Check for contradictory memories
                existing_memories = await conn.fetch("""
                    SELECT memory_text FROM NPCMemories
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                    AND memory_type = $4
                    ORDER BY created_at DESC
                    LIMIT 10
                """, npc_id, self.user_id, self.conversation_id, memory_type)
                
                if existing_memories:
                    validation_agent = await self._get_validation_agent()
                    memory_texts = [m["memory_text"] for m in existing_memories]
                    
                    prompt = f"""
                    Existing memories: {json.dumps(memory_texts, indent=2)}
                    New memory: "{memory_text}"
                    
                    Does this contradict existing memories?
                    Reply with JSON: {{"contradicts": true/false, "severity": "none/minor/major"}}
                    """
                    
                    result = await Runner.run(validation_agent.agent, prompt)
                    try:
                        check = json.loads(result.final_output)
                        if check["contradicts"] and check["severity"] == "major":
                            logger.warning(f"Memory contradicts canon: {memory_text}")
                            # You might want to handle this differently
                    except:
                        pass
        
        # Add memory through appropriate system
        if use_nyx_governance:
            memory_access = await self._get_nyx_memory_access(npc_id)
            result = await memory_access.remember(
                memory_text=memory_text,
                importance="high" if significance >= 7 else "medium" if significance >= 4 else "low",
                emotional=abs(emotional_valence) > 5,
                tags=tags
            )
            memory_id = result.get("memory_id")
        else:
            memory_manager = await self._get_memory_manager(npc_id)
            memory_id = await memory_manager.add_memory(
                memory_text=memory_text,
                memory_type=memory_type,
                significance=significance,
                emotional_valence=emotional_valence,
                tags=tags,
                feminine_context=feminine_context
            )
        
        # Canonize significant memories
        if self.enable_canon and self.auto_canonize and significance >= 7:
            ctx = self._get_canonical_context()
            async with get_db_connection_context() as conn:
                npc_data = await conn.fetchrow("""
                    SELECT npc_name FROM NPCStats
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                """, npc_id, self.user_id, self.conversation_id)
                
                if npc_data:
                    await canon.log_canonical_event(
                        ctx, conn,
                        f"{npc_data['npc_name']} formed significant memory: {memory_text[:100]}",
                        tags=["npc", "memory", memory_type] + (tags or []),
                        significance=significance
                    )
        
        return memory_id
    
    async def recall_memories_for_npc(
        self,
        npc_id: int,
        query: str = "",
        context: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        femdom_focus: bool = False,
        use_nyx_governance: bool = True
    ) -> Dict[str, Any]:
        """Recall memories for an NPC."""
        if use_nyx_governance:
            # Use Nyx-governed memory access
            memory_access = await self._get_nyx_memory_access(npc_id)
            return await memory_access.recall(
                query=query,
                context=str(context) if context else None,
                limit=limit
            )
        else:
            # Use direct memory manager
            memory_manager = await self._get_memory_manager(npc_id)
            return await memory_manager.retrieve_memories(
                query=query,
                context=context,
                limit=limit,
                femdom_focus=femdom_focus
            )
    
    async def update_npc_emotional_state(
        self,
        npc_id: int,
        primary_emotion: str,
        intensity: float,
        trigger: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update an NPC's emotional state."""
        memory_manager = await self._get_memory_manager(npc_id)
        result = await memory_manager.update_emotional_state(
            primary_emotion=primary_emotion,
            intensity=intensity,
            trigger=trigger
        )
        
        # Invalidate cache
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        
        return result
    
    async def trigger_npc_mask_slippage(
        self,
        npc_id: int,
        trigger: str,
        severity: Optional[int] = None,
        femdom_context: bool = False
    ) -> Dict[str, Any]:
        """Trigger a mask slippage event for an NPC."""
        memory_manager = await self._get_memory_manager(npc_id)
        result = await memory_manager.generate_mask_slippage(
            trigger=trigger,
            severity=severity,
            femdom_context=femdom_context
        )
        
        # Invalidate cache
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        
        return result
    
    async def run_memory_maintenance_for_npc(
        self,
        npc_id: int,
        include_femdom_maintenance: bool = True
    ) -> Dict[str, Any]:
        """Run memory maintenance for an NPC."""
        memory_manager = await self._get_memory_manager(npc_id)
        result = await memory_manager.run_memory_maintenance(
            include_femdom_maintenance=include_femdom_maintenance
        )
        
        # Invalidate cache
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        
        return result
    
    # ==================== PERCEPTION OPERATIONS ====================
    
    async def get_npc_perception(
        self,
        npc_id: int,
        context: Optional[Dict[str, Any]] = None,
        detail_level: str = "auto"
    ) -> Dict[str, Any]:
        """Get environment perception for an NPC."""
        perception_system = await self._get_perception_system(npc_id)
        
        # Build perception context
        if context:
            perception_context = PerceptionContext(**context) if not isinstance(context, PerceptionContext) else context
        else:
            perception_context = await self._build_npc_perception_context(npc_id)
        
        # Get perception
        result = await perception_system.perceive_environment(
            perception_context,
            detail_level=detail_level
        )
        
        return result.dict() if hasattr(result, 'dict') else result
    
    async def _build_npc_perception_context(self, npc_id: int) -> PerceptionContext:
        """Build perception context for an NPC."""
        # Get current location
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT current_location FROM NPCStats
                WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
            """, npc_id, self.user_id, self.conversation_id)
            
            location = row['current_location'] if row else "Unknown"
        
        # Get time of day
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay
                WHERE key = 'TimeOfDay' AND user_id = $1 AND conversation_id = $2
            """, self.user_id, self.conversation_id)
            
            time_of_day = row['value'] if row else "Unknown"
        
        # Get entities at location
        entities = []
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT npc_id, npc_name FROM NPCStats
                WHERE current_location = $1 AND user_id = $2 AND conversation_id = $3
                AND npc_id != $4
            """, location, self.user_id, self.conversation_id, npc_id)
            
            for row in rows:
                entities.append({
                    "type": "npc",
                    "id": row["npc_id"],
                    "name": row["npc_name"]
                })
        
        # Add player
        entities.append({
            "type": "player",
            "id": self.user_id,
            "name": "Player"
        })
        
        return PerceptionContext(
            location=location,
            time_of_day=time_of_day,
            entities_present=entities
        )
    
    async def evaluate_action_significance_for_npc(
        self,
        npc_id: int,
        action: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate if an action is significant enough for an NPC to remember."""
        perception_system = await self._get_perception_system(npc_id)
        significance = await perception_system.evaluate_action_significance(action, result)
        return significance.dict() if hasattr(significance, 'dict') else significance
    
    # ==================== DECISION MAKING (CANON-ENHANCED) ====================
    
    async def make_npc_decision(
        self,
        npc_id: int,
        perception: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a decision for an NPC using the decision engine with canon constraints."""
        self._npc_status[npc_id] = NPCStatus.DECISION_MAKING
        
        try:
            # Get or create decision engine
            decision_engine = await self._get_decision_engine(npc_id)
            
            # Get perception if not provided
            if perception is None:
                perception = await self.get_npc_perception(npc_id)
            
            # Add canonical constraints if enabled
            if self.enable_canon:
                async with get_db_connection_context() as conn:
                    # Get canonical events
                    events = await conn.fetch("""
                        SELECT event_text, tags, significance, timestamp
                        FROM CanonicalEvents
                        WHERE user_id = $1 AND conversation_id = $2
                        ORDER BY timestamp DESC
                        LIMIT 5
                    """, self.user_id, self.conversation_id)
                    
                    perception["canonical_constraints"] = {
                        "recent_events": [dict(e) for e in events]
                    }
            
            # Make decision
            decision = await decision_engine.decide(perception)
            
            # Check decision consistency with canon
            if self.enable_canon and self.check_canon_consistency and "action" in decision:
                consistency_check = await self._check_npc_canon_consistency(
                    npc_id, decision["action"], perception
                )
                
                if not consistency_check.is_consistent:
                    decision["canon_warnings"] = consistency_check.warnings
                    decision["canon_adjustments"] = consistency_check.suggestions
            
            # Canonize significant decisions
            if self.enable_canon and self.auto_canonize and decision.get("significance", 0) >= self.canon_significance_threshold:
                canon_result = await self._canonize_npc_action(
                    npc_id=npc_id,
                    action=decision.get("action", {}),
                    result=decision,
                    significance=decision.get("significance", 5)
                )
                decision["canonized"] = canon_result
            
            # Invalidate cache
            if npc_id in self._snapshot_cache:
                del self._snapshot_cache[npc_id]
            
            return decision
        finally:
            self._npc_status[npc_id] = NPCStatus.IDLE
    
    # ==================== BEHAVIOR EVOLUTION ====================
    
    async def evaluate_npc_scheming(
        self,
        npc_id: int,
        with_user_model: bool = False
    ) -> Dict[str, Any]:
        """Evaluate and update NPC scheming behavior."""
        behavior_evolution = await self._get_behavior_evolution()
        
        if with_user_model:
            adjustments = await behavior_evolution.evaluate_npc_scheming_with_user_model(npc_id)
        else:
            adjustments = await behavior_evolution.evaluate_npc_scheming(npc_id)
        
        # Apply adjustments
        if "error" not in adjustments:
            await behavior_evolution.apply_scheming_adjustments(npc_id, adjustments)
            
            # Update status if scheming
            if adjustments.get("scheme_level", 0) >= 5:
                self._npc_status[npc_id] = NPCStatus.SCHEMING
        
        # Invalidate cache
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        
        return adjustments
    
    async def evaluate_all_npc_scheming(self) -> Dict[int, Dict[str, Any]]:
        """Evaluate scheming for all active NPCs."""
        behavior_evolution = await self._get_behavior_evolution()
        return await behavior_evolution.evaluate_npc_scheming_for_all(list(self._active_npcs))
    
    async def generate_scheming_opportunity(
        self,
        npc_id: int,
        trigger_event: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate a scheming opportunity for an NPC based on an event."""
        behavior_evolution = await self._get_behavior_evolution()
        return await behavior_evolution.generate_scheming_opportunity(npc_id, trigger_event)
    
    # ==================== BELIEF MANAGEMENT (CANON-ENHANCED) ====================
    
    async def form_npc_belief(
        self,
        npc_id: int,
        observation: str,
        factuality: float = 1.0,
        use_nyx_governance: bool = True
    ) -> Dict[str, Any]:
        """Form a subjective belief for an NPC with canon consistency checking."""
        
        # Check belief consistency with canon if enabled
        if self.enable_canon and self.check_canon_consistency:
            async with get_db_connection_context() as conn:
                existing_beliefs = await conn.fetch("""
                    SELECT belief_text, confidence FROM NPCBeliefs
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                    AND confidence > 0.5
                    ORDER BY confidence DESC
                    LIMIT 10
                """, npc_id, self.user_id, self.conversation_id)
                
                if existing_beliefs:
                    validation_agent = await self._get_validation_agent()
                    belief_texts = [b["belief_text"] for b in existing_beliefs]
                    
                    prompt = f"""
                    Existing beliefs: {json.dumps(belief_texts, indent=2)}
                    New belief: "{observation}"
                    
                    Does this contradict existing beliefs?
                    Reply with JSON: {{"contradicts": true/false, "conflicting_beliefs": []}}
                    """
                    
                    result = await Runner.run(validation_agent.agent, prompt)
                    try:
                        check = json.loads(result.final_output)
                        if check["contradicts"]:
                            logger.warning(f"Belief contradicts existing: {observation}")
                            # Could reconcile or adjust factuality
                            factuality *= 0.7  # Reduce confidence due to contradiction
                    except:
                        pass
        
        # Form belief through appropriate system
        if use_nyx_governance:
            memory_access = await self._get_nyx_memory_access(npc_id)
            result = await memory_access.create_belief(
                belief_text=observation,
                confidence=factuality
            )
        else:
            belief_formation = await self._get_belief_formation(npc_id)
            ctx = RunContextWrapper(context={
                'user_id': self.user_id,
                'conversation_id': self.conversation_id,
                'npc_id': npc_id
            })
            result = await belief_formation.form_subjective_belief_from_observation(
                ctx, observation, factuality
            )
        
        # Canonize significant beliefs
        if self.enable_canon and self.auto_canonize and factuality >= 0.8:
            ctx = self._get_canonical_context()
            async with get_db_connection_context() as conn:
                await canon.log_canonical_event(
                    ctx, conn,
                    f"NPC {npc_id} formed strong belief: {observation[:100]}",
                    tags=["npc", "belief", "canonical"],
                    significance=5
                )
        
        # Invalidate cache
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        
        return result
    
    async def form_narrative_belief(
        self,
        npc_id: int,
        related_memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Form a narrative belief connecting multiple memories."""
        belief_formation = await self._get_belief_formation(npc_id)
        
        ctx = RunContextWrapper(context={
            'user_id': self.user_id,
            'conversation_id': self.conversation_id,
            'npc_id': npc_id
        })
        
        result = await belief_formation.form_narrative_from_memories(ctx, related_memories)
        
        # Invalidate cache
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        
        return result
    
    async def form_cultural_belief(
        self,
        npc_id: int,
        subject: str
    ) -> Dict[str, Any]:
        """Form a culturally-influenced belief."""
        belief_formation = await self._get_belief_formation(npc_id)
        
        ctx = RunContextWrapper(context={
            'user_id': self.user_id,
            'conversation_id': self.conversation_id,
            'npc_id': npc_id
        })
        
        result = await belief_formation.form_culturally_influenced_belief(ctx, subject)
        
        # Invalidate cache
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        
        return result
    
    async def get_npc_beliefs(
        self,
        npc_id: int,
        topic: Optional[str] = None,
        use_nyx_governance: bool = True
    ) -> List[Dict[str, Any]]:
        """Get beliefs for an NPC."""
        if use_nyx_governance:
            memory_access = await self._get_nyx_memory_access(npc_id)
            return await memory_access.get_beliefs(topic=topic)
        else:
            memory_manager = await self._get_memory_manager(npc_id)
            return await memory_manager.get_beliefs(topic=topic)
    
    async def get_npc_femdom_beliefs(
        self,
        npc_id: int,
        min_confidence: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Get femdom-related beliefs for an NPC."""
        memory_manager = await self._get_memory_manager(npc_id)
        return await memory_manager.get_femdom_beliefs(min_confidence=min_confidence)
    
    async def process_event_for_beliefs(
        self,
        event_text: str,
        event_type: str,
        npc_ids: List[int],
        factuality: float = 1.0
    ) -> Dict[str, Any]:
        """Process an event to generate beliefs for multiple NPCs."""
        belief_integration = await self._get_belief_integration()
        
        ctx = RunContextWrapper(context={
            'user_id': self.user_id,
            'conversation_id': self.conversation_id
        })
        
        result = await belief_integration.process_event_for_beliefs(
            ctx, event_text, event_type, npc_ids, factuality
        )
        
        # Invalidate caches
        for npc_id in npc_ids:
            if npc_id in self._snapshot_cache:
                del self._snapshot_cache[npc_id]
        
        return result
    
    # ==================== LORE CONTEXT ====================
    
    async def get_lore_context_for_npc(
        self,
        npc_id: int,
        context_type: str = "general"
    ) -> Dict[str, Any]:
        """Get lore context for an NPC."""
        lore_manager = await self._get_lore_context_manager()
        return await lore_manager.get_lore_context(npc_id, context_type)
    
    async def analyze_lore_impact(
        self,
        lore_change: Dict[str, Any],
        affected_npcs: List[int]
    ) -> Dict[str, Any]:
        """Analyze how a lore change affects NPCs."""
        analyzer = await self._get_lore_impact_analyzer()
        return await analyzer.analyze_lore_impact(lore_change, affected_npcs)
    
    async def propagate_lore_change(
        self,
        lore_change: Dict[str, Any],
        source_npc_id: int,
        target_npcs: List[int]
    ) -> Dict[str, Any]:
        """Propagate a lore change through NPC network."""
        propagation = await self._get_lore_propagation_system()
        return await propagation.propagate_lore_change(lore_change, source_npc_id, target_npcs)
    
    async def handle_lore_change(
        self,
        lore_change: Dict[str, Any],
        source_npc_id: int,
        affected_npcs: List[int]
    ) -> Dict[str, Any]:
        """Handle a lore change with full impact analysis and propagation."""
        lore_manager = await self._get_lore_context_manager()
        result = await lore_manager.handle_lore_change(lore_change, source_npc_id, affected_npcs)
        
        # Invalidate caches for affected NPCs
        for npc_id in affected_npcs:
            if npc_id in self._snapshot_cache:
                del self._snapshot_cache[npc_id]
        
        return result
    
    # ==================== LEARNING & ADAPTATION ====================
    
    async def record_npc_learning_interaction(
        self,
        npc_id: int,
        interaction_type: str,
        interaction_details: Dict[str, Any],
        player_response: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Record an interaction for NPC learning."""
        learning_adaptation = await self._get_learning_adaptation(npc_id)
        
        ctx = RunContextWrapper(context={
            'user_id': self.user_id,
            'conversation_id': self.conversation_id,
            'npc_id': npc_id
        })
        
        result = await learning_adaptation.record_player_interaction(
            ctx=ctx,
            interaction_type=interaction_type,
            interaction_details=interaction_details,
            player_response=player_response
        )
        
        # Invalidate cache
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        
        return result
    
    async def respond_to_learning_trigger(
        self,
        npc_id: int,
        trigger_type: str,
        trigger_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Respond to a specific learning trigger for an NPC."""
        learning_adaptation = await self._get_learning_adaptation(npc_id)
        
        ctx = RunContextWrapper(context={
            'user_id': self.user_id,
            'conversation_id': self.conversation_id,
            'npc_id': npc_id
        })
        
        result = await learning_adaptation.respond_to_trigger(
            ctx=ctx,
            trigger_type=trigger_type,
            trigger_details=trigger_details
        )
        
        # Invalidate cache
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        
        return result
    
    async def process_npc_learning_cycle(
        self,
        npc_id: int,
        days: int = 7
    ) -> Dict[str, Any]:
        """Process a learning cycle for an NPC."""
        learning_adaptation = await self._get_learning_adaptation(npc_id)
        
        ctx = RunContextWrapper(context={
            'user_id': self.user_id,
            'conversation_id': self.conversation_id,
            'npc_id': npc_id
        })
        
        # Process recent memories
        memory_result = await learning_adaptation.process_recent_memories_for_learning(ctx, days)
        
        # Adapt to relationship changes
        relationship_result = await learning_adaptation.adapt_to_relationship_changes(ctx)
        
        # Invalidate cache
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        
        return {
            "npc_id": npc_id,
            "memory_learning": memory_result,
            "relationship_adaptation": relationship_result
        }
    
    async def process_all_npc_learning(self) -> Dict[int, Dict[str, Any]]:
        """Process learning cycles for all active NPCs."""
        results = {}
        for npc_id in self._active_npcs:
            try:
                results[npc_id] = await self.process_npc_learning_cycle(npc_id)
            except Exception as e:
                logger.error(f"Error processing learning for NPC {npc_id}: {e}")
                results[npc_id] = {"error": str(e)}
        return results
    
    async def process_event_for_learning(
        self,
        event_text: str,
        event_type: str,
        npc_ids: List[int],
        player_response: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process an event for learning by multiple NPCs."""
        learning_manager = await self._get_learning_manager()
        result = await learning_manager.process_event_for_learning(
            event_text=event_text,
            event_type=event_type,
            npc_ids=npc_ids,
            player_response=player_response
        )
        
        # Invalidate caches
        for npc_id in npc_ids:
            if npc_id in self._snapshot_cache:
                del self._snapshot_cache[npc_id]
        
        return result
    
    async def run_learning_adaptation_cycle(
        self,
        npc_ids: List[int]
    ) -> Dict[str, Any]:
        """Run regular adaptation cycle for multiple NPCs."""
        learning_manager = await self._get_learning_manager()
        return await learning_manager.run_regular_adaptation_cycle(npc_ids)
    
    # ==================== RELATIONSHIP MANAGEMENT ====================
    
    async def update_npc_relationship(
        self,
        npc_id: int,
        entity_type: str,
        entity_id: int,
        player_action: Dict[str, Any],
        npc_action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update a relationship for an NPC."""
        relationship_manager = await self._get_relationship_manager(npc_id)
        result = await relationship_manager.update_relationship_from_interaction(
            entity_type=entity_type,
            entity_id=entity_id,
            player_action=player_action,
            npc_action=npc_action,
            context=context
        )
        
        # Invalidate cache
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        
        return result
    
    async def apply_relationship_decay_for_npc(
        self,
        npc_id: int,
        days_since_interaction: int = 1
    ) -> Dict[str, Any]:
        """Apply relationship decay for an NPC."""
        relationship_manager = await self._get_relationship_manager(npc_id)
        result = await relationship_manager.apply_relationship_decay(days_since_interaction)
        
        # Invalidate cache
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        
        return result
    
    async def update_npc_narrative_progression(
        self,
        npc_id: int,
        interaction_type: str,
        interaction_intensity: float = 1.0
    ) -> Dict[str, Any]:
        """Update narrative progression for an NPC."""
        relationship_manager = await self._get_relationship_manager(npc_id)
        result = await relationship_manager.update_narrative_progression(
            entity_id=self.user_id,
            interaction_type=interaction_type,
            interaction_intensity=interaction_intensity
        )
        
        # Invalidate cache
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        
        return result
    
    async def get_npc_relationship_history(
        self,
        npc_id: int,
        entity_type: str,
        entity_id: int,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get relationship history for an NPC."""
        relationship_manager = await self._get_relationship_manager(npc_id)
        return await relationship_manager.get_relationship_history(
            entity_type=entity_type,
            entity_id=entity_id,
            limit=limit
        )
    
    async def get_npc_relationship_memories(
        self,
        npc_id: int,
        entity_type: str,
        entity_id: int,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get relationship-related memories for an NPC."""
        relationship_manager = await self._get_relationship_manager(npc_id)
        return await relationship_manager.get_relationship_memories(
            entity_type=entity_type,
            entity_id=entity_id,
            limit=limit
        )
    
    async def evaluate_npc_coalitions_and_rivalries(
        self,
        npc_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate coalitions and rivalries for an NPC."""
        relationship_manager = await self._get_relationship_manager(npc_id)
        result = await relationship_manager.evaluate_coalitions_and_rivalries(context)
        
        # Invalidate cache
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        
        return result
    
    # ==================== NPC INFORMATION (WITH CANON) ====================
    
    async def get_npc_snapshot(
        self,
        npc_id: int,
        force_refresh: bool = False
    ) -> NPCSnapshot:
        """
        Get a comprehensive snapshot of an NPC's current state.
        This is the primary method for narrative generators to get NPC context.
        Now includes canonical events when canon is enabled.
        """
        # Check cache first
        if not force_refresh and npc_id in self._snapshot_cache:
            snapshot, timestamp = self._snapshot_cache[npc_id]
            if datetime.now() - timestamp < self._snapshot_ttl:
                return snapshot
        
        # Build fresh snapshot
        snapshot = await self._build_npc_snapshot(npc_id)
        
        # Cache it
        self._snapshot_cache[npc_id] = (snapshot, datetime.now())
        
        return snapshot
    
    async def _build_npc_snapshot(self, npc_id: int) -> NPCSnapshot:
        """Build a comprehensive snapshot of an NPC including canonical events."""
        # Get basic NPC data
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT npc_name, current_location, dominance, cruelty, 
                       closeness, trust, respect, intensity, mask_integrity,
                       personality_traits, schedule, scheming_level, betrayal_planning,
                       special_mechanics
                FROM NPCStats
                WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
            """, npc_id, self.user_id, self.conversation_id)
            
            if not row:
                raise ValueError(f"NPC {npc_id} not found")
            
            # Get canonical events if canon enabled
            canonical_events = []
            if self.enable_canon:
                events = await conn.fetch("""
                    SELECT event_text, tags, significance, timestamp
                    FROM CanonicalEvents
                    WHERE user_id = $1 AND conversation_id = $2
                    AND event_text LIKE '%' || $3 || '%'
                    ORDER BY timestamp DESC
                    LIMIT 5
                """, self.user_id, self.conversation_id, row["npc_name"])
                
                canonical_events = [dict(e) for e in events]
        
        # Get emotional state
        emotional_state = await self._memory_system.get_npc_emotion(npc_id)
        
        # Get recent memories
        memory_result = await self._memory_system.recall(
            entity_type="npc",
            entity_id=npc_id,
            query="",
            limit=5
        )
        recent_memories = memory_result.get("memories", [])
        
        # Get active relationships
        relationships = await self._get_npc_relationships(npc_id)
        
        # Get current beliefs
        beliefs = await self._memory_system.get_beliefs(
            entity_type="npc",
            entity_id=npc_id,
            topic="player"
        )
        
        # Get narrative stage with player
        narrative_stage = await get_npc_narrative_stage(
            self.user_id, self.conversation_id, npc_id
        )
        
        # Determine current activity
        current_activity = await self._determine_npc_activity(npc_id, row['schedule'])
        
        # Get decision history if available
        decision_history = []
        if npc_id in self._decision_engines:
            engine = self._decision_engines[npc_id]
            decision_history = engine.context.decision_log[-5:] if engine.context.decision_log else []
        
        # Get current goals if available
        current_goals = []
        if npc_id in self._decision_engines:
            engine = self._decision_engines[npc_id]
            current_goals = engine.context.long_term_goals
        
        # Calculate paranoia level from behavior evolution
        behavior_evolution = await self._get_behavior_evolution()
        npc_data = await behavior_evolution._get_npc_data(npc_id)
        paranoia_level = 5 if npc_data and "paranoid" in npc_data.get("personality_traits", []) else 2
        
        # Parse special mechanics if present
        special_mechanics = {}
        if row['special_mechanics']:
            try:
                special_mechanics = json.loads(row['special_mechanics'])
            except:
                special_mechanics = {}
        
        return NPCSnapshot(
            npc_id=npc_id,
            name=row['npc_name'],
            location=row['current_location'] or "unknown",
            status=self._npc_status.get(npc_id, NPCStatus.IDLE),
            current_activity=current_activity,
            emotional_state=emotional_state or {},
            recent_memories=recent_memories,
            active_relationships=relationships,
            current_beliefs=beliefs or [],
            narrative_stage=narrative_stage.name if narrative_stage else None,
            mask_integrity=row['mask_integrity'] or 100.0,
            stats={
                "dominance": row['dominance'],
                "cruelty": row['cruelty'],
                "closeness": row['closeness'],
                "trust": row['trust'],
                "respect": row['respect'],
                "intensity": row['intensity']
            },
            schedule=row['schedule'],
            scheming_level=row['scheming_level'] or 0,
            paranoia_level=paranoia_level,
            current_goals=current_goals,
            decision_history=decision_history,
            nyx_directives=[],  # Would need implementation
            special_mechanics=special_mechanics,
            canonical_events=canonical_events  # Added canon events
        )
    
    # ==================== CANON-SPECIFIC METHODS ====================
    
    async def get_npc_canonical_context(
        self,
        npc_id: int,
        include_relationships: bool = True,
        include_memories: bool = True,
        include_beliefs: bool = True,
        include_events: bool = True
    ) -> Dict[str, Any]:
        """Get comprehensive canonical context for an NPC."""
        if not self.enable_canon:
            return {"canon_disabled": True}
        
        context = {
            "npc_id": npc_id,
            "canonical_data": {},
            "relationships": [],
            "memories": [],
            "beliefs": [],
            "events": [],
            "locations_visited": [],
            "items_owned": []
        }
        
        async with get_db_connection_context() as conn:
            # Get core NPC data
            npc_data = await conn.fetchrow("""
                SELECT * FROM NPCStats
                WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
            """, npc_id, self.user_id, self.conversation_id)
            
            if not npc_data:
                return context
            
            context["canonical_data"] = dict(npc_data)
            
            # Get relationships
            if include_relationships:
                relationships = await conn.fetch("""
                    SELECT * FROM SocialLinks
                    WHERE entity1_type = 'npc' AND entity1_id = $1
                    AND user_id = $2 AND conversation_id = $3
                    ORDER BY link_level DESC
                """, npc_id, self.user_id, self.conversation_id)
                
                context["relationships"] = [dict(r) for r in relationships]
            
            # Get canonical memories
            if include_memories:
                memories = await conn.fetch("""
                    SELECT memory_text, memory_type, significance, tags, created_at
                    FROM NPCMemories
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                    ORDER BY significance DESC, created_at DESC
                    LIMIT 20
                """, npc_id, self.user_id, self.conversation_id)
                
                context["memories"] = [dict(m) for m in memories]
            
            # Get beliefs
            if include_beliefs:
                beliefs = await conn.fetch("""
                    SELECT belief_text, confidence, formed_at, last_reinforced
                    FROM NPCBeliefs
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                    AND confidence > 0.3
                    ORDER BY confidence DESC
                    LIMIT 15
                """, npc_id, self.user_id, self.conversation_id)
                
                context["beliefs"] = [dict(b) for b in beliefs]
            
            # Get canonical events
            if include_events:
                events = await conn.fetch("""
                    SELECT ce.event_text, ce.tags, ce.significance, ce.timestamp
                    FROM CanonicalEvents ce
                    WHERE ce.user_id = $1 AND ce.conversation_id = $2
                    AND ce.event_text LIKE '%' || $3 || '%'
                    ORDER BY ce.timestamp DESC
                    LIMIT 10
                """, self.user_id, self.conversation_id, npc_data["npc_name"])
                
                context["events"] = [dict(e) for e in events]
        
        return context
    
    async def update_npc_location(
        self,
        npc_id: int,
        new_location: str,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update NPC location with automatic canonization."""
        ctx = self._get_canonical_context()
        
        async with get_db_connection_context() as conn:
            # Ensure location exists in canon if canon enabled
            if self.enable_canon:
                location_id = await canon.find_or_create_location(
                    ctx, conn,
                    location_name=new_location
                )
            
            # Get NPC name and old location
            npc_data = await conn.fetchrow("""
                SELECT npc_name, current_location FROM NPCStats
                WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
            """, npc_id, self.user_id, self.conversation_id)
            
            if not npc_data:
                return {"success": False, "error": "NPC not found"}
            
            old_location = npc_data["current_location"]
            
            # Update location
            await conn.execute("""
                UPDATE NPCStats
                SET current_location = $1
                WHERE npc_id = $2 AND user_id = $3 AND conversation_id = $4
            """, new_location, npc_id, self.user_id, self.conversation_id)
            
            # Canonize if enabled
            if self.enable_canon and self.auto_canonize:
                event_text = f"{npc_data['npc_name']} moved from {old_location} to {new_location}"
                if reason:
                    event_text += f" ({reason})"
                
                await canon.log_canonical_event(
                    ctx, conn,
                    event_text,
                    tags=["npc", "movement", "location"],
                    significance=4
                )
        
        # Invalidate cache
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        
        return {
            "success": True,
            "npc_id": npc_id,
            "old_location": old_location,
            "new_location": new_location
        }
    
    async def set_canon_config(
        self,
        enable_canon: Optional[bool] = None,
        auto_canonize: Optional[bool] = None,
        check_consistency: Optional[bool] = None,
        significance_threshold: Optional[int] = None
    ) -> Dict[str, Any]:
        """Configure canon integration settings."""
        old_config = {
            "enable_canon": self.enable_canon,
            "auto_canonize": self.auto_canonize,
            "check_canon_consistency": self.check_canon_consistency,
            "canon_significance_threshold": self.canon_significance_threshold
        }
        
        if enable_canon is not None:
            self.enable_canon = enable_canon
        if auto_canonize is not None:
            self.auto_canonize = auto_canonize
        if check_consistency is not None:
            self.check_canon_consistency = check_consistency
        if significance_threshold is not None:
            self.canon_significance_threshold = significance_threshold
        
        new_config = {
            "enable_canon": self.enable_canon,
            "auto_canonize": self.auto_canonize,
            "check_canon_consistency": self.check_canon_consistency,
            "canon_significance_threshold": self.canon_significance_threshold
        }
        
        logger.info(f"Canon configuration updated: {new_config}")
        
        return {
            "old_config": old_config,
            "new_config": new_config
        }
    
    # ==================== TEMPLATE GENERATION ====================
    
    async def generate_mask_slippage_triggers(
        self,
        stat: str,
        environment_desc: str = ""
    ) -> List[Dict[str, Any]]:
        """Generate mask slippage triggers for an NPC stat."""
        return await get_mask_slippage_triggers(stat, environment_desc)
    
    async def generate_relationship_stages(
        self,
        scenario: str,
        environment_desc: str = ""
    ) -> List[Dict[str, Any]]:
        """Generate relationship progression stages."""
        return await get_relationship_stages(scenario, environment_desc)
    
    async def generate_relationship_memory(
        self,
        npc_name: str,
        target_name: str,
        relationship: str,
        location: str,
        environment_desc: str = ""
    ) -> Optional[str]:
        """Generate a relationship memory."""
        return await generate_relationship_memory(
            npc_name, target_name, relationship, location, environment_desc
        )
    
    async def get_calendar_data(
        self,
        environment_desc: str = ""
    ) -> Dict[str, Any]:
        """Get calendar month and day names for the environment."""
        months = await get_calendar_months(environment_desc)
        days = await get_calendar_day_names(environment_desc)
        return {"months": months, "days": days}
    
    async def get_trauma_keywords_for_environment(
        self,
        environment_desc: str = ""
    ) -> List[str]:
        """Get trauma keywords relevant to the environment."""
        return await get_trauma_keywords(environment_desc)
    
    # ==================== GROUP OPERATIONS ====================
    
    async def coordinate_group_decision(
        self,
        npc_ids: List[int],
        shared_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Coordinate decision-making for a group of NPCs."""
        coordinator = await self._get_coordinator()
        return await coordinator.make_group_decisions(npc_ids, shared_context)
    
    async def coordinate_group_with_nyx(
        self,
        npc_ids: List[int],
        shared_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Coordinate group decisions with Nyx governance."""
        coordinator = await self._get_coordinator()
        return await coordinator.make_group_decisions_with_nyx(npc_ids, shared_context)
    
    async def batch_update_npcs(
        self,
        npc_ids: List[int],
        update_type: str,
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Batch update multiple NPCs."""
        coordinator = await self._get_coordinator()
        result = await coordinator.batch_update_npcs(npc_ids, update_type, update_data)
        
        # Invalidate caches
        for npc_id in npc_ids:
            if npc_id in self._snapshot_cache:
                del self._snapshot_cache[npc_id]
        
        return result
    
    # ==================== EXISTING METHODS (from original) ====================
    
    async def get_all_npc_snapshots(self) -> Dict[int, NPCSnapshot]:
        """Get snapshots for all active NPCs."""
        snapshots = {}
        for npc_id in self._active_npcs:
            try:
                snapshots[npc_id] = await self.get_npc_snapshot(npc_id)
            except Exception as e:
                logger.error(f"Error getting snapshot for NPC {npc_id}: {e}")
        return snapshots
    
    async def get_npcs_at_location(self, location: str) -> List[NPCSnapshot]:
        """Get all NPCs at a specific location."""
        npcs = []
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT npc_id FROM NPCStats
                WHERE current_location = $1 
                AND user_id = $2 AND conversation_id = $3
                AND introduced = TRUE
            """, location, self.user_id, self.conversation_id)
            
            for row in rows:
                try:
                    snapshot = await self.get_npc_snapshot(row['npc_id'])
                    npcs.append(snapshot)
                except Exception as e:
                    logger.error(f"Error getting snapshot for NPC {row['npc_id']}: {e}")
        
        return npcs
    
    async def process_player_interaction(
        self,
        npc_id: int,
        player_input: str,
        interaction_type: str = "standard",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a player interaction with an NPC, with canon consistency checking."""
        self._npc_status[npc_id] = NPCStatus.IN_CONVERSATION
        
        try:
            # Canon consistency check if enabled
            if self.enable_canon and self.check_canon_consistency:
                action = {
                    "type": interaction_type,
                    "player_input": player_input,
                    "location": context.get("location") if context else None
                }
                
                consistency_check = await self._check_npc_canon_consistency(
                    npc_id, action, context
                )
                
                if not consistency_check.is_consistent and consistency_check.conflicts:
                    # Add warnings to context
                    if context is None:
                        context = {}
                    context["canon_warnings"] = consistency_check.warnings
                    context["canon_suggestions"] = consistency_check.suggestions
                    
                    # For critical conflicts, we might want to block
                    critical_conflicts = [c for c in consistency_check.conflicts 
                                         if c.get("severity") == "critical"]
                    if critical_conflicts:
                        return {
                            "success": False,
                            "reason": "Canon consistency violation",
                            "conflicts": critical_conflicts
                        }
            
            # Process interaction through handler
            handler = await self._get_handler()
            result = await handler.handle_interaction(
                npc_id=npc_id,
                interaction_type=interaction_type,
                player_input=player_input,
                context=context
            )
            
            # Canonize significant results
            if self.enable_canon and self.auto_canonize and result.get("significance", 0) >= self.canon_significance_threshold:
                canon_result = await self._canonize_npc_action(
                    npc_id=npc_id,
                    action={"type": interaction_type, "input": player_input},
                    result=result,
                    significance=result.get("significance", 5)
                )
                result["canonization"] = canon_result
            
            # Invalidate snapshot cache for this NPC
            if npc_id in self._snapshot_cache:
                del self._snapshot_cache[npc_id]
            
            return result
        finally:
            self._npc_status[npc_id] = NPCStatus.IDLE
    
    async def process_group_interaction(
        self,
        npc_ids: List[int],
        player_action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a player interaction with multiple NPCs."""
        for npc_id in npc_ids:
            self._npc_status[npc_id] = NPCStatus.IN_CONVERSATION
        
        try:
            agent_system = await self._get_agent_system()
            result = await agent_system.handle_group_npc_interaction(
                npc_ids, player_action, context or {}
            )
            
            # Invalidate snapshot caches
            for npc_id in npc_ids:
                if npc_id in self._snapshot_cache:
                    del self._snapshot_cache[npc_id]
            
            return result
        finally:
            for npc_id in npc_ids:
                self._npc_status[npc_id] = NPCStatus.IDLE
    
    async def generate_npc_to_npc_interaction(
        self,
        npc1_id: int,
        npc2_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate an interaction between two NPCs."""
        handler = await self._get_handler()
        result = await handler.generate_npc_npc_interaction(
            npc1_id, npc2_id, context
        )
        
        # Invalidate caches
        for npc_id in [npc1_id, npc2_id]:
            if npc_id in self._snapshot_cache:
                del self._snapshot_cache[npc_id]
        
        return result
    
    async def process_scheduled_activities(self) -> Dict[str, Any]:
        """Process scheduled activities for all NPCs."""
        agent_system = await self._get_agent_system()
        result = await agent_system.process_npc_scheduled_activities()
        
        # Invalidate all snapshot caches
        self._snapshot_cache.clear()
        
        return result
    
    async def process_daily_activities(self) -> Dict[str, Any]:
        """Process daily activities for all NPCs."""
        handler = await self._get_handler()
        result = await handler.process_daily_npc_activities()
        
        # Update locations and invalidate caches
        self._snapshot_cache.clear()
        
        return result
    
    async def _determine_npc_activity(
        self,
        npc_id: int,
        schedule: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Determine what an NPC is currently doing based on schedule and time."""
        if not schedule:
            return None
        
        # Get current game time
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay
                WHERE key = 'TimeOfDay' 
                AND user_id = $1 AND conversation_id = $2
            """, self.user_id, self.conversation_id)
            
            if not row:
                return None
            
            time_of_day = row['value']
            
            # Check schedule for current time
            for day, day_schedule in schedule.items():
                if isinstance(day_schedule, dict) and time_of_day in day_schedule:
                    return day_schedule[time_of_day]
        
        return None
    
    async def create_memory_for_npc(
        self,
        npc_id: int,
        memory_text: str,
        importance: str = "medium",
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a memory for an NPC."""
        result = await self._memory_system.remember(
            entity_type="npc",
            entity_id=npc_id,
            memory_text=memory_text,
            importance=importance,
            tags=tags or []
        )
        
        # Invalidate cache
        if npc_id in self._snapshot_cache:
            del self._snapshot_cache[npc_id]
        
        return result
    
    async def _get_npc_relationships(self, npc_id: int) -> Dict[str, Any]:
        """Get active relationships for an NPC."""
        relationships = {}
        
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT entity2_type, entity2_id, link_type, link_level
                FROM SocialLinks
                WHERE entity1_type = 'npc' AND entity1_id = $1
                AND user_id = $2 AND conversation_id = $3
                ORDER BY link_level DESC
                LIMIT 5
            """, npc_id, self.user_id, self.conversation_id)
            
            for row in rows:
                key = f"{row['entity2_type']}_{row['entity2_id']}"
                relationships[key] = {
                    "type": row['link_type'],
                    "level": row['link_level']
                }
        
        return relationships
    
    async def update_relationship(
        self,
        npc1_id: int,
        entity2_type: str,
        entity2_id: int,
        change_amount: int
    ) -> Dict[str, Any]:
        """Update a relationship between entities."""
        manager = NPCRelationshipManager(npc1_id, self.user_id, self.conversation_id)
        
        # Create interaction context
        player_action = {"type": "relationship_change"}
        npc_action = {"type": "response"}
        context = {"change_amount": change_amount}
        
        result = await manager.update_relationship_from_interaction(
            entity2_type, entity2_id, player_action, npc_action, context
        )
        
        # Invalidate cache
        if npc1_id in self._snapshot_cache:
            del self._snapshot_cache[npc1_id]
        
        return result
    
    async def get_narrative_context(
        self,
        focus_npc_ids: Optional[List[int]] = None,
        location: Optional[str] = None,
        include_relationships: bool = True,
        include_beliefs: bool = True,
        include_decision_patterns: bool = True,
        include_learning_data: bool = True,
        include_nyx_governance: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive narrative context for the narrative generator.
        Enhanced with all subsystem data and canon integration.
        """
        context = {
            "timestamp": datetime.now().isoformat(),
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "canon_enabled": self.enable_canon,
            "npcs": {},
            "locations": {},
            "active_relationships": [],
            "group_dynamics": {},
            "narrative_threads": [],
            "belief_networks": {},
            "lore_context": {},
            "behavior_patterns": {},
            "learning_insights": {},
            "nyx_governance": {},
            "canonical_events": []
        }
        
        # Add recent canonical events if canon enabled
        if self.enable_canon:
            async with get_db_connection_context() as conn:
                events = await conn.fetch("""
                    SELECT event_text, tags, significance, timestamp
                    FROM CanonicalEvents
                    WHERE user_id = $1 AND conversation_id = $2
                    ORDER BY timestamp DESC
                    LIMIT 10
                """, self.user_id, self.conversation_id)
                
                context["canonical_events"] = [dict(e) for e in events]
        
        # Determine which NPCs to include
        if focus_npc_ids:
            npc_ids = focus_npc_ids
        elif location:
            location_npcs = await self.get_npcs_at_location(location)
            npc_ids = [npc.npc_id for npc in location_npcs]
        else:
            npc_ids = list(self._active_npcs)
        
        # Get snapshots for each NPC
        for npc_id in npc_ids:
            try:
                snapshot = await self.get_npc_snapshot(npc_id)
                context["npcs"][npc_id] = snapshot.to_narrative_context()
                
                # Track locations
                if snapshot.location not in context["locations"]:
                    context["locations"][snapshot.location] = []
                context["locations"][snapshot.location].append(snapshot.name)
                
                # Add belief network
                if include_beliefs:
                    beliefs = await self._memory_system.get_beliefs(
                        entity_type="npc",
                        entity_id=npc_id
                    )
                    context["belief_networks"][npc_id] = beliefs
                
                # Add lore context
                lore_manager = await self._get_lore_context_manager()
                lore_context = await lore_manager.get_lore_context(npc_id, "narrative")
                context["lore_context"][npc_id] = lore_context
                
                # Add behavior patterns
                if include_decision_patterns and npc_id in self._decision_engines:
                    engine = self._decision_engines[npc_id]
                    context["behavior_patterns"][npc_id] = {
                        "decision_log": engine.context.decision_log[-10:],
                        "long_term_goals": engine.context.long_term_goals,
                        "behavior_evolution": engine.context.behavior_evolution_factors.__dict__
                    }
                
                # Add learning insights
                if include_learning_data and npc_id in self._learning_adaptations:
                    adaptation = self._learning_adaptations[npc_id]
                    context["learning_insights"][npc_id] = {
                        "pattern_count": len(adaptation.interaction_patterns) if hasattr(adaptation, 'interaction_patterns') else 0,
                        "preference_model": adaptation.preference_model if hasattr(adaptation, 'preference_model') else {}
                    }
                
                # Add Nyx governance data
                if include_nyx_governance and self._nyx_bridge:
                    # This would need to be implemented in the NyxNPCBridge
                    context["nyx_governance"][npc_id] = snapshot.nyx_directives or []
                    
            except Exception as e:
                logger.error(f"Error getting narrative context for NPC {npc_id}: {e}")
        
        # Add relationship network if requested
        if include_relationships:
            context["active_relationships"] = await self._build_relationship_network(npc_ids)
        
        # Add group dynamics
        if len(npc_ids) > 1:
            context["group_dynamics"] = await self._analyze_group_dynamics(npc_ids)
        
        # Add narrative threads
        context["narrative_threads"] = await self._identify_narrative_threads(npc_ids)
        
        return context
    
    async def _build_relationship_network(self, npc_ids: List[int]) -> List[Dict[str, Any]]:
        """Build a network of relationships between NPCs."""
        relationships = []
        
        async with get_db_connection_context() as conn:
            for npc_id in npc_ids:
                rows = await conn.fetch("""
                    SELECT entity2_type, entity2_id, link_type, link_level
                    FROM SocialLinks
                    WHERE entity1_type = 'npc' AND entity1_id = $1
                    AND user_id = $2 AND conversation_id = $3
                    AND link_level != 50
                """, npc_id, self.user_id, self.conversation_id)
                
                for row in rows:
                    relationships.append({
                        "from": npc_id,
                        "to": f"{row['entity2_type']}_{row['entity2_id']}",
                        "type": row['link_type'],
                        "strength": row['link_level']
                    })
        
        return relationships
    
    async def _analyze_group_dynamics(self, npc_ids: List[int]) -> Dict[str, Any]:
        """Analyze dynamics within a group of NPCs."""
        dynamics = {
            "average_dominance": 0,
            "tension_level": 0,
            "cohesion": 0,
            "leader_candidates": [],
            "potential_conflicts": [],
            "scheming_npcs": []
        }
        
        if not npc_ids:
            return dynamics
        
        total_dominance = 0
        dominance_scores = []
        scheming_scores = []
        
        for npc_id in npc_ids:
            snapshot = await self.get_npc_snapshot(npc_id)
            dominance = snapshot.stats["dominance"]
            total_dominance += dominance
            dominance_scores.append((npc_id, dominance, snapshot.name))
            
            # Track scheming NPCs
            if snapshot.scheming_level >= 5:
                scheming_scores.append({
                    "npc_id": npc_id,
                    "name": snapshot.name,
                    "scheming_level": snapshot.scheming_level,
                    "paranoia_level": snapshot.paranoia_level
                })
        
        dynamics["average_dominance"] = total_dominance / len(npc_ids)
        dynamics["scheming_npcs"] = scheming_scores
        
        # Identify leader candidates
        dominance_scores.sort(key=lambda x: x[1], reverse=True)
        for npc_id, dominance, name in dominance_scores[:2]:
            if dominance > 70:
                dynamics["leader_candidates"].append({
                    "npc_id": npc_id,
                    "name": name,
                    "dominance": dominance
                })
        
        # Check for potential conflicts
        if len(dominance_scores) > 1:
            if dominance_scores[0][1] > 70 and dominance_scores[1][1] > 70:
                dynamics["potential_conflicts"].append({
                    "type": "dominance_conflict",
                    "between": [dominance_scores[0][2], dominance_scores[1][2]]
                })
        
        # Check for scheming conflicts
        if len(scheming_scores) > 1:
            dynamics["potential_conflicts"].append({
                "type": "scheming_conflict",
                "description": "Multiple NPCs with high scheming levels may work against each other"
            })
        
        return dynamics
    
    async def _identify_narrative_threads(self, npc_ids: List[int]) -> List[Dict[str, Any]]:
        """Identify active narrative threads involving these NPCs."""
        threads = []
        
        # Check for player-NPC narrative progressions
        for npc_id in npc_ids:
            stage = await get_npc_narrative_stage(
                self.user_id, self.conversation_id, npc_id
            )
            if stage and stage.name != "Innocent Beginning":
                threads.append({
                    "type": "player_npc_progression",
                    "npc_id": npc_id,
                    "stage": stage.name,
                    "corruption": stage.corruption,
                    "dependency": stage.dependency,
                    "realization": stage.realization
                })
        
        # Check for active conflicts
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT c.conflict_id, c.conflict_type, cn.npc_id
                FROM Conflicts c
                JOIN ConflictNPCs cn ON c.conflict_id = cn.conflict_id
                WHERE c.user_id = $1 AND c.conversation_id = $2
                AND c.is_active = TRUE
                AND cn.npc_id = ANY($3)
            """, self.user_id, self.conversation_id, npc_ids)
            
            conflicts = {}
            for row in rows:
                if row['conflict_id'] not in conflicts:
                    conflicts[row['conflict_id']] = {
                        "type": "active_conflict",
                        "conflict_type": row['conflict_type'],
                        "involved_npcs": []
                    }
                conflicts[row['conflict_id']]["involved_npcs"].append(row['npc_id'])
            
            threads.extend(conflicts.values())
        
        return threads
    
    # ==================== UTILITY METHODS ====================
    
    async def refresh_all_caches(self) -> None:
        """Force refresh all caches."""
        self._snapshot_cache.clear()
        self._npc_cache.clear()
        if self._lore_context_cache:
            self._lore_context_cache.clear()
        logger.info("All NPC caches refreshed")
    
    async def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get status information about the orchestrator including canon status."""
        return {
            "active_npcs": len(self._active_npcs),
            "cached_snapshots": len(self._snapshot_cache),
            "npc_statuses": dict(self._npc_status),
            "canon_config": {
                "enabled": self.enable_canon,
                "auto_canonize": self.auto_canonize,
                "check_consistency": self.check_canon_consistency,
                "significance_threshold": self.canon_significance_threshold
            },
            "systems_initialized": {
                "agent_system": self._agent_system is not None,
                "coordinator": self._coordinator is not None,
                "handler": self._handler is not None,
                "creation_handler": self._creation_handler is not None,
                "preset_handler": self._preset_handler is not None,
                "memory_system": self._memory_system is not None,
                "lore_system": self._lore_system is not None,
                "belief_integration": self._belief_integration is not None,
                "lore_context_manager": self._lore_context_manager is not None,
                "behavior_evolution": self._behavior_evolution is not None,
                "learning_manager": self._learning_manager is not None,
                "nyx_bridge": self._nyx_bridge is not None
            },
            "subsystem_counts": {
                "belief_formations": len(self._belief_formations),
                "npc_behaviors": len(self._npc_behaviors),
                "decision_engines": len(self._decision_engines),
                "learning_adaptations": len(self._learning_adaptations),
                "memory_managers": len(self._memory_managers),
                "perception_systems": len(self._perception_systems),
                "relationship_managers": len(self._relationship_managers),
                "nyx_memory_access": len(self._nyx_memory_access)
            }
        }
    
    async def shutdown(self) -> None:
        """Clean shutdown of the orchestrator."""
        logger.info("Shutting down NPC Orchestrator")
        self._snapshot_cache.clear()
        self._npc_cache.clear()
        if self._lore_context_cache:
            self._lore_context_cache.clear()
        self._active_npcs.clear()
        self._npc_status.clear()
        self._belief_formations.clear()
        self._npc_behaviors.clear()
        self._decision_engines.clear()
        self._learning_adaptations.clear()
        self._memory_managers.clear()
        self._perception_systems.clear()
        self._relationship_managers.clear()
        self._nyx_memory_access.clear()


# ==================== CONVENIENCE FUNCTIONS ====================

async def create_orchestrator(
    user_id: int, 
    conversation_id: int,
    enable_canon: bool = True
) -> NPCOrchestrator:
    """
    Create and initialize an NPC orchestrator with canon integration.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        enable_canon: Whether to enable canon integration (default: True)
    """
    orchestrator = NPCOrchestrator(user_id, conversation_id, enable_canon=enable_canon)
    await orchestrator.initialize()
    return orchestrator


async def get_npc_context_for_narrative(
    user_id: int,
    conversation_id: int,
    focus_npc_ids: Optional[List[int]] = None,
    location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick function to get NPC context for narrative generation.
    Creates a temporary orchestrator if needed.
    """
    orchestrator = NPCOrchestrator(user_id, conversation_id, enable_canon=True)
    await orchestrator.initialize()
    
    try:
        return await orchestrator.get_narrative_context(
            focus_npc_ids=focus_npc_ids,
            location=location,
            include_relationships=True,
            include_beliefs=True,
            include_decision_patterns=True,
            include_learning_data=True,
            include_nyx_governance=True
        )
    finally:
        await orchestrator.shutdown()
