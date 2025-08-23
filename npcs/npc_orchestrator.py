# npcs/npc_orchestrator.py

"""
NPC System Orchestrator - Single Access Point for All NPC Operations

This module provides a unified interface to all NPC systems, making it easy
to integrate with narrative generators and other game systems.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
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
from npcs.belief_system_integration import NPCBeliefSystemIntegration
from npcs.lore_context_manager import LoreContextManager
from npcs.new_npc_creation import NPCCreationHandler
from npcs.npc_learning_adaptation import NPCLearningManager
from npcs.dynamic_templates import async_result_cache

# Import related systems
from memory.wrapper import MemorySystem
from lore.core.lore_system import LoreSystem
from logic.dynamic_relationships import OptimizedRelationshipManager
from logic.npc_narrative_progression import (
    get_npc_narrative_stage,
    progress_npc_narrative_stage,
    NPCNarrativeStage
)
from db.connection import get_db_connection_context
from agents import RunContextWrapper

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
                "recent_thoughts": [m.get("memory_text") for m in self.recent_memories[:3]]
            },
            "social": {
                "relationships": self.active_relationships,
                "narrative_stage": self.narrative_stage
            },
            "traits": self.stats
        }


class NPCOrchestrator:
    """
    Master orchestrator for all NPC operations.
    Single access point for narrative generation and game systems.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the orchestrator with all subsystems."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Core systems (lazy-loaded)
        self._agent_system: Optional[NPCAgentSystem] = None
        self._coordinator: Optional[NPCAgentCoordinator] = None
        self._handler: Optional[NPCHandler] = None
        self._creation_handler: Optional[NPCCreationHandler] = None
        self._memory_system: Optional[MemorySystem] = None
        self._lore_system: Optional[LoreSystem] = None
        self._relationship_manager: Optional[OptimizedRelationshipManager] = None
        self._belief_integration: Optional[NPCBeliefSystemIntegration] = None
        self._lore_context_manager: Optional[LoreContextManager] = None
        self._learning_manager: Optional[NPCLearningManager] = None
        
        # Caches
        self._npc_cache: Dict[int, NPCAgent] = {}
        self._snapshot_cache: Dict[int, Tuple[NPCSnapshot, datetime]] = {}
        self._snapshot_ttl = timedelta(minutes=5)
        
        # Tracking
        self._active_npcs: Set[int] = set()
        self._npc_status: Dict[int, NPCStatus] = {}
        
    # ==================== INITIALIZATION ====================
    
    async def initialize(self) -> None:
        """Initialize all core systems."""
        logger.info(f"Initializing NPC Orchestrator for user {self.user_id}, conversation {self.conversation_id}")
        
        # Initialize agent system (manages all NPCs)
        self._agent_system = NPCAgentSystem(self.user_id, self.conversation_id, None)
        await self._agent_system.initialize_agents()
        
        # Initialize other systems
        self._memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        self._lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
        
        # Load active NPCs
        await self._load_active_npcs()
        
    async def _load_active_npcs(self) -> None:
        """Load all active NPCs from the database."""
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT npc_id, npc_name, current_location
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2
                AND introduced = TRUE
            """, self.user_id, self.conversation_id)
            
            for row in rows:
                self._active_npcs.add(row['npc_id'])
                self._npc_status[row['npc_id']] = NPCStatus.IDLE
    
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
    
    async def _get_belief_integration(self) -> NPCBeliefSystemIntegration:
        """Get or create belief integration."""
        if self._belief_integration is None:
            self._belief_integration = NPCBeliefSystemIntegration(self.user_id, self.conversation_id)
            await self._belief_integration.initialize()
        return self._belief_integration
    
    async def _get_learning_manager(self) -> NPCLearningManager:
        """Get or create learning manager."""
        if self._learning_manager is None:
            self._learning_manager = NPCLearningManager(self.user_id, self.conversation_id)
        return self._learning_manager
    
    # ==================== NPC CREATION & MANAGEMENT ====================
    
    async def create_npc(
        self,
        environment_desc: Optional[str] = None,
        archetype_names: Optional[List[str]] = None,
        specific_traits: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new NPC."""
        handler = await self._get_creation_handler()
        result = await handler.create_npc(
            environment_desc=environment_desc,
            archetype_names=archetype_names,
            specific_traits=specific_traits,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        if "npc_id" in result:
            self._active_npcs.add(result["npc_id"])
            self._npc_status[result["npc_id"]] = NPCStatus.IDLE
            
        return result
    
    async def spawn_multiple_npcs(
        self,
        count: int = 3,
        environment_desc: Optional[str] = None
    ) -> List[int]:
        """Spawn multiple NPCs at once."""
        handler = await self._get_creation_handler()
        npc_ids = await handler.spawn_multiple_npcs(
            count=count,
            environment_desc=environment_desc,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        for npc_id in npc_ids:
            self._active_npcs.add(npc_id)
            self._npc_status[npc_id] = NPCStatus.IDLE
            
        return npc_ids
    
    # ==================== NPC INFORMATION ====================
    
    async def get_npc_snapshot(
        self,
        npc_id: int,
        force_refresh: bool = False
    ) -> NPCSnapshot:
        """
        Get a comprehensive snapshot of an NPC's current state.
        This is the primary method for narrative generators to get NPC context.
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
        """Build a comprehensive snapshot of an NPC."""
        # Get basic NPC data
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT npc_name, current_location, dominance, cruelty, 
                       closeness, trust, respect, intensity, mask_integrity,
                       personality_traits, schedule
                FROM NPCStats
                WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
            """, npc_id, self.user_id, self.conversation_id)
            
            if not row:
                raise ValueError(f"NPC {npc_id} not found")
        
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
            schedule=row['schedule']
        )
    
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
    
    # ==================== NPC INTERACTIONS ====================
    
    async def process_player_interaction(
        self,
        npc_id: int,
        player_input: str,
        interaction_type: str = "standard",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a player interaction with an NPC."""
        # Update NPC status
        self._npc_status[npc_id] = NPCStatus.IN_CONVERSATION
        
        try:
            handler = await self._get_handler()
            result = await handler.handle_interaction(
                npc_id=npc_id,
                interaction_type=interaction_type,
                player_input=player_input,
                context=context
            )
            
            # Invalidate snapshot cache for this NPC
            if npc_id in self._snapshot_cache:
                del self._snapshot_cache[npc_id]
            
            return result
        finally:
            # Reset status
            self._npc_status[npc_id] = NPCStatus.IDLE
    
    async def process_group_interaction(
        self,
        npc_ids: List[int],
        player_action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a player interaction with multiple NPCs."""
        # Update statuses
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
            # Reset statuses
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
    
    # ==================== NPC ACTIVITIES & SCHEDULING ====================
    
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
            # Schedule format: {"Monday": {"Morning": "work", "Afternoon": "market"}}
            for day, day_schedule in schedule.items():
                if isinstance(day_schedule, dict) and time_of_day in day_schedule:
                    return day_schedule[time_of_day]
        
        return None
    
    # ==================== MEMORY & BELIEFS ====================
    
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
    
    async def process_event_for_beliefs(
        self,
        event_text: str,
        event_type: str,
        npc_ids: List[int],
        factuality: float = 1.0
    ) -> Dict[str, Any]:
        """Process an event to generate beliefs for multiple NPCs."""
        belief_integration = await self._get_belief_integration()
        
        # Create context
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
    
    # ==================== RELATIONSHIPS ====================
    
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
    
    # ==================== NARRATIVE CONTEXT ====================
    
    async def get_narrative_context(
        self,
        focus_npc_ids: Optional[List[int]] = None,
        location: Optional[str] = None,
        include_relationships: bool = True,
        include_beliefs: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive narrative context for the narrative generator.
        This is the main method for getting NPC context for story generation.
        """
        context = {
            "timestamp": datetime.now().isoformat(),
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "npcs": {},
            "locations": {},
            "active_relationships": [],
            "group_dynamics": {},
            "narrative_threads": []
        }
        
        # Determine which NPCs to include
        if focus_npc_ids:
            npc_ids = focus_npc_ids
        elif location:
            # Get NPCs at location
            location_npcs = await self.get_npcs_at_location(location)
            npc_ids = [npc.npc_id for npc in location_npcs]
        else:
            # Get all active NPCs
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
                    AND link_level != 50  -- Exclude neutral relationships
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
            "potential_conflicts": []
        }
        
        if not npc_ids:
            return dynamics
        
        # Calculate averages and identify patterns
        total_dominance = 0
        dominance_scores = []
        
        for npc_id in npc_ids:
            snapshot = await self.get_npc_snapshot(npc_id)
            dominance = snapshot.stats["dominance"]
            total_dominance += dominance
            dominance_scores.append((npc_id, dominance, snapshot.name))
        
        dynamics["average_dominance"] = total_dominance / len(npc_ids)
        
        # Identify leader candidates (high dominance)
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
            # High dominance NPCs might conflict
            if dominance_scores[0][1] > 70 and dominance_scores[1][1] > 70:
                dynamics["potential_conflicts"].append({
                    "type": "dominance_conflict",
                    "between": [dominance_scores[0][2], dominance_scores[1][2]]
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
        logger.info("All NPC caches refreshed")
    
    async def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get status information about the orchestrator."""
        return {
            "active_npcs": len(self._active_npcs),
            "cached_snapshots": len(self._snapshot_cache),
            "npc_statuses": dict(self._npc_status),
            "systems_initialized": {
                "agent_system": self._agent_system is not None,
                "coordinator": self._coordinator is not None,
                "handler": self._handler is not None,
                "memory_system": self._memory_system is not None,
                "lore_system": self._lore_system is not None
            }
        }
    
    async def shutdown(self) -> None:
        """Clean shutdown of the orchestrator."""
        logger.info("Shutting down NPC Orchestrator")
        self._snapshot_cache.clear()
        self._npc_cache.clear()
        self._active_npcs.clear()
        self._npc_status.clear()


# ==================== CONVENIENCE FUNCTIONS ====================

async def create_orchestrator(user_id: int, conversation_id: int) -> NPCOrchestrator:
    """Create and initialize an NPC orchestrator."""
    orchestrator = NPCOrchestrator(user_id, conversation_id)
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
    orchestrator = NPCOrchestrator(user_id, conversation_id)
    await orchestrator.initialize()
    
    try:
        return await orchestrator.get_narrative_context(
            focus_npc_ids=focus_npc_ids,
            location=location
        )
    finally:
        await orchestrator.shutdown()
