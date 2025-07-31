# logic/fully_integrated_npc_system.py

import os
import json
import logging
import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from functools import lru_cache
import contextlib

# UPDATED: Import async connection context manager
from db.connection import get_db_connection_context

# Import existing modules to utilize their functionality
from logic.time_cycle import (
    advance_time_with_events,
    get_current_time,
    set_current_time,
    update_npc_schedules_for_time,
    TIME_PHASES,
    ActivityManager
)
from logic.memory_logic import (
    record_npc_event,
    MemoryType,
    MemorySignificance
)
from logic.stats_logic import (
    apply_stat_change,
    apply_activity_effects,
    get_player_current_tier,
    check_for_combination_triggers,
    record_stat_change_event,
    STAT_THRESHOLDS,
    STAT_COMBINATIONS
)

# Import the new dynamic relationships system
from logic.dynamic_relationships import (
    OptimizedRelationshipManager,
    RelationshipState,
    RelationshipDimensions,
    event_generator,
    process_relationship_interaction_tool,
    get_relationship_summary_tool,
    poll_relationship_events_tool,
    drain_relationship_events_tool
)

# Import agent-based architecture components
from npcs.npc_agent import NPCAgent
from npcs.npc_agent_system import NPCAgentSystem
from npcs.npc_coordinator import NPCAgentCoordinator
from npcs.npc_decisions import NPCDecisionEngine
from npcs.npc_relationship import NPCRelationshipManager
from npcs.npc_memory import MemoryContext, NPCMemoryManager

from npcs.belief_system_integration import NPCBeliefSystemIntegration, enhance_npc_with_belief_system
from npcs.lore_context_manager import LoreContextManager
from npcs.npc_behavior import BehaviorEvolution, NPCBehavior
from npcs.npc_learning_adaptation import NPCLearningAdaptation, NPCLearningManager
from npcs.npc_perception import EnvironmentPerception, PerceptionContext

# Import for memory system
try:
    from memory.wrapper import MemorySystem
    from memory.core import Memory
    from memory.emotional import EmotionalMemoryManager
    from memory.schemas import MemorySchemaManager
    from memory.masks import ProgressiveRevealManager
    MEMORY_SYSTEM_AVAILABLE = True
except ImportError:
    MEMORY_SYSTEM_AVAILABLE = False
    logging.warning("Advanced memory system not available. Some features will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom exception classes for better error handling
class NPCSystemError(Exception):
    """Base exception for NPC system errors"""
    pass

class NPCNotFoundError(NPCSystemError):
    """Exception raised when an NPC cannot be found"""
    pass

class MemorySystemError(NPCSystemError):
    """Exception raised when there's an issue with the memory system"""
    pass

class RelationshipError(NPCSystemError):
    """Exception raised when there's an issue with relationships"""
    pass

class NPCCreationError(NPCSystemError):
    """Exception raised when there's an issue creating an NPC"""
    pass

class TimeSystemError(NPCSystemError):
    """Exception raised when there's an issue with the time system"""
    pass

@dataclass
class CrossroadsEvent:
    """Data class for relationship crossroads events"""
    entity1_type: str
    entity1_id: int
    entity2_type: str
    entity2_id: int
    relationship_state: RelationshipState
    event_type: str
    description: str
    options: List[Dict[str, Any]]
    expires_in: int

class IntegratedNPCSystem:
    """
    Central system that integrates NPC creation, social dynamics, time management,
    memory systems, and stat progression using an agent-based architecture.
    """
    
    def __init__(self, user_id: int, conversation_id: int, connection_pool=None):
        """
        Initialize the NPC system.
        
        Args:
            user_id: The user ID
            conversation_id: The conversation ID
        """
        from npcs.new_npc_creation import (
            NPCCreationHandler, 
            RunContextWrapper, 
            NPCCreationResult
        )       
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize caching structures with more granular control
        self.npc_cache = {}  # Cache for NPC data
        self.relationship_cache = {}  # Cache for relationship data
        self.memory_cache = {}  # Cache for frequently accessed memories
        self.last_cache_refresh = datetime.now()
        self.cache_ttl = timedelta(minutes=5)  # Cache time-to-live
        self.cache_hit_counts = {}  # Track cache effectiveness
        
        # Performance monitoring
        self.perf_metrics = {
            'db_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_query_time': 0,
            'query_times': [],
            'memory_retrieval_time': []
        }
        
        # Initialize the activity manager
        self.activity_manager = ActivityManager()
        
        # Initialize the agent system - core component for NPC agentic behavior
        self.agent_system = NPCAgentSystem(user_id, conversation_id, connection_pool)
        
        # Initialize the NPC creation handler from the new system
        self.npc_creation_handler = NPCCreationHandler()
        
        # Initialize the relationship manager
        self.relationship_manager = OptimizedRelationshipManager(user_id, conversation_id)
        
        logger.info(f"Initialized IntegratedNPCSystem for user={user_id}, conversation={conversation_id}")
        
        # Initialize memory system
        self._memory_system = None
        self.is_initialized: bool = False
        
        # Set up periodic cache cleanup and metrics reporting
        self._setup_cache_cleanup()
        self._setup_metrics_reporting()

    def _setup_cache_cleanup(self):
        """Set up periodic cache cleanup task with memory pressure detection."""
        async def cache_cleanup_task():
            while True:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Check for memory pressure
                memory_pressure = self._detect_memory_pressure()
                await self._cleanup_cache(force=memory_pressure)
        
        # Start the task without waiting for it
        asyncio.create_task(cache_cleanup_task())

    async def initialize(self):
        """Initialize any async components that weren't initialized in the constructor."""
        # Initialize memory system if needed
        if MEMORY_SYSTEM_AVAILABLE and self._memory_system is None:
            try:
                self._memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
            except Exception as e:
                logger.error(f"Failed to initialize memory system: {e}")

        self.is_initialized = True
        return self

    async def _get_active_npcs(self) -> List[Dict[str, Any]]:
        """Get all active NPCs in the current conversation."""
        try:
            # Using get_db_connection_context() instead of connection_pool.acquire()
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location, schedule
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                    AND is_active = true
                """, self.user_id, self.conversation_id)
                
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting active NPCs: {e}")
            return []

    def _detect_memory_pressure(self):
        """Detect memory pressure in the application."""
        try:
            # Try to get memory info using psutil if available
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Log memory usage
            logger.debug(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB ({memory_percent:.1f}%)")
            
            # Consider high pressure if memory usage is above 75%
            if memory_percent > 75.0:
                logger.warning(f"High memory pressure detected: {memory_percent:.1f}%")
                return True
                
            # Check if we have too many cached items
            cache_size = (
                len(self.npc_cache) + 
                len(self.relationship_cache) + 
                len(self.memory_cache)
            )
            
            if cache_size > 1000:  # Arbitrary threshold
                logger.warning(f"Large cache size detected: {cache_size} items")
                return True
                
            return False
        except ImportError:
            # If psutil isn't available, use a simpler heuristic based on cache size
            cache_size = (
                len(self.npc_cache) + 
                len(self.relationship_cache) + 
                len(self.memory_cache)
            )
            
            return cache_size > 500  # Lower threshold when we can't check actual memory
        except Exception as e:
            logger.error(f"Error detecting memory pressure: {e}")
            return False
    
    async def _cleanup_cache(self, force=False):
        """
        Clean up expired cache entries with granular control.
        
        Args:
            force: Force cleanup regardless of TTL
        """
        now = datetime.now()
        
        # Check global TTL first
        global_expired = now - self.last_cache_refresh > self.cache_ttl
        if global_expired or force:
            self.last_cache_refresh = now
            logger.debug(f"{'Forced' if force else 'Global TTL expired'}, clearing all caches")
            
            # Clear all caches
            self.npc_cache.clear()
            self.relationship_cache.clear()
            self.memory_cache.clear()
            return
        
        # Check individual entries for expiration
        expired_npc_keys = []
        for key, data in self.npc_cache.items():
            if now - data.get("last_updated", datetime.min) > self.cache_ttl:
                expired_npc_keys.append(key)
        
        expired_rel_keys = []
        for key, data in self.relationship_cache.items():
            if now - data.get("last_updated", datetime.min) > self.cache_ttl:
                expired_rel_keys.append(key)
        
        expired_mem_keys = []
        for key, data in self.memory_cache.items():
            if now - data.get("last_updated", datetime.min) > self.cache_ttl:
                expired_mem_keys.append(key)
        
        # Remove expired entries
        for key in expired_npc_keys:
            del self.npc_cache[key]
        
        for key in expired_rel_keys:
            del self.relationship_cache[key]
        
        for key in expired_mem_keys:
            del self.memory_cache[key]
        
        logger.debug(f"Cleaned up {len(expired_npc_keys)} NPC cache entries, " +
                    f"{len(expired_rel_keys)} relationship cache entries, " +
                    f"{len(expired_mem_keys)} memory cache entries")
    
    # Database access methods
    async def execute_query(self, query, *args, timeout=10.0):
        """Execute a database query with error handling."""
        start_time = datetime.now()
        self.perf_metrics['db_queries'] += 1
        
        try:
            async with get_db_connection_context(timeout=timeout) as conn:
                result = await conn.execute(query, *args)
                
                # Track query performance
                query_time = (datetime.now() - start_time).total_seconds()
                self.perf_metrics['query_times'].append(query_time)
                self._update_avg_query_time(query_time)
                
                return result
        except Exception as e:
            logger.error(f"Database query error: {e}, Query: {query[:100]}...")
            raise
    
    async def fetch_row(self, query, *args, timeout=10.0):
        """Fetch a single row with error handling."""
        start_time = datetime.now()
        self.perf_metrics['db_queries'] += 1
        
        try:
            async with get_db_connection_context(timeout=timeout) as conn:
                result = await conn.fetchrow(query, *args)
                
                # Track query performance
                query_time = (datetime.now() - start_time).total_seconds()
                self.perf_metrics['query_times'].append(query_time)
                self._update_avg_query_time(query_time)
                
                return result
        except Exception as e:
            logger.error(f"Database query error: {e}, Query: {query[:100]}...")
            raise
    
    async def fetch_all(self, query, *args, timeout=10.0):
        """Fetch multiple rows with error handling."""
        start_time = datetime.now()
        self.perf_metrics['db_queries'] += 1
        
        try:
            async with get_db_connection_context(timeout=timeout) as conn:
                result = await conn.fetch(query, *args)
                
                # Track query performance
                query_time = (datetime.now() - start_time).total_seconds()
                self.perf_metrics['query_times'].append(query_time)
                self._update_avg_query_time(query_time)
                
                return result
        except Exception as e:
            logger.error(f"Database query error: {e}, Query: {query[:100]}...")
            raise
    
    def _update_avg_query_time(self, new_time):
        """Update the average query time metric."""
        times = self.perf_metrics['query_times'][-100:]  # Only keep last 100 times
        self.perf_metrics['avg_query_time'] = sum(times) / len(times)
    
    def _setup_metrics_reporting(self):
        """Set up periodic metrics reporting for performance monitoring."""
        async def metrics_reporting_task():
            while True:
                await asyncio.sleep(600)  # Report every 10 minutes
                await self._report_performance_metrics()
        
        # Start the task without waiting for it
        asyncio.create_task(metrics_reporting_task())
    
    async def _report_performance_metrics(self):
        """Report detailed performance metrics."""
        # Calculate additional derived metrics
        db_metrics = {
            "db_queries": self.perf_metrics['db_queries'],
            "avg_query_time": self.perf_metrics['avg_query_time'],
            "max_query_time": max(self.perf_metrics['query_times']) if self.perf_metrics['query_times'] else 0,
            "min_query_time": min(self.perf_metrics['query_times']) if self.perf_metrics['query_times'] else 0,
            "slow_query_count": sum(1 for t in self.perf_metrics['query_times'] if t > 0.5)
        }
        
        cache_metrics = {
            "cache_hits": self.perf_metrics['cache_hits'],
            "cache_misses": self.perf_metrics['cache_misses'],
            "hit_ratio": self.perf_metrics['cache_hits'] / (self.perf_metrics['cache_hits'] + self.perf_metrics['cache_misses']) 
                if (self.perf_metrics['cache_hits'] + self.perf_metrics['cache_misses']) > 0 else 0,
            "cache_size": {
                "npc_cache": len(self.npc_cache),
                "relationship_cache": len(self.relationship_cache),
                "memory_cache": len(self.memory_cache)
            }
        }
        
        # Combine all metrics
        full_metrics = {
            "timestamp": datetime.now().isoformat(),
            "db": db_metrics,
            "cache": cache_metrics
        }
        
        # Log metrics at appropriate level
        if db_metrics["slow_query_count"] > 10:
            logger.warning(f"Performance metrics indicate issues: {full_metrics}")
        else:
            logger.info(f"Performance metrics: {full_metrics}")
        
        # Reset counters
        self.perf_metrics['db_queries'] = 0
        self.perf_metrics['cache_hits'] = 0
        self.perf_metrics['cache_misses'] = 0
        self.perf_metrics['query_times'] = self.perf_metrics['query_times'][-100:]
    
    async def _get_memory_system(self):
        """Lazy-load the memory system."""
        if self._memory_system is None:
            try:
                self._memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
            except Exception as e:
                raise MemorySystemError(f"Failed to initialize memory system: {e}")
        return self._memory_system
    
    #=================================================================
    # NPC CREATION AND MANAGEMENT
    #=================================================================
    
    async def create_new_npc(self, environment_desc: str, day_names: List[str], sex: str = "female", specific_traits: Dict[str, Any] = None) -> int:
        """
        Create a new NPC using the enhanced NPCCreationHandler.
        
        Args:
            environment_desc: Description of the environment
            day_names: List of day names
            sex: Gender of the NPC
            specific_traits: Dictionary with specific traits to incorporate (optional)
            
        Returns:
            NPC ID
        """
        logger.info(f"Creating new NPC in environment: {environment_desc[:30]}...")
        from npcs.new_npc_creation import (
            NPCCreationHandler, 
            RunContextWrapper, 
            NPCCreationResult
        )        
                
        try:
            # Create context for the NPC creation handler
            ctx = RunContextWrapper({
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
            # Use the enhanced NPC creation system
            creation_result = await self.npc_creation_handler.create_npc_with_context(
                environment_desc=environment_desc,
                archetype_names=None,  # Let the system choose appropriate archetypes
                specific_traits=specific_traits or {"sex": sex},
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            npc_id = creation_result.npc_id
            npc_name = creation_result.npc_name
            current_location = creation_result.current_location
            memories = creation_result.memories
            
            # Initialize advanced systems for the new NPC
            await self._initialize_advanced_npc_systems(
                npc_id=npc_id,
                npc_name=npc_name,
                memories=memories,
                current_location=current_location
            )
            
            # Create initial relationship with player if this is the first NPC
            await self._initialize_npc_player_relationship(npc_id)
            
            logger.info(f"Successfully created NPC {npc_id} ({npc_name})")
            return npc_id
            
        except Exception as e:
            logger.error(f"Error creating NPC: {e}")
            # Try fallback approach
            return await self._create_minimal_fallback_npc(
                f"NPC_{datetime.now().timestamp()}",
                sex
            )
    
    async def _initialize_npc_player_relationship(self, npc_id: int):
        """Initialize the relationship between a new NPC and the player."""
        try:
            # Create an initial neutral relationship with the player
            interaction = {
                "type": "first_encounter",
                "context": "meeting"
            }
            
            await self.relationship_manager.process_interaction(
                entity1_type="npc",
                entity1_id=npc_id,
                entity2_type="player", 
                entity2_id=self.user_id,
                interaction=interaction
            )
            
            logger.info(f"Initialized relationship between NPC {npc_id} and player")
            
        except Exception as e:
            logger.error(f"Error initializing NPC-player relationship: {e}")
    
    async def _initialize_advanced_npc_systems(
        self, 
        npc_id: int, 
        npc_name: str, 
        memories: List[Dict[str, Any]],
        current_location: str
    ):
        """Initialize all advanced systems for a new NPC."""
        try:
            # Initialize NPC agent as before
            if npc_id not in self.agent_system.npc_agents:
                self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
            
            agent = self.agent_system.npc_agents[npc_id]
            
            # Enhance agent with belief system
            enhance_npc_with_belief_system(agent)
            
            # Initialize perception system
            agent.perception_system = EnvironmentPerception(npc_id, self.user_id, self.conversation_id)
            
            # Initialize behavior evolution
            agent.behavior_evolution = BehaviorEvolution(self.user_id, self.conversation_id)
            
            # Initialize learning adaptation
            agent.learning_system = NPCLearningAdaptation(self.user_id, self.conversation_id, npc_id)
            await agent.learning_system.initialize()
            
            # Initialize lore context
            agent.lore_context = LoreContextManager(self.user_id, self.conversation_id)
            
            # Process memories as before
            memory_system = await agent._get_memory_system()
            for memory in memories:
                memory_text = memory if isinstance(memory, str) else memory.get("text", "")
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=npc_id,
                    memory_text=memory_text,
                    importance="medium",
                    tags=["creation", "origin"]
                )
        
        except Exception as e:
            logger.error(f"Error initializing advanced systems for NPC {npc_id}: {e}")

    async def process_event_for_beliefs(self, event_text: str, event_type: str, npc_ids: List[int], factuality: float = 1.0) -> Dict[str, Any]:
        """
        Process a game event to generate beliefs for multiple NPCs.
        
        Args:
            event_text: Description of the event
            event_type: Type of event
            npc_ids: List of NPC IDs who witnessed the event
            factuality: Base factuality level for beliefs
            
        Returns:
            Information about the beliefs formed
        """
        # Create a belief system integration if not already cached
        belief_system = NPCBeliefSystemIntegration(self.user_id, self.conversation_id)
        await belief_system.initialize()
        
        # Process the event for beliefs
        result = await belief_system.process_event_for_beliefs(
            event_text=event_text,
            event_type=event_type,
            npc_ids=npc_ids,
            factuality=factuality
        )
        
        return result
    
    async def process_conversation_for_beliefs(self, conversation_text: str, speaker_id: Union[int, str], listener_id: int, topic: str = "general", credibility: float = 0.7) -> Dict[str, Any]:
        """
        Process a conversation to generate beliefs for an NPC.
        
        Args:
            conversation_text: The content of what was said
            speaker_id: ID of the speaker (NPC ID or 'player')
            listener_id: NPC ID of the listener
            topic: Topic of conversation
            credibility: How credible the speaker is (0.0-1.0)
            
        Returns:
            Information about the beliefs formed
        """
        belief_system = NPCBeliefSystemIntegration(self.user_id, self.conversation_id)
        await belief_system.initialize()
        
        result = await belief_system.process_conversation_for_beliefs(
            conversation_text=conversation_text,
            speaker_id=speaker_id,
            listener_id=listener_id,
            topic=topic,
            credibility=credibility
        )
        
        return result

    async def perceive_environment_for_npc(self, npc_id: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Have an NPC perceive their environment.
        
        Args:
            npc_id: ID of the NPC
            context: Context information for perception
            
        Returns:
            Perception results
        """
        # Get or create NPC agent
        if npc_id not in self.agent_system.npc_agents:
            self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
        
        agent = self.agent_system.npc_agents[npc_id]
        
        # Initialize perception system if not already done
        if not hasattr(agent, "perception_system"):
            agent.perception_system = EnvironmentPerception(npc_id, self.user_id, self.conversation_id)
        
        # Convert to PerceptionContext if needed
        if not isinstance(context, PerceptionContext):
            context = PerceptionContext(**context)
        
        # Perform perception
        perception_result = await agent.perception_system.perceive_environment(context)
        
        return perception_result.dict()

    async def evaluate_npc_scheming(self, npc_id: int) -> Dict[str, Any]:
        """
        Evaluate if an NPC should adjust their behavior, escalate plans, or set new secret goals.
        
        Args:
            npc_id: ID of the NPC to evaluate
            
        Returns:
            Dictionary with scheming adjustments
        """
        # Create a behavior evolution instance
        behavior_evolution = BehaviorEvolution(self.user_id, self.conversation_id)
        
        # Evaluate scheming
        adjustments = await behavior_evolution.evaluate_npc_scheming(npc_id)
        
        # Apply adjustments
        if "error" not in adjustments:
            await behavior_evolution.apply_scheming_adjustments(npc_id, adjustments)
        
        return adjustments
    
    async def evaluate_npc_scheming_for_all(self, npc_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Evaluate and update scheming behavior for multiple NPCs.
        
        Args:
            npc_ids: List of NPC IDs to evaluate
            
        Returns:
            Dictionary mapping NPC IDs to their scheming adjustments
        """
        behavior_evolution = BehaviorEvolution(self.user_id, self.conversation_id)
        return await behavior_evolution.evaluate_npc_scheming_for_all(npc_ids)

    async def record_player_interaction_for_npc(self, npc_id: int, interaction_type: str, interaction_details: Dict[str, Any], player_response: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Record a player interaction to drive NPC learning and adaptation.
        
        Args:
            npc_id: ID of the NPC
            interaction_type: Type of interaction
            interaction_details: Details about the interaction
            player_response: Optional information about how the player responded
            
        Returns:
            Learning outcomes and adaptation results
        """
        # Create a learning system for this NPC
        learning_system = NPCLearningAdaptation(self.user_id, self.conversation_id, npc_id)
        await learning_system.initialize()
        
        # Record the interaction
        result = await learning_system.record_player_interaction(
            interaction_type=interaction_type,
            interaction_details=interaction_details,
            player_response=player_response
        )
        
        return result
    
    async def process_recent_memories_for_learning(self, npc_id: int, days: int = 7) -> Dict[str, Any]:
        """
        Process recent memories to drive NPC learning.
        
        Args:
            npc_id: ID of the NPC
            days: Number of days of memories to process
            
        Returns:
            Learning outcomes and adaptation results
        """
        learning_system = NPCLearningAdaptation(self.user_id, self.conversation_id, npc_id)
        await learning_system.initialize()
        
        result = await learning_system.process_recent_memories_for_learning(days=days)
        
        return result

    async def get_lore_context_for_npc(self, npc_id: int, context_type: str) -> Dict[str, Any]:
        """
        Get lore context for an NPC.
        
        Args:
            npc_id: ID of the NPC
            context_type: Type of context to retrieve
            
        Returns:
            Lore context information
        """
        lore_manager = LoreContextManager(self.user_id, self.conversation_id)
        
        context = await lore_manager.get_lore_context(npc_id, context_type)
        
        return context
    
    async def handle_lore_change(self, lore_change: Dict[str, Any], source_npc_id: int, affected_npcs: List[int]) -> Dict[str, Any]:
        """
        Handle a lore change, analyzing impact and propagating to NPCs.
        
        Args:
            lore_change: Details of the lore change
            source_npc_id: ID of the NPC who is the source of the change
            affected_npcs: List of NPC IDs affected by the change
            
        Returns:
            Analysis and propagation results
        """
        lore_manager = LoreContextManager(self.user_id, self.conversation_id)
        
        result = await lore_manager.handle_lore_change(
            lore_change=lore_change,
            source_npc_id=source_npc_id,
            affected_npcs=affected_npcs
        )
        
        return result

    async def _initialize_npc_agent(
        self, 
        npc_id: int, 
        npc_name: str, 
        memories: List[Dict[str, Any]],
        current_location: str,
        time_of_day: str
    ):
        """Initialize NPC agent with memory and perception in the background."""
        try:
            # Create agent if it doesn't exist
            if npc_id not in self.agent_system.npc_agents:
                self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
            
            agent = self.agent_system.npc_agents[npc_id]
            
            # Initialize memories in batches for better performance
            if memories:
                memory_system = await agent._get_memory_system()
                
                # Process in batches of 5 for optimal performance
                batch_size = 5
                for i in range(0, len(memories), batch_size):
                    batch = memories[i:i+batch_size]
                    
                    # Create batch memory tasks
                    memory_tasks = []
                    for memory in batch:
                        memory_text = memory if isinstance(memory, str) else memory.get("text", "")
                        memory_tasks.append(
                            memory_system.remember(
                                entity_type="npc",
                                entity_id=npc_id,
                                memory_text=memory_text,
                                importance="medium",
                                tags=["creation", "origin"]
                            )
                        )
                    
                    # Process batch concurrently
                    await asyncio.gather(*memory_tasks)
                    
                    # Small delay between batches to prevent overloading
                    await asyncio.sleep(0.1)
            
            # Initialize mask
            mask_manager = await agent._get_mask_manager()
            await mask_manager.initialize_npc_mask(npc_id)
            
            # Create initial perception
            initial_context = {
                "location": current_location,
                "time_of_day": time_of_day,
                "description": f"Initial perception upon creation at {current_location}"
            }
            
            await agent.perceive_environment(initial_context)
            
        except Exception as e:
            logger.error(f"Error initializing NPC agent {npc_id}: {e}")
            # Non-critical error, just log

    async def _create_minimal_fallback_npc(self, npc_name: str, sex: str) -> int:
        """
        Create a minimal viable NPC as a fallback when full creation fails.
        
        Args:
            npc_name: NPC name to use
            sex: NPC sex
            
        Returns:
            NPC ID of the created minimal NPC
        """
        async with get_db_connection_context() as conn:
            npc_id = await conn.fetchval("""
                INSERT INTO NPCStats (
                    user_id, conversation_id, npc_name, sex, 
                    dominance, cruelty, introduced, 
                    current_location
                )
                VALUES ($1, $2, $3, $4, 50, 50, FALSE, 'Unknown')
                RETURNING npc_id
            """, self.user_id, self.conversation_id, npc_name, sex)
            
            # Create a basic minimal schedule
            basic_schedule = {}
            for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
                basic_schedule[day] = {
                    "Morning": "Goes about their day",
                    "Afternoon": "Continues their routine",
                    "Evening": "Relaxes at home",
                    "Night": "Sleeps"
                }
                
            # Update with schedule
            await conn.execute("""
                UPDATE NPCStats
                SET schedule = $1
                WHERE npc_id = $2
            """, json.dumps(basic_schedule), npc_id)
            
            return npc_id

    async def create_multiple_npcs(self, environment_desc: str, day_names: List[str], count: int = 3) -> List[int]:
        """
        Create multiple NPCs using the new NPCCreationHandler.
        
        Args:
            environment_desc: Description of the environment
            day_names: List of day names
            count: Number of NPCs to create
            
        Returns:
            List of created NPC IDs
        """
        from npcs.new_npc_creation import (
            NPCCreationHandler, 
            RunContextWrapper, 
            NPCCreationResult
        )        
        # Create context for the NPC creation handler
        ctx = RunContextWrapper({
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # Spawn multiple NPCs using the new handler
        npc_ids = await self.npc_creation_handler.spawn_multiple_npcs(
            ctx, 
            count=count
        )
        
        # Initialize agents for the new NPCs
        for npc_id in npc_ids:
            # Get NPC details to initialize agent
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow("""
                    SELECT npc_name, current_location, memory 
                    FROM NPCStats 
                    WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
                """, npc_id, self.user_id, self.conversation_id)
                
                if row:
                    npc_name = row[0]
                    current_location = row[1] or "Unknown"
                    memories = row[2] if row[2] else []
                    
                    # Parse memories if needed
                    if isinstance(memories, str):
                        try:
                            memories = json.loads(memories)
                        except:
                            memories = []
                    
                    # Initialize agent in background
                    asyncio.create_task(
                        self._initialize_npc_agent(
                            npc_id=npc_id,
                            npc_name=npc_name,
                            memories=memories,
                            current_location=current_location,
                            time_of_day="Morning"  # Default
                        )
                    )
                    
                    # Update cache
                    self.npc_cache[npc_id] = {
                        "npc_id": npc_id,
                        "npc_name": npc_name,
                        "last_updated": datetime.now()
                    }
                    
            # Initialize relationship with player
            await self._initialize_npc_player_relationship(npc_id)
        
        return npc_ids

    async def batch_create_memories(
        self, 
        npc_id: int,
        memories: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Create multiple memories in a single operation for better performance.
        
        Args:
            npc_id: ID of the NPC
            memories: List of memory objects with text, type, significance, etc.
            
        Returns:
            List of created memory IDs
        """
        if not memories:
            return []
            
        memory_system = await self._get_memory_system()
        
        # Prepare all memories for batch insertion
        batch_values = []
        for memory in memories:
            memory_text = memory if isinstance(memory, str) else memory.get("text", "")
            
            # Get emotional analysis for each memory text
            emotion_analysis = await memory_system.emotional_manager.analyze_emotional_content(
                memory_text
            )
            
            batch_values.append({
                "entity_type": "npc",
                "entity_id": npc_id,
                "memory_text": memory_text,
                "importance": memory.get("importance", "medium") if isinstance(memory, dict) else "medium",
                "emotional": memory.get("emotional", False) if isinstance(memory, dict) else False,
                "primary_emotion": emotion_analysis.get("primary_emotion", "neutral"),
                "emotion_intensity": emotion_analysis.get("intensity", 0.5),
                "tags": memory.get("tags", []) if isinstance(memory, dict) else []
            })
        
        # Execute batch insert
        results = await memory_system.batch_remember(batch_values)
        
        # Process schemas in background task for efficiency
        asyncio.create_task(self._process_batch_schemas(results, npc_id))
        
        return results.get("memory_ids", [])
    
    async def _process_batch_schemas(self, results, npc_id):
        """Process batch memory schemas in the background."""
        try:
            memory_system = await self._get_memory_system()
            schema_manager = await memory_system.get_schema_manager()
            
            # Process at most 5 schemas to avoid overloading
            memory_ids = results.get("memory_ids", [])[:5]
            if memory_ids:
                await schema_manager.process_memories_for_schemas(
                    "npc", npc_id, memory_ids
                )
        except Exception as e:
            logger.error(f"Error processing batch schemas for NPC {npc_id}: {e}")
        
    async def get_npc_details(self, npc_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an NPC, enhanced with agent-based data.
        Uses caching for performance optimization and connection reuse.
        """
        # Check cache first
        cache_key = f"npc:{npc_id}"
        if cache_key in self.npc_cache and (datetime.now() - self.npc_cache[cache_key].get("last_updated", datetime.min)).seconds < 300:
            self.perf_metrics['cache_hits'] += 1
            return self.npc_cache[cache_key]
        
        self.perf_metrics['cache_misses'] += 1
        
        # Get or create NPC agent
        if npc_id not in self.agent_system.npc_agents:
            self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
        
        agent = self.agent_system.npc_agents[npc_id]
        
        try:
            async with get_db_connection_context() as conn:
                # Use a single connection for all related queries
                start_time = datetime.now()
                
                npc_data = await self._fetch_npc_basic_data(conn, npc_id)
                if not npc_data:
                    error_msg = f"NPC with ID {npc_id} not found"
                    logger.error(error_msg)
                    raise NPCNotFoundError(error_msg)
                
                # Get relationships using the new system
                relationships = await self._fetch_npc_relationships_dynamic(npc_id)
                
                query_time = (datetime.now() - start_time).total_seconds()
                self.perf_metrics['query_times'].append(query_time)
                self.perf_metrics['db_queries'] += 1
                
                # Get enhanced memory from the agent's memory system
                memory_system = await agent._get_memory_system()
                memory_result = await memory_system.recall(
                    entity_type="npc",
                    entity_id=npc_id,
                    limit=5
                )
                
                agent_memories = memory_result.get("memories", [])
                
                # Get mask information using agent
                mask_info = await agent._get_mask_manager().get_npc_mask(npc_id)
                
                # Get emotional state using agent
                emotional_state = await memory_system.get_npc_emotion(npc_id)
                
                # Get beliefs using agent
                beliefs = await memory_system.get_beliefs(
                    entity_type="npc",
                    entity_id=npc_id,
                    topic="player"
                )
                
                # Get current perception through agent
                current_perception = None
                if agent.last_perception:
                    current_perception = {
                        "location": agent.last_perception.get("environment", {}).get("location"),
                        "time_of_day": agent.last_perception.get("environment", {}).get("time_of_day"),
                        "entities_present": agent.last_perception.get("environment", {}).get("entities_present", [])
                    }
                
                # Build enhanced response with agent-based data
                npc_details = {
                    **npc_data,  # Merge basic data
                    "memories": agent_memories or npc_data.get("memories", [])[:5],  # Prefer agent memories
                    "memory_count": len(npc_data.get("memories", [])),
                    "mask": mask_info if mask_info and "error" not in mask_info else {"integrity": 100},
                    "emotional_state": emotional_state,
                    "beliefs": beliefs,
                    "current_perception": current_perception,
                    "relationships": relationships
                }
                
                # Update cache
                self.npc_cache[cache_key] = npc_details
                self.npc_cache[cache_key]["last_updated"] = datetime.now()
                
                return npc_details
        
        except NPCNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Error getting NPC details: {e}"
            logger.error(error_msg)
            return None

    async def _fetch_npc_basic_data(self, conn, npc_id: int) -> Dict[str, Any]:
        """Fetch basic NPC data from database using provided connection."""
        row = await conn.fetchrow("""
            SELECT npc_id, npc_name, introduced, sex, dominance, cruelty, 
                   closeness, trust, respect, intensity, archetype_summary,
                   physical_description, current_location, memory
            FROM NPCStats
            WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
        """, self.user_id, self.conversation_id, npc_id)
        
        if not row:
            return None
            
        # Parse memory with error handling
        memories = []
        if row["memory"]:
            try:
                if isinstance(row["memory"], str):
                    memories = json.loads(row["memory"])
                else:
                    memories = row["memory"]
            except json.JSONDecodeError:
                logger.warning(f"Error parsing memory for NPC {npc_id}")
                memories = []
        
        return {
            "npc_id": row["npc_id"],
            "npc_name": row["npc_name"],
            "introduced": row["introduced"],
            "sex": row["sex"],
            "stats": {
                "dominance": row["dominance"],
                "cruelty": row["cruelty"],
                "closeness": row["closeness"],
                "trust": row["trust"],
                "respect": row["respect"],
                "intensity": row["intensity"]
            },
            "archetype_summary": row["archetype_summary"],
            "physical_description": row["physical_description"],
            "current_location": row["current_location"],
            "memories": memories
        }
    
    async def _fetch_npc_relationships_dynamic(self, npc_id: int) -> List[Dict[str, Any]]:
        """
        Fetch relationships for an NPC using the new dynamic relationship system.
        
        Args:
            npc_id: The NPC ID
            
        Returns:
            List of relationship information
        """
        relationships = []
        
        try:
            # Get player relationship
            player_rel = await self.relationship_manager.get_relationship_state(
                entity1_type="npc",
                entity1_id=npc_id,
                entity2_type="player",
                entity2_id=self.user_id
            )
            
            if player_rel:
                relationships.append({
                    "target_type": "player",
                    "target_id": self.user_id,
                    "target_name": "Chase",
                    "dimensions": player_rel.dimensions.to_dict(),
                    "momentum": player_rel.momentum.get_magnitude(),
                    "patterns": list(player_rel.history.active_patterns),
                    "archetypes": list(player_rel.active_archetypes)
                })
            
            # Get other NPC relationships if needed
            # This could be expanded to fetch relationships with other NPCs
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error fetching dynamic relationships for NPC {npc_id}: {e}")
            return []
    
    async def introduce_npc(self, npc_id: int) -> bool:
        """
        Mark an NPC as introduced, updating agent memory.
        
        Args:
            npc_id: The ID of the NPC to introduce
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            NPCNotFoundError: If the NPC cannot be found
        """
        # Get or create the NPC agent
        if npc_id not in self.agent_system.npc_agents:
            self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
        
        agent = self.agent_system.npc_agents[npc_id]
        
        try:
            # Use connection for performance
            async with get_db_connection_context() as conn:
                # Update NPC status with RETURNING to verify existence
                row = await conn.fetchrow("""
                    UPDATE NPCStats
                    SET introduced=TRUE
                    WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                    RETURNING npc_name
                """, self.user_id, self.conversation_id, npc_id)
                
                if not row:
                    error_msg = f"NPC with ID {npc_id} not found"
                    logger.error(error_msg)
                    raise NPCNotFoundError(error_msg)
                    
                npc_name = row["npc_name"]
                
                # Add to player journal
                await conn.execute("""
                    INSERT INTO PlayerJournal 
                    (user_id, conversation_id, entry_type, entry_text, timestamp)
                    VALUES ($1, $2, 'npc_introduction', $3, CURRENT_TIMESTAMP)
                """, 
                    self.user_id, self.conversation_id,
                    f"Met {npc_name} for the first time."
                )
            
            # Create an introduction memory using the agent's memory system
            memory_system = await agent._get_memory_system()
            introduction_memory = f"I was formally introduced to the player today."
            
            await memory_system.remember(
                entity_type="npc",
                entity_id=npc_id,
                memory_text=introduction_memory,
                importance="medium",
                tags=["introduction", "player_interaction", "first_meeting"]
            )
            
            # Update the agent's emotional state based on introduction
            await memory_system.update_npc_emotion(
                npc_id=npc_id,
                emotion="curiosity", 
                intensity=0.7
            )
            
            # Update the relationship with a formal introduction interaction
            interaction = {
                "type": "formal_introduction",
                "context": "meeting"
            }
            
            await self.relationship_manager.process_interaction(
                entity1_type="npc",
                entity1_id=npc_id,
                entity2_type="player",
                entity2_id=self.user_id,
                interaction=interaction
            )
            
            # Update cache if present
            cache_key = f"npc:{npc_id}"
            if cache_key in self.npc_cache:
                self.npc_cache[cache_key]["introduced"] = True
                self.npc_cache[cache_key]["last_updated"] = datetime.now()
            
            return True
            
        except NPCNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Error introducing NPC: {e}"
            logger.error(error_msg)
            return False

    #=================================================================
    # SOCIAL LINKS AND RELATIONSHIPS - REFACTORED FOR DYNAMIC SYSTEM
    #=================================================================
    
    async def create_direct_social_link(self, 
                                      entity1_type: str, entity1_id: int,
                                      entity2_type: str, entity2_id: int,
                                      link_type: str = "neutral", 
                                      link_level: int = 0) -> Dict[str, Any]:
        """
        Create a relationship between two entities using the dynamic system.
        
        Args:
            entity1_type: Type of first entity
            entity1_id: ID of first entity
            entity2_type: Type of second entity
            entity2_id: ID of second entity
            link_type: Type of link (now maps to interaction type)
            link_level: Level of the link (now affects initial dimensions)
            
        Returns:
            Dictionary with relationship information
            
        Raises:
            RelationshipError: If there's an issue creating the relationship
        """
        try:
            # Map old link types to new interaction types
            interaction_type_map = {
                "friendly": "genuine_compliment",
                "hostile": "criticism_harsh",
                "neutral": "first_encounter",
                "rival": "conflict_resolved",
                "romantic": "vulnerability_shared"
            }
            
            interaction_type = interaction_type_map.get(link_type, "first_encounter")
            
            # Create initial interaction
            interaction = {
                "type": interaction_type,
                "context": "establishing_relationship",
                "initial_level": link_level
            }
            
            # Process the interaction
            result = await self.relationship_manager.process_interaction(
                entity1_type=entity1_type,
                entity1_id=entity1_id,
                entity2_type=entity2_type,
                entity2_id=entity2_id,
                interaction=interaction
            )
            
            # If this involves an NPC, update their memory
            if entity1_type == "npc":
                # Get or create NPC agent
                if entity1_id not in self.agent_system.npc_agents:
                    self.agent_system.npc_agents[entity1_id] = NPCAgent(entity1_id, self.user_id, self.conversation_id)
                
                # Create memory of this link
                agent = self.agent_system.npc_agents[entity1_id]
                memory_system = await agent._get_memory_system()
                
                # Get target name for better memory context
                target_name = await self._get_entity_name(entity2_type, entity2_id)
                
                memory_text = f"I formed a {link_type} relationship with {target_name}."
                
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=entity1_id,
                    memory_text=memory_text,
                    importance="medium",
                    tags=["relationship", entity2_type, link_type]
                )
            
            # Invalidate relationship cache
            cache_key = f"rel:{entity1_type}:{entity1_id}:{entity2_type}:{entity2_id}"
            if cache_key in self.relationship_cache:
                del self.relationship_cache[cache_key]
            
            logger.info(f"Created relationship between {entity1_type}:{entity1_id} and {entity2_type}:{entity2_id}")
            
            return {
                "success": result.get("success", True),
                "dimensions": result.get("dimensions_diff", {}),
                "patterns": result.get("new_patterns", []),
                "archetypes": result.get("new_archetypes", [])
            }
            
        except Exception as e:
            error_msg = f"Failed to create relationship: {e}"
            logger.error(error_msg)
            raise RelationshipError(error_msg)
    
    async def update_link_details(self, entity1_type: str, entity1_id: int, 
                                entity2_type: str, entity2_id: int,
                                new_type: str = None, level_change: int = 0) -> Dict[str, Any]:
        """
        Update a relationship using the dynamic system.
        
        Args:
            entity1_type: Type of first entity
            entity1_id: ID of first entity
            entity2_type: Type of second entity
            entity2_id: ID of second entity
            new_type: New type for the link (maps to interaction)
            level_change: Amount to change dimensions by
            
        Returns:
            Dictionary with update results
            
        Raises:
            RelationshipError: If there's an issue updating the relationship
        """
        try:
            # Map link type changes to interactions
            if new_type:
                interaction_map = {
                    "friendly": "support_provided",
                    "hostile": "boundary_violated",
                    "rival": "shared_success",
                    "romantic": "vulnerability_shared"
                }
                interaction_type = interaction_map.get(new_type, "helpful_action")
            else:
                # Use level change to determine interaction
                if level_change > 0:
                    interaction_type = "helpful_action"
                elif level_change < 0:
                    interaction_type = "criticism_harsh"
                else:
                    interaction_type = "neutral_interaction"
            
            # Create interaction
            interaction = {
                "type": interaction_type,
                "context": "relationship_update",
                "intensity": abs(level_change) / 10.0  # Scale to 0-1
            }
            
            # Process the interaction
            result = await self.relationship_manager.process_interaction(
                entity1_type=entity1_type,
                entity1_id=entity1_id,
                entity2_type=entity2_type,
                entity2_id=entity2_id,
                interaction=interaction
            )
            
            # Update agent memory if an NPC is involved
            if entity1_type == "npc":
                # Get or create NPC agent
                if entity1_id not in self.agent_system.npc_agents:
                    self.agent_system.npc_agents[entity1_id] = NPCAgent(entity1_id, self.user_id, self.conversation_id)
                
                agent = self.agent_system.npc_agents[entity1_id]
                memory_system = await agent._get_memory_system()
                
                # Get target name for better memory context
                target_name = await self._get_entity_name(entity2_type, entity2_id)
                
                if new_type:
                    memory_text = f"My relationship with {target_name} changed to {new_type}."
                else:
                    direction = "improved" if level_change > 0 else "worsened"
                    memory_text = f"My relationship with {target_name} {direction}."
                
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=entity1_id,
                    memory_text=memory_text,
                    importance="medium",
                    tags=["relationship_change", entity2_type]
                )
                
                # Update emotional state based on relationship change
                if abs(level_change) >= 10:
                    if level_change > 0:
                        await memory_system.update_npc_emotion(
                            npc_id=entity1_id,
                            emotion="joy",
                            intensity=0.6
                        )
                    else:
                        await memory_system.update_npc_emotion(
                            npc_id=entity1_id,
                            emotion="sadness",
                            intensity=0.6
                        )
            
            # Invalidate relationship cache
            cache_key = f"rel:{entity1_type}:{entity1_id}:{entity2_type}:{entity2_id}"
            if cache_key in self.relationship_cache:
                del self.relationship_cache[cache_key]
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to update relationship: {e}"
            logger.error(error_msg)
            raise RelationshipError(error_msg)
    
    async def add_event_to_link(self, entity1_type: str, entity1_id: int,
                              entity2_type: str, entity2_id: int,
                              event_text: str) -> bool:
        """
        Add an event to a relationship using the dynamic system.
        
        Args:
            entity1_type: Type of first entity
            entity1_id: ID of first entity
            entity2_type: Type of second entity
            entity2_id: ID of second entity
            event_text: Text describing the event
            
        Returns:
            True if successful
            
        Raises:
            RelationshipError: If there's an issue adding the event
        """
        try:
            # Parse event text to determine interaction type
            interaction_type = "neutral_interaction"
            if any(word in event_text.lower() for word in ["help", "assist", "support"]):
                interaction_type = "helpful_action"
            elif any(word in event_text.lower() for word in ["betray", "hurt", "attack"]):
                interaction_type = "betrayal"
            elif any(word in event_text.lower() for word in ["love", "care", "intimate"]):
                interaction_type = "vulnerability_shared"
            
            # Create interaction
            interaction = {
                "type": interaction_type,
                "context": "event",
                "description": event_text
            }
            
            # Process the interaction
            result = await self.relationship_manager.process_interaction(
                entity1_type=entity1_type,
                entity1_id=entity1_id,
                entity2_type=entity2_type,
                entity2_id=entity2_id,
                interaction=interaction
            )
            
            # Create memory record for NPC agents involved
            if entity1_type == "npc":
                # Get or create NPC agent
                if entity1_id not in self.agent_system.npc_agents:
                    self.agent_system.npc_agents[entity1_id] = NPCAgent(entity1_id, self.user_id, self.conversation_id)
                
                agent = self.agent_system.npc_agents[entity1_id]
                memory_system = await agent._get_memory_system()
                
                # Get target name for better memory context
                target_name = await self._get_entity_name(entity2_type, entity2_id)
                
                memory_text = f"With {target_name}: {event_text}"
                
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=entity1_id,
                    memory_text=memory_text,
                    importance="medium",
                    tags=["relationship_event", entity2_type]
                )
            
            # Invalidate relationship cache
            cache_key = f"rel:{entity1_type}:{entity1_id}:{entity2_type}:{entity2_id}"
            if cache_key in self.relationship_cache:
                del self.relationship_cache[cache_key]
            
            logger.info(f"Added event to relationship: {event_text[:50]}...")
            return True
            
        except Exception as e:
            error_msg = f"Failed to add event to relationship: {e}"
            logger.error(error_msg)
            raise RelationshipError(error_msg)
    
    async def update_relationship_from_interaction(self, 
                                               npc_id: int, 
                                               player_action: Dict[str, Any],
                                               npc_action: Dict[str, Any],
                                               context: Dict[str, Any] = None) -> bool:
        """
        Update relationship between NPC and player based on an interaction.
        Enhanced to use the dynamic relationship system.
        
        Args:
            npc_id: ID of the NPC
            player_action: Description of the player's action
            npc_action: Description of the NPC's action
            context: Additional context for the interaction
            
        Returns:
            True if successful
            
        Raises:
            RelationshipError: If there's an issue updating the relationship
        """
        try:
            # Get or create NPC agent
            if npc_id not in self.agent_system.npc_agents:
                self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
            
            agent = self.agent_system.npc_agents[npc_id]
            
            # Get agent's current emotional state and perception for context
            memory_system = await agent._get_memory_system()
            emotional_state = await memory_system.get_npc_emotion(npc_id)
            
            # Default context if none provided
            if context is None:
                context = {}
            
            # Enhance context with emotional state
            enhanced_context = {
                "emotional_state": emotional_state,
                "player_action_type": player_action.get("type", "unknown"),
                "npc_action_type": npc_action.get("type", "unknown")
            }
            
            # Update context with provided context
            enhanced_context.update(context)
            
            # Determine interaction type based on actions
            interaction_type = self._determine_interaction_type(player_action, npc_action)
            
            # Create interaction
            interaction = {
                "type": interaction_type,
                "context": enhanced_context.get("situation", "general"),
                "player_action": player_action.get("description", ""),
                "npc_response": npc_action.get("description", "")
            }
            
            # Process through dynamic relationship system
            result = await self.relationship_manager.process_interaction(
                entity1_type="npc",
                entity1_id=npc_id,
                entity2_type="player",
                entity2_id=self.user_id,
                interaction=interaction
            )
            
            # Create a memory of this relationship change
            # Format memory text based on action types
            if player_action.get("type") in ["help", "assist", "support"]:
                memory_text = "The player helped me, improving our relationship."
            elif player_action.get("type") in ["insult", "mock", "threaten"]:
                memory_text = "The player was hostile to me, damaging our relationship."
            else:
                memory_text = f"My relationship with the player changed after they {player_action.get('description', 'interacted with me')}."
            
            await memory_system.remember(
                entity_type="npc",
                entity_id=npc_id,
                memory_text=memory_text,
                importance="medium",
                tags=["relationship_change", "player_interaction"]
            )
            
            # Invalidate relationship cache
            cache_key = f"rel:npc:{npc_id}:player:{self.user_id}"
            if cache_key in self.relationship_cache:
                del self.relationship_cache[cache_key]
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to update relationship from interaction: {e}"
            logger.error(error_msg)
            raise RelationshipError(error_msg)
    
    def _determine_interaction_type(self, player_action: Dict[str, Any], npc_action: Dict[str, Any]) -> str:
        """Determine the interaction type based on player and NPC actions."""
        player_type = player_action.get("type", "").lower()
        
        # Map player action types to interaction types
        if player_type in ["help", "assist", "support"]:
            return "helpful_action"
        elif player_type in ["betray", "lie", "deceive"]:
            return "betrayal"
        elif player_type in ["compliment", "praise", "flatter"]:
            return "genuine_compliment"
        elif player_type in ["share", "confide", "reveal"]:
            return "vulnerability_shared"
        elif player_type in ["resolve", "apologize", "reconcile"]:
            return "conflict_resolved"
        elif player_type in ["violate", "cross", "ignore"]:
            return "boundary_violated"
        elif player_type in ["provide", "give", "offer"]:
            return "support_provided"
        elif player_type in ["criticize", "insult", "mock"]:
            return "criticism_harsh"
        elif player_type in ["achieve", "accomplish", "win"]:
            return "shared_success"
        elif player_type in ["lie", "deceive", "trick"]:
            return "deception_discovered"
        else:
            return "neutral_interaction"

    async def check_for_relationship_events(self) -> List[Dict[str, Any]]:
        """
        Check for special relationship events using the dynamic event system.
        
        Returns:
            List of relationship events
            
        Raises:
            RelationshipError: If there's an issue checking for events
        """
        try:
            events = []
            
            # Poll for events from the dynamic system
            event_data = await event_generator.get_next_event(timeout=0.5)
            
            while event_data:
                # Parse the event
                if event_data and "event" in event_data:
                    event = event_data["event"]
                    state_key = event_data.get("state_key", "")
                    
                    # Extract entity information from state key
                    parts = state_key.split("_")
                    if len(parts) >= 4:
                        entity1_type = parts[0]
                        entity1_id = int(parts[1])
                        entity2_type = parts[2]
                        entity2_id = int(parts[3])
                        
                        # Get the relationship state
                        rel_state = await self.relationship_manager.get_relationship_state(
                            entity1_type, entity1_id,
                            entity2_type, entity2_id
                        )
                        
                        # Create crossroads event if appropriate
                        if event.get("type") in ["moment_of_truth", "pattern_crisis", 
                                               "archetype_crisis", "reconnection_opportunity"]:
                            
                            crossroads = CrossroadsEvent(
                                entity1_type=entity1_type,
                                entity1_id=entity1_id,
                                entity2_type=entity2_type,
                                entity2_id=entity2_id,
                                relationship_state=rel_state,
                                event_type=event.get("type"),
                                description=event.get("description", ""),
                                options=event.get("choices", []),
                                expires_in=3
                            )
                            
                            events.append({
                                "type": "relationship_crossroads",
                                "data": crossroads
                            })
                        else:
                            # Other event types
                            events.append({
                                "type": event.get("type"),
                                "data": event
                            })
                
                # Try to get another event
                event_data = await event_generator.get_next_event(timeout=0.1)
            
            return events
            
        except Exception as e:
            error_msg = f"Error checking for relationship events: {e}"
            logger.error(error_msg)
            raise RelationshipError(error_msg)
    
    async def apply_crossroads_choice(self, crossroads: CrossroadsEvent, choice_index: int) -> Dict[str, Any]:
        """
        Apply a choice in a relationship crossroads using the dynamic system.
        
        Args:
            crossroads: The crossroads event
            choice_index: Index of the chosen option
            
        Returns:
            Result of the choice
            
        Raises:
            RelationshipError: If there's an issue applying the choice
        """
        try:
            options = crossroads.options
            
            if choice_index < 0 or choice_index >= len(options):
                return {"error": "Invalid choice index"}
            
            chosen_option = options[choice_index]
            
            # Apply stat effects
            if "stat_effects" in chosen_option:
                await self.apply_stat_changes(
                    chosen_option["stat_effects"],
                    f"Crossroads choice in relationship"
                )
            
            # Create interaction based on choice
            interaction_map = {
                "confront": "conflict_resolved",
                "avoid": "boundary_violated",
                "trust": "vulnerability_shared",
                "distrust": "deception_discovered",
                "embrace": "vulnerability_shared",
                "boundaries": "boundary_setting",
                "commit": "support_provided",
                "independence": "boundary_setting"
            }
            
            choice_id = chosen_option.get("id", "")
            interaction_type = interaction_map.get(choice_id, "neutral_interaction")
            
            # Process interaction
            interaction = {
                "type": interaction_type,
                "context": "crossroads_choice",
                "choice": chosen_option.get("text", ""),
                "impacts": chosen_option.get("potential_impacts", {})
            }
            
            result = await self.relationship_manager.process_interaction(
                entity1_type=crossroads.entity1_type,
                entity1_id=crossroads.entity1_id,
                entity2_type=crossroads.entity2_type,
                entity2_id=crossroads.entity2_id,
                interaction=interaction
            )
            
            # Create memory for NPC
            if crossroads.entity1_type == "npc":
                npc_id = crossroads.entity1_id
            elif crossroads.entity2_type == "npc":
                npc_id = crossroads.entity2_id
            else:
                npc_id = None
                
            if npc_id:
                await self.add_memory_to_npc(
                    npc_id,
                    f"The player made a choice about our relationship: {chosen_option['text']}",
                    importance="high",
                    tags=["crossroads", "relationship_choice"]
                )
            
            # Invalidate relationship cache
            cache_key = f"rel:{crossroads.entity1_type}:{crossroads.entity1_id}:{crossroads.entity2_type}:{crossroads.entity2_id}"
            if cache_key in self.relationship_cache:
                del self.relationship_cache[cache_key]
            
            return {
                "success": True,
                "outcome_text": chosen_option.get("outcome", "Your choice has been recorded."),
                "stat_effects": chosen_option.get("stat_effects", {}),
                "relationship_changes": result
            }
        
        except Exception as e:
            error_msg = f"Error applying crossroads choice: {e}"
            logger.error(error_msg)
            raise RelationshipError(error_msg)
    
    async def get_relationship(self, entity1_type: str, entity1_id: int, entity2_type: str, entity2_id: int) -> Dict[str, Any]:
        """
        Get the relationship between two entities using the dynamic system.
        
        Args:
            entity1_type: Type of first entity
            entity1_id: ID of first entity
            entity2_type: Type of second entity
            entity2_id: ID of second entity
            
        Returns:
            Dictionary with relationship details
            
        Raises:
            RelationshipError: If there's an issue retrieving the relationship
        """
        try:
            # Check cache first
            cache_key = f"rel:{entity1_type}:{entity1_id}:{entity2_type}:{entity2_id}"
            if cache_key in self.relationship_cache and (datetime.now() - self.relationship_cache[cache_key].get("last_updated", datetime.min)).seconds < 300:
                return self.relationship_cache[cache_key]
            
            # Get relationship state
            rel_state = await self.relationship_manager.get_relationship_state(
                entity1_type=entity1_type,
                entity1_id=entity1_id,
                entity2_type=entity2_type,
                entity2_id=entity2_id
            )
            
            # Get entity names
            entity1_name = await self._get_entity_name(entity1_type, entity1_id)
            entity2_name = await self._get_entity_name(entity2_type, entity2_id)
            
            # Use agent system for memory-enriched relationship data if one of the entities is an NPC
            relationship_memories = []
            if entity1_type == "npc" and entity1_id in self.agent_system.npc_agents:
                agent = self.agent_system.npc_agents[entity1_id]
                memory_system = await agent._get_memory_system()
                
                # Get memories related to the relationship
                query = f"{entity2_name}"
                memory_result = await memory_system.recall(
                    entity_type="npc",
                    entity_id=entity1_id,
                    query=query,
                    limit=5
                )
                
                relationship_memories.extend(memory_result.get("memories", []))
            
            elif entity2_type == "npc" and entity2_id in self.agent_system.npc_agents:
                agent = self.agent_system.npc_agents[entity2_id]
                memory_system = await agent._get_memory_system()
                
                # Get memories related to the relationship
                query = f"{entity1_name}"
                memory_result = await memory_system.recall(
                    entity_type="npc",
                    entity_id=entity2_id,
                    query=query,
                    limit=5
                )
                
                relationship_memories.extend(memory_result.get("memories", []))
            
            # Build relationship info
            relationship = {
                "entity1_type": entity1_type,
                "entity1_id": entity1_id,
                "entity1_name": entity1_name,
                "entity2_type": entity2_type,
                "entity2_id": entity2_id,
                "entity2_name": entity2_name,
                "dimensions": rel_state.dimensions.to_dict(),
                "momentum": rel_state.momentum.get_magnitude(),
                "patterns": list(rel_state.history.active_patterns),
                "archetypes": list(rel_state.active_archetypes),
                "relationship_memories": relationship_memories,
                "created_at": rel_state.created_at.isoformat(),
                "last_interaction": rel_state.last_interaction.isoformat(),
                "duration_days": rel_state.get_duration_days()
            }
            
            # Update cache
            self.relationship_cache[cache_key] = relationship
            self.relationship_cache[cache_key]["last_updated"] = datetime.now()
            
            return relationship
            
        except Exception as e:
            error_msg = f"Error getting relationship: {e}"
            logger.error(error_msg)
            raise RelationshipError(error_msg)
    
    async def _get_entity_name(self, entity_type: str, entity_id: int) -> str:
        """
        Get the name of an entity using connection for better performance.
        
        Args:
            entity_type: Entity type (e.g., "npc", "player")
            entity_id: Entity ID
            
        Returns:
            Entity name
        """
        if entity_type == "player":
            return "Player"
        elif entity_type == "npc":
            # Check cache first
            cache_key = f"npc:{entity_id}"
            if cache_key in self.npc_cache:
                return self.npc_cache[cache_key].get("npc_name", f"NPC-{entity_id}")
            
            # Use connection
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow("""
                    SELECT npc_name 
                    FROM NPCStats
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                """, entity_id, self.user_id, self.conversation_id)
                
                if row:
                    return row["npc_name"]
            
            return f"NPC-{entity_id}"
        else:
            return f"{entity_type}-{entity_id}"
    
    async def generate_multi_npc_scene(self, npc_ids: List[int], location: str = None, include_player: bool = True) -> Dict[str, Any]:
        """
        Generate a scene with multiple NPCs interacting.
        
        Args:
            npc_ids: List of NPC IDs to include in the scene
            location: Optional location for the scene
            include_player: Whether to include the player in the scene
            
        Returns:
            Scene information
            
        Raises:
            NPCSystemError: If there's an issue generating the scene
        """
        try:
            # Get current time information
            year, month, day, time_of_day = await self.get_current_game_time()
            
            # If location not provided, get a common location for the NPCs
            if not location:
                location = await self._find_common_location(npc_ids)
            
            # Initialize the scene data
            scene = {
                "location": location,
                "time_of_day": time_of_day,
                "day": day,
                "npc_ids": npc_ids,
                "include_player": include_player,
                "interactions": [],
                "description": f"Scene at {location} during {time_of_day}"
            }
            
            # Use the agent coordinator for group behavior
            coordinator = self.agent_system.coordinator
            
            # Prepare context for the scene
            context = {
                "location": location,
                "time_of_day": time_of_day,
                "day": day,
                "include_player": include_player,
                "description": f"NPCs interacting at {location} during {time_of_day}"
            }
            
            # Generate group actions using the coordinator
            action_plan = await coordinator.make_group_decisions(npc_ids, context)
            
            # Add actions to the scene
            scene["group_actions"] = action_plan.get("group_actions", [])
            scene["individual_actions"] = action_plan.get("individual_actions", {})
            
            # Create interactions
            interactions = []
            
            # Process group actions
            for group_action in action_plan.get("group_actions", []):
                npc_id = group_action.get("npc_id")
                if npc_id is None:
                    continue
                    
                action_data = group_action.get("action", {})
                
                # Get NPC name
                npc_name = await self.get_npc_name(npc_id)
                
                interaction = {
                    "type": "group_action",
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "action": action_data.get("type", "interact"),
                    "description": action_data.get("description", "does something"),
                    "target": action_data.get("target", "group")
                }
                
                interactions.append(interaction)
            
            # Process individual actions
            for npc_id, actions in action_plan.get("individual_actions", {}).items():
                npc_name = await self.get_npc_name(npc_id)
                
                for action_data in actions:
                    target = action_data.get("target")
                    target_name = action_data.get("target_name")
                    
                    interaction = {
                        "type": "individual_action",
                        "npc_id": npc_id,
                        "npc_name": npc_name,
                        "action": action_data.get("type", "interact"),
                        "description": action_data.get("description", "does something"),
                        "target": target,
                        "target_name": target_name
                    }
                    
                    interactions.append(interaction)
            
            # Add interactions to the scene
            scene["interactions"] = interactions
            
            return scene
            
        except Exception as e:
            error_msg = f"Error generating multi-NPC scene: {e}"
            logger.error(error_msg)
            raise NPCSystemError(error_msg)
    
    async def _find_common_location(self, npc_ids: List[int]) -> str:
        """
        Find a common location for a group of NPCs.
        
        Args:
            npc_ids: List of NPC IDs
            
        Returns:
            Common location name
        """
        if not npc_ids:
            return "Unknown"
        
        try:
            # Use connection for performance
            async with get_db_connection_context() as conn:
                # Get all locations for the NPCs in a single query with count
                rows = await conn.fetch("""
                    SELECT current_location, COUNT(*) as location_count
                    FROM NPCStats
                    WHERE npc_id = ANY($1) AND user_id = $2 AND conversation_id = $3
                    GROUP BY current_location
                    ORDER BY location_count DESC
                """, npc_ids, self.user_id, self.conversation_id)
                
                # Find the most common location
                if rows:
                    common_location = rows[0]["current_location"]
                    return common_location
                
                return "Unknown"
            
        except Exception as e:
            logger.error(f"Error finding common location: {e}")
            return "Unknown"
    
    async def generate_overheard_conversation(self, npc_ids: List[int], topic: str = None, about_player: bool = False) -> Dict[str, Any]:
        """
        Generate a conversation between NPCs that the player can overhear.
        
        Args:
            npc_ids: List of NPCs involved in the conversation
            topic: Optional topic of conversation
            about_player: Whether the conversation is about the player
            
        Returns:
            Conversation details
            
        Raises:
            NPCSystemError: If there's an issue generating the conversation
        """
        try:
            if len(npc_ids) < 2:
                return {"error": "Need at least 2 NPCs for a conversation"}
            
            # Get current location and time
            year, month, day, time_of_day = await self.get_current_game_time()
            location = await self._find_common_location(npc_ids)
            
            # Get NPC details (batch query for performance)
            npc_details = await self._get_multiple_npc_details(npc_ids)
            
            # Prepare context for conversation
            topic_text = topic or ("the player" if about_player else "general matters")
            context = {
                "location": location,
                "time_of_day": time_of_day,
                "topic": topic_text,
                "about_player": about_player,
                "description": f"NPCs conversing about {topic_text} at {location}"
            }
            
            # Use agent system for generating conversation
            conversation_lines = []
            
            # Generate initial statement from first NPC
            first_npc = npc_ids[0]
            agent1 = self.agent_system.npc_agents.get(first_npc)
            if not agent1:
                self.agent_system.npc_agents[first_npc] = NPCAgent(first_npc, self.user_id, self.conversation_id)
                agent1 = self.agent_system.npc_agents[first_npc]
            
            first_perception = await agent1.perceive_environment(context)
            first_action = await agent1.make_decision(first_perception)
            
            npc1_name = npc_details.get(first_npc, {}).get("npc_name", f"NPC-{first_npc}")
            first_line = {
                "npc_id": first_npc,
                "npc_name": npc1_name,
                "text": first_action.get("description", f"starts talking about {topic_text}")
            }
            conversation_lines.append(first_line)
            
            # Generate responses from other NPCs
            for npc_id in npc_ids[1:]:
                agent = self.agent_system.npc_agents.get(npc_id)
                if not agent:
                    self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
                    agent = self.agent_system.npc_agents[npc_id]
                
                # Add the previous statement to context
                response_context = context.copy()
                response_context["previous_statement"] = first_line.get("text")
                response_context["previous_speaker"] = first_line.get("npc_name")
                
                # Generate response using agent
                perception = await agent.perceive_environment(response_context)
                response_action = await agent.make_decision(perception)
                
                npc_name = npc_details.get(npc_id, {}).get("npc_name", f"NPC-{npc_id}")
                response_line = {
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "text": response_action.get("description", f"responds about {topic_text}")
                }
                conversation_lines.append(response_line)
            
            # Create memories of this conversation for each NPC
            for npc_id in npc_ids:
                # Create memory in background task to avoid blocking
                asyncio.create_task(
                    self._create_conversation_memory(npc_id, npc_ids, npc_details, topic_text, location, about_player)
                )
            
            # Format the conversation
            conversation = {
                "location": location,
                "time_of_day": time_of_day,
                "topic": topic_text,
                "about_player": about_player,
                "lines": conversation_lines
            }
            
            return conversation
            
        except Exception as e:
            error_msg = f"Error generating overheard conversation: {e}"
            logger.error(error_msg)
            raise NPCSystemError(error_msg)
    
    async def _create_conversation_memory(self, npc_id: int, all_npc_ids: List[int], 
                                       npc_details: Dict[int, Dict[str, Any]], 
                                       topic: str, location: str, about_player: bool):
        """
        Create a memory of a conversation for an NPC.
        
        Args:
            npc_id: The NPC ID
            all_npc_ids: All NPCs in the conversation
            npc_details: Details of all NPCs
            topic: Conversation topic
            location: Conversation location
            about_player: Whether the conversation was about the player
        """
        try:
            other_npcs = [n for n in all_npc_ids if n != npc_id]
            other_names = [npc_details.get(n, {}).get("npc_name", f"NPC-{n}") for n in other_npcs]
            
            memory_text = f"I had a conversation with {', '.join(other_names)} about {topic} at {location}."
            
            await self.add_memory_to_npc(
                npc_id,
                memory_text,
                importance="medium" if about_player else "low",
                tags=["conversation", "overheard"] + (["player_related"] if about_player else [])
            )
        except Exception as e:
            logger.error(f"Error creating conversation memory for NPC {npc_id}: {e}")
    
    async def _get_multiple_npc_details(self, npc_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Get details for multiple NPCs in a single database query.
        
        Args:
            npc_ids: List of NPC IDs
            
        Returns:
            Dictionary mapping NPC IDs to their details
        """
        result = {}
        
        if not npc_ids:
            return result
        
        try:
            # Use connection for performance
            async with get_db_connection_context() as conn:
                # Batch query for efficiency
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, introduced
                    FROM NPCStats
                    WHERE npc_id = ANY($1) AND user_id = $2 AND conversation_id = $3
                """, npc_ids, self.user_id, self.conversation_id)
                
                for row in rows:
                    result[row["npc_id"]] = {
                        "npc_id": row["npc_id"],
                        "npc_name": row["npc_name"],
                        "introduced": row["introduced"]
                    }
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting multiple NPC details: {e}")
            # Return whatever we could get
            return result
    
    #=================================================================
    # MEMORY MANAGEMENT
    #=================================================================
    
    async def add_memory_to_npc(self, 
                              npc_id: int, 
                              memory_text: str,
                              importance: str = "medium",
                              emotional: bool = False,
                              tags: List[str] = None) -> bool:
        """
        Add a memory to an NPC using the agent architecture.
        
        Args:
            npc_id: ID of the NPC
            memory_text: Text of the memory
            importance: Importance of the memory ("low", "medium", "high")
            emotional: Whether the memory has emotional content
            tags: List of tags for the memory
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            MemorySystemError: If there's an issue adding the memory
        """
        try:
            # Get or create NPC agent
            if npc_id not in self.agent_system.npc_agents:
                self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
            
            agent = self.agent_system.npc_agents[npc_id]
            memory_system = await agent._get_memory_system()
            
            # Create memory using the agent's memory system
            memory_id = await memory_system.remember(
                entity_type="npc",
                entity_id=npc_id,
                memory_text=memory_text,
                importance=importance,
                emotional=emotional,
                tags=tags or []
            )
            
            return memory_id is not None
            
        except Exception as e:
            error_msg = f"Error adding memory to NPC: {e}"
            logger.error(error_msg)
            raise MemorySystemError(error_msg)

    async def retrieve_relevant_memories(self, context, limit=5, memory_types=None):
            """
            Unified memory retrieval method that delegates to appropriate implementation.
            
            Args:
                context: Query context (text or dict)
                limit: Maximum memories to retrieve
                memory_types: Optional filter for memory types
                
            Returns:
                List of relevant memories
            """
            memory_types = memory_types or ["observation", "reflection", "semantic", "secondhand"]
            
            # Extract query text for cache key
            query_text = self._extract_query_text(context)
            cache_key = f"mem_{hash(query_text)}_{limit}"
            
            # Check cache
            cached_memories = self._check_memory_cache(cache_key)
            if cached_memories:
                return cached_memories
            
            try:
                # Delegate to appropriate implementation
                if MEMORY_SYSTEM_AVAILABLE:
                    memories = await self._retrieve_memories_optimized(context, limit, memory_types)
                else:
                    memories = await self._retrieve_memories_fallback(context, limit, memory_types)
                
                # Cache the result
                self._cache_memories(cache_key, memories)
                return memories
            except Exception as e:
                logger.error(f"Error retrieving memories: {e}")
                # Return emergency fallback memories
                return self._get_fallback_memories()
    
    def _extract_query_text(self, context):
        """Extract query text from context."""
        if isinstance(context, str):
            return context
        elif isinstance(context, dict):
            return context.get("text", context.get("description", ""))
        return ""
    
    def _check_memory_cache(self, cache_key):
        """Check if memories are cached and still valid."""
        if not hasattr(self, '_memory_cache'):
            self._memory_cache = {}
            self._memory_cache_times = {}
            return None
        
        if cache_key in self._memory_cache:
            cache_time = self._memory_cache_times.get(cache_key)
            if cache_time and (datetime.now() - cache_time).seconds < 30:
                return self._memory_cache[cache_key]
        return None
    
    def _cache_memories(self, cache_key, memories):
        """Cache memories for future use."""
        if not hasattr(self, '_memory_cache'):
            self._memory_cache = {}
            self._memory_cache_times = {}
        
        self._memory_cache[cache_key] = memories
        self._memory_cache_times[cache_key] = datetime.now()
        
        # Keep cache size reasonable
        if len(self._memory_cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                self._memory_cache_times.keys(),
                key=lambda k: self._memory_cache_times[k]
            )[:20]  # Remove 20 oldest
            
            for key in oldest_keys:
                if key in self._memory_cache:
                    del self._memory_cache[key]
                if key in self._memory_cache_times:
                    del self._memory_cache_times[key]
    
    def _get_fallback_memories(self):
        """Get fallback memories when retrieval fails."""
        # Create a basic fallback memory
        return [{
            "id": f"fallback_{datetime.now().timestamp()}",
            "text": "I exist in this world.",
            "type": "fallback",
            "significance": 1
        }]
    
    async def _retrieve_memories_optimized(self, context, limit=5, memory_types=None):
        """
        Optimized implementation of memory retrieval with vector search.
        
        Args:
            context: Query context (text or dict)
            limit: Maximum memories to retrieve
            memory_types: Optional filter for memory types
            
        Returns:
            List of relevant memories
        """
        # Start timing for performance metrics
        start_time = datetime.now()
        
        # Get memory system
        memory_system = await self._get_memory_system()
        
        # Process context to enhance recall
        context_dict = {}
        if isinstance(context, str):
            context_dict = {"text": context}
        elif isinstance(context, dict):
            context_dict = context
        
        # Prepare query text
        query_text = context_dict.get("text", "")
        
        # Build enhanced query incorporating context
        enhanced_query = query_text
        if context_dict:
            # Add location if available
            if "location" in context_dict:
                enhanced_query += f" location:{context_dict['location']}"
            
            # Add time info if available
            if "time_of_day" in context_dict:
                enhanced_query += f" time:{context_dict['time_of_day']}"
            
            # Add entity info if available
            if "entities_present" in context_dict:
                entities_str = " ".join(context_dict["entities_present"])
                enhanced_query += f" entities:{entities_str}"
        
        # Use vector search through memory system
        vector_results = await memory_system.recall(
            entity_type="npc",
            entity_id=None,  # Will be filled by specific NPC ID in actual use
            query=enhanced_query,
            limit=limit,
            use_vector_search=True
        )
        
        # Get memories and update timing metrics
        memories = vector_results.get("memories", [])
        
        # Update performance metrics
        elapsed = (datetime.now() - start_time).total_seconds()
        self.perf_metrics['memory_retrieval_time'].append(elapsed)
        
        return memories
    
    async def _retrieve_memories_fallback(self, context, limit=5, memory_types=None):
        """
        Fallback implementation of memory retrieval when optimized system not available.
        
        Args:
            context: Query context (text or dict)
            limit: Maximum memories to retrieve
            memory_types: Optional filter for memory types
            
        Returns:
            List of relevant memories
        """
        # Extract query text
        query_text = self._extract_query_text(context)
        
        # UPDATED: Use async connection context manager
        try:
            async with get_db_connection_context() as conn:
                # Basic search using text similarity
                rows = await conn.fetch("""
                    SELECT id, memory_text, memory_type, significance, emotional_intensity
                    FROM NPCMemories
                    WHERE user_id = $1 AND conversation_id = $2
                    ORDER BY 
                        CASE 
                            WHEN LOWER(memory_text) LIKE $3 THEN 1
                            ELSE 2
                        END,
                        significance DESC,
                        created_at DESC
                    LIMIT $4
                """, self.user_id, self.conversation_id, f"%{query_text.lower()}%", limit)
                
                memories = []
                for row in rows:
                    memories.append({
                        "id": row[0],
                        "text": row[1],
                        "type": row[2],
                        "significance": row[3],
                        "emotional_intensity": row[4]
                    })
                
                return memories
        except Exception as e:
            logger.error(f"Error in fallback memory retrieval: {e}")
            return self._get_fallback_memories()
    
    async def propagate_memory_to_related_npcs(self, 
                                           source_npc_id: int,
                                           memory_text: str,
                                           importance: str = "medium") -> bool:
        """
        Propagate a memory to NPCs related to the source NPC.
        
        Args:
            source_npc_id: ID of the source NPC
            memory_text: Text of the memory to propagate
            importance: Importance of the memory
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            MemorySystemError: If there's an issue propagating the memory
        """
        try:
            # First, add the memory to the source NPC
            await self.add_memory_to_npc(
                source_npc_id, memory_text, importance
            )
            
            # Get or create source NPC agent
            if source_npc_id not in self.agent_system.npc_agents:
                self.agent_system.npc_agents[source_npc_id] = NPCAgent(source_npc_id, self.user_id, self.conversation_id)
            
            # Find related NPCs using the new relationship system
            related_npcs = []
            source_name = "Unknown"
            
            # Get all NPCs
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
                
                source_name_row = [r for r in rows if r["npc_id"] == source_npc_id]
                if source_name_row:
                    source_name = source_name_row[0]["npc_name"]
                
                # Check relationships with each NPC
                for row in rows:
                    if row["npc_id"] != source_npc_id:
                        # Get relationship state
                        rel_state = await self.relationship_manager.get_relationship_state(
                            entity1_type="npc",
                            entity1_id=source_npc_id,
                            entity2_type="npc",
                            entity2_id=row["npc_id"]
                        )
                        
                        # Check if relationship is strong enough for memory propagation
                        if (rel_state.dimensions.trust > 30 or 
                            rel_state.dimensions.intimacy > 30 or
                            rel_state.dimensions.frequency > 40):
                            
                            # Calculate propagation strength based on relationship
                            propagation_strength = max(
                                rel_state.dimensions.trust / 100,
                                rel_state.dimensions.intimacy / 100,
                                rel_state.dimensions.frequency / 100
                            )
                            
                            related_npcs.append((row["npc_id"], propagation_strength))
            
            # Propagate memory to each related NPC
            propagation_tasks = []
            for npc_id, strength in related_npcs:
                # Add task to propagate memory
                propagation_tasks.append(
                    self._propagate_single_memory_dynamic(npc_id, source_name, memory_text, strength, importance)
                )
            
            # Run all propagation tasks concurrently
            if propagation_tasks:
                await asyncio.gather(*propagation_tasks)
            
            return True
            
        except Exception as e:
            error_msg = f"Error propagating memory: {e}"
            logger.error(error_msg)
            raise MemorySystemError(error_msg)
    
    async def _propagate_single_memory_dynamic(self, npc_id: int, source_name: str, memory_text: str, 
                                             propagation_strength: float, importance: str):
        """
        Propagate a single memory to an NPC with dynamic distortion.
        
        Args:
            npc_id: Target NPC ID
            source_name: Source NPC name
            memory_text: Memory text
            propagation_strength: Strength of propagation (0.0-1.0)
            importance: Memory importance
        """
        try:
            # Higher relationship strength means more accurate propagation
            if propagation_strength > 0.7:
                propagated_text = f"I heard from {source_name} that {memory_text}"
            else:
                # Add potential distortion
                words = memory_text.split()
                if len(words) > 5 and propagation_strength < 0.5:
                    # More distortion for weaker relationships
                    distortion_count = int((1 - propagation_strength) * 3)  # 1-3 words
                    for _ in range(min(distortion_count, len(words) // 3)):
                        if len(words) > 3:
                            idx = random.randint(0, len(words) - 1)
                            
                            # Replace with similar word or opposite
                            replacements = {
                                "good": "nice", "bad": "terrible", "happy": "pleased",
                                "sad": "unhappy", "angry": "upset", "large": "big",
                                "small": "tiny", "important": "critical", "interesting": "fascinating"
                            }
                            
                            if words[idx].lower() in replacements:
                                words[idx] = replacements[words[idx].lower()]
                
                distorted_text = " ".join(words)
                propagated_text = f"I heard from {source_name} that {distorted_text}"
            
            # Reduce importance for propagated memories
            new_importance = "low" if importance == "medium" else "medium" if importance == "high" else "low"
            
            # Create the propagated memory
            await self.add_memory_to_npc(
                npc_id, 
                propagated_text, 
                new_importance,
                tags=["hearsay", "secondhand", "rumor"]
            )
        except Exception as e:
            logger.error(f"Error propagating memory to NPC {npc_id}: {e}")
    
    #=================================================================
    # MASK AND EMOTIONAL STATE MANAGEMENT
    #=================================================================
    
    async def get_npc_mask(self, npc_id: int) -> Dict[str, Any]:
        """
        Get mask information for an NPC.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Dictionary with mask information
            
        Raises:
            MemorySystemError: If there's an issue retrieving the mask
        """
        try:
            # Get or create NPC agent
            if npc_id not in self.agent_system.npc_agents:
                self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
            
            agent = self.agent_system.npc_agents[npc_id]
            mask_manager = await agent._get_mask_manager()
            
            # Get mask using the agent's mask manager
            result = await mask_manager.get_npc_mask(npc_id)
            
            return result
            
        except Exception as e:
            error_msg = f"Error getting NPC mask: {e}"
            logger.error(error_msg)
            raise MemorySystemError(error_msg)
    
    async def generate_mask_slippage(self, 
                                       npc_id: int, 
                                       trigger: str = None,
                                       severity: int = None) -> Dict[str, Any]:
        """
        Generate a mask slippage event for an NPC.
        
        Args:
            npc_id: ID of the NPC
            trigger: What triggered the slippage
            severity: Severity level of the slippage
            
        Returns:
            Dictionary with slippage information
            
        Raises:
            MemorySystemError: If there's an issue generating the slippage
        """
        try:
            # Get or create NPC agent
            if npc_id not in self.agent_system.npc_agents:
                self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
            
            agent = self.agent_system.npc_agents[npc_id]
            memory_system = await agent._get_memory_system()
            
            # Generate mask slippage using the memory system
            result = await memory_system.reveal_npc_trait(
                npc_id=npc_id,
                trigger=trigger,
                severity=severity
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Error generating mask slippage: {e}"
            logger.error(error_msg)
            raise MemorySystemError(error_msg)
    
    async def update_npc_emotional_state(self, 
                                      npc_id: int, 
                                      emotion: str,
                                      intensity: float) -> Dict[str, Any]:
        """
        Update an NPC's emotional state.
        
        Args:
            npc_id: ID of the NPC
            emotion: Primary emotion
            intensity: Intensity of the emotion (0.0-1.0)
            
        Returns:
            Updated emotional state
            
        Raises:
            MemorySystemError: If there's an issue updating the emotional state
        """
        try:
            # Get or create NPC agent
            if npc_id not in self.agent_system.npc_agents:
                self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
            
            agent = self.agent_system.npc_agents[npc_id]
            memory_system = await agent._get_memory_system()
            
            # Update emotional state using the memory system
            result = await memory_system.update_npc_emotion(
                npc_id=npc_id,
                emotion=emotion,
                intensity=intensity
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Error updating NPC emotional state: {e}"
            logger.error(error_msg)
            raise MemorySystemError(error_msg)
    
    async def get_npc_emotional_state(self, npc_id: int) -> Dict[str, Any]:
        """
        Get an NPC's current emotional state.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Current emotional state
            
        Raises:
            MemorySystemError: If there's an issue retrieving the emotional state
        """
        try:
            # Get or create NPC agent
            if npc_id not in self.agent_system.npc_agents:
                self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
            
            agent = self.agent_system.npc_agents[npc_id]
            memory_system = await agent._get_memory_system()
            
            # Get emotional state using the memory system
            result = await memory_system.get_npc_emotion(npc_id)
            
            return result
            
        except Exception as e:
            error_msg = f"Error getting NPC emotional state: {e}"
            logger.error(error_msg)
            raise MemorySystemError(error_msg)
    
    #=================================================================
    # TIME MANAGEMENT AND NPC ACTIVITIES
    #=================================================================
    
    async def get_current_game_time(self) -> Tuple[int, int, int, str]:
        """
        Get the current game time.
        
        Returns:
            Tuple of (year, month, day, time_of_day)
            
        Raises:
            TimeSystemError: If there's an issue getting the current time
        """
        try:
            # Using get_current_time
            time_info = get_current_time(self.user_id, self.conversation_id)
            
            # Log using TIME_PHASES
            current_phase_idx = TIME_PHASES.index(time_info[3]) if time_info[3] in TIME_PHASES else 0
            next_phase_idx = (current_phase_idx + 1) % len(TIME_PHASES)
            logger.info(f"Current time phase: {time_info[3]}, Next phase: {TIME_PHASES[next_phase_idx]}")
            
            return time_info
            
        except Exception as e:
            error_msg = f"Error getting current game time: {e}"
            logger.error(error_msg)
            raise TimeSystemError(error_msg)
    
    async def set_game_time(self, year: int, month: int, day: int, time_of_day: str) -> bool:
        """
        Set the game time.
        
        Args:
            year: Year to set
            month: Month to set
            day: Day to set
            time_of_day: Time of day to set
            
        Returns:
            True if successful
            
        Raises:
            TimeSystemError: If there's an issue setting the time
        """
        try:
            # Validate time_of_day using TIME_PHASES
            if time_of_day not in TIME_PHASES:
                logger.warning(f"Invalid time phase '{time_of_day}'. Using {TIME_PHASES[0]} instead.")
                time_of_day = TIME_PHASES[0]
            
            # Using set_current_time
            set_current_time(self.user_id, self.conversation_id, year, month, day, time_of_day)
            
            # Using update_npc_schedules_for_time
            update_npc_schedules_for_time(self.user_id, self.conversation_id, day, time_of_day)
            
            return True
            
        except Exception as e:
            error_msg = f"Error setting game time: {e}"
            logger.error(error_msg)
            raise TimeSystemError(error_msg)
    
    async def advance_time_with_activity(self, activity_type: str) -> Dict[str, Any]:
        """
        Advance time based on an activity type.
        
        Args:
            activity_type: Type of activity
            
        Returns:
            Dictionary with results including any events that occurred
            
        Raises:
            TimeSystemError: If there's an issue advancing time
        """
        try:
            # Using advance_time_with_events
            result = await advance_time_with_events(
                self.user_id, self.conversation_id, activity_type
            )
            
            # After time advances, process scheduled activities for all NPCs
            if result.get("time_advanced", False):
                # Process in background task to avoid blocking the main thread
                asyncio.create_task(self.process_npc_scheduled_activities())
                
                # Check for relationship events after time advance
                asyncio.create_task(self._check_relationship_events_after_time())
            
            return result
            
        except Exception as e:
            error_msg = f"Error advancing time with activity: {e}"
            logger.error(error_msg)
            raise TimeSystemError(error_msg)
    
    async def _check_relationship_events_after_time(self):
        """Check for relationship events after time advances."""
        try:
            # Drain any pending relationship events
            events = await event_generator.drain_events(max_events=10)
            
            # Process each event
            for event_data in events:
                if event_data and "event" in event_data:
                    # Log the event for now
                    logger.info(f"Relationship event occurred: {event_data['event'].get('type', 'unknown')}")
                    # You could store these events for later retrieval or process them immediately
        except Exception as e:
            logger.error(f"Error checking relationship events after time advance: {e}")
    
    async def process_player_activity(
        self, 
        player_input: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Enhanced player activity processing with parallel NPC notifications.
        
        Args:
            player_input: Player's input text
            context: Additional context
            
        Returns:
            Dictionary with processing results
        """
        # Create base context if not provided
        context_obj = context or {}
        
        # Create standardized player action
        player_action = {
            "type": "activity",
            "description": player_input,
            "text": player_input,
            "context": context_obj
        }
        
        try:
            # Use NLP-enhanced activity detection
            activity_type = await self._detect_activity_type(player_input, context_obj)
            player_action["type"] = activity_type
            
            # Get current location for NPC notifications
            current_location = context_obj.get("location")
            
            if current_location:
                # Get NPCs at current location in a single query
                nearby_npcs = await self._fetch_npcs_at_location(current_location)
                
                # Prioritize NPCs for notifications based on relationship and traits
                prioritized_npcs = await self._prioritize_npcs_for_notification(
                    nearby_npcs, 
                    activity_type,
                    player_action
                )
                
                # Create perception tasks for NPCs (parallel processing)
                perception_tasks = []
                
                # Process in batches for better performance
                batch_size = 5
                for i in range(0, len(prioritized_npcs), batch_size):
                    batch = prioritized_npcs[i:i+batch_size]
                    
                    # Prepare perception contexts for this batch
                    batch_tasks = []
                    for npc_data in batch:
                        npc_id = npc_data["npc_id"]
                        
                        # Get or create agent
                        if npc_id not in self.agent_system.npc_agents:
                            self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
                        
                        agent = self.agent_system.npc_agents[npc_id]
                        
                        # Create perception context
                        perception_context = {
                            "location": current_location,
                            "player_action": player_action,
                            "description": f"Player {player_input}",
                            "npc_priority": npc_data.get("priority", 1.0)
                        }
                        
                        # Add perception task
                        batch_tasks.append(
                            self._process_npc_perception(agent, npc_id, perception_context, player_action)
                        )
                    
                    # Process batch concurrently
                    await asyncio.gather(*batch_tasks)
            
            # Return result with calculated activity progression
            return {
                "activity_type": activity_type,
                "time_advanced": self._should_advance_time(activity_type),
                "would_advance": True if activity_type in ["sleep", "rest", "eat", "travel", "wait"] else False,
                "activity_progression": self._get_activity_progression(activity_type, player_input)
            }
            
        except Exception as e:
            logger.error(f"Error processing player activity: {e}")
            # Provide fallback classification 
            return {
                "activity_type": "unknown_activity",
                "time_advanced": False,
                "error": str(e)
            }
    
    async def _detect_activity_type(self, player_input: str, context: Dict[str, Any]) -> str:
        """
        Detect activity type using NLP techniques and context.
        
        Args:
            player_input: Player's input text
            context: Additional context
            
        Returns:
            Detected activity type
        """
        # Check for context-provided activity first
        if "activity_type" in context:
            return context["activity_type"]
        
        # Use more sophisticated detection with classification
        lower_input = player_input.lower()
        
        # Check exact activity mentions first
        if "sleep" in lower_input or "rest" in lower_input or "nap" in lower_input:
            return "sleep"
        elif "eat" in lower_input or "food" in lower_input or "meal" in lower_input:
            return "eat"
        elif "wait" in lower_input:
            return "wait"
        
        # Check for movement/travel activities
        if any(word in lower_input for word in ["go", "walk", "move", "travel", "head"]):
            return "travel"
        
        # Check for interaction patterns
        if "talk" in lower_input or "ask" in lower_input or "tell" in lower_input:
            return "conversation"
        
        # Look for action words
        action_words = ["look", "examine", "search", "find", "take", "grab", "use"]
        for action in action_words:
            if action in lower_input:
                return "action"
        
        # Fall back to activity manager for complex cases
        activity_result = await self.activity_manager.process_activity(
            self.user_id, self.conversation_id, player_input, context
        )
        
        return activity_result.get("activity_type", "generic_activity")
    
    async def _fetch_npcs_at_location(self, location: str) -> List[Dict[str, Any]]:
        """
        Fetch NPCs at a given location with a single optimized query.
        
        Args:
            location: Location to check
            
        Returns:
            List of NPCs at the location
        """
        # UPDATED: Use async connection context manager
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT 
                    n.npc_id, n.npc_name, 
                    n.dominance, n.cruelty, 
                    n.current_location
                FROM NPCStats n
                WHERE n.user_id = $1 
                  AND n.conversation_id = $2 
                  AND n.current_location = $3
                ORDER BY n.dominance DESC
                LIMIT 10
            """, self.user_id, self.conversation_id, location)
            
            return [dict(row) for row in rows]
    
    async def _prioritize_npcs_for_notification(
        self,
        npcs: List[Dict[str, Any]],
        activity_type: str,
        player_action: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Prioritize NPCs for notifications based on relevance to the activity.
        
        Args:
            npcs: List of NPCs to prioritize
            activity_type: Type of activity
            player_action: Player's action
            
        Returns:
            Prioritized list of NPCs
        """
        # Calculate priority scores for each NPC
        for npc in npcs:
            priority = 1.0  # Base priority
            
            # Dominant NPCs pay more attention to player activities
            if npc.get("dominance", 0) > 70:
                priority += 0.5
            
            # Cruel NPCs pay attention to potentially vulnerable actions
            if npc.get("cruelty", 0) > 70 and activity_type in ["sleep", "vulnerable_position"]:
                priority += 0.7
            
            # Default priority for active NPCs to avoid missing notifications
            if priority < 0.5:
                priority = 0.5
                
            npc["priority"] = priority
        
        # Sort by priority (highest first)
        return sorted(npcs, key=lambda x: x.get("priority", 0), reverse=True)
    
    def _should_advance_time(self, activity_type: str) -> bool:
        """
        Determine if an activity should advance time.
        
        Args:
            activity_type: Type of activity
            
        Returns:
            Whether time should advance
        """
        # List of activities that advance time
        time_advancing = [
            "sleep", "rest", "eat", "travel", "wait", 
            "extended_activity", "training", "exercise"
        ]
        
        return activity_type in time_advancing
    
    def _get_activity_progression(self, activity_type: str, player_input: str) -> float:
        """
        Calculate activity progression percentage.
        
        Args:
            activity_type: Type of activity
            player_input: Player's input text
            
        Returns:
            Progression value (0.0-1.0)
        """
        lower_input = player_input.lower()
        
        # Certain keywords indicate progression level
        if "completely" in lower_input or "fully" in lower_input:
            return 1.0
        elif "partially" in lower_input or "start" in lower_input:
            return 0.3
        elif "half" in lower_input or "midway" in lower_input:
            return 0.5
        elif "nearly" in lower_input or "almost" in lower_input:
            return 0.8
        
        # Default progression based on activity type
        activity_defaults = {
            "sleep": 1.0,    # Sleep fully advances time
            "rest": 0.5,     # Rest partially advances
            "eat": 0.3,      # Eating is quick
            "travel": 0.7,   # Travel takes time
            "wait": 1.0      # Wait fully advances
        }
        
        return activity_defaults.get(activity_type, 0.3)  # Default to 30% for unknown activities
    
    async def _process_npc_perception(self, agent, npc_id, perception_context, player_action):
        """
        Process an NPC's perception and memory of a player action.
        
        Args:
            agent: The NPC agent
            npc_id: NPC ID
            perception_context: Perception context
            player_action: Player's action
        """
        try:
            # Update agent's perception
            await agent.perceive_environment(perception_context)
            
            # Create a memory of observing the player's action
            memory_system = await agent._get_memory_system()
            await memory_system.remember(
                entity_type="npc",
                entity_id=npc_id,
                memory_text=f"I observed the player {player_action.get('description', 'doing something')} at {perception_context.get('location', 'somewhere')}",
                importance="low",  # Low importance for routine observations
                tags=["player_observation", player_action.get("type", "unknown")]
            )
        except Exception as e:
            logger.error(f"Error processing NPC perception for NPC {npc_id}: {e}")
    
    async def process_npc_scheduled_activities(self) -> Dict[str, Any]:
        """
        Process scheduled activities for all NPCs using enhanced systems.
        """
        logger.info("Processing scheduled activities with enhanced systems")
        
        try:
            # Get current time information for context
            year, month, day, time_of_day = await self.get_current_game_time()
            
            # Create base context for all NPCs
            base_context = {
                "year": year,
                "month": month,
                "day": day,
                "time_of_day": time_of_day,
                "activity_type": "scheduled"
            }
            
            # Get all NPCs with their current locations
            npc_data = await self._fetch_all_npc_data_for_activities()
            
            # Count total NPCs to process
            total_npcs = len(npc_data)
            if total_npcs == 0:
                return {"npc_responses": [], "count": 0}
                
            logger.info(f"Processing scheduled activities for {total_npcs} NPCs")
            
            # Process in batches for better performance
            npc_responses = []
            behavior_updates = {}
            learning_updates = {}
            
            # Process each NPC
            for npc_id, data in npc_data.items():
                try:
                    # Get or create NPC agent
                    if npc_id not in self.agent_system.npc_agents:
                        self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
                    
                    agent = self.agent_system.npc_agents[npc_id]
                    
                    # Enhance with advanced systems if needed
                    if not hasattr(agent, "perception_system"):
                        agent.perception_system = EnvironmentPerception(npc_id, self.user_id, self.conversation_id)
                    
                    if not hasattr(agent, "behavior_evolution"):
                        agent.behavior_evolution = BehaviorEvolution(self.user_id, self.conversation_id)
                    
                    if not hasattr(agent, "learning_system"):
                        agent.learning_system = NPCLearningAdaptation(self.user_id, self.conversation_id, npc_id)
                        await agent.learning_system.initialize()
                    
                    # Perform perception
                    perception_context = {
                        "location": data["location"],
                        "time_of_day": base_context["time_of_day"],
                        "description": f"Scheduled activity at {data['location']}"
                    }
                    
                    await agent.perception_system.perceive_environment(perception_context)
                    
                    # Process behavior evolution
                    behavior_result = await agent.behavior_evolution.evaluate_npc_scheming(npc_id)
                    behavior_updates[npc_id] = behavior_result
                    
                    # Process learning
                    learning_result = await agent.learning_system.process_recent_memories_for_learning(days=1)
                    learning_updates[npc_id] = learning_result
                    
                    # Perform scheduled activity
                    activity_result = await agent.perform_scheduled_activity()
                    
                    if activity_result:
                        # Format result for output
                        formatted_result = {
                            "npc_id": npc_id,
                            "npc_name": data["name"],
                            "location": data["location"],
                            "action": activity_result.get("action", {}),
                            "result": activity_result.get("result", {}),
                            "behavior_update": behavior_result,
                            "learning_update": learning_result
                        }
                        
                        npc_responses.append(formatted_result)
                
                except Exception as e:
                    logger.error(f"Error processing activity for NPC {npc_id}: {e}")
            
            # Apply daily drift to relationships
            await self.relationship_manager.apply_daily_drift()
            
            # Return combined results
            return {
                "npc_responses": npc_responses,
                "behavior_updates": behavior_updates,
                "learning_updates": learning_updates,
                "count": len(npc_responses)
            }
                
        except Exception as e:
            error_msg = f"Error processing NPC scheduled activities: {e}"
            logger.error(error_msg)
            raise NPCSystemError(error_msg)
    
    async def _fetch_all_npc_data_for_activities(self) -> Dict[int, Dict[str, Any]]:
        """Fetch all NPCs with basic data needed for activities."""
        npc_data = {}
        
        try:
            # UPDATED: Use async connection context manager
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location, schedule
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
                
                for row in rows:
                    schedule = row["schedule"]
                    if isinstance(schedule, str):
                        try:
                            schedule = json.loads(schedule)
                        except:
                            schedule = {}
                    
                    npc_data[row["npc_id"]] = {
                        "name": row["npc_name"],
                        "location": row["current_location"],
                        "schedule": schedule
                    }
                
                return npc_data
        except Exception as e:
            logger.error(f"Error fetching NPC data for activities: {e}")
            return {}
    
    async def _process_coordination_activities(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process NPC coordination activities - handling group interactions and influences.
        This separates the coordination logic from the main activity processing to avoid recursion.
        
        Args:
            context: Base context for activities
            
        Returns:
            List of coordination activity responses
        """
        logger.info("Processing NPC coordination activities")
        
        try:
            # Get NPCs that are in the same location (for potential group activities)
            location_groups = await self._get_npcs_by_location()
            
            # Process each location group for potential interactions
            coordination_responses = []
            
            for location, npc_ids in location_groups.items():
                # Only process groups with multiple NPCs
                if len(npc_ids) < 2:
                    continue
                    
                # Find dominant NPCs in each location who might initiate group activities
                dominant_npcs = await self._find_dominant_npcs(npc_ids)
                
                for dom_npc_id in dominant_npcs:
                    # Create group context
                    group_context = context.copy()
                    group_context["location"] = location
                    group_context["group_members"] = npc_ids
                    group_context["initiator_id"] = dom_npc_id
                    
                    # Check if this group should interact
                    if await self._should_group_interact(dom_npc_id, npc_ids, group_context):
                        # Use coordinator to handle the group interaction
                        group_result = await self.agent_system.coordinator.make_group_decisions(
                            npc_ids, 
                            group_context
                        )
                        
                        if group_result:
                            coordination_responses.append({
                                "type": "group_interaction",
                                "location": location,
                                "initiator": dom_npc_id,
                                "participants": npc_ids,
                                "result": group_result
                            })
            
            return coordination_responses
            
        except Exception as e:
            logger.error(f"Error in coordination activities: {e}")
            return []
    
    async def _get_npcs_by_location(self) -> Dict[str, List[int]]:
        """Group NPCs by their current location."""
        location_groups = {}
        
        try:
            # UPDATED: Use async connection context manager
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT npc_id, current_location
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                    AND current_location IS NOT NULL
                """, self.user_id, self.conversation_id)
                
                for row in rows:
                    npc_id = row["npc_id"]
                    location = row["current_location"]
                    
                    if location not in location_groups:
                        location_groups[location] = []
                        
                    location_groups[location].append(npc_id)
                
                return location_groups
        except Exception as e:
            logger.error(f"Error grouping NPCs by location: {e}")
            return {}
    
    async def _find_dominant_npcs(self, npc_ids: List[int]) -> List[int]:
        """Find NPCs with high dominance that might initiate group activities."""
        dominant_npcs = []
        
        try:
            # UPDATED: Use async connection context manager
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT npc_id 
                    FROM NPCStats
                    WHERE npc_id = ANY($1) AND user_id = $2 AND conversation_id = $3
                    AND dominance > 65
                    ORDER BY dominance DESC
                """, npc_ids, self.user_id, self.conversation_id)
                
                dominant_npcs = [row["npc_id"] for row in rows]
                
                return dominant_npcs
        except Exception as e:
            logger.error(f"Error finding dominant NPCs: {e}")
            return []
    
    async def _should_group_interact(self, initiator_id: int, group_members: List[int], context: Dict[str, Any]) -> bool:
        """Determine if a group interaction should occur based on social dynamics."""
        # Base chance - 30%
        interaction_chance = 0.3
        
        try:
            # Check time of day - certain times are more social
            time_of_day = context.get("time_of_day", "")
            if time_of_day == "evening":
                interaction_chance += 0.2  # More group interactions in evening
            
            # Check if initiator has high dominance and cruelty (femdom context)
            # UPDATED: Use async connection context manager
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow("""
                    SELECT dominance, cruelty
                    FROM NPCStats
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                """, initiator_id, self.user_id, self.conversation_id)
                
                if row:
                    dominance = row["dominance"]
                    cruelty = row["cruelty"]
                    
                    # Highly dominant NPCs more likely to initiate group dynamics
                    if dominance > 80:
                        interaction_chance += 0.2
                        
                    # Cruel dominants enjoy group discipline scenes
                    if cruelty > 70:
                        interaction_chance += 0.15
            
            # Random roll against the calculated chance
            return random.random() < interaction_chance
        except Exception as e:
            logger.error(f"Error determining if group should interact: {e}")
            return False
    
    async def _process_single_npc_activity(self, npc_id, data, base_context):
        """
        Process the scheduled activity for a single NPC.
        
        Args:
            npc_id: NPC ID
            data: NPC data including name and location
            base_context: Base context for all NPCs
            
        Returns:
            Formatted activity result
        """
        try:
            # Get or create NPC agent
            if npc_id not in self.agent_system.npc_agents:
                self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
            
            agent = self.agent_system.npc_agents[npc_id]
            
            # Create context with location
            context = base_context.copy()
            context["location"] = data["location"]
            context["description"] = f"Scheduled activity at {data['location']} during {base_context['time_of_day']}"
            
            # Use agent to perceive environment and perform activity
            await agent.perceive_environment(context)
            activity_result = await agent.perform_scheduled_activity()
            
            if activity_result:
                # Format result for output
                formatted_result = {
                    "npc_id": npc_id,
                    "npc_name": data["name"],
                    "location": data["location"],
                    "action": activity_result.get("action", {}),
                    "result": activity_result.get("result", {})
                }
                
                # Add emotional impact data from agent if available
                memory_system = await agent._get_memory_system()
                emotional_state = await memory_system.get_npc_emotion(npc_id)
                
                if emotional_state:
                    formatted_result["emotional_state"] = emotional_state
                
                return formatted_result
            
            return None
        
        except Exception as e:
            logger.error(f"Error processing activity for NPC {npc_id}: {e}")
            return None
    
    async def run_memory_maintenance(self) -> Dict[str, Any]:
        """
        Run memory maintenance for all agents with enhanced processing.
        
        Returns:
            Results of maintenance operations
            
        Raises:
            MemorySystemError: If there's an issue running maintenance
        """
        try:
            # Run system-wide maintenance first
            system_results = await self.agent_system.run_maintenance()
            
            # Run individual agent maintenance for any that need special processing
            individual_results = {}
            
            # Batch database query for performance
            active_agents = {}
            important_relationships = set()
            
            # Get active and important NPCs in a single query
            # UPDATED: Use async connection context manager
            async with get_db_connection_context() as conn:
                # Find NPCs with important relationships using the dynamic system
                # Get all NPCs first
                npc_rows = await conn.fetch("""
                    SELECT npc_id
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
                
                # Check relationships for each NPC
                for row in npc_rows:
                    npc_id = row["npc_id"]
                    
                    # Get relationship with player
                    rel_state = await self.relationship_manager.get_relationship_state(
                        entity1_type="npc",
                        entity1_id=npc_id,
                        entity2_type="player",
                        entity2_id=self.user_id
                    )
                    
                    # Check if relationship is important
                    if (rel_state.dimensions.trust > 50 or 
                        rel_state.dimensions.affection > 50 or
                        rel_state.dimensions.intimacy > 50):
                        important_relationships.add(npc_id)
                
                # Find recently active NPCs
                act_rows = await conn.fetch("""
                    SELECT DISTINCT npc_id
                    FROM NPCAgentState
                    WHERE user_id = $1 AND conversation_id = $2
                    AND last_updated > NOW() - INTERVAL '1 day'
                """, self.user_id, self.conversation_id)
                
                for row in act_rows:
                    active_agents[row["npc_id"]] = True
            
            # Process selective maintenance for high-priority NPCs
            maintenance_tasks = []
            
            for npc_id, agent in self.agent_system.npc_agents.items():
                # Only run individual maintenance for NPCs that need it
                if npc_id in important_relationships or npc_id in active_agents:
                    # Add maintenance task
                    maintenance_tasks.append(self._run_agent_maintenance(npc_id, agent))
            
            # Run maintenance tasks concurrently
            maintenance_results = await asyncio.gather(*maintenance_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(maintenance_results):
                if isinstance(result, Exception):
                    logger.error(f"Error in agent maintenance: {result}")
                elif result:  # Skip None results
                    npc_id, maintenance_result = result
                    individual_results[npc_id] = maintenance_result
            
            # Combine results
            combined_results = {
                "system_maintenance": system_results,
                "individual_maintenance": individual_results,
                "maintenance_count": len(individual_results)
            }
            
            return combined_results
            
        except Exception as e:
            error_msg = f"Error running memory maintenance: {e}"
            logger.error(error_msg)
            raise MemorySystemError(error_msg)
    
    async def _run_agent_maintenance(self, npc_id, agent):
        """
        Run maintenance for a single agent.
        
        Args:
            npc_id: NPC ID
            agent: The NPC agent
            
        Returns:
            Tuple of (npc_id, maintenance_result)
        """
        try:
            maintenance_result = await agent.run_memory_maintenance()
            return (npc_id, maintenance_result)
        except Exception as e:
            logger.error(f"Error running maintenance for NPC {npc_id}: {e}")
            return (npc_id, {"error": str(e)})
    
    #=================================================================
    # STATS AND PROGRESSION
    #=================================================================
    
    async def apply_stat_changes(self, changes: Dict[str, int], cause: str = "") -> bool:
        """
        Apply multiple stat changes to the player.
        
        Args:
            changes: Dictionary of stat changes
            cause: Reason for the changes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Using apply_stat_change
            result = apply_stat_change(self.user_id, self.conversation_id, changes, cause)
            
            # UPDATED: Use async connection context manager
            async with get_db_connection_context() as conn:
                # Get all current values in a single query
                stat_names = list(changes.keys())
                placeholders = ", ".join(f"{name}" for name in stat_names)
                query = f"""
                    SELECT {placeholders}
                    FROM PlayerStats 
                    WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
                """
                
                row = await conn.fetchrow(query, self.user_id, self.conversation_id)
                
                if row:
                    # Record stat changes in a batch operation
                    for stat_name, change_value in changes.items():
                        old_value = getattr(row, stat_name, 0)
                        new_value = old_value + change_value
                        
                        # Using record_stat_change_event
                        record_stat_change_event(
                            self.user_id, self.conversation_id, 
                            stat_name, old_value, new_value, cause
                        )
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying stat changes: {e}")
            return False
    
    async def apply_activity_effects(self, activity_name: str, intensity: float = 1.0) -> bool:
        """
        Apply stat changes based on a specific activity.
        
        Args:
            activity_name: Name of the activity
            intensity: Intensity multiplier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Using apply_activity_effects
            result = apply_activity_effects(
                self.user_id, self.conversation_id, activity_name, intensity
            )
            return result
            
        except Exception as e:
            logger.error(f"Error applying activity effects: {e}")
            return False
    
    async def get_player_current_tier(self, stat_name: str) -> Optional[Dict[str, Any]]:
        """
        Determine which tier a player is in for a given stat.
        
        Args:
            stat_name: Name of the stat
            
        Returns:
            Threshold dictionary or None
        """
        try:
            # Using get_player_current_tier
            result = get_player_current_tier(
                self.user_id, self.conversation_id, stat_name
            )
            
            # Using STAT_THRESHOLDS
            if result and stat_name in STAT_THRESHOLDS:
                logger.info(f"Player tier for {stat_name}: {result['name']} (level {result['level']})")
                logger.info(f"Possible tiers for {stat_name}: {[tier['name'] for tier in STAT_THRESHOLDS[stat_name]]}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting player current tier: {e}")
            return None
    
    async def check_for_combination_triggers(self) -> List[Dict[str, Any]]:
        """
        Check if player stats trigger any special combination states.
        
        Returns:
            List of triggered combinations
        """
        try:
            # Using check_for_combination_triggers
            result = check_for_combination_triggers(
                self.user_id, self.conversation_id
            )
            
            # Using STAT_COMBINATIONS
            if result:
                logger.info(f"Triggered combinations: {[combo['name'] for combo in result]}")
                logger.info(f"Available combinations: {[combo['name'] for combo in STAT_COMBINATIONS]}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking for combination triggers: {e}")
            return []
    
    #=================================================================
    # HIGH-LEVEL INTERACTION HANDLERS
    #=================================================================
         
    async def handle_npc_interaction(
        self, 
        npc_id: int, 
        interaction_type: str,
        player_input: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Handle interaction between player and NPC with improved systems.
        
        Args:
            npc_id: ID of the NPC
            interaction_type: Type of interaction
            player_input: Player's input text
            context: Additional context information
            
        Returns:
            Dictionary with results including events and stat changes
        """
        cache_key = f"interaction:{npc_id}:{interaction_type}:{hash(player_input)}"
        
        # Check cache for similar recent interactions
        cached_result = self._check_interaction_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Get or create NPC agent
            if npc_id not in self.agent_system.npc_agents:
                self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
            
            agent = self.agent_system.npc_agents[npc_id]
            
            # Create player action object
            player_action = {
                "type": interaction_type,
                "description": player_input,
                "target_npc_id": npc_id
            }
            
            # Prepare context
            context_obj = context or {}
            context_obj["interaction_type"] = interaction_type
            
            # Environment perception
            if not hasattr(agent, "perception_system"):
                agent.perception_system = EnvironmentPerception(npc_id, self.user_id, self.conversation_id)
            
            perception_context = {
                "location": context_obj.get("location"),
                "time_of_day": context_obj.get("time_of_day"),
                "description": f"Player says: {player_input}",
                "player_action": player_action
            }
            
            perception_result = await agent.perception_system.perceive_environment(perception_context)
            
            # Process through agent system
            npc_response = await self.agent_system.handle_player_action(player_action, context_obj)
            
            # Record for learning and adaptation
            if not hasattr(agent, "learning_system"):
                agent.learning_system = NPCLearningAdaptation(self.user_id, self.conversation_id, npc_id)
                await agent.learning_system.initialize()
            
            interaction_details = {
                "summary": f"Player said: {player_input}",
                "type": interaction_type
            }
            
            learning_result = await agent.learning_system.record_player_interaction(
                interaction_type=interaction_type,
                interaction_details=interaction_details,
                player_response={"text": player_input}
            )
            
            # Update beliefs if appropriate
            if not hasattr(agent, "belief_system"):
                enhance_npc_with_belief_system(agent)
            
            # Only form beliefs for significant interactions
            if interaction_type in ["command", "question", "statement", "emotional"]:
                belief_result = await agent.form_belief_about(
                    observation=f"Player {interaction_type}: {player_input}",
                    factuality=0.8
                )
            else:
                belief_result = None
            
            # Update relationship
            npc_action = npc_response.get("npc_responses", [{}])[0] if npc_response.get("npc_responses") else {}
            await self.update_relationship_from_interaction(
                npc_id=npc_id,
                player_action=player_action,
                npc_action=npc_action,
                context=context_obj
            )
            
            # Calculate stat changes based on interaction
            stat_changes = await self._calculate_stat_changes(
                await self.get_npc_details(npc_id),
                interaction_type
            )
            
            if stat_changes:
                await self.apply_stat_changes(
                    stat_changes,
                    f"Interaction with NPC {npc_id}: {interaction_type}"
                )
            
            # Combine results
            combined_result = {
                "npc_id": npc_id,
                "interaction_type": interaction_type,
                "npc_responses": npc_response.get("npc_responses", []),
                "perception": perception_result.dict() if hasattr(perception_result, "dict") else perception_result,
                "learning": learning_result,
                "belief_formed": belief_result is not None,
                "stat_changes": stat_changes
            }
            
            # Cache the result
            self._cache_interaction_result(cache_key, combined_result)
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Error handling NPC interaction: {e}")
            return {"error": str(e), "npc_id": npc_id}
    
    async def _calculate_stat_changes(self, npc_details: Dict[str, Any], interaction_type: str) -> Dict[str, int]:
        """Calculate stat changes based on NPC traits and interaction type."""
        if not npc_details:
            return {}
            
        stat_changes = {}
        dominance = npc_details.get("stats", {}).get("dominance", 50)
        cruelty = npc_details.get("stats", {}).get("cruelty", 50)
        
        if interaction_type == "submissive_response":
            # Submitting to a dominant NPC increases corruption and obedience
            dominance_factor = dominance / 100
            stat_changes = {
                "corruption": int(2 + (dominance_factor * 3)),
                "obedience": int(3 + (dominance_factor * 4)),
                "willpower": -2,
                "confidence": -1
            }
        elif interaction_type == "defiant_response":
            # Defying increases willpower but may decrease mental resilience
            cruelty_factor = cruelty / 100
            stat_changes = {
                "willpower": 3,
                "confidence": 2,
                "mental_resilience": int(-1 - (cruelty_factor * 3))
            }
        elif interaction_type == "flirtatious_remark":
            # Flirting increases lust and can affect various stats
            dominance_factor = dominance / 100
            stat_changes = {
                "lust": 3, 
                "corruption": int(1 + (dominance_factor * 2))
            }
        
        return stat_changes
    
    async def _apply_stat_changes_transaction(self, conn, stat_changes, cause):
        """Apply stat changes within a transaction."""
        try:
            # Apply each stat change
            for stat_name, change_value in stat_changes.items():
                # Get current value
                query_text = f"SELECT {stat_name} FROM PlayerStats WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'"
                row = await conn.fetchrow(query_text, self.user_id, self.conversation_id)
                
                if row:
                    old_value = row[stat_name]
                    new_value = old_value + change_value
                    
                    # Update the stat
                    update_query = f"UPDATE PlayerStats SET {stat_name} = $1 WHERE user_id=$2 AND conversation_id=$3 AND player_name='Chase'"
                    await conn.execute(update_query, new_value, self.user_id, self.conversation_id)
                    
                    # Log the change
                    await conn.execute("""
                        INSERT INTO StatHistory (user_id, conversation_id, stat_name, old_value, new_value, cause, timestamp)
                        VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
                    """, self.user_id, self.conversation_id, stat_name, old_value, new_value, cause)
        except Exception as e:
            logger.error(f"Error applying stat changes in transaction: {e}")
            raise
    
    def _check_interaction_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check cache for recent similar interactions."""
        if not hasattr(self, '_interaction_cache'):
            self._interaction_cache = {}
            self._interaction_cache_timestamps = {}
            return None
            
        # Check if we have this exact interaction cached recently (30 seconds TTL)
        if cache_key in self._interaction_cache:
            timestamp = self._interaction_cache_timestamps.get(cache_key)
            if timestamp and (datetime.now() - timestamp).total_seconds() < 30:
                # Use cached result for very recent identical interactions
                return self._interaction_cache[cache_key]
        
        return None
    
    def _cache_interaction_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache interaction result with TTL."""
        if not hasattr(self, '_interaction_cache'):
            self._interaction_cache = {}
            self._interaction_cache_timestamps = {}
        
        # Cache successful results (avoid caching errors)
        if "error" not in result:
            self._interaction_cache[cache_key] = result
            self._interaction_cache_timestamps[cache_key] = datetime.now()
            
            # Keep cache size reasonable
            if len(self._interaction_cache) > 100:
                # Remove oldest entries
                oldest_keys = sorted(
                    self._interaction_cache_timestamps.keys(), 
                    key=lambda k: self._interaction_cache_timestamps[k]
                )[:20]  # Remove 20 oldest
                
                for key in oldest_keys:
                    if key in self._interaction_cache:
                        del self._interaction_cache[key]
                    if key in self._interaction_cache_timestamps:
                        del self._interaction_cache_timestamps[key]
        
    async def handle_group_interaction(self,
                                    npc_ids: List[int],
                                    interaction_type: str,
                                    player_input: str,
                                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle an interaction between player and multiple NPCs using the agent architecture.
        
        Args:
            npc_ids: List of NPC IDs to interact with
            interaction_type: Type of interaction
            player_input: Player's input text
            context: Additional context
            
        Returns:
            Comprehensive result dictionary
            
        Raises:
            NPCSystemError: If there's an issue handling the group interaction
        """
        try:
            # Create player action object
            player_action = {
                "type": interaction_type,
                "description": player_input,
                "group_interaction": True
            }
            
            # Prepare context
            context_obj = context or {}
            context_obj["interaction_type"] = interaction_type
            context_obj["group_interaction"] = True
            context_obj["affected_npcs"] = npc_ids
            
            # Process through the agent system's coordinator
            result = await self.agent_system.handle_group_npc_interaction(npc_ids, player_action, context_obj)
            
            # Process the activity and potentially advance time
            activity_result = await self.process_player_activity(player_input, context_obj)
            
            # Update relationships with each NPC
            for npc_id in npc_ids:
                npc_response = next((r for r in result.get("npc_responses", []) if r.get("npc_id") == npc_id), {})
                await self.update_relationship_from_interaction(
                    npc_id=npc_id,
                    player_action=player_action,
                    npc_action=npc_response,
                    context=context_obj
                )
            
            # Combine results
            combined_result = {
                "npc_ids": npc_ids,
                "interaction_type": interaction_type,
                "npc_responses": result.get("npc_responses", []),
                "events": [],
                "stat_changes": {},
                "time_advanced": activity_result.get("time_advanced", False)
            }
            
            # Add time advancement info if applicable
            if activity_result.get("time_advanced", False):
                combined_result["new_time"] = activity_result.get("new_time")
                
                # If time advanced, add any events that occurred
                for event in activity_result.get("events", []):
                    combined_result["events"].append(event)
            
            # Apply stat effects to player based on the group interaction
            stat_changes = {}
            
            # Calculate group dominance average with a single batch query
            # UPDATED: Use async connection context manager
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT AVG(dominance) as avg_dominance, AVG(cruelty) as avg_cruelty, COUNT(*) as npc_count
                    FROM NPCStats
                    WHERE npc_id = ANY($1) AND user_id = $2 AND conversation_id = $3
                """, npc_ids, self.user_id, self.conversation_id)
                
                if rows and rows[0]["npc_count"] > 0:
                    avg_dominance = rows[0]["avg_dominance"]
                    avg_cruelty = rows[0]["avg_cruelty"]
                    npc_count = rows[0]["npc_count"]
                    
                    if interaction_type == "submissive_response":
                        # Submitting to a group increases effects
                        dominance_factor = avg_dominance / 100  # 0.0 to 1.0
                        stat_changes = {
                            "corruption": int(3 + (dominance_factor * 4)),
                            "obedience": int(4 + (dominance_factor * 5)),
                            "willpower": -3,
                            "confidence": -2
                        }
                    elif interaction_type == "defiant_response":
                        # Defying a group is more impactful
                        cruelty_factor = avg_cruelty / 100  # 0.0 to 1.0
                        stat_changes = {
                            "willpower": +4,
                            "confidence": +3,
                            "mental_resilience": int(-2 - (cruelty_factor * 4))
                        }
            
            if stat_changes:
                # Apply stat changes
                await self.apply_stat_changes(
                    stat_changes, 
                    cause=f"Group interaction with {len(npc_ids)} NPCs: {interaction_type}"
                )
                combined_result["stat_changes"] = stat_changes
            
            return combined_result
            
        except Exception as e:
            error_msg = f"Error handling group interaction: {e}"
            logger.error(error_msg)
            raise NPCSystemError(error_msg)
    
    #=================================================================
    # UTILITY METHODS
    #=================================================================
    
    async def get_npc_name(self, npc_id: int) -> str:
        """
        Get the name of an NPC. Uses caching for better performance.
        
        Args:
            npc_id: The NPC ID
            
        Returns:
            NPC name
        """
        # Check cache first
        cache_key = f"npc:{npc_id}"
        if cache_key in self.npc_cache:
            return self.npc_cache[cache_key].get("npc_name", f"NPC-{npc_id}")
        
        try:
            # UPDATED: Use async connection context manager
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow("""
                    SELECT npc_name 
                    FROM NPCStats
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                """, npc_id, self.user_id, self.conversation_id)
                
                if row:
                    npc_name = row["npc_name"]
                    
                    # Update cache
                    self.npc_cache[cache_key] = {
                        "npc_id": npc_id,
                        "npc_name": npc_name,
                        "last_updated": datetime.now()
                    }
                    
                    return npc_name
                
                return f"NPC-{npc_id}"
                
        except Exception as e:
            logger.error(f"Error getting NPC name: {e}")
            return f"NPC-{npc_id}"
