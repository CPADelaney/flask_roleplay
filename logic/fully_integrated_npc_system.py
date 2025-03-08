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

# Import existing modules to utilize their functionality
from db.connection import get_db_connection, get_connection_pool
from logic.npc_creation import (
    create_npc_partial, 
    insert_npc_stub_into_db, 
    assign_random_relationships,
    gpt_generate_physical_description,
    gpt_generate_schedule,
    gpt_generate_memories,
    gpt_generate_affiliations,
    integrate_femdom_elements,
    propagate_shared_memories
)
from logic.social_links import (
    create_social_link,
    update_link_type_and_level,
    add_link_event
)
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

# Import agent-based architecture components
from logic.npc_agents.npc_agent import NPCAgent
from logic.npc_agents.agent_system import NPCAgentSystem
from logic.npc_agents.agent_coordinator import NPCAgentCoordinator
from logic.npc_agents.decision_engine import NPCDecisionEngine
from logic.npc_agents.relationship_manager import NPCRelationshipManager
from logic.npc_agents.memory_manager import EnhancedMemoryManager

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
    link_id: int
    npc_id: int
    npc_name: str
    type: str
    description: str
    options: List[Dict[str, Any]]
    expires_in: int

class IntegratedNPCSystem:
    """
    Central system that integrates NPC creation, social dynamics, time management,
    memory systems, and stat progression using an agent-based architecture.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the NPC system with enhanced connection pooling.
        
        Args:
            user_id: The user ID
            conversation_id: The conversation ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Enhanced connection pool with better configuration
        self._pool = None
        self._pool_initialized = asyncio.Event()
        # Start async initialization in the background
        asyncio.create_task(self._initialize_pool())
        
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
        }
        
        # Initialize the activity manager
        self.activity_manager = ActivityManager()
        
        # Initialize the agent system - core component for NPC agentic behavior
        self.agent_system = NPCAgentSystem(user_id, conversation_id)
        
        logger.info(f"Initialized IntegratedNPCSystem for user={user_id}, conversation={conversation_id}")
        
        # Initialize memory system
        self._memory_system = None
        
        # Set up periodic cache cleanup and metrics reporting
        self._setup_cache_cleanup()
        self._setup_metrics_reporting()
    
    async def _initialize_pool(self):
        """Initialize the connection pool with optimized settings."""
        try:
            # Get connection string from environment
            dsn = os.getenv("DB_DSN")
            self._pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=5,      # Minimum connections in pool
                max_size=20,     # Maximum connections in pool
                command_timeout=60.0,
                max_inactive_connection_lifetime=300.0,  # 5 minutes
                max_queries=50000,
                statement_cache_size=0,  # Disable statement cache for less memory usage
            )
            logger.info("Database connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing connection pool: {e}")
            # Create minimal pool as fallback
            self._pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5)
        finally:
            # Signal that initialization is complete
            self._pool_initialized.set()
    
    async def get_connection_pool(self):
        """Get the connection pool, waiting for initialization if needed."""
        await self._pool_initialized.wait()
        return self._pool
    
    async def execute_with_pool(self, query, *args, timeout=10.0):
        """Execute a database query with the connection pool and error handling."""
        start_time = datetime.now()
        pool = await self.get_connection_pool()
        self.perf_metrics['db_queries'] += 1
        
        try:
            async with pool.acquire() as conn:
                result = await conn.execute(query, *args)
                
                # Track query performance
                query_time = (datetime.now() - start_time).total_seconds()
                self.perf_metrics['query_times'].append(query_time)
                self._update_avg_query_time(query_time)
                
                return result
        except asyncpg.PostgresConnectionError:
            # Connection error - try to reinitialize pool
            logger.error("Database connection error, reinitializing pool...")
            await self._initialize_pool()
            # Retry once with fresh connection
            async with pool.acquire() as conn:
                return await conn.execute(query, *args)
        except Exception as e:
            logger.error(f"Database query error: {e}, Query: {query[:100]}...")
            raise
    
    async def fetchrow_with_pool(self, query, *args, timeout=10.0):
        """Fetch a single row with the connection pool and error handling."""
        start_time = datetime.now()
        pool = await self.get_connection_pool()
        self.perf_metrics['db_queries'] += 1
        
        try:
            async with pool.acquire() as conn:
                result = await conn.fetchrow(query, *args)
                
                # Track query performance
                query_time = (datetime.now() - start_time).total_seconds()
                self.perf_metrics['query_times'].append(query_time)
                self._update_avg_query_time(query_time)
                
                return result
        except asyncpg.PostgresConnectionError:
            # Connection error - try to reinitialize pool
            logger.error("Database connection error, reinitializing pool...")
            await self._initialize_pool()
            # Retry once with fresh connection
            async with pool.acquire() as conn:
                return await conn.fetchrow(query, *args)
        except Exception as e:
            logger.error(f"Database query error: {e}, Query: {query[:100]}...")
            raise
    
    async def fetch_with_pool(self, query, *args, timeout=10.0):
        """Fetch multiple rows with the connection pool and error handling."""
        start_time = datetime.now()
        pool = await self.get_connection_pool()
        self.perf_metrics['db_queries'] += 1
        
        try:
            async with pool.acquire() as conn:
                result = await conn.fetch(query, *args)
                
                # Track query performance
                query_time = (datetime.now() - start_time).total_seconds()
                self.perf_metrics['query_times'].append(query_time)
                self._update_avg_query_time(query_time)
                
                return result
        except asyncpg.PostgresConnectionError:
            # Connection error - try to reinitialize pool
            logger.error("Database connection error, reinitializing pool...")
            await self._initialize_pool()
            # Retry once with fresh connection
            async with pool.acquire() as conn:
                return await conn.fetch(query, *args)
        except Exception as e:
            logger.error(f"Database query error: {e}, Query: {query[:100]}...")
            raise
    
    def _update_avg_query_time(self, new_time):
        """Update the average query time metric."""
        times = self.perf_metrics['query_times'][-100:]  # Only keep last 100 times
        self.perf_metrics['avg_query_time'] = sum(times) / len(times)
    
    def _setup_cache_cleanup(self):
        """Set up periodic cache cleanup task with improved granularity."""
        async def cache_cleanup_task():
            while True:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_cache()
        
        # Start the task without waiting for it
        asyncio.create_task(cache_cleanup_task())
    
    def _setup_metrics_reporting(self):
        """Set up periodic metrics reporting for performance monitoring."""
        async def metrics_reporting_task():
            while True:
                await asyncio.sleep(600)  # Report every 10 minutes
                logger.info(f"Performance metrics: {self.perf_metrics}")
                # Reset some counters
                self.perf_metrics['db_queries'] = 0
                self.perf_metrics['cache_hits'] = 0
                self.perf_metrics['cache_misses'] = 0
                self.perf_metrics['query_times'] = self.perf_metrics['query_times'][-100:]
        
        # Start the task without waiting for it
        asyncio.create_task(metrics_reporting_task())
    
    async def _cleanup_cache(self):
        """Clean up expired cache entries with granular control."""
        now = datetime.now()
        
        # Check global TTL first
        global_expired = now - self.last_cache_refresh > self.cache_ttl
        if global_expired:
            self.last_cache_refresh = now
            logger.debug(f"Global cache TTL expired, clearing all caches")
            
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
        """
        Initialize the NPC system with caching and connection pooling.
        
        Args:
            user_id: The user ID
            conversation_id: The conversation ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize connection pool for better database performance
        self.connection_pool = get_connection_pool()
        
        # Initialize caching structures
        self.npc_cache = {}  # Cache for NPC data
        self.relationship_cache = {}  # Cache for relationship data
        self.last_cache_refresh = datetime.now()
        self.cache_ttl = timedelta(minutes=5)  # Cache time-to-live
        
        # Initialize the activity manager
        self.activity_manager = ActivityManager()
        
        # Initialize the agent system - core component for NPC agentic behavior
        self.agent_system = NPCAgentSystem(user_id, conversation_id)
        
        logger.info(f"Initialized IntegratedNPCSystem with NPCAgentSystem for user={user_id}, conversation={conversation_id}")
        logger.info(f"Available time phases: {TIME_PHASES}")
        
        # Initialize memory system
        self._memory_system = None
        
        # Set up periodic cache cleanup
        self._setup_cache_cleanup()
    
    def _setup_cache_cleanup(self):
        """Set up periodic cache cleanup task"""
        async def cache_cleanup_task():
            while True:
                await asyncio.sleep(300)  # Run every 5 minutes
                self._cleanup_cache()
        
        # Start the task without waiting for it
        asyncio.create_task(cache_cleanup_task())
    
    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        now = datetime.now()
        if now - self.last_cache_refresh > self.cache_ttl:
            self.npc_cache.clear()
            self.relationship_cache.clear()
            self.last_cache_refresh = now
            logger.debug("Cleared NPC system caches due to TTL expiration")
    
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
    
    async def create_new_npc(self, environment_desc: str, day_names: List[str], sex: str = "female") -> int:
        """
        Create a new NPC with improved error recovery.
        """
        logger.info(f"Creating new NPC in environment: {environment_desc[:30]}...")
        
        retry_count = 0
        max_retries = 3
        backoff_factor = 1.5
        
        while retry_count <= max_retries:
            try:
                # Step 1: Create the partial NPC (base data)
                partial_npc = create_npc_partial(
                    user_id=self.user_id,
                    conversation_id=self.conversation_id,
                    sex=sex,
                    total_archetypes=4,
                    environment_desc=environment_desc
                )
                
                # Create a transaction for the entire NPC creation process
                pool = await self.get_connection_pool()
                async with pool.acquire() as conn:
                    async with conn.transaction():
                        # Step 2: Insert the partial NPC into the database
                        insert_query = """
                            INSERT INTO NPCStats (user_id, conversation_id, npc_name, sex, 
                                                archetype_summary, archetypes, introduced)
                            VALUES ($1, $2, $3, $4, $5, $6, FALSE)
                            RETURNING npc_id
                        """
                        npc_id = await conn.fetchval(
                            insert_query,
                            self.user_id, self.conversation_id,
                            partial_npc['npc_name'], partial_npc['sex'],
                            partial_npc['archetype_summary'], json.dumps(partial_npc.get('archetypes', []))
                        )
                        
                        logger.info(f"Created NPC stub with ID {npc_id} and name {partial_npc['npc_name']}")

            
            # Step 3: Assign relationships (with more error handling)
            try:
                await assign_random_relationships(
                    user_id=self.user_id,
                    conversation_id=self.conversation_id,
                    new_npc_id=npc_id,
                    new_npc_name=partial_npc["npc_name"],
                    npc_archetypes=partial_npc.get("archetypes", [])
                )
            except Exception as e:
                logger.warning(f"Error assigning relationships for NPC {npc_id}: {e}")
                # Continue despite relationship assignment errors
            
            # Step 4: Get relationships for memory generation (using connection pool)
            relationships = []
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT relationships 
                    FROM NPCStats 
                    WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                    """,
                    self.user_id, self.conversation_id, npc_id
                )
                
                if row and row["relationships"]:
                    if isinstance(row["relationships"], str):
                        relationships = json.loads(row["relationships"])
                    else:
                        relationships = row["relationships"]
            
            # Parallelize NPC generation tasks for better performance
            generation_tasks = [
                gpt_generate_physical_description(self.user_id, self.conversation_id, partial_npc, environment_desc),
                gpt_generate_schedule(self.user_id, self.conversation_id, partial_npc, environment_desc, day_names),
                gpt_generate_memories(self.user_id, self.conversation_id, partial_npc, environment_desc, relationships),
                gpt_generate_affiliations(self.user_id, self.conversation_id, partial_npc, environment_desc)
            ]
            
            results = await asyncio.gather(*generation_tasks, return_exceptions=True)
            
            # Handle results and potential exceptions
            physical_description, schedule, memories, affiliations = None, None, None, None
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error in NPC generation task {i}: {result}")
                    # Use default values for failed tasks
                    if i == 0:  # Physical description
                        physical_description = f"{partial_npc['npc_name']} has a typical appearance."
                    elif i == 1:  # Schedule
                        schedule = {day: {"Morning": "Goes about their day", 
                                         "Afternoon": "Continues their routine", 
                                         "Evening": "Relaxes", 
                                         "Night": "Sleeps"} for day in day_names}
                    elif i == 2:  # Memories
                        memories = [{"text": f"I am {partial_npc['npc_name']}.", "importance": "high"}]
                    elif i == 3:  # Affiliations
                        affiliations = []
                else:
                    if i == 0:
                        physical_description = result
                    elif i == 1:
                        schedule = result
                    elif i == 2:
                        memories = result
                    elif i == 3:
                        affiliations = result
            
            # Step 6: Determine current location based on time of day and schedule
            current_year, current_month, current_day, time_of_day = get_current_time(
                self.user_id, self.conversation_id
            )
            
            # Calculate day index
            day_index = (current_day - 1) % len(day_names)
            current_day_name = day_names[day_index]
            
            # Extract current location from schedule
            current_location = await self._extract_location_from_schedule(
                schedule, current_day_name, time_of_day
            )
            
            # Step 7: Update the NPC with all refined data (batch update for efficiency)
            async with self.connection_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE NPCStats 
                    SET physical_description=$1,
                        schedule=$2,
                        memory=$3,
                        current_location=$4,
                        affiliations=$5
                    WHERE user_id=$6 AND conversation_id=$7 AND npc_id=$8
                """, 
                    physical_description,
                    json.dumps(schedule),
                    json.dumps(memories),
                    current_location,
                    json.dumps(affiliations),
                    self.user_id, self.conversation_id, npc_id
                )
            
            logger.info(f"Successfully refined NPC {npc_id} ({partial_npc['npc_name']})")
            
            # Step 8: Propagate memories to other connected NPCs (in background)
            asyncio.create_task(
                propagate_shared_memories(
                    user_id=self.user_id,
                    conversation_id=self.conversation_id,
                    source_npc_id=npc_id,
                    source_npc_name=partial_npc["npc_name"],
                    memories=memories
                )
            )
            
            # Step 9: Create NPC Agent and initialize mask
            # This utilizes the agent framework by explicitly creating the agent
            agent = NPCAgent(npc_id, self.user_id, self.conversation_id)
            self.agent_system.npc_agents[npc_id] = agent
            
            # Initialize mask using the agent's capabilities
            try:
                mask_manager = await agent._get_mask_manager()
                await mask_manager.initialize_npc_mask(npc_id)
            except Exception as e:
                logger.warning(f"Error initializing mask for NPC {npc_id}: {e}")
                # Continue despite mask initialization errors
            
            # Step 10: Create a direct memory event using the agent's memory system
            try:
                memory_system = await agent._get_memory_system()
                creation_memory = f"I was created on {current_year}-{current_month}-{current_day} during {time_of_day}."
                
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=npc_id,
                    memory_text=creation_memory,
                    importance="medium",
                    tags=["creation", "origin"]
                )
            except Exception as e:
                logger.warning(f"Error creating initial memory for NPC {npc_id}: {e}")
                # Continue despite memory creation errors
            
            # Step 11: Initialize agent's perception of environment
            try:
                initial_context = {
                    "location": current_location,
                    "time_of_day": time_of_day,
                    "description": f"Initial perception upon creation at {current_location}"
                }
                await agent.perceive_environment(initial_context)
            except Exception as e:
                logger.warning(f"Error initializing perception for NPC {npc_id}: {e}")
                # Continue despite perception errors
            
                # Update cache with new NPC
                self.npc_cache[npc_id] = {
                    "npc_id": npc_id,
                    "npc_name": partial_npc["npc_name"],
                    "last_updated": datetime.now()
                }
                
                return npc_id
                
            except asyncpg.PostgresConnectionError as e:
                # Connection error - try to reconnect
                retry_count += 1
                if retry_count <= max_retries:
                    # Exponential backoff
                    wait_time = (backoff_factor ** retry_count) * 0.5
                    logger.warning(f"Database connection error, retrying in {wait_time:.1f}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed to create new NPC after {max_retries} retries: {e}")
                    raise NPCCreationError(f"Database connection failure: {e}")
                    
            except Exception as e:
                error_msg = f"Failed to create new NPC: {e}"
                logger.error(error_msg)
                
                # Try to create a minimal viable NPC rather than failing completely
                if retry_count == max_retries:
                    logger.warning("Attempting to create minimal viable NPC as fallback")
                    try:
                        # Create minimal NPC with just the essential fields
                        minimal_npc_id = await self._create_minimal_fallback_npc(
                            partial_npc.get("npc_name", f"NPC_{datetime.now().timestamp()}"),
                            sex
                        )
                        return minimal_npc_id
                    except Exception as fallback_error:
                        logger.error(f"Even fallback NPC creation failed: {fallback_error}")
                        raise NPCCreationError(f"Complete NPC creation failure: {e}, fallback also failed: {fallback_error}")
                
                retry_count += 1
                if retry_count <= max_retries:
                    # Exponential backoff
                    wait_time = (backoff_factor ** retry_count) * 0.5
                    logger.warning(f"NPC creation error, retrying in {wait_time:.1f}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    raise NPCCreationError(error_msg)
    
    async def _create_minimal_fallback_npc(self, npc_name: str, sex: str) -> int:
        """
        Create a minimal viable NPC as a fallback when full creation fails.
        
        Args:
            npc_name: NPC name to use
            sex: NPC sex
            
        Returns:
            NPC ID of the created minimal NPC
        """
        pool = await self.get_connection_pool()
        async with pool.acquire() as conn:
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
    
    async def _extract_location_from_schedule(self, schedule, current_day_name, time_of_day):
        """
        Extract location from a scheduled activity.
        
        Args:
            schedule: The NPC's schedule
            current_day_name: Current day name
            time_of_day: Current time of day
            
        Returns:
            Extracted location or "Unknown"
        """
        # Default location
        current_location = "Unknown"
        
        # Extract from schedule if available
        if schedule and current_day_name in schedule and time_of_day in schedule[current_day_name]:
            activity = schedule[current_day_name][time_of_day]
            # Extract location from activity description
            location_keywords = ["at the", "in the", "at", "in"]
            for keyword in location_keywords:
                if keyword in activity:
                    parts = activity.split(keyword, 1)
                    if len(parts) > 1:
                        potential_location = parts[1].split(".")[0].split(",")[0].strip()
                        if len(potential_location) > 3:  # Avoid very short fragments
                            current_location = potential_location
                            break
        
        # If we couldn't extract a location, use a random location from the database
        if current_location == "Unknown":
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT location_name 
                    FROM Locations 
                    WHERE user_id=$1 AND conversation_id=$2 
                    ORDER BY RANDOM() LIMIT 1
                    """,
                    self.user_id, self.conversation_id
                )
                
                if row:
                    current_location = row["location_name"]
        
        return current_location
    
    async def create_multiple_npcs(self, environment_desc: str, day_names: List[str], count: int = 3) -> List[int]:
        """
        Create multiple NPCs in the system with parallel execution.
        
        Args:
            environment_desc: Description of the environment
            day_names: List of day names used in the calendar
            count: Number of NPCs to create
            
        Returns:
            List of created NPC IDs
        """
        # Create NPCs in parallel for better performance
        creation_tasks = [
            self.create_new_npc(environment_desc, day_names)
            for _ in range(count)
        ]
        
        # Handle exceptions individually to prevent one failure from stopping all
        npc_ids = []
        results = await asyncio.gather(*creation_tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error creating NPC: {result}")
            else:
                npc_ids.append(result)
        
        return npc_ids
    
    async def get_npc_details(self, npc_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an NPC, enhanced with agent-based data.
        Uses caching for performance optimization.
        
        Args:
            npc_id: The ID of the NPC to retrieve
            
        Returns:
            Dictionary with NPC details or None if not found
            
        Raises:
            NPCNotFoundError: If the NPC cannot be found
        """
        # Check cache first
        cache_key = f"npc:{npc_id}"
        if cache_key in self.npc_cache and (datetime.now() - self.npc_cache[cache_key].get("last_updated", datetime.min)).seconds < 300:
            return self.npc_cache[cache_key]
        
        # Get or create NPC agent
        if npc_id not in self.agent_system.npc_agents:
            self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
        
        agent = self.agent_system.npc_agents[npc_id]
        
        try:
            # Use connection pool for better performance
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT npc_id, npc_name, introduced, sex, dominance, cruelty, 
                           closeness, trust, respect, intensity, archetype_summary,
                           physical_description, current_location, memory
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                """, self.user_id, self.conversation_id, npc_id)
                
                if not row:
                    error_msg = f"NPC with ID {npc_id} not found"
                    logger.error(error_msg)
                    raise NPCNotFoundError(error_msg)
                
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
                
                # Get social links (with batched query for performance)
                links = await self._fetch_npc_relationships(npc_id, conn)
                
                # Build enhanced response with agent-based data
                npc_details = {
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
                    "memories": agent_memories or memories[:5],  # Prefer agent memories
                    "memory_count": len(memories),
                    "mask": mask_info if mask_info and "error" not in mask_info else {"integrity": 100},
                    "emotional_state": emotional_state,
                    "beliefs": beliefs,
                    "current_perception": current_perception,
                    "relationships": links
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
    
    async def _fetch_npc_relationships(self, npc_id: int, conn) -> List[Dict[str, Any]]:
        """
        Fetch relationships for an NPC using a single database query.
        
        Args:
            npc_id: The NPC ID
            conn: Database connection
            
        Returns:
            List of relationship information
        """
        links = []
        
        # Get all relationships in a single query
        rows = await conn.fetch("""
            SELECT sl.link_id, sl.entity2_type, sl.entity2_id, sl.link_type, sl.link_level,
                   CASE WHEN sl.entity2_type = 'npc' THEN n.npc_name ELSE 'Chase' END as target_name
            FROM SocialLinks sl
            LEFT JOIN NPCStats n ON sl.entity2_type = 'npc' AND sl.entity2_id = n.npc_id 
                                 AND n.user_id = sl.user_id AND n.conversation_id = sl.conversation_id
            WHERE sl.entity1_type = 'npc' 
              AND sl.entity1_id = $1
              AND sl.user_id = $2 
              AND sl.conversation_id = $3
        """, npc_id, self.user_id, self.conversation_id)
        
        for row in rows:
            links.append({
                "link_id": row["link_id"],
                "target_type": row["entity2_type"],
                "target_id": row["entity2_id"],
                "target_name": row["target_name"],
                "link_type": row["link_type"],
                "link_level": row["link_level"]
            })
        
        return links
    
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
            # Use connection pool for performance
            async with self.connection_pool.acquire() as conn:
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
    # SOCIAL LINKS AND RELATIONSHIPS
    #=================================================================
    
    async def create_direct_social_link(self, 
                                      entity1_type: str, entity1_id: int,
                                      entity2_type: str, entity2_id: int,
                                      link_type: str = "neutral", 
                                      link_level: int = 0) -> int:
        """
        Create a direct social link between two entities.
        
        Args:
            entity1_type: Type of first entity
            entity1_id: ID of first entity
            entity2_type: Type of second entity
            entity2_id: ID of second entity
            link_type: Type of link
            link_level: Level of the link
            
        Returns:
            The ID of the created link
            
        Raises:
            RelationshipError: If there's an issue creating the relationship
        """
        try:
            # Using create_social_link
            link_id = create_social_link(
                self.user_id, self.conversation_id,
                entity1_type, entity1_id,
                entity2_type, entity2_id,
                link_type, link_level
            )
            
            # If this involves an NPC, update their relationship manager
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
            
            logger.info(f"Created social link (ID: {link_id}) between {entity1_type}:{entity1_id} and {entity2_type}:{entity2_id}")
            return link_id
            
        except Exception as e:
            error_msg = f"Failed to create social link: {e}"
            logger.error(error_msg)
            raise RelationshipError(error_msg)
    
    async def update_link_details(self, link_id: int, new_type: str = None, level_change: int = 0) -> Dict[str, Any]:
        """
        Update the type and level of a social link.
        
        Args:
            link_id: ID of the link
            new_type: New type for the link (or None to keep current type)
            level_change: Amount to change the level by
            
        Returns:
            Dictionary with update results
            
        Raises:
            RelationshipError: If there's an issue updating the relationship
        """
        try:
            # Get the relationship details before update
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT entity1_type, entity1_id, entity2_type, entity2_id, link_type, link_level
                    FROM SocialLinks
                    WHERE link_id=$1 AND user_id=$2 AND conversation_id=$3
                """, link_id, self.user_id, self.conversation_id)
                
                if not row:
                    return {"error": "Link not found"}
                    
                entity1_type = row["entity1_type"]
                entity1_id = row["entity1_id"]
                entity2_type = row["entity2_type"]
                entity2_id = row["entity2_id"]
                old_type = row["link_type"]
                old_level = row["link_level"]
            
            # Using update_link_type_and_level
            result = update_link_type_and_level(
                self.user_id, self.conversation_id,
                link_id, new_type, level_change
            )
            
            # Update agent memory if an NPC is involved
            if result and entity1_type == "npc":
                # Get or create NPC agent
                if entity1_id not in self.agent_system.npc_agents:
                    self.agent_system.npc_agents[entity1_id] = NPCAgent(entity1_id, self.user_id, self.conversation_id)
                
                agent = self.agent_system.npc_agents[entity1_id]
                memory_system = await agent._get_memory_system()
                
                # Get target name for better memory context
                target_name = await self._get_entity_name(entity2_type, entity2_id)
                
                if new_type and new_type != old_type:
                    memory_text = f"My relationship with {target_name} changed from {old_type} to {new_type}."
                else:
                    direction = "improved" if level_change > 0 else "worsened"
                    memory_text = f"My relationship with {target_name} {direction} from level {old_level} to {result['new_level']}."
                
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
            
            if result:
                logger.info(f"Updated link {link_id}: type={result['new_type']}, level={result['new_level']}")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to update link details: {e}"
            logger.error(error_msg)
            raise RelationshipError(error_msg)
    
    async def add_event_to_link(self, link_id: int, event_text: str) -> bool:
        """
        Add an event to a social link's history.
        
        Args:
            link_id: ID of the link
            event_text: Text describing the event
            
        Returns:
            True if successful
            
        Raises:
            RelationshipError: If there's an issue adding the event
        """
        try:
            # Get the relationship details
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT entity1_type, entity1_id, entity2_type, entity2_id
                    FROM SocialLinks
                    WHERE link_id=$1 AND user_id=$2 AND conversation_id=$3
                """, link_id, self.user_id, self.conversation_id)
                
                if not row:
                    return False
                    
                entity1_type = row["entity1_type"]
                entity1_id = row["entity1_id"]
                entity2_type = row["entity2_type"]
                entity2_id = row["entity2_id"]
            
            # Using add_link_event
            add_link_event(
                self.user_id, self.conversation_id,
                link_id, event_text
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
            
            logger.info(f"Added event to link {link_id}: {event_text[:50]}...")
            return True
            
        except Exception as e:
            error_msg = f"Failed to add event to link: {e}"
            logger.error(error_msg)
            raise RelationshipError(error_msg)
    
    async def update_relationship_from_interaction(self, 
                                               npc_id: int, 
                                               player_action: Dict[str, Any],
                                               npc_action: Dict[str, Any],
                                               context: Dict[str, Any] = None) -> bool:
        """
        Update relationship between NPC and player based on an interaction.
        Enhanced to use agent's relationship manager and memory system for more nuanced updates.
        
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
            # Get or create NPCRelationshipManager for this NPC
            relationship_manager = NPCRelationshipManager(npc_id, self.user_id, self.conversation_id)
            
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
            
            # Enhance relationship update with emotional context
            enhanced_context = {
                "emotional_state": emotional_state,
                "recent_interactions": [],  # Will populate from memory
                "interaction_type": player_action.get("type", "unknown")
            }
            
            # Update context with provided context
            enhanced_context.update(context)
            
            # Get recent memories to inform relationship change
            memory_result = await memory_system.recall(
                entity_type="npc",
                entity_id=npc_id,
                query="player interaction",
                limit=3
            )
            
            enhanced_context["recent_interactions"] = memory_result.get("memories", [])
            
            # Update relationship with enhanced context
            await relationship_manager.update_relationship_from_interaction(
                "player", self.user_id, player_action, npc_action, enhanced_context
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

    async def record_memory_event(self, npc_id: int, memory_text: str, tags: List[str] = None) -> bool:
        """
        Record a memory event for an NPC using the agent's memory system.
        
        Args:
            npc_id: ID of the NPC
            memory_text: The memory text to record
            tags: Optional tags for the memory
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            MemorySystemError: If there's an issue recording the memory
        """
        try:
            # Get or create the NPC agent
            if npc_id not in self.agent_system.npc_agents:
                self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
            
            agent = self.agent_system.npc_agents[npc_id]
            memory_system = await agent._get_memory_system()
            
            # Create the memory
            await memory_system.remember(
                entity_type="npc",
                entity_id=npc_id,
                memory_text=memory_text,
                importance="medium",  # Default importance
                tags=tags or ["player_interaction"]
            )
            return True
            
        except Exception as e:
            error_msg = f"Error recording memory for NPC {npc_id}: {e}"
            logger.error(error_msg)
            raise MemorySystemError(error_msg)

    async def check_for_relationship_events(self) -> List[Dict[str, Any]]:
        """
        Check for special relationship events like crossroads or rituals.
        
        Returns:
            List of relationship events
            
        Raises:
            RelationshipError: If there's an issue checking for events
        """
        try:
            events = []
            
            # Use connection pool for performance
            async with self.connection_pool.acquire() as conn:
                # Get social links with sufficiently high levels that might trigger events
                rows = await conn.fetch("""
                    SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id, 
                           link_type, link_level
                    FROM SocialLinks
                    WHERE user_id = $1 AND conversation_id = $2
                      AND (entity1_type = 'player' OR entity2_type = 'player')
                      AND link_level >= 50
                """, self.user_id, self.conversation_id)
                
                links = []
                for row in rows:
                    link_id = row["link_id"]
                    e1_type = row["entity1_type"]
                    e1_id = row["entity1_id"]
                    e2_type = row["entity2_type"]
                    e2_id = row["entity2_id"]
                    link_type = row["link_type"]
                    link_level = row["link_level"]
                    
                    # Get NPC details if applicable
                    npc_id = None
                    npc_name = None
                    
                    if e1_type == 'npc':
                        npc_id = e1_id
                    elif e2_type == 'npc':
                        npc_id = e2_id
                    
                    if npc_id:
                        npc_row = await conn.fetchrow("""
                            SELECT npc_name FROM NPCStats
                            WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                        """, npc_id, self.user_id, self.conversation_id)
                        
                        if npc_row:
                            npc_name = npc_row["npc_name"]
                    
                    links.append({
                        "link_id": link_id,
                        "entity1_type": e1_type,
                        "entity1_id": e1_id,
                        "entity2_type": e2_type,
                        "entity2_id": e2_id,
                        "link_type": link_type,
                        "link_level": link_level,
                        "npc_id": npc_id,
                        "npc_name": npc_name
                    })
            
            # Check each link for potential events
            for link in links:
                # Check for crossroads event - significant decision point
                if link["link_level"] >= 70 and random.random() < 0.2:  # 20% chance for high level links
                    # Get or create NPC agent for better decision making
                    npc_id = link["npc_id"]
                    if npc_id and npc_id not in self.agent_system.npc_agents:
                        self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
                    
                    # Get NPC agent if available
                    agent = self.agent_system.npc_agents.get(npc_id) if npc_id else None
                    
                    # Generate crossroads event (potentially using agent for better decision modeling)
                    crossroads_data = await self._generate_relationship_crossroads(link, agent)
                    
                    if crossroads_data:
                        events.append({
                            "type": "relationship_crossroads",
                            "data": crossroads_data
                        })
            
            return events
            
        except Exception as e:
            error_msg = f"Error checking for relationship events: {e}"
            logger.error(error_msg)
            raise RelationshipError(error_msg)
    
    async def _generate_relationship_crossroads(self, link: Dict[str, Any], agent: Optional[NPCAgent] = None) -> Dict[str, Any]:
        """
        Generate a relationship crossroads event based on link details and NPC agent.
        
        Args:
            link: The social link data
            agent: Optional NPC agent for better decision modeling
            
        Returns:
            Crossroads event data
        """
        # Default crossroads types based on relationship level
        crossroads_types = [
            "trust_test",
            "commitment_decision",
            "loyalty_challenge",
            "boundary_setting",
            "power_dynamic_shift"
        ]
        
        # Use agent to refine crossroads type if available
        selected_type = random.choice(crossroads_types)
        if agent:
            try:
                # Get agent's current emotional state and perception for better context
                memory_system = await agent._get_memory_system()
                emotional_state = await memory_system.get_npc_emotion(link["npc_id"])
                
                # Use emotional state to influence crossroads type
                if emotional_state and "current_emotion" in emotional_state:
                    emotion = emotional_state["current_emotion"]
                    primary = emotion.get("primary", {})
                    emotion_name = primary.get("name", "neutral")
                    
                    # Adjust crossroads type based on emotional state
                    if emotion_name == "anger":
                        selected_type = "boundary_setting" if random.random() < 0.7 else "power_dynamic_shift"
                    elif emotion_name == "joy":
                        selected_type = "commitment_decision" if random.random() < 0.7 else "trust_test"
                    elif emotion_name == "fear":
                        selected_type = "trust_test" if random.random() < 0.7 else "loyalty_challenge"
            except Exception as e:
                logger.warning(f"Error using agent for crossroads generation: {e}")
                # Fall back to random selection
        
        # Generate crossroads options based on type
        options = self._generate_crossroads_options(selected_type, link)
        
        # Create crossroads data
        crossroads_data = {
            "link_id": link["link_id"],
            "npc_id": link["npc_id"],
            "npc_name": link["npc_name"],
            "type": selected_type,
            "description": self._get_crossroads_description(selected_type, link),
            "options": options,
            "expires_in": 3  # Number of interactions before expiring
        }
        
        return crossroads_data
    
    def _generate_crossroads_options(self, crossroads_type: str, link: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate options for a relationship crossroads based on type."""
        npc_name = link.get('npc_name', 'The NPC')
        
        if crossroads_type == "trust_test":
            return [
                {
                    "text": "Trust completely",
                    "stat_effects": {"trust": 10, "respect": 5, "willpower": -5},
                    "outcome": f"Your trust in {npc_name} deepens significantly."
                },
                {
                    "text": "Remain cautious",
                    "stat_effects": {"trust": 0, "willpower": 5, "respect": 0},
                    "outcome": f"You maintain your guard with {npc_name}."
                },
                {
                    "text": "Express distrust",
                    "stat_effects": {"trust": -10, "respect": 0, "willpower": 10},
                    "outcome": f"Your relationship with {npc_name} becomes more distant."
                }
            ]
        elif crossroads_type == "commitment_decision":
            # Generate options based on commitment decision
            return [
                {
                    "text": "Commit fully",
                    "stat_effects": {"closeness": 15, "willpower": -10, "obedience": 10},
                    "outcome": f"Your relationship with {npc_name} becomes much closer."
                },
                {
                    "text": "Partial commitment",
                    "stat_effects": {"closeness": 5, "willpower": 0, "obedience": 0},
                    "outcome": f"You become somewhat closer to {npc_name}."
                },
                {
                    "text": "Maintain independence",
                    "stat_effects": {"willpower": 10, "closeness": -5, "obedience": -5},
                    "outcome": f"You maintain your independence from {npc_name}."
                }
            ]
        elif crossroads_type == "loyalty_challenge":
            return [
                {
                    "text": "Demonstrate unwavering loyalty",
                    "stat_effects": {"respect": 15, "obedience": 10, "willpower": -5},
                    "outcome": f"{npc_name} is deeply impressed by your loyalty."
                },
                {
                    "text": "Balance loyalty with personal needs",
                    "stat_effects": {"respect": 5, "willpower": 5, "obedience": 0},
                    "outcome": f"You find a middle ground with {npc_name}."
                },
                {
                    "text": "Prioritize your own interests",
                    "stat_effects": {"willpower": 10, "respect": -5, "obedience": -10},
                    "outcome": f"You choose your own path, disappointing {npc_name}."
                }
            ]
        elif crossroads_type == "boundary_setting":
            return [
                {
                    "text": "Allow boundaries to be pushed",
                    "stat_effects": {"obedience": 15, "willpower": -10, "corruption": 5},
                    "outcome": f"You let {npc_name} push your boundaries further."
                },
                {
                    "text": "Negotiate reasonable boundaries",
                    "stat_effects": {"respect": 5, "willpower": 5, "obedience": 0},
                    "outcome": f"You establish healthy boundaries with {npc_name}."
                },
                {
                    "text": "Firmly maintain strict boundaries",
                    "stat_effects": {"willpower": 15, "respect": -5, "obedience": -10},
                    "outcome": f"You stand firm against {npc_name}'s pressure."
                }
            ]
        elif crossroads_type == "power_dynamic_shift":
            return [
                {
                    "text": "Submit to their authority",
                    "stat_effects": {"obedience": 15, "willpower": -10, "corruption": 10},
                    "outcome": f"You accept {npc_name}'s dominance in the relationship."
                },
                {
                    "text": "Seek balanced power",
                    "stat_effects": {"respect": 5, "willpower": 5, "confidence": 5},
                    "outcome": f"You work toward a more equal relationship with {npc_name}."
                },
                {
                    "text": "Assert your dominance",
                    "stat_effects": {"willpower": 10, "confidence": 15, "obedience": -15},
                    "outcome": f"You take control in your relationship with {npc_name}."
                }
            ]
        
        # Default options if type not recognized
        return [
            {
                "text": "Strengthen relationship",
                "stat_effects": {"closeness": 10, "respect": 5},
                "outcome": f"Your bond with {npc_name} strengthens."
            },
            {
                "text": "Maintain status quo",
                "stat_effects": {"closeness": 0, "respect": 0},
                "outcome": f"Your relationship with {npc_name} continues unchanged."
            },
            {
                "text": "Create distance",
                "stat_effects": {"closeness": -10, "respect": -5, "willpower": 5},
                "outcome": f"You create some distance between yourself and {npc_name}."
            }
        ]
    
    def _get_crossroads_description(self, crossroads_type: str, link: Dict[str, Any]) -> str:
        """Get description text for a relationship crossroads."""
        npc_name = link.get("npc_name", "The NPC")
        
        descriptions = {
            "trust_test": f"{npc_name} has shared something important with you. How much will you trust them?",
            "commitment_decision": f"Your relationship with {npc_name} has reached a critical point. How committed will you be?",
            "loyalty_challenge": f"{npc_name} is testing your loyalty. How will you respond?",
            "boundary_setting": f"{npc_name} is pushing boundaries in your relationship. How will you establish limits?",
            "power_dynamic_shift": f"The power dynamic with {npc_name} is shifting. How will you position yourself?"
        }
        
        return descriptions.get(crossroads_type, f"You've reached a crossroads in your relationship with {npc_name}.")
    
    async def apply_crossroads_choice(self, link_id: int, crossroads_name: str, choice_index: int) -> Dict[str, Any]:
        """
        Apply a choice in a relationship crossroads.
        
        Args:
            link_id: ID of the social link
            crossroads_name: Type/name of the crossroads
            choice_index: Index of the chosen option
            
        Returns:
            Result of the choice
            
        Raises:
            RelationshipError: If there's an issue applying the choice
        """
        try:
            # Use connection pool for performance
            async with self.connection_pool.acquire() as conn:
                # Get link details
                row = await conn.fetchrow("""
                    SELECT entity1_type, entity1_id, entity2_type, entity2_id, link_type, link_level
                    FROM SocialLinks
                    WHERE link_id = $1 AND user_id = $2 AND conversation_id = $3
                """, link_id, self.user_id, self.conversation_id)
                
                if not row:
                    return {"error": "Link not found"}
                
                e1_type = row["entity1_type"]
                e1_id = row["entity1_id"]
                e2_type = row["entity2_type"]
                e2_id = row["entity2_id"]
                link_type = row["link_type"]
                link_level = row["link_level"]
                
                # Get NPC details if applicable
                npc_id = e1_id if e1_type == 'npc' else e2_id if e2_type == 'npc' else None
                npc_name = await self._get_entity_name(e1_type if e1_type == 'npc' else e2_type, 
                                                     npc_id) if npc_id else None
                
                # Reconstruct link data
                link = {
                    "link_id": link_id,
                    "entity1_type": e1_type,
                    "entity1_id": e1_id,
                    "entity2_type": e2_type,
                    "entity2_id": e2_id,
                    "link_type": link_type,
                    "link_level": link_level,
                    "npc_id": npc_id,
                    "npc_name": npc_name
                }
                
                # Generate options to find the chosen one
                options = self._generate_crossroads_options(crossroads_name, link)
                
                if choice_index < 0 or choice_index >= len(options):
                    return {"error": "Invalid choice index"}
                
                chosen_option = options[choice_index]
                
                # Apply stat effects
                if "stat_effects" in chosen_option:
                    # Convert dict to a list of separate changes for apply_stat_change
                    await self.apply_stat_changes(
                        chosen_option["stat_effects"],
                        f"Crossroads choice in relationship with {npc_name}"
                    )
                
                # Apply relationship changes
                # Determine change based on choice index
                level_change = 10 if choice_index == 0 else 0 if choice_index == 1 else -10
                
                # Update relationship
                await conn.execute("""
                    UPDATE SocialLinks
                    SET link_level = GREATEST(0, LEAST(100, link_level + $1))
                    WHERE link_id = $2
                """, level_change, link_id)
                
                # Add event to link history
                await conn.execute("""
                    UPDATE SocialLinks
                    SET link_history = COALESCE(link_history, '[]'::jsonb) || $1::jsonb
                    WHERE link_id = $2
                """, json.dumps([f"Crossroads choice: {chosen_option['text']}"]), link_id)
            
            # Create memory for NPC
            if npc_id:
                await self.add_memory_to_npc(
                    npc_id,
                    f"The player made a choice about our relationship: {chosen_option['text']}",
                    importance="high",
                    tags=["crossroads", "relationship_choice"]
                )
            
            # Invalidate relationship cache
            cache_key = f"rel:{e1_type}:{e1_id}:{e2_type}:{e2_id}"
            if cache_key in self.relationship_cache:
                del self.relationship_cache[cache_key]
            
            return {
                "success": True,
                "outcome_text": chosen_option.get("outcome", "Your choice has been recorded."),
                "stat_effects": chosen_option.get("stat_effects", {})
            }
        
        except Exception as e:
            error_msg = f"Error applying crossroads choice: {e}"
            logger.error(error_msg)
            raise RelationshipError(error_msg)
    
    async def get_relationship(self, entity1_type: str, entity1_id: int, entity2_type: str, entity2_id: int) -> Dict[str, Any]:
        """
        Get the relationship between two entities.
        
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
            
            # Use connection pool for performance
            async with self.connection_pool.acquire() as conn:
                # Check for direct relationship
                row = await conn.fetchrow("""
                    SELECT link_id, link_type, link_level, link_history
                    FROM SocialLinks
                    WHERE user_id = $1 AND conversation_id = $2
                      AND ((entity1_type = $3 AND entity1_id = $4 AND entity2_type = $5 AND entity2_id = $6)
                         OR (entity1_type = $5 AND entity1_id = $6 AND entity2_type = $3 AND entity2_id = $4))
                """, 
                    self.user_id, self.conversation_id,
                    entity1_type, entity1_id, entity2_type, entity2_id
                )
                
                if not row:
                    return None
                
                link_id = row["link_id"]
                link_type = row["link_type"]
                link_level = row["link_level"]
                link_history = row["link_history"]
                
                # Convert link_history to Python list if it's not None
                if link_history:
                    if isinstance(link_history, str):
                        try:
                            history = json.loads(link_history)
                        except json.JSONDecodeError:
                            history = []
                    else:
                        history = link_history
                else:
                    history = []
                
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
            
            relationship = {
                "link_id": link_id,
                "entity1_type": entity1_type,
                "entity1_id": entity1_id,
                "entity1_name": entity1_name,
                "entity2_type": entity2_type,
                "entity2_id": entity2_id,
                "entity2_name": entity2_name,
                "link_type": link_type,
                "link_level": link_level,
                "link_history": history[-5:],  # Get last 5 events
                "relationship_memories": relationship_memories
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
        Get the name of an entity using connection pooling for better performance.
        
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
            
            # Use connection pool
            async with self.connection_pool.acquire() as conn:
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
            # Use connection pool for performance
            async with self.connection_pool.acquire() as conn:
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
            # Use connection pool for performance
            async with self.connection_pool.acquire() as conn:
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
    
    async def retrieve_relevant_memories(self, 
                                       npc_id: int, 
                                       query: str = None,
                                       context: Dict[str, Any] = None,
                                       limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to a context.
        
        Args:
            npc_id: ID of the NPC
            query: Search query
            context: Context dictionary
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memory objects
            
        Raises:
            MemorySystemError: If there's an issue retrieving memories
        """
        try:
            # Get or create NPC agent
            if npc_id not in self.agent_system.npc_agents:
                self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
            
            agent = self.agent_system.npc_agents[npc_id]
            memory_system = await agent._get_memory_system()
            
            # Prepare context for recall
            context_obj = context or {}
            if query:
                context_obj["query"] = query
            
            # Retrieve memories using the agent's memory system
            result = await memory_system.recall(
                entity_type="npc",
                entity_id=npc_id,
                query=query,
                context=context_obj,
                limit=limit
            )
            
            return result.get("memories", [])
            
        except Exception as e:
            error_msg = f"Error retrieving relevant memories: {e}"
            logger.error(error_msg)
            raise MemorySystemError(error_msg)
    
    async def generate_flashback(self, npc_id: int, current_context: str) -> Optional[Dict[str, Any]]:
        """
        Generate a flashback for an NPC using the agent's capabilities.
        
        Args:
            npc_id: ID of the NPC
            current_context: Current context that may trigger a flashback
            
        Returns:
            Flashback data or None if no flashback was generated
            
        Raises:
            MemorySystemError: If there's an issue generating the flashback
        """
        try:
            # Get or create NPC agent
            if npc_id not in self.agent_system.npc_agents:
                self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
            
            agent = self.agent_system.npc_agents[npc_id]
            memory_system = await agent._get_memory_system()
            
            # Generate flashback using the agent's memory system
            flashback = await memory_system.npc_flashback(
                npc_id=npc_id,
                context=current_context
            )
            
            return flashback
            
        except Exception as e:
            error_msg = f"Error generating flashback: {e}"
            logger.error(error_msg)
            raise MemorySystemError(error_msg)
    
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
            
            # Use connection pool for better performance
            async with self.connection_pool.acquire() as conn:
                # Find related NPCs with a single query
                rows = await conn.fetch("""
                    SELECT sl.entity2_id, sl.link_level, n.npc_name as source_name
                    FROM SocialLinks sl
                    JOIN NPCStats n ON n.npc_id = sl.entity1_id 
                                   AND n.user_id = sl.user_id 
                                   AND n.conversation_id = sl.conversation_id
                    WHERE sl.user_id = $1 
                      AND sl.conversation_id = $2
                      AND sl.entity1_type = 'npc' 
                      AND sl.entity1_id = $3
                      AND sl.entity2_type = 'npc'
                      AND sl.link_level > 30
                """, self.user_id, self.conversation_id, source_npc_id)
                
                source_name = "Unknown"
                related_npcs = []
                
                for row in rows:
                    related_npcs.append((row["entity2_id"], row["link_level"]))
                    if not source_name or source_name == "Unknown":
                        source_name = row["source_name"]
            
            # Propagate memory to each related NPC
            propagation_tasks = []
            for npc_id, link_level in related_npcs:
                # Add task to propagate memory
                propagation_tasks.append(
                    self._propagate_single_memory(npc_id, source_name, memory_text, link_level, importance)
                )
            
            # Run all propagation tasks concurrently
            if propagation_tasks:
                await asyncio.gather(*propagation_tasks)
            
            return True
            
        except Exception as e:
            error_msg = f"Error propagating memory: {e}"
            logger.error(error_msg)
            raise MemorySystemError(error_msg)
    
    async def _propagate_single_memory(self, npc_id: int, source_name: str, memory_text: str, 
                                    link_level: int, importance: str):
        """
        Propagate a single memory to an NPC.
        
        Args:
            npc_id: Target NPC ID
            source_name: Source NPC name
            memory_text: Memory text
            link_level: Relationship level
            importance: Memory importance
        """
        try:
            # Modify the memory text based on relationship
            relationship_factor = link_level / 100.0  # 0.0 to 1.0
            
            # Higher relationship means more accurate propagation
            if relationship_factor > 0.7:
                propagated_text = f"I heard from {source_name} that {memory_text}"
            else:
                # Add potential distortion
                words = memory_text.split()
                if len(words) > 5:
                    # Replace 1-2 words to create slight distortion
                    for _ in range(random.randint(1, 2)):
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
            
            # Create the propagated memory
            await self.add_memory_to_npc(
                npc_id, 
                propagated_text, 
                "low" if importance == "medium" else "medium" if importance == "high" else "low",
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
            
            return result
            
        except Exception as e:
            error_msg = f"Error advancing time with activity: {e}"
            logger.error(error_msg)
            raise TimeSystemError(error_msg)
    
    async def process_player_activity(self, player_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a player's activity, determining if time should advance and handling events.
        Enhanced with agent perception and memory formation.
        
        Args:
            player_input: Player's input text
            context: Additional context
            
        Returns:
            Dictionary with processing results
            
        Raises:
            NPCSystemError: If there's an issue processing the activity
        """
        try:
            # Create base context if not provided
            context_obj = context or {}
            
            # Create standardized player action
            player_action = {
                "type": "activity",
                "description": player_input,
                "text": player_input,
                "context": context_obj
            }
            
            # Determine activity type using activity manager
            activity_result = await self.activity_manager.process_activity(
                self.user_id, self.conversation_id, player_input, context_obj
            )
            
            # Update player action with determined activity type
            player_action["type"] = activity_result.get("activity_type", "generic_activity")
            
            # Add activity perception to nearby NPCs via agent system
            # This ensures NPCs are aware of what the player is doing
            current_location = context_obj.get("location")
            
            if current_location:
                # Get NPCs at current location (batch query for performance)
                async with self.connection_pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT npc_id 
                        FROM NPCStats 
                        WHERE user_id=$1 AND conversation_id=$2 AND current_location=$3
                    """, self.user_id, self.conversation_id, current_location)
                    
                    nearby_npc_ids = [row["npc_id"] for row in rows]
                
                # Prepare perception tasks for all NPCs
                perception_tasks = []
                
                for npc_id in nearby_npc_ids:
                    if npc_id not in self.agent_system.npc_agents:
                        self.agent_system.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
                    
                    agent = self.agent_system.npc_agents[npc_id]
                    
                    # Create perception context
                    perception_context = {
                        "location": current_location,
                        "player_action": player_action,
                        "description": f"Player {player_input}"
                    }
                    
                    # Add perception and memory tasks
                    perception_tasks.append(
                        self._process_npc_perception(agent, npc_id, perception_context, player_action)
                    )
                
                # Run all perception tasks concurrently
                if perception_tasks:
                    await asyncio.gather(*perception_tasks)
            
            # Return the original activity result
            return activity_result
            
        except Exception as e:
            error_msg = f"Error processing player activity: {e}"
            logger.error(error_msg)
            raise NPCSystemError(error_msg)
    
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
        Process scheduled activities for all NPCs using the agent system.
        Enhanced with agent-based decision making and memory formation.
        
        Returns:
            Dictionary with results of NPC activities
            
        Raises:
            NPCSystemError: If there's an issue processing scheduled activities
        """
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
            
            # Get all NPCs with their current locations (batch query for performance)
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location 
                    FROM NPCStats 
                    WHERE user_id=$1 AND conversation_id=$2
                """, self.user_id, self.conversation_id)
                
                npc_data = {row["npc_id"]: {"name": row["npc_name"], "location": row["current_location"]} for row in rows}
            
            # Process activities in parallel for better performance
            activity_tasks = []
            
            # For each NPC, process their scheduled activity
            for npc_id, data in npc_data.items():
                # Create task to process NPC activity
                activity_tasks.append(
                    self._process_single_npc_activity(npc_id, data, base_context)
                )
            
            # Run all activity tasks concurrently and collect results
            results = await asyncio.gather(*activity_tasks, return_exceptions=True)
            
            # Format results, filtering out exceptions
            npc_responses = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error processing scheduled activity: {result}")
                elif result:  # Skip None results
                    npc_responses.append(result)
            
            # Let agent system process all scheduled activities
            # This handles coordination between NPCs, shared observations, etc.
            agent_system_result = await self.agent_system.process_npc_scheduled_activities()
            
            # Combine our results with agent system results
            combined_results = {
                "npc_responses": npc_responses,
                "agent_system_responses": agent_system_result.get("npc_responses", []),
                "count": len(npc_responses)
            }
            
            return combined_results
            
        except Exception as e:
            error_msg = f"Error processing NPC scheduled activities: {e}"
            logger.error(error_msg)
            raise NPCSystemError(error_msg)
    
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
            async with self.connection_pool.acquire() as conn:
                # Find NPCs with important relationships
                rel_rows = await conn.fetch("""
                    SELECT DISTINCT entity1_id as npc_id
                    FROM SocialLinks
                    WHERE user_id = $1 AND conversation_id = $2
                    AND entity1_type = 'npc'
                    AND link_level > 50
                """, self.user_id, self.conversation_id)
                
                for row in rel_rows:
                    important_relationships.add(row["npc_id"])
                
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
            
            # Use connection pool for better performance
            async with self.connection_pool.acquire() as conn:
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
    
    async def handle_npc_interaction(self, 
                                   npc_id: int, 
                                   interaction_type: str,
                                   player_input: str,
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle a complete interaction between player and NPC using the agent architecture.
        
        Args:
            npc_id: ID of the NPC
            interaction_type: Type of interaction (conversation, command, etc.)
            player_input: Player's input text
            context: Additional context
            
        Returns:
            Comprehensive result dictionary
            
        Raises:
            NPCSystemError: If there's an issue handling the interaction
        """
        try:
            # Create player action object
            player_action = {
                "type": interaction_type,
                "description": player_input,
                "target_npc_id": npc_id
            }
            
            # Prepare context
            context_obj = context or {}
            context_obj["interaction_type"] = interaction_type
            
            # Process through the agent system - this is the key change utilizing the agent architecture
            result = await self.agent_system.handle_player_action(player_action, context_obj)
            
            # Process the activity and potentially advance time
            activity_result = await self.process_player_activity(player_input, context_obj)
            
            # Combine results
            combined_result = {
                "npc_id": npc_id,
                "interaction_type": interaction_type,
                "npc_responses": result.get("npc_responses", []),
                "events": [],
                "memories_created": [],
                "stat_changes": {},
                "time_advanced": activity_result.get("time_advanced", False)
            }
            
            # Add time advancement info if applicable
            if activity_result.get("time_advanced", False):
                combined_result["new_time"] = activity_result.get("new_time")
                
                # If time advanced, add any events that occurred
                for event in activity_result.get("events", []):
                    combined_result["events"].append(event)
            
            # Apply stat effects to player
            stat_changes = {}
            
            # Get NPC details
            npc_details = await self.get_npc_details(npc_id)
            
            if npc_details:
                dominance = npc_details["stats"]["dominance"]
                cruelty = npc_details["stats"]["cruelty"]
                
                if interaction_type == "submissive_response":
                    # Submitting to a dominant NPC increases corruption and obedience
                    dominance_factor = dominance / 100  # 0.0 to 1.0
                    stat_changes = {
                        "corruption": int(2 + (dominance_factor * 3)),
                        "obedience": int(3 + (dominance_factor * 4)),
                        "willpower": -2,
                        "confidence": -1
                    }
                elif interaction_type == "defiant_response":
                    # Defying increases willpower and confidence but may decrease other stats
                    # More cruel NPCs cause more mental damage when defied
                    cruelty_factor = cruelty / 100  # 0.0 to 1.0
                    stat_changes = {
                        "willpower": +3,
                        "confidence": +2,
                        "mental_resilience": int(-1 - (cruelty_factor * 3))
                    }
            
            if stat_changes:
                # Apply stat changes
                await self.apply_stat_changes(
                    stat_changes, 
                    cause=f"Interaction with {npc_details['npc_name'] if npc_details else 'NPC'}: {interaction_type}"
                )
                combined_result["stat_changes"] = stat_changes
            
            return combined_result
            
        except Exception as e:
            error_msg = f"Error handling NPC interaction: {e}"
            logger.error(error_msg)
            raise NPCSystemError(error_msg)
    
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
            async with self.connection_pool.acquire() as conn:
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
            # Use connection pool for better performance
            async with self.connection_pool.acquire() as conn:
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

#=================================================================
# USAGE EXAMPLES
#=================================================================

async def example_usage():
    """Example demonstrating key agent-based functionality."""
    user_id = 1
    conversation_id = 123
    
    # Initialize the system
    npc_system = IntegratedNPCSystem(user_id, conversation_id)
    
    # Create a new NPC
    environment_desc = "A mansion with sprawling gardens and opulent interior."
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    npc_id = await npc_system.create_new_npc(environment_desc, day_names)
    
    # Get NPC details
    npc_details = await npc_system.get_npc_details(npc_id)
    print(f"Created NPC: {npc_details['npc_name']}")
    
    # Introduce the NPC
    await npc_system.introduce_npc(npc_id)
    
    # Handle an interaction with the NPC using the agent architecture
    interaction_result = await npc_system.handle_npc_interaction(
        npc_id, "conversation", "Hello, nice to meet you."
    )
    print(f"Interaction result: {interaction_result}")
    
    # Update NPC's emotional state
    await npc_system.update_npc_emotional_state(npc_id, "joy", 0.7)
    
    # Create another NPC
    second_npc_id = await npc_system.create_new_npc(environment_desc, day_names)
    
    # Handle a group interaction
    group_result = await npc_system.handle_group_interaction(
        [npc_id, second_npc_id], "conversation", "Hello everyone!"
    )
    print(f"Group interaction result: {group_result}")
    
    # Generate a scene with both NPCs
    scene = await npc_system.generate_npc_scene([npc_id, second_npc_id], "Garden")
    print(f"Generated scene: {scene}")
    
    # Process scheduled activities
    await npc_system.process_npc_scheduled_activities()
    
    # Run memory maintenance
    await npc_system.run_memory_maintenance()
    
    print("Agent-based NPC system demo completed successfully!")

if __name__ == "__main__":
    # Run the agent-based example
    import asyncio
    asyncio.run(example_usage())
