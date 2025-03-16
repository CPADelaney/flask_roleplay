# context/unified_context_service.py

"""
Unified Context Service that integrates all context optimization components
to provide a streamlined API for context retrieval and management.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from context.context_config import get_config
from context.unified_cache import context_cache
from context.vector_service import get_vector_service
from context.memory_manager import get_memory_manager
from context.context_performance import PerformanceMonitor, track_performance

logger = logging.getLogger(__name__)

class UnifiedContextService:
    """
    Central service for context management that integrates:
    - Caching
    - Vector search
    - Memory consolidation
    - Performance monitoring
    - Incremental updates
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.config = get_config()
        self.last_context = None
        self.last_context_hash = None
        self.performance_monitor = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize the context service"""
        if self.initialized:
            return
            
        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor.get_instance(
            self.user_id, self.conversation_id
        )
        
        self.initialized = True
        logger.info(f"Initialized unified context service for user {self.user_id}, conversation {self.conversation_id}")
    
    async def close(self):
        """Close the context service"""
        self.initialized = False
        logger.info(f"Closed unified context service for user {self.user_id}, conversation {self.conversation_id}")
    
    @track_performance("get_context")
    async def get_context(
        self,
        input_text: str = "",
        location: Optional[str] = None,
        context_budget: int = 4000,
        use_vector_search: Optional[bool] = None,
        use_delta: bool = True,
        include_memories: bool = True,
        include_npcs: bool = True,
        include_location: bool = True,
        include_quests: bool = True,
        return_full_context: bool = True
    ) -> Dict[str, Any]:
        """
        Get optimized context for the current interaction
        
        Args:
            input_text: Current user input
            location: Optional current location
            context_budget: Maximum token budget
            use_vector_search: Override for vector search setting
            use_delta: Whether to include delta changes
            include_memories: Whether to include memories
            include_npcs: Whether to include NPCs
            include_location: Whether to include location details
            include_quests: Whether to include quests
            return_full_context: Whether to return full context even with delta
            
        Returns:
            Optimized context dictionary
        """
        # Start timers and record request
        timer_id = self.performance_monitor.start_timer("get_context")
        request_start = time.time()
        
        # Check if vector search should be used
        if use_vector_search is None:
            use_vector_search = self.config.is_enabled("use_vector_search")
        
        # Cache key based on parameters
        cache_key_parts = [
            f"context:{self.user_id}:{self.conversation_id}",
            f"loc:{location or 'none'}",
            f"vec:{use_vector_search}",
            f"inc:{include_memories}:{include_npcs}:{include_location}:{include_quests}"
        ]
        
        # Add input_text hash if not empty (for search relevance)
        if input_text:
            import hashlib
            text_hash = hashlib.md5(input_text.encode()).hexdigest()[:8]
            cache_key_parts.append(f"input:{text_hash}")
        
        cache_key = ":".join(cache_key_parts)
        
        # Function to fetch context if not in cache
        async def fetch_context():
            # Initialize sub-components we'll need
            vector_service = None
            memory_manager = None
            
            if use_vector_search:
                vector_service = await get_vector_service(self.user_id, self.conversation_id)
            
            if include_memories:
                memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
            
            # Collect context components in parallel
            tasks = []
            
            # 1. Get base context from database
            tasks.append(self._get_base_context(location))
            
            # 2. Get relevant NPC information
            if include_npcs:
                tasks.append(self._get_relevant_npcs(input_text, location, vector_service))
            else:
                tasks.append(asyncio.create_task(asyncio.sleep(0)))
            
            # 3. Get relevant memories
            if include_memories and memory_manager:
                tasks.append(self._get_relevant_memories(input_text, memory_manager))
            else:
                tasks.append(asyncio.create_task(asyncio.sleep(0)))
            
            # 4. Get location details
            if include_location:
                tasks.append(self._get_location_details(location, vector_service))
            else:
                tasks.append(asyncio.create_task(asyncio.sleep(0)))
            
            # 5. Get active quests
            if include_quests:
                tasks.append(self._get_quest_information())
            else:
                tasks.append(asyncio.create_task(asyncio.sleep(0)))
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)
            
            # Combine results
            context = results[0]  # Base context
            if include_npcs:
                context["npcs"] = results[1]
            if include_memories and memory_manager:
                context["memories"] = results[2]
            if include_location:
                context["location_details"] = results[3]
            if include_quests:
                context["quests"] = results[4]
            
            # Add metadata
            context["timestamp"] = datetime.now().isoformat()
            context["context_budget"] = context_budget
            
            # Calculate token usage
            token_usage = self._calculate_token_usage(context)
            context["token_usage"] = token_usage
            
            # Trim to fit budget if needed
            if sum(token_usage.values()) > context_budget:
                context = await self._trim_to_budget(context, context_budget)
                # Recalculate token usage after trimming
                token_usage = self._calculate_token_usage(context)
                context["token_usage"] = token_usage
            
            return context
        
        try:
            # Get from cache or fetch
            importance = 0.5  # Default importance
            if "quest" in input_text.lower():
                importance = 0.8  # More important for quest-related inputs
            
            # Try to get from cache with a short TTL
            context = await context_cache.get(
                cache_key, 
                fetch_context, 
                cache_level=1,
                importance=importance,
                ttl_override=10  # Short TTL for context
            )
            
            # If using delta updates, calculate delta if needed
            if use_delta:
                context = self._apply_delta_tracking(context)
            
            # Record token usage for performance monitoring
            token_usage = context.get("token_usage", {})
            total_tokens = sum(token_usage.values()) if token_usage else 0
            self.performance_monitor.record_token_usage(total_tokens)
            
            # Record memory usage
            self.performance_monitor.record_memory_usage()
            
            return context
        
        finally:
            # Stop timer and calculate elapsed time
            elapsed = self.performance_monitor.stop_timer(timer_id)
            logger.debug(f"Context retrieval completed in {elapsed:.4f}s")
    
    async def _get_base_context(self, location: Optional[str] = None) -> Dict[str, Any]:
        """
        Get base context information from database
        
        Args:
            location: Optional current location
            
        Returns:
            Dictionary with base context
        """
        # Use cache with longer TTL for base context
        cache_key = f"base_context:{self.user_id}:{self.conversation_id}:{location or 'none'}"
        
        async def fetch_base_context():
            try:
                # Get database connection
                from db.connection import get_db_connection
                import asyncpg
                
                conn = await asyncpg.connect(dsn=get_db_connection())
                try:
                    # Get time information
                    time_info = {
                        "year": "1040",
                        "month": "6",
                        "day": "15",
                        "time_of_day": "Morning"
                    }
                    
                    # Query time information
                    for key in ["CurrentYear", "CurrentMonth", "CurrentDay", "TimeOfDay"]:
                        row = await conn.fetchrow("""
                            SELECT value
                            FROM CurrentRoleplay
                            WHERE user_id=$1 AND conversation_id=$2 AND key=$3
                        """, self.user_id, self.conversation_id, key)
                        
                        if row:
                            if key == "CurrentYear":
                                time_info["year"] = row["value"]
                            elif key == "CurrentMonth":
                                time_info["month"] = row["value"]
                            elif key == "CurrentDay":
                                time_info["day"] = row["value"]
                            elif key == "TimeOfDay":
                                time_info["time_of_day"] = row["value"]
                    
                    # Get player stats
                    player_stats = {}
                    player_row = await conn.fetchrow("""
                        SELECT corruption, confidence, willpower,
                               obedience, dependency, lust,
                               mental_resilience, physical_endurance
                        FROM PlayerStats
                        WHERE user_id=$1 AND conversation_id=$2
                        LIMIT 1
                    """, self.user_id, self.conversation_id)
                    
                    if player_row:
                        player_stats = dict(player_row)
                    
                    # Get current roleplay data
                    roleplay_data = {}
                    rp_rows = await conn.fetch("""
                        SELECT key, value
                        FROM CurrentRoleplay
                        WHERE user_id=$1 AND conversation_id=$2
                        AND key IN ('CurrentLocation', 'EnvironmentDesc', 'PlayerRole', 'MainQuest')
                    """, self.user_id, self.conversation_id)
                    
                    for row in rp_rows:
                        roleplay_data[row["key"]] = row["value"]
                    
                    # Create base context
                    context = {
                        "time_info": time_info,
                        "player_stats": player_stats,
                        "current_roleplay": roleplay_data,
                        "current_location": location or roleplay_data.get("CurrentLocation", "Unknown")
                    }
                    
                    return context
                
                finally:
                    await conn.close()
            
            except Exception as e:
                logger.error(f"Error getting base context: {e}")
                return {
                    "time_info": {
                        "year": "1040",
                        "month": "6",
                        "day": "15",
                        "time_of_day": "Morning"
                    },
                    "player_stats": {},
                    "current_roleplay": {},
                    "current_location": location or "Unknown",
                    "error": str(e)
                }
        
        # Get from cache or fetch (30 second TTL, medium importance)
        return await context_cache.get(
            cache_key, 
            fetch_base_context, 
            cache_level=1,
            importance=0.7,
            ttl_override=30
        )
    
    async def _get_relevant_npcs(
        self,
        input_text: str,
        location: Optional[str] = None,
        vector_service = None
    ) -> List[Dict[str, Any]]:
        """
        Get NPCs relevant to current input and location
        
        Args:
            input_text: Current user input
            location: Optional current location
            vector_service: Optional vector service instance
            
        Returns:
            List of relevant NPCs
        """
        # Use vector search if available
        if vector_service and input_text:
            try:
                # Get vector context for input
                vector_context = await vector_service.get_context_for_input(
                    input_text=input_text,
                    current_location=location
                )
                
                # Extract NPCs
                if "npcs" in vector_context:
                    return vector_context["npcs"]
            except Exception as e:
                logger.error(f"Error getting NPCs from vector service: {e}")
        
        # Fallback to database query
        try:
            # Get database connection
            from db.connection import get_db_connection
            import asyncpg
            
            conn = await asyncpg.connect(dsn=get_db_connection())
            try:
                # Query params
                params = [self.user_id, self.conversation_id]
                query = """
                    SELECT npc_id, npc_name,
                           dominance, cruelty, closeness,
                           trust, respect, intensity,
                           current_location, physical_description
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 AND introduced=TRUE
                """
                
                # Add location filter if provided
                if location:
                    query += f" AND (current_location IS NULL OR current_location=$3)"
                    params.append(location)
                
                # Limit results
                query += " LIMIT 10"
                
                # Execute query
                rows = await conn.fetch(query, *params)
                
                # Process results
                npcs = []
                for row in rows:
                    npcs.append({
                        "npc_id": row["npc_id"],
                        "npc_name": row["npc_name"],
                        "dominance": row["dominance"],
                        "cruelty": row["cruelty"],
                        "closeness": row["closeness"],
                        "trust": row["trust"],
                        "respect": row["respect"],
                        "intensity": row["intensity"],
                        "current_location": row["current_location"] or "Unknown",
                        "physical_description": row["physical_description"] or "",
                        # Estimated relevance since we're not using vector search
                        "relevance": 0.7 if row["current_location"] == location else 0.5
                    })
                
                return npcs
            
            finally:
                await conn.close()
        
        except Exception as e:
            logger.error(f"Error getting NPCs from database: {e}")
            return []
    
    async def _get_relevant_memories(
        self,
        input_text: str,
        memory_manager
    ) -> List[Dict[str, Any]]:
        """
        Get memories relevant to current input
        
        Args:
            input_text: Current user input
            memory_manager: Memory manager instance
            
        Returns:
            List of relevant memories
        """
        try:
            # Search for relevant memories
            if input_text:
                memories = await memory_manager.search_memories(
                    query_text=input_text,
                    limit=5
                )
            else:
                # Get recent memories if no input
                memories = await memory_manager.get_recent_memories(
                    days=3,
                    limit=5
                )
            
            # Convert to dictionaries
            memory_dicts = []
            for memory in memories:
                memory_dicts.append({
                    "memory_id": memory.memory_id,
                    "content": memory.content,
                    "memory_type": memory.memory_type,
                    "created_at": memory.created_at.isoformat(),
                    "importance": memory.importance,
                    "tags": memory.tags
                })
            
            return memory_dicts
        
        except Exception as e:
            logger.error(f"Error getting relevant memories: {e}")
            return []
    
    async def _get_location_details(
        self,
        location: Optional[str] = None,
        vector_service = None
    ) -> Dict[str, Any]:
        """
        Get details about the current location
        
        Args:
            location: Optional current location
            vector_service: Optional vector service instance
            
        Returns:
            Dictionary with location details
        """
        if not location:
            return {}
        
        # Use vector search if available
        if vector_service:
            try:
                # Get vector context for location
                vector_context = await vector_service.get_context_for_input(
                    input_text=f"Details about {location}",
                    current_location=location
                )
                
                # Extract locations
                if "locations" in vector_context and vector_context["locations"]:
                    for loc in vector_context["locations"]:
                        if loc.get("location_name", "").lower() == location.lower():
                            return loc
            except Exception as e:
                logger.error(f"Error getting location from vector service: {e}")
        
        # Fallback to database query
        try:
            # Get database connection
            from db.connection import get_db_connection
            import asyncpg
            
            conn = await asyncpg.connect(dsn=get_db_connection())
            try:
                # Query location
                row = await conn.fetchrow("""
                    SELECT id, location_name, description
                    FROM Locations
                    WHERE user_id=$1 AND conversation_id=$2 AND location_name=$3
                    LIMIT 1
                """, self.user_id, self.conversation_id, location)
                
                if row:
                    return {
                        "location_id": row["id"],
                        "location_name": row["location_name"],
                        "description": row["description"]
                    }
                
                return {}
            
            finally:
                await conn.close()
        
        except Exception as e:
            logger.error(f"Error getting location details: {e}")
            return {}
    
    async def _get_quest_information(self) -> List[Dict[str, Any]]:
        """
        Get information about active quests
        
        Returns:
            List of active quests
        """
        try:
            # Get database connection
            from db.connection import get_db_connection
            import asyncpg
            
            conn = await asyncpg.connect(dsn=get_db_connection())
            try:
                # Query active quests
                rows = await conn.fetch("""
                    SELECT quest_id, quest_name, status, progress_detail,
                           quest_giver, reward
                    FROM Quests
                    WHERE user_id=$1 AND conversation_id=$2
                    AND status IN ('active', 'in_progress')
                    ORDER BY quest_id
                """, self.user_id, self.conversation_id)
                
                # Process results
                quests = []
                for row in rows:
                    quests.append({
                        "quest_id": row["quest_id"],
                        "quest_name": row["quest_name"],
                        "status": row["status"],
                        "progress_detail": row["progress_detail"],
                        "quest_giver": row["quest_giver"],
                        "reward": row["reward"]
                    })
                
                return quests
            
            finally:
                await conn.close()
        
        except Exception as e:
            logger.error(f"Error getting quest information: {e}")
            return []
    
    def _apply_delta_tracking(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply delta tracking to context
        
        Args:
            context: Full context
            
        Returns:
            Context with delta information
        """
        # Calculate hash of context (excluding certain fields)
        context_for_hash = {k: v for k, v in context.items() 
                           if k not in ["timestamp", "token_usage", "is_delta", "delta_changes"]}
        
        context_hash = self._hash_context(context_for_hash)
        
        # If this is the first request, just return full context
        if not self.last_context_hash:
            self.last_context = context_for_hash
            self.last_context_hash = context_hash
            return {**context, "is_delta": False}
        
        # If hash matches, nothing changed
        if context_hash == self.last_context_hash:
            return {**context, "is_delta": False, "no_changes": True}
        
        # Something changed, compute delta
        delta_changes = self._compute_changes(self.last_context, context_for_hash)
        
        # Update stored state
        self.last_context = context_for_hash
        self.last_context_hash = context_hash
        
        # Return context with delta information
        return {
            **context,
            "is_delta": True,
            "delta_changes": delta_changes
        }
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """
        Create a hash representation of context
        
        Args:
            context: Context to hash
            
        Returns:
            Hash string
        """
        import hashlib
        import json
        
        # Sort keys for consistent ordering
        serialized = json.dumps(context, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()
    
    def _compute_changes(self, old_context: Dict[str, Any], new_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute changes between contexts
        
        Args:
            old_context: Previous context
            new_context: Current context
            
        Returns:
            Dictionary of changes
        """
        changes = {
            "added": {},
            "modified": {},
            "removed": {}
        }
        
        # Check for added or modified items
        for key, value in new_context.items():
            if key not in old_context:
                changes["added"][key] = value
            elif old_context[key] != value:
                changes["modified"][key] = {
                    "old": old_context[key],
                    "new": value
                }
        
        # Check for removed items
        for key in old_context:
            if key not in new_context:
                changes["removed"][key] = old_context[key]
        
        return changes
    
    def _calculate_token_usage(self, context: Dict[str, Any]) -> Dict[str, int]:
        """
        Calculate token usage for context components
        
        Args:
            context: Context dictionary
            
        Returns:
            Dictionary with token usage by component
        """
        # Simple estimation based on text length
        def estimate_tokens(obj) -> int:
            if obj is None:
                return 0
            elif isinstance(obj, (str, int, float, bool)):
                # Approximate tokens as 4 characters per token
                return max(1, len(str(obj)) // 4)
            elif isinstance(obj, list):
                return sum(estimate_tokens(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(estimate_tokens(k) + estimate_tokens(v) for k, v in obj.items())
            else:
                return 1
        
        # Calculate tokens for each major component
        token_usage = {}
        
        if "player_stats" in context:
            token_usage["player_stats"] = estimate_tokens(context["player_stats"])
        
        if "npcs" in context:
            token_usage["npcs"] = estimate_tokens(context["npcs"])
        
        if "memories" in context:
            token_usage["memories"] = estimate_tokens(context["memories"])
        
        if "location_details" in context:
            token_usage["location"] = estimate_tokens(context["location_details"])
        
        if "quests" in context:
            token_usage["quests"] = estimate_tokens(context["quests"])
        
        if "time_info" in context:
            token_usage["time"] = estimate_tokens(context["time_info"])
        
        if "current_roleplay" in context:
            token_usage["roleplay"] = estimate_tokens(context["current_roleplay"])
        
        # Calculate remaining fields
        other_keys = [k for k in context if k not in token_usage 
                     and k not in ["token_usage", "is_delta", "delta_changes", "timestamp"]]
        
        if other_keys:
            token_usage["other"] = sum(estimate_tokens(context[k]) for k in other_keys)
        
        return token_usage
    
    async def _trim_to_budget(self, context: Dict[str, Any], budget: int) -> Dict[str, Any]:
        """
        Trim context to fit within token budget
        
        Args:
            context: Context to trim
            budget: Token budget
            
        Returns:
            Trimmed context
        """
        # Get token usage
        token_usage = self._calculate_token_usage(context)
        total = sum(token_usage.values())
        
        # If within budget, return as is
        if total <= budget:
            return context
        
        # Define trim priority (higher value = higher priority to keep)
        trim_priority = {
            "player_stats": 10,      # Highest priority
            "time": 9,               # Very important
            "roleplay": 8,           # Important
            "location": 7,           # Important
            "quests": 6,             # Medium-high priority
            "npcs": 5,               # Medium priority
            "memories": 4,           # Medium-low priority
            "other": 1               # Lowest priority
        }
        
        # Calculate how much to trim
        reduction_needed = total - budget
        
        # Create a working copy
        trimmed = context.copy()
        
        # Sort components by priority (lowest first)
        components = sorted([
            (k, v) for k, v in token_usage.items()
        ], key=lambda x: trim_priority.get(x[0], 0))
        
        # Trim components until within budget
        for component_name, component_tokens in components:
            # Skip if reduction achieved
            if reduction_needed <= 0:
                break
                
            # Skip high-priority components if possible
            priority = trim_priority.get(component_name, 0)
            if priority >= 8 and reduction_needed < total * 0.3:
                continue
                
            # Skip player_stats entirely
            if component_name == "player_stats":
                continue
            
            # Get the component
            if component_name == "npcs" and "npcs" in trimmed:
                # For NPCs, keep most relevant and trim details of others
                npcs = trimmed["npcs"]
                if not npcs:
                    continue
                
                # Sort by relevance
                sorted_npcs = sorted(npcs, key=lambda x: x.get("relevance", 0.5), reverse=True)
                
                # Calculate max to keep fully
                keep_full = max(1, len(sorted_npcs) // 3)  # Keep at least top 1/3
                
                # Create trimmed list
                new_npcs = []
                tokens_saved = 0
                
                for i, npc in enumerate(sorted_npcs):
                    if i < keep_full:
                        # Keep full NPC
                        new_npcs.append(npc)
                    else:
                        # Trim details
                        essential = {
                            "npc_id": npc.get("npc_id"),
                            "npc_name": npc.get("npc_name"),
                            "current_location": npc.get("current_location"),
                            "relevance": npc.get("relevance", 0.5)
                        }
                        
                        # Calculate tokens saved
                        orig_tokens = self._calculate_token_usage({"npc": npc}).get("npc", 0)
                        new_tokens = self._calculate_token_usage({"npc": essential}).get("npc", 0)
                        tokens_saved += orig_tokens - new_tokens
                        
                        new_npcs.append(essential)
                
                # Update context
                trimmed["npcs"] = new_npcs
                
                # Update reduction needed
                reduction_needed -= tokens_saved
            
            elif component_name == "memories" and "memories" in trimmed:
                # For memories, keep most important/relevant and discard others
                memories = trimmed["memories"]
                if not memories:
                    continue
                
                # Sort by importance
                sorted_memories = sorted(memories, 
                                       key=lambda x: x.get("importance", 0.5), 
                                       reverse=True)
                
                # Calculate how many to keep
                memories_budget = budget * 0.15  # Max 15% of budget for memories
                current_tokens = token_usage.get("memories", 0)
                
                if current_tokens > memories_budget:
                    # Keep a portion of memories
                    keep_ratio = memories_budget / current_tokens
                    keep_count = max(1, int(len(sorted_memories) * keep_ratio))
                    
                    # Trim memories
                    new_memories = sorted_memories[:keep_count]
                    tokens_saved = self._calculate_token_usage({"memories": memories}).get("memories", 0) - \
                                  self._calculate_token_usage({"memories": new_memories}).get("memories", 0)
                    
                    # Update context
                    trimmed["memories"] = new_memories
                    
                    # Update reduction needed
                    reduction_needed -= tokens_saved
            
            elif component_name == "quests" and "quests" in trimmed:
                # For quests, keep active ones and summarize others
                quests = trimmed["quests"]
                if not quests:
                    continue
                
                # Prioritize active quests
                active_quests = [q for q in quests if q.get("status") == "active"]
                other_quests = [q for q in quests if q.get("status") != "active"]
                
                # Keep all active quests
                new_quests = active_quests
                
                # Only keep crucial info for other quests
                for quest in other_quests:
                    summary = {
                        "quest_id": quest.get("quest_id"),
                        "quest_name": quest.get("quest_name"),
                        "status": quest.get("status")
                    }
                    new_quests.append(summary)
                
                # Calculate tokens saved
                tokens_saved = self._calculate_token_usage({"quests": quests}).get("quests", 0) - \
                              self._calculate_token_usage({"quests": new_quests}).get("quests", 0)
                
                # Update context
                trimmed["quests"] = new_quests
                
                # Update reduction needed
                reduction_needed -= tokens_saved
        
        # If we still need to trim, remove the lowest priority components entirely
        if reduction_needed > 0:
            for component_name, _ in components:
                # Skip critical components
                if trim_priority.get(component_name, 0) >= 7:
                    continue
                
                # Remove component entirely
                if component_name in trimmed:
                    tokens_saved = token_usage.get(component_name, 0)
                    del trimmed[component_name]
                    reduction_needed -= tokens_saved
                
                # Break if we've reduced enough
                if reduction_needed <= 0:
                    break
        
        return trimmed
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """
        Run maintenance tasks for context optimization
        
        Returns:
            Dictionary with maintenance results
        """
        results = {
            "memory_maintenance": None,
            "vector_maintenance": None,
            "cache_maintenance": None,
            "performance_metrics": None
        }
        
        # 1. Memory maintenance
        memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
        memory_result = await memory_manager.run_maintenance()
        results["memory_maintenance"] = memory_result
        
        # 2. Vector maintenance
        if self.config.is_enabled("use_vector_search"):
            vector_service = await get_vector_service(self.user_id, self.conversation_id)
            # If there were vector maintenance methods, we'd call them here
            results["vector_maintenance"] = {"status": "vector_service_active"}
        
        # 3. Cache maintenance
        cache_items = len(context_cache.l1_cache) + len(context_cache.l2_cache) + len(context_cache.l3_cache)
        results["cache_maintenance"] = {
            "cache_items": cache_items,
            "levels": {
                "l1": len(context_cache.l1_cache),
                "l2": len(context_cache.l2_cache),
                "l3": len(context_cache.l3_cache)
            }
        }
        
        # 4. Performance metrics
        if self.performance_monitor:
            results["performance_metrics"] = self.performance_monitor.get_metrics()
        
        return results

# Global function to get or create context service
async def get_context_service(user_id: int, conversation_id: int) -> UnifiedContextService:
    """
    Get or create a context service instance
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        UnifiedContextService instance
    """
    # Use the cache to avoid creating multiple instances
    cache_key = f"context_service:{user_id}:{conversation_id}"
    
    async def create_service():
        service = UnifiedContextService(user_id, conversation_id)
        await service.initialize()
        return service
    
    # Get from cache or create new with 10 minute TTL (level 2 cache)
    return await context_cache.get(
        cache_key, 
        create_service, 
        cache_level=2, 
        ttl_override=600
    )

# Main function for context retrieval with performance tracking
@track_performance("get_comprehensive_context")
async def get_comprehensive_context(
    user_id: int,
    conversation_id: int,
    input_text: str = "",
    location: Optional[str] = None,
    context_budget: int = 4000,
    use_vector_search: Optional[bool] = None,
    use_delta: bool = True
) -> Dict[str, Any]:
    """
    Get comprehensive context optimized for token efficiency and relevance
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        input_text: Current user input
        location: Optional current location
        context_budget: Token budget (default 4000)
        use_vector_search: Whether to use vector search
        use_delta: Whether to track and return delta changes
        
    Returns:
        Optimized context dictionary
    """
    # Get context service
    service = await get_context_service(user_id, conversation_id)
    
    # Get context
    context = await service.get_context(
        input_text=input_text,
        location=location,
        context_budget=context_budget,
        use_vector_search=use_vector_search,
        use_delta=use_delta
    )
    
    return context

# Cleanup function
async def cleanup_all_services():
    """Close all services"""
    # 1. Clean up context services
    context_service_keys = [
        key for key in context_cache.l1_cache.keys() 
        if key.startswith("context_service:")
    ]
    context_service_keys.extend([
        key for key in context_cache.l2_cache.keys() 
        if key.startswith("context_service:")
    ])
    
    for key in set(context_service_keys):
        service = context_cache.l1_cache.get(key) or context_cache.l2_cache.get(key)
        if service:
            await service.close()
    
    # 2. Clean up vector services
    from context.vector_service import cleanup_vector_services
    await cleanup_vector_services()
    
    # 3. Clean up memory managers
    from context.memory_manager import cleanup_memory_managers
    await cleanup_memory_managers()
    
    # 4. Clear caches
    context_cache.invalidate(None)  # Invalidate all
