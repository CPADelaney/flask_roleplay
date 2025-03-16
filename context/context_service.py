# context/context_service.py

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import hashlib

from context.context_config import get_config
from context.unified_cache import context_cache
from context.vector_service import get_vector_service
from context.memory_manager import get_memory_manager
from context.context_manager import get_context_manager
from context.performance import PerformanceMonitor, track_performance

logger = logging.getLogger(__name__)

class ContextService:
    """
    Unified context service that integrates all context components.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.config = get_config()
        self.initialized = False
        self.performance_monitor = None
        self.context_manager = None
        self.memory_manager = None
        self.vector_service = None
        self.last_context = None
        self.last_context_hash = None
    
    async def initialize(self):
        """Initialize the context service"""
        if self.initialized:
            return
        
        # Get core components (lazy initialization)
        self.context_manager = get_context_manager()
        self.performance_monitor = PerformanceMonitor.get_instance(self.user_id, self.conversation_id)
        
        self.initialized = True
        logger.info(f"Initialized context service for user {self.user_id}, conversation {self.conversation_id}")
    
    async def close(self):
        """Close the context service"""
        self.initialized = False
        logger.info(f"Closed context service for user {self.user_id}, conversation {self.conversation_id}")
    
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
        include_quests: bool = True
    ) -> Dict[str, Any]:
        """
        Get optimized context for the current interaction
        
        Args:
            input_text: Current user input
            location: Optional current location
            context_budget: Maximum token budget
            use_vector_search: Whether to use vector search
            use_delta: Whether to include delta changes
            include_memories: Whether to include memories
            include_npcs: Whether to include NPCs
            include_location: Whether to include location details
            include_quests: Whether to include quests
            
        Returns:
            Optimized context
        """
        # Initialize if needed
        await self.initialize()
        
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
        
        # Add input text hash if not empty
        if input_text:
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
            
            # 2. Get relevant NPC information if requested
            if include_npcs:
                tasks.append(self._get_relevant_npcs(input_text, location, vector_service))
            else:
                tasks.append(asyncio.create_task(asyncio.sleep(0)))
            
            # 3. Get relevant memories if requested
            if include_memories and memory_manager:
                if input_text:
                    tasks.append(memory_manager.search_memories(query_text=input_text, limit=5))
                else:
                    tasks.append(memory_manager.get_recent_memories(days=3, limit=5))
            else:
                tasks.append(asyncio.create_task(asyncio.sleep(0)))
            
            # 4. Get location details if requested
            if include_location:
                tasks.append(self._get_location_details(location, vector_service))
            else:
                tasks.append(asyncio.create_task(asyncio.sleep(0)))
            
            # 5. Get quest information if requested
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
                memories = results[2]
                # Convert memories to dictionaries if they're objects
                memory_dicts = []
                for memory in memories:
                    if hasattr(memory, 'to_dict'):
                        memory_dicts.append(memory.to_dict())
                    else:
                        memory_dicts.append(memory)
                context["memories"] = memory_dicts
            
            if include_location:
                context["location_details"] = results[3]
            
            if include_quests:
                context["quests"] = results[4]
            
            # Add vector-based content if enabled
            if use_vector_search and vector_service:
                try:
                    vector_context = await vector_service.get_context_for_input(
                        input_text=input_text,
                        current_location=location
                    )
                    
                    # Merge vector results if not already included
                    if "npcs" in vector_context and vector_context["npcs"] and not context.get("npcs"):
                        context["npcs"] = vector_context["npcs"]
                    if "locations" in vector_context and vector_context["locations"] and not context.get("location_details"):
                        context["locations"] = vector_context["locations"]
                    if "narratives" in vector_context and vector_context["narratives"]:
                        context["narratives"] = vector_context["narratives"]
                except Exception as e:
                    logger.error(f"Error getting vector context: {e}")
            
            # Add metadata
            context["timestamp"] = datetime.now().isoformat()
            context["context_budget"] = context_budget
            
            # Calculate token usage
            token_usage = self._calculate_token_usage(context)
            context["token_usage"] = token_usage
            context["total_tokens"] = sum(token_usage.values())
            
            # Trim to budget if needed
            if context["total_tokens"] > context_budget:
                context = await self._trim_to_budget(context, context_budget)
                # Recalculate token usage
                token_usage = self._calculate_token_usage(context)
                context["token_usage"] = token_usage
                context["total_tokens"] = sum(token_usage.values())
            
            # Record elapsed time
            context["retrieval_time"] = time.time()
            
            return context
        
        try:
            # Calculate importance based on input
            importance = 0.5
            if input_text:
                if "quest" in input_text.lower() or "mission" in input_text.lower():
                    importance = 0.7
                elif "character" in input_text.lower() or "stats" in input_text.lower():
                    importance = 0.6
            
            # Get from cache or fetch
            context = await context_cache.get(
                cache_key, 
                fetch_context, 
                cache_level=1,  # L1 for short TTL
                importance=importance,
                ttl_override=15  # Short TTL for context
            )
            
            # Apply delta tracking if requested
            if use_delta:
                context = self._apply_delta_tracking(context)
            
            # Record performance metrics
            token_usage = context.get("token_usage", {})
            total_tokens = sum(token_usage.values()) if token_usage else 0
            self.performance_monitor.record_token_usage(total_tokens)
            self.performance_monitor.record_memory_usage()
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            # Return a minimal context in case of error
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_base_context(self, location: Optional[str] = None) -> Dict[str, Any]:
        """
        Get base context from database
        
        Args:
            location: Optional current location
            
        Returns:
            Dictionary with base context
        """
        # Use cache for base context
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
                    time_keys = ["CurrentYear", "CurrentMonth", "CurrentDay", "TimeOfDay"]
                    for key in time_keys:
                        row = await conn.fetchrow("""
                            SELECT value
                            FROM CurrentRoleplay
                            WHERE user_id=$1 AND conversation_id=$2 AND key=$3
                        """, self.user_id, self.conversation_id, key)
                        
                        if row:
                            value = row["value"]
                            if key == "CurrentYear":
                                time_info["year"] = value
                            elif key == "CurrentMonth":
                                time_info["month"] = value
                            elif key == "CurrentDay":
                                time_info["day"] = value
                            elif key == "TimeOfDay":
                                time_info["time_of_day"] = value
                    
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
        
        # Get from cache or fetch
        return await context_cache.get(
            cache_key, 
            fetch_base_context, 
            cache_level=1,
            importance=0.7,
            ttl_override=30  # 30 seconds
        )
    
    async def _get_relevant_npcs(
        self,
        input_text: str,
        location: Optional[str] = None,
        vector_service = None
    ) -> List[Dict[str, Any]]:
        """Get NPCs relevant to current input and location"""
        # Use vector search if available
        if vector_service and input_text:
            try:
                # Get vector context for input
                vector_context = await vector_service.get_context_for_input(
                    input_text=input_text,
                    current_location=location
                )
                
                # Extract NPCs
                if "npcs" in vector_context and vector_context["npcs"]:
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
                query += " ORDER BY closeness DESC, trust DESC LIMIT 10"
                
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
    
    async def _get_location_details(
        self,
        location: Optional[str] = None,
        vector_service = None
    ) -> Dict[str, Any]:
        """Get details about the current location"""
        if not location:
            return {}
        
        # Use vector search if available
        if vector_service:
            try:
                # Get vector context for location
                vector_context = await vector_service.get_context_for_input(
                    input_text=f"Location: {location}",
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
        """Get information about active quests"""
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
        """Apply delta tracking to context"""
        # Calculate hash of context (excluding certain fields)
        context_for_hash = {k: v for k, v in context.items() 
                           if k not in ["timestamp", "token_usage", "is_delta", "delta_changes", "retrieval_time"]}
        
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
        """Create a hash representation of context"""
        serialized = json.dumps(context, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode('utf-8')).hexdigest()
    
    def _compute_changes(self, old_context: Dict[str, Any], new_context: Dict[str, Any]) -> Dict[str, Any]:
        """Compute changes between contexts"""
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
        """Calculate token usage for context components"""
        # Simple estimation based on text length
        def estimate_tokens(obj):
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
        
        components = {
            "player_stats": ["player_stats"],
            "npcs": ["npcs"],
            "memories": ["memories"],
            "location": ["location_details", "locations"],
            "quests": ["quests"],
            "time": ["time_info"],
            "roleplay": ["current_roleplay"],
            "narratives": ["narratives"]
        }
        
        for category, keys in components.items():
            total = 0
            for key in keys:
                if key in context:
                    total += estimate_tokens(context[key])
            if total > 0:
                token_usage[category] = total
        
        # Calculate remaining fields
        skip_keys = set()
        for keys in components.values():
            skip_keys.update(keys)
        skip_keys.update(["token_usage", "is_delta", "delta_changes", "timestamp", "total_tokens"])
        
        other_keys = [k for k in context if k not in skip_keys]
        if other_keys:
            token_usage["other"] = sum(estimate_tokens(context[k]) for k in other_keys)
        
        return token_usage
    
    async def _trim_to_budget(self, context: Dict[str, Any], budget: int) -> Dict[str, Any]:
        """Trim context to fit within token budget"""
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
            "narratives": 3,         # Lower priority
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
            
            # Different trimming strategies for different components
            if component_name == "npcs" and "npcs" in trimmed:
                # For NPCs, keep most relevant and trim details of others
                npcs = trimmed["npcs"]
                if not npcs:
                    continue
                
                # Sort by relevance
                sorted_npcs = sorted(npcs, key=lambda x: x.get("relevance", 0.5), reverse=True)
                
                # Keep top 1/3 fully, trim the rest
                keep_full = max(1, len(sorted_npcs) // 3)
                
                new_npcs = []
                for i, npc in enumerate(sorted_npcs):
                    if i < keep_full:
                        new_npcs.append(npc)
                    else:
                        # Only keep essential info
                        new_npcs.append({
                            "npc_id": npc.get("npc_id"),
                            "npc_name": npc.get("npc_name"),
                            "current_location": npc.get("current_location"),
                            "relevance": npc.get("relevance", 0.5)
                        })
                
                # Calculate tokens saved
                old_tokens = token_usage[component_name]
                new_tokens = self._calculate_token_usage({"npcs": new_npcs}).get("npcs", 0)
                
                # Update context and reduction needed
                trimmed["npcs"] = new_npcs
                reduction_needed -= (old_tokens - new_tokens)
            
            elif component_name == "memories" and "memories" in trimmed:
                # For memories, keep most important ones
                memories = trimmed["memories"]
                if not memories:
                    continue
                
                # Sort by importance
                sorted_memories = sorted(memories, key=lambda x: x.get("importance", 0.5), reverse=True)
                
                # Keep only top half
                keep_count = max(1, len(sorted_memories) // 2)
                new_memories = sorted_memories[:keep_count]
                
                # Calculate tokens saved
                old_tokens = token_usage[component_name]
                new_tokens = self._calculate_token_usage({"memories": new_memories}).get("memories", 0)
                
                # Update context and reduction needed
                trimmed["memories"] = new_memories
                reduction_needed -= (old_tokens - new_tokens)
            
            elif component_name == "narratives" and "narratives" in trimmed:
                # For narratives, we can remove entirely if needed
                old_tokens = token_usage[component_name]
                del trimmed["narratives"]
                reduction_needed -= old_tokens
        
        # If we still need to trim, remove the lowest priority components entirely
        if reduction_needed > 0:
            for component_name, _ in components:
                # Skip critical components
                if trim_priority.get(component_name, 0) >= 7:
                    continue
                
                # Remove component entirely
                if component_name in token_usage and component_name in trimmed:
                    component_keys = []
                    
                    # Find all keys for this component
                    for category, keys in components.items():
                        if category == component_name:
                            component_keys.extend(keys)
                    
                    # Remove all keys for this component
                    for key in component_keys:
                        if key in trimmed:
                            del trimmed[key]
                    
                    reduction_needed -= token_usage[component_name]
                
                # Break if we've reduced enough
                if reduction_needed <= 0:
                    break
        
        return trimmed
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """Run maintenance tasks for context optimization"""
        # Initialize if needed
        await self.initialize()
        
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
        results["performance_metrics"] = self.performance_monitor.get_metrics()
        
        return results


# Global service registry
_context_services = {}

async def get_context_service(user_id: int, conversation_id: int) -> ContextService:
    """Get or create a context service instance"""
    key = f"{user_id}:{conversation_id}"
    
    if key not in _context_services:
        service = ContextService(user_id, conversation_id)
        await service.initialize()
        _context_services[key] = service
    
    return _context_services[key]

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
        context_budget: Token budget
        use_vector_search: Whether to use vector search
        use_delta: Whether to include delta changes
        
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

async def cleanup_context_services():
    """Close all context services"""
    global _context_services
    
    # Close each service
    for service in _context_services.values():
        await service.close()
    
    # Clear registry
    _context_services.clear()
    
    # Close other components
    from context.vector_service import cleanup_vector_services
    from context.memory_manager import cleanup_memory_managers
    
    await cleanup_vector_services()
    await cleanup_memory_managers()
