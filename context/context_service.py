# context/context_service.py

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
        use_delta: bool = True
    ) -> Dict[str, Any]:
        """
        Get optimized context for the current interaction
        
        Args:
            input_text: Current user input
            location: Optional current location
            context_budget: Maximum token budget
            use_vector_search: Whether to use vector search
            use_delta: Whether to include delta changes
            
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
            f"vec:{use_vector_search}"
        ]
        
        # Add input text hash if not empty
        if input_text:
            import hashlib
            text_hash = hashlib.md5(input_text.encode()).hexdigest()[:8]
            cache_key_parts.append(f"input:{text_hash}")
        
        cache_key = ":".join(cache_key_parts)
        
        # Function to fetch context if not in cache
        async def fetch_context():
            # Record start time
            start_time = time.time()
            
            # Get base context
            context = await self._get_base_context(location)
            
            # Add memories if available
            if self.config.get("features", "use_memories", True):
                memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
                
                if input_text:
                    # Search memories relevant to input
                    memories = await memory_manager.search_memories(
                        query_text=input_text,
                        limit=5,
                        use_vector=use_vector_search
                    )
                else:
                    # Get recent memories
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
                
                context["memories"] = memory_dicts
            
            # Add vector-based content if enabled
            if use_vector_search:
                vector_service = await get_vector_service(self.user_id, self.conversation_id)
                vector_context = await vector_service.get_context_for_input(
                    input_text=input_text,
                    current_location=location
                )
                
                # Merge vector results
                if "npcs" in vector_context and vector_context["npcs"]:
                    context["npcs"] = vector_context["npcs"]
                if "locations" in vector_context and vector_context["locations"]:
                    context["locations"] = vector_context["locations"]
                if "narratives" in vector_context and vector_context["narratives"]:
                    context["narratives"] = vector_context["narratives"]
            
            # Trim to budget if needed
                token_usage = self._estimate_token_usage(context)
                total_tokens = sum(token_usage.values())
                
                if total_tokens > context_budget:
                    context = await self._trim_to_budget(context, context_budget)
                    # Recalculate token usage
                    token_usage = self._estimate_token_usage(context)
                
                # Add metadata
                context["timestamp"] = datetime.now().isoformat()
                context["context_budget"] = context_budget
                context["token_usage"] = token_usage
                context["total_tokens"] = sum(token_usage.values())
                
                # Record elapsed time
                elapsed = time.time() - start_time
                context["retrieval_time"] = elapsed
                
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
                if use_delta and self.context_manager:
                    # Get current version to check for changes
                    current_state = await self.context_manager.get_context()
                    
                    # Only compute delta if we're not coming from cache
                    if context.get("retrieval_time"):
                        # Update the context manager with our current context
                        was_changed, changes = await self.context_manager.update_context(context)
                        
                        if was_changed:
                            context["is_delta"] = True
                            context["delta_changes"] = changes
                        else:
                            context["is_delta"] = False
                            context["no_changes"] = True
                
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
        
        def _estimate_token_usage(self, context: Dict[str, Any]) -> Dict[str, int]:
            """
            Estimate token usage for context components
            
            Args:
                context: Context dictionary
                
            Returns:
                Dictionary with token usage by component
            """
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
            """
            Trim context to fit within token budget
            
            Args:
                context: Context to trim
                budget: Token budget
                
            Returns:
                Trimmed context
            """
            # Get token usage
            token_usage = self._estimate_token_usage(context)
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
                    
                    # Calculate max to keep fully
                    keep_full = max(1, len(sorted_npcs) // 3)  # Keep at least top 1/3
                    
                    # Create trimmed list
                    new_npcs = []
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
                            new_npcs.append(essential)
                    
                    # Update context
                    trimmed["npcs"] = new_npcs
                    
                    # Estimate reduction
                    old_tokens = token_usage[component_name]
                    new_tokens = self._estimate_token_usage({"npcs": new_npcs}).get("npcs", 0)
                    reduction_achieved = old_tokens - new_tokens
                    reduction_needed -= reduction_achieved
                
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
                    
                    # Update context
                    trimmed["memories"] = new_memories
                    
                    # Estimate reduction
                    old_tokens = token_usage[component_name]
                    new_tokens = self._estimate_token_usage({"memories": new_memories}).get("memories", 0)
                    reduction_achieved = old_tokens - new_tokens
                    reduction_needed -= reduction_achieved
                
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
                        del trimmed[component_name]
                        reduction_needed -= token_usage[component_name]
                    
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
        """
        Get or create a context service instance
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            ContextService instance
        """
        global _context_services
        
        key = f"{user_id}:{conversation_id}"
        
        if key not in _context_services:
            service = ContextService(user_id, conversation_id)
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
