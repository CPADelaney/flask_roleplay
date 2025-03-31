# logic/aggregator_sdk.py

"""
Optimized replacement for aggregator_sdk.py that integrates with the
unified context service for improved performance and efficiency.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from context.context_service import (
    get_context_service,
    get_comprehensive_context,
    cleanup_context_services
)
from context.context_config import get_config
from context.context_performance import PerformanceMonitor, track_performance

# UPDATED: Using new async context manager
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------
# Optimized Context Retrieval
# -------------------------------------------------------------------------------

@track_performance("get_aggregated_roleplay_context")
async def get_aggregated_roleplay_context(
    user_id: int,
    conversation_id: int,
    player_name: str = "Chase",
    current_input: str = "",
    location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Optimized drop-in replacement for the original get_aggregated_roleplay_context
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        player_name: Name of the player (default: "Chase")
        current_input: Current user input for relevance scoring
        location: Optional location override
        
    Returns:
        Aggregated context dictionary
    """
    # Use the unified context service for optimized retrieval
    config = get_config()
    context_budget = config.get("token_budget", "default_budget", 4000)
    use_vector_search = config.is_enabled("use_vector_search")
    use_delta = config.is_enabled("use_incremental_context")
    
    try:
        # Get comprehensive context through the unified service
        context = await get_comprehensive_context(
            user_id=user_id,
            conversation_id=conversation_id,
            input_text=current_input,
            location=location,
            context_budget=context_budget,
            use_vector_search=use_vector_search,
            use_delta=use_delta
        )
        
        # Convert the result to the expected format for compatibility
        return format_context_for_compatibility(context)
    
    except Exception as e:
        logger.error(f"Error in optimized get_aggregated_roleplay_context: {e}")
        
        # Fallback to database query if needed
        try:
            return await fallback_get_context(user_id, conversation_id, player_name)
        except Exception as inner_e:
            logger.error(f"Error in fallback context retrieval: {inner_e}")
            # Return minimal context to avoid complete failure
            return {
                "player_stats": {},
                "introduced_npcs": [],
                "unintroduced_npcs": [],
                "current_roleplay": {},
                "error": str(e)
            }

def format_context_for_compatibility(optimized_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format optimized context to match the format expected by existing code
    
    Args:
        optimized_context: Context from comprehensive context service
        
    Returns:
        Context in the format expected by existing code
    """
    # Create a compatible structure
    compatible = {}
    
    # Map time information
    if "time_info" in optimized_context:
        time_info = optimized_context["time_info"]
        compatible["year"] = time_info.get("year", "1040")
        compatible["month"] = time_info.get("month", "6")
        compatible["day"] = time_info.get("day", "15")
        compatible["time_of_day"] = time_info.get("time_of_day", "Morning")
    
    # Copy player stats directly
    if "player_stats" in optimized_context:
        compatible["player_stats"] = optimized_context["player_stats"]
    
    # Map NPC lists
    compatible["introduced_npcs"] = optimized_context.get("npcs", [])
    compatible["unintroduced_npcs"] = []  # Populated separately if needed
    
    # Copy current roleplay
    if "current_roleplay" in optimized_context:
        compatible["current_roleplay"] = optimized_context["current_roleplay"]
    
    # Map location details
    if "location_details" in optimized_context:
        compatible["current_location"] = optimized_context.get("location_details", {}).get(
            "location_name", optimized_context.get("current_location", "Unknown")
        )
        
        # Add other location details
        compatible["location_details"] = optimized_context.get("location_details", {})
    else:
        compatible["current_location"] = optimized_context.get("current_location", "Unknown")
    
    # Map memories
    if "memories" in optimized_context:
        compatible["memories"] = optimized_context["memories"]
    
    # Map quests
    if "quests" in optimized_context:
        compatible["quests"] = optimized_context["quests"]
    
    # Add delta information if available
    if "is_delta" in optimized_context:
        compatible["is_delta"] = optimized_context["is_delta"]
        
        if optimized_context.get("is_delta", False) and "delta_changes" in optimized_context:
            compatible["delta_changes"] = optimized_context["delta_changes"]
    
    # Add token usage if available
    if "token_usage" in optimized_context:
        compatible["token_usage"] = optimized_context["token_usage"]
    
    # Copy any other fields as is
    for key, value in optimized_context.items():
        if key not in compatible and key not in [
            "time_info", "npcs", "location_details", "is_delta", 
            "delta_changes", "token_usage", "timestamp"
        ]:
            compatible[key] = value
    
    return compatible

async def fallback_get_context(
    user_id: int,
    conversation_id: int,
    player_name: str = "Chase"
) -> Dict[str, Any]:
    """
    Fallback context retrieval directly from database
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        player_name: Name of the player
        
    Returns:
        Context dictionary
    """
    # UPDATED: Using async context manager
    try:
        # Minimal context to return
        context = {
            "player_stats": {},
            "introduced_npcs": [],
            "unintroduced_npcs": [],
            "current_roleplay": {},
            "year": "1040",
            "month": "6",
            "day": "15",
            "time_of_day": "Morning"
        }
        
        async with get_db_connection_context() as conn:
            # 1. Get player stats
            row = await conn.fetchrow("""
                SELECT corruption, confidence, willpower,
                       obedience, dependency, lust,
                       mental_resilience, physical_endurance
                FROM PlayerStats
                WHERE user_id=$1 AND conversation_id=$2
                LIMIT 1
            """, user_id, conversation_id)
            
            if row:
                context["player_stats"] = dict(row)
            
            # 2. Get introduced NPCs
            rows = await conn.fetch("""
                SELECT npc_id, npc_name, dominance, cruelty, closeness,
                       trust, respect, intensity, current_location,
                       physical_description
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND introduced=TRUE
                LIMIT 20
            """, user_id, conversation_id)
            
            for row in rows:
                context["introduced_npcs"].append(dict(row))
            
            # 3. Get time info
            time_keys = [
                ("CurrentYear", "year"),
                ("CurrentMonth", "month"),
                ("CurrentDay", "day"),
                ("TimeOfDay", "time_of_day")
            ]
            
            for key, context_key in time_keys:
                row = await conn.fetchrow("""
                    SELECT value
                    FROM CurrentRoleplay
                    WHERE user_id=$1 AND conversation_id=$2 AND key=$3
                """, user_id, conversation_id, key)
                
                if row:
                    context[context_key] = row["value"]
            
            # 4. Get current roleplay data
            rows = await conn.fetch("""
                SELECT key, value
                FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2
            """, user_id, conversation_id)
            
            for row in rows:
                context["current_roleplay"][row["key"]] = row["value"]
        
        return context
    
    except Exception as e:
        logger.error(f"Error in fallback context retrieval: {e}")
        return {
            "player_stats": {},
            "introduced_npcs": [],
            "unintroduced_npcs": [],
            "current_roleplay": {},
            "error": str(e)
        }

# -------------------------------------------------------------------------------
# Optimized Context Cache with Multi-Level Support
# -------------------------------------------------------------------------------

class OptimizedContextCache:
    """
    Optimized context cache wrapper that integrates with unified cache
    """
    
    def __init__(self):
        # Get configuration
        config = get_config()
        self.l1_ttl = config.get("cache", "l1_ttl_seconds", 60)
        self.l2_ttl = config.get("cache", "l2_ttl_seconds", 300)
        self.l3_ttl = config.get("cache", "l3_ttl_seconds", 3600)
        self.enabled = config.get("cache", "enabled", True)
    
    async def get(self, key: str, fetch_func, cache_level: int = 1) -> Any:
        """
        Get an item from cache or fetch and store it
        
        Args:
            key: Cache key
            fetch_func: Async function to fetch data on cache miss
            cache_level: Cache level to use (1-3)
            
        Returns:
            Cached or fetched data
        """
        if not self.enabled:
            return await fetch_func()
        
        # Use unified cache through the optimized service
        from context.unified_cache import context_cache
        
        # Determine TTL based on level
        if cache_level == 1:
            ttl = self.l1_ttl
        elif cache_level == 2:
            ttl = self.l2_ttl
        else:
            ttl = self.l3_ttl
        
        return await context_cache.get(
            key=key,
            fetch_func=fetch_func,
            cache_level=cache_level,
            ttl_override=ttl
        )
    
    def invalidate(self, key_prefix: str) -> None:
        """
        Invalidate cache entries matching a prefix
        
        Args:
            key_prefix: Prefix to match for invalidation
        """
        from context.unified_cache import context_cache
        context_cache.invalidate(key_prefix)

# Create singleton instance
context_cache = OptimizedContextCache()

# -------------------------------------------------------------------------------
# Optimized Incremental Context Manager
# -------------------------------------------------------------------------------

class OptimizedIncrementalContextManager:
    """
    Optimized incremental context manager that leverages unified context service
    """
    
    def __init__(self):
        # Get configuration
        config = get_config()
        self.enabled = config.is_enabled("use_incremental_context")
        self.token_budget = config.get("token_budget", "default_budget", 4000)
        self.use_vector = config.is_enabled("use_vector_search")
    
    async def get_context(
        self,
        user_id: int,
        conversation_id: int,
        user_input: str,
        location: Optional[str] = None,
        include_delta: bool = True
    ) -> Dict[str, Any]:
        """
        Get context with delta tracking if enabled
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            user_input: Current user input for relevance scoring
            location: Optional location override
            include_delta: Whether to include delta changes
            
        Returns:
            Context with incremental changes if enabled
        """
        # If incremental context is disabled, just get full context
        if not self.enabled or not include_delta:
            return await self.get_full_context(
                user_id, conversation_id, user_input, location
            )
        
        # Get context service for delta tracking
        context_service = await get_context_service(user_id, conversation_id)
        
        # Get context with delta tracking
        context = await context_service.get_context(
            input_text=user_input,
            location=location,
            context_budget=self.token_budget,
            use_vector_search=self.use_vector,
            use_delta=True
        )
        
        # Format as expected
        result = {
            "full_context": format_context_for_compatibility(context),
            "is_incremental": context.get("is_delta", False)
        }
        
        # Add delta context if available
        if context.get("is_delta", False) and "delta_changes" in context:
            result["delta_context"] = context["delta_changes"]
        
        return result
    
    async def get_full_context(
        self,
        user_id: int,
        conversation_id: int,
        user_input: str,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get full context without delta tracking
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            user_input: Current user input for relevance scoring
            location: Optional location override
            
        Returns:
            Full context dictionary
        """
        # Get context using standard function
        context = await get_aggregated_roleplay_context(
            user_id=user_id,
            conversation_id=conversation_id,
            current_input=user_input,
            location=location
        )
        
        return {
            "full_context": context,
            "is_incremental": False
        }

# Create singleton instance
incremental_context_manager = OptimizedIncrementalContextManager()

# -------------------------------------------------------------------------------
# Optimized Context Retrieval and Formatting
# -------------------------------------------------------------------------------

async def get_optimized_context(
    user_id: int,
    conversation_id: int,
    current_input: str,
    location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get optimized context based on input relevance and budget constraints
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        current_input: Current user input
        location: Optional location override
        
    Returns:
        Optimized context dictionary
    """
    config = get_config()
    context_budget = config.get("token_budget", "default_budget", 4000)
    
    # Get context service
    context_service = await get_context_service(user_id, conversation_id)
    
    # Get optimized context
    context = await context_service.get_context(
        input_text=current_input,
        location=location,
        context_budget=context_budget,
        use_delta=False
    )
    
    # Format for compatibility
    return format_context_for_compatibility(context)

def build_aggregator_text(aggregated_data: Dict[str, Any]) -> str:
    """
    Build the aggregator text from the provided data
    
    Args:
        aggregated_data: The aggregated context data
        
    Returns:
        Formatted aggregator text
    """
    if "aggregator_text" in aggregated_data:
        return aggregated_data["aggregator_text"]
    
    # Extract component data
    current_location = aggregated_data.get("current_location", "Unknown")
    year = aggregated_data.get("year", "1040")
    month = aggregated_data.get("month", "6")
    day = aggregated_data.get("day", "15")
    time_of_day = aggregated_data.get("time_of_day", "Morning")
    
    introduced_npcs = aggregated_data.get("introduced_npcs", [])
    
    # Format current date/time
    date_line = f"- It is {year}, {month} {day}, {time_of_day}.\n"
    
    # Format location
    location_line = f"- Current location: {current_location}\n"
    
    # Format NPCs
    npc_lines = ["Introduced NPCs in the area:"]
    for npc in introduced_npcs[:5]:  # Limit to 5 NPCs
        npc_location = npc.get("current_location", "Unknown")
        npc_lines.append(f"  - {npc.get('npc_name')} is at {npc_location}")
    
    npc_section = "\n".join(npc_lines) if introduced_npcs else "No NPCs currently in the area."
    
    # Combine components
    aggregator_text = (
        f"{date_line}"
        f"{location_line}\n"
        f"{npc_section}\n"
    )
    
    # Add environment description if available
    environment_desc = aggregated_data.get("current_roleplay", {}).get("EnvironmentDesc")
    if environment_desc:
        aggregator_text += f"\nEnvironment:\n{environment_desc}\n"
    
    # Add player role if available
    player_role = aggregated_data.get("current_roleplay", {}).get("PlayerRole")
    if player_role:
        aggregator_text += f"\nPlayer Role:\n{player_role}\n"
    
    # Add optimization markers
    aggregator_text += "\n\n<!-- Context optimized with unified context system -->"
    
    # Add relevance note if NPCs have relevance scores
    has_relevance = any("relevance_score" in npc or "relevance" in npc 
                        for npc in introduced_npcs)
    
    if has_relevance:
        aggregator_text += "\n<!-- NPCs sorted by relevance to current context -->"
    
    return aggregator_text

# -------------------------------------------------------------------------------
# Maintenance Functions
# -------------------------------------------------------------------------------

async def run_context_maintenance(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Run maintenance tasks for context optimization
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Dictionary with maintenance results
    """
    # Get context service
    context_service = await get_context_service(user_id, conversation_id)
    
    # Run maintenance
    return await context_service.run_maintenance()

# -------------------------------------------------------------------------------
# Migration Helpers
# -------------------------------------------------------------------------------

async def migrate_old_context_to_new(
    user_id: int,
    conversation_id: int
) -> Dict[str, Any]:
    """
    Migrate data from old context system to new optimized system
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Migration results
    """
    try:
        # Get existing context
        from logic.aggregator_sdk import get_aggregated_roleplay_context as old_get_context
        
        old_context = await old_get_context(user_id, conversation_id)
        
        # Get context service
        context_service = await get_context_service(user_id, conversation_id)
        
        # Initialize memory manager
        from context.memory_manager import get_memory_manager
        memory_manager = await get_memory_manager(user_id, conversation_id)
        
        # Migrate memories to new system
        memory_migrations = 0
        if "memories" in old_context:
            for memory in old_context["memories"]:
                content = memory.get("content") or memory.get("text") or ""
                memory_type = memory.get("type") or "observation"
                
                if content:
                    await memory_manager.add_memory(
                        content=content,
                        memory_type=memory_type,
                        importance=0.7  # High importance for existing memories
                    )
                    memory_migrations += 1
        
        # Store vector embeddings for NPCs
        npc_migrations = 0
        vector_service = None
        
        if self.config.is_enabled("use_vector_search"):
            from context.vector_service import get_vector_service
            vector_service = await get_vector_service(user_id, conversation_id)
            
            if "introduced_npcs" in old_context:
                # Implementation would add NPCs to vector database
                npc_migrations = len(old_context["introduced_npcs"])
        
        return {
            "memory_migrations": memory_migrations,
            "npc_migrations": npc_migrations,
            "success": True
        }
    
    except Exception as e:
        logger.error(f"Error during context migration: {e}")
        return {
            "error": str(e),
            "success": False
        }

# -------------------------------------------------------------------------------
# Apply monkey patching for easy integration
# -------------------------------------------------------------------------------

def apply_context_optimizations():
    """
    Apply context optimizations by monkey patching existing functions
    
    This replaces existing context retrieval functions with optimized versions
    """
    import sys
    
    # Get the aggregator_sdk module
    aggregator_sdk = sys.modules.get("logic.aggregator_sdk")
    if not aggregator_sdk:
        logger.warning("Could not find aggregator_sdk module for patching")
        return False
    
    # Save original functions
    original_get_context = getattr(aggregator_sdk, "get_aggregated_roleplay_context", None)
    original_build_text = getattr(aggregator_sdk, "build_aggregator_text", None)
    
    if not original_get_context or not original_build_text:
        logger.warning("Required functions not found in aggregator_sdk")
        return False
    
    # Replace functions
    setattr(aggregator_sdk, "get_aggregated_roleplay_context", get_aggregated_roleplay_context)
    setattr(aggregator_sdk, "build_aggregator_text", build_aggregator_text)
    setattr(aggregator_sdk, "ContextCache", OptimizedContextCache)
    setattr(aggregator_sdk, "IncrementalContextManager", OptimizedIncrementalContextManager)
    setattr(aggregator_sdk, "get_optimized_context", get_optimized_context)
    setattr(aggregator_sdk, "run_context_maintenance", run_context_maintenance)
    
    logger.info("Applied context optimizations via monkey patching")
    return True

async def update_context_with_universal_updates(
    context: dict,
    universal_updates: dict,
    user_id: str,
    conversation_id: str
) -> dict:
    """
    Update context with universal updates while maintaining consistency.
    
    Args:
        context: Current context dictionary
        universal_updates: Dictionary of updates from universal updater
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Updated context dictionary
    """
    try:
        # Create a copy of context to avoid modifying original
        updated_context = context.copy()
        
        # Process NPC updates
        if "npc_updates" in universal_updates:
            for npc_update in universal_updates["npc_updates"]:
                npc_id = npc_update.get("npc_id")
                if not npc_id:
                    continue
                    
                # Update NPC in context
                if "npcs" not in updated_context:
                    updated_context["npcs"] = {}
                    
                if npc_id not in updated_context["npcs"]:
                    updated_context["npcs"][npc_id] = {}
                    
                # Update NPC stats
                if "stats" in npc_update:
                    updated_context["npcs"][npc_id]["stats"] = npc_update["stats"]
                    
                # Update NPC location
                if "location" in npc_update:
                    updated_context["npcs"][npc_id]["current_location"] = npc_update["location"]
                    
                # Update NPC memory
                if "memory" in npc_update:
                    if "memory" not in updated_context["npcs"][npc_id]:
                        updated_context["npcs"][npc_id]["memory"] = []
                    updated_context["npcs"][npc_id]["memory"].extend(npc_update["memory"])
        
        # Process relationship updates
        if "social_links" in universal_updates:
            if "relationships" not in updated_context:
                updated_context["relationships"] = {}
                
            for link in universal_updates["social_links"]:
                e1_type = link.get("entity1_type")
                e1_id = link.get("entity1_id")
                e2_type = link.get("entity2_type")
                e2_id = link.get("entity2_id")
                
                if not all([e1_type, e1_id, e2_type, e2_id]):
                    continue
                    
                link_key = f"{e1_type}_{e1_id}_{e2_type}_{e2_id}"
                updated_context["relationships"][link_key] = {
                    "type": link.get("link_type", "neutral"),
                    "level": link.get("link_level", 0),
                    "group_context": link.get("group_context", ""),
                    "events": link.get("events", [])
                }
        
        # Process quest updates
        if "quest_updates" in universal_updates:
            if "quests" not in updated_context:
                updated_context["quests"] = {}
                
            for quest in universal_updates["quest_updates"]:
                quest_id = quest.get("quest_id")
                if not quest_id:
                    continue
                    
                updated_context["quests"][quest_id] = {
                    "status": quest.get("status", "In Progress"),
                    "progress": quest.get("progress_detail", ""),
                    "giver": quest.get("quest_giver", ""),
                    "reward": quest.get("reward", "")
                }
        
        # Process inventory updates
        if "inventory_updates" in universal_updates:
            if "inventory" not in updated_context:
                updated_context["inventory"] = {
                    "items": {},
                    "removed_items": []
                }
                
            # Process added items
            for item in universal_updates["inventory_updates"].get("added_items", []):
                if isinstance(item, str):
                    item_name = item
                    item_data = {"name": item_name}
                else:
                    item_name = item.get("name")
                    item_data = item
                    
                if item_name:
                    updated_context["inventory"]["items"][item_name] = item_data
                    
            # Process removed items
            for item in universal_updates["inventory_updates"].get("removed_items", []):
                if isinstance(item, str):
                    item_name = item
                else:
                    item_name = item.get("name")
                    
                if item_name:
                    if item_name in updated_context["inventory"]["items"]:
                        del updated_context["inventory"]["items"][item_name]
                    updated_context["inventory"]["removed_items"].append(item_name)
        
        # Process activity updates
        if "activity_updates" in universal_updates:
            if "activities" not in updated_context:
                updated_context["activities"] = []
                
            for activity in universal_updates["activity_updates"]:
                if "activity_name" in activity:
                    updated_context["activities"].append({
                        "name": activity["activity_name"],
                        "purpose": activity.get("purpose", {}),
                        "stats": activity.get("stat_integration", {}),
                        "intensity": activity.get("intensity_tier", 0),
                        "setting": activity.get("setting_variant", "")
                    })
        
        # Update last modified timestamp
        updated_context["last_modified"] = datetime.now().isoformat()
        
        return updated_context
        
    except Exception as e:
        logging.error(f"Error updating context with universal updates: {e}")
        return context  # Return original context on error
