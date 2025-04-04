# aggregator_sdk.py

"""
Optimized replacement for aggregator.py that integrates with the
unified context service for improved performance and efficiency.

Usage:
  1) Import in your main or startup code:
       from logic.aggregator_sdk import init_singletons
  2) Call `await init_singletons()` once in an async startup routine
     to initialize global singletons like context_cache or incremental_context_manager.

Then use:
  from logic.aggregator_sdk import (
      get_aggregated_roleplay_context,
      context_cache,
      incremental_context_manager,
      ...
  )
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

# Using new async context manager for DB connections
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

###############################################################################
# Global Singletons: Will be set by init_singletons()
###############################################################################
context_cache: Optional["OptimizedContextCache"] = None
incremental_context_manager: Optional["OptimizedIncrementalContextManager"] = None


###############################################################################
# Public Initialization
###############################################################################
async def init_singletons() -> None:
    """
    Initialize global singletons in aggregator_sdk. Must be called once in an async context.
    For example, in main.py after creating your Flask app, do:

        import asyncio
        from logic.aggregator_sdk import init_singletons

        async def startup():
            await init_singletons()

        if __name__ == "__main__":
            asyncio.run(startup())
            # Then run your Flask/Quart server

    or, if you're using an async framework's startup event (FastAPI, etc.):

        @app.on_event("startup")
        async def startup():
            await init_singletons()

    After calling init_singletons(), you can safely access context_cache or
    incremental_context_manager in aggregator_sdk.
    """
    global context_cache, incremental_context_manager
    logger.info("Initializing aggregator_sdk singletons...")

    context_cache = await OptimizedContextCache.create()
    incremental_context_manager = await OptimizedIncrementalContextManager.create()

    logger.info("Aggregator SDK singletons initialized successfully.")


###############################################################################
# Main Context Retrieval
###############################################################################

@track_performance("get_aggregated_roleplay_context")
async def get_aggregated_roleplay_context(
    user_id: int,
    conversation_id: int,
    player_name: str = "Chase",
    current_input: str = "",
    location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Optimized drop-in replacement for the original get_aggregated_roleplay_context.

    Args:
        user_id: User ID
        conversation_id: Conversation ID
        player_name: Name of the player (default: "Chase")
        current_input: Current user input for relevance scoring
        location: Optional location override

    Returns:
        Aggregated context dictionary
    """
    config = await get_config()
    context_budget = config.get("token_budget", "default_budget", 4000)
    use_vector_search = config.is_enabled("use_vector_search")
    use_delta = config.is_enabled("use_incremental_context")

    try:
        context_data = await get_comprehensive_context(
            user_id=user_id,
            conversation_id=conversation_id,
            input_text=current_input,
            location=location,
            context_budget=context_budget,
            use_vector_search=use_vector_search,
            use_delta=use_delta
        )
        return format_context_for_compatibility(context_data)

    except Exception as e:
        logger.error(f"Error in get_aggregated_roleplay_context: {e}", exc_info=True)
        # Fallback to DB if needed
        try:
            return await fallback_get_context(user_id, conversation_id, player_name)
        except Exception as fallback_err:
            logger.error(f"Error in fallback context retrieval: {fallback_err}", exc_info=True)
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
    Format optimized context to match the format expected by existing code.
    """
    compatible = {}

    # Time info
    if "time_info" in optimized_context:
        time_info = optimized_context["time_info"]
        compatible["year"] = time_info.get("year", "1040")
        compatible["month"] = time_info.get("month", "6")
        compatible["day"] = time_info.get("day", "15")
        compatible["time_of_day"] = time_info.get("time_of_day", "Morning")

    # Player stats
    if "player_stats" in optimized_context:
        compatible["player_stats"] = optimized_context["player_stats"]

    # NPC lists
    compatible["introduced_npcs"] = optimized_context.get("npcs", [])
    compatible["unintroduced_npcs"] = []

    # Current roleplay
    if "current_roleplay" in optimized_context:
        compatible["current_roleplay"] = optimized_context["current_roleplay"]

    # Location details
    if "location_details" in optimized_context:
        location_details = optimized_context["location_details"]
        compatible["current_location"] = location_details.get(
            "location_name",
            optimized_context.get("current_location", "Unknown")
        )
        compatible["location_details"] = location_details
    else:
        compatible["current_location"] = optimized_context.get("current_location", "Unknown")

    # Memories
    if "memories" in optimized_context:
        compatible["memories"] = optimized_context["memories"]

    # Quests
    if "quests" in optimized_context:
        compatible["quests"] = optimized_context["quests"]

    # Delta info
    if "is_delta" in optimized_context:
        compatible["is_delta"] = optimized_context["is_delta"]
        if optimized_context.get("is_delta", False) and "delta_changes" in optimized_context:
            compatible["delta_changes"] = optimized_context["delta_changes"]

    # Token usage
    if "token_usage" in optimized_context:
        compatible["token_usage"] = optimized_context["token_usage"]

    # Copy any other fields
    ignored_keys = {
        "time_info", "npcs", "location_details", "is_delta",
        "delta_changes", "token_usage", "timestamp"
    }
    for key, value in optimized_context.items():
        if key not in compatible and key not in ignored_keys:
            compatible[key] = value

    return compatible


async def fallback_get_context(
    user_id: int,
    conversation_id: int,
    player_name: str = "Chase"
) -> Dict[str, Any]:
    """
    Fallback context retrieval directly from the database.
    """
    try:
        minimal_context = {
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
            # 1. Player stats
            row = await conn.fetchrow(
                """
                SELECT corruption, confidence, willpower,
                       obedience, dependency, lust,
                       mental_resilience, physical_endurance
                FROM PlayerStats
                WHERE user_id=$1 AND conversation_id=$2
                LIMIT 1
                """,
                user_id, conversation_id
            )
            if row:
                minimal_context["player_stats"] = dict(row)

            # 2. Introduced NPCs
            rows = await conn.fetch(
                """
                SELECT npc_id, npc_name, dominance, cruelty, closeness,
                       trust, respect, intensity, current_location,
                       physical_description
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND introduced=TRUE
                LIMIT 20
                """,
                user_id, conversation_id
            )
            for r in rows:
                minimal_context["introduced_npcs"].append(dict(r))

            # 3. Time info
            time_keys = [
                ("CurrentYear", "year"),
                ("CurrentMonth", "month"),
                ("CurrentDay", "day"),
                ("TimeOfDay", "time_of_day")
            ]
            for key, context_key in time_keys:
                row = await conn.fetchrow(
                    """
                    SELECT value
                    FROM CurrentRoleplay
                    WHERE user_id=$1 AND conversation_id=$2 AND key=$3
                    """,
                    user_id, conversation_id, key
                )
                if row:
                    minimal_context[context_key] = row["value"]

            # 4. Current roleplay data
            rows = await conn.fetch(
                """
                SELECT key, value
                FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2
                """,
                user_id, conversation_id
            )
            for r in rows:
                minimal_context["current_roleplay"][r["key"]] = r["value"]

        return minimal_context

    except Exception as e:
        logger.error(f"Error in fallback_get_context: {e}", exc_info=True)
        return {
            "player_stats": {},
            "introduced_npcs": [],
            "unintroduced_npcs": [],
            "current_roleplay": {},
            "error": str(e)
        }


###############################################################################
# Optimized Context Cache
###############################################################################

class OptimizedContextCache:
    """
    Multi-level context cache wrapper that integrates with the unified cache.
    """

    def __init__(self, config):
        # All the real config usage is here, no `await`.
        self.l1_ttl = config.get("cache", "l1_ttl_seconds", 60)
        self.l2_ttl = config.get("cache", "l2_ttl_seconds", 300)
        self.l3_ttl = config.get("cache", "l3_ttl_seconds", 3600)
        self.enabled = config.get("cache", "enabled", True)

    @classmethod
    async def create(cls) -> "OptimizedContextCache":
        """
        Async factory that awaits get_config() so we don't do it in __init__.
        """
        cfg = await get_config()
        return cls(cfg)

    async def get(self, key: str, fetch_func, cache_level: int = 1) -> Any:
        """
        Get an item from cache or fetch it if not found. Then store it.

        Args:
            key: Cache key
            fetch_func: Async function to fetch data if cache miss
            cache_level: Which level of the cache (1, 2, or 3)
        """
        if not self.enabled:
            return await fetch_func()

        from context.unified_cache import context_cache

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
        Invalidate cache entries that match a given prefix.
        """
        from context.unified_cache import context_cache
        context_cache.invalidate(key_prefix)


###############################################################################
# Optimized Incremental Context Manager
###############################################################################

class OptimizedIncrementalContextManager:
    """
    Incremental context manager that leverages the unified context service
    for partial updates.
    """

    def __init__(self, config):
        self.enabled = config.is_enabled("use_incremental_context")
        self.token_budget = config.get("token_budget", "default_budget", 4000)
        self.use_vector = config.is_enabled("use_vector_search")

    @classmethod
    async def create(cls) -> "OptimizedIncrementalContextManager":
        """
        Async factory method to create an instance once we can await get_config().
        """
        cfg = await get_config()
        return cls(cfg)

    async def get_context(
        self,
        user_id: int,
        conversation_id: int,
        user_input: str,
        location: Optional[str] = None,
        include_delta: bool = True
    ) -> Dict[str, Any]:
        """
        Get context with delta tracking if enabled.
        """
        if not self.enabled or not include_delta:
            return await self.get_full_context(user_id, conversation_id, user_input, location)

        context_service = await get_context_service(user_id, conversation_id)
        context_data = await context_service.get_context(
            input_text=user_input,
            location=location,
            context_budget=self.token_budget,
            use_vector_search=self.use_vector,
            use_delta=True
        )

        result = {
            "full_context": format_context_for_compatibility(context_data),
            "is_incremental": context_data.get("is_delta", False)
        }

        if context_data.get("is_delta", False) and "delta_changes" in context_data:
            result["delta_context"] = context_data["delta_changes"]

        return result

    async def get_full_context(
        self,
        user_id: int,
        conversation_id: int,
        user_input: str,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get the full context without delta tracking.
        """
        context_data = await get_aggregated_roleplay_context(
            user_id=user_id,
            conversation_id=conversation_id,
            current_input=user_input,
            location=location
        )
        return {
            "full_context": context_data,
            "is_incremental": False
        }


###############################################################################
# Additional Utilities
###############################################################################

async def get_optimized_context(
    user_id: int,
    conversation_id: int,
    current_input: str,
    location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Similar to get_aggregated_roleplay_context, but specifically for an "optimized" flow.
    """
    config = await get_config()
    context_budget = config.get("token_budget", "default_budget", 4000)

    context_service = await get_context_service(user_id, conversation_id)
    context_data = await context_service.get_context(
        input_text=current_input,
        location=location,
        context_budget=context_budget,
        use_delta=False
    )
    return format_context_for_compatibility(context_data)


def build_aggregator_text(aggregated_data: Dict[str, Any]) -> str:
    """
    Build aggregator text from the provided data for display or logging.
    """
    if "aggregator_text" in aggregated_data:
        return aggregated_data["aggregator_text"]

    current_location = aggregated_data.get("current_location", "Unknown")
    year = aggregated_data.get("year", "1040")
    month = aggregated_data.get("month", "6")
    day = aggregated_data.get("day", "15")
    time_of_day = aggregated_data.get("time_of_day", "Morning")

    introduced_npcs = aggregated_data.get("introduced_npcs", [])

    date_line = f"- It is {year}, {month} {day}, {time_of_day}.\n"
    location_line = f"- Current location: {current_location}\n"

    npc_lines = ["Introduced NPCs in the area:"]
    for npc in introduced_npcs[:5]:
        npc_loc = npc.get("current_location", "Unknown")
        npc_name = npc.get("npc_name", "Unnamed NPC")
        npc_lines.append(f"  - {npc_name} is at {npc_loc}")

    if introduced_npcs:
        npc_section = "\n".join(npc_lines)
    else:
        npc_section = "No NPCs currently in the area."

    text = f"{date_line}{location_line}\n{npc_section}\n"

    environment_desc = aggregated_data.get("current_roleplay", {}).get("EnvironmentDesc")
    if environment_desc:
        text += f"\nEnvironment:\n{environment_desc}\n"

    player_role = aggregated_data.get("current_roleplay", {}).get("PlayerRole")
    if player_role:
        text += f"\nPlayer Role:\n{player_role}\n"

    text += "\n\n<!-- Context optimized with unified context system -->"

    has_relevance = any("relevance_score" in npc or "relevance" in npc for npc in introduced_npcs)
    if has_relevance:
        text += "\n<!-- NPCs sorted by relevance to current context -->"

    return text


###############################################################################
# Maintenance and Migration
###############################################################################

async def run_context_maintenance(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Run maintenance tasks for context optimization.
    """
    context_service = await get_context_service(user_id, conversation_id)
    return await context_service.run_maintenance()


async def migrate_old_context_to_new(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Migrate data from old context system to the new optimized system.
    """
    try:
        from logic.aggregator_sdk import get_aggregated_roleplay_context as old_get_context
        old_context = await old_get_context(user_id, conversation_id)

        context_service = await get_context_service(user_id, conversation_id)
        from context.memory_manager import get_memory_manager
        memory_manager = await get_memory_manager(user_id, conversation_id)

        memory_migrations = 0
        if "memories" in old_context:
            for mem in old_context["memories"]:
                content = mem.get("content") or mem.get("text") or ""
                memory_type = mem.get("type", "observation")
                if content:
                    await memory_manager.add_memory(
                        content=content,
                        memory_type=memory_type,
                        importance=0.7
                    )
                    memory_migrations += 1

        npc_migrations = 0
        config = await get_config()
        if config.is_enabled("use_vector_search"):
            from context.vector_service import get_vector_service
            vector_service = await get_vector_service(user_id, conversation_id)
            if "introduced_npcs" in old_context:
                npc_migrations = len(old_context["introduced_npcs"])

        return {
            "memory_migrations": memory_migrations,
            "npc_migrations": npc_migrations,
            "success": True
        }

    except Exception as e:
        logger.error(f"Error during context migration: {e}", exc_info=True)
        return {
            "error": str(e),
            "success": False
        }


###############################################################################
# Optional Monkey Patching
###############################################################################

def apply_context_optimizations() -> bool:
    """
    Apply context optimizations by monkey patching existing aggregator functions.
    """
    import sys
    aggregator_sdk = sys.modules.get("logic.aggregator_sdk")
    if not aggregator_sdk:
        logger.warning("Could not find aggregator_sdk module for patching")
        return False

    original_get_context = getattr(aggregator_sdk, "get_aggregated_roleplay_context", None)
    original_build_text = getattr(aggregator_sdk, "build_aggregator_text", None)
    if not original_get_context or not original_build_text:
        logger.warning("Required aggregator functions not found in aggregator_sdk")
        return False

    setattr(aggregator_sdk, "get_aggregated_roleplay_context", get_aggregated_roleplay_context)
    setattr(aggregator_sdk, "build_aggregator_text", build_aggregator_text)
    setattr(aggregator_sdk, "ContextCache", OptimizedContextCache)
    setattr(aggregator_sdk, "IncrementalContextManager", OptimizedIncrementalContextManager)
    setattr(aggregator_sdk, "get_optimized_context", get_optimized_context)
    setattr(aggregator_sdk, "run_context_maintenance", run_context_maintenance)

    logger.info("Applied context optimizations via monkey patching.")
    return True


async def update_context_with_universal_updates(
    context: dict,
    universal_updates: dict,
    user_id: str,
    conversation_id: str
) -> dict:
    """
    Update the context with universal updates while maintaining consistency.
    """
    try:
        updated_context = context.copy()

        # NPC Updates
        if "npc_updates" in universal_updates:
            for npc_update in universal_updates["npc_updates"]:
                npc_id = npc_update.get("npc_id")
                if not npc_id:
                    continue
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

        # Relationship updates
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

        # Quest updates
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

        # Inventory updates
        if "inventory_updates" in universal_updates:
            if "inventory" not in updated_context:
                updated_context["inventory"] = {"items": {}, "removed_items": []}

            for item in universal_updates["inventory_updates"].get("added_items", []):
                if isinstance(item, str):
                    item_name = item
                    item_data = {"name": item_name}
                else:
                    item_name = item.get("name")
                    item_data = item

                if item_name:
                    updated_context["inventory"]["items"][item_name] = item_data

            for item in universal_updates["inventory_updates"].get("removed_items", []):
                if isinstance(item, str):
                    item_name = item
                else:
                    item_name = item.get("name")
                if item_name:
                    if item_name in updated_context["inventory"]["items"]:
                        del updated_context["inventory"]["items"][item_name]
                    updated_context["inventory"]["removed_items"].append(item_name)

        # Activity updates
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

        updated_context["last_modified"] = datetime.now().isoformat()
        return updated_context

    except Exception as e:
        logger.error(f"Error updating context with universal updates: {e}", exc_info=True)
        return context  # Return original context on error
