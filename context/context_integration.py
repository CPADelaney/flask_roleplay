# context/context_integration.py

"""
Integration module to connect the enhanced context system with existing code.

This module provides simple hooks to replace or enhance existing context 
retrieval functions with the new optimized context system.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

from context_optimization import (
    get_comprehensive_context,
    consolidate_memories,
    get_context_service,
    EnhancedContextCache,
    EnhancedIncrementalContextManager
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------
# Integration with Existing Context Systems
# -------------------------------------------------------------------------------

async def get_enhanced_context(user_id, conversation_id, user_input, location=None, token_budget=4000):
    """
    Enhanced drop-in replacement for get_aggregated_roleplay_context
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        user_input: Current user input text
        location: Optional current location
        token_budget: Maximum token budget
        
    Returns:
        Optimized context dictionary
    """
    from logic.aggregator_sdk import get_aggregated_roleplay_context
    
    try:
        # Use the new comprehensive context service
        context = await get_comprehensive_context(
            user_id=user_id,
            conversation_id=conversation_id,
            input_text=user_input,
            location=location,
            context_budget=token_budget
        )
        
        # Convert the optimized context structure to match the expected format
        # from get_aggregated_roleplay_context
        return format_context_for_compatibility(context)
    except Exception as e:
        logger.error(f"Error getting enhanced context: {e}, falling back to standard context")
        # Fallback to original function if there's an error
        return await get_aggregated_roleplay_context(user_id, conversation_id)

def format_context_for_compatibility(optimized_context):
    """
    Format the optimized context to match the format expected by existing code.
    
    Args:
        optimized_context: Context from the comprehensive context service
        
    Returns:
        Formatted context compatible with existing code
    """
    # Create a compatible structure
    compatible = {}
    
    # Copy core fields directly
    for key in ["player_stats", "year", "month", "day", "time_of_day"]:
        if key in optimized_context:
            compatible[key] = optimized_context[key]
    
    # Map NPC lists
    if "npcs" in optimized_context:
        compatible["introduced_npcs"] = optimized_context["npcs"]
    
    # Map other key fields
    if "active_conflicts" in optimized_context:
        compatible["active_conflicts"] = optimized_context["active_conflicts"]
    
    if "memories" in optimized_context:
        compatible["memories"] = optimized_context["memories"]
    
    if "quests" in optimized_context:
        compatible["quests"] = optimized_context["quests"]
    
    # Copy any other fields as is
    for key, value in optimized_context.items():
        if key not in compatible and key not in ["npcs", "source", "token_usage", "delta_changes", "is_delta"]:
            compatible[key] = value
    
    return compatible

# -------------------------------------------------------------------------------
# Enhanced Aggregator Text Builder
# -------------------------------------------------------------------------------

def build_enhanced_aggregator_text(aggregated_data):
    """
    Enhanced version of build_aggregator_text with better formatting and optimization.
    
    Args:
        aggregated_data: Aggregated context data
        
    Returns:
        Formatted aggregator text
    """
    from routes.story_routes import build_aggregator_text
    
    # First get the basic aggregator text
    base_text = build_aggregator_text(aggregated_data)
    
    # Add additional optimization marker for parsers
    text_with_markers = base_text + "\n\n<!-- Context optimized with advanced retrieval system -->"
    
    # Add relevance scores if available
    if "introduced_npcs" in aggregated_data:
        has_relevance = any("relevance_score" in npc for npc in aggregated_data["introduced_npcs"])
        
        if has_relevance:
            # Sort NPCs by relevance for better presentation
            sorted_npcs = sorted(
                aggregated_data["introduced_npcs"], 
                key=lambda x: x.get("relevance_score", 0),
                reverse=True
            )
            
            # Add relevance notes
            text_with_markers += "\n\n<!-- NPCs sorted by relevance to current context -->"
    
    return text_with_markers

# -------------------------------------------------------------------------------
# Maintenance Functions
# -------------------------------------------------------------------------------

async def run_context_maintenance(user_id, conversation_id):
    """
    Run maintenance tasks for context optimization.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Maintenance results
    """
    results = {
        "memory_consolidation": None,
        "cache_cleanup": False,
        "preloading": None
    }
    
    try:
        # 1. Consolidate old memories
        try:
            memory_results = await consolidate_memories(user_id, conversation_id)
            results["memory_consolidation"] = memory_results
        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")
            results["memory_consolidation"] = {"error": str(e)}
        
        # 2. Clean up cache as needed
        try:
            service = get_context_service()
            cache = service.context_cache
            
            # Count items in each cache level
            l1_count = len(cache.l1_cache)
            l2_count = len(cache.l2_cache)
            l3_count = len(cache.l3_cache)
            
            # Force eviction if caches are too large
            if l1_count > 200:
                cache._evict_from_cache(cache.l1_cache, cache.l1_timestamps, cache.l1_access_count, 0.5)
            if l2_count > 500:
                cache._evict_from_cache(cache.l2_cache, cache.l2_timestamps, cache.l2_access_count, 0.3)
            if l3_count > 1000:
                cache._evict_from_cache(cache.l3_cache, cache.l3_timestamps, cache.l3_access_count, 0.2)
            
            results["cache_cleanup"] = True
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            results["cache_cleanup"] = False
        
        # 3. Run predictive preloading for common locations
        try:
            # Get current location
            from logic.aggregator_sdk import get_aggregated_roleplay_context
            context = await get_aggregated_roleplay_context(user_id, conversation_id)
            current_location = context.get("current_location", 
                                          context.get("currentRoleplay", {}).get("CurrentLocation"))
            
            if current_location:
                # Run preloading as background task
                preload_task = asyncio.create_task(
                    service.predictive_loader.preload_likely_contexts(
                        user_id, conversation_id, current_location
                    )
                )
                
                # Wait for it to complete with timeout
                try:
                    preload_results = await asyncio.wait_for(preload_task, timeout=10.0)
                    results["preloading"] = preload_results
                except asyncio.TimeoutError:
                    results["preloading"] = {"timeout": True}
        except Exception as e:
            logger.error(f"Error in predictive preloading: {e}")
            results["preloading"] = {"error": str(e)}
    
    except Exception as e:
        logger.error(f"Error in context maintenance: {e}")
        results["error"] = str(e)
    
    return results

# -------------------------------------------------------------------------------
# Monkey Patching for Easy Integration
# -------------------------------------------------------------------------------

def apply_context_optimizations():
    """
    Apply context optimizations by monkey patching existing functions.
    
    This replaces the existing context retrieval functions with enhanced versions.
    Only use this if you can't directly modify the code that calls these functions.
    """
    import logic.aggregator_sdk
    import routes.story_routes
    
    # Replace get_aggregated_roleplay_context with our enhanced version
    original_get_context = logic.aggregator_sdk.get_aggregated_roleplay_context
    
    async def enhanced_get_context_wrapper(user_id, conversation_id, player_name="Chase"):
        """Enhanced wrapper for get_aggregated_roleplay_context"""
        # Figure out if we have user input from somewhere
        user_input = ""
        
        # Try to get from global frame or last messages
        import inspect
        frame = inspect.currentframe()
        try:
            while frame:
                if 'user_input' in frame.f_locals:
                    user_input = frame.f_locals['user_input']
                    break
                if 'input_text' in frame.f_locals:
                    user_input = frame.f_locals['input_text']
                    break
                frame = frame.f_back
        finally:
            del frame
        
        # If we found user input, use enhanced version
        if user_input:
            return await get_enhanced_context(user_id, conversation_id, user_input)
        
        # Otherwise fall back to original
        return await original_get_context(user_id, conversation_id, player_name)
    
    # Replace the function
    logic.aggregator_sdk.get_aggregated_roleplay_context = enhanced_get_context_wrapper
    
    # Replace build_aggregator_text with our enhanced version
    original_build_text = routes.story_routes.build_aggregator_text
    
    def enhanced_build_text_wrapper(aggregated_data):
        """Enhanced wrapper for build_aggregator_text"""
        return build_enhanced_aggregator_text(aggregated_data)
    
    # Replace the function
    routes.story_routes.build_aggregator_text = enhanced_build_text_wrapper
    
    logger.info("Context optimization monkey patching applied successfully")
