# context/context_service.py

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import hashlib

from context.context_config import get_config
from context.unified_cache import context_cache
from context.vector_service import get_vector_service
from context.memory_manager import get_memory_manager
from context.context_manager import get_context_manager
from context.context_performance import PerformanceMonitor, track_performance

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
        
        # NEW: Progressive summarization integration
        self.narrative_manager = None
    
    async def initialize(self):
        """Initialize the context service"""
        if self.initialized:
            return
        
        # Get core components (lazy initialization)
        self.context_manager = get_context_manager()
        self.performance_monitor = PerformanceMonitor.get_instance(self.user_id, self.conversation_id)
        
        # NEW: Initialize memory manager
        self.memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
        
        # NEW: Initialize vector service if enabled
        if self.config.is_enabled("use_vector_search"):
            self.vector_service = await get_vector_service(self.user_id, self.conversation_id)
        
        # NEW: Initialize narrative manager for progressive summarization
        try:
            from story_agent.progressive_summarization import RPGNarrativeManager
            from db.connection import get_db_connection
            
            self.narrative_manager = RPGNarrativeManager(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                db_connection_string=get_db_connection()
            )
            await self.narrative_manager.initialize()
        except ImportError:
            logger.info("Progressive summarization not available - narrative manager not initialized")
            self.narrative_manager = None
        
        self.initialized = True
        logger.info(f"Initialized context service for user {self.user_id}, conversation {self.conversation_id}")
    
    async def close(self):
        """Close the context service"""
        self.initialized = False
        
        # NEW: Close narrative manager if initialized
        if self.narrative_manager:
            await self.narrative_manager.close()
        
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
        include_quests: bool = True,
        source_version: Optional[int] = None  # NEW: Version tracking for delta updates
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
            source_version: Optional source version for delta tracking
            
        Returns:
            Optimized context
        """
        # Initialize if needed
        await self.initialize()
        
        # Check if vector search should be used
        if use_vector_search is None:
            use_vector_search = self.config.is_enabled("use_vector_search")
        
        # NEW: Dynamic token budget based on narrative stage
        if not context_budget:
            # Try to get the narrative stage to adjust budget
            narrative_stage = await self._get_narrative_stage()
            if narrative_stage:
                # Adjust budget based on stage
                stage_name = narrative_stage.get("name", "").lower()
                if "revelation" in stage_name:
                    # Later stages need more detail
                    context_budget = self.config.get_token_budget("default") * 1.2
                elif "beginning" in stage_name:
                    # Early stages need less detail
                    context_budget = self.config.get_token_budget("default") * 0.8
                else:
                    context_budget = self.config.get_token_budget("default")
            else:
                context_budget = self.config.get_token_budget("default")
        
        # Check for delta tracking if source_version is provided
        if source_version is not None and use_delta:
            # Get context from context manager with delta tracking
            context_result = await self.context_manager.get_context(source_version)
            
            # If it's incremental, return as is
            if context_result.get("is_incremental", False):
                return context_result
            else:
                # If not incremental, continue with normal context retrieval
                pass
        
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
                    tasks.append(memory_manager.search_memories(query_text=input_text, limit=5, use_vector=True))
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
            
            # NEW: 6. Get progressive summaries if available
            if self.narrative_manager and input_text:
                tasks.append(self._get_summarized_narratives(input_text))
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
            
            # NEW: Add summarized narratives if available
            if self.narrative_manager and input_text:
                context["narrative_summaries"] = results[5]
            
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
            
            # Store in context manager for delta tracking
            await self.context_manager.update_context(context)
            
            # Add version for delta tracking
            context["version"] = self.context_manager.version
            
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

    # NEW: Get narrative stage for context budget adjustment
    async def _get_narrative_stage(self) -> Dict[str, Any]:
        """Get the current narrative stage"""
        try:
            from logic.narrative_progression import get_current_narrative_stage
            
            stage = await get_current_narrative_stage(self.user_id, self.conversation_id)
            if stage:
                return {"name": stage.name, "description": stage.description}
            return None
        except:
            return None
    
    # NEW: Get summarized narratives from narrative manager
    async def _get_summarized_narratives(self, query_text: str) -> Dict[str, Any]:
        """Get summarized narratives relevant to the query"""
        if not self.narrative_manager:
            return {}
        
        try:
            # Get optimal narrative context with query
            return await self.narrative_manager.get_optimal_narrative_context(
                query=query_text,
                max_tokens=1000
            )
        except Exception as e:
            logger.error(f"Error getting summarized narratives: {e}")
            return {}
    
    # NEW: Method to get context with appropriate summarization
    async def get_summarized_context(
        self,
        input_text: str = "",
        summary_level: int = 1,  # 0=Detailed, 1=Condensed, 2=Summary, 3=Headline
        context_budget: int = 2000,
        use_vector_search: bool = True,
    ) -> Dict[str, Any]:
        """
        Get context with automatic summarization based on importance and recency
        
        Args:
            input_text: Current user input
            summary_level: Level of summarization to apply
            context_budget: Maximum token budget
            use_vector_search: Whether to use vector search
            
        Returns:
            Summarized context
        """
        # Initialize if needed
        await self.initialize()
        
        # Get base context
        context = await self.get_context(
            input_text=input_text,
            context_budget=context_budget,
            use_vector_search=use_vector_search,
            use_delta=False
        )
        
        # If narrative manager is available, use it for summarization
        if self.narrative_manager:
            try:
                # Get summarized memories
                narrative_context = await self.narrative_manager.get_optimal_narrative_context(
                    query=input_text,
                    max_tokens=context_budget // 4  # Use 25% of budget for narrative context
                )
                
                # Add summarized narratives to context
                context["narrative_summaries"] = narrative_context
                
            except Exception as e:
                logger.error(f"Error getting narrative summaries: {e}")
        
        # Apply summarization to existing memories based on importance/recency
        if "memories" in context:
            for i, memory in enumerate(context["memories"]):
                # Skip recent or important memories
                importance = memory.get("importance", 0.5)
                
                # Parse timestamp to check recency
                is_recent = False
                if "created_at" in memory:
                    created_at = memory["created_at"]
                    if isinstance(created_at, str):
                        try:
                            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                            days_old = (datetime.now() - created_at).days
                            is_recent = days_old < 7  # Less than a week old
                        except:
                            pass
                
                # Skip summarization for recent or important memories
                if importance > 0.7 or is_recent:
                    continue
                
                # Get the content to summarize
                content = memory.get("content", "")
                if not content:
                    continue
                
                # Apply appropriate summarization level
                summarized = await self._summarize_text(content, summary_level)
                
                # Replace content with summarized version
                context["memories"][i]["content"] = summarized
                context["memories"][i]["summarized"] = True
                context["memories"][i]["summary_level"] = summary_level
        
        return context
    
    # Helper method for summarization
    async def _summarize_text(self, text: str, level: int) -> str:
        """
        Summarize text to the specified level
        
        Args:
            text: Text to summarize
            level: Summarization level (0-3)
            
        Returns:
            Summarized text
        """
        if level == 0:  # Detailed - no summarization
            return text
            
        # Simple rule-based summarization if narrative manager not available
        if not self.narrative_manager:
            sentences = text.split(". ")
            
            if level == 1:  # Condensed
                # Keep first, middle and last sentence
                if len(sentences) >= 3:
                    middle_idx = len(sentences) // 2
                    return f"{sentences[0]}. {sentences[middle_idx]}. {sentences[-1]}."
                return text
                
            elif level == 2:  # Summary
                # Keep just first and last sentences
                if len(sentences) >= 2:
                    return f"{sentences[0]}. {sentences[-1]}."
                return text
                
            elif level == 3:  # Headline
                # Just keep first sentence, truncated if needed
                if sentences:
                    if len(sentences[0]) > 100:
                        return sentences[0][:97] + "..."
                    return sentences[0]
                return text
        
        # Use narrative manager's summarizer if available
        try:
            from story_agent.progressive_summarization import SummaryLevel
            
            # Map our levels to SummaryLevel
            summary_level_map = {
                0: SummaryLevel.DETAILED,
                1: SummaryLevel.CONDENSED,
                2: SummaryLevel.SUMMARY,
                3: SummaryLevel.HEADLINE
            }
            
            summarizer = self.narrative_manager.narrative.summarizer
            result = await summarizer.summarize(text, summary_level_map[level])
            return result
        except Exception as e:
            logger.error(f"Error summarizing with narrative manager: {e}")
            
            # Fallback to simple summarization
            sentences = text.split(". ")
            if level == 1:  # Condensed
                return ". ".join(sentences[:max(1, len(sentences) // 2)])
            elif level == 2:  # Summary
                return ". ".join(sentences[:max(1, len(sentences) // 3)])
            elif level == 3:  # Headline
                return sentences[0] if sentences else text
            
            return text
    
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
                    
                    # NEW: Get narrative stage
                    from logic.narrative_progression import get_current_narrative_stage
                    narrative_stage = await get_current_narrative_stage(self.user_id, self.conversation_id)
                    narrative_stage_info = None
                    if narrative_stage:
                        narrative_stage_info = {
                            "name": narrative_stage.name,
                            "description": narrative_stage.description
                        }
                    
                    # Create base context
                    context = {
                        "time_info": time_info,
                        "player_stats": player_stats,
                        "current_roleplay": roleplay_data,
                        "current_location": location or roleplay_data.get("CurrentLocation", "Unknown"),
                        "narrative_stage": narrative_stage_info
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
        # If there's an vector search and input text, prioritize vector search
        if vector_service and input_text and await vector_service.is_initialized():
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
        if vector_service and await vector_service.is_initialized():
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
                           if k not in ["timestamp", "token_usage", "is_delta", "delta_changes", "retrieval_time", "version"]}
        
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
            "narratives": ["narratives"],
            "summaries": ["narrative_summaries"]  # NEW: Track narrative summaries
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
        skip_keys.update(["token_usage", "is_delta", "delta_changes", "timestamp", "total_tokens", "version"])
        
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
            "summaries": 2,          # Can be trimmed first
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
                
                # Additionally, summarize what we keep if needed
                if self.narrative_manager and reduction_needed > 0:
                    for i, memory in enumerate(new_memories):
                        if "content" in memory and len(memory["content"]) > 200:
                            # Get a summarized version
                            content = memory["content"]
                            summarized = await self._summarize_text(content, 2)  # Level 2 summary
                            if len(summarized) < len(content):
                                new_memories[i]["content"] = summarized
                                new_memories[i]["summarized"] = True
                
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
            
            elif component_name == "summaries" and "narrative_summaries" in trimmed:
                # Remove narrative summaries if needed
                old_tokens = token_usage[component_name]
                del trimmed["narrative_summaries"]
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
            "performance_metrics": None,
            "narrative_maintenance": None  # NEW: Track narrative maintenance
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
        
        # NEW: 5. Narrative maintenance if available
        if self.narrative_manager:
            try:
                narrative_result = await self.narrative_manager.run_maintenance()
                results["narrative_maintenance"] = narrative_result
            except Exception as e:
                logger.error(f"Error running narrative maintenance: {e}")
                results["narrative_maintenance"] = {"error": str(e)}
        
        return results

    async def _get_direct_relationships(self, npc_id: int) -> Dict[str, Any]:
        """Get direct relationships for an NPC."""
        try:
            from db.connection import get_db_connection
            import asyncpg
            
            conn = await asyncpg.connect(dsn=get_db_connection())
            try:
                rows = await conn.fetch("""
                    SELECT r.*, n.name as npc_name, n.role as npc_role
                    FROM NPCRelationships r
                    JOIN NPCs n ON r.related_npc_id = n.id
                    WHERE r.npc_id = $1 AND r.user_id = $2 AND r.conversation_id = $3
                """, npc_id, self.user_id, self.conversation_id)
                
                return [dict(row) for row in rows]
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error getting direct relationships: {e}")
            return []

    async def _get_relationship_history(self, npc_id: int) -> List[Dict[str, Any]]:
        """Get relationship history for an NPC."""
        try:
            from db.connection import get_db_connection
            import asyncpg
            
            conn = await asyncpg.connect(dsn=get_db_connection())
            try:
                rows = await conn.fetch("""
                    SELECT rh.*, n.name as npc_name
                    FROM RelationshipHistory rh
                    JOIN NPCs n ON rh.related_npc_id = n.id
                    WHERE rh.npc_id = $1 AND rh.user_id = $2 AND rh.conversation_id = $3
                    ORDER BY rh.timestamp DESC
                    LIMIT 10
                """, npc_id, self.user_id, self.conversation_id)
                
                return [dict(row) for row in rows]
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error getting relationship history: {e}")
            return []

    async def _get_shared_memories(self, npc_id: int) -> List[Dict[str, Any]]:
        """Get shared memories with an NPC."""
        try:
            # Get memories from memory manager
            memories = await self.memory_manager.get_memories_by_npc(npc_id)
            
            # Filter and sort by importance
            shared_memories = sorted(
                [m for m in memories if m.metadata.get("npc_id") == npc_id],
                key=lambda x: x.importance,
                reverse=True
            )[:5]
            
            return [m.to_dict() for m in shared_memories]
        except Exception as e:
            logger.error(f"Error getting shared memories: {e}")
            return []

    async def _analyze_relationship_dynamics(self, npc_id: int) -> Dict[str, Any]:
        """Analyze relationship dynamics for an NPC."""
        try:
            # Get relationship history
            history = await self._get_relationship_history(npc_id)
            
            # Calculate relationship trends
            trends = {
                "overall_trend": 0,
                "recent_changes": [],
                "key_events": []
            }
            
            if history:
                # Calculate overall trend
                trend_values = [h["relationship_change"] for h in history]
                trends["overall_trend"] = sum(trend_values) / len(trend_values)
                
                # Get recent changes
                trends["recent_changes"] = history[:3]
                
                # Identify key events
                key_events = [h for h in history if abs(h["relationship_change"]) > 0.5]
                trends["key_events"] = key_events[:3]
            
            return trends
        except Exception as e:
            logger.error(f"Error analyzing relationship dynamics: {e}")
            return {}

    async def _get_relationship_context(self, npc_id: int) -> Dict[str, Any]:
        """Get comprehensive relationship context for an NPC."""
        try:
            # Get direct relationships
            relationships = await self._get_direct_relationships(npc_id)
            
            # Get relationship history
            history = await self._get_relationship_history(npc_id)
            
            # Get shared memories
            shared_memories = await self._get_shared_memories(npc_id)
            
            # Get relationship dynamics
            dynamics = await self._analyze_relationship_dynamics(npc_id)
            
            return {
                "relationships": relationships,
                "history": history,
                "shared_memories": shared_memories,
                "dynamics": dynamics
            }
        except Exception as e:
            logger.error(f"Error getting relationship context: {e}")
            return {}

    async def _get_narrative_state(self) -> Dict[str, Any]:
        """Get current narrative state."""
        try:
            from db.connection import get_db_connection
            import asyncpg
            
            conn = await asyncpg.connect(dsn=get_db_connection())
            try:
                # Get active narrative threads
                threads = await conn.fetch("""
                    SELECT * FROM NarrativeThreads
                    WHERE user_id = $1 AND conversation_id = $2
                    AND is_active = true
                """, self.user_id, self.conversation_id)
                
                # Get narrative metadata
                metadata = await conn.fetchrow("""
                    SELECT * FROM NarrativeMetadata
                    WHERE user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
                
                return {
                    "threads": [dict(t) for t in threads],
                    "metadata": dict(metadata) if metadata else {}
                }
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error getting narrative state: {e}")
            return {"threads": [], "metadata": {}}

    async def _check_narrative_inconsistencies(self, context: Dict[str, Any], narrative_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for narrative inconsistencies in the context."""
        inconsistencies = []
        
        try:
            # Check NPC consistency
            npc_ids = set(npc["id"] for npc in context.get("npcs", []))
            thread_npc_ids = set(npc_id for thread in narrative_state["threads"] 
                               for npc_id in thread.get("involved_npcs", []))
            
            # Find NPCs in threads but not in context
            missing_npcs = thread_npc_ids - npc_ids
            if missing_npcs:
                inconsistencies.append({
                    "type": "missing_npcs",
                    "details": list(missing_npcs),
                    "severity": "medium"
                })
            
            # Check location consistency
            current_location = context.get("location_details", {}).get("name")
            thread_locations = set(loc for thread in narrative_state["threads"] 
                                 for loc in thread.get("locations", []))
            
            if current_location and current_location not in thread_locations:
                inconsistencies.append({
                    "type": "location_mismatch",
                    "details": {"current": current_location, "expected": list(thread_locations)},
                    "severity": "low"
                })
            
            # Check quest consistency
            active_quests = set(q["id"] for q in context.get("quests", []))
            thread_quests = set(q_id for thread in narrative_state["threads"] 
                              for q_id in thread.get("related_quests", []))
            
            # Find quests in threads but not in context
            missing_quests = thread_quests - active_quests
            if missing_quests:
                inconsistencies.append({
                    "type": "missing_quests",
                    "details": list(missing_quests),
                    "severity": "high"
                })
            
            return inconsistencies
        except Exception as e:
            logger.error(f"Error checking narrative inconsistencies: {e}")
            return []

    async def _resolve_narrative_inconsistencies(self, context: Dict[str, Any], inconsistencies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve narrative inconsistencies in the context."""
        try:
            resolved_context = context.copy()
            
            for inconsistency in inconsistencies:
                if inconsistency["type"] == "missing_npcs":
                    # Fetch missing NPCs
                    missing_npcs = await self._fetch_npcs(inconsistency["details"])
                    resolved_context["npcs"].extend(missing_npcs)
                
                elif inconsistency["type"] == "location_mismatch":
                    # Update location context
                    location_context = await self._get_location_context(inconsistency["details"]["expected"][0])
                    resolved_context["location_details"] = location_context
                
                elif inconsistency["type"] == "missing_quests":
                    # Fetch missing quests
                    missing_quests = await self._fetch_quests(inconsistency["details"])
                    resolved_context["quests"].extend(missing_quests)
            
            return resolved_context
        except Exception as e:
            logger.error(f"Error resolving narrative inconsistencies: {e}")
            return context

    async def _calculate_coherence_score(self, context: Dict[str, Any]) -> float:
        """Calculate narrative coherence score."""
        try:
            score = 1.0
            
            # Check NPC presence
            npc_count = len(context.get("npcs", []))
            if npc_count < 3:
                score *= 0.9
            
            # Check quest consistency
            quest_count = len(context.get("quests", []))
            if quest_count < 2:
                score *= 0.9
            
            # Check memory relevance
            memories = context.get("memories", [])
            if not memories:
                score *= 0.8
            else:
                # Check memory recency
                recent_memories = [m for m in memories 
                                 if (datetime.now() - m["created_at"]).days < 7]
                if len(recent_memories) < 3:
                    score *= 0.9
            
            # Check relationship consistency
            relationships = context.get("relationships", {})
            if not relationships:
                score *= 0.9
            else:
                # Check relationship depth
                deep_relationships = [r for r in relationships.values() 
                                   if r.get("depth", 0) > 0.5]
                if len(deep_relationships) < 2:
                    score *= 0.9
            
            return round(score, 2)
        except Exception as e:
            logger.error(f"Error calculating coherence score: {e}")
            return 0.5

    async def _get_active_narrative_threads(self) -> List[Dict[str, Any]]:
        """Get active narrative threads."""
        try:
            from db.connection import get_db_connection
            import asyncpg
            
            conn = await asyncpg.connect(dsn=get_db_connection())
            try:
                rows = await conn.fetch("""
                    SELECT * FROM NarrativeThreads
                    WHERE user_id = $1 AND conversation_id = $2
                    AND is_active = true
                    ORDER BY importance DESC, last_updated DESC
                    LIMIT 5
                """, self.user_id, self.conversation_id)
                
                return [dict(row) for row in rows]
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error getting active narrative threads: {e}")
            return []

    async def _ensure_narrative_coherence(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure narrative coherence across context."""
        try:
            # Get current narrative state
            narrative_state = await self._get_narrative_state()
            
            # Check for narrative inconsistencies
            inconsistencies = await self._check_narrative_inconsistencies(context, narrative_state)
            
            # Resolve inconsistencies
            if inconsistencies:
                context = await self._resolve_narrative_inconsistencies(context, inconsistencies)
            
            # Add narrative metadata
            context["narrative_metadata"] = {
                "state": narrative_state,
                "coherence_score": await self._calculate_coherence_score(context),
                "narrative_threads": await self._get_active_narrative_threads()
            }
            
            return context
        except Exception as e:
            logger.error(f"Error ensuring narrative coherence: {e}")
            return context

    async def initialize(self, user_id: int, conversation_id: int):
        """Initialize the context service."""
        try:
            # Initialize core components
            self.context_manager = ContextManager()
            self.memory_manager = MemoryManager()
            self.vector_service = VectorService()
            
            # Initialize Nyx integration
            await self.context_manager.initialize_nyx_integration(user_id, conversation_id)
            
            # Load initial context
            await self._load_initial_context(user_id, conversation_id)
            
            logger.info("Context service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing context service: {e}")
            raise

    async def get_context(self, user_id: int, conversation_id: int, 
                         context_type: str = "full") -> Dict[str, Any]:
        """Get optimized context for interactions."""
        try:
            # Get base context
            context = await self._get_base_context(user_id, conversation_id)
            
            # Apply Nyx directives
            context = await self._apply_nyx_directives(context)
            
            # Get narrative state
            narrative_state = await self._get_narrative_state(user_id, conversation_id)
            
            # Get active conflicts
            conflicts = await self._get_active_conflicts(user_id, conversation_id)
            
            # Get relevant NPCs
            npcs = await self._get_relevant_npcs(user_id, conversation_id)
            
            # Get story progression
            story_progress = await self._get_story_progression(user_id, conversation_id)
            
            # Combine all context
            full_context = {
                "base_context": context,
                "narrative_state": narrative_state,
                "active_conflicts": conflicts,
                "relevant_npcs": npcs,
                "story_progress": story_progress,
                "timestamp": datetime.now().isoformat()
            }
            
            # Apply context type filtering
            if context_type == "condensed":
                return await self._condense_context(full_context)
            elif context_type == "summary":
                return await self._summarize_context(full_context)
            else:
                return full_context
                
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return {}

    async def _apply_nyx_directives(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Nyx directives to context."""
        try:
            # Get active directives
            directives = await self.context_manager.get_active_directives()
            
            # Apply each directive
            for directive in directives:
                if directive["type"] == "override":
                    context = await self._apply_override_directive(context, directive)
                elif directive["type"] == "prohibition":
                    context = await self._apply_prohibition_directive(context, directive)
                elif directive["type"] == "action":
                    context = await self._apply_action_directive(context, directive)
            
            return context
        except Exception as e:
            logger.error(f"Error applying Nyx directives: {e}")
            return context

    async def _apply_override_directive(self, context: Dict[str, Any], 
                                     directive: Dict[str, Any]) -> Dict[str, Any]:
        """Apply an override directive to context."""
        try:
            override_action = directive.get("override_action", {})
            applies_to = directive.get("applies_to", [])
            
            # Apply override to specified paths
            for path in applies_to:
                if path in context:
                    context[path] = override_action.get(path, context[path])
            
            return context
        except Exception as e:
            logger.error(f"Error applying override directive: {e}")
            return context

    async def _apply_prohibition_directive(self, context: Dict[str, Any], 
                                        directive: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a prohibition directive to context."""
        try:
            prohibited_actions = directive.get("prohibited_actions", [])
            
            # Remove prohibited actions from context
            for action in prohibited_actions:
                if action in context:
                    del context[action]
            
            return context
        except Exception as e:
            logger.error(f"Error applying prohibition directive: {e}")
            return context

    async def _apply_action_directive(self, context: Dict[str, Any], 
                                   directive: Dict[str, Any]) -> Dict[str, Any]:
        """Apply an action directive to context."""
        try:
            instruction = directive.get("instruction", "")
            params = directive.get("parameters", {})
            
            if "prioritize" in instruction.lower():
                # Apply prioritization rules
                priority_rules = params.get("priority_rules", {})
                context = await self._prioritize_context(context, priority_rules)
            
            elif "consolidate" in instruction.lower():
                # Apply consolidation rules
                consolidation_rules = params.get("consolidation_rules", {})
                context = await self._consolidate_context(context, consolidation_rules)
            
            elif "filter" in instruction.lower():
                # Apply filtering rules
                filter_rules = params.get("filter_rules", {})
                context = await self._filter_context(context, filter_rules)
            
            return context
        except Exception as e:
            logger.error(f"Error applying action directive: {e}")
            return context

    async def _prioritize_context(self, context: Dict[str, Any], 
                                rules: Dict[str, Any]) -> Dict[str, Any]:
        """Prioritize context based on rules."""
        try:
            prioritized = {}
            
            # Calculate priority scores
            for key, value in context.items():
                score = 0.0
                
                # Apply type-based scoring
                if key in rules.get("type_scores", {}):
                    score += rules["type_scores"][key]
                
                # Apply relationship-based scoring
                if key in rules.get("relationship_weights", {}):
                    score += rules["relationship_weights"][key]
                
                # Apply recency-based scoring
                if isinstance(value, dict) and "timestamp" in value:
                    age = (datetime.now() - datetime.fromisoformat(value["timestamp"])).total_seconds()
                    score += max(0, 1 - (age / rules.get("recency_threshold", 86400)))
                
                prioritized[key] = {"value": value, "score": score}
            
            # Sort by priority score
            sorted_items = sorted(
                prioritized.items(),
                key=lambda x: x[1]["score"],
                reverse=True
            )
            
            # Return prioritized context
            return {k: v["value"] for k, v in sorted_items}
        except Exception as e:
            logger.error(f"Error prioritizing context: {e}")
            return context

    async def _consolidate_context(self, context: Dict[str, Any], 
                                 rules: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate context based on rules."""
        try:
            consolidated = context.copy()
            
            # Apply grouping rules
            if "group_by" in rules:
                for group_key in rules["group_by"]:
                    if group_key in consolidated:
                        consolidated[group_key] = await self._consolidate_group(
                            consolidated[group_key],
                            rules.get("group_rules", {}).get(group_key, {})
                        )
            
            # Apply memory consolidation
            if "consolidate_memories" in rules and "memories" in consolidated:
                consolidated["memories"] = await self._consolidate_memories(
                    consolidated["memories"],
                    rules["consolidate_memories"]
                )
            
            # Apply relationship consolidation
            if "consolidate_relationships" in rules and "relationships" in consolidated:
                consolidated["relationships"] = await self._consolidate_relationships(
                    consolidated["relationships"],
                    rules["consolidate_relationships"]
                )
            
            return consolidated
        except Exception as e:
            logger.error(f"Error consolidating context: {e}")
            return context

    async def _filter_context(self, context: Dict[str, Any], 
                            rules: Dict[str, Any]) -> Dict[str, Any]:
        """Filter context based on rules."""
        try:
            filtered = context.copy()
            
            # Apply inclusion rules
            if "include_only" in rules:
                filtered = {
                    k: v for k, v in filtered.items()
                    if k in rules["include_only"]
                }
            
            # Apply exclusion rules
            if "exclude" in rules:
                for key in rules["exclude"]:
                    filtered.pop(key, None)
            
            # Apply importance threshold
            if "importance_threshold" in rules:
                filtered = {
                    k: v for k, v in filtered.items()
                    if self._calculate_importance(v) >= rules["importance_threshold"]
                }
            
            return filtered
        except Exception as e:
            logger.error(f"Error filtering context: {e}")
            return context

    def _calculate_importance(self, value: Any) -> float:
        """Calculate importance score for a context value."""
        try:
            importance = 0.0
            
            # Base importance by type
            if isinstance(value, dict):
                importance += 1.0
                # Add importance based on key presence
                if "error" in value or "critical" in value:
                    importance += 2.0
                if "player" in value or "npc" in value:
                    importance += 1.5
            elif isinstance(value, list):
                importance += 0.5
                # Add importance based on list length
                importance += min(1.0, len(value) * 0.1)
            
            return importance
        except Exception as e:
            logger.error(f"Error calculating importance: {e}")
            return 0.0


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
    use_delta: bool = True,
    source_version: Optional[int] = None,  # NEW: Version tracking for delta updates
    summary_level: Optional[int] = None  # NEW: Summary level option
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
        source_version: Optional source version for delta tracking 
        summary_level: Optional summary level (0-3)
        
    Returns:
        Optimized context dictionary
    """
    # Get context service
    service = await get_context_service(user_id, conversation_id)
    
    # If summarization requested, use that method
    if summary_level is not None:
        context = await service.get_summarized_context(
            input_text=input_text,
            summary_level=summary_level,
            context_budget=context_budget,
            use_vector_search=use_vector_search if use_vector_search is not None else True
        )
    else:
        # Get regular context
        context = await service.get_context(
            input_text=input_text,
            location=location,
            context_budget=context_budget,
            use_vector_search=use_vector_search,
            use_delta=use_delta,
            source_version=source_version
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
