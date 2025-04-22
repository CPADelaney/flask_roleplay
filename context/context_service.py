# context/context_service.py

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import hashlib

# Agent SDK imports
from agents import (
    Agent, Runner, function_tool, RunContextWrapper, trace, RunConfig,
    ModelSettings, InputGuardrail, OutputGuardrail, GuardrailFunctionOutput,
    handoff
)
from pydantic import BaseModel, Field

from context.context_config import get_config
from context.unified_cache import context_cache
from context.vector_service import get_vector_service
from context.memory_manager import get_memory_manager, get_memory_agent
from context.context_manager import get_context_manager, get_context_manager_agent
from context.models import (
    ContextRequest, ContextOutput, TokenUsage, 
    NPCData, Memory, QuestData, LocationData
)

logger = logging.getLogger(__name__)


class ContextGuardrailResult(BaseModel):
    """Result of context guardrail check"""
    valid: bool = True
    reason: Optional[str] = None
    budget_exceeded: bool = False
    actual_tokens: Optional[int] = None
    max_tokens: Optional[int] = None


class ContextService:
    """
    Unified context service that integrates all context components.
    Refactored to use OpenAI Agents SDK but without using @function_tool on instance methods.
    """

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.config = None  # Will be initialized asynchronously
        self.initialized = False
        self.performance_monitor = None
        self.context_manager = None
        self.memory_manager = None
        self.vector_service = None
        self.last_context = None
        self.last_context_hash = None
        self.narrative_manager = None
        
        # Orchestrator agent
        self.orchestrator_agent = None
        
        # Specialized agents
        self.context_agent = None
        self.memory_agent = None
        self.vector_agent = None
        self.narrative_agent = None
        
        # Shared context for agents
        self.agent_context = {"user_id": user_id, "conversation_id": conversation_id}
    
    async def initialize(self):
        """Initialize the context service"""
        if self.initialized:
            return
        
        # Get configuration
        self.config = await get_config()
        
        # Get or create the core components
        self.context_manager = get_context_manager()
        self.memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
        
        # Initialize vector service if enabled
        if self.config.is_enabled("use_vector_search"):
            self.vector_service = await get_vector_service(self.user_id, self.conversation_id)
        
        # Initialize narrative manager if available
        try:
            from story_agent.progressive_summarization import RPGNarrativeManager
            from db.connection import get_db_connection_context
            self.narrative_manager = RPGNarrativeManager(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                db_connection_string=get_db_connection_context()
            )
            await self.narrative_manager.initialize()
        except ImportError:
            logger.info("Progressive summarization not available - narrative manager not initialized")
            self.narrative_manager = None
        
        # Initialize specialized agents
        await self._initialize_agents()
        
        self.initialized = True
        logger.info(f"Initialized context service for user {self.user_id}, conversation {self.conversation_id}")
    
    async def _initialize_agents(self):
        """Initialize the specialized agents for context, memory, etc."""
        self.context_agent = get_context_manager_agent()
        self.memory_agent = get_memory_agent()
        
        # Create orchestrator agent with handoffs
        self.orchestrator_agent = Agent(
            name="Context Orchestrator",
            instructions="""
            You are the orchestrator for context management in an RPG system.
            You decide which specialized agent should handle each request 
            based on the task:

            1. For memory-related tasks: Use the Memory Manager agent
            2. For context management tasks: Use the Context Manager agent
            3. For vector search tasks: Use the Vector Search agent
            4. For narrative tasks: Use the Narrative agent
            """,
            handoffs=[
                handoff(
                    self.context_agent,
                    tool_name_override="use_context_manager",
                    tool_description_override="Use this for context management tasks"
                ),
                handoff(
                    self.memory_agent,
                    tool_name_override="use_memory_manager",
                    tool_description_override="Use this for memory-related tasks"
                )
                # We could add vector/narrative agent handoffs if needed
            ]
        )
    
    async def close(self):
        """Close the context service"""
        self.initialized = False
        
        # Close narrative manager if initialized
        if self.narrative_manager:
            await self.narrative_manager.close()
        
        logger.info(f"Closed context service for user {self.user_id}, conversation {self.conversation_id}")
    
    # ---------------------------------------------------------------------
    # INTERNAL (PRIVATE) METHODS that used to be @function_tool
    # ---------------------------------------------------------------------

    async def _validate_context_budget(
        self, 
        context: Dict[str, Any], 
        context_budget: int
    ) -> GuardrailFunctionOutput:
        """
        Internal: Validate that the context stays within token budget.

        Returns a GuardrailFunctionOutput indicating if we triggered a tripwire.
        """
        try:
            # Calculate token usage
            token_usage = self._calculate_token_usage(context)
            total_tokens = sum(token_usage.values())
            
            # Check if we're exceeding budget
            if total_tokens > context_budget:
                return GuardrailFunctionOutput(
                    output_info=ContextGuardrailResult(
                        valid=False,
                        reason="Context exceeds token budget",
                        budget_exceeded=True,
                        actual_tokens=total_tokens,
                        max_tokens=context_budget
                    ),
                    tripwire_triggered=True
                )
            
            # Context is valid
            return GuardrailFunctionOutput(
                output_info=ContextGuardrailResult(
                    valid=True,
                    actual_tokens=total_tokens,
                    max_tokens=context_budget
                ),
                tripwire_triggered=False
            )
        except Exception as e:
            logger.error(f"Error validating context budget: {e}")
            return GuardrailFunctionOutput(
                output_info=ContextGuardrailResult(
                    valid=False,
                    reason=f"Error validating context: {str(e)}"
                ),
                tripwire_triggered=True
            )
    
    async def _run_maintenance(self) -> Dict[str, Any]:
        """
        Internal method to run maintenance tasks for context optimization.
        """
        # Initialize if needed
        if not self.initialized:
            await self.initialize()
        
        results = {
            "memory_maintenance": None,
            "vector_maintenance": None,
            "cache_maintenance": None,
            "performance_metrics": None,
            "narrative_maintenance": None
        }
        
        # 1. Memory maintenance
        memory_manager = await get_memory_manager(self.user_id, self.conversation_id)
        memory_result = await memory_manager.run_maintenance(None)  # pass None for ctx
        results["memory_maintenance"] = memory_result
        
        # 2. Vector maintenance
        if self.config.is_enabled("use_vector_search"):
            vector_service = await get_vector_service(self.user_id, self.conversation_id)
            results["vector_maintenance"] = {"status": "vector_service_active"}
        
        # 3. Cache maintenance
        cache_items = (len(context_cache.l1_cache)
                       + len(context_cache.l2_cache)
                       + len(context_cache.l3_cache))
        results["cache_maintenance"] = {
            "cache_items": cache_items,
            "levels": {
                "l1": len(context_cache.l1_cache),
                "l2": len(context_cache.l2_cache),
                "l3": len(context_cache.l3_cache)
            }
        }
        
        # 4. Performance metrics (placeholder)
        if self.performance_monitor:
            results["performance_metrics"] = self.performance_monitor.get_metrics()
        
        # 5. Narrative maintenance if available
        if self.narrative_manager:
            try:
                narrative_result = await self.narrative_manager.run_maintenance()
                results["narrative_maintenance"] = narrative_result
            except Exception as e:
                logger.error(f"Error running narrative maintenance: {e}")
                results["narrative_maintenance"] = {"error": str(e)}
        
        return results
    
    async def _get_base_context(
        self, 
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Internal method: get base context from database or fallback.
        """
        cache_key = f"base_context:{self.user_id}:{self.conversation_id}:{location or 'none'}"
        
        async def fetch_base_context():
            try:
                from db.connection import get_db_connection_context
                import asyncpg
                
                conn = await asyncpg.connect(dsn=get_db_connection_context())
                try:
                    time_info = {
                        "year": "1040",
                        "month": "6",
                        "day": "15",
                        "time_of_day": "Morning"
                    }
                    
                    # Attempt to fetch from CurrentRoleplay table
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
                    
                    # Player stats
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
                    
                    # Roleplay data
                    roleplay_data = {}
                    rp_rows = await conn.fetch("""
                        SELECT key, value
                        FROM CurrentRoleplay
                        WHERE user_id=$1 AND conversation_id=$2
                        AND key IN ('CurrentLocation', 'EnvironmentDesc', 'PlayerRole', 'MainQuest')
                    """, self.user_id, self.conversation_id)
                    
                    for row in rp_rows:
                        roleplay_data[row["key"]] = row["value"]
                    
                    # Narrative stage
                    from logic.narrative_progression import get_current_narrative_stage
                    narrative_stage = await get_current_narrative_stage(self.user_id, self.conversation_id)
                    narrative_stage_info = None
                    if narrative_stage:
                        narrative_stage_info = {
                            "name": narrative_stage.name,
                            "description": narrative_stage.description
                        }
                    
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
        location: Optional[str] = None
    ) -> List[NPCData]:
        """
        Internal method: get NPCs relevant to current input & location 
        (fallback to DB if vector search not available).
        """
        # If vector search is enabled
        if self.vector_service and input_text and self.config.is_enabled("use_vector_search"):
            try:
                vector_context = await self.vector_service.get_context_for_input(
                    input_text=input_text,
                    current_location=location
                )
                if "npcs" in vector_context and vector_context["npcs"]:
                    npcs = []
                    for npc in vector_context["npcs"]:
                        npcs.append(NPCData(
                            npc_id=npc.get("npc_id", ""),
                            npc_name=npc.get("npc_name", ""),
                            dominance=npc.get("dominance"),
                            cruelty=npc.get("cruelty"),
                            closeness=npc.get("closeness"),
                            trust=npc.get("trust"),
                            respect=npc.get("respect"),
                            intensity=npc.get("intensity"),
                            current_location=npc.get("location"),
                            physical_description=npc.get("description", ""),
                            relevance=npc.get("relevance", 0.5)
                        ))
                    return npcs
            except Exception as e:
                logger.error(f"Error getting NPCs from vector service: {e}")
        
        # Fallback to database
        try:
            from db.connection import get_db_connection_context
            import asyncpg
            conn = await asyncpg.connect(dsn=get_db_connection_context())
            try:
                params = [self.user_id, self.conversation_id]
                query = """
                    SELECT npc_id, npc_name,
                           dominance, cruelty, closeness,
                           trust, respect, intensity,
                           current_location, physical_description
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 AND introduced=TRUE
                """
                
                if location:
                    query += " AND (current_location IS NULL OR current_location=$3)"
                    params.append(location)
                
                query += " ORDER BY closeness DESC, trust DESC LIMIT 10"
                
                rows = await conn.fetch(query, *params)
                npcs = []
                for row in rows:
                    npcs.append(NPCData(
                        npc_id=row["npc_id"],
                        npc_name=row["npc_name"],
                        dominance=row["dominance"],
                        cruelty=row["cruelty"],
                        closeness=row["closeness"],
                        trust=row["trust"],
                        respect=row["respect"],
                        intensity=row["intensity"],
                        current_location=row["current_location"] or "Unknown",
                        physical_description=row["physical_description"] or "",
                        relevance=0.7 if row["current_location"] == location else 0.5
                    ))
                return npcs
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error getting NPCs from database: {e}")
            return []
    
    async def _get_location_details(
        self,
        location: Optional[str] = None
    ) -> LocationData:
        """
        Internal method: get details about the current location 
        (fallback to DB if vector not available).
        """
        if not location:
            return LocationData(location_name="Unknown")
        
        if self.vector_service and self.config.is_enabled("use_vector_search"):
            try:
                vc = await self.vector_service.get_context_for_input(
                    input_text=f"Location: {location}",
                    current_location=location
                )
                if "locations" in vc and vc["locations"]:
                    # Attempt to find the best match
                    for loc in vc["locations"]:
                        if loc.get("location_name", "").lower() == location.lower():
                            return LocationData(
                                location_id=loc.get("location_id"),
                                location_name=loc.get("location_name"),
                                description=loc.get("description"),
                                connected_locations=loc.get("connected_locations"),
                                relevance=loc.get("relevance", 0.5)
                            )
            except Exception as e:
                logger.error(f"Error in vector location: {e}")
        
        # Fallback to DB
        try:
            from db.connection import get_db_connection_context
            import asyncpg
            conn = await asyncpg.connect(dsn=get_db_connection_context())
            try:
                row = await conn.fetchrow("""
                    SELECT id, location_name, description
                    FROM Locations
                    WHERE user_id=$1 AND conversation_id=$2 AND location_name=$3
                    LIMIT 1
                """, self.user_id, self.conversation_id, location)
                if row:
                    return LocationData(
                        location_id=str(row["id"]),
                        location_name=row["location_name"],
                        description=row["description"]
                    )
                return LocationData(location_name=location)
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error in get_location_details: {e}")
            return LocationData(location_name=location)
    
    async def _get_quest_information(self) -> List[QuestData]:
        """Internal method: get info about active quests."""
        try:
            from db.connection import get_db_connection_context
            import asyncpg
            conn = await asyncpg.connect(dsn=get_db_connection_context())
            try:
                rows = await conn.fetch("""
                    SELECT quest_id, quest_name, status, progress_detail,
                           quest_giver, reward
                    FROM Quests
                    WHERE user_id=$1 AND conversation_id=$2
                    AND status IN ('active', 'in_progress')
                    ORDER BY quest_id
                """, self.user_id, self.conversation_id)
                
                quests = []
                for row in rows:
                    quests.append(QuestData(
                        quest_id=row["quest_id"],
                        quest_name=row["quest_name"],
                        status=row["status"],
                        progress_detail=row["progress_detail"],
                        quest_giver=row["quest_giver"],
                        reward=row["reward"]
                    ))
                return quests
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error getting quest info: {e}")
            return []
    
    async def _get_narrative_summaries(self, input_text: str) -> Dict[str, Any]:
        """Internal method: get summarized narratives from the narrative manager."""
        if not self.narrative_manager:
            return {}
        try:
            return await self.narrative_manager.get_optimal_narrative_context(
                query=input_text,
                max_tokens=1000
            )
        except Exception as e:
            logger.error(f"Error getting summarized narratives: {e}")
            return {}
    
    def _calculate_token_usage(self, context: Dict[str, Any]) -> Dict[str, int]:
        """
        Internal token usage estimation for the context dictionary,
        counting approximate tokens for each major component.
        """
        def estimate_tokens(obj):
            if obj is None:
                return 0
            elif isinstance(obj, (str, int, float, bool)):
                return max(1, len(str(obj)) // 4)
            elif isinstance(obj, list):
                return sum(estimate_tokens(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(estimate_tokens(k) + estimate_tokens(v) for k, v in obj.items())
            else:
                return 1
        
        token_usage = {}
        components = {
            "player_stats": ["player_stats"],
            "npcs": ["npcs"],
            "memories": ["memories"],
            "location": ["location_details", "locations", "current_location"],
            "quests": ["quests"],
            "time": ["time_info"],
            "roleplay": ["current_roleplay"],
            "narratives": ["narratives"],
            "summaries": ["narrative_summaries"]
        }
        
        for category, keys in components.items():
            total = 0
            for k in keys:
                if k in context:
                    total += estimate_tokens(context[k])
            if total > 0:
                token_usage[category] = total
        
        skip_keys = set()
        for keys in components.values():
            skip_keys.update(keys)
        skip_keys.update(["token_usage", "is_delta", "delta_changes", 
                          "timestamp", "total_tokens", "version"])
        
        other_keys = [k for k in context if k not in skip_keys]
        if other_keys:
            token_usage["other"] = sum(estimate_tokens(context[k]) for k in other_keys)
        
        return token_usage
    
    async def _trim_to_budget(
        self, 
        context: Dict[str, Any], 
        budget: int
    ) -> Dict[str, Any]:
        """
        Internal method: trim context to fit within token budget.
        """
        token_usage = self._calculate_token_usage(context)
        total = sum(token_usage.values())
        
        if total <= budget:
            return context
        
        # Basic priority dict
        trim_priority = {
            "player_stats": 10,
            "time": 9,
            "roleplay": 8,
            "location": 7,
            "quests": 6,
            "npcs": 5,
            "memories": 4,
            "narratives": 3,
            "summaries": 2,
            "other": 1
        }
        reduction_needed = total - budget
        trimmed = context.copy()
        
        # Sort by priority ascending
        components = sorted(token_usage.items(), key=lambda x: trim_priority.get(x[0], 0))
        
        for component_name, usage_val in components:
            if reduction_needed <= 0:
                break
            # Skip very high priority if partial trim is enough
            priority = trim_priority.get(component_name, 0)
            if priority >= 8 and reduction_needed < total * 0.3:
                continue
            
            # Example trimming logic for each component
            if component_name == "npcs" and "npcs" in trimmed:
                npcs = trimmed["npcs"]
                if not npcs:
                    continue
                # Keep top 1/3 
                sorted_npcs = sorted(npcs, key=lambda x: x.get("relevance", 0.5), reverse=True)
                keep_full = max(1, len(sorted_npcs)//3)
                new_npcs = []
                for i, npc in enumerate(sorted_npcs):
                    if i < keep_full:
                        new_npcs.append(npc)
                    else:
                        new_npcs.append({
                            "npc_id": npc.get("npc_id"),
                            "npc_name": npc.get("npc_name"),
                            "current_location": npc.get("current_location"),
                            "relevance": npc.get("relevance", 0.5)
                        })
                old_tokens = usage_val
                new_tokens = self._calculate_token_usage({"npcs": new_npcs}).get("npcs", 0)
                trimmed["npcs"] = new_npcs
                reduction_needed -= (old_tokens - new_tokens)
            
            elif component_name == "memories" and "memories" in trimmed:
                memories = trimmed["memories"]
                if not memories:
                    continue
                sorted_memories = sorted(memories, key=lambda x: x.get("importance", 0.5), reverse=True)
                keep_count = max(1, len(sorted_memories)//2)
                new_memories = sorted_memories[:keep_count]
                
                # Summarize if we still have to trim
                if reduction_needed > 0 and self.narrative_manager:
                    for i, mem in enumerate(new_memories):
                        content = mem.get("content", "")
                        if len(content) > 200:
                            summarized_text = await self._summarize_text(content, 2)
                            new_memories[i]["content"] = summarized_text
                            new_memories[i]["summarized"] = True
                
                old_tokens = usage_val
                new_tokens = self._calculate_token_usage({"memories": new_memories}).get("memories", 0)
                trimmed["memories"] = new_memories
                reduction_needed -= (old_tokens - new_tokens)
            
            elif component_name == "narratives" and "narratives" in trimmed:
                old_tokens = usage_val
                del trimmed["narratives"]
                reduction_needed -= old_tokens
            
            elif component_name == "summaries" and "narrative_summaries" in trimmed:
                old_tokens = usage_val
                del trimmed["narrative_summaries"]
                reduction_needed -= old_tokens
        
        return trimmed
    
    async def _summarize_text(self, text: str, level: int) -> str:
        """
        Internal method: Summarize text for trimming or user requests.
        level=1 => condensed, 2 => summary, 3 => headline
        """
        if level <= 0:
            return text
        
        # Fallback if no narrative_manager
        if not self.narrative_manager:
            sentences = text.split(". ")
            if level == 1:
                if len(sentences) >= 3:
                    mid = len(sentences)//2
                    return f"{sentences[0]}. {sentences[mid]}. {sentences[-1]}."
                return text
            elif level == 2:
                if len(sentences) >= 2:
                    return f"{sentences[0]}. {sentences[-1]}."
                return text
            elif level == 3:
                if sentences:
                    if len(sentences[0])>100:
                        return sentences[0][:97]+"..."
                    return sentences[0]
                return text
        
        # If narrative manager available
        try:
            from story_agent.progressive_summarization import SummaryLevel
            summary_map = {
                0: SummaryLevel.DETAILED,
                1: SummaryLevel.CONDENSED,
                2: SummaryLevel.SUMMARY,
                3: SummaryLevel.HEADLINE
            }
            summarizer = self.narrative_manager.narrative.summarizer
            res = await summarizer.summarize(text, summary_map[level])
            return res
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return text

    # ---------------------------------------------------------------------
    # Public (non-tool) Methods used by get_context / get_summarized_context
    # ---------------------------------------------------------------------

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
        source_version: Optional[int] = None,
        summary_level: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for obtaining context 
        (non-tool method, directly called from outside or via aggregator).
        """
        if not self.initialized:
            await self.initialize()
        
        request = ContextRequest(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            input_text=input_text,
            location=location,
            context_budget=context_budget,
            use_vector_search=use_vector_search,
            use_delta=use_delta,
            include_memories=include_memories,
            include_npcs=include_npcs,
            include_location=include_location,
            include_quests=include_quests,
            source_version=source_version,
            summary_level=summary_level
        )
        
        # Example: we might orchestrate specialized calls, but let's do a direct approach for demonstration.
        base_context = await self._get_base_context(location)
        
        # Possibly get relevant NPCs
        if include_npcs:
            npcs = await self._get_relevant_npcs(input_text=input_text, location=location)
            base_context["npcs"] = [npc.dict() for npc in npcs]
        
        # Possibly get location details
        if include_location:
            loc_data = await self._get_location_details(location)
            base_context["location_details"] = loc_data.dict()
        
        # Possibly get quest info
        if include_quests:
            quests = await self._get_quest_information()
            base_context["quests"] = [q.dict() for q in quests]
        
        # Possibly get memories from memory manager
        if include_memories:
            # Use memory tool calls or direct manager calls
            # E.g. memory_manager.get_recent_memories, etc.
            pass
        
        # If delta requested, we can fetch from context_manager
        if use_delta and source_version is not None:
            # For demonstration, let's just store a partial delta
            delta_context = await self.context_manager._get_context(source_version)
            base_context["is_delta"] = delta_context.get("is_incremental", False)
            if "delta_context" in delta_context:
                base_context["delta_changes"] = delta_context["delta_context"]
            base_context["version"] = delta_context["version"]
        else:
            base_context["version"] = self.context_manager.version
        
        # Trim to budget
        final_context = await self._trim_to_budget(base_context, context_budget)
        
        # Return final
        return final_context
    
    async def get_summarized_context(
        self,
        input_text: str = "",
        summary_level: int = 1,
        context_budget: int = 2000,
        use_vector_search: bool = True
    ) -> Dict[str, Any]:
        """
        Summarized version of context. 
        Might call get_context + additional summarization steps.
        """
        if not self.initialized:
            await self.initialize()
        
        # For demonstration, just call get_context with smaller budget
        raw_context = await self.get_context(
            input_text=input_text,
            context_budget=context_budget,
            use_vector_search=use_vector_search
        )
        # Then do an extra summarization pass
        # (We can refine as needed)
        summarized = raw_context.copy()
        
        # Summarize memories, NPC descriptions, location details, etc.
        # Minimal example:
        if "npcs" in summarized:
            for npc in summarized["npcs"]:
                desc = npc.get("physical_description", "")
                if len(desc)>100 and summary_level>1:
                    npc["physical_description"] = desc[:100] + "..."
        
        return summarized
    
    # ---------------------------------------------------------------------
    # Private helper for the "trim" logic
    # ---------------------------------------------------------------------
    # (already in _trim_to_budget, etc.)
    
# ---------------------------------------------------------------------
# Global Registry for ContextService
# ---------------------------------------------------------------------
_context_services = {}

async def get_context_service(user_id: int, conversation_id: int) -> ContextService:
    """Get or create a context service instance"""
    key = f"{user_id}:{conversation_id}"
    if key not in _context_services:
        service = ContextService(user_id, conversation_id)
        await service.initialize()
        _context_services[key] = service
    return _context_services[key]

async def get_comprehensive_context(
    user_id: int,
    conversation_id: int,
    input_text: str = "",
    location: Optional[str] = None,
    context_budget: int = 4000,
    use_vector_search: Optional[bool] = None,
    use_delta: bool = True,
    source_version: Optional[int] = None,
    summary_level: Optional[int] = None
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
    for svc in _context_services.values():
        await svc.close()
    _context_services.clear()
    
    # Also close vector & memory if needed
    from context.vector_service import cleanup_vector_services
    from context.memory_manager import cleanup_memory_managers
    await cleanup_vector_services()
    await cleanup_memory_managers()


# ---------------------------------------------------------------------
# STANDALONE TOOL FUNCTIONS (with @function_tool, "ctx" is first param)
# ---------------------------------------------------------------------

@function_tool
async def validate_context_budget_tool(
    ctx: RunContextWrapper,
    context: Dict[str, Any],
    context_budget: int
) -> GuardrailFunctionOutput:
    """
    Standalone tool: Validate that the context stays within token budget.
    Library sees `ctx` as the first parameter => correct signature.
    """
    # We must figure out how to get the relevant ContextService instance (by user_id?).
    # If your code can pass user_id in the signature, do so. Otherwise assume "ctx" includes it.
    
    # Example: assume "ctx" has "user_id" / "conversation_id" in it
    user_id = ctx.get("user_id", 0)
    conversation_id = ctx.get("conversation_id", 0)
    
    service = await get_context_service(user_id, conversation_id)
    return await service._validate_context_budget(context, context_budget)


@function_tool
async def run_context_maintenance_tool(
    ctx: RunContextWrapper
) -> Dict[str, Any]:
    """
    Standalone tool: run maintenance tasks 
    (calls the private method _run_maintenance inside the ContextService).
    """
    user_id = ctx.get("user_id", 0)
    conversation_id = ctx.get("conversation_id", 0)
    service = await get_context_service(user_id, conversation_id)
    return await service._run_maintenance()


@function_tool
async def get_base_context_tool(
    ctx: RunContextWrapper,
    location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Standalone tool: get base context from DB or fallback,
    calling the private method `_get_base_context`.
    """
    user_id = ctx.get("user_id", 0)
    conversation_id = ctx.get("conversation_id", 0)
    service = await get_context_service(user_id, conversation_id)
    return await service._get_base_context(location)


@function_tool
async def get_relevant_npcs_tool(
    ctx: RunContextWrapper,
    input_text: str,
    location: Optional[str] = None
) -> List[NPCData]:
    """
    Standalone tool: get NPCs relevant to the current input & location
    from the private method `_get_relevant_npcs`.
    """
    user_id = ctx.get("user_id", 0)
    conversation_id = ctx.get("conversation_id", 0)
    service = await get_context_service(user_id, conversation_id)
    return await service._get_relevant_npcs(input_text, location)


@function_tool
async def get_location_details_tool(
    ctx: RunContextWrapper,
    location: Optional[str] = None
) -> LocationData:
    """
    Standalone tool: get details about the current location
    from `_get_location_details`.
    """
    user_id = ctx.get("user_id", 0)
    conversation_id = ctx.get("conversation_id", 0)
    service = await get_context_service(user_id, conversation_id)
    return await service._get_location_details(location)


@function_tool
async def get_quest_information_tool(
    ctx: RunContextWrapper
) -> List[QuestData]:
    """
    Standalone tool: get info about active quests
    via `_get_quest_information`.
    """
    user_id = ctx.get("user_id", 0)
    conversation_id = ctx.get("conversation_id", 0)
    service = await get_context_service(user_id, conversation_id)
    return await service._get_quest_information()


@function_tool
async def get_narrative_summaries_tool(
    ctx: RunContextWrapper,
    input_text: str
) -> Dict[str, Any]:
    """
    Standalone tool: get summarized narratives from `_get_narrative_summaries`.
    """
    user_id = ctx.get("user_id", 0)
    conversation_id = ctx.get("conversation_id", 0)
    service = await get_context_service(user_id, conversation_id)
    return await service._get_narrative_summaries(input_text)


@function_tool
async def trim_to_budget_tool(
    ctx: RunContextWrapper,
    context: Dict[str, Any],
    budget: int
) -> Dict[str, Any]:
    """
    Standalone tool: trim context to fit within `budget` tokens.
    """
    user_id = ctx.get("user_id", 0)
    conversation_id = ctx.get("conversation_id", 0)
    service = await get_context_service(user_id, conversation_id)
    return await service._trim_to_budget(context, budget)


# ---------------------------------------------------------------------
# Agent Creation referencing the standalone tool functions
# ---------------------------------------------------------------------

def create_context_service_orchestrator() -> Agent:
    """
    Create the orchestration agent for the context service,
    now referencing the standalone tool functions instead of instance methods.
    """
    agent = Agent(
        name="Context Service Orchestrator",
        instructions="""
        You are a context service orchestrator specialized in managing 
        context for RPG interactions.
        """,
        tools=[
            validate_context_budget_tool,
            run_context_maintenance_tool,
            get_base_context_tool,
            get_relevant_npcs_tool,
            get_location_details_tool,
            get_quest_information_tool,
            get_narrative_summaries_tool,
            trim_to_budget_tool
        ]
    )
    return agent


def get_context_service_orchestrator() -> Agent:
    """Get the context service orchestrator agent."""
    return create_context_service_orchestrator()
