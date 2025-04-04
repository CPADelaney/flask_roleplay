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

# --- Pydantic Models for Context Requests and Responses ---

class ContextGuardrailResult(BaseModel):
    """Result of context guardrail check"""
    valid: bool = True
    reason: Optional[str] = None
    budget_exceeded: bool = False
    actual_tokens: Optional[int] = None
    max_tokens: Optional[int] = None


# --- Context Service ---

class ContextService:
    """
    Unified context service that integrates all context components.
    Refactored to use OpenAI Agents SDK
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
        
        # Agent components
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
        
        # Initialize narrative manager for progressive summarization
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
        
        # Initialize agents
        await self._initialize_agents()
        
        self.initialized = True
        logger.info(f"Initialized context service for user {self.user_id}, conversation {self.conversation_id}")
    
    async def _initialize_agents(self):
        """Initialize the agent ecosystem"""
        # Get the specialized agents
        self.context_agent = get_context_manager_agent()
        self.memory_agent = get_memory_agent()
        
        # Create orchestrator agent with handoffs
        self.orchestrator_agent = Agent(
            name="Context Orchestrator",
            instructions="""
            You are the orchestrator for context management in an RPG system.
            You decide which specialized agent should handle each request based on the task:
            
            1. For memory-related tasks: Use the Memory Manager agent
            2. For context management tasks: Use the Context Manager agent
            3. For vector search tasks: Use the Vector Search agent
            4. For narrative tasks: Use the Narrative agent
            
            Make the handoff decision based on the request type and content.
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
            ],
            # Add more handoffs when we implement those agents
        )
    
    async def close(self):
        """Close the context service"""
        self.initialized = False
        
        # Close narrative manager if initialized
        if self.narrative_manager:
            await self.narrative_manager.close()
        
        logger.info(f"Closed context service for user {self.user_id}, conversation {self.conversation_id}")
    
    @function_tool
    async def validate_context_budget(
        self, 
        ctx: RunContextWrapper,
        context: Dict[str, Any], 
        context_budget: int
    ) -> GuardrailFunctionOutput:
        """
        Validate that the context stays within token budget
        
        Args:
            context: The context to validate
            context_budget: Maximum token budget
            
        Returns:
            Validation result with budget information
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
        source_version: Optional[int] = None,  # Version tracking for delta updates
        summary_level: Optional[int] = None  # Summary level option
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
            summary_level: Optional summary level (0-3)
            
        Returns:
            Optimized context
        """
        # Initialize if needed
        if not self.initialized:
            await self.initialize()
        
        # Create request model
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
        
        # Get model settings from config
        model_settings = self.config.get_model_settings()
        model = self.config.get_default_model()
        
        # Create the output guardrail for token budget
        async def context_budget_guardrail(ctx, agent, output):
            result = await self.validate_context_budget(ctx, output.dict(), context_budget)
            return result
        
        # Create run configuration
        run_config = RunConfig(
            model=model,
            model_settings=model_settings,
            workflow_name="get_context",
            output_guardrails=[OutputGuardrail(guardrail_function=context_budget_guardrail)]
        )
        
        # Run the context agent
        with trace(workflow_name="context_service.get_context", 
                  group_id=f"{self.user_id}:{self.conversation_id}"):
            try:
                # If summarization requested, use a different approach
                if summary_level is not None:
                    result = await Runner.run(
                        self.context_agent,
                        {"request": request.dict(), "action": "get_summarized_context"},
                        context=self.agent_context,
                        run_config=run_config
                    )
                else:
                    # Use the orchestrator agent to decide which specialized agent to use
                    result = await Runner.run(
                        self.orchestrator_agent,
                        {"request": request.dict(), "action": "get_context"},
                        context=self.agent_context,
                        run_config=run_config
                    )
                
                return result.final_output
            except Exception as e:
                logger.error(f"Error getting context: {e}")
                return {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
    
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
        if not self.initialized:
            await self.initialize()
        
        # Create request model
        request = ContextRequest(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            input_text=input_text,
            context_budget=context_budget,
            use_vector_search=use_vector_search,
            summary_level=summary_level
        )
        
        # Run the summarization process
        with trace(workflow_name="context_service.get_summarized_context", 
                  group_id=f"{self.user_id}:{self.conversation_id}"):
            try:
                # First get the base context
                base_context = await self.get_context(
                    input_text=input_text,
                    context_budget=context_budget,
                    use_vector_search=use_vector_search,
                    use_delta=False
                )
                
                # Apply summarization if needed
                if summary_level <= 0:  # Detailed - no summarization
                    return base_context
                
                # Create a new instance of the context with summarization
                result = await self._summarize_context(base_context, summary_level)
                return result
            except Exception as e:
                logger.error(f"Error getting summarized context: {e}")
                return {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
    
    async def _summarize_context(self, context: Dict[str, Any], level: int) -> Dict[str, Any]:
        """Summarize context at the specified level"""
        # Make a copy to avoid modifying the original
        summarized = context.copy()
        
        # Summarize memories if present
        if "memories" in summarized:
            for i, memory in enumerate(summarized["memories"]):
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
                summarized_content = await self._summarize_text(content, level)
                
                # Replace content with summarized version
                summarized["memories"][i]["content"] = summarized_content
                summarized["memories"][i]["summarized"] = True
                summarized["memories"][i]["summary_level"] = level
        
        # If narrative manager is available, use it for summarization
        if self.narrative_manager and "narrative_summaries" not in summarized:
            try:
                # Get summarized narratives
                input_text = context.get("input_text", "")
                narrative_context = await self.narrative_manager.get_optimal_narrative_context(
                    query=input_text,
                    max_tokens=context.get("context_budget", 4000) // 4  # Use 25% of budget for narrative
                )
                
                # Add to context
                summarized["narrative_summaries"] = narrative_context
            except Exception as e:
                logger.error(f"Error getting narrative summaries: {e}")
        
        return summarized
    
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
    
    @function_tool
    async def _get_base_context(
        self, 
        ctx: RunContextWrapper,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
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
                    
                    # Get narrative stage
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
    
    @function_tool
    async def _get_relevant_npcs(
        self,
        ctx: RunContextWrapper,
        input_text: str,
        location: Optional[str] = None
    ) -> List[NPCData]:
        """
        Get NPCs relevant to current input and location
        
        Args:
            input_text: Current input text
            location: Optional location
            
        Returns:
            List of relevant NPCs
        """
        # If there's vector search, prioritize it
        if self.vector_service and input_text and self.config.is_enabled("use_vector_search"):
            try:
                # Get vector context for input
                vector_context = await self.vector_service.get_context_for_input(
                    input_text=input_text,
                    current_location=location
                )
                
                # Extract NPCs
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
                            current_location=npc.get("current_location"),
                            physical_description=npc.get("physical_description", ""),
                            relevance=npc.get("relevance", 0.5)
                        ))
                    return npcs
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
                        # Estimated relevance since we're not using vector search
                        relevance=0.7 if row["current_location"] == location else 0.5
                    ))
                
                return npcs
            
            finally:
                await conn.close()
        
        except Exception as e:
            logger.error(f"Error getting NPCs from database: {e}")
            return []
    
    @function_tool
    async def _get_location_details(
        self,
        ctx: RunContextWrapper,
        location: Optional[str] = None
    ) -> LocationData:
        """
        Get details about the current location
        
        Args:
            location: Current location
            
        Returns:
            Location details
        """
        if not location:
            return LocationData(location_name="Unknown")
        
        # Use vector search if available
        if self.vector_service and self.config.is_enabled("use_vector_search"):
            try:
                # Get vector context for location
                vector_context = await self.vector_service.get_context_for_input(
                    input_text=f"Location: {location}",
                    current_location=location
                )
                
                # Extract locations
                if "locations" in vector_context and vector_context["locations"]:
                    for loc in vector_context["locations"]:
                        if loc.get("location_name", "").lower() == location.lower():
                            return LocationData(
                                location_id=loc.get("location_id"),
                                location_name=loc.get("location_name"),
                                description=loc.get("description"),
                                connected_locations=loc.get("connected_locations"),
                                relevance=loc.get("relevance", 0.5)
                            )
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
                    return LocationData(
                        location_id=str(row["id"]),
                        location_name=row["location_name"],
                        description=row["description"]
                    )
                
                return LocationData(location_name=location)
            
            finally:
                await conn.close()
        
        except Exception as e:
            logger.error(f"Error getting location details: {e}")
            return LocationData(location_name=location)
    
    @function_tool
    async def _get_quest_information(self, ctx: RunContextWrapper) -> List[QuestData]:
        """
        Get information about active quests
        
        Args:
            None
            
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
            logger.error(f"Error getting quest information: {e}")
            return []
    
    @function_tool
    async def _get_narrative_summaries(
        self,
        ctx: RunContextWrapper,
        input_text: str
    ) -> Dict[str, Any]:
        """
        Get summarized narratives from narrative manager
        
        Args:
            input_text: Current input text
            
        Returns:
            Narrative summaries
        """
        if not self.narrative_manager:
            return {}
        
        try:
            # Get optimal narrative context with query
            return await self.narrative_manager.get_optimal_narrative_context(
                query=input_text,
                max_tokens=1000
            )
        except Exception as e:
            logger.error(f"Error getting summarized narratives: {e}")
            return {}
    
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
            "summaries": ["narrative_summaries"]  # Track narrative summaries
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
    
    @function_tool
    async def _trim_to_budget(
        self, 
        ctx: RunContextWrapper,
        context: Dict[str, Any], 
        budget: int
    ) -> Dict[str, Any]:
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
    
    @function_tool
    async def run_maintenance(self, ctx: RunContextWrapper) -> Dict[str, Any]:
        """
        Run maintenance tasks for context optimization
        
        Returns:
            Maintenance results
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
        memory_result = await memory_manager.run_maintenance(ctx)
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


# --- Global Service Registry and Factory ---
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


def create_context_service_orchestrator() -> Agent:
    """Create the orchestration agent for context service"""
    # Create a new instance of ContextService
    # This is a placeholder - in real usage, you'd initialize with actual user_id and conversation_id
    service = ContextService(user_id=0, conversation_id=0)
    
    # Define the agent with tools from the service
    agent = Agent(
        name="Context Service Orchestrator",
        instructions="""
        You are a context service orchestrator specialized in managing the context for RPG interactions.
        Your tasks include:
        
        1. Getting appropriate context for the current interaction
        2. Trimming context to fit token budgets
        3. Summarizing context based on importance
        4. Running maintenance tasks on the context system
        
        Work with specialized agents for memory management, vector search, and narrative management.
        """,
        tools=[
            service._get_base_context,
            service._get_relevant_npcs,
            service._get_location_details,
            service._get_quest_information,
            service._get_narrative_summaries,
            service._trim_to_budget,
            service.validate_context_budget,
            service.run_maintenance,
        ],
    )
    
    return agent


def get_context_service_orchestrator() -> Agent:
    """Get the context service orchestrator agent"""
    return create_context_service_orchestrator()
