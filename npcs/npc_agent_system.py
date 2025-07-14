# npcs/npc_agent_system.py

"""
Main system that integrates NPC agents with the game loop, using OpenAI Agents SDK.
Refactored to use the new canon/LoreSystem architecture.
"""

import logging
import asyncio
import random
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, validator
from agents import Agent, Runner, function_tool, handoff, trace, ModelSettings, input_guardrail, GuardrailFunctionOutput

from npcs.npc_agent import NPCAgent, ResourcePool
from memory.wrapper import MemorySystem
from .lore_context_manager import LoreContextManager
from db.connection import get_db_connection_context  # Updated import
from lore.core import canon
from lore.lore_system import LoreSystem
from agents import RunContextWrapper

logger = logging.getLogger(__name__)

class InitializeAgentsResult(BaseModel):
    """Result from initializing agents."""
    npc_count: int
    initialized_ids: List[int]

class PlayerAction(BaseModel):
    """Model for player actions."""
    type: str
    description: str
    target_npc_id: Optional[int] = None
    target_location: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Removed extra = "allow" to comply with strict mode
    # If you need additional fields, you can add them explicitly

class ActionContext(BaseModel):
    """Model for action context."""
    location: Optional[str] = None
    time_of_day: Optional[str] = None
    year: Optional[int] = None
    month: Optional[str] = None
    day: Optional[int] = None
    activity_type: Optional[str] = None
    is_group_interaction: bool = False
    affected_npcs: List[int] = Field(default_factory=list)
    emotional_states: Dict[int, Dict[str, Any]] = Field(default_factory=dict)
    mask_states: Dict[int, Dict[str, Any]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Removed extra = "allow" to comply with strict mode

class NPCResponse(BaseModel):
    """Model for NPC response."""
    npc_id: int
    npc_name: str
    response_type: str
    response_text: str
    emotional_impact: Optional[Dict[str, Any]] = None
    stat_changes: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HandlePlayerActionResult(BaseModel):
    """Result from handling player action."""
    npc_responses: List[Union[NPCResponse, Dict[str, Any]]]

class DetermineAffectedNPCsInput(BaseModel):
    """Input for determine_affected_npcs."""
    player_action: PlayerAction
    context: ActionContext

class BatchUpdateData(BaseModel):
    """Data for batch updates."""
    # For location changes
    new_location: Optional[str] = None
    # For emotional updates  
    emotion: Optional[str] = None
    intensity: Optional[float] = 0.5
    # For trait updates
    traits: Optional[Dict[str, Any]] = None
    # For any other update type
    data: Dict[str, Any] = Field(default_factory=dict)
    
    # Removed extra = "allow" to comply with strict mode

class BatchUpdateResult(BaseModel):
    """Result from batch update."""
    success_count: int = 0
    error_count: int = 0
    details: Dict[int, Dict[str, Any]] = Field(default_factory=dict)
    error: Optional[str] = None
    valid_types: Optional[List[str]] = None

class GameTime(BaseModel):
    """Game time information."""
    year: int = 2023
    month: str = "January"
    day: int = 1
    time_of_day: str = "afternoon"

class ScheduledActivitiesResult(BaseModel):
    """Result from processing scheduled activities."""
    npc_responses: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    time_of_day: str = "afternoon"

class MaintenanceResult(BaseModel):
    """Result from memory maintenance."""
    player_maintenance: Optional[Dict[str, Any]] = None
    npc_maintenance: Dict[int, Dict[str, Any]] = Field(default_factory=dict)
    nyx_maintenance: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class FlashbackResult(BaseModel):
    """Result from generating flashback."""
    triggered: bool
    reason: Optional[str] = None
    text: Optional[str] = None
    memory_id: Optional[str] = None
    triggered_emotion: Optional[str] = None
    emotion_intensity: Optional[float] = None
    time_since_last_seconds: Optional[float] = None
    error: Optional[str] = None

class NPCSystemError(Exception):
    """Error in NPC system operations."""
    pass

class SystemInput(BaseModel):
    """Input for the system agent."""
    command: str
    parameters: Dict[str, Any]

class SystemOutput(BaseModel):
    """Output from the system agent."""
    result: Dict[str, Any]
    status: str
    message: str

class ModerationCheck(BaseModel):
    """Output for moderation guardrail check."""
    is_appropriate: bool = True
    reasoning: str = ""

class NPCAgentSystem:
    """
    Main system that integrates individual NPC agents with the game loop using the OpenAI Agents SDK.
    Refactored to use canon for creation and LoreSystem for updates.

    Responsibilities:
    - Load and store a reference to each NPC agent (NPCAgent).
    - Provide methods to handle player actions directed at NPC(s),
      determining which NPCs are affected, and dispatching to single
      or group interaction handlers.
    - Process scheduled activities for all NPCs.
    - Coordinate memory-related operations across NPCs.
    """

    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        connection_pool  # No change to this parameter as it's passed in
    ):
        """
        Initialize the agent system for a specific user & conversation.
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.npc_agents: Dict[int, NPCAgent] = {}
        self._memory_system = None
        self.connection_pool = connection_pool
        self._system_agent = None
        self._lore_system = None
        self.lore_context_manager = LoreContextManager(user_id, conversation_id)

        # Track when memory maintenance was last run
        self._last_memory_maintenance = datetime.now() - timedelta(hours=1)  # Run on first init
        # Track NPC emotional states to detect significant changes
        self._npc_emotional_states = {}
        # Track last flashback times to prevent too-frequent occurrences
        self._last_flashback_times = {}

        # Resource pools for different operations
        self.resource_pools = {
            "decisions": ResourcePool(max_concurrent=10, timeout=45.0),
            "perceptions": ResourcePool(max_concurrent=15, timeout=30.0),
            "memory_operations": ResourcePool(max_concurrent=20, timeout=20.0)
        }

        # Schedule periodic memory maintenance
        self._setup_memory_maintenance_schedule()

    def _setup_memory_maintenance_schedule(self):
        """Schedule periodic memory maintenance for all NPCs."""
        async def run_maintenance_cycle():
            while True:
                try:
                    # Run maintenance every 15 minutes
                    await asyncio.sleep(900)  # 15 minutes in seconds

                    # Log maintenance start
                    logger.info("Starting scheduled memory maintenance cycle")

                    # Run maintenance
                    results = await self.run_memory_maintenance()

                    # Log completion
                    logger.info(f"Completed memory maintenance: {results}")

                except Exception as e:
                    logger.error(f"Error in scheduled memory maintenance: {e}")

        # Start the maintenance task
        asyncio.create_task(run_maintenance_cycle())

    async def _get_lore_system(self):
        """Lazy-load the lore system."""
        if self._lore_system is None:
            self._lore_system = await LoreSystem.get_instance(
                self.user_id, self.conversation_id
            )
        return self._lore_system

    async def update_npc_directive(self, npc_id: int, directive: Dict[str, Any]) -> Dict[str, Any]:
        """Process a directive from Nyx for a specific NPC."""
        if npc_id in self.npc_agents:
            # Set directive in NPC's context
            self.npc_agents[npc_id].context.current_directive = directive
            
            # Log directive for debugging
            logger.info(f"NPC {npc_id} received directive from Nyx: {directive}")
            
            return {"success": True, "npc_id": npc_id}
        else:
            return {"success": False, "error": f"NPC {npc_id} not found"}

    async def _get_memory_system(self):
        """Lazy-load the memory system."""
        if self._memory_system is None:
            self._memory_system = await MemorySystem.get_instance(
                self.user_id, self.conversation_id
            )
        return self._memory_system

    async def _get_system_agent(self):
        """Lazy-load the system agent."""
        if self._system_agent is None:
            # Set up the moderation check agent
            moderation_agent = Agent(
                name="Content_Moderator",
                instructions="""
                Your job is to check if content is appropriate and doesn't violate content policies.
                Flag any content that:
                1. Contains explicit or graphic violence
                2. Contains explicit sexual content
                3. Promotes illegal activities
                4. Contains hate speech or discriminatory content
                
                Output true for is_appropriate if the content is appropriate, false otherwise.
                """,
                output_type=ModerationCheck
            )
            
            async def moderation_guardrail(
                ctx, 
                agent: Agent, 
                input_data: Union[str, Dict[str, Any]]
            ) -> GuardrailFunctionOutput:
                # Extract text from input
                text = ""
                if isinstance(input_data, str):
                    text = input_data
                elif isinstance(input_data, dict):
                    if "command" in input_data and input_data["command"] == "process_player_action":
                        if "parameters" in input_data and "player_action" in input_data["parameters"]:
                            player_action = input_data["parameters"]["player_action"]
                            text = player_action.get("description", "")
                
                # Check moderation if we have text
                if text:
                    result = await Runner.run(moderation_agent, text, context=ctx.context)
                    final_output = result.final_output_as(ModerationCheck)
                    return GuardrailFunctionOutput(
                        output_info=final_output,
                        tripwire_triggered=not final_output.is_appropriate
                    )
                
                # Default to appropriate if no text to check
                return GuardrailFunctionOutput(
                    output_info=ModerationCheck(is_appropriate=True, reasoning="No text to check"),
                    tripwire_triggered=False
                )
            
            # Create the main system agent
            self._system_agent = Agent(
                name="NPC_System_Agent",
                instructions="""
                You are the main coordinator for an NPC agent system. You handle requests related to:
                
                1. Processing player actions directed at NPCs
                2. Managing scheduled activities for NPCs
                3. Coordinating memory operations
                4. Loading and initializing NPC agents
                
                Your responses should be efficient and focused on the specific task requested.
                """,
                model="gpt-4.1-nano",
                model_settings=ModelSettings(temperature=0.2),
                tools=[
                    function_tool(self.handle_player_action, strict_mode=False),
                    function_tool(self.process_npc_scheduled_activities, strict_mode=False),
                    function_tool(self.run_memory_maintenance, strict_mode=False),
                    function_tool(self.initialize_agents, strict_mode=False),
                    function_tool(self.batch_update_npcs, strict_mode=False),
                    function_tool(self.generate_npc_flashback, strict_mode=False)
                ],
                input_guardrails=[input_guardrail(moderation_guardrail)],
                output_type=SystemOutput
            )
        
        return self._system_agent

    @function_tool(strict_mode=False)
    async def initialize_agents(self, npc_ids: Optional[List[int]] = None) -> InitializeAgentsResult:
        """
        Initialize NPCAgent objects for specified NPCs or all NPCs in the conversation.
        This is a READ-ONLY operation - no database writes.
        """
        logger.info(
            "Initializing NPC agents for user=%s, conversation=%s",
            self.user_id, self.conversation_id
        )

        query = """
            SELECT npc_id
            FROM NPCStats
            WHERE user_id=$1
              AND conversation_id=$2
        """
        params = [self.user_id, self.conversation_id]
        
        if npc_ids:
            query += " AND npc_id = ANY($3)"
            params.append(npc_ids)

        # Using the provided connection pool - READ ONLY
        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
            # Create tasks to initialize agents concurrently
            init_tasks = []
            for row in rows:
                npc_id = row["npc_id"]
                if npc_id not in self.npc_agents:
                    self.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
                    init_tasks.append(self.npc_agents[npc_id].initialize())
            
            # Wait for all initializations to complete
            if init_tasks:
                await asyncio.gather(*init_tasks)

        npc_count = len(self.npc_agents)
        logger.info("Loaded %d NPC agents", npc_count)
        
        return InitializeAgentsResult(
            npc_count=npc_count,
            initialized_ids=list(self.npc_agents.keys())
        )
        
    @function_tool(strict_mode=False)
    async def handle_player_action(
        self,
        player_action: PlayerAction,
        context: Optional[ActionContext] = None
    ) -> HandlePlayerActionResult:
        """
        Handle a player action and determine NPC responses with memory integration.
        This creates memories but doesn't modify core NPC state directly.
        """
        if context is None:
            context = ActionContext()
    
        # Create trace for debugging and monitoring
        with trace(
            f"player_action_{self.user_id}_{self.conversation_id}",
            group_id=f"user_{self.user_id}_conv_{self.conversation_id}"
        ):
            # Get the memory system for creating player action memories
            memory_system = await self._get_memory_system()
    
            # Create a memory of this action from the player's perspective
            player_memory_text = f"I {player_action.description}"
    
            # Determine emotional content based on action type
            is_emotional = player_action.type in ["express_emotion", "shout", "cry", "laugh", "threaten"]
    
            # Add memory with appropriate tags
            await memory_system.remember(
                entity_type="player",
                entity_id=self.user_id,
                memory_text=player_memory_text,
                importance="medium",
                emotional=is_emotional,
                tags=["player_action", player_action.type]
            )
    
            # Determine which NPCs are affected by the action
            # Convert models to dicts for compatibility with existing code
            affected_npcs = await self.determine_affected_npcs(
                player_action.model_dump(), 
                context.model_dump()
            )
            if not affected_npcs:
                logger.debug("No NPCs were affected by this action: %s", player_action)
                return HandlePlayerActionResult(npc_responses=[])
    
            # Single NPC path
            if len(affected_npcs) == 1:
                npc_id = affected_npcs[0]
                result = await self.handle_single_npc_interaction(
                    npc_id, 
                    player_action.model_dump(), 
                    context.model_dump()
                )
                return HandlePlayerActionResult(npc_responses=result.get("npc_responses", []))
    
            # Multiple NPCs => group logic
            result = await self.handle_group_npc_interaction(
                affected_npcs, 
                player_action.model_dump(), 
                context.model_dump()
            )
            return HandlePlayerActionResult(npc_responses=result.get("npc_responses", []))

    @function_tool(strict_mode=False)
    async def determine_affected_npcs(
        self,
        player_action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[int]:
        """
        Figure out which NPCs are affected by a given player action.
        This is a READ-ONLY operation.
        """
        # [Keep the existing implementation as-is]
        target_npc_id = player_action.get("target_npc_id")
        if target_npc_id:
            return [target_npc_id]

        location = player_action.get("target_location", context.get("location"))
        if not location:
            # Attempt to get location from CurrentRoleplay table
            location = await self._fetch_current_location()
            if not location:
                logger.debug(
                    "No location found in context or CurrentRoleplay; can't determine affected NPCs."
                )

        location_npcs = []
        if location:
            try:
                # Using the provided connection pool - READ ONLY
                async with self.connection_pool.acquire() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT npc_id
                        FROM NPCStats
                        WHERE user_id=$1
                          AND conversation_id=$2
                          AND current_location=$3
                          AND introduced = TRUE
                        """,
                        self.user_id, self.conversation_id, location
                    )
                    location_npcs = [row["npc_id"] for row in rows]
            except Exception as e:
                logger.error("Error getting NPCs in location '%s': %s", location, e)

        if location_npcs:
            # If we have action text, we can prioritize NPCs by memory relevance
            if "description" in player_action and player_action.get("description"):
                try:
                    memory_system = await self._get_memory_system()

                    relevant_npcs = []
                    for npc_id in location_npcs:
                        # Get NPC memories relevant to this action
                        memory_result = await memory_system.recall(
                            entity_type="npc",
                            entity_id=npc_id,
                            query=player_action.get("description", ""),
                            limit=3
                        )
                        memories = memory_result.get("memories", [])
                        # Calculate relevance based on memory count and significance
                        relevance = len(memories)
                        for memory in memories:
                            if "significance" in memory:
                                relevance += memory["significance"]

                        # Check for beliefs that might make this NPC more interested
                        beliefs = await memory_system.get_beliefs(
                            entity_type="npc",
                            entity_id=npc_id,
                            topic="player"
                        )
                        belief_relevance = 0
                        for belief in beliefs:
                            if any(
                                term in player_action.get("description", "").lower()
                                for term in belief.get("belief", "").lower().split()
                            ):
                                belief_relevance += belief.get("confidence", 0.5) * 2

                        relevance += belief_relevance
                        relevant_npcs.append((npc_id, relevance))

                    # Sort by relevance, highest first
                    relevant_npcs.sort(key=lambda x: x[1], reverse=True)

                    # If we have relevant NPCs, prioritize them
                    if relevant_npcs and relevant_npcs[0][1] > 0:
                        # Get up to 3 most relevant NPCs
                        return [npc_id for npc_id, _ in relevant_npcs[:3]]
                except Exception as e:
                    logger.error("Error calculating NPC relevance: %s", e)

            # Default to location-based NPCs if we can't determine relevance
            return location_npcs

        # Fallback: if the action is "talk", get the last 3 recently active NPCs
        if player_action.get("type") == "talk":
            try:
                async with self.connection_pool.acquire() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT DISTINCT npc_id
                        FROM NPCAgentState
                        WHERE user_id = $1
                          AND conversation_id = $2
                        ORDER BY last_updated DESC
                        LIMIT 3
                        """,
                        self.user_id, self.conversation_id
                    )
                    npc_ids = [row["npc_id"] for row in rows]
                    return npc_ids
            except Exception as e:
                logger.error("Error getting recently active NPCs: %s", e)

        return []


    @function_tool(strict_mode=False)
    async def batch_update_npcs(
        self,
        npc_ids: List[int],
        update_type: str,
        update_data: BatchUpdateData
    ) -> BatchUpdateResult:
        """
        Update multiple NPCs in a single batch operation.
        REFACTORED: Now uses LoreSystem for updates instead of direct DB writes.
        """
        results = BatchUpdateResult()

        try:
            # Get the lore system
            lore_system = await self._get_lore_system()
            
            if update_type == "location_change":
                if not update_data.new_location:
                    results.error = "No location specified"
                    return results
                
                # Create context for governance
                ctx = RunContextWrapper(context={
                    'user_id': self.user_id,
                    'conversation_id': self.conversation_id
                })

                
                # Use LoreSystem to update each NPC's location
                for npc_id in npc_ids:
                    try:
                        result = await lore_system.propose_and_enact_change(
                            ctx=ctx,
                            entity_type="NPCStats",
                            entity_identifier={"npc_id": npc_id},
                            updates={"current_location": update_data.new_location},
                            reason=f"Batch location update to {update_data.new_location}"
                        )
                        
                        if result.get("status") == "committed":
                            results.success_count += 1
                            results.details[npc_id] = {"success": True}
                        else:
                            results.error_count += 1
                            results.details[npc_id] = {"error": result.get("message", "Update failed")}
                    except Exception as e:
                        results.error_count += 1
                        results.details[npc_id] = {"error": str(e)}
                
            elif update_type == "emotional_update":
                # Emotional updates through memory system (which is allowed)
                if not update_data.emotion:
                    results.error = "No emotion specified"
                    return results
                    
                # Get memory system
                memory_system = await self._get_memory_system()
                
                # Process in smaller batches for better control
                batch_size = 5
                for i in range(0, len(npc_ids), batch_size):
                    batch = npc_ids[i:i+batch_size]
                    
                    # Process each NPC in batch
                    batch_tasks = []
                    for npc_id in batch:
                        task = memory_system.update_npc_emotion(
                            npc_id=npc_id,
                            emotion=update_data.emotion,
                            intensity=update_data.intensity
                        )
                        batch_tasks.append(task)
                        
                    # Execute batch concurrently
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # Process results
                    for npc_id, result in zip(batch, batch_results):
                        if isinstance(result, Exception):
                            results.error_count += 1
                            results.details[npc_id] = {"error": str(result)}
                        else:
                            results.success_count += 1
                            results.details[npc_id] = {"success": True}
                    
                    # Small delay between batches
                    if i + batch_size < len(npc_ids):
                        await asyncio.sleep(0.1)
            
            # Additional update types would be implemented here
            else:
                results.error = f"Unknown update type: {update_type}"
                results.valid_types = [
                    "location_change", "emotional_update"
                ]
            
            return results
                
        except Exception as e:
            logger.error(f"Error in batch update: {e}")
            results.error = str(e)
            results.error_count = len(npc_ids)
            return results
            
    async def _fetch_current_location(self) -> Optional[str]:
        """
        Attempt to retrieve the current location from the CurrentRoleplay table.
        READ-ONLY operation.
        """
        logger.debug("Fetching current location from CurrentRoleplay")
        try:
            # Using the provided connection pool
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT value
                    FROM CurrentRoleplay
                    WHERE user_id = $1
                      AND conversation_id = $2
                      AND key = 'CurrentLocation'
                """, self.user_id, self.conversation_id)
                return row["value"] if row else None
        except Exception as e:
            logger.error("Error getting CurrentLocation: %s", e)
            return None

    async def handle_single_npc_interaction(
        self,
        npc_id: int,
        player_action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle a player action directed at a single NPC with enhanced memory integration.
        This doesn't directly modify NPC state, only processes actions.
        """
        logger.info("Handling single NPC interaction with npc_id=%s", npc_id)

        # Make sure the NPC agent is loaded and initialized
        if npc_id not in self.npc_agents:
            self.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
            await self.npc_agents[npc_id].initialize()

        # Process the player action with the NPC agent
        response = await self.npc_agents[npc_id].process_player_action(player_action, context)

        return {"npc_responses": [response]}

    async def handle_group_npc_interaction(
        self,
        npc_ids: List[int],
        player_action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle a player action directed at multiple NPCs."""
        logger.info("Handling group NPC interaction: %s", npc_ids)
        
        # Ask Nyx to approve or modify this group interaction
        try:
            from nyx.integrate import get_central_governance
            
            governance = await get_central_governance(self.user_id, self.conversation_id)
            
            # Check permission for group interaction
            approval = await governance.check_action_permission(
                agent_type="npc_system",
                agent_id="group_coordinator",
                action_type="group_interaction",
                action_details={
                    "npc_ids": npc_ids,
                    "context": context,
                    "player_action": player_action
                }
            )
            approval = {
                "approved": approval.get("approved", True),
                "modified_context": approval.get("modified_action_details", {}).get("context", context),
                "modified_npc_ids": approval.get("modified_action_details", {}).get("npc_ids", npc_ids)
            }
            
            # Apply Nyx's modifications if any
            if approval.get("modified_context"):
                context = approval.get("modified_context")
            
            if approval.get("modified_npc_ids"):
                npc_ids = approval.get("modified_npc_ids")
                
        except Exception as e:
            logger.error(f"Error consulting Nyx for group interaction: {e}")
        
        # Initialize all affected agents
        await self.initialize_agents(npc_ids)
        
        npc_responses = []
        for npc_id in npc_ids:
            # Process with each NPC agent
            try:
                agent = self.npc_agents.get(npc_id)
                if agent:
                    response = await agent.process_player_action(player_action, context)
                    npc_responses.append(response)
            except Exception as e:
                logger.error(f"Error handling group interaction with NPC {npc_id}: {e}")
        
        return {"npc_responses": npc_responses}

    @function_tool(strict_mode=False)
    async def get_current_game_time(self) -> GameTime:
        """
        Get the current in-game time information.
        READ-ONLY operation.
        """
        year, month, day, time_of_day = None, None, None, None
        try:
            # Using the provided connection pool
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT key, value
                    FROM CurrentRoleplay
                    WHERE key IN ('CurrentYear', 'CurrentMonth', 'CurrentDay', 'TimeOfDay')
                      AND user_id = $1
                      AND conversation_id = $2
                    """,
                    self.user_id, self.conversation_id
                )
                for row in rows:
                    key = row["key"]
                    value = row["value"]
                    if key == "CurrentYear":
                        year = int(value) if value.isdigit() else 2023
                    elif key == "CurrentMonth":
                        month = value
                    elif key == "CurrentDay":
                        day = int(value) if value.isdigit() else 1
                    elif key == "TimeOfDay":
                        time_of_day = value

        except Exception as e:
            logger.error(f"Error getting game time: {e}")

        # Return with defaults
        return GameTime(
            year=year or 2023,
            month=month or "January",
            day=day or 1,
            time_of_day=time_of_day or "afternoon"
        )

    @function_tool(strict_mode=False)
    async def process_npc_scheduled_activities(self) -> ScheduledActivitiesResult:
        """
        Process scheduled activities for all NPCs using the agent system.
        This doesn't directly update NPC state but processes activities.
        """
        logger.info("Processing scheduled activities")

        try:
            # Get current game time
            time_data = await self.get_current_game_time()
            base_context = {
                "year": time_data.year,
                "month": time_data.month,
                "day": time_data.day,
                "time_of_day": time_data.time_of_day,
                "activity_type": "scheduled"
            }

            # Fetch all NPC data for activities - READ ONLY
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, schedule, current_location,
                           dominance, cruelty
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2
                """, self.user_id, self.conversation_id)
                
                npc_data = {}
                for row in rows:
                    npc_id = row["npc_id"]
                    
                    # Parse schedule if needed
                    schedule = row["schedule"]
                    if schedule and isinstance(schedule, str):
                        try:
                            import json
                            schedule = json.loads(schedule)
                        except json.JSONDecodeError:
                            schedule = {}
                    
                    npc_data[npc_id] = {
                        "name": row["npc_name"],
                        "location": row["current_location"],
                        "schedule": schedule,
                        "dominance": row["dominance"],
                        "cruelty": row["cruelty"]
                    }

            total_npcs = len(npc_data)
            if total_npcs == 0:
                return ScheduledActivitiesResult(
                    npc_responses=[], 
                    count=0,
                    time_of_day=time_data.time_of_day
                )

            logger.info(f"Processing scheduled activities for {total_npcs} NPCs")

            # Process NPCs in batches
            batch_size = 20
            npc_responses = []
            for i in range(0, total_npcs, batch_size):
                batch = list(npc_data.items())[i:i + batch_size]
                batch_tasks = []
                for npc_id, data in batch:
                    # Make sure NPC agent is initialized
                    if npc_id not in self.npc_agents:
                        self.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
                        await self.npc_agents[npc_id].initialize()
                    
                    npc_context = base_context.copy()
                    npc_context.update({
                        "npc_name": data["name"],
                        "location": data["location"],
                        "dominance": data["dominance"],
                        "cruelty": data["cruelty"]
                    })
                    
                    # Generate and execute an action
                    action = await self.npc_agents[npc_id].make_decision(npc_context)
                    batch_tasks.append((npc_id, action, npc_context))

                # Process batch results
                for npc_id, action, npc_context in batch_tasks:
                    try:
                        result = await self.npc_agents[npc_id].process_player_action(
                            {"type": "system", "description": "scheduled activity"}, 
                            npc_context
                        )
                        npc_responses.append(result)
                    except Exception as e:
                        logger.error(f"Error processing scheduled activity for NPC {npc_id}: {e}")

                # Small delay between batches
                if i + batch_size < total_npcs:
                    await asyncio.sleep(0.1)

            return ScheduledActivitiesResult(
                npc_responses=npc_responses,
                count=len(npc_responses),
                time_of_day=time_data.time_of_day
            )

        except Exception as e:
            error_msg = f"Error processing NPC scheduled activities: {e}"
            logger.error(error_msg)
            raise NPCSystemError(error_msg)


    @function_tool(strict_mode=False)
    async def run_memory_maintenance(self) -> MaintenanceResult:
        """
        Run comprehensive maintenance tasks on all NPCs' memory systems.
        This includes consolidation, decay, schema formation, and belief updates.
        Memory operations are allowed as they don't modify core game state.
        """
        results = MaintenanceResult()
        try:
            memory_system = await self._get_memory_system()

            # Player memory maintenance
            results.player_maintenance = await memory_system.maintain(
                entity_type="player",
                entity_id=self.user_id
            )

            # NPC maintenance
            npc_results = {}
            for npc_id, agent in self.npc_agents.items():
                try:
                    npc_result = await self._run_comprehensive_npc_maintenance(npc_id)
                    npc_results[npc_id] = npc_result
                except Exception as e:
                    logger.error(f"Error in memory maintenance for NPC {npc_id}: {e}")
                    npc_results[npc_id] = {"error": str(e)}
            results.npc_maintenance = npc_results

            # DM (Nyx) memory
            try:
                results.nyx_maintenance = await memory_system.maintain(entity_type="nyx", entity_id=0)
            except Exception as e:
                logger.error(f"Error in Nyx memory maintenance: {e}")
                results.nyx_maintenance = {"error": str(e)}

            # Update last maintenance time
            self._last_memory_maintenance = datetime.now()
            
            return results
        except Exception as e:
            logger.error(f"Error in system-wide memory maintenance: {e}")
            results.error = str(e)
            return results

    async def _run_comprehensive_npc_maintenance(self, npc_id: int) -> Dict[str, Any]:
        """
        Run comprehensive memory maintenance for a single NPC.
        Memory operations are allowed.
        """
        results = {}
        memory_system = await self._get_memory_system()

        # Basic memory maintenance
        try:
            basic_result = await memory_system.maintain(
                entity_type="npc",
                entity_id=npc_id
            )
            results["basic_maintenance"] = basic_result
        except Exception as e:
            logger.error(f"Error in basic maintenance for NPC {npc_id}: {e}")
            results["basic_maintenance"] = {"error": str(e)}

        # Schema detection
        try:
            schema_result = await memory_system.generate_schemas(
                entity_type="npc",
                entity_id=npc_id
            )
            results["schema_generation"] = schema_result
        except Exception as e:
            logger.error(f"Error in schema generation for NPC {npc_id}: {e}")
            results["schema_generation"] = {"error": str(e)}

        # Belief updates
        try:
            beliefs = await memory_system.get_beliefs(entity_type="npc", entity_id=npc_id)
            belief_updates = 0
            for belief in beliefs:
                belief_id = belief.get("id")
                if belief_id and random.random() < 0.3:
                    confidence_change = random.uniform(-0.05, 0.05)
                    new_confidence = max(
                        0.1,
                        min(0.95, belief.get("confidence", 0.5) + confidence_change)
                    )
                    await memory_system.update_belief_confidence(
                        entity_type="npc",
                        entity_id=npc_id,
                        belief_id=belief_id,
                        new_confidence=new_confidence,
                        reason="Regular belief reassessment"
                    )
                    belief_updates += 1
            results["belief_updates"] = belief_updates
        except Exception as e:
            logger.error(f"Error updating beliefs for NPC {npc_id}: {e}")
            results["belief_updates"] = {"error": str(e)}

        # Emotional decay
        try:
            emotional_state = await memory_system.get_npc_emotion(npc_id)
            if emotional_state and "current_emotion" in emotional_state:
                current = emotional_state["current_emotion"]
                
                # Handle different data structures
                if isinstance(current.get("primary"), dict):
                    emotion_name = current.get("primary", {}).get("name", "neutral")
                    intensity = current.get("primary", {}).get("intensity", 0.0)
                else:
                    emotion_name = current.get("primary", "neutral")
                    intensity = current.get("intensity", 0.0)
                
                if emotion_name != "neutral" and intensity > 0.3:
                    decay_amount = 0.1 if intensity > 0.7 else 0.05
                    new_intensity = max(0.1, intensity - decay_amount)
                    if new_intensity < 0.25:
                        emotion_name = "neutral"
                        new_intensity = 0.1
                    await memory_system.update_npc_emotion(
                        npc_id=npc_id,
                        emotion=emotion_name,
                        intensity=new_intensity
                    )
                    results["emotional_decay"] = {
                        "old_emotion": emotion_name,
                        "old_intensity": intensity,
                        "new_intensity": new_intensity,
                        "decayed": True
                    }
                else:
                    results["emotional_decay"] = {"decayed": False}
            else:
                results["emotional_decay"] = {"state_found": False}
        except Exception as e:
            logger.error(f"Error in emotional decay for NPC {npc_id}: {e}")
            results["emotional_decay"] = {"error": str(e)}

        # Mask evolution
        try:
            mask_info = await memory_system.get_npc_mask(npc_id)
            if mask_info:
                integrity = mask_info.get("integrity", 100)
                if 0 < integrity < 100:
                    if random.random() < 0.8:
                        new_integrity = min(100, integrity + random.uniform(0.2, 1.0))
                        results["mask_evolution"] = {
                            "old_integrity": integrity,
                            "estimated_new_integrity": new_integrity,
                            "direction": "rebuilding"
                        }
                    else:
                        await memory_system.reveal_npc_trait(
                            npc_id=npc_id,
                            trigger="introspection",
                            severity=1
                        )
                        results["mask_evolution"] = {
                            "old_integrity": integrity,
                            "direction": "weakening",
                            "slip_generated": True
                        }
                else:
                    results["mask_evolution"] = {"no_change": True}
            else:
                results["mask_evolution"] = {"mask_not_found": True}
        except Exception as e:
            logger.error(f"Error in mask evolution for NPC {npc_id}: {e}")
            results["mask_evolution"] = {"error": str(e)}

        return results

    @function_tool(strict_mode=False)
    async def generate_npc_flashback(
        self, 
        npc_id: int, 
        context_text: str
    ) -> FlashbackResult:
        """
        Generate a flashback for an NPC based on specific context.
        This is a memory operation which is allowed.
        """
        try:
            # Check if we've recently triggered a flashback for this NPC
            last_flashback_time = self._last_flashback_times.get(
                npc_id, 
                datetime.now() - timedelta(hours=1)
            )
            time_since_last = (datetime.now() - last_flashback_time).total_seconds()
            if time_since_last < 300:  # 5 minutes minimum between flashbacks
                return FlashbackResult(
                    triggered=False,
                    reason="too_soon",
                    time_since_last_seconds=time_since_last
                )

            memory_system = await self._get_memory_system()
            flashback = await memory_system.npc_flashback(npc_id, context_text)
            
            if flashback:
                self._last_flashback_times[npc_id] = datetime.now()

                # Update emotional state based on flashback content
                emotional_state = await memory_system.get_npc_emotion(npc_id)
                if emotional_state:
                    # Determine appropriate emotion from flashback content
                    emotion = "fear"  # Default
                    intensity = 0.7
                    text_lower = flashback.get("text", "").lower()
                    
                    if "happy" in text_lower or "joy" in text_lower or "good" in text_lower:
                        emotion = "joy"
                    elif "anger" in text_lower or "angr" in text_lower or "rage" in text_lower:
                        emotion = "anger"
                    elif "sad" in text_lower or "sorrow" in text_lower:
                        emotion = "sadness"

                    await memory_system.update_npc_emotion(
                        npc_id=npc_id,
                        emotion=emotion,
                        intensity=intensity
                    )
                    
                    return FlashbackResult(
                        triggered=True,
                        text=flashback.get("text"),
                        memory_id=flashback.get("memory_id"),
                        triggered_emotion=emotion,
                        emotion_intensity=intensity
                    )
                else:
                    return FlashbackResult(
                        triggered=True,
                        text=flashback.get("text"),
                        memory_id=flashback.get("memory_id")
                    )
            else:
                return FlashbackResult(
                    triggered=False,
                    reason="no_relevant_memory"
                )
        except Exception as e:
            logger.error(f"Error generating flashback for NPC {npc_id}: {e}")
            return FlashbackResult(
                triggered=False,
                error=str(e)
            )

    async def process_command(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a command from the game system using the system agent.
        """
        # Get the system agent
        system_agent = await self._get_system_agent()
        
        # Convert parameters to appropriate models if needed
        if command == "process_player_action" and "player_action" in parameters:
            # Convert player_action dict to model
            player_action_dict = parameters["player_action"]
            player_action = PlayerAction(**player_action_dict)
            
            # Convert context dict to model if present
            context = None
            if "context" in parameters:
                context = ActionContext(**parameters["context"])
            
            # Call the method directly with models
            result = await self.handle_player_action(player_action, context)
            return result.model_dump()
            
        elif command == "batch_update":
            # Convert update_data to model
            update_data = BatchUpdateData(**parameters.get("update_data", {}))
            result = await self.batch_update_npcs(
                parameters["npc_ids"],
                parameters["update_type"],
                update_data
            )
            return result.model_dump()
        
        # For other commands, use the existing approach
        input_data = {
            "command": command,
            "parameters": parameters
        }
        
        # Create trace for debugging
        with trace(
            f"system_command_{self.user_id}_{self.conversation_id}", 
            group_id=f"user_{self.user_id}_conv_{self.conversation_id}"
        ):
            # Run the system agent
            result = await Runner.run(system_agent, input_data)
            
            # Return the result
            return result.final_output.result
            
    async def get_npc_name(self, npc_id: int) -> str:
        """
        Get the name of an NPC by ID.
        READ-ONLY operation.
        """
        try:
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT npc_name
                    FROM NPCStats
                    WHERE npc_id = $1
                      AND user_id = $2
                      AND conversation_id = $3
                """, npc_id, self.user_id, self.conversation_id)
                
                if row:
                    return row["npc_name"]
                return f"NPC_{npc_id}"
        except Exception as e:
            logger.error(f"Error getting NPC name: {e}")
            return f"NPC_{npc_id}"
            
    async def get_all_npc_beliefs_about_player(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get all NPCs' beliefs about the player.
        READ-ONLY operation through memory system.
        """
        results = {}
        memory_system = await self._get_memory_system()
        for npc_id in self.npc_agents:
            try:
                beliefs = await memory_system.get_beliefs(
                    entity_type="npc",
                    entity_id=npc_id,
                    topic="player"
                )
                if beliefs:
                    results[npc_id] = beliefs
            except Exception as e:
                logger.error(f"Error getting beliefs for NPC {npc_id}: {e}")
        return results

    async def get_player_beliefs_about_npcs(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get player's beliefs about each NPC.
        READ-ONLY operation through memory system.
        """
        results = {}
        memory_system = await self._get_memory_system()
        for npc_id in self.npc_agents:
            try:
                topic = f"npc_{npc_id}"
                beliefs = await memory_system.get_beliefs(
                    entity_type="player",
                    entity_id=self.user_id,
                    topic=topic
                )
                if beliefs:
                    results[npc_id] = beliefs
            except Exception as e:
                logger.error(f"Error getting player beliefs about NPC {npc_id}: {e}")
        return results

    async def report_npc_action_to_nyx(self, npc_id: int, action: Dict[str, Any]) -> None:
        """Report significant NPC actions back to Nyx for narrative coordination."""
        # This creates a memory, which is allowed
        memory_system = await self._get_memory_system()
        
        await memory_system.remember(
            entity_type="nyx",
            entity_id=0,
            memory_text=f"NPC {npc_id} performed action: {action.get('description', 'unknown action')}",
            importance="medium",
            tags=["npc_action", f"npc_{npc_id}"],
            metadata={
                "npc_id": npc_id,
                "action": action
            }
        )

    # ================== RESTORED HELPER METHODS ==================

    async def _get_current_time(self) -> Optional[Dict[str, Any]]:
        """Get current game time - READ ONLY."""
        return await self.get_current_game_time()

    async def _get_active_npcs(self) -> List[Dict[str, Any]]:
        """Get all active NPCs in the current conversation - READ ONLY."""
        try:
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location, schedule
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                    AND is_active = true
                """, self.user_id, self.conversation_id)
                
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting active NPCs: {e}")
            return []

    async def _get_npc_agent(self, npc_id: int) -> Optional[NPCAgent]:
        """Get or create an NPC agent."""
        if npc_id not in self.npc_agents:
            self.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
            await self.npc_agents[npc_id].initialize()
        return self.npc_agents.get(npc_id)

    async def _is_npc_in_conflict(self, npc_id: int) -> bool:
        """Check if an NPC is currently involved in a conflict - READ ONLY."""
        try:
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT COUNT(*) as count
                    FROM ConflictNPCs
                    WHERE npc_id = $1
                    AND conflict_id IN (
                        SELECT conflict_id 
                        FROM Conflicts 
                        WHERE user_id = $2 
                        AND conversation_id = $3 
                        AND is_active = true
                    )
                """, npc_id, self.user_id, self.conversation_id)
                
                return row["count"] > 0
        except Exception as e:
            logger.error(f"Error checking NPC conflict status: {e}")
            return False

    async def _should_perform_activity(self, npc_agent: NPCAgent, current_time: Dict[str, Any]) -> bool:
        """Check if an NPC should perform an activity based on their schedule and current time."""
        try:
            # Get NPC's schedule
            schedule = await npc_agent._get_current_schedule()
            if not schedule:
                return False
                
            # Get time of day
            time_of_day = current_time.get("time_of_day")
            if not time_of_day:
                return False
                
            # Check if NPC has scheduled activity for current time
            scheduled_activity = schedule.get(time_of_day)
            if not scheduled_activity:
                return False
                
            # Check if NPC has already performed activity
            last_activity = await self._get_last_activity(npc_agent.npc_id)
            if last_activity:
                last_activity_time = last_activity.get("time")
                if last_activity_time and self._is_same_time_period(last_activity_time, current_time):
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking activity schedule: {e}")
            return False

    async def _get_last_activity(self, npc_id: int) -> Optional[Dict[str, Any]]:
        """Get the last recorded activity for an NPC - READ ONLY."""
        # This would check activity logs/memories
        memory_system = await self._get_memory_system()
        memories = await memory_system.recall(
            entity_type="npc",
            entity_id=npc_id,
            query="performed activity",
            limit=1,
            recency_days=1
        )
        
        if memories and memories.get("memories"):
            latest_memory = memories["memories"][0]
            return {
                "data": latest_memory,
                "time": latest_memory.get("timestamp")
            }
        return None

    def _is_same_time_period(self, time1: Any, time2: Dict[str, Any]) -> bool:
        """Check if two times are in the same time period."""
        try:
            # Simple check - are they on the same day and time period
            if isinstance(time1, str):
                # Parse timestamp if it's a string
                time1 = datetime.fromisoformat(time1.replace('Z', '+00:00'))
            
            # Check if same day
            return (time1.date() == datetime.now().date() and 
                    time2.get("time_of_day") == "current")
            
        except Exception as e:
            logger.error(f"Error comparing time periods: {e}")
            return False

    async def _process_activity_result(self, npc_agent: NPCAgent, result: Dict[str, Any], lore_system) -> None:
        """Process the result of an NPC's activity using LoreSystem for updates."""
        try:
            # Record activity in memory (allowed)
            await self._record_activity(npc_agent.npc_id, result)
            
            # Check if stats need updating
            if result.get("result", {}).get("status") == "success":
                stat_changes = result.get("stat_changes", {})
                if stat_changes:
                    # Use LoreSystem to update stats
                    ctx = RunContextWrapper(context={
                        'user_id': self.user_id,
                        'conversation_id': self.conversation_id
                    })
                    
                    await lore_system.propose_and_enact_change(
                        ctx=ctx,
                        entity_type="NPCStats",
                        entity_identifier={"npc_id": npc_agent.npc_id},
                        updates=stat_changes,
                        reason=f"Activity result: {result.get('action', {}).get('description', 'unknown')}"
                    )
                
            # Check for new conflicts
            await self._check_for_conflicts(npc_agent.npc_id, result)
            
            # Update relationships if social activity
            if result.get("action", {}).get("type") == "socialize":
                await self._update_relationships(npc_agent.npc_id, result, lore_system)
                
        except Exception as e:
            logger.error(f"Error processing activity result: {e}")

    async def _record_activity(self, npc_id: int, result: Dict[str, Any]) -> None:
        """Record an NPC's activity in memory system."""
        memory_system = await self._get_memory_system()
        
        action = result.get("action", {})
        description = action.get("description", "performed an activity")
        
        await memory_system.remember(
            entity_type="npc",
            entity_id=npc_id,
            memory_text=f"I {description}",
            importance="low",
            tags=["activity", action.get("type", "unknown")],
            metadata=result
        )

    async def _check_for_conflicts(self, npc_id: int, result: Dict[str, Any]) -> None:
        """Check if an activity might lead to a conflict."""
        # This would analyze the result and potentially create conflict memories
        # but not directly create conflicts in the database
        action = result.get("action", {})
        if action.get("type") == "socialize" and result.get("tension_level", 0) > 0.7:
            memory_system = await self._get_memory_system()
            target_npc = result.get("result", {}).get("target_npc")
            
            if target_npc:
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=npc_id,
                    memory_text=f"I had a tense interaction with {target_npc.get('name', 'someone')}",
                    importance="medium",
                    tags=["conflict_potential", "social"],
                    emotional=True
                )

    async def _update_relationships(self, npc_id: int, result: Dict[str, Any], lore_system) -> None:
        """Update relationships based on social activity result using LoreSystem."""
        try:
            # Get interaction details
            interaction = result.get("result", {}).get("interaction", {})
            target_npc = result.get("result", {}).get("target_npc")
            
            if not target_npc:
                return
                
            # Calculate relationship changes
            changes = self._calculate_relationship_changes(interaction)
            
            if changes:
                # Use LoreSystem to update the relationship
                ctx = RunContextWrapper(context={
                    'user_id': self.user_id,
                    'conversation_id': self.conversation_id
                })
                
                # Find existing social link
                async with self.connection_pool.acquire() as conn:
                    link = await conn.fetchrow("""
                        SELECT link_id FROM SocialLinks
                        WHERE user_id = $1 AND conversation_id = $2
                        AND entity1_type = 'npc' AND entity1_id = $3
                        AND entity2_type = 'npc' AND entity2_id = $4
                    """, self.user_id, self.conversation_id, npc_id, target_npc["npc_id"])
                
                if link:
                    # Update existing link
                    await lore_system.propose_and_enact_change(
                        ctx=ctx,
                        entity_type="SocialLinks",
                        entity_identifier={"link_id": link["link_id"]},
                        updates=changes,
                        reason=f"Social interaction: {interaction.get('type', 'unknown')}"
                    )
                else:
                    # Create new link through canon
                    async with get_db_connection_context() as conn:
                        await canon.establish_relationship(
                            ctx, conn,
                            entity1_type="npc",
                            entity1_id=npc_id,
                            entity2_type="npc", 
                            entity2_id=target_npc["npc_id"],
                            link_type=changes.get("link_type", "neutral"),
                            link_level=changes.get("link_level", 50)
                        )
            
        except Exception as e:
            logger.error(f"Error updating relationships: {e}")

    def _calculate_relationship_changes(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate relationship changes based on interaction."""
        try:
            changes = {}
            
            # Get interaction type and details
            interaction_type = interaction.get("type")
            details = interaction.get("details", {})
            
            # Calculate changes based on interaction type
            link_level_change = 0
            if interaction_type == "friendly":
                link_level_change = 5
            elif interaction_type == "hostile":
                link_level_change = -10
            elif interaction_type == "professional":
                link_level_change = 2
                
            # Adjust based on interaction details
            if details.get("success"):
                link_level_change = int(link_level_change * 1.2)
            else:
                link_level_change = int(link_level_change * 0.8)
            
            if link_level_change != 0:
                changes["link_level"] = link_level_change
                        
            return changes
            
        except Exception as e:
            logger.error(f"Error calculating relationship changes: {e}")
            return {}

    async def _get_npc_active_conflict(self, npc_id: int) -> Optional[Dict[str, Any]]:
        """Get the active conflict an NPC is involved in - READ ONLY."""
        try:
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT c.*, cn.faction, cn.role, cn.influence_level
                    FROM Conflicts c
                    JOIN ConflictNPCs cn ON c.conflict_id = cn.conflict_id
                    WHERE cn.npc_id = $1
                    AND c.user_id = $2 
                    AND c.conversation_id = $3
                    AND c.is_active = true
                    ORDER BY c.created_at DESC
                    LIMIT 1
                """, npc_id, self.user_id, self.conversation_id)
                
                if row:
                    conflict_data = dict(row)
                    # Add conflict goals and strategies
                    conflict_data["goals"] = await self._get_conflict_goals(npc_id, conflict_data)
                    conflict_data["strategies"] = await self._get_conflict_strategies(npc_id, conflict_data)
                    return conflict_data
                return None
        except Exception as e:
            logger.error(f"Error getting NPC conflict: {e}")
            return None

    async def _get_conflict_goals(self, npc_id: int, conflict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get the NPC's goals for resolving the conflict - READ ONLY."""
        # This would analyze NPC personality and generate goals
        # For now, return basic goals based on NPC traits
        return [
            {
                "type": "resolve",
                "description": "Find a peaceful resolution",
                "priority": 0.5,
                "success_conditions": ["conflict_resolved", "relationship_improved"]
            }
        ]

    async def _get_conflict_strategies(self, npc_id: int, conflict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get the NPC's strategies for achieving conflict goals - READ ONLY."""
        # This would analyze NPC personality and generate strategies
        return [
            {
                "type": "negotiation",
                "description": "Attempt to negotiate a resolution",
                "risk_level": "medium",
                "potential_reward": "medium",
                "conditions": ["neutral_location", "both_npcs_calm"]
            }
        ]

    async def _handle_conflict_activity(self, npc_agent: NPCAgent, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an NPC's activity while they are in a conflict."""
        try:
            # Get current goals and strategies
            goals = conflict.get("goals", [])
            strategies = conflict.get("strategies", [])
            
            if not goals or not strategies:
                return None
            
            # Select best goal and strategy based on current situation
            selected_goal = self._select_best_goal(goals, conflict)
            selected_strategy = self._select_best_strategy(strategies, conflict)
            
            # Create activity context
            activity_context = {
                "conflict_id": conflict["conflict_id"],
                "goal": selected_goal,
                "strategy": selected_strategy,
                "conflict_data": conflict
            }
            
            # Generate and execute conflict resolution action
            action = await npc_agent.make_decision(activity_context)
            result = await npc_agent.process_player_action(
                {"type": "conflict_resolution", "description": action.description},
                activity_context
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling conflict activity: {e}")
            return None

    def _select_best_goal(self, goals: List[Dict[str, Any]], conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best goal based on current conflict state."""
        # Simple selection - return first goal
        return goals[0] if goals else None

    def _select_best_strategy(self, strategies: List[Dict[str, Any]], conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best strategy based on current conflict state."""
        # Simple selection - return first strategy
        return strategies[0] if strategies else None

    # ================== ADDITIONAL HELPER METHODS FOR RELATIONSHIPS/STATS ==================

    async def _get_npc_stats(self, npc_id: int) -> Optional[Dict[str, Any]]:
        """Get NPC stats - READ ONLY."""
        try:
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM NPCStats
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                """, npc_id, self.user_id, self.conversation_id)
                
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting NPC stats: {e}")
            return None

    async def _get_npc_traits(self, npc_id: int) -> Dict[str, Any]:
        """Get NPC personality traits - READ ONLY."""
        try:
            stats = await self._get_npc_stats(npc_id)
            if stats:
                return {
                    "dominance": stats.get("dominance", 50),
                    "cruelty": stats.get("cruelty", 50),
                    "confidence": stats.get("confidence", 50),
                    "loyalty": stats.get("loyalty", 50)
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting NPC traits: {e}")
            return {}

    async def _get_relationship(self, npc1_id: int, npc2_id: int) -> Dict[str, Any]:
        """Get relationship data between two NPCs - READ ONLY."""
        try:
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM SocialLinks
                    WHERE user_id = $1 AND conversation_id = $2
                    AND ((entity1_type = 'npc' AND entity1_id = $3 
                          AND entity2_type = 'npc' AND entity2_id = $4)
                    OR (entity1_type = 'npc' AND entity1_id = $4 
                        AND entity2_type = 'npc' AND entity2_id = $3))
                """, self.user_id, self.conversation_id, npc1_id, npc2_id)
                
                if row:
                    return {
                        "trust": row.get("trust_level", 50),
                        "respect": row.get("respect_level", 50),
                        "affection": row.get("affection_level", 50),
                        "tension": row.get("tension_level", 0),
                        "link_level": row.get("link_level", 50)
                    }
                return {
                    "trust": 50,
                    "respect": 50,
                    "affection": 50,
                    "tension": 0,
                    "link_level": 50
                }
        except Exception as e:
            logger.error(f"Error getting relationship: {e}")
            return {}

    async def _check_relationship_tension(self, npc1_id: int, npc2_id: int) -> float:
        """Check tension level between two NPCs - READ ONLY."""
        try:
            relationship = await self._get_relationship(npc1_id, npc2_id)
            return relationship.get("tension", 0) / 100.0
        except Exception as e:
            logger.error(f"Error checking relationship tension: {e}")
            return 0.0

    async def _get_workplace_rivals(self, npc_id: int, location: str) -> List[Dict[str, Any]]:
        """Get NPCs who are rivals at the same workplace - READ ONLY."""
        try:
            async with self.connection_pool.acquire() as conn:
                # Get NPCs at same location
                rows = await conn.fetch("""
                    SELECT n.npc_id, n.npc_name
                    FROM NPCStats n
                    WHERE n.user_id = $1 AND n.conversation_id = $2
                    AND n.current_location = $3 AND n.npc_id != $4
                """, self.user_id, self.conversation_id, location, npc_id)
                
                rivals = []
                for row in rows:
                    # Check relationship for rivalry
                    relationship = await self._get_relationship(npc_id, row["npc_id"])
                    if relationship.get("tension", 0) > 50 or relationship.get("trust", 50) < 30:
                        rivals.append(dict(row))
                
                return rivals
        except Exception as e:
            logger.error(f"Error getting workplace rivals: {e}")
            return []

    def _calculate_stat_changes(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate stat changes based on activity result."""
        try:
            changes = {}
            
            # Example stat change logic
            if result.get("status") == "success":
                activity_type = result.get("action", {}).get("type")
                if activity_type == "work":
                    changes["experience"] = 1
                    changes["fatigue"] = 5
                elif activity_type == "socialize":
                    changes["social_satisfaction"] = 10
                    changes["mood"] = 5
            
            return changes
        except Exception as e:
            logger.error(f"Error calculating stat changes: {e}")
            return {}

    async def _update_conflict_data(self, conflict_id: int, data: Dict[str, Any]) -> None:
        """Update conflict data - NOW USES LORE SYSTEM."""
        try:
            lore_system = await self._get_lore_system()
            ctx = RunContextWrapper(context={
                'user_id': self.user_id,
                'conversation_id': self.conversation_id
            })
            
            await lore_system.propose_and_enact_change(
                ctx=ctx,
                entity_type="Conflicts",
                entity_identifier={"conflict_id": conflict_id},
                updates={"conflict_data": data},
                reason="Updating conflict progress"
            )
        except Exception as e:
            logger.error(f"Error updating conflict data: {e}")

    # ================== NEW METHODS FOR EXTENDED FUNCTIONALITY ==================

    async def process_scheduled_activities(self) -> None:
        """
        Process scheduled activities for all NPCs.
        This uses LoreSystem for any necessary updates.
        """
        try:
            # Get current time
            current_time = await self._get_current_time()
            if not current_time:
                return
                
            # Get all active NPCs - READ ONLY
            active_npcs = await self._get_active_npcs()
            
            # Get the lore system for any updates
            lore_system = await self._get_lore_system()
            
            # Process each NPC's activities
            for npc in active_npcs:
                try:
                    # Get NPC agent
                    npc_agent = await self._get_npc_agent(npc["npc_id"])
                    if not npc_agent:
                        continue
                        
                    # Check if NPC is in conflict
                    conflict = await self._get_npc_active_conflict(npc["npc_id"])
                    if conflict:
                        # Handle conflict-based activity
                        result = await self._handle_conflict_activity(npc_agent, conflict)
                    else:
                        # Handle normal scheduled activity
                        if await self._should_perform_activity(npc_agent, current_time):
                            result = await npc_agent.perform_scheduled_activity()
                        else:
                            continue
                    
                    # Process activity result
                    if result:
                        await self._process_activity_result(npc_agent, result, lore_system)
                        
                except Exception as e:
                    logger.error(f"Error processing activities for NPC {npc['npc_id']}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing scheduled activities: {e}")

    async def handle_lore_change(self, lore_change: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a lore change across all NPCs."""
        # Get all affected NPCs
        affected_npcs = await self._get_affected_npcs(lore_change)
        
        # Process change for each affected NPC
        results = []
        for npc_id in affected_npcs:
            if npc_id in self.npc_agents:
                result = await self.npc_agents[npc_id]._process_lore_change(lore_change)
                results.append(result)
                
        return {
            "lore_change": lore_change,
            "affected_npcs": affected_npcs,
            "results": results
        }
        
    async def _get_affected_npcs(self, lore_change: Dict[str, Any]) -> List[int]:
        """Get all NPCs affected by a lore change."""
        affected_npcs = set()
        
        # Check each NPC's context
        for npc_id, agent in self.npc_agents.items():
            # Get NPC's lore context
            context = await self.lore_context_manager.get_lore_context(
                npc_id,
                "change_impact"
            )
            
            # Check if NPC is affected
            if self._is_npc_affected(context, lore_change):
                affected_npcs.add(npc_id)
                
        return list(affected_npcs)
        
    def _is_npc_affected(self, context: Dict[str, Any], lore_change: Dict[str, Any]) -> bool:
        """Determine if an NPC is affected by a lore change."""
        # Check direct impact
        if lore_change.get("npc_id") == context.get("npc_id"):
            return True
            
        # Check relationship impact
        if lore_change.get("relationship_impact"):
            for relationship in context.get("relationships", []):
                if relationship.get("npc_id") == lore_change.get("npc_id"):
                    return True
                    
        # Check knowledge impact
        if lore_change.get("knowledge_impact"):
            for knowledge in context.get("knowledge", []):
                if knowledge.get("topic") == lore_change.get("topic"):
                    return True
                    
        return False
        
    async def get_npc_lore_context(self, npc_id: int, context_type: str) -> Dict[str, Any]:
        """Get lore context for an NPC."""
        return await self.lore_context_manager.get_lore_context(npc_id, context_type)
        
    async def propagate_lore_change(self, lore_change: Dict[str, Any], 
                                  source_npc_id: int, 
                                  target_npcs: List[int]) -> Dict[str, Any]:
        """Propagate a lore change through the NPC network."""
        return await self.lore_context_manager.propagation_system.propagate_lore_change(
            lore_change,
            source_npc_id,
            target_npcs
        )
