# logic/npc_agents/agent_system.py

"""
Main system that integrates NPC agents with the game loop, using OpenAI Agents SDK.
"""

import logging
import asyncio
import random
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from pydantic import BaseModel

import asyncpg
from agents import Agent, Runner, function_tool, handoff, trace, ModelSettings, TraceProvider, input_guardrail, GuardrailFunctionOutput

from .npc_agent import NPCAgent
from .agent_coordinator import NPCAgentCoordinator
from memory.wrapper import MemorySystem

logger = logging.getLogger(__name__)

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
    is_appropriate: bool
    reasoning: str

class NPCAgentSystem:
    """
    Main system that integrates individual NPC agents with the game loop using the OpenAI Agents SDK.

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
        connection_pool: asyncpg.Pool
    ):
        """
        Initialize the agent system for a specific user & conversation.

        Args:
            user_id: The ID of the user/player
            conversation_id: The ID of the current conversation/scene
            connection_pool: An asyncpg.Pool for database access
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.coordinator = NPCAgentCoordinator(user_id, conversation_id)
        self.npc_agents: Dict[int, NPCAgent] = {}
        self._memory_system = None
        self.connection_pool = connection_pool  # Store asyncpg connection pool
        self._system_agent = None

        # Track when memory maintenance was last run
        self._last_memory_maintenance = datetime.now() - timedelta(hours=1)  # Run on first init
        # Track NPC emotional states to detect significant changes
        self._npc_emotional_states = {}
        # Track last flashback times to prevent too-frequent occurrences
        self._last_flashback_times = {}

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
                model="gpt-4o",
                model_settings=ModelSettings(temperature=0.2),
                tools=[
                    function_tool(self.handle_player_action),
                    function_tool(self.process_npc_scheduled_activities),
                    function_tool(self.run_memory_maintenance),
                    function_tool(self.initialize_agents),
                    function_tool(self.batch_update_npcs),
                    function_tool(self.generate_npc_flashback)
                ],
                input_guardrails=[input_guardrail(moderation_guardrail)],
                output_type=SystemOutput
            )
        
        return self._system_agent

    @function_tool
    async def initialize_agents(self, npc_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Initialize NPCAgent objects for specified NPCs or all NPCs in the conversation.
        
        Args:
            npc_ids: Optional list of specific NPC IDs to initialize
            
        Returns:
            Dictionary with initialization results
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

        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
            # Create tasks to initialize agents concurrently
            init_tasks = []
            for row in rows:
                npc_id = row["npc_id"]
                if npc_id not in self.npc_agents:
                    self.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
                    init_tasks.append(self.npc_agents[npc_id].initialize())
                else:
                    # Already initialized
                    pass
            
            # Wait for all initializations to complete
            if init_tasks:
                await asyncio.gather(*init_tasks)

        npc_count = len(self.npc_agents)
        logger.info("Loaded %d NPC agents", npc_count)
        
        return {
            "npc_count": npc_count,
            "initialized_ids": list(self.npc_agents.keys())
        }

    @function_tool
    async def handle_player_action(
        self,
        player_action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle a player action and determine NPC responses with memory integration.

        Depending on how many NPCs are affected (one vs many), this
        method will delegate to the appropriate single/group logic.

        Args:
            player_action: A dictionary describing what the player is doing or saying
            context: Optional additional context (like current location/time)

        Returns:
            A dictionary with NPC responses
        """
        if context is None:
            context = {}

        # Create trace for debugging and monitoring
        with trace(
            f"player_action_{self.user_id}_{self.conversation_id}", 
            group_id=f"user_{self.user_id}_conv_{self.conversation_id}"
        ):
            # Get memory system for creating player action memories
            memory_system = await self._get_memory_system()
    
            # Create a memory of this action from the player's perspective
            player_memory_text = f"I {player_action.get('description', 'did something')}"
    
            # Determine emotional content based on action type
            action_type = player_action.get("type", "unknown")
            is_emotional = action_type in ["express_emotion", "shout", "cry", "laugh", "threaten"]
    
            # Add memory with appropriate tags
            await memory_system.remember(
                entity_type="player",
                entity_id=self.user_id,
                memory_text=player_memory_text,
                importance="medium",
                emotional=is_emotional,
                tags=["player_action", action_type]
            )
    
            # Determine which NPCs are affected by the action
            affected_npcs = await self.determine_affected_npcs(player_action, context)
            if not affected_npcs:
                logger.debug("No NPCs were affected by this action: %s", player_action)
                return {"npc_responses": []}
    
            # Single NPC path
            if len(affected_npcs) == 1:
                npc_id = affected_npcs[0]
                return await self.handle_single_npc_interaction(npc_id, player_action, context)
    
            # Multiple NPCs => group logic
            return await self.handle_group_npc_interaction(affected_npcs, player_action, context)

    @function_tool
    async def determine_affected_npcs(
        self,
        player_action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[int]:
        """
        Figure out which NPCs are affected by a given player action with improved memory awareness.

        Prioritizes:
        1. A 'target_npc_id' in the player_action
        2. NPCs in the specified location
        3. If action is "talk", fallback to the last 3 recently active NPCs
        4. NPCs with high relevance to the context based on memories

        Returns:
            A list of NPC IDs that are relevant to this action.
        """
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

    @function_tool
    async def batch_update_npcs(
        self,
        npc_ids: List[int],
        update_type: str,
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update multiple NPCs in a single batch operation for better performance.
        
        Args:
            npc_ids: List of NPC IDs to update
            update_type: Type of update (location_change, emotional_update, etc.)
            update_data: Data for the update
            
        Returns:
            Results of the batch update
        """
        results = {
            "success_count": 0,
            "error_count": 0,
            "details": {}
        }

        # Delegate to coordinator for batch updates
        return await self.coordinator.batch_update_npcs(npc_ids, update_type, update_data)

    async def _fetch_current_location(self) -> Optional[str]:
        """
        Attempt to retrieve the current location from the CurrentRoleplay table.

        Returns:
            The current location string, or None if not found or on error.
        """
        logger.debug("Fetching current location from CurrentRoleplay")
        query = """
            SELECT value
            FROM CurrentRoleplay
            WHERE user_id = $1
              AND conversation_id = $2
              AND key = 'CurrentLocation'
        """
        try:
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow(query, self.user_id, self.conversation_id)
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

        Args:
            npc_id: The ID of the targeted NPC
            player_action: The player's action
            context: Additional context

        Returns:
            A dictionary with the NPC's response in npc_responses
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
        """
        Handle a player action directed at multiple NPCs, delegating to the coordinator.

        Args:
            npc_ids: List of NPC IDs that are all affected
            player_action: The player's action
            context: Additional context

        Returns:
            A dictionary possibly containing "npc_responses"
        """
        logger.info("Handling group NPC interaction: %s", npc_ids)

        # Delegate to the coordinator
        return await self.coordinator.handle_player_action(player_action, context, npc_ids)

    @function_tool
    async def get_current_game_time(self) -> Dict[str, Any]:
        """
        Get the current in-game time information.

        Returns:
            Dictionary with year, month, day, time_of_day
        """
        year, month, day, time_of_day = None, None, None, None
        try:
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
                        year = int(value) if value.isdigit() else value
                    elif key == "CurrentMonth":
                        month = value
                    elif key == "CurrentDay":
                        day = int(value) if value.isdigit() else value
                    elif key == "TimeOfDay":
                        time_of_day = value

                # Set defaults if values are missing
                if year is None:
                    year = 2023
                if month is None:
                    month = "January"
                if day is None:
                    day = 1
                if time_of_day is None:
                    time_of_day = "afternoon"

        except Exception as e:
            logger.error(f"Error getting game time: {e}")
            # Return defaults if query fails
            return {"year": 2023, "month": "January", "day": 1, "time_of_day": "afternoon"}

        return {"year": year, "month": month, "day": day, "time_of_day": time_of_day}

    @function_tool
    async def process_npc_scheduled_activities(self) -> Dict[str, Any]:
        """
        Process scheduled activities for all NPCs using the agent system.
        Optimized for better performance with many NPCs through batching.
        
        Returns:
            Results of the scheduled activities
        """
        logger.info("Processing scheduled activities")

        try:
            # Get current game time
            time_data = await self.get_current_game_time()
            base_context = {
                "year": time_data["year"],
                "month": time_data["month"],
                "day": time_data["day"],
                "time_of_day": time_data["time_of_day"],
                "activity_type": "scheduled"
            }

            # Fetch all NPC data for activities
            npc_data = await self._fetch_all_npc_data_for_activities()
            total_npcs = len(npc_data)
            if total_npcs == 0:
                return {"npc_responses": [], "count": 0}

            logger.info(f"Processing scheduled activities for {total_npcs} NPCs")

            # Process NPCs in batches
            batch_size = 20
            npc_responses = []
            for i in range(0, total_npcs, batch_size):
                batch = list(npc_data.items())[i:i + batch_size]
                batch_tasks = []
                for npc_id, data in batch:
                    batch_tasks.append(self._process_single_npc_activity(npc_id, data, base_context))

                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Error processing scheduled activity: {result}")
                    elif result:
                        npc_responses.append(result)

                # Small delay between batches to prevent resource contention
                if i + batch_size < total_npcs:
                    await asyncio.sleep(0.1)

            # Process group interactions
            group_responses = await self._process_group_activities(base_context)

            return {
                "npc_responses": npc_responses,
                "group_responses": group_responses,
                "count": len(npc_responses) + len(group_responses),
                "time_of_day": time_data["time_of_day"]
            }

        except Exception as e:
            error_msg = f"Error processing NPC scheduled activities: {e}"
            logger.error(error_msg)
            raise NPCSystemError(error_msg)

    async def _process_single_npc_activity(
        self,
        npc_id: int,
        data: Dict[str, Any],
        base_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a scheduled activity for a single NPC.

        Args:
            npc_id: ID of the NPC
            data: NPC data including location, schedule, etc.
            base_context: Base context for the activity

        Returns:
            Activity result or None if processing failed
        """
        try:
            # Make sure the NPC agent is loaded and initialized
            if npc_id not in self.npc_agents:
                self.npc_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
                await self.npc_agents[npc_id].initialize()
            
            agent = self.npc_agents[npc_id]

            # Create NPC-specific context
            npc_context = base_context.copy()
            npc_context.update({
                "npc_name": data.get("name"),
                "location": data.get("location"),
                "dominance": data.get("dominance", 50),
                "cruelty": data.get("cruelty", 50)
            })

            # Get schedule entry for current time if available
            schedule = data.get("schedule", {})
            time_of_day = base_context.get("time_of_day", "afternoon")
            current_schedule = None

            if schedule and isinstance(schedule, dict):
                current_schedule = schedule.get(time_of_day)
                if not current_schedule and "default" in schedule:
                    current_schedule = schedule.get("default")

            # Create perception of environment
            perception = await agent.perceive_environment(npc_context)

            # Determine available actions based on schedule
            available_actions = []
            if current_schedule:
                activity_type = current_schedule.get("activity", "idle")
                description = current_schedule.get("description", f"perform {activity_type} activity")
                available_actions.append({
                    "type": "scheduled",
                    "description": description,
                    "target": "environment",
                    "weight": 2.0  # Prioritize scheduled activities
                })

            # Always add some default actions
            default_actions = [
                {
                    "type": "idle",
                    "description": "spend time in current location",
                    "target": "environment"
                },
                {
                    "type": "observe",
                    "description": "observe surroundings",
                    "target": "environment"
                }
            ]
            available_actions.extend(default_actions)

            # Make a decision using the agent
            action = await agent.make_decision(perception, available_actions)

            # Execute the action
            result = await agent.execute_action(action, npc_context)

            # Determine significance for memory formation
            significance = self._determine_activity_significance(action, result)
            if significance >= 2:
                memory_system = await self._get_memory_system()
                memory_text = f"{data.get('name')} {action.description} at {data.get('location', 'somewhere')}"

                await memory_system.remember(
                    entity_type="player",
                    entity_id=self.user_id,
                    memory_text=memory_text,
                    importance="low" if significance < 3 else "medium",
                    tags=["npc_activity", action.type]
                )

            return {
                "npc_id": npc_id,
                "npc_name": data.get("name"),
                "location": data.get("location"),
                "action": action,
                "result": result,
                "significance": significance
            }
        except Exception as e:
            logger.error(f"Error processing activity for NPC {npc_id}: {e}")
            return None

    async def _process_group_activities(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process group interactions between NPCs.
        
        Args:
            context: Shared context for group activities
            
        Returns:
            List of group activity results
        """
        # Group NPCs by location
        location_groups = await self._get_npcs_by_location()
        group_results = []
        
        for location, npc_ids in location_groups.items():
            # Only process locations with multiple NPCs
            if len(npc_ids) < 2:
                continue
                
            # Determine if a group interaction should occur (random chance)
            if random.random() < 0.3:  # 30% chance of group interaction
                try:
                    # Create location-specific context
                    group_context = context.copy()
                    group_context["location"] = location
                    
                    # Use the coordinator to process group decisions
                    result = await self.coordinator.make_group_decisions(npc_ids, group_context)
                    
                    # Add to results
                    group_results.append({
                        "location": location,
                        "npc_ids": npc_ids,
                        "group_actions": result.get("group_actions", []),
                        "individual_actions": result.get("individual_actions", {})
                    })
                except Exception as e:
                    logger.error(f"Error processing group activity at {location}: {e}")
        
        return group_results

    async def _get_npcs_by_location(self) -> Dict[str, List[int]]:
        """
        Group NPCs by their current location.
        
        Returns:
            Dictionary mapping locations to lists of NPC IDs
        """
        location_groups = {}
        
        try:
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT npc_id, current_location
                    FROM NPCStats
                    WHERE user_id = $1
                      AND conversation_id = $2
                      AND current_location IS NOT NULL
                """, self.user_id, self.conversation_id)
                
                for row in rows:
                    npc_id = row["npc_id"]
                    location = row["current_location"]
                    if location not in location_groups:
                        location_groups[location] = []
                    location_groups[location].append(npc_id)
        except Exception as e:
            logger.error(f"Error grouping NPCs by location: {e}")
            
        return location_groups

    async def _fetch_all_npc_data_for_activities(self) -> Dict[int, Dict[str, Any]]:
        """
        Batch fetch NPC data needed for scheduled activities.
        
        Returns:
            Dictionary mapping NPC IDs to NPC data
        """
        npc_data = {}
        
        try:
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location, schedule,
                           dominance, cruelty, introduced
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
                
                import json
                for row in rows:
                    npc_id = row["npc_id"]
                    
                    # Parse schedule JSON
                    schedule_value = row["schedule"]
                    if schedule_value is not None:
                        if isinstance(schedule_value, str):
                            try:
                                schedule_value = json.loads(schedule_value)
                            except json.JSONDecodeError:
                                schedule_value = {}
                    else:
                        schedule_value = {}
                    
                    npc_data[npc_id] = {
                        "name": row["npc_name"],
                        "location": row["current_location"],
                        "schedule": schedule_value,
                        "dominance": row["dominance"],
                        "cruelty": row["cruelty"],
                        "introduced": row["introduced"]
                    }
        except Exception as e:
            logger.error(f"Error fetching NPC data for activities: {e}")
            
        return npc_data

    def _determine_activity_significance(self, action: Dict[str, Any], result: Dict[str, Any]) -> int:
        """
        Determine how significant an NPC activity is for player memory formation.

        Returns:
            0: Not worth remembering
            1: Minor significance
            2: Moderate significance
            3: High significance
        """
        action_type = action.get("type", "unknown")
        outcome = result.get("outcome", "")
        emotional_impact = result.get("emotional_impact", 0)

        if action_type in ["think", "plan"]:
            return 0
        if abs(emotional_impact) > 2:
            return 3
        if action_type in ["talk", "command", "mock", "emotional_outburst"]:
            return 2
        if "visibly" in outcome or "noticeably" in outcome:
            return 2
        return 1

    @function_tool
    async def run_memory_maintenance(self) -> Dict[str, Any]:
        """
        Run comprehensive maintenance tasks on all NPCs' memory systems.
        This includes consolidation, decay, schema formation, and belief updates.
        
        Returns:
            Results of memory maintenance tasks
        """
        results = {}
        try:
            memory_system = await self._get_memory_system()

            # Player memory maintenance
            player_result = await memory_system.maintain(
                entity_type="player",
                entity_id=self.user_id
            )
            results["player_maintenance"] = player_result

            # NPC maintenance
            npc_results = {}
            for npc_id, agent in self.npc_agents.items():
                try:
                    npc_result = await self._run_comprehensive_npc_maintenance(npc_id)
                    npc_results[npc_id] = npc_result
                except Exception as e:
                    logger.error(f"Error in memory maintenance for NPC {npc_id}: {e}")
                    npc_results[npc_id] = {"error": str(e)}
            results["npc_maintenance"] = npc_results

            # DM (Nyx) memory
            try:
                nyx_result = await memory_system.maintain(entity_type="nyx", entity_id=0)
                results["nyx_maintenance"] = nyx_result
            except Exception as e:
                logger.error(f"Error in Nyx memory maintenance: {e}")
                results["nyx_maintenance"] = {"error": str(e)}

            # Update last maintenance time
            self._last_memory_maintenance = datetime.now()
            
            return results
        except Exception as e:
            logger.error(f"Error in system-wide memory maintenance: {e}")
            return {"error": str(e)}

    async def _run_comprehensive_npc_maintenance(self, npc_id: int) -> Dict[str, Any]:
        """
        Run comprehensive memory maintenance for a single NPC.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Results of maintenance tasks
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

    @function_tool
    async def generate_npc_flashback(
        self, 
        npc_id: int, 
        context_text: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a flashback for an NPC based on specific context.
        
        Args:
            npc_id: ID of the NPC
            context_text: Context to trigger flashback
            
        Returns:
            Flashback data or None if no flashback triggered
        """
        try:
            # Check if we've recently triggered a flashback for this NPC
            last_flashback_time = self._last_flashback_times.get(
                npc_id, 
                datetime.now() - timedelta(hours=1)
            )
            time_since_last = (datetime.now() - last_flashback_time).total_seconds()
            if time_since_last < 300:  # 5 minutes minimum between flashbacks
                return {
                    "triggered": False,
                    "reason": "too_soon",
                    "time_since_last_seconds": time_since_last
                }

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
                    
                    flashback["triggered_emotion"] = emotion
                    flashback["emotion_intensity"] = intensity
                
                flashback["triggered"] = True
                return flashback
            else:
                return {
                    "triggered": False,
                    "reason": "no_relevant_memory"
                }
        except Exception as e:
            logger.error(f"Error generating flashback for NPC {npc_id}: {e}")
            return {
                "triggered": False,
                "error": str(e)
            }

    async def process_command(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a command from the game system using the system agent.
        
        Args:
            command: Command to process
            parameters: Parameters for the command
            
        Returns:
            Result of the command
        """
        # Get the system agent
        system_agent = await self._get_system_agent()
        
        # Create input data
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
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Name of the NPC
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

    async def generate_npc_flashback(self, npc_id: int, context_text: str) -> Optional[Dict[str, Any]]:
        """
        Generate a flashback for an NPC based on specific context.
        """
        try:
            memory_system = await self._get_memory_system()
            flashback = await memory_system.npc_flashback(npc_id, context_text)
            if flashback:
                self._last_flashback_times[npc_id] = datetime.now()

                emotional_state = await memory_system.get_npc_emotion(npc_id)
                if emotional_state:
                    emotion = "fear"
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
            return flashback
        except Exception as e:
            logger.error(f"Error generating flashback for NPC {npc_id}: {e}")
            return None

    # ------------------------------------------------------------------
    # Helper methods for emotion and mask processing
    # ------------------------------------------------------------------
    def _should_update_emotion(
        self,
        player_action: Dict[str, Any],
        npc_response: Dict[str, Any]
    ) -> bool:
        action_type = player_action.get("type", "").lower()
        emotional_action = action_type in [
            "express_emotion", "flirt", "threaten", "comfort",
            "insult", "praise", "mock", "support", "challenge", "provoke"
        ]
        result = npc_response.get("result", {})
        emotional_impact = result.get("emotional_impact", 0)
        return emotional_action or abs(emotional_impact) >= 2

    async def _determine_new_emotion(
        self,
        npc_id: int,
        player_action: Dict[str, Any],
        npc_response: Dict[str, Any],
        current_state: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        action_type = player_action.get("type", "").lower()
        result = npc_response.get("result", {})
        emotional_impact = result.get("emotional_impact", 0)

        new_emotion = {"name": "neutral", "intensity": 0.3}

        if action_type in ["praise", "comfort", "support"]:
            new_emotion["name"] = "joy"
            new_emotion["intensity"] = 0.6
        elif action_type in ["insult", "mock"]:
            try:
                async with self.connection_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        SELECT dominance, cruelty
                        FROM NPCStats
                        WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
                        """,
                        npc_id, self.user_id, self.conversation_id
                    )
                    if row:
                        dominance, cruelty = row["dominance"], row["cruelty"]
                        if dominance > 60 or cruelty > 60:
                            new_emotion["name"] = "anger"
                        else:
                            new_emotion["name"] = "sadness"
                        new_emotion["intensity"] = min(
                            0.8,
                            0.5 + (max(dominance, cruelty) / 100.0)
                        )
                    else:
                        new_emotion["name"] = "sadness"
                        new_emotion["intensity"] = 0.6
            except Exception as e:
                logger.error(f"Error getting NPC stats for emotion: {e}")
                new_emotion["name"] = "sadness"
                new_emotion["intensity"] = 0.6
        elif action_type == "threaten":
            new_emotion["name"] = "fear"
            new_emotion["intensity"] = 0.7
        elif action_type == "flirt":
            try:
                memory_system = await self._get_memory_system()
                beliefs = await memory_system.get_beliefs(entity_type="npc", entity_id=npc_id, topic="player")
                positive_belief = False
                for belief in beliefs:
                    b_text = belief.get("belief", "").lower()
                    if any(word in b_text for word in ["attract", "like", "interest"]) and \
                       belief.get("confidence", 0) > 0.5:
                        positive_belief = True
                        break
                if positive_belief:
                    new_emotion["name"] = "joy"
                    new_emotion["intensity"] = 0.7
                else:
                    new_emotion["name"] = "surprise"
                    new_emotion["intensity"] = 0.5
            except Exception as e:
                logger.error(f"Error determining flirt response: {e}")
                new_emotion["name"] = "surprise"
                new_emotion["intensity"] = 0.5

        # Overwrite if emotional_impact is very strong
        if abs(emotional_impact) >= 3:
            if emotional_impact > 3:
                new_emotion["name"] = "joy"
                new_emotion["intensity"] = 0.7
            elif emotional_impact > 0:
                new_emotion["name"] = "joy"
                new_emotion["intensity"] = 0.5
            elif emotional_impact < -3:
                new_emotion["name"] = "anger" if random.random() < 0.5 else "sadness"
                new_emotion["intensity"] = 0.7
            else:
                new_emotion["name"] = "sadness"
                new_emotion["intensity"] = 0.5

        if current_state and "current_emotion" in current_state:
            current = current_state["current_emotion"]
            current_name = current.get("primary")
            current_intensity = current.get("intensity", 0)
            if current_name == new_emotion["name"] and \
               abs(current_intensity - new_emotion["intensity"]) < 0.2:
                return None

        return new_emotion

    def _should_check_mask_slippage(
        self,
        player_action: Dict[str, Any],
        npc_response: Dict[str, Any],
        mask_info: Optional[Dict[str, Any]]
    ) -> bool:
        if not mask_info or mask_info.get("integrity", 100) >= 95:
            return False
        action_type = player_action.get("type", "").lower()
        challenging_action = action_type in [
            "threaten", "challenge", "accuse", "provoke", "mock", "insult"
        ]
        result = npc_response.get("result", {})
        emotional_impact = abs(result.get("emotional_impact", 0))
        compromised_mask = mask_info.get("integrity", 100) < 70
        return challenging_action or emotional_impact > 2 or compromised_mask

    def _calculate_mask_slippage_chance(
        self,
        player_action: Dict[str, Any],
        npc_response: Dict[str, Any],
        mask_info: Optional[Dict[str, Any]]
    ) -> float:
        if not mask_info:
            return 0.0
        integrity = mask_info.get("integrity", 100)
        base_chance = max(0.0, (100 - integrity) / 200.0)  # up to 0.5

        action_type = player_action.get("type", "").lower()
        if action_type in ["threaten", "challenge", "accuse", "provoke", "mock", "insult"]:
            base_chance += 0.1

        result = npc_response.get("result", {})
        emotional_impact = abs(result.get("emotional_impact", 0))
        impact_factor = min(0.3, emotional_impact * 0.05)
        base_chance += impact_factor
        return min(0.75, base_chance)

    def _process_belief_updates(
        self,
        npc_id: int,
        player_action: Dict[str, Any],
        npc_response: Dict[str, Any],
        existing_beliefs: List[Dict[str, Any]]
    ) -> None:
        """
        Process belief updates based on player-NPC interaction.
        Note this method is synchronous logic, but actual DB calls are handled
        by memory_system asynchronously. We're only shaping the logic here.
        """
        action_type = player_action.get("type", "").lower()
        result = npc_response.get("result", {})
        emotional_impact = result.get("emotional_impact", 0)

        should_update = False
        belief_forming_actions = [
            "threaten", "praise", "insult", "support", "betray", "help",
            "challenge", "defend", "protect", "attack", "share", "confide",
            "flirt", "reject"
        ]
        if action_type in belief_forming_actions:
            should_update = True
        if abs(emotional_impact) >= 3:
            should_update = True

        # Actual updates to beliefs are done with memory_system calls inside the method
        # This code just sets up the logic. No DB calls directly here.

# End of agent_system.py
