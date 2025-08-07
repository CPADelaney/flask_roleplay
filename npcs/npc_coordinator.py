# npcs/npc_coordinator.py

"""
Coordinates multiple NPC agents for group interactions, using OpenAI Agents SDK.
Refactored from the original agent_coordinator.py.
"""
from lore.core import canon
from lore.core.lore_system import LoreSystem

import logging
import asyncio
import random
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pydantic import BaseModel

from agents import Agent, Runner, function_tool, handoff, trace, InputGuardrail, RunContextWrapper, GuardrailFunctionOutput
from db.connection import get_db_connection_context  # Updated import
from memory.wrapper import MemorySystem

from npcs.npc_agent import NPCAgent, ResourcePool

logger = logging.getLogger(__name__)

class GroupContext(BaseModel):
    """Context for group interactions."""
    location: str
    time_of_day: str
    participants: List[int]
    shared_history: Optional[List[Dict[str, Any]]] = None
    emotional_states: Optional[Dict[str, Dict[str, Any]]] = None

class GroupDecisionOutput(BaseModel):
    """Output from group decision-making process."""
    group_actions: List[Dict[str, Any]]
    individual_actions: Dict[str, List[Dict[str, Any]]]
    reasoning: str

class HomeworkCheck(BaseModel):
    """Output for homework guardrail check."""
    is_homework: bool
    reasoning: str

class NPCAgentCoordinator:
    """Coordinates the behavior of multiple NPC agents using the Agents SDK."""

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.active_agents: Dict[int, NPCAgent] = {}
        self._memory_system = None
        self._coordinator_agent = None

        # Add resource pools for different operation types
        self.resource_pools = {
            "decisions": ResourcePool(max_concurrent=10, timeout=45.0),
            "perceptions": ResourcePool(max_concurrent=15, timeout=30.0),
            "memory_operations": ResourcePool(max_concurrent=20, timeout=20.0)
        }

        # Cache systems to reduce repeated queries
        self._emotional_states = {}  # Cache of emotional states
        self._emotional_states_timestamps = {}  # When the states were last updated
        self._mask_states = {}       # Cache of mask states
        self._mask_states_timestamps = {}  # When the states were last updated

        # Cache TTL settings
        self._cache_ttl = {
            "emotional_state": 120,  # 2 minutes
            "mask": 300,             # 5 minutes
        }

        # Locks for synchronization
        self._memory_system_lock = asyncio.Lock()
        self._emotional_state_lock = asyncio.Lock()
        self._mask_state_lock = asyncio.Lock()
        self._agent_init_lock = asyncio.Lock()
        self._batch_update_lock = asyncio.Lock()

    async def _get_memory_system(self):
        """Lazy-load the memory system with synchronization."""
        if self._memory_system is None:
            async with self._memory_system_lock:
                if self._memory_system is None:
                    self._memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        return self._memory_system

    async def make_group_decisions_with_nyx(
        self,
        npc_ids: List[int],
        shared_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make group decisions with Nyx's guidance"""
        # Get Nyx's guidance for the scene
        from .nyx import NyxAgent  # Import here to avoid circular imports
        nyx_agent = NyxAgent(self.user_id, self.conversation_id)
        nyx_guidance = await nyx_agent.get_scene_guidance(shared_context)
        
        # Add Nyx's guidance to the context
        enhanced_context = await self._prepare_group_context(npc_ids, shared_context)
        enhanced_context["nyx_guidance"] = nyx_guidance
        
        # Make decisions with Nyx's influence
        return await self.make_group_decisions(npc_ids, enhanced_context)

    async def _get_coordinator_agent(self):
        """Lazy-load the coordinator agent with synchronization."""
        if self._coordinator_agent is None:
            async with self._memory_system_lock:  # Reusing the memory system lock is fine here
                if self._coordinator_agent is None:
                    self._coordinator_agent = Agent(
                        name="NPC_Group_Coordinator",
                        instructions="""
                            You coordinate interactions between multiple NPCs in a group setting.
                            Your job is to decide what actions each NPC should take based on their traits,
                            relationships, and the context of the interaction.

                            Consider the following factors:
                            1. Each NPC's dominance and cruelty levels
                            2. Relationships between NPCs
                            3. Emotional states of NPCs
                            4. The specific context of the interaction
                            5. Previous group interactions

                            Your output should include:
                            - Group actions: Actions that affect the entire group
                            - Individual actions: Actions specific to each NPC
                            - Reasoning: Explanation for your decisions

                            Ensure that actions align with each NPC's personality and maintain consistent
                            character behavior.
                        """,
                        model="gpt-5-nano",
                        tools=[
                            function_tool(self._get_npc_emotional_state),
                            function_tool(self._get_npc_mask),
                            function_tool(self._get_npc_traits),
                            function_tool(self._get_relationships_between_npcs),
                            function_tool(self._create_group_memory),
                        ],
                        output_type=GroupDecisionOutput
                    )
        return self._coordinator_agent

    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get statistics about resource pool usage."""
        stats = {}
        for name, pool in self.resource_pools.items():
            stats[name] = pool.stats.copy()
        return stats

    async def _log_resource_stats(self):
        """Log resource usage statistics periodically."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                stats = await self.get_resource_stats()
                logger.info(f"Resource pool stats: {stats}")
            except Exception as e:
                logger.error(f"Error logging resource stats: {e}")

    async def load_agents(self, npc_ids: Optional[List[int]] = None) -> List[int]:
        """
        Load specified NPC agents into memory, or load all if none specified.
        Thread-safe implementation that prevents duplicate initialization.
        """
        if npc_ids is None:
            logger.info("Loading all NPC agents for user=%s, conversation=%s", self.user_id, self.conversation_id)
        else:
            logger.info("Loading NPC agents: %s", npc_ids)

        query = """
            SELECT npc_id
            FROM NPCStats
            WHERE user_id = %s
              AND conversation_id = %s
        """
        params = [self.user_id, self.conversation_id]

        if npc_ids:
            query += " AND npc_id = ANY(%s)"
            params.append(npc_ids)

        loaded_ids: List[int] = []

        try:
            # Refactored to use async connection
            async with get_db_connection_context() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(query, params)
                    rows = await cursor.fetchall()

                    # First collect all IDs to load
                    to_load = []
                    for row in rows:
                        npc_id = row[0]
                        if npc_id not in self.active_agents:
                            to_load.append(npc_id)
                        loaded_ids.append(npc_id)

                    # Then initialize them with proper locking
                    for npc_id in to_load:
                        async with self._agent_init_lock:
                            if npc_id not in self.active_agents:
                                self.active_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
                                await self.active_agents[npc_id].initialize()

            logger.info("Loaded agents: %s", loaded_ids)
            return loaded_ids
        except Exception as e:
            logger.error(f"Error loading agents: {e}")
            return []

    @function_tool
    async def _get_npc_emotional_state(self, npc_id: int) -> Dict[str, Any]:
        """
        Get an NPC's current emotional state, with caching for performance.
        Thread-safe implementation.
        """
        now = datetime.now()

        # Quick check without lock
        if npc_id in self._emotional_states:
            timestamp = self._emotional_states_timestamps.get(npc_id)
            if timestamp and (now - timestamp).total_seconds() < self._cache_ttl["emotional_state"]:
                return self._emotional_states[npc_id]

        # If we need to fetch or update, acquire lock
        async with self._emotional_state_lock:
            # Check again within the lock
            if npc_id in self._emotional_states:
                timestamp = self._emotional_states_timestamps.get(npc_id)
                if timestamp and (now - timestamp).total_seconds() < self._cache_ttl["emotional_state"]:
                    return self._emotional_states[npc_id]

            try:
                memory_system = await self._get_memory_system()
                emotional_state = await memory_system.get_npc_emotion(npc_id)

                self._emotional_states[npc_id] = emotional_state
                self._emotional_states_timestamps[npc_id] = now
                return emotional_state
            except Exception as e:
                logger.error(f"Error getting emotional state for NPC {npc_id}: {e}")
                return {}

    @function_tool
    async def _get_npc_mask(self, npc_id: int) -> Dict[str, Any]:
        """
        Get an NPC's mask information, with caching for performance.
        Thread-safe implementation.
        """
        now = datetime.now()

        # Quick check without lock
        if npc_id in self._mask_states:
            timestamp = self._mask_states_timestamps.get(npc_id)
            if timestamp and (now - timestamp).total_seconds() < self._cache_ttl["mask"]:
                return self._mask_states[npc_id]

        # If we need to fetch or update, acquire lock
        async with self._mask_state_lock:
            if npc_id in self._mask_states:
                timestamp = self._mask_states_timestamps.get(npc_id)
                if timestamp and (now - timestamp).total_seconds() < self._cache_ttl["mask"]:
                    return self._mask_states[npc_id]

            try:
                memory_system = await self._get_memory_system()
                mask_info = await memory_system.get_npc_mask(npc_id)

                self._mask_states[npc_id] = mask_info
                self._mask_states_timestamps[npc_id] = now
                return mask_info
            except Exception as e:
                logger.error(f"Error getting mask info for NPC {npc_id}: {e}")
                return {}

    @function_tool
    async def _get_npc_traits(self, npc_id: int) -> Dict[str, Any]:
        """
        Get an NPC's traits and personality information.
        """
        try:
            # Refactored to use async connection
            async with get_db_connection_context() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        """
                        SELECT npc_name, dominance, cruelty, personality_traits 
                        FROM NPCStats
                        WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                        """,
                        (npc_id, self.user_id, self.conversation_id),
                    )
                    row = await cursor.fetchone()
                    if not row:
                        return {"error": f"NPC {npc_id} not found"}

                    npc_name, dominance, cruelty, personality_traits = row

                    # Parse personality traits if it's a JSON string
                    if personality_traits and isinstance(personality_traits, str):
                        try:
                            personality_traits = json.loads(personality_traits)
                        except json.JSONDecodeError:
                            personality_traits = []

                    return {
                        "npc_id": npc_id,
                        "npc_name": npc_name,
                        "dominance": dominance,
                        "cruelty": cruelty,
                        "personality_traits": personality_traits,
                    }
        except Exception as e:
            logger.error(f"Error getting NPC traits for {npc_id}: {e}")
            return {"error": str(e)}

    @function_tool
    async def _get_relationships_between_npcs(self, npc_ids: List[int]) -> Dict[str, Any]:
        """
        Get relationship information between a group of NPCs.
        """
        if not npc_ids or len(npc_ids) < 2:
            return {}

        relationships = {}

        try:
            # Refactored to use async connection
            async with get_db_connection_context() as conn:
                async with conn.cursor() as cursor:
                    for i, npc1 in enumerate(npc_ids):
                        for npc2 in npc_ids[i + 1:]:
                            await cursor.execute(
                                """
                                SELECT link_type, link_level 
                                FROM SocialLinks
                                WHERE user_id = %s AND conversation_id = %s
                                  AND (
                                    (entity1_type = 'npc' AND entity1_id = %s AND entity2_type = 'npc' AND entity2_id = %s)
                                    OR
                                    (entity1_type = 'npc' AND entity1_id = %s AND entity2_type = 'npc' AND entity2_id = %s)
                                  )
                                """,
                                (self.user_id, self.conversation_id, npc1, npc2, npc2, npc1),
                            )
                            row = await cursor.fetchone()
                            key = f"{min(npc1, npc2)}_{max(npc1, npc2)}"
                            if row:
                                link_type, link_level = row
                                relationships[key] = {
                                    "npc1": npc1,
                                    "npc2": npc2,
                                    "link_type": link_type,
                                    "link_level": link_level,
                                }
                            else:
                                # No established relationship
                                relationships[key] = {
                                    "npc1": npc1,
                                    "npc2": npc2,
                                    "link_type": "neutral",
                                    "link_level": 50,
                                }

            return relationships
        except Exception as e:
            logger.error(f"Error getting relationships between NPCs: {e}")
            return {}
            
    @function_tool
    async def _create_group_memory(
        self,
        npc_ids: List[int],
        memory_text: str,
        importance: str = "medium",
        tags: List[str] = ["group_interaction"],
    ) -> Dict[str, Any]:
        """
        Create a memory of a group interaction for all participating NPCs.
        """
        memory_system = await self._get_memory_system()
        results = {}

        # Create tasks for all NPCs but execute in smaller batches
        batch_size = 5
        for i in range(0, len(npc_ids), batch_size):
            batch = npc_ids[i : i + batch_size]
            tasks = []

            for npc_id in batch:
                task = memory_system.remember(
                    entity_type="npc",
                    entity_id=npc_id,
                    memory_text=memory_text,
                    importance=importance,
                    tags=tags,
                )
                tasks.append(task)

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for npc_id, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error creating memory for NPC {npc_id}: {result}")
                    results[npc_id] = {"status": "error", "message": str(result)}
                else:
                    results[npc_id] = {"status": "success", "memory_id": result.get("id")}

            if i + batch_size < len(npc_ids):
                await asyncio.sleep(0.05)

        return results

    async def make_group_decisions(
        self,
        npc_ids: List[int],
        shared_context: Dict[str, Any],
        available_actions: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """Coordinate decision-making for a group of NPCs."""
        # NEW: Always get Nyx's approval first
        nyx_approval = await self._get_nyx_scene_approval(npc_ids, shared_context)
        
        if not nyx_approval.get("approved", False):
            logger.info(f"Nyx rejected group action: {nyx_approval.get('reason', 'No reason provided')}")
            return {
                "group_actions": [],
                "individual_actions": {},
                "reasoning": f"Nyx has not approved this group interaction: {nyx_approval.get('reason', 'Unknown reason')}",
                "nyx_override": True
            }
        
        # Use Nyx's modifications if provided
        if "modified_context" in nyx_approval:
            shared_context = nyx_approval["modified_context"]
            
        decision_resource = await self.resource_pools["decisions"].acquire()

        try:
            await self.load_agents(npc_ids)
            coordinator_agent = await self._get_coordinator_agent()

            # 1. Prepare enhanced group context with memory integration
            enhanced_context = await self._prepare_group_context(npc_ids, shared_context)

            # 2. Generate actions if not provided
            if available_actions is None:
                if not decision_resource:
                    logger.warning("Decision resources constrained, using smaller batches")
                    available_actions = await self.generate_group_actions(npc_ids[:5], enhanced_context)
                else:
                    available_actions = await self.generate_group_actions(npc_ids, enhanced_context)

            # 3. Prepare input for the coordinator agent
            input_data = {
                "context": enhanced_context,
                "npc_ids": npc_ids,
                "available_actions": available_actions,
            }

            # 4. Create trace for debugging
            with trace(
                f"group_decision_{self.user_id}_{self.conversation_id}",
                group_id=f"user_{self.user_id}_conv_{self.conversation_id}",
            ):
                # 5. Run the coordinator agent
                result = await Runner.run(coordinator_agent, input_data)
                output = result.final_output_as(GroupDecisionOutput)

                # 6. Create memories for all NPCs based on the decision
                location = enhanced_context.get("location", "Unknown")
                memory_text = f"I participated in a group interaction at {location} with {len(npc_ids)} others"

                memory_resource = await self.resource_pools["memory_operations"].acquire()
                try:
                    await self._create_group_memory(
                        npc_ids=npc_ids,
                        memory_text=memory_text,
                        importance="medium",
                        tags=["group_interaction", "group_decision"],
                    )
                finally:
                    if memory_resource:
                        self.resource_pools["memory_operations"].release()

                # 7. Return the action plan
                return {
                    "group_actions": output.group_actions,
                    "individual_actions": output.individual_actions,
                    "reasoning": output.reasoning,
                    "context": enhanced_context,
                }
        finally:
            if decision_resource:
                self.resource_pools["decisions"].release()

    async def _get_nyx_scene_approval(
        self, 
        npc_ids: List[int], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get Nyx's approval for a group interaction."""
        try:
            # Import here to avoid circular imports
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
                    "requested_at": datetime.now().isoformat()
                }
            )
            
            # Transform to expected format
            approval_result = {
                "approved": approval.get("approved", True),
                "reason": approval.get("reasoning", ""),
                "modified_context": approval.get("modified_action_details", {}).get("context")
            }
            
            return approval_result
        except Exception as e:
            logger.error(f"Error getting Nyx approval: {e}")
            # Default to approved if we can't reach Nyx to prevent game blocking
            return {"approved": True, "reason": "Failed to contact Nyx, proceeding by default"}

    async def _prepare_group_context(self, npc_ids: List[int], shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare enhanced context for group interactions with memory.
        Thread-safe implementation with batched processing.
        """
        # Acquire perception resource
        perception_resource = await self.resource_pools["perceptions"].acquire()

        try:
            memory_system = await self._get_memory_system()

            # Create enhanced context
            enhanced_context = shared_context.copy()
            enhanced_context["participants"] = npc_ids
            enhanced_context["type"] = "group_interaction"

            # Add location if not present
            if "location" not in enhanced_context:
                try:
                    with get_db_connection() as conn, conn.cursor() as cursor:
                        cursor.execute(
                            """
                            SELECT current_location
                            FROM NPCStats
                            WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                            """,
                            (npc_ids[0], self.user_id, self.conversation_id),
                        )
                        row = cursor.fetchone()
                        if row and row[0]:
                            enhanced_context["location"] = row[0]
                except Exception as e:
                    logger.error(f"Error getting location for context: {e}")

            # Add time if not present
            if "time_of_day" not in enhanced_context:
                try:
                    with get_db_connection() as conn, conn.cursor() as cursor:
                        cursor.execute(
                            """
                            SELECT value
                            FROM CurrentRoleplay
                            WHERE user_id = %s AND conversation_id = %s AND key = 'TimeOfDay'
                            """,
                            (self.user_id, self.conversation_id),
                        )
                        row = cursor.fetchone()
                        if row:
                            enhanced_context["time_of_day"] = row[0]
                except Exception as e:
                    logger.error(f"Error getting time for context: {e}")

            if "npc_context" not in enhanced_context:
                enhanced_context["npc_context"] = {}

            # Build up NPC-specific contexts in batches
            batch_size = 5
            if not perception_resource:
                batch_size = 3
                logger.warning("Perception resources constrained, using smaller batch size")

            npc_contexts = {}

            for i in range(0, len(npc_ids), batch_size):
                batch_npc_ids = npc_ids[i : i + batch_size]
                batch_tasks = []
                for npc_id in batch_npc_ids:
                    batch_tasks.append(self._prepare_single_npc_context(npc_id, npc_ids, enhanced_context))
                batch_results = await asyncio.gather(*batch_tasks)

                # Add each result to the npc_context
                for result in batch_results:
                    npc_id = result.pop("npc_id")
                    npc_contexts[npc_id] = result

                if i + batch_size < len(npc_ids):
                    await asyncio.sleep(0.05)

            enhanced_context["npc_context"] = npc_contexts

            # Add shared group memories
            shared_memories = await memory_system.recall(
                entity_type="npc",
                entity_id=npc_ids[0],  # Just use the first NPC as reference
                query="group interaction",
                context={"location": enhanced_context.get("location", "Unknown")},
                limit=2,
            )
            enhanced_context["shared_history"] = shared_memories.get("memories", [])

            return enhanced_context
        finally:
            # Make sure we always release this
            if perception_resource:
                self.resource_pools["perceptions"].release()

    async def _prepare_single_npc_context(
        self,
        npc_id: int,
        group_npc_ids: List[int],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare context for a single NPC within a group."""
        memory_system = await self._get_memory_system()

        group_memories = await memory_system.recall(
            entity_type="npc",
            entity_id=npc_id,
            query="group interaction",
            context=context,
            limit=3,
        )

        # Random chance of a flashback
        flashback = None
        if random.random() < 0.15:
            context_text = f"group interaction at {context.get('location', 'Unknown')}"
            flashback = await memory_system.npc_flashback(npc_id, context_text)

        # Get NPC's emotional state (cached)
        emotional_state = await self._get_npc_emotional_state(npc_id)
        # Get NPC's mask info (cached)
        mask_info = await self._get_npc_mask(npc_id)
        # Get NPC's traits
        traits = await self._get_npc_traits(npc_id)

        # Collect beliefs about others in the group
        beliefs = {}
        other_npc_ids = [x for x in group_npc_ids if x != npc_id]
        batch_size = 3
        for i in range(0, len(other_npc_ids), batch_size):
            batch = other_npc_ids[i : i + batch_size]
            batch_tasks = []
            for other_id in batch:
                task = memory_system.get_beliefs(entity_type="npc", entity_id=npc_id, topic=f"npc_{other_id}")
                batch_tasks.append(task)

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for other_id, result in zip(batch, batch_results):
                if not isinstance(result, Exception) and result:
                    beliefs[other_id] = result

            if i + batch_size < len(other_npc_ids):
                await asyncio.sleep(0.02)

        return {
            "npc_id": npc_id,
            "group_memories": group_memories.get("memories", []),
            "emotional_state": emotional_state,
            "mask_info": mask_info,
            "traits": traits,
            "flashback": flashback,
            "beliefs": beliefs,
        }

    async def generate_group_actions(
        self, npc_ids: List[int], context: Dict[str, Any]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Generate possible actions for each NPC in a group context.
        Enhanced with memory-based influences.
        Thread-safe implementation with batched processing.
        """
        group_actions: Dict[int, List[Dict[str, Any]]] = {}
        memory_system = await self._get_memory_system()

        batch_size = 5
        for i in range(0, len(npc_ids), batch_size):
            batch = npc_ids[i : i + batch_size]
            batch_tasks = []
            for npc_id in batch:
                batch_tasks.append(self._generate_actions_for_npc(npc_id, npc_ids, context, memory_system))
            batch_results = await asyncio.gather(*batch_tasks)
            for npc_id, actions in batch_results:
                group_actions[npc_id] = actions
            if i + batch_size < len(npc_ids):
                await asyncio.sleep(0.05)

        return group_actions

    async def _generate_actions_for_npc(
        self,
        npc_id: int,
        all_npc_ids: List[int],
        context: Dict[str, Any],
        memory_system: MemorySystem,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Generate actions for a single NPC.
        """
        npc_traits = await self._get_npc_traits(npc_id)
        if "error" in npc_traits:
            return npc_id, []

        dom = npc_traits.get("dominance", 50)
        cru = npc_traits.get("cruelty", 50)
        name = npc_traits.get("npc_name", f"NPC_{npc_id}")

        actions = [
            {"type": "talk", "description": "Talk to the group", "target": "group"},
            {"type": "observe", "description": "Observe the group", "target": "group"},
            {"type": "leave", "description": "Leave the group", "target": "group"},
        ]

        emotional_state = await self._get_npc_emotional_state(npc_id)
        mask_info = await self._get_npc_mask(npc_id)
        beliefs = await memory_system.get_beliefs(entity_type="npc", entity_id=npc_id, topic="group_interaction")

        # Dominance-based actions
        if dom > 60:
            actions.append({
                "type": "command",
                "description": "Give an authoritative command",
                "target": "group",
                "stats_influenced": {"dominance": 1, "trust": -1},
            })
            actions.append({
                "type": "test",
                "description": "Test group's obedience",
                "target": "group",
                "stats_influenced": {"dominance": 2, "respect": -1},
            })
            if dom > 75:
                actions.append({
                    "type": "dominate",
                    "description": "Assert dominance forcefully",
                    "target": "group",
                    "stats_influenced": {"dominance": 3, "fear": 2},
                })

        # Cruelty-based actions
        if cru > 60:
            actions.append({
                "type": "mock",
                "description": "Mock or belittle the group",
                "target": "group",
                "stats_influenced": {"cruelty": 1, "closeness": -2},
            })
            if cru > 70:
                actions.append({
                    "type": "humiliate",
                    "description": "Deliberately humiliate the group",
                    "target": "group",
                    "stats_influenced": {"cruelty": 2, "fear": 2},
                })

        # Emotionally-influenced
        if emotional_state and "current_emotion" in emotional_state:
            current_emotion = emotional_state["current_emotion"]
            primary = current_emotion.get("primary", {})
            if isinstance(primary, dict) and "name" in primary:
                emotion_name = primary.get("name", "neutral")
                intensity = primary.get("intensity", 0.0)
            else:
                emotion_name = primary if primary else "neutral"
                intensity = current_emotion.get("intensity", 0.0)

            if intensity > 0.7:
                if emotion_name == "anger":
                    actions.append({
                        "type": "express_anger",
                        "description": "Express anger forcefully",
                        "target": "group",
                        "stats_influenced": {"dominance": 2, "closeness": -3},
                    })
                elif emotion_name == "fear":
                    actions.append({
                        "type": "act_defensive",
                        "description": "Act defensively and guarded",
                        "target": "environment",
                        "stats_influenced": {"trust": -2},
                    })
                elif emotion_name == "joy":
                    actions.append({
                        "type": "celebrate",
                        "description": "Share happiness enthusiastically",
                        "target": "group",
                        "stats_influenced": {"closeness": 3},
                    })

        # Mask integrity
        mask_integrity = 100
        hidden_traits = {}
        if mask_info:
            mask_integrity = mask_info.get("integrity", 100)
            hidden_traits = mask_info.get("hidden_traits", {})

        if mask_integrity < 70:
            # As mask breaks down, hidden traits might show
            if isinstance(hidden_traits, dict):
                for trait, value in hidden_traits.items():
                    if trait == "dominant" and value:
                        actions.append({
                            "type": "mask_slip",
                            "description": "Show unexpected dominance",
                            "target": "group",
                            "stats_influenced": {"dominance": 3, "fear": 2},
                        })
                    elif trait == "cruel" and value:
                        actions.append({
                            "type": "mask_slip",
                            "description": "Reveal unexpected cruelty",
                            "target": "group",
                            "stats_influenced": {"cruelty": 2, "fear": 1},
                        })
                    elif trait == "submissive" and value:
                        actions.append({
                            "type": "mask_slip",
                            "description": "Show unexpected submission",
                            "target": "group",
                            "stats_influenced": {"dominance": -2},
                        })
            elif isinstance(hidden_traits, list):
                if "dominant" in hidden_traits:
                    actions.append({
                        "type": "mask_slip",
                        "description": "Show unexpected dominance",
                        "target": "group",
                        "stats_influenced": {"dominance": 3, "fear": 2},
                    })
                if "cruel" in hidden_traits:
                    actions.append({
                        "type": "mask_slip",
                        "description": "Reveal unexpected cruelty",
                        "target": "group",
                        "stats_influenced": {"cruelty": 2, "fear": 1},
                    })
                if "submissive" in hidden_traits:
                    actions.append({
                        "type": "mask_slip",
                        "description": "Show unexpected submission",
                        "target": "group",
                        "stats_influenced": {"dominance": -2},
                    })

        # Beliefs
        if beliefs:
            for belief in beliefs:
                belief_text = belief.get("belief", "").lower()
                if "dangerous" in belief_text or "threat" in belief_text:
                    actions.append({
                        "type": "defensive",
                        "description": "Take a defensive stance",
                        "target": "group",
                        "stats_influenced": {"trust": -2},
                    })
                elif "opportunity" in belief_text or "beneficial" in belief_text:
                    actions.append({
                        "type": "engage",
                        "description": "Actively engage with the group",
                        "target": "group",
                        "stats_influenced": {"closeness": 2},
                    })

        # Location-based
        location = context.get("location", "").lower()
        if any(loc in location for loc in ["cafe", "restaurant", "bar", "party"]):
            actions.append({
                "type": "socialize",
                "description": "Engage in group conversation",
                "target": "group",
                "stats_influenced": {"closeness": 1},
            })

        # Target-specific actions
        other_npcs = [o for o in all_npc_ids if o != npc_id]
        batch_size = 3
        for i in range(0, len(other_npcs), batch_size):
            batch = other_npcs[i : i + batch_size]
            for other_id in batch:
                other_traits = await self._get_npc_traits(other_id)
                if "error" in other_traits:
                    continue

                other_name = other_traits.get("npc_name", f"NPC_{other_id}")
                actions.append({
                    "type": "talk_to",
                    "description": f"Talk to {other_name}",
                    "target": str(other_id),
                    "target_name": other_name,
                    "stats_influenced": {"closeness": 1},
                })
                if dom > 60:
                    actions.append({
                        "type": "command",
                        "description": f"Command {other_name}",
                        "target": str(other_id),
                        "target_name": other_name,
                        "stats_influenced": {"dominance": 1, "trust": -1},
                    })
                if cru > 60:
                    actions.append({
                        "type": "mock",
                        "description": f"Mock {other_name}",
                        "target": str(other_id),
                        "target_name": other_name,
                        "stats_influenced": {"cruelty": 1, "closeness": -2},
                    })
            if i + batch_size < len(other_npcs):
                await asyncio.sleep(0.01)

        return npc_id, actions

    async def handle_player_action(
        self,
        player_action: Dict[str, Any],
        context: Dict[str, Any],
        npc_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Handle a player action directed at multiple NPCs.
        """
        guardrail_agent = Agent(
            name="Content Guardrail",
            instructions=(
                "Check if the player is asking about homework or schoolwork. If they "
                "are asking for solutions to actual homework, we must flag it."
            ),
            output_type=HomeworkCheck,
        )

        async def homework_guardrail(ctx: RunContextWrapper, agent: Agent, input_data: Dict[str, Any]) -> GuardrailFunctionOutput:
            result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
            final_output = result.final_output_as(HomeworkCheck)
            return GuardrailFunctionOutput(
                output_info=final_output,
                tripwire_triggered=final_output.is_homework
            )

        coordinator = Agent(
            name="Player Action Coordinator",
            instructions="""
                You coordinate NPC responses to player actions.
                Consider each NPC's personality, emotional state, and relationships
                when determining how they should respond.
            """,
            input_guardrails=[InputGuardrail(guardrail_function=homework_guardrail)],
            tools=[function_tool(self._process_player_action_for_npcs)],
        )

        if npc_ids is None:
            npc_ids = await self._determine_affected_npcs(player_action, context)
        
        if not npc_ids:
            logger.debug("No NPCs were affected by this action: %s", player_action)
            return {"npc_responses": []}

        input_data = {
            "player_action": player_action,
            "context": context,
            "npc_ids": npc_ids,
        }

        with trace(
            f"player_action_{self.user_id}_{self.conversation_id}",
            group_id=f"user_{self.user_id}_conv_{self.conversation_id}",
        ):
            result = await Runner.run(coordinator, input_data)
            return result.final_output

    @function_tool(strict_mode=False)
    async def _process_player_action_for_npcs(
        self,
        player_action: Dict[str, Any],
        context: Dict[str, Any],
        npc_ids: List[int],
    ) -> Dict[str, Any]:
        """
        Process a player action for multiple NPCs.
        """
        await self.load_agents(npc_ids)
        enhanced_context = context.copy()
        enhanced_context["is_group_interaction"] = True
        enhanced_context["affected_npcs"] = npc_ids
        memory_system = await self._get_memory_system()

        # Emotional states in batches
        emotional_states = {}
        for i in range(0, len(npc_ids), 5):
            batch = npc_ids[i : i + 5]
            batch_tasks = [self._get_npc_emotional_state(n) for n in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for npc_id, result in zip(batch, batch_results):
                if not isinstance(result, Exception):
                    emotional_states[npc_id] = result
            if i + 5 < len(npc_ids):
                await asyncio.sleep(0.05)
        enhanced_context["emotional_states"] = emotional_states

        # Mask states in batches
        mask_states = {}
        for i in range(0, len(npc_ids), 5):
            batch = npc_ids[i : i + 5]
            batch_tasks = [self._get_npc_mask(n) for n in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for npc_id, result in zip(batch, batch_results):
                if not isinstance(result, Exception):
                    mask_states[npc_id] = result
            if i + 5 < len(npc_ids):
                await asyncio.sleep(0.05)
        enhanced_context["mask_states"] = mask_states

        # Process player action in batches
        npc_responses = []
        batch_size = 3
        for i in range(0, len(npc_ids), batch_size):
            batch = npc_ids[i : i + batch_size]
            batch_tasks = []
            for npc_id in batch:
                agent = self.active_agents.get(npc_id)
                if agent:
                    npc_context = enhanced_context.copy()
                    npc_context["emotional_state"] = emotional_states.get(npc_id)
                    npc_context["mask_info"] = mask_states.get(npc_id)
                    batch_tasks.append(agent.process_player_action(player_action, npc_context))
                else:
                    batch_tasks.append(asyncio.sleep(0))
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in batch_results:
                if not isinstance(result, Exception) and result is not None:
                    npc_responses.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error processing NPC response: {result}")
            if i + batch_size < len(npc_ids):
                await asyncio.sleep(0.1)

        # Create group memories if more than one NPC involved
        if len(npc_ids) > 1:
            await self._create_player_group_interaction_memories(npc_ids, player_action, context)

        return {"npc_responses": npc_responses}

    async def _determine_affected_npcs(
        self,
        player_action: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[int]:
        """
        Determine which NPCs are affected by a player action.
        """
        if "target_npc_id" in player_action:
            return [player_action["target_npc_id"]]

        current_location = context.get("location", "Unknown")
        logger.debug("Determining NPCs at location=%s", current_location)
        try:
            # Refactored to use async connection
            async with get_db_connection_context() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        """
                        SELECT npc_id
                        FROM NPCStats
                        WHERE user_id = %s
                          AND conversation_id = %s
                          AND current_location = %s
                        """,
                        (self.user_id, self.conversation_id, current_location),
                    )
                    rows = await cursor.fetchall()
                    return [row[0] for row in rows]
        except Exception as e:
            logger.error(f"Error determining affected NPCs: {e}")
            return []

    async def _create_player_group_interaction_memories(
        self,
        npc_ids: List[int],
        player_action: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """
        Create memories of a group interaction with the player.
        Thread-safe implementation with batched processing.
        """
        memory_system = await self._get_memory_system()
        npc_names_dict = {}
        batch_size = 5

        # Pull NPC names
        for i in range(0, len(npc_ids), batch_size):
            batch = npc_ids[i : i + batch_size]
            batch_tasks = [self._get_npc_traits(n) for n in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for npc_id, result in zip(batch, batch_results):
                if not isinstance(result, Exception) and "error" not in result:
                    npc_names_dict[npc_id] = result.get("npc_name", f"NPC_{npc_id}")
                else:
                    npc_names_dict[npc_id] = f"NPC_{npc_id}"
            if i + batch_size < len(npc_ids):
                await asyncio.sleep(0.05)

        # Create memories in smaller batches
        for i in range(0, len(npc_ids), batch_size):
            batch = npc_ids[i : i + batch_size]
            batch_tasks = []

            for npc_id in batch:
                other_npcs = [npc_names_dict.get(o, f"NPC_{o}") for o in npc_ids if o != npc_id]
                others_text = ", ".join(other_npcs) if other_npcs else "no one else"

                memory_text = (
                    f"The player {player_action.get('description','interacted with us')} "
                    f"while I was with {others_text}"
                )
                action_type = player_action.get("type", "").lower()
                is_emotional = any(x in action_type for x in ["emotion", "challenge", "threaten", "mock", "praise"])

                batch_tasks.append(
                    memory_system.remember(
                        entity_type="npc",
                        entity_id=npc_id,
                        memory_text=memory_text,
                        importance="medium",
                        emotional=is_emotional,
                        tags=["group_interaction", "player_action"],
                    )
                )
            await asyncio.gather(*batch_tasks, return_exceptions=True)
            if i + batch_size < len(npc_ids):
                await asyncio.sleep(0.1)
    
    async def batch_update_npcs(
        self,
        npc_ids: List[int],
        update_type: str,
        update_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update multiple NPCs in a single batch operation for better performance.
        Thread-safe implementation using LoreSystem.
        """
        async with self._batch_update_lock:
            results = {
                "success_count": 0,
                "error_count": 0,
                "details": {},
            }
    
            try:
                # Get LoreSystem instance
                lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)
                
                # Create context for operations
                from agents import RunContextWrapper
                
                ctx = RunContextWrapper(context={
                    'user_id': self.user_id,
                    'conversation_id': self.conversation_id
                })

    
                # ------------------------------------------------------------------
                # LOCATION CHANGE
                # ------------------------------------------------------------------
                if update_type == "location_change":
                    new_location = update_data.get("new_location")
                    if not new_location:
                        return {"error": "No location specified"}
    
                    # First ensure location exists canonically
                    async with get_db_connection_context() as conn:
                        location_name = await canon.find_or_create_location(ctx, conn, new_location)
    
                    # Process each NPC through LoreSystem
                    for npc_id in npc_ids:
                        try:
                            result = await lore_system.propose_and_enact_change(
                                ctx=ctx,
                                entity_type="NPCStats",
                                entity_identifier={"npc_id": npc_id},
                                updates={"current_location": location_name},
                                reason=f"NPC moved to {location_name}"
                            )
                            
                            if result.get("status") == "committed":
                                results["success_count"] += 1
                                results["details"][npc_id] = {"success": True}
                            else:
                                results["error_count"] += 1
                                results["details"][npc_id] = {"error": result.get("message", "Unknown error")}
                        except Exception as e:
                            results["error_count"] += 1
                            results["details"][npc_id] = {"error": str(e)}
    
                # ------------------------------------------------------------------
                # TRAIT UPDATE
                # ------------------------------------------------------------------
                elif update_type == "trait_update":
                    traits = update_data.get("traits", {})
                    if not traits:
                        return {"error": "No traits specified for update"}
    
                    # Use LoreSystem for each NPC
                    for npc_id in npc_ids:
                        try:
                            result = await lore_system.propose_and_enact_change(
                                ctx=ctx,
                                entity_type="NPCStats",
                                entity_identifier={"npc_id": npc_id},
                                updates=traits,
                                reason=f"Batch trait update: {', '.join(traits.keys())}"
                            )
                            
                            if result.get("status") == "committed":
                                results["success_count"] += 1
                                results["details"][npc_id] = {"success": True}
                            else:
                                results["error_count"] += 1
                                results["details"][npc_id] = {"error": result.get("message", "Unknown error")}
                        except Exception as e:
                            results["error_count"] += 1
                            results["details"][npc_id] = {"error": str(e)}
    
                # ------------------------------------------------------------------
                # EMOTIONAL UPDATE
                # ------------------------------------------------------------------
                elif update_type == "emotional_update":
                    emotion = update_data.get("emotion")
                    intensity = update_data.get("intensity", 0.5)
                    if not emotion:
                        return {"error": "No emotion specified"}
    
                    memory_system = await self._get_memory_system()
    
                    # Process in batches
                    batch_size = 5
                    for i in range(0, len(npc_ids), batch_size):
                        batch = npc_ids[i : i + batch_size]
    
                        batch_tasks = []
                        for npc_id in batch:
                            task = memory_system.update_npc_emotion(
                                npc_id=npc_id,
                                emotion=emotion,
                                intensity=intensity
                            )
                            batch_tasks.append(task)
    
                        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
    
                        # Process results
                        for npc_id, result in zip(batch, batch_results):
                            if isinstance(result, Exception):
                                results["error_count"] += 1
                                results["details"][npc_id] = {"error": str(result)}
                            else:
                                results["success_count"] += 1
                                results["details"][npc_id] = {"success": True}
                                # Invalidate cached emotional state
                                async with self._emotional_state_lock:
                                    if npc_id in self._emotional_states:
                                        del self._emotional_states[npc_id]
                                    if npc_id in self._emotional_states_timestamps:
                                        del self._emotional_states_timestamps[npc_id]
    
                        # Small delay between batches
                        if i + batch_size < len(npc_ids):
                            await asyncio.sleep(0.1)
    
                # ------------------------------------------------------------------
                # MASK UPDATE
                # ------------------------------------------------------------------
                elif update_type == "mask_update":
                    mask_action = update_data.get("action")
                    if not mask_action:
                        return {"error": "No mask action specified"}
    
                    memory_system = await self._get_memory_system()
    
                    if mask_action == "reveal_trait":
                        trait = update_data.get("trait")
                        trigger = update_data.get("trigger", "forced revelation")
                        severity = update_data.get("severity", 1)
                        if not trait:
                            return {"error": "No trait specified for reveal"}
    
                        batch_size = 5
                        for i in range(0, len(npc_ids), batch_size):
                            batch = npc_ids[i : i + batch_size]
                            batch_tasks = []
    
                            for npc_id in batch:
                                task = memory_system.reveal_npc_trait(
                                    npc_id=npc_id,
                                    trigger=trigger,
                                    trait=trait,
                                    severity=severity
                                )
                                batch_tasks.append(task)
    
                            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
    
                            # Process results
                            for npc_id, result in zip(batch, batch_results):
                                if isinstance(result, Exception):
                                    results["error_count"] += 1
                                    results["details"][npc_id] = {"error": str(result)}
                                else:
                                    results["success_count"] += 1
                                    results["details"][npc_id] = {"success": True}
                                    # Invalidate cached mask
                                    async with self._mask_state_lock:
                                        if npc_id in self._mask_states:
                                            del self._mask_states[npc_id]
                                        if npc_id in self._mask_states_timestamps:
                                            del self._mask_states_timestamps[npc_id]
    
                            # Small delay between batches
                            if i + batch_size < len(npc_ids):
                                await asyncio.sleep(0.1)
    
                    elif mask_action == "adjust_integrity":
                        value = update_data.get("value")
                        absolute = update_data.get("absolute", False)
                        if value is None:
                            return {"error": "No value specified for mask integrity adjustment"}
    
                        try:
                            async with get_db_connection_context() as conn:
                                async with conn.cursor() as cursor:
                                    await cursor.execute("BEGIN")
                                    updated_ids = []
    
                                    if absolute:
                                        for npc_id in npc_ids:
                                            await cursor.execute(
                                                """
                                                UPDATE NPCStats
                                                SET mask_integrity = %s
                                                WHERE npc_id = %s
                                                  AND user_id = %s
                                                  AND conversation_id = %s
                                                RETURNING npc_id
                                                """,
                                                (value, npc_id, self.user_id, self.conversation_id)
                                            )
                                            row = await cursor.fetchone()
                                            if row:
                                                updated_ids.append(row[0])
                                    else:
                                        # Relative adjustment
                                        for npc_id in npc_ids:
                                            await cursor.execute(
                                                """
                                                UPDATE NPCStats
                                                SET mask_integrity = GREATEST(0, LEAST(100, mask_integrity + %s))
                                                WHERE npc_id = %s
                                                  AND user_id = %s
                                                  AND conversation_id = %s
                                                RETURNING npc_id
                                                """,
                                                (value, npc_id, self.user_id, self.conversation_id)
                                            )
                                            row = await cursor.fetchone()
                                            if row:
                                                updated_ids.append(row[0])
    
                                    results["success_count"] = len(updated_ids)
                                    results["updated_npcs"] = updated_ids
    
                                    await cursor.execute("COMMIT")
    
                                    # Invalidate cached masks
                                    async with self._mask_state_lock:
                                        for npc_id in updated_ids:
                                            if npc_id in self._mask_states:
                                                del self._mask_states[npc_id]
                                            if npc_id in self._mask_states_timestamps:
                                                del self._mask_states_timestamps[npc_id]
    
                        except Exception as e:
                            logger.error(f"Error updating mask integrity: {e}")
                            results["error"] = str(e)
                            results["error_count"] = len(npc_ids)
    
                            # Attempt to rollback transaction
                            try:
                                async with get_db_connection_context() as conn:
                                    async with conn.cursor() as cursor:
                                        await cursor.execute("ROLLBACK")
                            except Exception as rollback_error:
                                logger.error(f"Error rolling back transaction: {rollback_error}")
    
                    else:
                        return {"error": f"Unknown mask action: {mask_action}"}
    
                # ------------------------------------------------------------------
                # RELATIONSHIP UPDATE
                # ------------------------------------------------------------------
                elif update_type == "relationship_update":
                    target_npc_ids = update_data.get("target_npc_ids", [])
                    link_type = update_data.get("link_type")
                    link_level = update_data.get("link_level")
                    adjustment = update_data.get("adjustment")
    
                    if not target_npc_ids:
                        return {"error": "No target NPCs specified"}
                    if (link_level is None and adjustment is None) or link_type is None:
                        return {"error": "Must specify link_type and either link_level or adjustment"}
    
                    try:
                        async with get_db_connection_context() as conn:
                            async with conn.cursor() as cursor:
                                await cursor.execute("BEGIN")
                                updated_pairs = []
    
                                for npc_id in npc_ids:
                                    # Process target NPCs in smaller batches
                                    batch_size = 5
                                    for i in range(0, len(target_npc_ids), batch_size):
                                        batch_targets = target_npc_ids[i : i + batch_size]
    
                                        for target_id in batch_targets:
                                            if npc_id == target_id:
                                                # Skip self-relationship
                                                continue
    
                                            # Check if relationship exists
                                            await cursor.execute(
                                                """
                                                SELECT link_level
                                                FROM SocialLinks
                                                WHERE user_id = %s
                                                  AND conversation_id = %s
                                                  AND (
                                                    (entity1_type = 'npc' AND entity1_id = %s
                                                     AND entity2_type = 'npc' AND entity2_id = %s)
                                                    OR
                                                    (entity1_type = 'npc' AND entity1_id = %s
                                                     AND entity2_type = 'npc' AND entity2_id = %s)
                                                  )
                                                """,
                                                (
                                                    self.user_id,
                                                    self.conversation_id,
                                                    npc_id,
                                                    target_id,
                                                    target_id,
                                                    npc_id
                                                )
                                            )
                                            row = await cursor.fetchone()
    
                                            if row:
                                                # Update existing relationship
                                                current_level = row[0]
                                                if link_level is not None:
                                                    new_level = link_level
                                                else:
                                                    new_level = max(0, min(100, current_level + adjustment))
    
                                                await cursor.execute(
                                                    """
                                                    UPDATE SocialLinks
                                                    SET link_type = %s, link_level = %s
                                                    WHERE user_id = %s
                                                      AND conversation_id = %s
                                                      AND (
                                                        (entity1_type = 'npc' AND entity1_id = %s
                                                         AND entity2_type = 'npc' AND entity2_id = %s)
                                                        OR
                                                        (entity1_type = 'npc' AND entity1_id = %s
                                                         AND entity2_type = 'npc' AND entity2_id = %s)
                                                      )
                                                    """,
                                                    (
                                                        link_type,
                                                        new_level,
                                                        self.user_id,
                                                        self.conversation_id,
                                                        npc_id,
                                                        target_id,
                                                        target_id,
                                                        npc_id
                                                    )
                                                )
                                            else:
                                                # Create new relationship
                                                new_level = link_level if link_level is not None else 50
                                                await cursor.execute(
                                                    """
                                                    INSERT INTO SocialLinks
                                                    (user_id, conversation_id,
                                                     entity1_type, entity1_id,
                                                     entity2_type, entity2_id,
                                                     link_type, link_level)
                                                    VALUES (%s, %s, 'npc', %s, 'npc', %s, %s, %s)
                                                    """,
                                                    (
                                                        self.user_id,
                                                        self.conversation_id,
                                                        npc_id,
                                                        target_id,
                                                        link_type,
                                                        new_level
                                                    )
                                                )
    
                                            updated_pairs.append((npc_id, target_id))
    
                                        # Small delay between batch target updates
                                        if i + batch_size < len(target_npc_ids):
                                            await asyncio.sleep(0.05)
    
                                await cursor.execute("COMMIT")
    
                                results["success_count"] = len(updated_pairs)
                                results["updated_relationships"] = updated_pairs
    
                    except Exception as e:
                        logger.error(f"Error updating relationships: {e}")
                        results["error"] = str(e)
                        results["error_count"] = len(npc_ids) * len(target_npc_ids)
    
                        # Rollback on error
                        try:
                            async with get_db_connection_context() as conn:
                                async with conn.cursor() as cursor:
                                    await cursor.execute("ROLLBACK")
                        except Exception as rollback_error:
                            logger.error(f"Error rolling back transaction: {rollback_error}")
    
                # ------------------------------------------------------------------
                # BELIEF UPDATE
                # ------------------------------------------------------------------
                elif update_type == "belief_update":
                    belief_text = update_data.get("belief_text")
                    topic = update_data.get("topic", "player")
                    confidence = update_data.get("confidence", 0.7)
                    if not belief_text:
                        return {"error": "No belief text specified"}
    
                    memory_system = await self._get_memory_system()
    
                    batch_size = 5
                    for i in range(0, len(npc_ids), batch_size):
                        batch = npc_ids[i : i + batch_size]
                        batch_tasks = []
    
                        for npc_id in batch:
                            task = memory_system.create_belief(
                                entity_type="npc",
                                entity_id=npc_id,
                                belief_text=belief_text,
                                confidence=confidence,
                                topic=topic
                            )
                            batch_tasks.append(task)
    
                        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
    
                        # Process results
                        for npc_id, result in zip(batch, batch_results):
                            if isinstance(result, Exception):
                                results["error_count"] += 1
                                results["details"][npc_id] = {"error": str(result)}
                            else:
                                results["success_count"] += 1
                                results["details"][npc_id] = {
                                    "success": True,
                                    "belief_id": result.get("id")
                                }
    
                        # Small delay between batches
                        if i + batch_size < len(npc_ids):
                            await asyncio.sleep(0.1)
    
                # ------------------------------------------------------------------
                # TRAIT UPDATE
                # ------------------------------------------------------------------
                elif update_type == "trait_update":
                    traits = update_data.get("traits", {})
                    if not traits:
                        return {"error": "No traits specified for update"}
    
                    try:
                        async with get_db_connection_context() as conn:
                            async with conn.cursor() as cursor:
                                await cursor.execute("BEGIN")
    
                                update_fields = []
                                update_values = []
    
                                # Build dynamic update fields
                                for trait, value in traits.items():
                                    if trait in ["dominance", "cruelty", "mental_resilience"]:
                                        # Numeric trait
                                        update_fields.append(f"{trait} = %s")
                                        update_values.append(value)
                                    elif trait == "personality_traits":
                                        # JSON trait
                                        update_fields.append(f"{trait} = %s")
                                        if isinstance(value, (list, dict)):
                                            update_values.append(json.dumps(value))
                                        else:
                                            update_values.append(value)
    
                                if update_fields:
                                    batch_size = 10
                                    updated_ids = []
    
                                    for i in range(0, len(npc_ids), batch_size):
                                        batch_npcs = npc_ids[i : i + batch_size]
                                        for npc_id in batch_npcs:
                                            query = f"""
                                                UPDATE NPCStats
                                                SET {', '.join(update_fields)}
                                                WHERE npc_id = %s
                                                  AND user_id = %s
                                                  AND conversation_id = %s
                                                RETURNING npc_id
                                            """
                                            query_params = update_values + [npc_id, self.user_id, self.conversation_id]
                                            await cursor.execute(query, query_params)
    
                                            row = await cursor.fetchone()
                                            if row:
                                                updated_ids.append(row[0])
    
                                        # Small delay between batches
                                        if i + batch_size < len(npc_ids):
                                            await asyncio.sleep(0.05)
    
                                    results["success_count"] = len(updated_ids)
                                    results["updated_npcs"] = updated_ids
    
                                await cursor.execute("COMMIT")
    
                    except Exception as e:
                        logger.error(f"Error updating NPC traits: {e}")
                        results["error"] = str(e)
                        results["error_count"] = len(npc_ids)
    
                        # Rollback on error
                        try:
                            async with get_db_connection_context() as conn:
                                async with conn.cursor() as cursor:
                                    await cursor.execute("ROLLBACK")
                        except Exception as rollback_error:
                            logger.error(f"Error rolling back transaction: {rollback_error}")
    
                # ------------------------------------------------------------------
                # MEMORY UPDATE
                # ------------------------------------------------------------------
                elif update_type == "memory_update":
                    memory_text = update_data.get("memory_text")
                    importance = update_data.get("importance", "medium")
                    emotional = update_data.get("emotional", False)
                    tags = update_data.get("tags", [])
    
                    if not memory_text:
                        return {"error": "No memory text specified"}
    
                    memory_system = await self._get_memory_system()
    
                    batch_size = 5
                    for i in range(0, len(npc_ids), batch_size):
                        batch = npc_ids[i : i + batch_size]
                        batch_tasks = []
    
                        for npc_id in batch:
                            task = memory_system.remember(
                                entity_type="npc",
                                entity_id=npc_id,
                                memory_text=memory_text,
                                importance=importance,
                                emotional=emotional,
                                tags=tags
                            )
                            batch_tasks.append(task)
    
                        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
    
                        # Process results
                        for npc_id, result in zip(batch, batch_results):
                            if isinstance(result, Exception):
                                results["error_count"] += 1
                                results["details"][npc_id] = {"error": str(result)}
                            else:
                                results["success_count"] += 1
                                results["details"][npc_id] = {
                                    "success": True,
                                    "memory_id": result.get("id")
                                }
    
                        # Small delay between batches
                        if i + batch_size < len(npc_ids):
                            await asyncio.sleep(0.1)
    
                # ------------------------------------------------------------------
                # SCHEDULE UPDATE
                # ------------------------------------------------------------------
                elif update_type == "schedule_update":
                    schedule_data = update_data.get("schedule_data")
                    time_period = update_data.get("time_period")  # "morning", "evening", "default", etc.
    
                    if not schedule_data:
                        return {"error": "No schedule data specified"}
    
                    try:
                        async with get_db_connection_context() as conn:
                            async with conn.cursor() as cursor:
                                await cursor.execute("BEGIN")
                                batch_size = 10
                                updated_ids = []
    
                                for i in range(0, len(npc_ids), batch_size):
                                    batch_npcs = npc_ids[i : i + batch_size]
    
                                    for npc_id in batch_npcs:
                                        # Get current schedule
                                        await cursor.execute(
                                            """
                                            SELECT schedule
                                            FROM NPCStats
                                            WHERE npc_id = %s
                                              AND user_id = %s
                                              AND conversation_id = %s
                                            """,
                                            (npc_id, self.user_id, self.conversation_id)
                                        )
                                        row = await cursor.fetchone()
                                        if not row:
                                            continue
    
                                        current_schedule = row[0]
                                        if current_schedule is None:
                                            current_schedule = {}
                                        elif isinstance(current_schedule, str):
                                            try:
                                                current_schedule = json.loads(current_schedule)
                                            except json.JSONDecodeError:
                                                current_schedule = {}
    
                                        if not isinstance(current_schedule, dict):
                                            current_schedule = {}
    
                                        # Update a specific time period or replace entire schedule
                                        if time_period:
                                            current_schedule[time_period] = schedule_data
                                        else:
                                            current_schedule = schedule_data
    
                                        # Save updated schedule
                                        await cursor.execute(
                                            """
                                            UPDATE NPCStats
                                            SET schedule = %s
                                            WHERE npc_id = %s
                                              AND user_id = %s
                                              AND conversation_id = %s
                                            RETURNING npc_id
                                            """,
                                            (
                                                json.dumps(current_schedule),
                                                npc_id,
                                                self.user_id,
                                                self.conversation_id
                                            )
                                        )
                                        row = await cursor.fetchone()
                                        if row:
                                            updated_ids.append(row[0])
    
                                    # Small delay between batches
                                    if i + batch_size < len(npc_ids):
                                        await asyncio.sleep(0.05)
    
                                await cursor.execute("COMMIT")
    
                                results["success_count"] = len(updated_ids)
                                results["updated_npcs"] = updated_ids
    
                    except Exception as e:
                        logger.error(f"Error updating NPC schedules: {e}")
                        results["error"] = str(e)
                        results["error_count"] = len(npc_ids)
    
                        # Rollback on error
                        try:
                            async with get_db_connection_context() as conn:
                                async with conn.cursor() as cursor:
                                    await cursor.execute("ROLLBACK")
                        except Exception as rollback_error:
                            logger.error(f"Error rolling back transaction: {rollback_error}")
    
                # ------------------------------------------------------------------
                # STATUS UPDATE
                # ------------------------------------------------------------------
                elif update_type == "status_update":
                    status_field = update_data.get("field")
                    status_value = update_data.get("value")
    
                    if status_field is None or status_value is None:
                        return {"error": "Must specify status field and value"}
    
                    allowed_fields = ["introduced", "active", "visible"]
                    if status_field not in allowed_fields:
                        return {"error": f"Invalid status field. Allowed: {allowed_fields}"}
    
                    try:
                        async with get_db_connection_context() as conn:
                            async with conn.cursor() as cursor:
                                await cursor.execute("BEGIN")
                                batch_size = 20
                                updated_ids = []
    
                                for i in range(0, len(npc_ids), batch_size):
                                    batch_npcs = npc_ids[i : i + batch_size]
    
                                    for npc_id in batch_npcs:
                                        await cursor.execute(
                                            f"""
                                            UPDATE NPCStats
                                            SET {status_field} = %s
                                            WHERE npc_id = %s
                                              AND user_id = %s
                                              AND conversation_id = %s
                                            RETURNING npc_id
                                            """,
                                            (status_value, npc_id, self.user_id, self.conversation_id)
                                        )
                                        row = await cursor.fetchone()
                                        if row:
                                            updated_ids.append(row[0])
    
                                    # Small delay between batches
                                    if i + batch_size < len(npc_ids):
                                        await asyncio.sleep(0.05)
    
                                await cursor.execute("COMMIT")
                                results["success_count"] = len(updated_ids)
                                results["updated_npcs"] = updated_ids
    
                    except Exception as e:
                        logger.error(f"Error updating NPC status: {e}")
                        results["error"] = str(e)
                        results["error_count"] = len(npc_ids)
    
                        # Rollback on error
                        try:
                            async with get_db_connection_context() as conn:
                                async with conn.cursor() as cursor:
                                    await cursor.execute("ROLLBACK")
                        except Exception as rollback_error:
                            logger.error(f"Error rolling back transaction: {rollback_error}")
    
                # ------------------------------------------------------------------
                # APPEARANCE UPDATE
                # ------------------------------------------------------------------
                elif update_type == "appearance_update":
                    appearance_data = update_data.get("appearance_data")
                    if not appearance_data:
                        return {"error": "No appearance data specified"}
    
                    try:
                        async with get_db_connection_context() as conn:
                            async with conn.cursor() as cursor:
                                await cursor.execute("BEGIN")
    
                                # Convert dict to JSON if needed
                                if isinstance(appearance_data, dict):
                                    appearance_json = json.dumps(appearance_data)
                                else:
                                    appearance_json = appearance_data
    
                                batch_size = 20
                                updated_ids = []
                                for i in range(0, len(npc_ids), batch_size):
                                    batch_npcs = npc_ids[i : i + batch_size]
    
                                    for npc_id in batch_npcs:
                                        await cursor.execute(
                                            """
                                            UPDATE NPCStats
                                            SET appearance = %s
                                            WHERE npc_id = %s
                                              AND user_id = %s
                                              AND conversation_id = %s
                                            RETURNING npc_id
                                            """,
                                            (
                                                appearance_json,
                                                npc_id,
                                                self.user_id,
                                                self.conversation_id
                                            )
                                        )
                                        row = await cursor.fetchone()
                                        if row:
                                            updated_ids.append(row[0])
    
                                    # Small delay between batches
                                    if i + batch_size < len(npc_ids):
                                        await asyncio.sleep(0.05)
    
                                await cursor.execute("COMMIT")
                                results["success_count"] = len(updated_ids)
                                results["updated_npcs"] = updated_ids
    
                    except Exception as e:
                        logger.error(f"Error updating NPC appearance: {e}")
                        results["error"] = str(e)
                        results["error_count"] = len(npc_ids)
    
                        # Rollback on error
                        try:
                            async with get_db_connection_context() as conn:
                                async with conn.cursor() as cursor:
                                    await cursor.execute("ROLLBACK")
                        except Exception as rollback_error:
                            logger.error(f"Error rolling back transaction: {rollback_error}")
    
                # ------------------------------------------------------------------
                # UNKNOWN UPDATE TYPE
                # ------------------------------------------------------------------
                else:
                    return {
                        "error": f"Unknown update type: {update_type}",
                        "valid_types": [
                            "location_change", "emotional_update", "mask_update",
                            "relationship_update", "belief_update", "trait_update",
                            "memory_update", "schedule_update", "status_update",
                            "appearance_update"
                        ]
                    }
    
                return results
    
            except Exception as e:
                logger.error(f"Error in batch update: {e}")
                return {
                    "error": str(e),
                    "success_count": 0,
                    "error_count": len(npc_ids)
                }
