# logic/npc_agents/agent_coordinator.py

"""
Coordinates multiple NPC agents for group interactions, using OpenAI Agents SDK.
"""

import logging
import asyncio
import random
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pydantic import BaseModel

from agents import Agent, Runner, function_tool, handoff, trace, InputGuardrail, RunContextWrapper, GuardrailFunctionOutput
from db.connection import get_db_connection
from memory.wrapper import MemorySystem
from .npc_agent import NPCAgent, NPCAction

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
        self.active_agents: Dict[int, NPCAgent] = {}  # Map of npc_id -> NPCAgent
        self._memory_system = None
        self._coordinator_agent = None
        
        # Cache systems to reduce repeated queries
        self._emotional_states = {}  # Cache of emotional states to avoid repeated queries
        self._emotional_states_timestamps = {}  # When the states were last updated
        self._mask_states = {}       # Cache of mask states to avoid repeated queries
        self._mask_states_timestamps = {}  # When the states were last updated
        
        # Cache TTL settings
        self._cache_ttl = {
            "emotional_state": 120,  # 2 minutes in seconds
            "mask": 300,             # 5 minutes in seconds
        }

    async def _get_memory_system(self):
        """Lazy-load the memory system."""
        if self._memory_system is None:
            self._memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        return self._memory_system
    
    async def _get_coordinator_agent(self):
        """Lazy-load the coordinator agent."""
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
                model="gpt-4o",
                tools=[
                    function_tool(self._get_npc_emotional_state),
                    function_tool(self._get_npc_mask),
                    function_tool(self._get_npc_traits),
                    function_tool(self._get_relationships_between_npcs),
                    function_tool(self._create_group_memory)
                ],
                output_type=GroupDecisionOutput
            )
        return self._coordinator_agent

    async def load_agents(self, npc_ids: Optional[List[int]] = None) -> List[int]:
        """
        Load specified NPC agents into memory, or load all if none specified.

        Returns:
            List of NPC IDs that were successfully loaded.
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
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()

                for row in rows:
                    npc_id = row[0]
                    if npc_id not in self.active_agents:
                        self.active_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
                        # Initialize the agent
                        await self.active_agents[npc_id].initialize()
                    loaded_ids.append(npc_id)

            logger.info("Loaded agents: %s", loaded_ids)
            return loaded_ids
        except Exception as e:
            logger.error(f"Error loading agents: {e}")
            return []
    
    @function_tool
    async def _get_npc_emotional_state(self, npc_id: int) -> Dict[str, Any]:
        """
        Get an NPC's current emotional state, with caching for performance.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Emotional state dictionary
        """
        now = datetime.now()
        
        # Check cache first
        if npc_id in self._emotional_states:
            timestamp = self._emotional_states_timestamps.get(npc_id)
            if timestamp and (now - timestamp).total_seconds() < self._cache_ttl["emotional_state"]:
                return self._emotional_states[npc_id]
            
        try:
            memory_system = await self._get_memory_system()
            emotional_state = await memory_system.get_npc_emotion(npc_id)
            
            # Cache the result
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
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Mask information dictionary
        """
        now = datetime.now()
        
        # Check cache first
        if npc_id in self._mask_states:
            timestamp = self._mask_states_timestamps.get(npc_id)
            if timestamp and (now - timestamp).total_seconds() < self._cache_ttl["mask"]:
                return self._mask_states[npc_id]
            
        try:
            memory_system = await self._get_memory_system()
            mask_info = await memory_system.get_npc_mask(npc_id)
            
            # Cache the result
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
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Dictionary with NPC traits
        """
        try:
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute("""
                    SELECT npc_name, dominance, cruelty, personality_traits 
                    FROM NPCStats
                    WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                """, (npc_id, self.user_id, self.conversation_id))
                
                row = cursor.fetchone()
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
                    "personality_traits": personality_traits
                }
        except Exception as e:
            logger.error(f"Error getting NPC traits for {npc_id}: {e}")
            return {"error": str(e)}
    
    @function_tool
    async def _get_relationships_between_npcs(self, npc_ids: List[int]) -> Dict[str, Any]:
        """
        Get relationship information between a group of NPCs.
        
        Args:
            npc_ids: List of NPC IDs
            
        Returns:
            Dictionary mapping NPC pairs to relationship information
        """
        if not npc_ids or len(npc_ids) < 2:
            return {}
            
        relationships = {}
        
        try:
            with get_db_connection() as conn, conn.cursor() as cursor:
                for i, npc1 in enumerate(npc_ids):
                    for npc2 in npc_ids[i+1:]:
                        cursor.execute("""
                            SELECT link_type, link_level 
                            FROM SocialLinks
                            WHERE user_id = %s AND conversation_id = %s
                              AND ((entity1_type = 'npc' AND entity1_id = %s AND entity2_type = 'npc' AND entity2_id = %s)
                                OR (entity1_type = 'npc' AND entity1_id = %s AND entity2_type = 'npc' AND entity2_id = %s))
                        """, (self.user_id, self.conversation_id, npc1, npc2, npc2, npc1))
                        
                        row = cursor.fetchone()
                        if row:
                            link_type, link_level = row
                            key = f"{min(npc1, npc2)}_{max(npc1, npc2)}"
                            relationships[key] = {
                                "npc1": npc1,
                                "npc2": npc2,
                                "link_type": link_type,
                                "link_level": link_level
                            }
                        else:
                            # No established relationship
                            key = f"{min(npc1, npc2)}_{max(npc1, npc2)}"
                            relationships[key] = {
                                "npc1": npc1,
                                "npc2": npc2,
                                "link_type": "neutral",
                                "link_level": 50
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
        tags: List[str] = ["group_interaction"]
    ) -> Dict[str, Any]:
        """
        Create a memory of a group interaction for all participating NPCs.
        
        Args:
            npc_ids: List of NPC IDs
            memory_text: Text of the memory
            importance: Importance level (low, medium, high)
            tags: Tags for the memory
            
        Returns:
            Status of memory creation
        """
        memory_system = await self._get_memory_system()
        results = {}
        
        for npc_id in npc_ids:
            try:
                memory = await memory_system.remember(
                    entity_type="npc",
                    entity_id=npc_id,
                    memory_text=memory_text,
                    importance=importance,
                    tags=tags
                )
                results[npc_id] = {"status": "success", "memory_id": memory.get("id")}
            except Exception as e:
                logger.error(f"Error creating memory for NPC {npc_id}: {e}")
                results[npc_id] = {"status": "error", "message": str(e)}
        
        return results
    
    async def make_group_decisions(
        self,
        npc_ids: List[int],
        shared_context: Dict[str, Any],
        available_actions: Optional[Dict[int, List[Dict[str, Any]]]] = None
    ) -> Dict[str, Any]:
        """
        Coordinate decision-making for a group of NPCs using the Agents SDK.
        
        Args:
            npc_ids: List of NPC IDs
            shared_context: Common context for all NPCs
            available_actions: Optional pre-defined actions for each NPC
            
        Returns:
            Dictionary with group action plan
        """
        # Ensure all NPCs are loaded
        await self.load_agents(npc_ids)
        
        # Get the coordinator agent
        coordinator_agent = await self._get_coordinator_agent()
        
        # 1. Prepare enhanced group context with memory integration
        enhanced_context = await self._prepare_group_context(npc_ids, shared_context)
        
        # 2. If actions are not provided, generate them
        if available_actions is None:
            available_actions = await self.generate_group_actions(npc_ids, enhanced_context)
        
        # 3. Prepare input for the coordinator agent
        input_data = {
            "context": enhanced_context,
            "npc_ids": npc_ids,
            "available_actions": available_actions
        }
        
        # 4. Create trace for debugging
        with trace(
            f"group_decision_{self.user_id}_{self.conversation_id}", 
            group_id=f"user_{self.user_id}_conv_{self.conversation_id}"
        ):
            # 5. Run the coordinator agent to make group decisions
            result = await Runner.run(coordinator_agent, input_data)
            
            # 6. Process the result
            output = result.final_output_as(GroupDecisionOutput)
            
            # 7. Create memories for all NPCs based on the decision
            location = enhanced_context.get("location", "Unknown")
            memory_text = f"I participated in a group interaction at {location} with {len(npc_ids)} others"
            
            await self._create_group_memory(
                npc_ids=npc_ids,
                memory_text=memory_text,
                importance="medium",
                tags=["group_interaction", "group_decision"]
            )
            
            # 8. Return the action plan
            return {
                "group_actions": output.group_actions,
                "individual_actions": output.individual_actions,
                "reasoning": output.reasoning,
                "context": enhanced_context
            }
    
    async def _prepare_group_context(self, npc_ids: List[int], shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare enhanced context for group interactions with memory.
        
        Args:
            npc_ids: List of NPC IDs
            shared_context: Base context to enhance
            
        Returns:
            Enhanced context with memory integration
        """
        memory_system = await self._get_memory_system()
        
        # Create enhanced context
        enhanced_context = shared_context.copy()
        enhanced_context["participants"] = npc_ids
        enhanced_context["type"] = "group_interaction"
        
        # Add location if not present
        if "location" not in enhanced_context:
            try:
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT current_location
                        FROM NPCStats
                        WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                    """, (npc_ids[0], self.user_id, self.conversation_id))
                    row = cursor.fetchone()
                    if row and row[0]:
                        enhanced_context["location"] = row[0]
            except Exception as e:
                logger.error(f"Error getting location for context: {e}")
        
        # Add time if not present
        if "time_of_day" not in enhanced_context:
            try:
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT value
                        FROM CurrentRoleplay
                        WHERE user_id = %s AND conversation_id = %s AND key = 'TimeOfDay'
                    """, (self.user_id, self.conversation_id))
                    row = cursor.fetchone()
                    if row:
                        enhanced_context["time_of_day"] = row[0]
            except Exception as e:
                logger.error(f"Error getting time for context: {e}")
        
        # Add NPC-specific context information
        if "npc_context" not in enhanced_context:
            enhanced_context["npc_context"] = {}
        
        # Process each NPC in parallel for better performance
        tasks = []
        for npc_id in npc_ids:
            tasks.append(self._prepare_single_npc_context(npc_id, npc_ids, enhanced_context))
        
        npc_contexts = await asyncio.gather(*tasks)
        
        # Store in enhanced context
        for npc_context in npc_contexts:
            npc_id = npc_context.pop("npc_id")
            enhanced_context["npc_context"][npc_id] = npc_context
        
        # Add shared group memories
        shared_memories = await memory_system.recall(
            entity_type="npc",
            entity_id=npc_ids[0],  # Use first NPC as reference
            query="group interaction",
            context={"location": enhanced_context.get("location", "Unknown")},
            limit=2
        )
        
        enhanced_context["shared_history"] = shared_memories.get("memories", [])
        
        return enhanced_context
    
    async def _prepare_single_npc_context(
        self, 
        npc_id: int, 
        group_npc_ids: List[int],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare context for a single NPC within a group.
        
        Args:
            npc_id: The NPC ID
            group_npc_ids: All NPCs in the group
            context: Shared context
            
        Returns:
            NPC-specific context
        """
        memory_system = await self._get_memory_system()
        
        # Get previous group memories
        group_memories = await memory_system.recall(
            entity_type="npc",
            entity_id=npc_id,
            query="group interaction",
            context=context,
            limit=3
        )
        
        # Check for flashback opportunity
        flashback = None
        if random.random() < 0.15:  # 15% chance of flashback in group setting
            context_text = f"group interaction at {context.get('location', 'Unknown')}"
            flashback = await memory_system.npc_flashback(npc_id, context_text)
        
        # Get NPC's emotional state
        emotional_state = await self._get_npc_emotional_state(npc_id)
        
        # Get NPC's mask status
        mask_info = await self._get_npc_mask(npc_id)
        
        # Get NPC's traits
        traits = await self._get_npc_traits(npc_id)
        
        # Get NPC's beliefs about other NPCs in the group
        beliefs = {}
        for other_id in group_npc_ids:
            if other_id != npc_id:
                npc_beliefs = await memory_system.get_beliefs(
                    entity_type="npc",
                    entity_id=npc_id,
                    topic=f"npc_{other_id}"
                )
                if npc_beliefs:
                    beliefs[other_id] = npc_beliefs
        
        return {
            "npc_id": npc_id,
            "group_memories": group_memories.get("memories", []),
            "emotional_state": emotional_state,
            "mask_info": mask_info,
            "traits": traits,
            "flashback": flashback,
            "beliefs": beliefs
        }
    
    async def generate_group_actions(
        self,
        npc_ids: List[int],
        context: Dict[str, Any]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Generate possible actions for each NPC in a group context.
        Enhanced with memory-based influences.

        Args:
            npc_ids: IDs of the NPCs
            context: Enhanced context with perceptions

        Returns:
            A dict of npc_id -> list of NPCAction objects
        """
        group_actions: Dict[int, List[Dict[str, Any]]] = {}
        
        # Get memory system
        memory_system = await self._get_memory_system()
        
        # Process each NPC
        for npc_id in npc_ids:
            # Get NPC traits
            npc_traits = await self._get_npc_traits(npc_id)
            
            if "error" in npc_traits:
                continue
                
            dom = npc_traits.get("dominance", 50)
            cru = npc_traits.get("cruelty", 50)
            name = npc_traits.get("npc_name", f"NPC_{npc_id}")
            
            # Basic actions available to all NPCs
            actions = [
                {
                    "type": "talk",
                    "description": "Talk to the group",
                    "target": "group"
                },
                {
                    "type": "observe",
                    "description": "Observe the group",
                    "target": "group"
                },
                {
                    "type": "leave",
                    "description": "Leave the group",
                    "target": "group"
                }
            ]
            
            # Get emotional state to influence available actions
            emotional_state = await self._get_npc_emotional_state(npc_id)
            
            # Get NPC's mask information
            mask_info = await self._get_npc_mask(npc_id)
            
            # Get beliefs that might influence actions
            beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=npc_id,
                topic="group_interaction"
            )
            
            # Add dominance-based actions
            if dom > 60:
                actions.append({
                    "type": "command",
                    "description": "Give an authoritative command",
                    "target": "group",
                    "stats_influenced": {"dominance": 1, "trust": -1}
                })
                actions.append({
                    "type": "test",
                    "description": "Test group's obedience",
                    "target": "group",
                    "stats_influenced": {"dominance": 2, "respect": -1}
                })
                
                if dom > 75:
                    actions.append({
                        "type": "dominate",
                        "description": "Assert dominance forcefully",
                        "target": "group",
                        "stats_influenced": {"dominance": 3, "fear": 2}
                    })
            
            # Add cruelty-based actions
            if cru > 60:
                actions.append({
                    "type": "mock",
                    "description": "Mock or belittle the group",
                    "target": "group",
                    "stats_influenced": {"cruelty": 1, "closeness": -2}
                })
                
                if cru > 70:
                    actions.append({
                        "type": "humiliate",
                        "description": "Deliberately humiliate the group",
                        "target": "group",
                        "stats_influenced": {"cruelty": 2, "fear": 2}
                    })
            
            # Add emotionally-influenced actions for strong emotions
            if emotional_state and "current_emotion" in emotional_state:
                current_emotion = emotional_state["current_emotion"]
                primary = current_emotion.get("primary", {})
                
                # Handle different data structures
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
                            "stats_influenced": {"dominance": 2, "closeness": -3}
                        })
                    elif emotion_name == "fear":
                        actions.append({
                            "type": "act_defensive",
                            "description": "Act defensively and guarded",
                            "target": "environment",
                            "stats_influenced": {"trust": -2}
                        })
                    elif emotion_name == "joy":
                        actions.append({
                            "type": "celebrate",
                            "description": "Share happiness enthusiastically",
                            "target": "group",
                            "stats_influenced": {"closeness": 3}
                        })
            
            # Add actions based on mask information
            mask_integrity = 100
            hidden_traits = {}
            
            if mask_info:
                mask_integrity = mask_info.get("integrity", 100)
                hidden_traits = mask_info.get("hidden_traits", {})
            
            if mask_integrity < 70:
                # As mask breaks down, hidden traits show through
                if isinstance(hidden_traits, dict):
                    for trait, value in hidden_traits.items():
                        if trait == "dominant" and value:
                            actions.append({
                                "type": "mask_slip",
                                "description": "Show unexpected dominance",
                                "target": "group",
                                "stats_influenced": {"dominance": 3, "fear": 2}
                            })
                        elif trait == "cruel" and value:
                            actions.append({
                                "type": "mask_slip",
                                "description": "Reveal unexpected cruelty",
                                "target": "group",
                                "stats_influenced": {"cruelty": 2, "fear": 1}
                            })
                        elif trait == "submissive" and value:
                            actions.append({
                                "type": "mask_slip",
                                "description": "Show unexpected submission",
                                "target": "group",
                                "stats_influenced": {"dominance": -2}
                            })
                elif isinstance(hidden_traits, list):
                    if "dominant" in hidden_traits:
                        actions.append({
                            "type": "mask_slip",
                            "description": "Show unexpected dominance",
                            "target": "group",
                            "stats_influenced": {"dominance": 3, "fear": 2}
                        })
                    elif "cruel" in hidden_traits:
                        actions.append({
                            "type": "mask_slip",
                            "description": "Reveal unexpected cruelty",
                            "target": "group",
                            "stats_influenced": {"cruelty": 2, "fear": 1}
                        })
                    elif "submissive" in hidden_traits:
                        actions.append({
                            "type": "mask_slip",
                            "description": "Show unexpected submission",
                            "target": "group",
                            "stats_influenced": {"dominance": -2}
                        })
            
            # Add actions based on beliefs
            if beliefs:
                for belief in beliefs:
                    belief_text = belief.get("belief", "").lower()
                    if "dangerous" in belief_text or "threat" in belief_text:
                        actions.append({
                            "type": "defensive",
                            "description": "Take a defensive stance",
                            "target": "group",
                            "stats_influenced": {"trust": -2}
                        })
                    elif "opportunity" in belief_text or "beneficial" in belief_text:
                        actions.append({
                            "type": "engage",
                            "description": "Actively engage with the group",
                            "target": "group",
                            "stats_influenced": {"closeness": 2}
                        })
            
            # Context-based actions
            location = context.get("location", "").lower()
            
            if location and any(loc in location for loc in ["cafe", "restaurant", "bar", "party"]):
                actions.append({
                    "type": "socialize",
                    "description": "Engage in group conversation",
                    "target": "group",
                    "stats_influenced": {"closeness": 1}
                })
            
            # Add target-specific actions for other NPCs
            for other_id in npc_ids:
                if other_id != npc_id:
                    other_traits = await self._get_npc_traits(other_id)
                    if "error" in other_traits:
                        continue
                        
                    other_name = other_traits.get("npc_name", f"NPC_{other_id}")
                    
                    # Basic interaction
                    actions.append({
                        "type": "talk_to",
                        "description": f"Talk to {other_name}",
                        "target": str(other_id),
                        "target_name": other_name,
                        "stats_influenced": {"closeness": 1}
                    })
                    
                    # High dominance actions
                    if dom > 60:
                        actions.append({
                            "type": "command",
                            "description": f"Command {other_name}",
                            "target": str(other_id),
                            "target_name": other_name,
                            "stats_influenced": {"dominance": 1, "trust": -1}
                        })
                    
                    # High cruelty actions
                    if cru > 60:
                        actions.append({
                            "type": "mock",
                            "description": f"Mock {other_name}",
                            "target": str(other_id),
                            "target_name": other_name,
                            "stats_influenced": {"cruelty": 1, "closeness": -2}
                        })
            
            group_actions[npc_id] = actions

        return group_actions
    
    async def handle_player_action(
        self,
        player_action: Dict[str, Any],
        context: Dict[str, Any],
        npc_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Handle a player action directed at multiple NPCs, using the Agents SDK.
        
        Args:
            player_action: The player's action
            context: Additional context
            npc_ids: Optional list of specific NPC IDs to target
            
        Returns:
            Dictionary with NPC responses
        """
        # Define the guardrail agent to check for homework
        guardrail_agent = Agent(
            name="Content Guardrail",
            instructions="Check if the player is asking about homework or schoolwork. If they are asking for answers or solutions to schoolwork, this is not allowed.",
            output_type=HomeworkCheck
        )
        
        async def homework_guardrail(
            ctx: RunContextWrapper, 
            agent: Agent, 
            input_data: Dict[str, Any]
        ) -> GuardrailFunctionOutput:
            result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
            final_output = result.final_output_as(HomeworkCheck)
            return GuardrailFunctionOutput(
                output_info=final_output,
                tripwire_triggered=final_output.is_homework
            )
        
        # Create the main coordinator agent with guardrails
        coordinator = Agent(
            name="Player Action Coordinator",
            instructions="""
            You coordinate NPC responses to player actions.
            Consider each NPC's personality, emotional state, and relationships
            when determining how they should respond.
            """,
            input_guardrails=[InputGuardrail(guardrail_function=homework_guardrail)],
            tools=[function_tool(self._process_player_action_for_npcs)]
        )
        
        # Determine affected NPCs if not specified
        if npc_ids is None:
            npc_ids = await self._determine_affected_npcs(player_action, context)
        
        if not npc_ids:
            return {"npc_responses": []}
        
        # Prepare input for the coordinator
        input_data = {
            "player_action": player_action,
            "context": context,
            "npc_ids": npc_ids
        }
        
        # Create trace for debugging
        with trace(
            f"player_action_{self.user_id}_{self.conversation_id}", 
            group_id=f"user_{self.user_id}_conv_{self.conversation_id}"
        ):
            # Run the coordinator
            result = await Runner.run(coordinator, input_data)
            
            # Return the result
            return result.final_output
    
    @function_tool
    async def _process_player_action_for_npcs(
        self,
        player_action: Dict[str, Any],
        context: Dict[str, Any],
        npc_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Process a player action for multiple NPCs.
        
        Args:
            player_action: The player's action
            context: Additional context
            npc_ids: List of NPC IDs to process the action for
            
        Returns:
            Dictionary with NPC responses
        """
        # Ensure all NPCs are loaded
        await self.load_agents(npc_ids)
        
        # Enhanced context with group information
        enhanced_context = context.copy()
        enhanced_context["is_group_interaction"] = True
        enhanced_context["affected_npcs"] = npc_ids
        
        # Add memory-based context enhancements
        memory_system = await self._get_memory_system()
        
        # Get emotional states for all NPCs
        emotional_states = {}
        for npc_id in npc_ids:
            try:
                emotional_state = await self._get_npc_emotional_state(npc_id)
                if emotional_state:
                    emotional_states[npc_id] = emotional_state
            except Exception as e:
                logger.error(f"Error getting emotional state for NPC {npc_id}: {e}")
        
        enhanced_context["emotional_states"] = emotional_states
        
        # Get mask information for all NPCs
        mask_states = {}
        for npc_id in npc_ids:
            try:
                mask_info = await self._get_npc_mask(npc_id)
                if mask_info:
                    mask_states[npc_id] = mask_info
            except Exception as e:
                logger.error(f"Error getting mask info for NPC {npc_id}: {e}")
        
        enhanced_context["mask_states"] = mask_states
        
        # Process player action for each NPC concurrently
        response_tasks = []
        for npc_id in npc_ids:
            agent = self.active_agents.get(npc_id)
            if agent:
                # Pass the enhanced context to each agent
                npc_context = enhanced_context.copy()
                # Add NPC-specific context elements
                npc_context["emotional_state"] = emotional_states.get(npc_id)
                npc_context["mask_info"] = mask_states.get(npc_id)
                
                response_tasks.append(agent.process_player_action(player_action, npc_context))
            else:
                response_tasks.append(asyncio.sleep(0))

        responses = await asyncio.gather(*response_tasks)
        filtered_responses = [r for r in responses if r is not None]
        
        # Create memories of this group interaction
        if len(npc_ids) > 1:
            await self._create_player_group_interaction_memories(
                npc_ids,
                player_action,
                context
            )
        
        return {"npc_responses": filtered_responses}
    
    async def _determine_affected_npcs(
        self,
        player_action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[int]:
        """
        Determine which NPCs are affected by a player action.
        
        Args:
            player_action: The player's action
            context: Context information
            
        Returns:
            List of affected NPC IDs
        """
        if "target_npc_id" in player_action:
            return [player_action["target_npc_id"]]
            
        current_location = context.get("location", "Unknown")
        logger.debug("Determining NPCs at location=%s", current_location)
        
        try:
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute("""
                    SELECT npc_id
                    FROM NPCStats
                    WHERE user_id = %s
                      AND conversation_id = %s
                      AND current_location = %s
                """, (self.user_id, self.conversation_id, current_location))
                
                npc_list = [row[0] for row in cursor.fetchall()]
                return npc_list
        except Exception as e:
            logger.error(f"Error determining affected NPCs: {e}")
            return []
    
    async def _create_player_group_interaction_memories(
        self,
        npc_ids: List[int],
        player_action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> None:
        """
        Create memories of a group interaction with the player.
        
        Args:
            npc_ids: List of NPC IDs
            player_action: The player's action
            context: Context information
        """
        memory_system = await self._get_memory_system()
        
        # Get NPC names for better memory context
        npc_names_dict = {}
        for npc_id in npc_ids:
            traits = await self._get_npc_traits(npc_id)
            if "error" not in traits:
                npc_names_dict[npc_id] = traits.get("npc_name", f"NPC_{npc_id}")
            else:
                npc_names_dict[npc_id] = f"NPC_{npc_id}"
        
        # Create a memory of this group interaction for each NPC
        for npc_id in npc_ids:
            # Filter out this NPC from the participant list
            other_npcs = [npc_names_dict.get(other_id, f"NPC_{other_id}") for other_id in npc_ids if other_id != npc_id]
            others_text = ", ".join(other_npcs) if other_npcs else "no one else"
            
            memory_text = f"The player {player_action.get('description', 'interacted with us')} while I was with {others_text}"
            
            # Determine emotional impact
            action_type = player_action.get("type", "").lower()
            is_emotional = "emotion" in action_type or action_type in ["challenge", "threaten", "mock", "praise"]
            
            await memory_system.remember(
                entity_type="npc",
                entity_id=npc_id,
                memory_text=memory_text,
                importance="medium",
                emotional=is_emotional,
                tags=["group_interaction", "player_action"]
            )
    
    async def batch_update_npcs(
        self,
        npc_ids: List[int],
        update_type: str,
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update multiple NPCs in a single batch operation for better performance."""
        results = {
            "success_count": 0,
            "error_count": 0,
            "details": {}
        }
        
        # Ensure all NPCs are loaded
        await self.load_agents(npc_ids)
        
        # Process different update types
        if update_type == "location_change":
            # Batch location update
            new_location = update_data.get("new_location")
            if not new_location:
                return {"error": "No location specified"}
                
            try:
                with get_db_connection() as conn, conn.cursor() as cursor:
                    # Begin transaction
                    cursor.execute("BEGIN")
                    
                    # Update all NPCs in a single query
                    cursor.execute(
                        """
                        UPDATE NPCStats
                        SET current_location = %s
                        WHERE npc_id = ANY(%s)
                        AND user_id = %s
                        AND conversation_id = %s
                        RETURNING npc_id
                        """,
                        (new_location, npc_ids, self.user_id, self.conversation_id)
                    )
                    
                    rows = cursor.fetchall()
                    results["success_count"] = len(rows)
                    results["updated_npcs"] = [r[0] for r in rows]
                    
                    # Commit transaction
                    cursor.execute("COMMIT")
                    
            except Exception as e:
                logger.error(f"Error updating NPC locations: {e}")
                results["error"] = str(e)
                results["error_count"] = len(npc_ids)
                
                # Attempt to rollback transaction
                try:
                    with get_db_connection() as conn, conn.cursor() as cursor:
                        cursor.execute("ROLLBACK")
                except Exception as rollback_error:
                    logger.error(f"Error rolling back transaction: {rollback_error}")
                    
        elif update_type == "emotional_update":
            # Batch emotional state update
            emotion = update_data.get("emotion")
            intensity = update_data.get("intensity", 0.5)
            
            if not emotion:
                return {"error": "No emotion specified"}
                
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
                        emotion=emotion,
                        intensity=intensity
                    )
                    batch_tasks.append(task)
                    
                # Execute batch concurrently
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
                        if npc_id in self._emotional_states:
                            del self._emotional_states[npc_id]
                            del self._emotional_states_timestamps[npc_id]
        
        # Mask update - update mask integrity or reveal/hide traits
        elif update_type == "mask_update":
            mask_action = update_data.get("action")
            
            if not mask_action:
                return {"error": "No mask action specified"}
                
            # Get memory system
            memory_system = await self._get_memory_system()
            
            if mask_action == "reveal_trait":
                # Reveal a hidden trait
                trait = update_data.get("trait")
                trigger = update_data.get("trigger", "forced revelation")
                severity = update_data.get("severity", 1)
                
                if not trait:
                    return {"error": "No trait specified for reveal"}
                
                batch_tasks = []
                for npc_id in npc_ids:
                    task = memory_system.reveal_npc_trait(
                        npc_id=npc_id, 
                        trigger=trigger, 
                        trait=trait,
                        severity=severity
                    )
                    batch_tasks.append(task)
                    
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results
                for npc_id, result in zip(npc_ids, batch_results):
                    if isinstance(result, Exception):
                        results["error_count"] += 1
                        results["details"][npc_id] = {"error": str(result)}
                    else:
                        results["success_count"] += 1
                        results["details"][npc_id] = {"success": True}
                        
                        # Invalidate cached mask
                        if npc_id in self._mask_states:
                            del self._mask_states[npc_id]
                            del self._mask_states_timestamps[npc_id]
                            
            elif mask_action == "adjust_integrity":
                # Adjust mask integrity
                value = update_data.get("value")
                absolute = update_data.get("absolute", False)
                
                if value is None:
                    return {"error": "No value specified for mask integrity adjustment"}
                
                try:
                    with get_db_connection() as conn, conn.cursor() as cursor:
                        if absolute:
                            cursor.execute("""
                                UPDATE NPCStats
                                SET mask_integrity = %s
                                WHERE npc_id = ANY(%s)
                                AND user_id = %s
                                AND conversation_id = %s
                                RETURNING npc_id
                            """, (value, npc_ids, self.user_id, self.conversation_id))
                        else:
                            # Relative adjustment
                            cursor.execute("""
                                UPDATE NPCStats
                                SET mask_integrity = GREATEST(0, LEAST(100, mask_integrity + %s))
                                WHERE npc_id = ANY(%s)
                                AND user_id = %s
                                AND conversation_id = %s
                                RETURNING npc_id
                            """, (value, npc_ids, self.user_id, self.conversation_id))
                            
                        updated_ids = [row[0] for row in cursor.fetchall()]
                        results["success_count"] = len(updated_ids)
                        results["updated_npcs"] = updated_ids
                        
                        # Invalidate cached masks
                        for npc_id in updated_ids:
                            if npc_id in self._mask_states:
                                del self._mask_states[npc_id]
                                del self._mask_states_timestamps[npc_id]
                except Exception as e:
                    logger.error(f"Error updating mask integrity: {e}")
                    results["error"] = str(e)
                    results["error_count"] = len(npc_ids)
            
            else:
                return {"error": f"Unknown mask action: {mask_action}"}
                
        # Relationship update - update relationships between NPCs
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
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute("BEGIN")
                    
                    updated_pairs = []
                    for npc_id in npc_ids:
                        for target_id in target_npc_ids:
                            if npc_id == target_id:
                                continue  # Skip self-relationship
                                
                            # Check if relationship exists
                            cursor.execute("""
                                SELECT link_level
                                FROM SocialLinks
                                WHERE user_id = %s
                                AND conversation_id = %s
                                AND ((entity1_type = 'npc' AND entity1_id = %s AND entity2_type = 'npc' AND entity2_id = %s)
                                    OR (entity1_type = 'npc' AND entity1_id = %s AND entity2_type = 'npc' AND entity2_id = %s))
                            """, (self.user_id, self.conversation_id, npc_id, target_id, target_id, npc_id))
                            
                            row = cursor.fetchone()
                            
                            if row:
                                # Update existing relationship
                                current_level = row[0]
                                new_level = link_level if link_level is not None else max(0, min(100, current_level + adjustment))
                                
                                cursor.execute("""
                                    UPDATE SocialLinks
                                    SET link_type = %s, link_level = %s
                                    WHERE user_id = %s
                                    AND conversation_id = %s
                                    AND ((entity1_type = 'npc' AND entity1_id = %s AND entity2_type = 'npc' AND entity2_id = %s)
                                        OR (entity1_type = 'npc' AND entity1_id = %s AND entity2_type = 'npc' AND entity2_id = %s))
                                """, (link_type, new_level, self.user_id, self.conversation_id, 
                                      npc_id, target_id, target_id, npc_id))
                            else:
                                # Create new relationship
                                new_level = link_level if link_level is not None else 50  # Default level
                                
                                cursor.execute("""
                                    INSERT INTO SocialLinks
                                    (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id, link_type, link_level)
                                    VALUES (%s, %s, 'npc', %s, 'npc', %s, %s, %s)
                                """, (self.user_id, self.conversation_id, npc_id, target_id, link_type, new_level))
                                
                            updated_pairs.append((npc_id, target_id))
                            
                    cursor.execute("COMMIT")
                    
                    results["success_count"] = len(updated_pairs)
                    results["updated_relationships"] = updated_pairs
                    
            except Exception as e:
                logger.error(f"Error updating relationships: {e}")
                results["error"] = str(e)
                results["error_count"] = len(npc_ids) * len(target_npc_ids)
                
                # Rollback on error
                try:
                    with get_db_connection() as conn, conn.cursor() as cursor:
                        cursor.execute("ROLLBACK")
                except Exception as rollback_error:
                    logger.error(f"Error rolling back transaction: {rollback_error}")
        
        # Belief update - create or update beliefs
        elif update_type == "belief_update":
            belief_text = update_data.get("belief_text")
            topic = update_data.get("topic", "player")
            confidence = update_data.get("confidence", 0.7)
            
            if not belief_text:
                return {"error": "No belief text specified"}
                
            # Get memory system
            memory_system = await self._get_memory_system()
            
            batch_tasks = []
            for npc_id in npc_ids:
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
            for npc_id, result in zip(npc_ids, batch_results):
                if isinstance(result, Exception):
                    results["error_count"] += 1
                    results["details"][npc_id] = {"error": str(result)}
                else:
                    results["success_count"] += 1
                    results["details"][npc_id] = {"success": True, "belief_id": result.get("id")}
                    
        # Trait update - update NPC traits
        elif update_type == "trait_update":
            traits = update_data.get("traits", {})
            
            if not traits:
                return {"error": "No traits specified for update"}
                
            try:
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute("BEGIN")
                    
                    # Build dynamic update SQL
                    update_fields = []
                    update_values = []
                    
                    for trait, value in traits.items():
                        if trait in ["dominance", "cruelty", "mental_resilience"]:
                            # Numeric trait
                            update_fields.append(f"{trait} = %s")
                            update_values.append(value)
                        elif trait == "personality_traits":
                            # JSON trait - handle as array or object
                            update_fields.append(f"{trait} = %s")
                            
                            if isinstance(value, list) or isinstance(value, dict):
                                import json
                                update_values.append(json.dumps(value))
                            else:
                                update_values.append(value)
                    
                    if update_fields:
                        query = f"""
                            UPDATE NPCStats
                            SET {', '.join(update_fields)}
                            WHERE npc_id = ANY(%s)
                            AND user_id = %s
                            AND conversation_id = %s
                            RETURNING npc_id
                        """
                        
                        cursor.execute(
                            query, 
                            [*update_values, npc_ids, self.user_id, self.conversation_id]
                        )
                        
                        updated_ids = [row[0] for row in cursor.fetchall()]
                        results["success_count"] = len(updated_ids)
                        results["updated_npcs"] = updated_ids
                        
                    cursor.execute("COMMIT")
                    
            except Exception as e:
                logger.error(f"Error updating NPC traits: {e}")
                results["error"] = str(e)
                results["error_count"] = len(npc_ids)
                
                # Rollback on error
                try:
                    with get_db_connection() as conn, conn.cursor() as cursor:
                        cursor.execute("ROLLBACK")
                except Exception as rollback_error:
                    logger.error(f"Error rolling back transaction: {rollback_error}")
                    
        # Memory update - add memory to NPCs
        elif update_type == "memory_update":
            memory_text = update_data.get("memory_text")
            importance = update_data.get("importance", "medium")
            emotional = update_data.get("emotional", False)
            tags = update_data.get("tags", [])
            
            if not memory_text:
                return {"error": "No memory text specified"}
                
            # Get memory system
            memory_system = await self._get_memory_system()
            
            batch_tasks = []
            for npc_id in npc_ids:
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
            for npc_id, result in zip(npc_ids, batch_results):
                if isinstance(result, Exception):
                    results["error_count"] += 1
                    results["details"][npc_id] = {"error": str(result)}
                else:
                    results["success_count"] += 1
                    results["details"][npc_id] = {"success": True, "memory_id": result.get("id")}
                    
        # Schedule update - update NPC schedules
        elif update_type == "schedule_update":
            schedule_data = update_data.get("schedule_data")
            time_period = update_data.get("time_period")  # e.g., "morning", "evening", or "default"
            
            if not schedule_data:
                return {"error": "No schedule data specified"}
                
            try:
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute("BEGIN")
                    
                    updated_ids = []
                    for npc_id in npc_ids:
                        # First get current schedule
                        cursor.execute("""
                            SELECT schedule
                            FROM NPCStats
                            WHERE npc_id = %s
                            AND user_id = %s
                            AND conversation_id = %s
                        """, (npc_id, self.user_id, self.conversation_id))
                        
                        row = cursor.fetchone()
                        if not row:
                            continue
                            
                        current_schedule = row[0]
                        
                        # Parse and update schedule
                        if current_schedule is None:
                            current_schedule = {}
                        elif isinstance(current_schedule, str):
                            import json
                            try:
                                current_schedule = json.loads(current_schedule)
                            except json.JSONDecodeError:
                                current_schedule = {}
                        
                        if not isinstance(current_schedule, dict):
                            current_schedule = {}
                            
                        # Update schedule for specific time period or create a new one
                        if time_period:
                            current_schedule[time_period] = schedule_data
                        else:
                            # Replace entire schedule
                            current_schedule = schedule_data
                            
                        # Save updated schedule
                        import json
                        cursor.execute("""
                            UPDATE NPCStats
                            SET schedule = %s
                            WHERE npc_id = %s
                            AND user_id = %s
                            AND conversation_id = %s
                        """, (json.dumps(current_schedule), npc_id, self.user_id, self.conversation_id))
                        
                        updated_ids.append(npc_id)
                        
                    cursor.execute("COMMIT")
                    
                    results["success_count"] = len(updated_ids)
                    results["updated_npcs"] = updated_ids
                    
            except Exception as e:
                logger.error(f"Error updating NPC schedules: {e}")
                results["error"] = str(e)
                results["error_count"] = len(npc_ids)
                
                # Rollback on error
                try:
                    with get_db_connection() as conn, conn.cursor() as cursor:
                        cursor.execute("ROLLBACK")
                except Exception as rollback_error:
                    logger.error(f"Error rolling back transaction: {rollback_error}")
                    
        # Status update - update NPC status (active, introduced, etc.)
        elif update_type == "status_update":
            status_field = update_data.get("field")
            status_value = update_data.get("value")
            
            if status_field is None or status_value is None:
                return {"error": "Must specify status field and value"}
                
            # Validate status field for security
            allowed_fields = ["introduced", "active", "visible"]
            if status_field not in allowed_fields:
                return {"error": f"Invalid status field. Allowed: {allowed_fields}"}
                
            try:
                with get_db_connection() as conn, conn.cursor() as cursor:
                    cursor.execute(f"""
                        UPDATE NPCStats
                        SET {status_field} = %s
                        WHERE npc_id = ANY(%s)
                        AND user_id = %s
                        AND conversation_id = %s
                        RETURNING npc_id
                    """, (status_value, npc_ids, self.user_id, self.conversation_id))
                    
                    updated_ids = [row[0] for row in cursor.fetchall()]
                    results["success_count"] = len(updated_ids)
                    results["updated_npcs"] = updated_ids
                    
            except Exception as e:
                logger.error(f"Error updating NPC status: {e}")
                results["error"] = str(e)
                results["error_count"] = len(npc_ids)
                
        # Appearance update - update NPC appearance
        elif update_type == "appearance_update":
            appearance_data = update_data.get("appearance_data")
            
            if not appearance_data:
                return {"error": "No appearance data specified"}
                
            try:
                with get_db_connection() as conn, conn.cursor() as cursor:
                    import json
                    
                    # Ensure appearance data is JSON
                    if isinstance(appearance_data, dict):
                        appearance_json = json.dumps(appearance_data)
                    else:
                        appearance_json = appearance_data
                        
                    cursor.execute("""
                        UPDATE NPCStats
                        SET appearance = %s
                        WHERE npc_id = ANY(%s)
                        AND user_id = %s
                        AND conversation_id = %s
                        RETURNING npc_id
                    """, (appearance_json, npc_ids, self.user_id, self.conversation_id))
                    
                    updated_ids = [row[0] for row in cursor.fetchall()]
                    results["success_count"] = len(updated_ids)
                    results["updated_npcs"] = updated_ids
                    
            except Exception as e:
                logger.error(f"Error updating NPC appearance: {e}")
                results["error"] = str(e)
                results["error_count"] = len(npc_ids)
                
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
