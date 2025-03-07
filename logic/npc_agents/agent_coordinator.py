# logic/npc_agents/agent_coordinator.py

"""
Coordinates multiple NPC agents for group interactions, with improved memory integration.
"""

import logging
import asyncio
import random
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from db.connection import get_db_connection
from .npc_agent import NPCAgent
from memory.wrapper import MemorySystem

logger = logging.getLogger(__name__)

@dataclass
class NPCAction:
    """
    Simple data class to represent an NPC's chosen action.
    """
    type: str
    description: str
    target: Optional[str] = None            # Could be 'group', another npc_id, etc.
    target_name: Optional[str] = None
    stats_influenced: Dict[str, int] = None


class NPCAgentCoordinator:
    """Coordinates the behavior of multiple NPC agents."""

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.active_agents: Dict[int, NPCAgent] = {}  # Map of npc_id -> NPCAgent
        self._memory_system = None

    async def _get_memory_system(self):
        """Lazy-load the memory system."""
        if self._memory_system is None:
            self._memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        return self._memory_system

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
            WHERE user_id=%s
              AND conversation_id=%s
        """
        params = [self.user_id, self.conversation_id]

        if npc_ids:
            query += " AND npc_id = ANY(%s)"
            params.append(npc_ids)

        loaded_ids: List[int] = []
        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

            for row in rows:
                npc_id = row[0]
                if npc_id not in self.active_agents:
                    self.active_agents[npc_id] = NPCAgent(npc_id, self.user_id, self.conversation_id)
                loaded_ids.append(npc_id)

        logger.info("Loaded agents: %s", loaded_ids)
        return loaded_ids

    async def make_group_decisions(
        self,
        npc_ids: List[int],
        shared_context: Dict[str, Any],
        available_actions: Optional[Dict[int, List[NPCAction]]] = None
    ) -> Dict[str, Any]:
        """
        Coordinate decision-making for a group of NPCs with improved memory integration.

        Args:
            npc_ids: List of NPC IDs in the group
            shared_context: A dictionary describing the shared context/environment
            available_actions: Optional dict of predefined actions per NPC

        Returns:
            A dictionary representing the coordinated 'action_plan' for the entire group.
        """
        # Ensure all NPCs are loaded
        await self.load_agents(npc_ids)

        # Get memory system for group context
        memory_system = await self._get_memory_system()
        
        # Get prior group interaction memories
        group_context = {
            "participants": npc_ids,
            "location": shared_context.get("location", "Unknown"),
            "type": "group_interaction"
        }
        
        # Enhance shared context with group history for each NPC
        for npc_id in npc_ids:
            # Get NPC's previous memories of group interactions with these participants
            npc_group_memories = await memory_system.recall(
                entity_type="npc",
                entity_id=npc_id,
                query="group interaction",
                context=group_context,
                limit=3
            )
            
            # Add to shared context for this NPC
            if "npc_context" not in shared_context:
                shared_context["npc_context"] = {}
                
            shared_context["npc_context"][npc_id] = {
                "group_memories": npc_group_memories.get("memories", [])
            }

        # 1) Each NPC perceives the environment (run concurrently for performance)
        perceive_tasks = []
        for npc_id in npc_ids:
            agent = self.active_agents.get(npc_id)
            if agent:
                # Include group memories in perception context
                perception_context = shared_context.copy()
                perception_context["group_interaction"] = True
                
                perceive_tasks.append(agent.perceive_environment(perception_context))
            else:
                perceive_tasks.append(asyncio.sleep(0))  # Filler if agent is missing

        perceptions_list = await asyncio.gather(*perceive_tasks)
        perceptions = {npc_id: perceptions_list[i] for i, npc_id in enumerate(npc_ids)}

        # 2) Determine available actions if not provided
        if available_actions is None:
            available_actions = await self.generate_group_actions(npc_ids, perceptions)

        # 3) Each NPC decides individually (also run concurrently)
        decision_tasks = []
        for npc_id in npc_ids:
            agent = self.active_agents.get(npc_id)
            npc_actions = available_actions.get(npc_id, [])
            if agent:
                decision_tasks.append(agent.make_decision(perceptions[npc_id], npc_actions))
            else:
                decision_tasks.append(asyncio.sleep(0))

        decisions_list = await asyncio.gather(*decision_tasks)
        decisions = {npc_id: decisions_list[i] for i, npc_id in enumerate(npc_ids)}

        # 4) Resolve conflicts into a coherent plan
        action_plan = await self.resolve_decision_conflicts(decisions, npc_ids, perceptions)
        
        # 5) Create memories of the group interaction
        await self._create_group_interaction_memories(npc_ids, action_plan, shared_context)
        
        return action_plan

    async def generate_group_actions(
        self,
        npc_ids: List[int],
        perceptions: Dict[int, Any]
    ) -> Dict[int, List[NPCAction]]:
        """
        Generate possible actions for each NPC in a group context.
        Enhanced with memory-based influences.

        Args:
            npc_ids: IDs of the NPCs
            perceptions: Each NPC's environment perception

        Returns:
            A dict of npc_id -> list of NPCAction objects
        """
        npc_data = self._fetch_basic_npc_data(npc_ids)
        group_actions: Dict[int, List[NPCAction]] = {}
        
        # Get memory system
        memory_system = await self._get_memory_system()

        for npc_id in npc_ids:
            if npc_id not in npc_data:
                continue

            # Basic actions available to all NPCs
            actions = [
                NPCAction(type="talk", description="Talk to the group", target="group"),
                NPCAction(type="observe", description="Observe the group", target="group"),
                NPCAction(type="leave", description="Leave the group", target="group")
            ]

            # Get NPC's personality stats
            dom = npc_data[npc_id]["dominance"]
            cru = npc_data[npc_id]["cruelty"]
            
            # Get emotional state to influence available actions
            emotional_state = None
            try:
                emotional_state = await memory_system.get_npc_emotion(npc_id)
            except Exception as e:
                logger.error(f"Error getting emotional state for NPC {npc_id}: {e}")
            
            # Get NPC's mask information
            mask_info = None
            try:
                mask_info = await memory_system.get_npc_mask(npc_id)
            except Exception as e:
                logger.error(f"Error getting mask info for NPC {npc_id}: {e}")
            
            # Add emotionally-influenced actions if in a strong emotional state
            if emotional_state and "current_emotion" in emotional_state:
                current_emotion = emotional_state["current_emotion"]
                primary = current_emotion.get("primary", {})
                emotion_name = primary.get("name", "neutral")
                intensity = primary.get("intensity", 0.0)
                
                if intensity > 0.6:
                    # Add emotion-based actions for high intensity emotions
                    if emotion_name == "anger":
                        actions.append(NPCAction(
                            type="emotional_outburst",
                            description="Express anger to the group",
                            target="group"
                        ))
                    elif emotion_name == "fear":
                        actions.append(NPCAction(
                            type="emotional_response",
                            description="Show fear and anxiety",
                            target="group"
                        ))
                    elif emotion_name == "joy":
                        actions.append(NPCAction(
                            type="emotional_expression",
                            description="Express happiness and excitement",
                            target="group"
                        ))
            
            # Add mask-appropriate actions
            mask_integrity = 100
            if mask_info:
                mask_integrity = mask_info.get("integrity", 100)
                
                # If mask is breaking down, add actions that reveal true nature
                if mask_integrity < 50:
                    # Determine true nature based on hidden traits
                    hidden_traits = mask_info.get("hidden_traits", {})
                    if "cruel" in hidden_traits or "harsh" in hidden_traits:
                        actions.append(NPCAction(
                            type="mask_slip",
                            description="Show unexpected cruelty",
                            target="group"
                        ))
                    elif "manipulative" in hidden_traits or "controlling" in hidden_traits:
                        actions.append(NPCAction(
                            type="mask_slip",
                            description="Attempt to manipulate the situation",
                            target="group"
                        ))

            # Add target-specific actions for other NPCs
            for other_id in npc_ids:
                if other_id != npc_id and other_id in npc_data:
                    other_name = npc_data[other_id]["npc_name"]
                    
                    # Basic interaction
                    actions.append(NPCAction(
                        type="talk_to",
                        description=f"Talk to {other_name}",
                        target=str(other_id),
                        target_name=other_name
                    ))
                    
                    # Get relationship with this NPC to influence actions
                    relationship_level = self._get_relationship_level(npc_id, other_id)
                    
                    # High dominance actions
                    if dom > 60:
                        actions.append(NPCAction(
                            type="command",
                            description=f"Command {other_name}",
                            target=str(other_id),
                            target_name=other_name
                        ))
                    
                    # High cruelty actions
                    if cru > 60:
                        actions.append(NPCAction(
                            type="mock",
                            description=f"Mock {other_name}",
                            target=str(other_id),
                            target_name=other_name
                        ))
                    
                    # Relationship-based actions
                    if relationship_level is not None:
                        if relationship_level > 70:  # Friendly relationship
                            actions.append(NPCAction(
                                type="support",
                                description=f"Support {other_name}",
                                target=str(other_id),
                                target_name=other_name
                            ))
                        elif relationship_level < 30:  # Antagonistic relationship
                            actions.append(NPCAction(
                                type="challenge",
                                description=f"Challenge {other_name}",
                                target=str(other_id),
                                target_name=other_name
                            ))

            group_actions[npc_id] = actions

        return group_actions

    async def resolve_decision_conflicts(
        self,
        decisions: Dict[int, NPCAction],
        npc_ids: List[int],
        perceptions: Dict[int, Any]
    ) -> Dict[str, Any]:
        """
        Resolve conflicts between NPC decisions to form a coherent plan.
        Enhanced with memory influence.

        This approach sorts NPCs by dominance and tries to apply each NPC's action
        in descending order, skipping an action if the target is already 'affected'.

        Returns:
            A structured dict with group_actions and individual_actions.
        """
        # Get memory system
        memory_system = await self._get_memory_system()
        
        npc_dominance = self._fetch_npc_dominance(npc_ids)
        
        # Apply emotional state modifiers to dominance
        for npc_id in npc_ids:
            try:
                emotional_state = await memory_system.get_npc_emotion(npc_id)
                if emotional_state and "current_emotion" in emotional_state:
                    current_emotion = emotional_state["current_emotion"]
                    primary = current_emotion.get("primary", {})
                    emotion_name = primary.get("name", "neutral")
                    intensity = primary.get("intensity", 0.0)
                    
                    # Certain emotions can temporarily increase/decrease dominance
                    if emotion_name == "anger" and intensity > 0.5:
                        npc_dominance[npc_id] = min(100, npc_dominance.get(npc_id, 50) + int(intensity * 20))
                    elif emotion_name == "fear" and intensity > 0.5:
                        npc_dominance[npc_id] = max(0, npc_dominance.get(npc_id, 50) - int(intensity * 20))
            except Exception as e:
                logger.error(f"Error applying emotional modifiers for NPC {npc_id}: {e}")
                
        # Apply mask integrity modifiers
        for npc_id in npc_ids:
            try:
                mask_info = await memory_system.get_npc_mask(npc_id)
                if mask_info:
                    mask_integrity = mask_info.get("integrity", 100)
                    hidden_traits = mask_info.get("hidden_traits", {})
                    
                    # Low mask integrity reveals true dominance
                    if mask_integrity < 50 and "domineering" in hidden_traits:
                        npc_dominance[npc_id] = min(100, npc_dominance.get(npc_id, 50) + 20)
            except Exception as e:
                logger.error(f"Error applying mask modifiers for NPC {npc_id}: {e}")
        
        # Sort by modified dominance descending
        sorted_npcs = sorted(npc_ids, key=lambda id_: npc_dominance.get(id_, 0), reverse=True)

        action_plan = {"group_actions": [], "individual_actions": {}}
        affected_npcs: Set[int] = set()

        for npc_id in sorted_npcs:
            action = decisions.get(npc_id)
            if not action:
                continue

            # Group actions
            if action.type in ["talk", "command", "emotional_outburst", "mask_slip"] and action.target == "group":
                action_plan["group_actions"].append({"npc_id": npc_id, "action": action.__dict__})
                affected_npcs.update(npc_ids)

            # Direct interactions
            elif action.type in ["talk_to", "command", "mock", "support", "challenge"] and action.target is not None:
                try:
                    target_id = int(action.target)  # The other NPC's ID
                except ValueError:
                    target_id = -1

                if target_id not in affected_npcs:
                    if npc_id not in action_plan["individual_actions"]:
                        action_plan["individual_actions"][npc_id] = []
                    action_plan["individual_actions"][npc_id].append(action.__dict__)
                    affected_npcs.add(target_id)

            # Other actions
            else:
                if npc_id not in action_plan["individual_actions"]:
                    action_plan["individual_actions"][npc_id] = []
                action_plan["individual_actions"][npc_id].append(action.__dict__)

        logger.info("Resolved action plan: %s", action_plan)
        return action_plan

    async def _create_group_interaction_memories(
        self,
        npc_ids: List[int],
        action_plan: Dict[str, Any],
        shared_context: Dict[str, Any]
    ) -> None:
        """
        Create memories of the group interaction for each participating NPC.
        """
        memory_system = await self._get_memory_system()
        
        # Get location and time information
        location = shared_context.get("location", "Unknown")
        time_of_day = shared_context.get("time_of_day", "Unknown")
        
        # Format participant list
        npc_names = {}
        with get_db_connection() as conn, conn.cursor() as cursor:
            for npc_id in npc_ids:
                cursor.execute("""
                    SELECT npc_name FROM NPCStats
                    WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                """, (npc_id, self.user_id, self.conversation_id))
                row = cursor.fetchone()
                if row:
                    npc_names[npc_id] = row[0]
                else:
                    npc_names[npc_id] = f"NPC_{npc_id}"
        
        participant_names = list(npc_names.values())
        participant_list = ", ".join(participant_names)
        
        # Process group actions
        group_actions = action_plan.get("group_actions", [])
        for group_action in group_actions:
            npc_id = group_action.get("npc_id")
            action = group_action.get("action", {})
            
            # Create memory for the acting NPC
            if npc_id:
                actor_name = npc_names.get(npc_id, f"NPC_{npc_id}")
                action_type = action.get("type", "unknown")
                action_desc = action.get("description", "did something")
                
                # Create memory of performing the action
                actor_memory_text = f"I {action_desc} to the group including {participant_list} at {location} during {time_of_day}"
                
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=npc_id,
                    memory_text=actor_memory_text,
                    importance="medium",
                    tags=["group_interaction", action_type]
                )
                
                # Create memories for other NPCs of observing this action
                for observer_id in npc_ids:
                    if observer_id != npc_id:
                        observer_memory_text = f"{actor_name} {action_desc} to our group at {location}"
                        
                        await memory_system.remember(
                            entity_type="npc",
                            entity_id=observer_id,
                            memory_text=observer_memory_text,
                            importance="low",
                            tags=["group_interaction", "observation"]
                        )
        
        # Process individual actions
        individual_actions = action_plan.get("individual_actions", {})
        for actor_id, actions in individual_actions.items():
            actor_name = npc_names.get(actor_id, f"NPC_{actor_id}")
            
            for action in actions:
                action_type = action.get("type", "unknown")
                action_desc = action.get("description", "did something")
                target = action.get("target")
                
                # Only process actions targeted at other NPCs
                if target and target.isdigit():
                    target_id = int(target)
                    target_name = npc_names.get(target_id, f"NPC_{target_id}")
                    
                    # Create memory for the actor
                    actor_memory_text = f"I {action_desc} to {target_name} during our group interaction at {location}"
                    
                    await memory_system.remember(
                        entity_type="npc",
                        entity_id=actor_id,
                        memory_text=actor_memory_text,
                        importance="medium",
                        tags=["group_interaction", action_type]
                    )
                    
                    # Create memory for the target
                    target_memory_text = f"{actor_name} {action_desc} to me during our group interaction at {location}"
                    
                    await memory_system.remember(
                        entity_type="npc",
                        entity_id=target_id,
                        memory_text=target_memory_text,
                        importance="medium",
                        tags=["group_interaction", "targeted"]
                    )
                    
                    # Create memories for observers
                    for observer_id in npc_ids:
                        if observer_id != actor_id and observer_id != target_id:
                            observer_memory_text = f"{actor_name} {action_desc} to {target_name} during our group interaction"
                            
                            await memory_system.remember(
                                entity_type="npc",
                                entity_id=observer_id,
                                memory_text=observer_memory_text,
                                importance="low",
                                tags=["group_interaction", "observation"]
                            )

    async def handle_player_action(
        self,
        player_action: Dict[str, Any],
        context: Dict[str, Any],
        npc_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Handle a player action directed at multiple NPCs.
        Enhanced with group memory formation.

        If npc_ids is None, we attempt to find them based on location or 'target_npc_id'.
        Returns a dict containing responses from each NPC.
        """
        affected_npcs = npc_ids or await self.determine_affected_npcs(player_action, context)
        if not affected_npcs:
            return {"npc_responses": []}

        await self.load_agents(affected_npcs)
        
        # Enhance context with group information
        enhanced_context = context.copy()
        enhanced_context["is_group_interaction"] = True
        enhanced_context["affected_npcs"] = affected_npcs
        
        # If many NPCs are affected, add group dynamics to context
        if len(affected_npcs) > 1:
            # Find dominant NPCs
            npc_dominance = self._fetch_npc_dominance(affected_npcs)
            sorted_by_dominance = sorted(affected_npcs, key=lambda id_: npc_dominance.get(id_, 0), reverse=True)
            
            enhanced_context["group_dynamics"] = {
                "dominant_npc_id": sorted_by_dominance[0] if sorted_by_dominance else None,
                "participant_count": len(affected_npcs)
            }

        response_tasks = []
        for npc_id in affected_npcs:
            agent = self.active_agents.get(npc_id)
            if agent:
                response_tasks.append(agent.process_player_action(player_action))
            else:
                response_tasks.append(asyncio.sleep(0))

        responses = await asyncio.gather(*response_tasks)
        
        # Create memories of this group interaction
        if len(affected_npcs) > 1:
            memory_system = await self._get_memory_system()
            
            # Get NPC names for better memory context
            npc_names = {}
            with get_db_connection() as conn, conn.cursor() as cursor:
                for npc_id in affected_npcs:
                    cursor.execute("""
                        SELECT npc_name FROM NPCStats
                        WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                    """, (npc_id, self.user_id, self.conversation_id))
                    row = cursor.fetchone()
                    if row:
                        npc_names[npc_id] = row[0]
                    else:
                        npc_names[npc_id] = f"NPC_{npc_id}"
            
            # Create a memory of this group interaction for each NPC
            for npc_id in affected_npcs:
                # Filter out this NPC from the participant list
                other_npcs = [npc_names.get(other_id, f"NPC_{other_id}") for other_id in affected_npcs if other_id != npc_id]
                others_text = ", ".join(other_npcs) if other_npcs else "no one else"
                
                memory_text = f"The player {player_action.get('description', 'interacted with us')} while I was with {others_text}"
                
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=npc_id,
                    memory_text=memory_text,
                    importance="medium",
                    tags=["group_interaction", "player_action"]
                )
        
        return {"npc_responses": responses}

    async def determine_affected_npcs(self, player_action: Dict[str, Any], context: Dict[str, Any]) -> List[int]:
        """
        Determine which NPCs are affected by a player action.

        If "target_npc_id" is in the action, return that. Otherwise, find all NPCs in the
        current location from the context.
        """
        if "target_npc_id" in player_action:
            return [player_action["target_npc_id"]]

        current_location = context.get("location", "Unknown")
        logger.debug("Determining NPCs at location=%s", current_location)

        query = """
            SELECT npc_id
            FROM NPCStats
            WHERE user_id=%s
              AND conversation_id=%s
              AND current_location=%s
        """
        params = (self.user_id, self.conversation_id, current_location)
        npc_list: List[int] = []

        with get_db_connection() as conn, conn.cursor() as cursor:
            try:
                cursor.execute(query, params)
                npc_list = [row[0] for row in cursor.fetchall()]
            except Exception as e:
                logger.error("Error determining affected NPCs: %s", e)

        return npc_list

    # ----------------------------------------------------------------
    # Internal helper methods
    # ----------------------------------------------------------------

    def _fetch_basic_npc_data(self, npc_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Fetch minimal NPC data (name, dominance, cruelty).
        """
        data_map: Dict[int, Dict[str, Any]] = {}
        if not npc_ids:
            return data_map

        query = f"""
            SELECT npc_id, npc_name, dominance, cruelty
            FROM NPCStats
            WHERE npc_id = ANY(%s)
              AND user_id = %s
              AND conversation_id = %s
        """
        params = (npc_ids, self.user_id, self.conversation_id)

        with get_db_connection() as conn, conn.cursor() as cursor:
            try:
                cursor.execute(query, params)
                for row in cursor.fetchall():
                    nid, name, dom, cru = row
                    data_map[nid] = {
                        "npc_name": name,
                        "dominance": dom,
                        "cruelty": cru
                    }
            except Exception as e:
                logger.error("Error fetching basic NPC data: %s", e)

        return data_map

    def _fetch_npc_dominance(self, npc_ids: List[int]) -> Dict[int, int]:
        """
        Fetch only dominance values for the NPCs, used for sorting, etc.
        """
        dom_map: Dict[int, int] = {}
        if not npc_ids:
            return dom_map

        query = f"""
            SELECT npc_id, dominance
            FROM NPCStats
            WHERE npc_id = ANY(%s)
              AND user_id = %s
              AND conversation_id = %s
        """
        params = (npc_ids, self.user_id, self.conversation_id)

        with get_db_connection() as conn, conn.cursor() as cursor:
            try:
                cursor.execute(query, params)
                for row in cursor.fetchall():
                    nid, dom = row
                    dom_map[nid] = dom
            except Exception as e:
                logger.error("Error fetching NPC dominance: %s", e)
        return dom_map
    
    def _get_relationship_level(self, npc_id1: int, npc_id2: int) -> Optional[int]:
        """
        Get relationship level between two NPCs.
        """
        with get_db_connection() as conn, conn.cursor() as cursor:
            try:
                cursor.execute("""
                    SELECT link_level
                    FROM SocialLinks
                    WHERE user_id = %s AND conversation_id = %s
                    AND ((entity1_type = 'npc' AND entity1_id = %s AND entity2_type = 'npc' AND entity2_id = %s)
                       OR (entity1_type = 'npc' AND entity1_id = %s AND entity2_type = 'npc' AND entity2_id = %s))
                """, (self.user_id, self.conversation_id, npc_id1, npc_id2, npc_id2, npc_id1))
                
                row = cursor.fetchone()
                if row:
                    return row[0]
                return None
            except Exception as e:
                logger.error(f"Error getting relationship between NPCs {npc_id1} and {npc_id2}: {e}")
                return None
