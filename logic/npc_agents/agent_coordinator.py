# logic/npc_agents/agent_coordinator.py

"""
Coordinates multiple NPC agents for group interactions, with improved memory integration.
"""

import logging
import asyncio
import random
from typing import List, Dict, Any, Optional, Set, Tuple
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
        self._emotional_states = {}  # Cache of emotional states to avoid repeated queries
        self._mask_states = {}       # Cache of mask states to avoid repeated queries

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
        Refactored into smaller, more maintainable methods.
        """
        # Ensure all NPCs are loaded
        await self.load_agents(npc_ids)
        
        # 1. Prepare enhanced group context with memory integration
        enhanced_context = await self._prepare_group_context(npc_ids, shared_context)
        
        # 2. Each NPC perceives the environment (run concurrently for performance)
        perceptions = await self._gather_individual_perceptions(npc_ids, enhanced_context)
        
        # 3. Determine available actions if not provided
        if available_actions is None:
            available_actions = await self.generate_group_actions(npc_ids, perceptions)
        
        # 4. Each NPC decides individually (also run concurrently)
        decisions = await self._collect_individual_decisions(npc_ids, perceptions, available_actions)
        
        # 5. Resolve conflicts into a coherent plan
        action_plan = await self._resolve_group_conflicts(decisions, npc_ids, perceptions)
        
        # 6. Create memories and handle emotional effects
        await self._create_group_memory_effects(npc_ids, action_plan, shared_context)
        
        return action_plan
    
    async def _prepare_group_context(self, npc_ids: List[int], shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare enhanced context for group interactions with memory."""
        memory_system = await self._get_memory_system()
        
        # Create enhanced context
        enhanced_context = shared_context.copy()
        enhanced_context["participants"] = npc_ids
        enhanced_context["type"] = "group_interaction"
        
        # Add NPC-specific context information
        if "npc_context" not in enhanced_context:
            enhanced_context["npc_context"] = {}
        
        # Populate context for each NPC
        for npc_id in npc_ids:
            # Get previous group memories
            group_memories = await memory_system.recall(
                entity_type="npc",
                entity_id=npc_id,
                query="group interaction",
                context=enhanced_context,
                limit=3
            )
            
            # Check for flashback opportunity
            flashback = None
            if random.random() < 0.15:  # 15% chance of flashback in group setting
                context_text = f"group interaction at {shared_context.get('location', 'Unknown')}"
                flashback = await memory_system.npc_flashback(npc_id, context_text)
            
            # Get NPC's emotional state
            emotional_state = await self._get_npc_emotional_state(npc_id)
            
            # Get NPC's mask status
            mask_info = await self._get_npc_mask(npc_id)
            
            # Get NPC's beliefs about other NPCs in the group
            beliefs = await self._get_npc_beliefs_about_others(npc_id, npc_ids)
            
            # Store in enhanced context
            enhanced_context["npc_context"][npc_id] = {
                "group_memories": group_memories.get("memories", []),
                "emotional_state": emotional_state,
                "mask_info": mask_info,
                "flashback": flashback,
                "beliefs": beliefs
            }
        
        return enhanced_context
    
    async def _gather_individual_perceptions(self, npc_ids: List[int], enhanced_context: Dict[str, Any]) -> Dict[int, Any]:
        """Gather perceptions from all NPCs concurrently."""
        perception_tasks = []
        for npc_id in npc_ids:
            agent = self.active_agents.get(npc_id)
            if agent:
                # Include group context in perception
                perception_context = enhanced_context.copy()
                perception_context["group_interaction"] = True
                
                # Add NPC-specific context
                npc_specific = enhanced_context.get("npc_context", {}).get(npc_id, {})
                for key, value in npc_specific.items():
                    perception_context[key] = value
                    
                perception_tasks.append(agent.perceive_environment(perception_context))
            else:
                perception_tasks.append(asyncio.sleep(0))  # Filler if agent is missing
        
        # Gather results concurrently
        perceptions_list = await asyncio.gather(*perception_tasks)
        return {npc_id: perceptions_list[i] for i, npc_id in enumerate(npc_ids)}
    
    async def _collect_individual_decisions(
        self, 
        npc_ids: List[int], 
        perceptions: Dict[int, Any],
        available_actions: Dict[int, List[NPCAction]]
    ) -> Dict[int, Any]:
        """Collect decisions from each NPC concurrently."""
        decision_tasks = []
        for npc_id in npc_ids:
            agent = self.active_agents.get(npc_id)
            npc_actions = available_actions.get(npc_id, [])
            if agent:
                decision_tasks.append(agent.make_decision(perceptions[npc_id], npc_actions))
            else:
                decision_tasks.append(asyncio.sleep(0))
        
        # Gather results concurrently
        decisions_list = await asyncio.gather(*decision_tasks)
        return {npc_id: decisions_list[i] for i, npc_id in enumerate(npc_ids)}
    
    async def _create_group_memory_effects(
        self,
        npc_ids: List[int],
        action_plan: Dict[str, Any],
        shared_context: Dict[str, Any]
    ) -> None:
        """Create memories and handle emotional effects of the group interaction."""
        # Create memories
        await self._create_group_interaction_memories(npc_ids, action_plan, shared_context)
        
        # Check for mask slippage
        await self._check_for_mask_slippage(npc_ids, action_plan, shared_context)
        
        # Process emotional contagion between NPCs
        await self._process_emotional_contagion(npc_ids, action_plan)

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
            emotional_state = await self._get_npc_emotional_state(npc_id)
            
            # Get NPC's mask information
            mask_info = await self._get_npc_mask(npc_id)
            
            # Get relevant beliefs that might influence actions
            beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=npc_id,
                topic="group_interaction"
            )
            
            # Get memories of past similar situations to influence actions
            relevant_memories = await memory_system.recall(
                entity_type="npc",
                entity_id=npc_id,
                query="similar group interaction",
                context={"location": perceptions[npc_id].get("location", "Unknown")},
                limit=2
            )
            
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
                    elif emotion_name == "sadness":
                        actions.append(NPCAction(
                            type="emotional_expression",
                            description="Express sadness or disappointment",
                            target="group"
                        ))
                    elif emotion_name == "disgust":
                        actions.append(NPCAction(
                            type="emotional_expression",
                            description="Express disgust or contempt",
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
                    elif "selfish" in hidden_traits or "callous" in hidden_traits:
                        actions.append(NPCAction(
                            type="mask_slip",
                            description="Reveal a selfish or callous side",
                            target="group"
                        ))

            # Add actions based on beliefs
            if beliefs:
                for belief in beliefs:
                    belief_text = belief.get("belief", "").lower()
                    if "dangerous" in belief_text or "threat" in belief_text:
                        actions.append(NPCAction(
                            type="defensive",
                            description="Take a defensive stance",
                            target="group"
                        ))
                    elif "opportunity" in belief_text or "beneficial" in belief_text:
                        actions.append(NPCAction(
                            type="engage",
                            description="Actively engage with the group",
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
                    
                    # Specific belief-based actions toward this NPC
                    other_beliefs = await memory_system.get_beliefs(
                        entity_type="npc",
                        entity_id=npc_id,
                        topic=f"npc_{other_id}"
                    )
                    
                    if other_beliefs:
                        # Process each belief about this specific NPC
                        for belief in other_beliefs:
                            belief_text = belief.get("belief", "").lower()
                            confidence = belief.get("confidence", 0.5)
                            
                            # Only act on beliefs with decent confidence
                            if confidence > 0.6:
                                if "friend" in belief_text or "ally" in belief_text:
                                    actions.append(NPCAction(
                                        type="confide",
                                        description=f"Confide in {other_name}",
                                        target=str(other_id),
                                        target_name=other_name
                                    ))
                                elif "enemy" in belief_text or "rival" in belief_text:
                                    actions.append(NPCAction(
                                        type="undermine",
                                        description=f"Undermine {other_name}",
                                        target=str(other_id),
                                        target_name=other_name
                                    ))
                                elif "untrustworthy" in belief_text or "liar" in belief_text:
                                    actions.append(NPCAction(
                                        type="doubt",
                                        description=f"Express doubt about {other_name}'s statements",
                                        target=str(other_id),
                                        target_name=other_name
                                    ))

            group_actions[npc_id] = actions

        return group_actions

    async def _process_emotional_contagion(self, npc_ids: List[int], action_plan: Dict[str, Any]) -> None:
        """Process emotional contagion between NPCs during group interactions."""
        memory_system = await self._get_memory_system()
        
        # Get emotional states for all NPCs
        emotional_states = {}
        for npc_id in npc_ids:
            emotional_state = await memory_system.get_npc_emotion(npc_id)
            if emotional_state and "current_emotion" in emotional_state:
                emotional_states[npc_id] = emotional_state
        
        # Calculate dominance factors (affects emotional influence)
        dominance_factors = self._fetch_npc_dominance(npc_ids)
        
        # Process contagion for each NPC
        for affected_id in npc_ids:
            # Skip NPCs with no emotional data
            if affected_id not in emotional_states:
                continue
                
            affected_state = emotional_states[affected_id]
            affected_dominance = dominance_factors.get(affected_id, 50)
            
            # Calculate susceptibility - lower dominance = more susceptible
            susceptibility = max(0.1, 1.0 - (affected_dominance / 100))
            
            # Calculate weighted influence from other NPCs
            new_emotion = None
            max_influence = 0
            
            for influencer_id in npc_ids:
                if influencer_id == affected_id or influencer_id not in emotional_states:
                    continue
                    
                # Get influencer data
                influencer_state = emotional_states[influencer_id]
                influencer_dominance = dominance_factors.get(influencer_id, 50)
                
                # Skip if neutral emotion or low intensity
                influencer_emotion = influencer_state["current_emotion"]
                primary = influencer_emotion.get("primary", {})
                emotion_name = primary.get("name", "neutral")
                intensity = primary.get("intensity", 0.0)
                
                if emotion_name == "neutral" or intensity < 0.5:
                    continue
                    
                # Calculate influence power based on dominance and emotion intensity
                influence_power = (influencer_dominance / 100) * intensity * susceptibility
                
                # If this is the strongest influence so far, it affects the NPC
                if influence_power > max_influence:
                    max_influence = influence_power
                    new_emotion = {
                        "name": emotion_name,
                        "intensity": intensity * susceptibility  # Weakened version of original
                    }
            
            # Apply new emotion if sufficient influence
            if new_emotion and max_influence > 0.3:
                await memory_system.update_npc_emotion(
                    npc_id=affected_id,
                    emotion=new_emotion["name"],
                    intensity=new_emotion["intensity"]
                )

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
                emotional_state = await self._get_npc_emotional_state(npc_id)
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
                    elif emotion_name == "confidence" and intensity > 0.5:
                        npc_dominance[npc_id] = min(100, npc_dominance.get(npc_id, 50) + int(intensity * 15))
                    elif emotion_name == "sadness" and intensity > 0.6:
                        npc_dominance[npc_id] = max(0, npc_dominance.get(npc_id, 50) - int(intensity * 15))
            except Exception as e:
                logger.error(f"Error applying emotional modifiers for NPC {npc_id}: {e}")

            coalitions = await self._detect_potential_coalitions(npc_ids, perceptions)
        
        # When sorting NPCs by dominance, modify to consider coalitions
        if coalitions:
            # Boost dominance for coalition members
            for coalition in coalitions:
                coalition_boost = 15 * coalition["strength"]  # 0-15 point boost
                leader = coalition["leader"]
                
                for member in coalition["members"]:
                    if member != leader:
                        # Followers get smaller boost
                        npc_dominance[member] = min(100, npc_dominance.get(member, 50) + (coalition_boost * 0.5))
                    else:
                        # Leader gets full boost
                        npc_dominance[member] = min(100, npc_dominance.get(member, 50) + coalition_boost)
                
        # Apply mask integrity modifiers
        for npc_id in npc_ids:
            try:
                mask_info = await self._get_npc_mask(npc_id)
                if mask_info:
                    mask_integrity = mask_info.get("integrity", 100)
                    hidden_traits = mask_info.get("hidden_traits", {})
                    
                    # Low mask integrity reveals true dominance
                    if mask_integrity < 50 and "domineering" in hidden_traits:
                        npc_dominance[npc_id] = min(100, npc_dominance.get(npc_id, 50) + 20)
                    elif mask_integrity < 50 and "submissive" in hidden_traits:
                        npc_dominance[npc_id] = max(0, npc_dominance.get(npc_id, 50) - 20)
            except Exception as e:
                logger.error(f"Error applying mask modifiers for NPC {npc_id}: {e}")
        
        # Apply modifiers based on trauma or significant memories
        for npc_id in npc_ids:
            try:
                # Check for traumatic memories that might be influencing current behavior
                trauma_memories = await memory_system.recall(
                    entity_type="npc",
                    entity_id=npc_id,
                    query="traumatic event",
                    context={"location": perceptions[npc_id].get("location", "Unknown")},
                    limit=1
                )
                
                if trauma_memories and trauma_memories.get("memories"):
                    # Trauma can reduce effective dominance in triggering contexts
                    npc_dominance[npc_id] = max(0, npc_dominance.get(npc_id, 50) - 15)
            except Exception as e:
                logger.error(f"Error applying trauma modifiers for NPC {npc_id}: {e}")
        
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
            elif action.type in ["talk_to", "command", "mock", "support", "challenge", "confide", "undermine", "doubt"] and action.target is not None:
                try:
                    target_id = int(action.target)  # The other NPC's ID
                except ValueError:
                    target_id = -1

                if target_id not in affected_npcs:
                    if npc_id not in action_plan["individual_actions"]:
                        action_plan["individual_actions"][npc_id] = []
                    action_plan["individual_actions"][npc_id].append(action.__dict__)
                    affected_npcs.add(target_id)
                    
                    # Record beliefs based on interactions
                    if action.type in ["confide", "support"]:
                        # Positive interaction strengthens positive beliefs or forms new ones
                        await self._update_npc_belief_from_interaction(
                            npc_id, target_id, positive=True,
                            interaction_type=action.type
                        )
                    elif action.type in ["mock", "challenge", "undermine", "doubt"]:
                        # Negative interaction strengthens negative beliefs or forms new ones
                        await self._update_npc_belief_from_interaction(
                            npc_id, target_id, positive=False,
                            interaction_type=action.type
                        )

            # Other actions
            else:
                if npc_id not in action_plan["individual_actions"]:
                    action_plan["individual_actions"][npc_id] = []
                action_plan["individual_actions"][npc_id].append(action.__dict__)

        logger.info("Resolved action plan: %s", action_plan)
        return action_plan

    async def _detect_potential_coalitions(
        self,
        npc_ids: List[int],
        perceptions: Dict[int, Any]
    ) -> List[Dict[str, Any]]:
        """Detect potential coalitions between NPCs based on aligned interests."""
        coalitions = []
        
        # Get relationships between all NPCs
        relationships = {}
        for npc_id in npc_ids:
            for other_id in npc_ids:
                if npc_id != other_id:
                    # Get relationship level
                    link_level = await self._get_relationship_level(npc_id, other_id)
                    if link_level:
                        relationships[(npc_id, other_id)] = link_level
        
        # Identify potential pairs with strong relationships (above 60)
        potential_pairs = []
        for (npc1, npc2), level in relationships.items():
            if level > 60:
                potential_pairs.append((npc1, npc2, level))
        
        # Sort pairs by relationship strength
        potential_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Form coalitions from these pairs
        assigned_npcs = set()
        for npc1, npc2, level in potential_pairs:
            # Skip if either NPC is already in a coalition
            if npc1 in assigned_npcs or npc2 in assigned_npcs:
                continue
                
            # Check for compatible goals/traits
            traits_compatible = await self._check_trait_compatibility(npc1, npc2)
            
            if traits_compatible:
                coalition = {
                    "members": [npc1, npc2],
                    "strength": level / 100,  # Normalize to 0-1
                    "leader": npc1 if self._fetch_npc_dominance([npc1]).get(npc1, 0) > 
                               self._fetch_npc_dominance([npc2]).get(npc2, 0) else npc2
                }
                coalitions.append(coalition)
                assigned_npcs.add(npc1)
                assigned_npcs.add(npc2)
        
        return coalitions

      async def _check_trait_compatibility(self, npc1: int, npc2: int) -> bool:
        """Check if two NPCs have compatible traits for coalition formation."""
        compatible = False
        
        with get_db_connection() as conn, conn.cursor() as cursor:
            # Get traits for both NPCs
            cursor.execute("""
                SELECT npc_id, dominance, cruelty, personality_traits
                FROM NPCStats
                WHERE npc_id IN (%s, %s)
                AND user_id = %s
                AND conversation_id = %s
            """, (npc1, npc2, self.user_id, self.conversation_id))
            
            trait_data = {}
            for row in cursor.fetchall():
                npc_id, dominance, cruelty, personality_traits = row
                
                # Parse traits
                if isinstance(personality_traits, str):
                    try:
                        traits = json.loads(personality_traits)
                    except:
                        traits = []
                else:
                    traits = personality_traits or []
                    
                trait_data[npc_id] = {
                    "dominance": dominance,
                    "cruelty": cruelty,
                    "traits": traits
                }
        
        # Check basic compatibility rules
        if npc1 in trait_data and npc2 in trait_data:
            data1 = trait_data[npc1]
            data2 = trait_data[npc2]
            
            # Rule 1: Very similar dominance levels often don't work well together
            dom_diff = abs(data1["dominance"] - data2["dominance"])
            if dom_diff < 20 or dom_diff > 40:
                # Either clear hierarchy or too similar
                compatible = True
                
            # Rule 2: Shared cruelty trait creates bonds
            if abs(data1["cruelty"] - data2["cruelty"]) < 20:
                compatible = True
                
            # Rule 3: Check for complementary traits
            trait_pairs = [
                ("dominant", "submissive"),
                ("leader", "follower"),
                ("teacher", "student"),
                ("sadistic", "masochistic")
            ]
            
            for trait1, trait2 in trait_pairs:
                if trait1 in data1["traits"] and trait2 in data2["traits"]:
                    compatible = True
                    break
                if trait1 in data2["traits"] and trait2 in data1["traits"]:
                    compatible = True
                    break
        
        return compatible  

    async def update_relationship_from_interaction(
        self,
        entity_type: str,
        entity_id: int,
        player_action: Dict[str, Any],
        npc_action: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Update the relationship based on an interaction with enhanced
        error handling and memory integration.
        
        Args:
            entity_type: "npc" or "player"
            entity_id: The ID of the entity on the other side of the relationship
            player_action: A dict describing what the other entity did
            npc_action: A dict describing what the NPC did
            context: Additional context for the interaction
            
        Returns:
            Dictionary with update results
        """
        # Default return structure
        result = {
            "success": False,
            "link_id": None,
            "old_level": None,
            "new_level": None,
            "old_type": None,
            "new_type": None,
            "changes": {}
        }
        
        try:
            # Get memory system for beliefs and emotional context
            memory_system = await self._get_memory_system()
            
            # Get beliefs about this entity to influence relationship changes
            beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=self.npc_id,
                topic=f"{entity_type}_{entity_id}" if entity_type == "npc" else "player"
            )
            
            # Calculate belief adjustment factor
            belief_adjustment = 0
            for belief in beliefs:
                belief_text = belief.get("belief", "").lower()
                confidence = belief.get("confidence", 0.5)
                
                # Positive beliefs make relationship changes more favorable
                if any(word in belief_text for word in ["trust", "friend", "like", "positive"]):
                    belief_adjustment += confidence * 2
                # Negative beliefs make relationship changes more negative
                elif any(word in belief_text for word in ["distrust", "enemy", "dislike", "negative"]):
                    belief_adjustment -= confidence * 2
            
            # Get emotional state to influence relationship changes
            emotional_state = None
            try:
                emotional_state = await memory_system.get_npc_emotion(self.npc_id)
            except Exception as e:
                logger.error(f"Error getting emotional state: {e}")
                
            # Calculate emotional adjustment factor
            emotional_adjustment = 0
            if emotional_state and "current_emotion" in emotional_state:
                current_emotion = emotional_state["current_emotion"]
                primary = current_emotion.get("primary", {})
                emotion_name = primary.get("name", "neutral")
                intensity = primary.get("intensity", 0.0)
                
                # Different emotions affect relationship changes differently
                if emotion_name == "joy":
                    emotional_adjustment += intensity * 3
                elif emotion_name == "anger":
                    emotional_adjustment -= intensity * 3
                elif emotion_name == "fear":
                    emotional_adjustment -= intensity * 2
                elif emotion_name == "sadness":
                    emotional_adjustment -= intensity * 1
            
            # Get context information
            context_obj = context or {}
            interaction_environment = context_obj.get("environment", {})
            location = interaction_environment.get("location", "Unknown")
            
            # Record all factors that influence the relationship change
            change_factors = {
                "belief_adjustment": belief_adjustment,
                "emotional_adjustment": emotional_adjustment,
                "location": location,
                "action_types": {
                    "player": player_action.get("type", "unknown"),
                    "npc": npc_action.get("type", "unknown")
                }
            }
            
            # Create a connection and transaction
            conn = get_db_connection()
            cursor = conn.cursor()
            
            try:
                # Begin transaction
                conn.begin()
                
                # 1) Check if a social link record already exists
                cursor.execute("""
                    SELECT link_id, link_type, link_level
                    FROM SocialLinks
                    WHERE entity1_type = 'npc'
                      AND entity1_id = %s
                      AND entity2_type = %s
                      AND entity2_id = %s
                      AND user_id = %s
                      AND conversation_id = %s
                """, (
                    self.npc_id, entity_type, entity_id,
                    self.user_id, self.conversation_id
                ))
                row = cursor.fetchone()
    
                if row:
                    link_id, link_type, link_level = row
                    result["link_id"] = link_id
                    result["old_level"] = link_level
                    result["old_type"] = link_type
                else:
                    # Create a new relationship if none exists
                    cursor.execute("""
                        INSERT INTO SocialLinks (
                            entity1_type, entity1_id,
                            entity2_type, entity2_id,
                            link_type, link_level,
                            user_id, conversation_id
                        )
                        VALUES (
                            'npc', %s,
                            %s, %s,
                            'neutral', 0,
                            %s, %s
                        )
                        RETURNING link_id
                    """, (
                        self.npc_id, entity_type, entity_id,
                        self.user_id, self.conversation_id
                    ))
                    link_id = cursor.fetchone()[0]
                    result["link_id"] = link_id
                    link_type = "neutral"
                    link_level = 0
                    result["old_level"] = 0
                    result["old_type"] = "neutral"
    
                # 2) Calculate level changes
                level_change = 0
    
                # Base relationship changes
                if player_action.get("type") == "talk" and npc_action.get("type") == "talk":
                    level_change += 1
                elif player_action.get("type") == "talk" and npc_action.get("type") == "leave":
                    level_change -= 1
                elif player_action.get("type") == "talk" and npc_action.get("type") == "mock":
                    level_change -= 2
                
                # Add more complex interaction rules
                if player_action.get("type") == "help":
                    level_change += 3
                elif player_action.get("type") == "gift":
                    level_change += 4
                elif player_action.get("type") == "insult":
                    level_change -= 4
                elif player_action.get("type") == "threaten":
                    level_change -= 5
                elif player_action.get("type") == "attack":
                    level_change -= 8
                
                # Apply mask slip effects if present
                if "mask_slippage" in npc_action:
                    # Mask slippages can cause more dramatic relationship changes
                    slip_severity = npc_action["mask_slippage"].get("severity", 1)
                    if slip_severity >= 3:  # Major slips have bigger impacts
                        if level_change > 0:
                            level_change = level_change * 2  # Amplify positive changes
                        else:
                            level_change = level_change * 2  # Amplify negative changes
                
                # Record base level change
                change_factors["base_level_change"] = level_change
                
                # Apply belief and emotional adjustments
                final_level_change = level_change
                if level_change > 0:
                    # For positive changes, positive beliefs/emotions amplify
                    final_level_change += belief_adjustment + emotional_adjustment
                else:
                    # For negative changes, negative beliefs/emotions amplify
                    final_level_change += belief_adjustment - emotional_adjustment
                
                # Round to integer, preventing tiny changes
                final_level_change = round(final_level_change)
                
                # Record final change
                change_factors["final_level_change"] = final_level_change
                result["changes"] = change_factors
    
                # 3) Apply changes
                new_level = link_level
                if final_level_change != 0:
                    new_level = max(0, min(100, link_level + final_level_change))
                    cursor.execute("""
                        UPDATE SocialLinks
                        SET link_level = %s
                        WHERE link_id = %s
                    """, (new_level, link_id))
                    result["new_level"] = new_level
                
                # Determine new link type based on level
                new_link_type = link_type
                if new_level > 75:
                    new_link_type = "close"
                elif new_level > 50:
                    new_link_type = "friendly"
                elif new_level < 25:
                    new_link_type = "hostile"
                
                if new_link_type != link_type:
                    cursor.execute("""
                        UPDATE SocialLinks
                        SET link_type = %s
                        WHERE link_id = %s
                    """, (new_link_type, link_id))
                    result["new_type"] = new_link_type
    
                # 4) Add event to the link history
                change_description = []
                if abs(level_change) > 0:
                    change_description.append(f"base:{level_change:+d}")
                if abs(belief_adjustment) > 0:
                    change_description.append(f"beliefs:{belief_adjustment:+.1f}")
                if abs(emotional_adjustment) > 0:
                    change_description.append(f"emotions:{emotional_adjustment:+.1f}")
                
                change_str = ", ".join(change_description)
                
                event_text = (
                    f"Interaction: {entity_type.capitalize()} {player_action.get('description','???')}, "
                    f"NPC {npc_action.get('description','???')}. "
                    f"Relationship change: {link_level} → {new_level} ({final_level_change:+d}) [Factors: {change_str}]"
                )
                
                cursor.execute("""
                    UPDATE SocialLinks
                    SET link_history = COALESCE(link_history, '[]'::jsonb) || %s::jsonb
                    WHERE link_id = %s
                """, (json.dumps([event_text]), link_id))
    
                # Commit the transaction
                conn.commit()
                result["success"] = True
                
                logger.debug(
                    "Updated relationship for NPC %s -> entity (%s:%s). "
                    "Change: level=%d => %d, type_change=%s",
                    self.npc_id, entity_type, entity_id,
                    link_level, new_level, new_link_type if new_link_type != link_type else None
                )
    
                # 5) Create a memory of this relationship change - outside transaction for safety
                try:
                    # Only create memories for significant changes
                    if abs(final_level_change) >= 3 or new_link_type != link_type:
                        entity_name = "the player"
                        if entity_type == "npc":
                            # Get NPC name
                            cursor.execute("""
                                SELECT npc_name FROM NPCStats
                                WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                            """, (entity_id, self.user_id, self.conversation_id))
                            name_row = cursor.fetchone()
                            if name_row:
                                entity_name = name_row[0]
                        
                        direction = "improved" if final_level_change > 0 else "worsened"
                        if new_link_type != link_type:
                            memory_text = f"My relationship with {entity_name} changed to {new_link_type} (level {new_level})"
                        else:
                            memory_text = f"My relationship with {entity_name} {direction} (now level {new_level})"
                        
                        # Create memory with appropriate tags and importance
                        importance = "medium" if abs(final_level_change) >= 5 or new_link_type != link_type else "low"
                        
                        await memory_system.remember(
                            entity_type="npc",
                            entity_id=self.npc_id,
                            memory_text=memory_text,
                            importance=importance,
                            tags=["relationship_change", entity_type]
                        )
                        
                        # Update beliefs based on relationship changes
                        await self._update_beliefs_from_relationship_change(
                            entity_type, entity_id, entity_name, 
                            link_level, new_level, final_level_change
                        )
                except Exception as memory_error:
                    # Don't fail the whole operation if memory creation fails
                    logger.error(f"Error creating relationship memory: {memory_error}")
    
            except Exception as sql_error:
                # Roll back transaction on database errors
                conn.rollback()
                logger.error(f"Database error updating relationship: {sql_error}")
                result["error"] = str(sql_error)
            finally:
                # Always clean up cursor and connection
                cursor.close()
                conn.close()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in update_relationship_from_interaction: {e}")
            result["error"] = str(e)
            return result

    async def _update_npc_belief_from_interaction(
        self,
        npc_id: int,
        target_id: int,
        positive: bool,
        interaction_type: str
    ) -> None:
        """
        Update an NPC's beliefs about another NPC based on their interaction.
        
        Args:
            npc_id: ID of the NPC forming/updating the belief
            target_id: ID of the NPC the belief is about
            positive: Whether the interaction was positive or negative
            interaction_type: Type of interaction that occurred
        """
        try:
            memory_system = await self._get_memory_system()
            
            # Get NPC target name
            target_name = await self._get_npc_name(target_id)
            
            # Get existing beliefs
            existing_beliefs = await memory_system.get_beliefs(
                entity_type="npc",
                entity_id=npc_id,
                topic=f"npc_{target_id}"
            )
            
            # Determine if we should update existing belief or create new one
            belief_to_update = None
            if existing_beliefs:
                # Look for relevant beliefs to update
                for belief in existing_beliefs:
                    if positive and ("friend" in belief.get("belief", "").lower() or 
                                    "ally" in belief.get("belief", "").lower() or
                                    "trust" in belief.get("belief", "").lower()):
                        belief_to_update = belief
                        break
                    elif not positive and ("enemy" in belief.get("belief", "").lower() or 
                                          "rival" in belief.get("belief", "").lower() or
                                          "distrust" in belief.get("belief", "").lower()):
                        belief_to_update = belief
                        break
            
            if belief_to_update:
                # Update confidence in existing belief
                old_confidence = belief_to_update.get("confidence", 0.5)
                # Increase confidence by 0.1, capped at 0.95
                new_confidence = min(0.95, old_confidence + 0.1)
                
                await memory_system.update_belief_confidence(
                    belief_id=belief_to_update.get("id"),
                    entity_type="npc",
                    entity_id=npc_id,
                    new_confidence=new_confidence,
                    reason=f"Reinforced by {interaction_type} interaction"
                )
            else:
                # Create new belief
                if positive:
                    belief_text = f"{target_name} is someone I can trust or ally with."
                else:
                    belief_text = f"{target_name} is someone I should be cautious around."
                
                await memory_system.create_belief(
                    entity_type="npc",
                    entity_id=npc_id,
                    belief_text=belief_text,
                    confidence=0.6  # Initial confidence based on a single interaction
                )
                
        except Exception as e:
            logger.error(f"Error updating NPC belief from interaction: {e}")

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
                
                # Determine the emotional content of the memory
                importance = "medium"
                emotional = action_type in ["emotional_outburst", "mask_slip"]
                
                # Create more detailed memory for emotional events
                if emotional:
                    importance = "high"
                    # Use emotional memory system for emotional events
                    
                    # Analyze the emotional content
                    emotional_content = {
                        "primary_emotion": "neutral",
                        "intensity": 0.5
                    }
                    
                    if action_type == "emotional_outburst":
                        if "anger" in action_desc.lower():
                            emotional_content["primary_emotion"] = "anger"
                            emotional_content["intensity"] = 0.8
                        elif "fear" in action_desc.lower():
                            emotional_content["primary_emotion"] = "fear"
                            emotional_content["intensity"] = 0.7
                        elif "happiness" in action_desc.lower() or "joy" in action_desc.lower():
                            emotional_content["primary_emotion"] = "joy"
                            emotional_content["intensity"] = 0.8
                    
                    # Create emotional memory
                    await memory_system.remember(
                        entity_type="npc",
                        entity_id=npc_id,
                        memory_text=actor_memory_text,
                        importance=importance,
                        emotional=True,
                        tags=["group_interaction", action_type, "emotional"]
                    )
                else:
                    # Create standard memory
                    await memory_system.remember(
                        entity_type="npc",
                        entity_id=npc_id,
                        memory_text=actor_memory_text,
                        importance=importance,
                        emotional=False,
                        tags=["group_interaction", action_type]
                    )
                
                # Create memories for other NPCs of observing this action
                for observer_id in npc_ids:
                    if observer_id != npc_id:
                        observer_memory_text = f"{actor_name} {action_desc} to our group at {location}"
                        
                        # Determine importance based on relationship
                        rel_level = self._get_relationship_level(observer_id, npc_id)
                        observer_importance = "medium" if rel_level and rel_level > 50 else "low"
                        
                        # Observers with strong relationships to actor should form stronger memories
                        await memory_system.remember(
                            entity_type="npc",
                            entity_id=observer_id,
                            memory_text=observer_memory_text,
                            importance=observer_importance,
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
                    
                    # Determine emotional content based on action type
                    emotional = action_type in ["mock", "challenge", "support", "confide"]
                    importance = "high" if emotional else "medium"
                    
                    # Create memory for the actor
                    actor_memory_text = f"I {action_desc} to {target_name} during our group interaction at {location}"
                    
                    await memory_system.remember(
                        entity_type="npc",
                        entity_id=actor_id,
                        memory_text=actor_memory_text,
                        importance=importance,
                        emotional=emotional,
                        tags=["group_interaction", action_type]
                    )
                    
                    # Create memory for the target - more emotional for the target
                    target_memory_text = f"{actor_name} {action_desc} to me during our group interaction at {location}"
                    
                    # Target experiences the action more strongly
                    target_importance = "high" if emotional else "medium"
                    
                    await memory_system.remember(
                        entity_type="npc",
                        entity_id=target_id,
                        memory_text=target_memory_text,
                        importance=target_importance,
                        emotional=emotional,
                        tags=["group_interaction", "targeted"]
                    )
                    
                    # Create memories for observers
                    for observer_id in npc_ids:
                        if observer_id != actor_id and observer_id != target_id:
                            observer_memory_text = f"{actor_name} {action_desc} to {target_name} during our group interaction"
                            
                            # Calculate observer importance based on relationships
                            actor_rel = self._get_relationship_level(observer_id, actor_id) or 0
                            target_rel = self._get_relationship_level(observer_id, target_id) or 0
                            
                            # Observers care more if they have strong relationships with either party
                            rel_strength = max(actor_rel, target_rel)
                            observer_importance = "medium" if rel_strength > 50 else "low"
                            
                            await memory_system.remember(
                                entity_type="npc",
                                entity_id=observer_id,
                                memory_text=observer_memory_text,
                                importance=observer_importance,
                                tags=["group_interaction", "observation"]
                            )

    async def _check_for_mask_slippage(
        self,
        npc_ids: List[int],
        action_plan: Dict[str, Any],
        shared_context: Dict[str, Any]
    ) -> None:
        """
        Check for opportunities for mask slippage during group interactions.
        
        Args:
            npc_ids: List of NPC IDs in the group
            action_plan: The resolved action plan
            shared_context: Context information
        """
        memory_system = await self._get_memory_system()
        
        # Focus on NPCs with mask slip actions
        for group_action in action_plan.get("group_actions", []):
            action = group_action.get("action", {})
            npc_id = group_action.get("npc_id")
            
            if action.get("type") == "mask_slip" and npc_id:
                # Generate mask slippage event
                try:
                    # Get trigger from context
                    trigger = f"group interaction at {shared_context.get('location', 'Unknown')}"
                    
                    # Generate mask slippage using the memory system
                    slippage = await memory_system.reveal_npc_trait(
                        npc_id=npc_id,
                        trigger=trigger
                    )
                    
                    # Update cached mask info if slippage occurred
                    if slippage and "integrity_after" in slippage:
                        # Clear cached mask info to ensure we get fresh data next time
                        if npc_id in self._mask_states:
                            del self._mask_states[npc_id]
                        
                except Exception as e:
                    logger.error(f"Error generating mask slippage for NPC {npc_id}: {e}")
        
        # Check each NPC for random mask slippage based on actions
        for npc_id in npc_ids:
            # Only check NPCs not already processed above
            if npc_id not in [ga.get("npc_id") for ga in action_plan.get("group_actions", []) 
                             if ga.get("action", {}).get("type") == "mask_slip"]:
                try:
                    # Get current mask status
                    mask_info = await self._get_npc_mask(npc_id)
                    
                    if mask_info and mask_info.get("integrity", 100) < 70:
                        # Higher chance of slippage as integrity decreases
                        slippage_chance = (70 - mask_info.get("integrity", 100)) / 200
                        
                        # Check for emotional actions that might increase slippage chance
                        for individual_actions in action_plan.get("individual_actions", {}).values():
                            for action in individual_actions:
                                if action.get("target") == str(npc_id) and action.get("type") in ["mock", "challenge"]:
                                    # Being targeted increases slippage chance
                                    slippage_chance += 0.1
                                    break
                        
                        # Roll for slippage
                        if random.random() < slippage_chance:
                            # Generate mask slippage
                            trigger = f"stress during group interaction at {shared_context.get('location', 'Unknown')}"
                            
                            await memory_system.reveal_npc_trait(
                                npc_id=npc_id,
                                trigger=trigger
                            )
                            
                            # Clear cached mask info
                            if npc_id in self._mask_states:
                                del self._mask_states[npc_id]
                            
                except Exception as e:
                    logger.error(f"Error checking for random mask slippage for NPC {npc_id}: {e}")

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
        
        # Add memory-based context enhancements
        memory_system = await self._get_memory_system()
        
        # Get emotional states for all NPCs
        emotional_states = {}
        for npc_id in affected_npcs:
            try:
                emotional_state = await self._get_npc_emotional_state(npc_id)
                if emotional_state:
                    emotional_states[npc_id] = emotional_state
            except Exception as e:
                logger.error(f"Error getting emotional state for NPC {npc_id}: {e}")
        
        enhanced_context["emotional_states"] = emotional_states
        
        # Get mask information for all NPCs
        mask_states = {}
        for npc_id in affected_npcs:
            try:
                mask_info = await self._get_npc_mask(npc_id)
                if mask_info:
                    mask_states[npc_id] = mask_info
            except Exception as e:
                logger.error(f"Error getting mask info for NPC {npc_id}: {e}")
        
        enhanced_context["mask_states"] = mask_states
        
        # Get player-related beliefs for all NPCs
        npc_beliefs = {}
        for npc_id in affected_npcs:
            try:
                beliefs = await memory_system.get_beliefs(
                    entity_type="npc",
                    entity_id=npc_id,
                    topic="player"
                )
                if beliefs:
                    npc_beliefs[npc_id] = beliefs
            except Exception as e:
                logger.error(f"Error getting player beliefs for NPC {npc_id}: {e}")
        
        enhanced_context["npc_beliefs"] = npc_beliefs
        
        # If many NPCs are affected, add group dynamics to context
        if len(affected_npcs) > 1:
            # Find dominant NPCs
            npc_dominance = self._fetch_npc_dominance(affected_npcs)
            sorted_by_dominance = sorted(affected_npcs, key=lambda id_: npc_dominance.get(id_, 0), reverse=True)
            
            enhanced_context["group_dynamics"] = {
                "dominant_npc_id": sorted_by_dominance[0] if sorted_by_dominance else None,
                "participant_count": len(affected_npcs),
                "dominance_order": sorted_by_dominance
            }

        # Handle potential flashbacks triggered by player action
        flashbacks = {}
        for npc_id in affected_npcs:
            try:
                # Check for flashback with 15% chance
                if random.random() < 0.15:
                    context_text = player_action.get("description", "")
                    flashback = await memory_system.npc_flashback(npc_id, context_text)
                    if flashback:
                        flashbacks[npc_id] = flashback
            except Exception as e:
                logger.error(f"Error checking flashback for NPC {npc_id}: {e}")
        
        enhanced_context["flashbacks"] = flashbacks

        # Process player action for each NPC
        response_tasks = []
        for npc_id in affected_npcs:
            agent = self.active_agents.get(npc_id)
            if agent:
                # Pass the enhanced context to each agent
                npc_context = enhanced_context.copy()
                # Add NPC-specific context elements
                npc_context["flashback"] = flashbacks.get(npc_id)
                npc_context["emotional_state"] = emotional_states.get(npc_id)
                npc_context["mask_info"] = mask_states.get(npc_id)
                npc_context["player_beliefs"] = npc_beliefs.get(npc_id)
                
                response_tasks.append(agent.process_player_action(player_action, npc_context))
            else:
                response_tasks.append(asyncio.sleep(0))

        responses = await asyncio.gather(*response_tasks)
        
        # Check for mask slippage due to player action
        for npc_id in affected_npcs:
            try:
                # Higher chance if action is challenging or emotional
                action_type = player_action.get("type", "").lower()
                is_challenging = action_type in ["challenge", "mock", "accuse", "threaten"]
                
                slippage_chance = 0.05  # Base chance
                if is_challenging:
                    slippage_chance = 0.15
                
                # Increase chance for NPCs with compromised masks
                mask_info = mask_states.get(npc_id)
                if mask_info:
                    integrity = mask_info.get("integrity", 100)
                    if integrity < 70:
                        slippage_chance += (70 - integrity) / 200
                
                # Roll for slippage
                if random.random() < slippage_chance:
                    # Generate mask slippage
                    await memory_system.reveal_npc_trait(
                        npc_id=npc_id,
                        trigger=player_action.get("description", "player action")
                    )
                    
                    # Clear cached mask info
                    if npc_id in self._mask_states:
                        del self._mask_states[npc_id]
            except Exception as e:
                logger.error(f"Error processing mask slippage for NPC {npc_id}: {e}")
        
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
                
                # Determine emotional impact
                emotional = False
                if is_challenging or "emotion" in action_type:
                    emotional = True
                
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=npc_id,
                    memory_text=memory_text,
                    importance="medium",
                    emotional=emotional,
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
    # Memory-specific helper methods
    # ----------------------------------------------------------------
    
    async def _get_npc_emotional_state(self, npc_id: int) -> Dict[str, Any]:
        """
        Get an NPC's current emotional state, with caching for performance.
        """
        # Check cache first
        if npc_id in self._emotional_states:
            return self._emotional_states[npc_id]
            
        try:
            memory_system = await self._get_memory_system()
            emotional_state = await memory_system.get_npc_emotion(npc_id)
            
            # Cache the result
            self._emotional_states[npc_id] = emotional_state
            return emotional_state
        except Exception as e:
            logger.error(f"Error getting emotional state for NPC {npc_id}: {e}")
            return None
    
    async def _get_npc_mask(self, npc_id: int) -> Dict[str, Any]:
        """
        Get an NPC's mask information, with caching for performance.
        """
        # Check cache first
        if npc_id in self._mask_states:
            return self._mask_states[npc_id]
            
        try:
            memory_system = await self._get_memory_system()
            mask_info = await memory_system.get_npc_mask(npc_id)
            
            # Cache the result
            self._mask_states[npc_id] = mask_info
            return mask_info
        except Exception as e:
            logger.error(f"Error getting mask info for NPC {npc_id}: {e}")
            return None
    
    async def _get_npc_name(self, npc_id: int) -> str:
        """Get an NPC's name."""
        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
                SELECT npc_name FROM NPCStats
                WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
            """, (npc_id, self.user_id, self.conversation_id))
            row = cursor.fetchone()
            if row:
                return row[0]
            return f"NPC_{npc_id}"

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
