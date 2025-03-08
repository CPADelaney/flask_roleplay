# logic/npc_agents/agent_coordinator.py

"""
Coordinates multiple NPC agents for group interactions, with improved memory integration.
"""

import logging
import asyncio
import random
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

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
                    loaded_ids.append(npc_id)

            logger.info("Loaded agents: %s", loaded_ids)
            return loaded_ids
        except Exception as e:
            logger.error(f"Error loading agents: {e}")
            return []
    
    async def make_group_decisions(
        self,
        npc_ids: List[int],
        shared_context: Dict[str, Any],
        available_actions: Optional[Dict[int, List[NPCAction]]] = None
    ) -> Dict[str, Any]:
        """
        Coordinate decision-making for a group of NPCs with improved memory integration.
        
        Args:
            npc_ids: List of NPC IDs
            shared_context: Common context for all NPCs
            available_actions: Optional pre-defined actions for each NPC
            
        Returns:
            Dictionary with group action plan
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
        
        # Add NPC-specific context information
        if "npc_context" not in enhanced_context:
            enhanced_context["npc_context"] = {}
        
        # Process each NPC in parallel for better performance
        async def prepare_npc_context(npc_id):
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
            
            return {
                "npc_id": npc_id,
                "group_memories": group_memories.get("memories", []),
                "emotional_state": emotional_state,
                "mask_info": mask_info,
                "flashback": flashback,
                "beliefs": beliefs
            }
        
        # Gather all NPC context data concurrently
        tasks = [prepare_npc_context(npc_id) for npc_id in npc_ids]
        npc_contexts = await asyncio.gather(*tasks)
        
        # Store in enhanced context
        for npc_context in npc_contexts:
            npc_id = npc_context.pop("npc_id")
            enhanced_context["npc_context"][npc_id] = npc_context
        
        return enhanced_context
    
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
            return None
    
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
            return None
    
    async def _get_npc_beliefs_about_others(self, npc_id: int, npc_ids: List[int]) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get an NPC's beliefs about other NPCs in the group.
        
        Args:
            npc_id: ID of the NPC
            npc_ids: List of other NPC IDs
            
        Returns:
            Dictionary mapping NPC IDs to lists of beliefs
        """
        memory_system = await self._get_memory_system()
        beliefs_by_npc = {}
        
        # Get beliefs about each other NPC
        for other_id in npc_ids:
            if other_id != npc_id:
                beliefs = await memory_system.get_beliefs(
                    entity_type="npc",
                    entity_id=npc_id,
                    topic=f"npc_{other_id}"
                )
                
                if beliefs:
                    beliefs_by_npc[other_id] = beliefs
        
        return beliefs_by_npc
    
    async def _create_empty_perception_future(self):
        """Create an empty perception result for missing agents."""
        return {
            "environment": {},
            "relevant_memories": [],
            "relationships": {},
            "timestamp": datetime.now().isoformat(),
            "missing_agent": True  # Flag to identify this as a placeholder
        }

    async def _gather_individual_perceptions(self, npc_ids: List[int], enhanced_context: Dict[str, Any]) -> Dict[int, Any]:
        """
        Gather perceptions from all NPCs concurrently for performance.
        
        Args:
            npc_ids: List of NPC IDs
            enhanced_context: Enhanced context with memory integration
            
        Returns:
            Dictionary mapping NPC IDs to perceptions
        """
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
                perception_tasks.append(self._create_empty_perception_future())
        
        # Gather results concurrently
        perceptions_list = await asyncio.gather(*perception_tasks)
        return {npc_id: perceptions_list[i] for i, npc_id in enumerate(npc_ids)}
    
    async def _collect_individual_decisions(
        self, 
        npc_ids: List[int], 
        perceptions: Dict[int, Any],
        available_actions: Dict[int, List[NPCAction]]
    ) -> Dict[int, Any]:
        """
        Collect decisions from each NPC concurrently.
        
        Args:
            npc_ids: List of NPC IDs
            perceptions: Dictionary of NPC perceptions
            available_actions: Dictionary of available actions for each NPC
            
        Returns:
            Dictionary mapping NPC IDs to decisions
        """
        decision_tasks = []
        
        for npc_id in npc_ids:
            agent = self.active_agents.get(npc_id)
            npc_actions = available_actions.get(npc_id, [])
            
            if agent:
                decision_tasks.append(agent.make_decision(perceptions[npc_id], npc_actions))
            else:
                decision_tasks.append(asyncio.sleep(0))  # Empty placeholder task
        
        # Gather results concurrently
        decisions_list = await asyncio.gather(*decision_tasks)
        return {npc_id: decisions_list[i] for i, npc_id in enumerate(npc_ids)}
    
    async def _resolve_group_conflicts(self, decisions, npc_ids, perceptions):
        """
        Resolve conflicts in group decisions.
        
        Args:
            decisions: Dictionary of NPC decisions
            npc_ids: List of NPC IDs
            perceptions: Dictionary of NPC perceptions
            
        Returns:
            Resolved action plan
        """
        try:
            # This should call the existing method with proper parameters
            return await self.resolve_decision_conflicts(decisions, npc_ids, perceptions)
        except Exception as e:
            logger.error(f"Error resolving group conflicts: {e}")
            # Return a minimal fallback action plan
            return {
                "group_actions": [],
                "individual_actions": {}
            }

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

        Args:
            decisions: Dictionary mapping NPC IDs to their decisions
            npc_ids: List of NPC IDs
            perceptions: Dictionary of NPC perceptions
            
        Returns:
            A structured dict with group_actions and individual_actions.
        """
        # Get memory system
        memory_system = await self._get_memory_system()
        
        # Get NPC dominance values
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

        # Detect coalitions among NPCs
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
        await self._apply_mask_modifiers_to_dominance(npc_ids, npc_dominance)
        
        # Apply modifiers based on trauma or significant memories
        await self._apply_trauma_modifiers_to_dominance(npc_ids, npc_dominance, perceptions)
        
        # Sort by modified dominance descending
        sorted_npcs = sorted(npc_ids, key=lambda id_: npc_dominance.get(id_, 0), reverse=True)

        # Create the action plan structure
        action_plan = {"group_actions": [], "individual_actions": {}}
        affected_npcs: Set[int] = set()

        # Apply actions in order of dominance
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
    
    async def _apply_mask_modifiers_to_dominance(self, npc_ids: List[int], npc_dominance: Dict[int, int]) -> None:
        """
        Apply modifiers to dominance based on mask integrity and traits.
        
        Args:
            npc_ids: List of NPC IDs
            npc_dominance: Dictionary to update with modified dominance values
        """
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
    
    async def _apply_trauma_modifiers_to_dominance(
        self, 
        npc_ids: List[int], 
        npc_dominance: Dict[int, int],
        perceptions: Dict[int, Any]
    ) -> None:
        """
        Apply modifiers to dominance based on trauma and significant memories.
        
        Args:
            npc_ids: List of NPC IDs
            npc_dominance: Dictionary to update with modified dominance values
            perceptions: Dictionary of NPC perceptions
        """
        memory_system = await self._get_memory_system()
        
        for npc_id in npc_ids:
            try:
                # Check for traumatic memories that might be influencing current behavior
                location = perceptions.get(npc_id, {}).get("environment", {}).get("location", "Unknown")
                
                trauma_memories = await memory_system.recall(
                    entity_type="npc",
                    entity_id=npc_id,
                    query="traumatic event",
                    context={"location": location},
                    limit=1
                )
                
                if trauma_memories and trauma_memories.get("memories"):
                    # Trauma can reduce effective dominance in triggering contexts
                    npc_dominance[npc_id] = max(0, npc_dominance.get(npc_id, 50) - 15)
            except Exception as e:
                logger.error(f"Error applying trauma modifiers for NPC {npc_id}: {e}")

    async def _detect_potential_coalitions(
        self,
        npc_ids: List[int],
        perceptions: Dict[int, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect potential coalitions between NPCs based on aligned interests.
        
        Args:
            npc_ids: List of NPC IDs
            perceptions: Dictionary of NPC perceptions
            
        Returns:
            List of coalition dictionaries
        """
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
        """
        Check if two NPCs have compatible traits for coalition formation.
        
        Args:
            npc1: First NPC ID
            npc2: Second NPC ID
            
        Returns:
            True if traits are compatible, False otherwise
        """
        compatibility_score = 0.0
        
        try:
            with get_db_connection() as conn, conn.cursor() as cursor:
                # Get traits for both NPCs
                cursor.execute("""
                    SELECT npc_id, dominance, cruelty, personality_traits, 
                           hobbies, affiliations
                    FROM NPCStats
                    WHERE npc_id IN (%s, %s)
                    AND user_id = %s
                    AND conversation_id = %s
                """, (npc1, npc2, self.user_id, self.conversation_id))
                
                trait_data = {}
                for row in cursor.fetchall():
                    npc_id, dominance, cruelty, personality_traits, hobbies, affiliations = row
                    
                    # Parse traits with error handling
                    traits = self._parse_json_field(personality_traits, [])
                    hobbies_list = self._parse_json_field(hobbies, [])
                    affiliations_list = self._parse_json_field(affiliations, [])
                    
                    trait_data[npc_id] = {
                        "dominance": dominance,
                        "cruelty": cruelty,
                        "traits": traits,
                        "hobbies": hobbies_list,
                        "affiliations": affiliations_list
                    }
            
            # If missing data for either NPC, can't form coalition
            if npc1 not in trait_data or npc2 not in trait_data:
                return False
                
            data1 = trait_data[npc1]
            data2 = trait_data[npc2]
            
            # 1. Dominance compatibility
            dom_diff = abs(data1["dominance"] - data2["dominance"])
            if dom_diff < 20:
                # Similar dominance often creates friction
                compatibility_score -= 1.0
            elif dom_diff > 40:
                # Clear dominance hierarchy works well
                compatibility_score += 1.5
                
            # 2. Cruelty similarity (shared values)
            cruelty_diff = abs(data1["cruelty"] - data2["cruelty"])
            if cruelty_diff < 20:
                compatibility_score += 1.0
            else:
                compatibility_score -= 0.5
            
            # 3. Check for complementary trait pairs
            complementary_pairs = [
                ("dominant", "submissive"),
                ("leader", "follower"),
                ("teacher", "student"),
                ("sadistic", "masochistic"),
                ("protective", "dependent"),
                ("controlling", "obedient")
            ]
            
            for trait1, trait2 in complementary_pairs:
                if (trait1 in data1["traits"] and trait2 in data2["traits"]):
                    compatibility_score += 2.0
                    break
                if (trait1 in data2["traits"] and trait2 in data1["traits"]):
                    compatibility_score += 2.0
                    break
            
            # 4. Check for shared hobbies/interests
            shared_hobbies = set(data1["hobbies"]) & set(data2["hobbies"])
            compatibility_score += len(shared_hobbies) * 0.5
            
            # 5. Check for shared affiliations
            shared_affiliations = set(data1["affiliations"]) & set(data2["affiliations"])
            compatibility_score += len(shared_affiliations) * 0.7
            
            # 6. Check for incompatible trait combinations
            incompatible_pairs = [
                ("loner", "socialite"),
                ("honest", "manipulative"),
                ("loyal", "treacherous"),
                ("patient", "impulsive")
            ]
            
            for trait1, trait2 in incompatible_pairs:
                if ((trait1 in data1["traits"] and trait2 in data2["traits"]) or
                    (trait1 in data2["traits"] and trait2 in data1["traits"])):
                    compatibility_score -= 1.5
            
            # Get current relationship level to influence compatibility
            link_level = await self._get_relationship_level(npc1, npc2) or 0
            compatibility_factor = link_level / 25.0  # Scale to 0-4 range
            compatibility_score += compatibility_factor
            
            # Final decision - positive score indicates compatibility
            return compatibility_score > 1.0
        
        except Exception as e:
            logger.error(f"Error checking trait compatibility: {e}")
            return False
    
    def _parse_json_field(self, field, default=None):
        """Helper to parse JSON fields from DB rows."""
        if field is None:
            return default or []
            
        if isinstance(field, str):
            try:
                return json.loads(field)
            except json.JSONDecodeError:
                return default or []
                
        if isinstance(field, list):
            return field
            
        return default or []

    async def _create_group_memory_effects(
        self,
        npc_ids: List[int],
        action_plan: Dict[str, Any],
        shared_context: Dict[str, Any]
    ) -> None:
        """
        Create memories and handle emotional effects of the group interaction.
        
        Args:
            npc_ids: List of NPC IDs
            action_plan: Resolved action plan
            shared_context: Context information
        """
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
            
            # Get beliefs that might influence actions
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
            
            # Add dominance-based actions - important for femdom context
            if dom > 60:
                actions.append(
                    NPCAction(
                        type="command",
                        description="Give an authoritative command",
                        target="group",
                        stats_influenced={"dominance": +1, "trust": -1}
                    )
                )
                actions.append(
                    NPCAction(
                        type="test",
                        description="Test group's obedience",
                        target="group",
                        stats_influenced={"dominance": +2, "respect": -1}
                    )
                )
                
                # More intense femdom themed actions for high dominance
                if dom > 75:
                    actions.append(
                        NPCAction(
                            type="dominate",
                            description="Assert dominance forcefully",
                            target="group",
                            stats_influenced={"dominance": +3, "fear": +2}
                        )
                    )
                    actions.append(
                        NPCAction(
                            type="punish",
                            description="Punish disobedience",
                            target="group", 
                            stats_influenced={"fear": +3, "obedience": +2}
                        )
                    )
            
            # Add cruelty-based actions
            if cru > 60:
                actions.append(
                    NPCAction(
                        type="mock",
                        description="Mock or belittle the group",
                        target="group",
                        stats_influenced={"cruelty": +1, "closeness": -2}
                    )
                )
                
                # More intense femdom themed cruel actions
                if cru > 70:
                    actions.append(
                        NPCAction(
                            type="humiliate",
                            description="Deliberately humiliate the group",
                            target="group",
                            stats_influenced={"cruelty": +2, "fear": +2}
                        )
                    )
            
            # Add emotionally-influenced actions for strong emotions
            if emotional_state and "current_emotion" in emotional_state:
                current_emotion = emotional_state["current_emotion"]
                primary = current_emotion.get("primary", {})
                emotion_name = primary.get("name", "neutral")
                intensity = primary.get("intensity", 0.0)
                
                if intensity > 0.7:
                    if emotion_name == "anger":
                        actions.append(
                            NPCAction(
                                type="express_anger",
                                description="Express anger forcefully",
                                target="group",
                                stats_influenced={"dominance": +2, "closeness": -3}
                            )
                        )
                    elif emotion_name == "fear":
                        actions.append(
                            NPCAction(
                                type="act_defensive",
                                description="Act defensively and guarded",
                                target="environment",
                                stats_influenced={"trust": -2}
                            )
                        )
                    elif emotion_name == "joy":
                        actions.append(
                            NPCAction(
                                type="celebrate",
                                description="Share happiness enthusiastically",
                                target="group",
                                stats_influenced={"closeness": +3}
                            )
                        )
                    elif emotion_name == "arousal" or emotion_name == "desire":
                        # For femdom context
                        actions.append(
                            NPCAction(
                                type="seduce",
                                description="Make seductive advances",
                                target="group",
                                stats_influenced={"closeness": +2, "fear": +1}
                            )
                        )
            
            # Add actions based on mask information - reveal true nature if integrity is low
            mask_integrity = 100 if not mask_info else mask_info.get("integrity", 100)
            hidden_traits = {} if not mask_info else mask_info.get("hidden_traits", {})
            
            if mask_integrity < 70:
                # As mask breaks down, hidden traits show through
                if "dominant" in hidden_traits:
                    actions.append(
                        NPCAction(
                            type="mask_slip",
                            description="Show unexpected dominance",
                            target="group",
                            stats_influenced={"dominance": +3, "fear": +2}
                        )
                    )
                elif "cruel" in hidden_traits:
                    actions.append(
                        NPCAction(
                            type="mask_slip",
                            description="Reveal unexpected cruelty",
                            target="group",
                            stats_influenced={"cruelty": +2, "fear": +1}
                        )
                    )
                elif "submissive" in hidden_traits:
                    actions.append(
                        NPCAction(
                            type="mask_slip",
                            description="Show unexpected submission",
                            target="group",
                            stats_influenced={"dominance": -2}
                        )
                    )
            
            # Add actions based on beliefs
            if beliefs:
                for belief in beliefs:
                    belief_text = belief.get("belief", "").lower()
                    if "dangerous" in belief_text or "threat" in belief_text:
                        actions.append(
                            NPCAction(
                                type="defensive",
                                description="Take a defensive stance",
                                target="group",
                                stats_influenced={"trust": -2}
                            )
                        )
                    elif "opportunity" in belief_text or "beneficial" in belief_text:
                        actions.append(
                            NPCAction(
                                type="engage",
                                description="Actively engage with the group",
                                target="group",
                                stats_influenced={"closeness": +2}
                            )
                        )
            
            # Context-based actions
            environment_data = perceptions[npc_id].get("environment", {})
            loc_str = environment_data.get("location", "").lower()
            
            if any(loc in loc_str for loc in ["cafe", "restaurant", "bar", "party"]):
                actions.append(
                    NPCAction(
                        type="socialize",
                        description="Engage in group conversation",
                        target="group",
                        stats_influenced={"closeness": +1}
                    )
                )
            
            # Add target-specific actions for other NPCs
            for other_id in npc_ids:
                if other_id != npc_id and other_id in npc_data:
                    other_name = npc_data[other_id]["npc_name"]
                    
                    # Basic interaction
                    actions.append(
                        NPCAction(
                            type="talk_to",
                            description=f"Talk to {other_name}",
                            target=str(other_id),
                            target_name=other_name,
                            stats_influenced={"closeness": +1}
                        )
                    )
                    
                    # Get relationship with this NPC to influence actions
                    relationship_level = await self._get_relationship_level(npc_id, other_id)
                    
                    # High dominance actions
                    if dom > 60:
                        actions.append(
                            NPCAction(
                                type="command",
                                description=f"Command {other_name}",
                                target=str(other_id),
                                target_name=other_name,
                                stats_influenced={"dominance": +1, "trust": -1}
                            )
                        )
                    
                    # High cruelty actions
                    if cru > 60:
                        actions.append(
                            NPCAction(
                                type="mock",
                                description=f"Mock {other_name}",
                                target=str(other_id),
                                target_name=other_name,
                                stats_influenced={"cruelty": +1, "closeness": -2}
                            )
                        )
                    
                    # Relationship-based actions
                    if relationship_level is not None:
                        if relationship_level > 70:  # Friendly relationship
                            actions.append(
                                NPCAction(
                                    type="support",
                                    description=f"Support {other_name}",
                                    target=str(other_id),
                                    target_name=other_name,
                                    stats_influenced={"closeness": +2, "respect": +1}
                                )
                            )
                        elif relationship_level < 30:  # Antagonistic relationship
                            actions.append(
                                NPCAction(
                                    type="challenge",
                                    description=f"Challenge {other_name}",
                                    target=str(other_id),
                                    target_name=other_name,
                                    stats_influenced={"dominance": +2, "respect": -1}
                                )
                            )
            
            group_actions[npc_id] = actions

        return group_actions
    
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
                    belief_text = belief.get("belief", "").lower()
                    if positive and ("trust" in belief_text or "friend" in belief_text or 
                                    "ally" in belief_text or "like" in belief_text):
                        belief_to_update = belief
                        break
                    elif not positive and ("distrust" in belief_text or "enemy" in belief_text or 
                                          "rival" in belief_text or "dislike" in belief_text):
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
                    belief_text = f"{target_name} is someone I can trust or ally with"
                else:
                    belief_text = f"{target_name} is someone I should be cautious around"
                
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
        
        Args:
            npc_ids: List of NPC IDs
            action_plan: Resolved action plan
            shared_context: Context information
        """
        memory_system = await self._get_memory_system()
        
        # Get location and time information
        location = shared_context.get("location", "Unknown")
        time_of_day = shared_context.get("time_of_day", "Unknown")
        
        # Get NPC names
        npc_names = await self._get_npc_names(npc_ids)
        
        # Format participant list
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
                await self._create_observer_memories(
                    npc_id=npc_id,
                    observer_ids=[id for id in npc_ids if id != npc_id],
                    actor_name=actor_name,
                    action_desc=action_desc,
                    action_type=action_type,
                    location=location,
                    memory_system=memory_system
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
                    
                    # Create memories of this targeted interaction
                    await self._create_targeted_interaction_memories(
                        actor_id=actor_id,
                        actor_name=actor_name,
                        target_id=target_id,
                        target_name=target_name,
                        action_type=action_type,
                        action_desc=action_desc,
                        location=location,
                        observer_ids=[id for id in npc_ids if id not in (actor_id, target_id)],
                        memory_system=memory_system
                    )

    async def _create_observer_memories(
        self,
        npc_id: int,
        observer_ids: List[int],
        actor_name: str,
        action_desc: str,
        action_type: str,
        location: str,
        memory_system: Any
    ) -> None:
        """
        Create memories for NPCs observing an action.
        
        Args:
            npc_id: ID of the acting NPC
            observer_ids: List of observer NPC IDs
            actor_name: Name of the acting NPC
            action_desc: Description of the action
            action_type: Type of the action
            location: Location where the action occurred
            memory_system: Memory system instance
        """
        for observer_id in observer_ids:
            # Observer memory text
            observer_memory_text = f"{actor_name} {action_desc} to our group at {location}"
            
            # Determine importance based on relationship
            rel_level = await self._get_relationship_level(observer_id, npc_id)
            observer_importance = "medium" if rel_level and rel_level > 50 else "low"
            
            # Observers with strong relationships to actor should form stronger memories
            await memory_system.remember(
                entity_type="npc",
                entity_id=observer_id,
                memory_text=observer_memory_text,
                importance=observer_importance,
                tags=["group_interaction", "observation"]
            )

    async def _create_targeted_interaction_memories(
        self,
        actor_id: int,
        actor_name: str,
        target_id: int,
        target_name: str,
        action_type: str,
        action_desc: str,
        location: str,
        observer_ids: List[int],
        memory_system: Any
    ) -> None:
        """
        Create memories for a targeted interaction between two NPCs.
        
        Args:
            actor_id: ID of the acting NPC
            actor_name: Name of the acting NPC
            target_id: ID of the target NPC
            target_name: Name of the target NPC
            action_type: Type of the action
            action_desc: Description of the action
            location: Location where the action occurred
            observer_ids: List of observer NPC IDs
            memory_system: Memory system instance
        """
        # Determine emotional content based on action type
        emotional = action_type in ["mock", "challenge", "support", "confide", "dominate", "punish"]
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
        for observer_id in observer_ids:
            observer_memory_text = f"{actor_name} {action_desc} to {target_name} during our group interaction"
            
            # Calculate observer importance based on relationships
            actor_rel = await self._get_relationship_level(observer_id, actor_id) or 0
            target_rel = await self._get_relationship_level(observer_id, target_id) or 0
            
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
        
        # Focus on NPCs with mask slip actions first
        mask_slip_npcs = []
        for group_action in action_plan.get("group_actions", []):
            action = group_action.get("action", {})
            npc_id = group_action.get("npc_id")
            
            if action.get("type") == "mask_slip" and npc_id:
                mask_slip_npcs.append(npc_id)
                
                # Generate mask slippage event
                try:
                    # Get trigger from context
                    trigger = f"group interaction at {shared_context.get('location', 'Unknown')}"
                    
                    # Generate mask slippage
                    await memory_system.reveal_npc_trait(
                        npc_id=npc_id,
                        trigger=trigger
                    )
                    
                    # Invalidate cached mask info
                    if npc_id in self._mask_states:
                        del self._mask_states[npc_id]
                        del self._mask_states_timestamps[npc_id]
                        
                except Exception as e:
                    logger.error(f"Error generating mask slippage for NPC {npc_id}: {e}")
        
        # Get remaining NPCs to check
        remaining_npcs = [id for id in npc_ids if id not in mask_slip_npcs]
        
        # Check other NPCs for random mask slippage based on actions
        for npc_id in remaining_npcs:
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
                        
                        # Invalidate cached mask info
                        if npc_id in self._mask_states:
                            del self._mask_states[npc_id]
                            del self._mask_states_timestamps[npc_id]
                        
            except Exception as e:
                logger.error(f"Error checking for random mask slippage for NPC {npc_id}: {e}")

    async def _process_emotional_contagion(self, npc_ids: List[int], action_plan: Dict[str, Any]) -> None:
        """
        Process emotional contagion between NPCs during group interactions.
        Enhanced with relationship factors and physical proximity.
        
        Args:
            npc_ids: List of NPC IDs
            action_plan: The resolved action plan
        """
        memory_system = await self._get_memory_system()
        
        # Get emotional states for all NPCs
        emotional_states = {}
        for npc_id in npc_ids:
            emotional_state = await self._get_npc_emotional_state(npc_id)
            if emotional_state and "current_emotion" in emotional_state:
                emotional_states[npc_id] = emotional_state
        
        # Short-circuit if not enough NPCs with emotional states
        if len(emotional_states) < 2:
            return
        
        # Calculate dominance factors (affects emotional influence)
        dominance_factors = self._fetch_npc_dominance(npc_ids)
        
        # Get relationship factors (affects influence strength)
        relationship_factors = await self._get_group_relationship_factors(npc_ids)
        
        # Get physical proximity from action plan
        proximity_factors = await self._get_physical_proximity_factors(npc_ids, action_plan)
        
        # Get mental resilience stats (resistance to influence)
        mental_resistance = await self._get_mental_resilience_factors(npc_ids)
        
        # Process contagion for each NPC
        for affected_id in npc_ids:
            # Skip NPCs with no emotional data
            if affected_id not in emotional_states:
                continue
                
            affected_state = emotional_states[affected_id]
            affected_dominance = dominance_factors.get(affected_id, 50)
            
            # Calculate base susceptibility - lower dominance = more susceptible
            base_susceptibility = max(0.1, 1.0 - (affected_dominance / 100))
            
            # Adjust for mental resilience - higher resilience = less susceptible
            resistance = mental_resistance.get(affected_id, 50) / 100
            susceptibility = base_susceptibility * (1.0 - (resistance * 0.5))
            
            # Calculate weighted influence from other NPCs
            new_emotion = None
            max_influence = 0
            best_influencer = None
            
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
                    
                # Get relationship factor between these NPCs
                relationship_factor = relationship_factors.get(
                    (min(affected_id, influencer_id), max(affected_id, influencer_id)), 
                    0.5  # Default to moderate influence if no relationship
                )
                
                # Get proximity factor
                proximity_factor = proximity_factors.get(
                    (min(affected_id, influencer_id), max(affected_id, influencer_id)), 
                    0.8  # Default to moderate proximity if unknown
                )
                
                # Calculate base influence power based on dominance and emotion intensity
                base_influence = (influencer_dominance / 100) * intensity
                
                # Apply modifiers
                influence_power = base_influence * susceptibility * relationship_factor * proximity_factor
                
                # Special case: fear and anger are more contagious
                if emotion_name in ["fear", "anger"]:
                    influence_power *= 1.5
                    
                # Special case: intimate emotions transfer more between close NPCs
                if emotion_name in ["arousal", "desire", "affection"] and relationship_factor > 0.7:
                    influence_power *= 1.3
                
                # If this is the strongest influence so far, it affects the NPC
                if influence_power > max_influence:
                    max_influence = influence_power
                    best_influencer = influencer_id
                    new_emotion = {
                        "name": emotion_name,
                        "intensity": min(0.9, intensity * susceptibility)  # Cap at 0.9 intensity
                    }
            
            # Apply new emotion if sufficient influence
            if new_emotion and max_influence > 0.3:
                await memory_system.update_npc_emotion(
                    npc_id=affected_id,
                    emotion=new_emotion["name"],
                    intensity=new_emotion["intensity"]
                )
                
                # Create memory of being influenced
                if best_influencer is not None:
                    influencer_name = await self._get_npc_name(best_influencer)
                    memory_text = f"I felt {new_emotion['name']} because of {influencer_name}'s emotional state"
                    
                    await memory_system.remember(
                        entity_type="npc",
                        entity_id=affected_id,
                        memory_text=memory_text,
                        importance="low",
                        tags=["emotional_contagion", "social_influence"]
                    )
                    
                # Invalidate cached emotional state
                if affected_id in self._emotional_states:
                    del self._emotional_states[affected_id]
                    del self._emotional_states_timestamps[affected_id]
    
    async def _get_group_relationship_factors(self, npc_ids: List[int]) -> Dict[Tuple[int, int], float]:
        """
        Get relationship factors between all NPCs in the group.
        Higher values mean stronger emotional influence.
        
        Args:
            npc_ids: List of NPC IDs
            
        Returns:
            Dictionary mapping NPC ID pairs to relationship factors
        """
        relationship_factors = {}
        
        # Query all relationships between these NPCs
        for i, npc1 in enumerate(npc_ids):
            for npc2 in npc_ids[i+1:]:  # Only check each pair once
                # Get relationship level between these NPCs
                relationship_level = await self._get_relationship_level(npc1, npc2)
                
                if relationship_level is not None:
                    # Convert to 0.1-2.0 range for relationship factor
                    # Higher relationships = stronger influence
                    factor = 0.1 + (relationship_level / 100) * 1.9
                    
                    # Store with consistent key ordering
                    relationship_factors[(min(npc1, npc2), max(npc1, npc2))] = factor
                else:
                    # Default if no relationship exists
                    relationship_factors[(min(npc1, npc2), max(npc1, npc2))] = 0.5
        
        return relationship_factors
    
    async def _get_physical_proximity_factors(self, npc_ids: List[int], action_plan: Dict[str, Any]) -> Dict[Tuple[int, int], float]:
        """
        Calculate physical proximity factors based on group interaction.
        Higher values mean closer physical proximity (stronger influence).
        
        Args:
            npc_ids: List of NPC IDs
            action_plan: The resolved action plan
            
        Returns:
            Dictionary mapping NPC ID pairs to proximity factors
        """
        proximity_factors = {}
        
        # Default proximity is relatively high in group setting
        default_proximity = 0.8
        
        # Extract interaction patterns from action plan
        group_actions = action_plan.get("group_actions", [])
        individual_actions = action_plan.get("individual_actions", {})
        
        # Set default proximity for all pairs
        for i, npc1 in enumerate(npc_ids):
            for npc2 in npc_ids[i+1:]:
                proximity_factors[(min(npc1, npc2), max(npc1, npc2))] = default_proximity
        
        # Adjust based on interactions
        for action in group_actions:
            npc_id = action.get("npc_id")
            action_data = action.get("action", {})
            
            # NPCs directly interacting have higher proximity
            if action_data.get("type") in ["talk", "command", "emotional_outburst"]:
                for other_id in npc_ids:
                    if other_id != npc_id:
                        proximity_factors[(min(npc_id, other_id), max(npc_id, other_id))] = 1.0
        
        # Individual actions indicate closer proximity for those specific pairs
        for npc_id, actions in individual_actions.items():
            for action in actions:
                target = action.get("target")
                if target and target.isdigit():
                    target_id = int(target)
                    proximity_factors[(min(npc_id, target_id), max(npc_id, target_id))] = 1.0
        
        return proximity_factors
    
    async def _get_mental_resilience_factors(self, npc_ids: List[int]) -> Dict[int, int]:
        """
        Get mental resilience for all NPCs to determine contagion resistance.
        
        Args:
            npc_ids: List of NPC IDs
            
        Returns:
            Dictionary mapping NPC IDs to mental resilience values
        """
        resistance_factors = {}
        
        try:
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute("""
                    SELECT npc_id, mental_resilience
                    FROM NPCStats
                    WHERE npc_id = ANY(%s)
                    AND user_id = %s
                    AND conversation_id = %s
                """, (npc_ids, self.user_id, self.conversation_id))
                
                for row in cursor.fetchall():
                    npc_id, mental_resilience = row
                    
                    # Default to 50 if not specified
                    resistance_factors[npc_id] = mental_resilience if mental_resilience is not None else 50
        except Exception as e:
            logger.error(f"Error getting mental resilience factors: {e}")
        
        return resistance_factors

    async def handle_player_action(
        self,
        player_action: Dict[str, Any],
        context: Dict[str, Any],
        npc_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Handle a player action directed at multiple NPCs.
        Enhanced with group memory formation.

        Args:
            player_action: The player's action
            context: Additional context
            npc_ids: Optional list of specific NPC IDs to target
            
        Returns:
            Dictionary with NPC responses
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

        # Process player action for each NPC concurrently
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
        filtered_responses = [r for r in responses if r is not None]
        
        # Check for mask slippage due to player action
        await self._check_mask_slippage_from_player_action(
            affected_npcs, 
            player_action, 
            mask_states, 
            emotional_states
        )
        
        # Create memories of this group interaction
        if len(affected_npcs) > 1:
            await self._create_player_group_interaction_memories(
                affected_npcs,
                player_action,
                context
            )
        
        return {"npc_responses": filtered_responses}
    
    async def _check_mask_slippage_from_player_action(
        self,
        npc_ids: List[int],
        player_action: Dict[str, Any],
        mask_states: Dict[int, Dict[str, Any]],
        emotional_states: Dict[int, Dict[str, Any]]
    ) -> None:
        """
        Check for mask slippage triggered by player action.
        
        Args:
            npc_ids: List of NPC IDs
            player_action: The player's action
            mask_states: Dictionary of NPC mask states
            emotional_states: Dictionary of NPC emotional states
        """
        memory_system = await self._get_memory_system()
        
        for npc_id in npc_ids:
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
                
                # Increase chance if NPC is in a strong emotional state
                emotional_state = emotional_states.get(npc_id)
                if emotional_state and "current_emotion" in emotional_state:
                    emotion = emotional_state["current_emotion"]
                    intensity = emotion.get("primary", {}).get("intensity", 0.0)
                    
                    if intensity > 0.7:
                        slippage_chance += 0.1
                
                # Roll for slippage
                if random.random() < slippage_chance:
                    # Generate mask slippage
                    await memory_system.reveal_npc_trait(
                        npc_id=npc_id,
                        trigger=player_action.get("description", "player action")
                    )
                    
                    # Invalidate cached mask info
                    if npc_id in self._mask_states:
                        del self._mask_states[npc_id]
                        del self._mask_states_timestamps[npc_id]
            except Exception as e:
                logger.error(f"Error processing mask slippage for NPC {npc_id}: {e}")

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
        npc_names = await self._get_npc_names(npc_ids)
        
        # Create a memory of this group interaction for each NPC
        for npc_id in npc_ids:
            # Filter out this NPC from the participant list
            other_npcs = [npc_names.get(other_id, f"NPC_{other_id}") for other_id in npc_ids if other_id != npc_id]
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

    async def determine_affected_npcs(self, player_action: Dict[str, Any], context: Dict[str, Any]) -> List[int]:
        """
        Determine which NPCs are affected by a player action.

        If "target_npc_id" is in the action, return that. Otherwise, find all NPCs in the
        current location from the context.
        
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

    async def _get_npc_name(self, npc_id: int) -> str:
        """
        Get an NPC's name.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            NPC's name
        """
        try:
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute("""
                    SELECT npc_name FROM NPCStats
                    WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
                """, (npc_id, self.user_id, self.conversation_id))
                
                row = cursor.fetchone()
                if row:
                    return row[0]
                
            return f"NPC_{npc_id}"
        except Exception as e:
            logger.error(f"Error getting NPC name: {e}")
            return f"NPC_{npc_id}"
    
    async def _get_npc_names(self, npc_ids: List[int]) -> Dict[int, str]:
        """
        Get names for multiple NPCs in one database query.
        
        Args:
            npc_ids: List of NPC IDs
            
        Returns:
            Dictionary mapping NPC IDs to names
        """
        npc_names = {}
        
        if not npc_ids:
            return npc_names
            
        try:
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute("""
                    SELECT npc_id, npc_name FROM NPCStats
                    WHERE npc_id = ANY(%s) 
                    AND user_id = %s 
                    AND conversation_id = %s
                """, (npc_ids, self.user_id, self.conversation_id))
                
                for row in cursor.fetchall():
                    npc_id, npc_name = row
                    npc_names[npc_id] = npc_name
            
            # Add default names for any missing NPCs
            for npc_id in npc_ids:
                if npc_id not in npc_names:
                    npc_names[npc_id] = f"NPC_{npc_id}"
                    
            return npc_names
        except Exception as e:
            logger.error(f"Error getting NPC names: {e}")
            # Return default names if query fails
            return {npc_id: f"NPC_{npc_id}" for npc_id in npc_ids}

    async def _get_relationship_level(self, npc_id1: int, npc_id2: int) -> Optional[int]:
        """
        Get relationship level between two NPCs.
        
        Args:
            npc_id1: First NPC ID
            npc_id2: Second NPC ID
            
        Returns:
            Relationship level or None if no relationship exists
        """
        try:
            with get_db_connection() as conn, conn.cursor() as cursor:
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
    
    def _fetch_basic_npc_data(self, npc_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Fetch minimal NPC data (name, dominance, cruelty) in a single query.
        
        Args:
            npc_ids: List of NPC IDs
            
        Returns:
            Dictionary mapping NPC IDs to basic data
        """
        data_map: Dict[int, Dict[str, Any]] = {}
        
        if not npc_ids:
            return data_map

        try:
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute("""
                    SELECT npc_id, npc_name, dominance, cruelty
                    FROM NPCStats
                    WHERE npc_id = ANY(%s)
                      AND user_id = %s
                      AND conversation_id = %s
                """, (npc_ids, self.user_id, self.conversation_id))
                
                for row in cursor.fetchall():
                    nid, name, dom, cru = row
                    data_map[nid] = {
                        "npc_name": name,
                        "dominance": dom,
                        "cruelty": cru
                    }
                    
            return data_map
        except Exception as e:
            logger.error(f"Error fetching basic NPC data: {e}")
            return {}

    def _fetch_npc_dominance(self, npc_ids: List[int]) -> Dict[int, int]:
        """
        Fetch only dominance values for the NPCs, used for sorting.
        
        Args:
            npc_ids: List of NPC IDs
            
        Returns:
            Dictionary mapping NPC IDs to dominance values
        """
        dom_map: Dict[int, int] = {}
        
        if not npc_ids:
            return dom_map

        try:
            with get_db_connection() as conn, conn.cursor() as cursor:
                cursor.execute("""
                    SELECT npc_id, dominance
                    FROM NPCStats
                    WHERE npc_id = ANY(%s)
                      AND user_id = %s
                      AND conversation_id = %s
                """, (npc_ids, self.user_id, self.conversation_id))
                
                for row in cursor.fetchall():
                    nid, dom = row
                    dom_map[nid] = dom
                    
            return dom_map
        except Exception as e:
            logger.error(f"Error fetching NPC dominance: {e}")
            return {}
