# logic/conflict_system/conflict_integration.py

"""
Conflict System Integration Module

This module provides the connection between the character-driven conflict system
and the rest of the game, with special focus on player manipulation mechanics.
"""

import logging
import json
import asyncio
import random
from typing import Dict, List, Any, Optional, Union, Tuple

from logic.conflict_system.conflict_manager import ConflictManager
from logic.stats_logic import apply_stat_change
from logic.time_cycle import get_current_time
from logic.resource_management import ResourceManager
from logic.calendar import load_calendar_names
from logic.npc_relationship_manager import get_relationship_status, get_manipulation_leverage
from utils.caching import CONFLICT_CACHE

logger = logging.getLogger(__name__)

class ConflictSystemIntegration:
    """
    Integration layer for the Conflict System.
    Provides methods to interact with the character-driven conflict system from other game components.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize with user and conversation context."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.conflict_manager = ConflictManager(user_id, conversation_id)
        self.resource_manager = ResourceManager(user_id, conversation_id)
    
    async def get_active_conflicts(self) -> List[Dict[str, Any]]:
        """Get all active conflicts."""
        return await self.conflict_manager.get_active_conflicts()
    
    async def get_conflict_details(self, conflict_id: int) -> Dict[str, Any]:
        """Get detailed information about a specific conflict."""
        return await self.conflict_manager.get_conflict(conflict_id)
    
    async def get_conflict_stakeholders(self, conflict_id: int) -> List[Dict[str, Any]]:
        """Get all stakeholders in a specific conflict."""
        return await self.conflict_manager.get_conflict_stakeholders(conflict_id)
    
    async def get_resolution_paths(self, conflict_id: int) -> List[Dict[str, Any]]:
        """Get all resolution paths for a specific conflict."""
        return await self.conflict_manager.get_resolution_paths(conflict_id)
    
    async def generate_new_conflict(self, conflict_type: Optional[str] = None) -> Dict[str, Any]:
        """Generate a new conflict of the specified type, or determine the appropriate type based on current game state."""
        return await self.conflict_manager.generate_conflict(conflict_type)
    
    async def track_story_beat(
        self,
        conflict_id: int,
        path_id: str,
        beat_description: str,
        involved_npcs: List[int],
        progress_value: float
    ) -> Dict[str, Any]:
        """
        Track a story beat for a resolution path, advancing progress.
        
        Args:
            conflict_id: ID of the conflict
            path_id: ID of the resolution path
            beat_description: Description of what happened
            involved_npcs: List of NPC IDs involved
            progress_value: Progress value (0-100)
            
        Returns:
            Updated path information
        """
        return await self.conflict_manager.track_story_beat(
            conflict_id, path_id, beat_description, involved_npcs, progress_value
        )
    
    async def set_player_involvement(
        self, 
        conflict_id: int, 
        involvement_level: str,
        faction: str = "neutral",
        money_committed: int = 0,
        supplies_committed: int = 0,
        influence_committed: int = 0,
        action: Optional[str] = None
    ) -> Dict[str, Any]:
        """Set the player's involvement in a conflict."""
        # Check if player has sufficient resources
        if money_committed > 0 or supplies_committed > 0 or influence_committed > 0:
            resource_check = await self.resource_manager.check_resources(
                money_committed, supplies_committed, influence_committed
            )
            
            if not resource_check["has_resources"]:
                return {
                    "error": "Insufficient resources to commit",
                    "missing": resource_check["missing"],
                    "current": resource_check["current"]
                }
            
            # Commit resources if available
            resource_result = await self.resource_manager.commit_resources_to_conflict(
                conflict_id, money_committed, supplies_committed, influence_committed
            )
            
            if not resource_result["success"]:
                return resource_result  # Return the error
        
        # Check if player is under manipulation
        player_involvement = await self.conflict_manager.get_player_involvement(conflict_id)
        
        if player_involvement["is_manipulated"]:
            manipulator_id = player_involvement["manipulated_by"].get("npc_id")
            manipulator_name = await self._get_npc_name(manipulator_id)
            
            # Apply stat changes for obeying/resisting manipulation
            manipulated_faction = player_involvement["manipulated_by"].get("faction", "neutral")
            manipulated_level = player_involvement["manipulated_by"].get("involvement_level", "observing")
            
            if faction == manipulated_faction and involvement_level == manipulated_level:
                # Player is obeying manipulation - increase obedience, dependency
                await apply_stat_change(self.user_id, self.conversation_id, "obedience", 3)
                await apply_stat_change(self.user_id, self.conversation_id, "dependency", 2)
            else:
                # Player is resisting manipulation - increase willpower, decrease obedience
                await apply_stat_change(self.user_id, self.conversation_id, "willpower", 3)
                await apply_stat_change(self.user_id, self.conversation_id, "obedience", -1)
        
        # Set player involvement
        result = await self.conflict_manager.set_player_involvement(
            conflict_id, involvement_level, faction,
            money_committed, supplies_committed, influence_committed, action
        )
        
        # Get updated resources
        resources = await self.resource_manager.get_resources()
        result["resources"] = resources
        
        return result
    
    async def get_player_manipulation_attempts(self, conflict_id: int) -> List[Dict[str, Any]]:
        """Get manipulation attempts targeted at the player for a specific conflict."""
        return await self.conflict_manager.get_player_manipulation_attempts(conflict_id)
    
    async def resolve_manipulation_attempt(
        self,
        attempt_id: int,
        success: bool,
        player_response: str
    ) -> Dict[str, Any]:
        """
        Resolve a manipulation attempt by the player.
        
        Args:
            attempt_id: ID of the manipulation attempt
            success: Whether the manipulation was successful
            player_response: The player's response to the manipulation
            
        Returns:
            Updated manipulation attempt and stat changes
        """
        result = await self.conflict_manager.resolve_manipulation_attempt(
            attempt_id, success, player_response
        )
        
        # Apply stat changes based on result
        stat_changes = {}
        
        if success:
            # If player succumbed to manipulation
            obedience_change = random.randint(2, 5)
            dependency_change = random.randint(1, 3)
            
            await apply_stat_change(self.user_id, self.conversation_id, "obedience", obedience_change)
            await apply_stat_change(self.user_id, self.conversation_id, "dependency", dependency_change)
            
            stat_changes["obedience"] = obedience_change
            stat_changes["dependency"] = dependency_change
        else:
            # If player resisted manipulation
            willpower_change = random.randint(2, 4)
            confidence_change = random.randint(1, 3)
            
            await apply_stat_change(self.user_id, self.conversation_id, "willpower", willpower_change)
            await apply_stat_change(self.user_id, self.conversation_id, "confidence", confidence_change)
            
            stat_changes["willpower"] = willpower_change
            stat_changes["confidence"] = confidence_change
        
        result["stat_changes"] = stat_changes
        
        return result
    
    async def create_manipulation_attempt(
        self,
        conflict_id: int,
        npc_id: int,
        manipulation_type: str,
        content: str,
        goal: Dict[str, Any],
        leverage_used: Dict[str, Any],
        intimacy_level: int = 0
    ) -> Dict[str, Any]:
        """
        Create a manipulation attempt by an NPC targeted at the player.
        
        This method is exposed to allow other systems (like the story director)
        to create manipulation attempts dynamically based on narrative context.
        
        Args:
            conflict_id: ID of the conflict
            npc_id: ID of the NPC doing the manipulation
            manipulation_type: Type of manipulation (domination, blackmail, seduction, etc.)
            content: Content of the manipulation attempt
            goal: What the NPC wants the player to do
            leverage_used: What leverage the NPC is using
            intimacy_level: Level of intimacy in the manipulation (0-10)
            
        Returns:
            The created manipulation attempt
        """
        return await self.conflict_manager.create_player_manipulation_attempt(
            conflict_id, npc_id, manipulation_type, content, 
            goal, leverage_used, intimacy_level
        )
    
    async def initiate_faction_power_struggle(
        self,
        conflict_id: int,
        faction_id: int,
        challenger_npc_id: int,
        target_npc_id: int,
        prize: str,
        approach: str,
        is_public: bool = False
    ) -> Dict[str, Any]:
        """
        Initiate a power struggle within a faction.
        
        Args:
            conflict_id: The main conflict ID
            faction_id: The faction where struggle occurs
            challenger_npc_id: NPC initiating the challenge
            target_npc_id: NPC being challenged (usually leader)
            prize: What's at stake (position, policy, etc.)
            approach: How the challenge is made (direct, subtle, etc.)
            is_public: Whether other stakeholders are aware
            
        Returns:
            The created internal faction conflict
        """
        return await self.conflict_manager.initiate_faction_power_struggle(
            conflict_id, faction_id, challenger_npc_id, target_npc_id,
            prize, approach, is_public
        )
    
    async def attempt_faction_coup(
        self,
        struggle_id: int,
        approach: str,
        supporting_npcs: List[int],
        resources_committed: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Attempt a coup within a faction to forcefully resolve a power struggle.
        
        Args:
            struggle_id: ID of the internal faction struggle
            approach: The approach used (direct, subtle, force, blackmail)
            supporting_npcs: List of NPCs supporting the coup
            resources_committed: Resources committed to the coup
            
        Returns:
            Coup attempt results
        """
        # Check if player has sufficient resources
        resource_total = sum(resources_committed.values())
        if resource_total > 0:
            resource_check = await self.resource_manager.check_resources(
                resources_committed.get("money", 0),
                resources_committed.get("supplies", 0),
                resources_committed.get("influence", 0)
            )
            
            if not resource_check["has_resources"]:
                return {
                    "error": "Insufficient resources to commit to coup",
                    "missing": resource_check["missing"],
                    "current": resource_check["current"]
                }
            
            # Commit resources
            await self.resource_manager.commit_resources(
                resources_committed.get("money", 0),
                resources_committed.get("supplies", 0),
                resources_committed.get("influence", 0),
                "Committed to faction coup attempt"
            )
        
        # Attempt the coup
        result = await self.conflict_manager.attempt_faction_coup(
            struggle_id, approach, supporting_npcs, resources_committed
        )
        
        # Apply stat changes based on outcome
        if result.get("success"):
            # Successful coup increases corruption and confidence
            await apply_stat_change(self.user_id, self.conversation_id, "corruption", 3)
            await apply_stat_change(self.user_id, self.conversation_id, "confidence", 5)
            
            result["stat_changes"] = {
                "corruption": 3,
                "confidence": 5
            }
        else:
            # Failed coup increases mental_resilience
            await apply_stat_change(self.user_id, self.conversation_id, "mental_resilience", 4)
            
            result["stat_changes"] = {
                "mental_resilience": 4
            }
        
        # Get updated resources
        resources = await self.resource_manager.get_resources()
        result["resources"] = resources
        
        return result
    
    async def resolve_conflict(self, conflict_id: int) -> Dict[str, Any]:
        """Resolve a conflict and apply consequences."""
        return await self.conflict_manager.resolve_conflict(conflict_id)
    
    async def add_conflict_to_narrative(self, narrative_text: str) -> Dict[str, Any]:
        """Analyze a narrative text to determine if it should trigger a conflict."""
        return await self.conflict_manager.add_conflict_to_narrative(narrative_text)
    
    async def analyze_manipulation_potential(
        self, 
        npc_id: int,
        player_stats: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze an NPC's potential to manipulate the player based on their relationship
        and the player's current stats.
        
        Args:
            npc_id: ID of the NPC
            player_stats: Optional player stats (will be fetched if not provided)
            
        Returns:
            Dictionary with manipulation potential analysis
        """
        # Get NPC details
        npc = await self._get_npc_details(npc_id)
        
        # Get relationship status
        relationship = await get_relationship_status(
            self.user_id, self.conversation_id, npc_id
        )
        
        # Get potential leverage
        leverage = await get_manipulation_leverage(
            self.user_id, self.conversation_id, npc_id
        )
        
        # Get player stats if not provided
        if not player_stats:
            player_stats = await self._get_player_stats()
        
        # Calculate manipulation potential for different types
        domination_potential = min(100, npc.get("dominance", 0) - player_stats.get("willpower", 50) + 50)
        seduction_potential = min(100, relationship.get("closeness", 0) + player_stats.get("lust", 20))
        blackmail_potential = min(100, 50 + (len(leverage) * 15))
        
        # Determine most effective manipulation type
        manipulation_types = [
            {"type": "domination", "potential": domination_potential},
            {"type": "seduction", "potential": seduction_potential},
            {"type": "blackmail", "potential": blackmail_potential}
        ]
        
        most_effective = max(manipulation_types, key=lambda x: x["potential"])
        
        # Determine overall manipulation potential
        overall_potential = most_effective["potential"]
        
        return {
            "npc_id": npc_id,
            "npc_name": npc.get("npc_name", "Unknown"),
            "overall_potential": overall_potential,
            "manipulation_types": manipulation_types,
            "most_effective_type": most_effective["type"],
            "relationship": relationship,
            "available_leverage": leverage,
            "femdom_compatible": npc.get("sex", "female") == "female" and domination_potential > 60
        }
    
    async def suggest_manipulation_content(
        self,
        npc_id: int,
        conflict_id: int,
        manipulation_type: str,
        goal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Suggest manipulation content for an NPC based on their relationship with the player
        and the desired outcome.
        
        Args:
            npc_id: ID of the NPC
            conflict_id: ID of the conflict
            manipulation_type: Type of manipulation (domination, blackmail, seduction, etc.)
            goal: What the NPC wants the player to do
            
        Returns:
            Suggested manipulation content
        """
        # Get NPC details
        npc = await self._get_npc_details(npc_id)
        
        # Get relationship status
        relationship = await get_relationship_status(
            self.user_id, self.conversation_id, npc_id
        )
        
        # Get conflict details
        conflict = await self.conflict_manager.get_conflict(conflict_id)
        
        # Generate content based on NPC traits and relationship
        if manipulation_type == "domination":
            content = self._generate_domination_content(npc, relationship, goal, conflict)
        elif manipulation_type == "seduction":
            content = self._generate_seduction_content(npc, relationship, goal, conflict)
        elif manipulation_type == "blackmail":
            leverage = await get_manipulation_leverage(
                self.user_id, self.conversation_id, npc_id
            )
            content = self._generate_blackmail_content(npc, relationship, goal, conflict, leverage)
        else:
            content = self._generate_generic_manipulation_content(npc, relationship, goal, conflict)
        
        # Generate appropriate leverage
        leverage_used = self._generate_leverage(npc, relationship, manipulation_type)
        
        # Determine intimacy level
        intimacy_level = self._calculate_intimacy_level(npc, relationship, manipulation_type)
        
        return {
            "npc_id": npc_id,
            "npc_name": npc.get("npc_name", "Unknown"),
            "manipulation_type": manipulation_type,
            "content": content,
            "leverage_used": leverage_used,
            "intimacy_level": intimacy_level,
            "goal": goal
        }
    
    # ----- Helper methods -----
    
    async def _get_npc_details(self, npc_id: int) -> Dict[str, Any]:
        """Get details for an NPC."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT npc_name, dominance, cruelty, closeness, trust, respect, intensity, sex
                FROM NPCStats
                WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
            """, (npc_id, self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            
            if not row:
                return {}
            
            npc_name, dominance, cruelty, closeness, trust, respect, intensity, sex = row
            
            return {
                "npc_id": npc_id,
                "npc_name": npc_name,
                "dominance": dominance,
                "cruelty": cruelty,
                "closeness": closeness,
                "trust": trust,
                "respect": respect,
                "intensity": intensity,
                "sex": sex
            }
        except Exception as e:
            logger.error(f"Error getting NPC details: {e}", exc_info=True)
            return {}
        finally:
            cursor.close()
            conn.close()
    
    async def _get_npc_name(self, npc_id: int) -> str:
        """Get an NPC's name."""
        npc = await self._get_npc_details(npc_id)
        return npc.get("npc_name", f"NPC {npc_id}")
    
    async def _get_player_stats(self) -> Dict[str, Any]:
        """Get player stats."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT corruption, confidence, willpower, obedience, dependency, lust,
                       mental_resilience, physical_endurance
                FROM PlayerStats
                WHERE user_id = %s AND conversation_id = %s
            """, (self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            
            if not row:
                return {}
            
            corruption, confidence, willpower, obedience, dependency, lust, mental_resilience, physical_endurance = row
            
            return {
                "corruption": corruption,
                "confidence": confidence,
                "willpower": willpower,
                "obedience": obedience,
                "dependency": dependency,
                "lust": lust,
                "mental_resilience": mental_resilience,
                "physical_endurance": physical_endurance
            }
        except Exception as e:
            logger.error(f"Error getting player stats: {e}", exc_info=True)
            return {}
        finally:
            cursor.close()
            conn.close()
    
    def _generate_domination_content(
        self,
        npc: Dict[str, Any],
        relationship: Dict[str, Any],
        goal: Dict[str, Any],
        conflict: Dict[str, Any]
    ) -> str:
        """Generate domination-based manipulation content."""
        npc_name = npc.get("npc_name", "")
        dominance = npc.get("dominance", 50)
        
        faction = goal.get("faction", "neutral")
        involvement = goal.get("involvement_level", "observing")
        
        # Get faction name if available
        faction_a_name = "first faction"
        faction_b_name = "second faction"
        
        if conflict:
            if "faction_a_name" in conflict:
                faction_a_name = conflict["faction_a_name"]
            if "faction_b_name" in conflict:
                faction_b_name = conflict["faction_b_name"]
        
        faction_name = faction_a_name if faction == "a" else faction_b_name if faction == "b" else "neutral party"
        
        # Templates based on dominance level
        if dominance > 80:  # Very dominant
            templates = [
                f"'{goal.get('specific_actions', 'You will do as I say')},' {npc_name} commands, her voice leaving no room for argument. 'Support {faction_name} by {involvement}. I will not tolerate disobedience.'",
                f"{npc_name} steps closer, towering over you despite her actual height. 'This isn't a request. You will help {faction_name} by {involvement}, or face consequences that you cannot imagine.'",
                f"'Look at me,' {npc_name} says firmly, tilting your chin up with one finger. 'You have no choice in this matter. You will support {faction_name} by {involvement}, and you will do it well. Is that understood?'"
            ]
        elif dominance > 60:  # Moderately dominant
            templates = [
                f"{npc_name} fixes you with a stern gaze. 'I expect you to support {faction_name} by {involvement}. It would be... unwise to disappoint me in this matter.'",
                f"'You will help {faction_name} by {involvement},' {npc_name} states with quiet authority. 'I know you understand the consequences of refusing me.'",
                f"{npc_name} smiles, but it doesn't reach her eyes. 'I need your help with {faction_name}, specifically by {involvement}. And I always get what I need from you, don't I?'"
            ]
        else:  # Mild dominance
            templates = [
                f"'I think it would be best if you supported {faction_name} by {involvement},' {npc_name} suggests with a hint of steel in her voice. 'Don't you agree?'",
                f"{npc_name} places a hand on your shoulder, subtly applying pressure. 'I'm counting on you to help {faction_name} by {involvement}. You wouldn't want to let me down.'",
                f"'You've always been so good at following directions,' {npc_name} says with a meaningful look. 'So you'll support {faction_name} by {involvement}, won't you?'"
            ]
        
        return random.choice(templates)
    
    def _generate_seduction_content(
        self,
        npc: Dict[str, Any],
        relationship: Dict[str, Any],
        goal: Dict[str, Any],
        conflict: Dict[str, Any]
    ) -> str:
        """Generate seduction-based manipulation content."""
        npc_name = npc.get("npc_name", "")
        closeness = relationship.get("closeness", 30)
        
        faction = goal.get("faction", "neutral")
        involvement = goal.get("involvement_level", "observing")
        
        # Get faction name if available
        faction_a_name = "first faction"
        faction_b_name = "second faction"
        
        if conflict:
            if "faction_a_name" in conflict:
                faction_a_name = conflict["faction_a_name"]
            if "faction_b_name" in conflict:
                faction_b_name = conflict["faction_b_name"]
        
        faction_name = faction_a_name if faction == "a" else faction_b_name if faction == "b" else "neutral party"
        
        # Templates based on closeness
        if closeness > 80:  # Very close
            templates = [
                f"{npc_name} trails her fingers down your cheek, her touch lingering. 'You know how much it would please me if you helped {faction_name} by {involvement},' she whispers. 'And I can be very... grateful when I'm pleased.'",
                f"'We have something special, don't we?' {npc_name} asks, pressing her body against yours. 'So of course you'll support {faction_name} by {involvement}. For me.' Her lips brush your ear as she says it.",
                f"{npc_name} takes your hand, guiding it to rest on her waist. 'Help {faction_name} by {involvement}, and I promise to make it worth every moment of your time,' she purrs, her meaning unmistakable."
            ]
        elif closeness > 60:  # Moderately close
            templates = [
                f"'I've been thinking about us,' {npc_name} says with a suggestive smile. 'About how things could... develop between us if you were to help {faction_name} by {involvement}.'",
                f"{npc_name} moves closer than strictly necessary, her perfume enveloping you. 'Support {faction_name} by {involvement}, and I'll show you just how appreciative I can be.'",
                f"'We could have a very special arrangement,' {npc_name} suggests, touching your arm lightly. 'You help {faction_name} by {involvement}, and I...' She leaves the rest unsaid, but her meaning is clear."
            ]
        else:  # Beginning closeness
            templates = [
                f"{npc_name} catches your eye, holding your gaze a moment longer than necessary. 'I find myself drawn to people who support {faction_name},' she says. 'Especially those who {involvement}.'",
                f"'I've noticed you,' {npc_name} admits with a shy smile that doesn't quite match her calculating eyes. 'And I could notice you even more if you were to help {faction_name} by {involvement}.'",
                f"{npc_name} leans in, her voice dropping to an intimate whisper. 'Between us, I think we could have something special if you were to support {faction_name} by {involvement}. Don't you think so?'"
            ]
        
        return random.choice(templates)
    
    def _generate_blackmail_content(
        self,
        npc: Dict[str, Any],
        relationship: Dict[str, Any],
        goal: Dict[str, Any],
        conflict: Dict[str, Any],
        leverage: List[Dict[str, Any]]
    ) -> str:
        """Generate blackmail-based manipulation content."""
        npc_name = npc.get("npc_name", "")
        cruelty = npc.get("cruelty", 30)
        
        faction = goal.get("faction", "neutral")
        involvement = goal.get("involvement_level", "observing")
        
        # Get faction name if available
        faction_a_name = "first faction"
        faction_b_name = "second faction"
        
        if conflict:
            if "faction_a_name" in conflict:
                faction_a_name = conflict["faction_a_name"]
            if "faction_b_name" in conflict:
                faction_b_name = conflict["faction_b_name"]
        
        faction_name = faction_a_name if faction == "a" else faction_b_name if faction == "b" else "neutral party"
        
        # Get leverage detail if available
        leverage_detail = "certain information"
        if leverage and len(leverage) > 0:
            leverage_detail = leverage[0].get("description", "certain information")
        
        # Templates based on cruelty
        if cruelty > 70:  # Very cruel
            templates = [
                f"{npc_name} smiles coldly. 'I know about {leverage_detail}. Help {faction_name} by {involvement}, or everyone else will know too. It's a simple choice, really.'",
                f"'Let me be clear,' {npc_name} says, her voice like ice. 'Either you support {faction_name} by {involvement}, or {leverage_detail} becomes public knowledge. What will it be?'",
                f"{npc_name} slides a folder across the table to you. Inside is proof of {leverage_detail}. 'Support {faction_name} by {involvement}, or this goes out to everyone who matters to you. Your choice.'"
            ]
        elif cruelty > 50:  # Moderately cruel
            templates = [
                f"'It would be a shame if people learned about {leverage_detail},' {npc_name} says with feigned concern. 'Fortunately, you can ensure my silence by helping {faction_name} with {involvement}.'",
                f"{npc_name} raises an eyebrow. 'We all have secrets, don't we? Yours involve {leverage_detail}. Mine... well, mine could involve keeping that quiet if you support {faction_name} by {involvement}.'",
                f"'I consider myself discreet,' {npc_name} says, studying her nails. 'Information about {leverage_detail} would never come from me... as long as you help {faction_name} by {involvement}, of course.'"
            ]
        else:  # Mildly cruel
            templates = [
                f"{npc_name} looks genuinely uncomfortable. 'I don't like doing this, but I need your help. I know about {leverage_detail}, and I'll use it if I have to. Please support {faction_name} by {involvement}.'",
                f"'This isn't how I wanted to ask,' {npc_name} says with a sigh, 'but I'm desperate. Help {faction_name} by {involvement}, or I'll have to tell people about {leverage_detail}.'",
                f"{npc_name} winces slightly. 'I hate to bring this up, but... {leverage_detail}. I need you to support {faction_name} by {involvement}, and we can both forget I ever mentioned it.'"
            ]
        
        return random.choice(templates)
    
    def _generate_generic_manipulation_content(
        self,
        npc: Dict[str, Any],
        relationship: Dict[str, Any],
        goal: Dict[str, Any],
        conflict: Dict[str, Any]
    ) -> str:
        """Generate generic manipulation content."""
        npc_name = npc.get("npc_name", "")
        
        faction = goal.get("faction", "neutral")
        involvement = goal.get("involvement_level", "observing")
        
        # Get faction name if available
        faction_a_name = "first faction"
        faction_b_name = "second faction"
        
        if conflict:
            if "faction_a_name" in conflict:
                faction_a_name = conflict["faction_a_name"]
            if "faction_b_name" in conflict:
                faction_b_name = conflict["faction_b_name"]
        
        faction_name = faction_a_name if faction == "a" else faction_b_name if faction == "b" else "neutral party"
        
        # Generic templates
        templates = [
            f"{npc_name} makes a compelling case for why you should support {faction_name} by {involvement}, appealing to your sense of reason.",
            f"'I need your help,' {npc_name} says earnestly. 'Please support {faction_name} by {involvement}. It would mean a lot to me.'",
            f"{npc_name} outlines the benefits you would receive if you were to help {faction_name} by {involvement}. The offer is tempting."
        ]
        
        return random.choice(templates)
    
    def _generate_leverage(
        self,
        npc: Dict[str, Any],
        relationship: Dict[str, Any],
        manipulation_type: str
    ) -> Dict[str, Any]:
        """Generate appropriate leverage based on manipulation type."""
        if manipulation_type == "domination":
            return {
                "type": "dominance",
                "description": "Authority and intimidation",
                "strength": npc.get("dominance", 50)
            }
        elif manipulation_type == "seduction":
            return {
                "type": "desire",
                "description": "Romantic or sexual interest",
                "strength": relationship.get("closeness", 30)
            }
        elif manipulation_type == "blackmail":
            return {
                "type": "information",
                "description": "Compromising information",
                "strength": npc.get("cruelty", 30)
            }
        else:
            return {
                "type": "persuasion",
                "description": "Logical argument",
                "strength": 50
            }
    
    def _calculate_intimacy_level(
        self,
        npc: Dict[str, Any],
        relationship: Dict[str, Any],
        manipulation_type: str
    ) -> int:
        """Calculate intimacy level (0-10) based on relationship and manipulation type."""
        base_intimacy = relationship.get("closeness", 0) // 10
        
        if manipulation_type == "seduction":
            # Seduction is more intimate
            return min(10, base_intimacy + 3)
        elif manipulation_type == "domination":
            # Domination can be intimate but depends on relationship
            dominance_factor = npc.get("dominance", 0) // 20
            return min(10, base_intimacy + dominance_factor)
        elif manipulation_type == "blackmail":
            # Blackmail is less intimate
            return max(0, base_intimacy - 2)
        else:
            # Generic manipulation is neutral
            return base_intimacy
async def register_with_governance(user_id: int, conversation_id: int):
    """
    Register conflict system with governance system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
    """
    governance = await get_central_governance(user_id, conversation_id)
    
    # Create conflict system instance
    conflict_system = ConflictSystemIntegration(user_id, conversation_id)
    
    # Register with governance
    await governance.register_agent(
        agent_type=AgentType.CONFLICT_ANALYST, 
        agent_instance=conflict_system, 
        agent_id="conflict_manager"
    )
    
    # Issue directive for conflict analysis
    await governance.issue_directive(
        agent_type=AgentType.CONFLICT_ANALYST,
        agent_id="conflict_manager",
        directive_type=DirectiveType.ACTION,
        directive_data={
            "instruction": "Manage conflicts and their progression in the game world",
            "scope": "game"
        },
        priority=DirectivePriority.MEDIUM,
        duration_minutes=24*60  # 24 hours
    )
    
    logger.info("Conflict System registered with Nyx governance")
