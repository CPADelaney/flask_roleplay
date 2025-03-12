"""
Conflict System Integration Module

This module provides the connection between the conflict system and the rest of the game.
It handles the integration with other game systems such as stats, time cycle, NPCs, etc.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Union

from logic.conflict_system.conflict_manager import ConflictManager
from logic.stats_logic import apply_stat_change
from logic.time_cycle import get_current_time
from logic.resource_management import ResourceManager
from logic.calendar import load_calendar_names
from utils.caching import CONFLICT_CACHE

logger = logging.getLogger(__name__)

class ConflictSystemIntegration:
    """
    Integration layer for the Conflict System.
    Provides methods to interact with the conflict system from other game components.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize with user and conversation context."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.conflict_manager = ConflictManager(user_id, conversation_id)
    
    async def get_active_conflicts(self) -> List[Dict[str, Any]]:
        """Get all active conflicts."""
        return await self.conflict_manager.get_active_conflicts()
    
    async def get_conflict_details(self, conflict_id: int) -> Dict[str, Any]:
        """Get detailed information about a specific conflict."""
        return await self.conflict_manager.get_conflict(conflict_id)
    
    async def generate_new_conflict(self, conflict_type: Optional[str] = None) -> Dict[str, Any]:
        """Generate a new conflict of the specified type, or determine the appropriate type based on current game state."""
        return await self.conflict_manager.generate_conflict(conflict_type)
    
    async def set_involvement(
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
        return await self.conflict_manager.set_player_involvement(
            conflict_id, involvement_level, faction,
            money_committed, supplies_committed, influence_committed, action
        )
    
    async def recruit_npc(self, conflict_id: int, npc_id: int, faction: Optional[str] = None) -> Dict[str, Any]:
        """Recruit an NPC to help with a conflict."""
        return await self.conflict_manager.recruit_npc_for_conflict(conflict_id, npc_id, faction)
    
    async def update_progress(self, conflict_id: int, progress_increment: float) -> Dict[str, Any]:
        """Update the progress of a conflict."""
        return await self.conflict_manager.update_conflict_progress(conflict_id, progress_increment)
    
    async def resolve_conflict(self, conflict_id: int) -> Dict[str, Any]:
        """Resolve a conflict and apply consequences."""
        return await self.conflict_manager.resolve_conflict(conflict_id)
    
    async def run_daily_update(self) -> Dict[str, Any]:
        """Run the daily conflict update."""
        return await self.conflict_manager.daily_conflict_update()
    
    async def get_player_vitals(self) -> Dict[str, Any]:
        """Get the current player vitals."""
        return await self.conflict_manager.get_player_vitals()
    
    async def update_player_vitals(self, activity_type: str = "standard") -> Dict[str, Any]:
        """Update player vitals based on activity type."""
        return await self.conflict_manager.update_player_vitals(activity_type)
    
    async def add_conflict_to_narrative(self, narrative_text: str) -> Dict[str, Any]:
        """Analyze a narrative text to determine if it should trigger a conflict."""
        return await self.conflict_manager.add_conflict_to_narrative(narrative_text)
    
    async def get_current_state(self) -> Dict[str, Any]:
        """Get the comprehensive state of the conflict system."""
        return await self.conflict_manager.get_current_state()
    
    async def process_activity_for_conflict_impact(self, activity_type: str, description: str) -> Dict[str, Any]:
        """
        Process an activity to determine its impact on active conflicts.
        This is used when the player performs actions that might affect conflicts.
        
        Args:
            activity_type: Type of activity (standard, intense, etc.)
            description: Description of the activity
            
        Returns:
            Dict with activity impact on conflicts
        """
        # Update player vitals first
        vitals_result = await self.conflict_manager.update_player_vitals(activity_type)
        
        # Process impact on conflicts
        result = {
            "vitals_updated": vitals_result,
            "conflicts_affected": [],
            "progress_made": False
        }
        
        # Get active conflicts
        active_conflicts = await self.conflict_manager.get_active_conflicts()
        if not active_conflicts:
            return result
        
        # Determine which conflicts might be affected by this activity
        for conflict in active_conflicts:
            # Simple text matching to see if activity is relevant
            relevance_score = self._calculate_activity_relevance(
                description, 
                conflict.get("conflict_name", ""),
                conflict.get("description", ""),
                conflict.get("faction_a_name", ""),
                conflict.get("faction_b_name", "")
            )
            
            if relevance_score > 0.3:  # Threshold for relevance
                # Calculate progress increment based on relevance and activity type
                progress_increment = self._calculate_progress_increment(
                    relevance_score, activity_type, conflict.get("conflict_type", "standard")
                )
                
                if progress_increment > 0:
                    # Update conflict progress
                    updated_conflict = await self.conflict_manager.update_conflict_progress(
                        conflict["conflict_id"], progress_increment
                    )
                    
                    if updated_conflict:
                        result["conflicts_affected"].append({
                            "conflict_id": conflict["conflict_id"],
                            "conflict_name": conflict["conflict_name"],
                            "relevance_score": relevance_score,
                            "progress_increment": progress_increment,
                            "new_progress": updated_conflict["progress"],
                            "new_phase": updated_conflict["phase"],
                            "phase_changed": updated_conflict["phase"] != conflict["phase"]
                        })
                        result["progress_made"] = True
        
        return result

    async def set_involvement(
        self, 
        conflict_id: int, 
        involvement_level: str,
        faction: str = "neutral",
        money_committed: int = 0,
        supplies_committed: int = 0,
        influence_committed: int = 0,
        action: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Set the player's involvement in a conflict, handling resource commitment.
        
        This method now integrates with the ResourceManager to properly deduct
        resources when they are committed to a conflict.
        """
        if involvement_level not in self.conflict_manager.INVOLVEMENT_LEVELS:
            return {"error": f"Invalid involvement level. Must be one of: {', '.join(self.conflict_manager.INVOLVEMENT_LEVELS)}"}
        
        if faction not in ["a", "b", "neutral"]:
            return {"error": "Invalid faction. Must be 'a', 'b', or 'neutral'"}
        
        # Initialize resource manager to handle resource commitment
        resource_manager = ResourceManager(self.user_id, self.conversation_id)
        
        # Check if player has sufficient resources
        if money_committed > 0 or supplies_committed > 0 or influence_committed > 0:
            resource_check = await resource_manager.check_resources(
                money_committed, supplies_committed, influence_committed
            )
            
            if not resource_check["has_resources"]:
                return {
                    "error": "Insufficient resources to commit",
                    "missing": resource_check["missing"],
                    "current": resource_check["current"]
                }
            
            # Commit resources if available
            resource_result = await resource_manager.commit_resources_to_conflict(
                conflict_id, money_committed, supplies_committed, influence_committed
            )
            
            if not resource_result["success"]:
                return resource_result  # Return the error
        
        # Call the original manager method to update involvement
        result = await self.conflict_manager.set_player_involvement(
            conflict_id, involvement_level, faction,
            money_committed, supplies_committed, influence_committed, action
        )
        
        # Add resource information to the result
        if "resources" not in result:
            result["resources"] = await resource_manager.get_resources()
        
        return result
    
    def _calculate_activity_relevance(self, activity_desc: str, *conflict_elements: str) -> float:
        """
        Calculate how relevant an activity is to a conflict.
        Uses simple text matching for keywords.
        
        Returns:
            Float between 0 and 1 representing relevance score
        """
        activity_desc = activity_desc.lower()
        words = set(w.lower() for element in conflict_elements for w in element.split())
        
        # Remove common words
        common_words = {"the", "a", "an", "and", "of", "in", "to", "for", "with", "as", "at", "by"}
        words = words - common_words
        
        # Count matches
        matches = sum(1 for word in words if word in activity_desc)
        
        # Calculate score
        if not words:
            return 0
        
        return min(1.0, matches / max(5, len(words)))
    
    def _calculate_progress_increment(self, relevance_score: float, activity_type: str, conflict_type: str) -> float:
        """
        Calculate progress increment based on relevance score, activity type, and conflict type.
        
        Returns:
            Float representing progress increment (0-100)
        """
        # Base increment based on activity type
        base_increments = {
            "standard": 2,
            "intense": 5,
            "restful": 0.5,
            "eating": 0.1,
            "social": 1,
            "training": 3,
            "work": 1
        }
        
        base = base_increments.get(activity_type, 1)
        
        # Modifier based on conflict type (major conflicts progress slower)
        type_modifiers = {
            "major": 0.4,
            "minor": 0.8,
            "standard": 1.0,
            "catastrophic": 0.2
        }
        
        type_mod = type_modifiers.get(conflict_type, 1.0)
        
        # Final calculation
        return base * relevance_score * type_mod
    
    async def analyze_conflict_narrative(self, narrative_text: str) -> Dict[str, Any]:
        """
        Analyze a narrative to extract information about active conflicts.
        This can be used to generate contextual descriptions about conflicts.
        
        Args:
            narrative_text: Text to analyze
            
        Returns:
            Dict with analysis results
        """
        # Get active conflicts
        active_conflicts = await self.get_active_conflicts()
        if not active_conflicts:
            return {
                "conflicts_mentioned": [],
                "has_conflict_content": False
            }
        
        narrative_lower = narrative_text.lower()
        
        # Check for mentions of active conflicts
        mentioned_conflicts = []
        for conflict in active_conflicts:
            if conflict.get("conflict_name", "").lower() in narrative_lower:
                mentioned_conflicts.append({
                    "conflict_id": conflict["conflict_id"],
                    "conflict_name": conflict["conflict_name"],
                    "exact_match": True
                })
            else:
                # Check for thematic matches
                relevance = self._calculate_activity_relevance(
                    narrative_lower, 
                    conflict.get("conflict_name", ""),
                    conflict.get("description", ""),
                    conflict.get("faction_a_name", ""),
                    conflict.get("faction_b_name", "")
                )
                
                if relevance > 0.4:  # Higher threshold for narrative analysis
                    mentioned_conflicts.append({
                        "conflict_id": conflict["conflict_id"],
                        "conflict_name": conflict["conflict_name"],
                        "relevance": relevance,
                        "exact_match": False
                    })
        
        # Analyze for conflict-related terms
        conflict_terms = {
            "tension": 0.3,
            "conflict": 0.5,
            "fight": 0.4,
            "battle": 0.4,
            "struggle": 0.3,
            "opposition": 0.3,
            "confrontation": 0.4,
            "argument": 0.3,
            "disagreement": 0.3
        }
        
        term_matches = []
        for term, weight in conflict_terms.items():
            if term in narrative_lower:
                term_matches.append({
                    "term": term,
                    "weight": weight
                })
        
        return {
            "conflicts_mentioned": mentioned_conflicts,
            "conflict_terms": term_matches,
            "has_conflict_content": bool(mentioned_conflicts) or bool(term_matches)
        }
    
    async def suggest_conflict_actions(self, conflict_id: int) -> List[Dict[str, Any]]:
        """
        Generate suggested actions for a specific conflict.
        This can be used to provide the player with options.
        
        Args:
            conflict_id: The conflict ID
            
        Returns:
            List of suggested actions
        """
        # Get conflict details
        conflict = await self.get_conflict_details(conflict_id)
        if not conflict:
            return []
        
        # Base suggestions based on phase
        phase = conflict.get("phase", "brewing")
        
        phase_suggestions = {
            "brewing": [
                {"action": "Gather information", "description": "Learn more about the brewing conflict", "progress_impact": 1},
                {"action": "Speak with involved parties", "description": "Talk to people on both sides", "progress_impact": 2},
                {"action": "Prepare resources", "description": "Gather money and supplies", "progress_impact": 0}
            ],
            "active": [
                {"action": "Take a side", "description": "Openly support one faction", "progress_impact": 3},
                {"action": "Mediate", "description": "Try to reduce tensions between factions", "progress_impact": 2},
                {"action": "Commit resources", "description": "Provide money or supplies to your chosen side", "progress_impact": 3}
            ],
            "climax": [
                {"action": "Direct intervention", "description": "Personally step in to influence the outcome", "progress_impact": 5},
                {"action": "Rally support", "description": "Get others to join your cause", "progress_impact": 4},
                {"action": "Final push", "description": "Commit everything to ensure victory", "progress_impact": 5}
            ],
            "resolution": [
                {"action": "Secure favorable terms", "description": "Ensure the outcome benefits you", "progress_impact": 3},
                {"action": "Consolidate gains", "description": "Make sure your side's victory is complete", "progress_impact": 2},
                {"action": "Prepare for aftermath", "description": "Plan for what comes after the conflict", "progress_impact": 1}
            ]
        }
        
        # Get base suggestions for this phase
        suggestions = phase_suggestions.get(phase, [])
        
        # Add faction-specific suggestions
        faction_a = conflict.get("faction_a_name", "Faction A")
        faction_b = conflict.get("faction_b_name", "Faction B")
        
        if phase != "brewing":  # Factions are more defined in later phases
            suggestions.append({
                "action": f"Support {faction_a}",
                "description": f"Actively work to help {faction_a} succeed", 
                "progress_impact": 4,
                "faction": "a"
            })
            suggestions.append({
                "action": f"Support {faction_b}",
                "description": f"Actively work to help {faction_b} succeed", 
                "progress_impact": 4,
                "faction": "b"
            })
        
        # Add NPC-specific suggestions
        for npc in conflict.get("involved_npcs", []):
            npc_name = npc.get("npc_name", "Unknown NPC")
            suggestions.append({
                "action": f"Recruit {npc_name}",
                "description": f"Convince {npc_name} to support your side", 
                "progress_impact": 3,
                "npc_id": npc.get("npc_id"),
                "faction": npc.get("faction")
            })
        
        return suggestions
