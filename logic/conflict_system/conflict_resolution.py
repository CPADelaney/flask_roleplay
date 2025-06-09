# logic/conflict_system/conflict_resolution.py

"""
Conflict Resolution System

This module provides sophisticated conflict resolution capabilities including
stakeholder management, resolution strategies, and dynamic conflict evolution.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import asyncio
from datetime import datetime

from agents import function_tool, RunContextWrapper
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance
from db.connection import get_db_connection_context

from logic.conflict_system.conflict_tools import (
    get_active_conflicts, get_conflict_details, get_conflict_stakeholders,
    get_resolution_paths, get_player_involvement, get_internal_conflicts,
    update_conflict_progress, update_stakeholder_status, add_resolution_path,
    update_player_involvement, add_internal_conflict, resolve_internal_conflict
)

logger = logging.getLogger(__name__)

class ConflictResolutionSystem:
    """
    Advanced conflict resolution system with sophisticated strategies
    and stakeholder management.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the conflict resolution system."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.resolution_agent = None
        self.stakeholder_agent = None
        self.strategy_agent = None
        self.is_initialized = False
        self.active_resolutions = {}
        self.resolution_history = []
        self.strategy_cache = {}
        
    async def initialize(self):
        """Initialize the conflict resolution system."""
        if not self.is_initialized:
            self.resolution_agent = await self._create_resolution_agent()
            self.stakeholder_agent = await self._create_stakeholder_agent()
            self.strategy_agent = await self._create_strategy_agent()
            self.is_initialized = True
            logger.info(f"Conflict resolution system initialized for user {self.user_id}")
        return self
        
    async def _create_resolution_agent(self):
        """Create the resolution agent for handling conflict resolutions."""
        governance = await get_central_governance(self.user_id, self.conversation_id)
        return await governance.create_agent(
            agent_type=AgentType.CONFLICT_RESOLVER,
            agent_id="conflict_resolver",
            capabilities=["resolution_planning", "outcome_prediction", "stakeholder_management"]
        )
        
    async def _create_stakeholder_agent(self):
        """Create the stakeholder agent for managing conflict stakeholders."""
        governance = await get_central_governance(self.user_id, self.conversation_id)
        return await governance.create_agent(
            agent_type=AgentType.STAKEHOLDER_MANAGER,
            agent_id="stakeholder_manager",
            capabilities=["stakeholder_analysis", "motivation_tracking", "alliance_management"]
        )
        
    async def _create_strategy_agent(self):
        """Create the strategy agent for developing resolution strategies."""
        governance = await get_central_governance(self.user_id, self.conversation_id)
        return await governance.create_agent(
            agent_type=AgentType.STRATEGY_PLANNER,
            agent_id="strategy_planner",
            capabilities=["strategy_development", "risk_assessment", "resource_optimization"]
        )
        
    @with_governance
    async def resolve_conflict(
        self,
        conflict_id: int,
        resolution_strategy: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Resolve a conflict using sophisticated resolution strategies.
        
        Args:
            conflict_id: ID of the conflict to resolve
            resolution_strategy: Optional pre-defined resolution strategy
            
        Returns:
            Resolution result with outcome details
        """
        try:
            # Get conflict details
            conflict = await get_conflict_details(RunContextWrapper(self.user_id, self.conversation_id), conflict_id)
            if not conflict:
                return {"success": False, "error": "Conflict not found"}
                
            # Get stakeholders
            stakeholders = await get_conflict_stakeholders(RunContextWrapper(self.user_id, self.conversation_id), conflict_id)
            
            # Get player involvement
            player_involvement = await get_player_involvement(RunContextWrapper(self.user_id, self.conversation_id), conflict_id)
            
            # Analyze conflict state
            conflict_state = await self._analyze_conflict_state(conflict, stakeholders, player_involvement)
            
            # Generate or use resolution strategy
            if not resolution_strategy:
                resolution_strategy = await self._generate_resolution_strategy(conflict_state)
                
            # Validate strategy
            validation_result = await self._validate_resolution_strategy(resolution_strategy, conflict_state)
            if not validation_result["success"]:
                return validation_result
                
            # Execute resolution
            resolution_result = await self._execute_resolution(conflict_state, resolution_strategy)
            
            # Update conflict state
            await self._update_conflict_state(conflict_id, resolution_result)
            
            # Record resolution
            self.resolution_history.append({
                "conflict_id": conflict_id,
                "timestamp": datetime.utcnow().isoformat(),
                "strategy": resolution_strategy,
                "result": resolution_result
            })
            
            return resolution_result
            
        except Exception as e:
            logger.error(f"Error resolving conflict: {e}")
            return {"success": False, "error": str(e)}
            
    async def _analyze_conflict_state(
        self,
        conflict: Dict[str, Any],
        stakeholders: List[Dict[str, Any]],
        player_involvement: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the current state of a conflict."""
        try:
            # Analyze stakeholder dynamics
            stakeholder_analysis = await self._analyze_stakeholder_dynamics(stakeholders)
            
            # Analyze player position
            player_analysis = await self._analyze_player_position(player_involvement, stakeholders)
            
            # Analyze conflict progression
            progression_analysis = await self._analyze_conflict_progression(conflict)
            
            # Analyze internal conflicts
            internal_conflicts = await get_internal_conflicts(RunContextWrapper(self.user_id, self.conversation_id), conflict["conflict_id"])
            internal_analysis = await self._analyze_internal_conflicts(internal_conflicts)
            
            return {
                "conflict": conflict,
                "stakeholder_analysis": stakeholder_analysis,
                "player_analysis": player_analysis,
                "progression_analysis": progression_analysis,
                "internal_analysis": internal_analysis,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing conflict state: {e}")
            return {}
            
    async def _analyze_stakeholder_dynamics(self, stakeholders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze stakeholder dynamics and relationships."""
        try:
            # Group stakeholders by faction
            faction_groups = {}
            for stakeholder in stakeholders:
                faction = stakeholder.get("faction_name", "independent")
                if faction not in faction_groups:
                    faction_groups[faction] = []
                faction_groups[faction].append(stakeholder)
                
            # Analyze faction relationships
            faction_relationships = {}
            for faction1 in faction_groups:
                faction_relationships[faction1] = {}
                for faction2 in faction_groups:
                    if faction1 != faction2:
                        relationship = await self._calculate_faction_relationship(
                            faction_groups[faction1],
                            faction_groups[faction2]
                        )
                        faction_relationships[faction1][faction2] = relationship
                        
            # Calculate stakeholder influence
            stakeholder_influence = {}
            for stakeholder in stakeholders:
                influence = await self._calculate_stakeholder_influence(stakeholder, stakeholders)
                stakeholder_influence[stakeholder["npc_id"]] = influence
                
            return {
                "faction_groups": faction_groups,
                "faction_relationships": faction_relationships,
                "stakeholder_influence": stakeholder_influence
            }
            
        except Exception as e:
            logger.error(f"Error analyzing stakeholder dynamics: {e}")
            return {}
            
    async def _analyze_player_position(
        self,
        player_involvement: Dict[str, Any],
        stakeholders: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze the player's position in the conflict."""
        try:
            # Calculate player influence
            player_influence = await self._calculate_player_influence(player_involvement, stakeholders)
            
            # Analyze player resources
            resource_analysis = await self._analyze_player_resources(player_involvement)
            
            # Analyze player relationships
            relationship_analysis = await self._analyze_player_relationships(player_involvement, stakeholders)
            
            return {
                "influence": player_influence,
                "resources": resource_analysis,
                "relationships": relationship_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing player position: {e}")
            return {}
            
    async def _analyze_conflict_progression(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the progression of the conflict."""
        try:
            # Get resolution paths
            resolution_paths = await get_resolution_paths(RunContextWrapper(self.user_id, self.conversation_id), conflict["conflict_id"])
            
            # Analyze path progress
            path_progress = {}
            for path in resolution_paths:
                progress = await self._analyze_path_progress(path)
                path_progress[path["path_id"]] = progress
                
            # Calculate overall progress
            overall_progress = await self._calculate_overall_progress(conflict, resolution_paths)
            
            return {
                "resolution_paths": resolution_paths,
                "path_progress": path_progress,
                "overall_progress": overall_progress
            }
            
        except Exception as e:
            logger.error(f"Error analyzing conflict progression: {e}")
            return {}
            
    async def _analyze_internal_conflicts(self, internal_conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze internal faction conflicts."""
        try:
            # Group by faction
            faction_internal_conflicts = {}
            for conflict in internal_conflicts:
                faction = conflict.get("faction_id")
                if faction not in faction_internal_conflicts:
                    faction_internal_conflicts[faction] = []
                faction_internal_conflicts[faction].append(conflict)
                
            # Analyze each faction's internal conflicts
            faction_analysis = {}
            for faction, conflicts in faction_internal_conflicts.items():
                analysis = await self._analyze_faction_internal_conflicts(conflicts)
                faction_analysis[faction] = analysis
                
            return {
                "faction_internal_conflicts": faction_internal_conflicts,
                "faction_analysis": faction_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing internal conflicts: {e}")
            return {}
            
    async def _generate_resolution_strategy(self, conflict_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a resolution strategy based on conflict state."""
        try:
            # Get stakeholder analysis
            stakeholder_analysis = conflict_state.get("stakeholder_analysis", {})
            
            # Get player analysis
            player_analysis = conflict_state.get("player_analysis", {})
            
            # Get progression analysis
            progression_analysis = conflict_state.get("progression_analysis", {})
            
            # Get internal analysis
            internal_analysis = conflict_state.get("internal_analysis", {})
            
            # Generate strategy using strategy agent
            strategy = await self.strategy_agent.generate_strategy({
                "stakeholder_analysis": stakeholder_analysis,
                "player_analysis": player_analysis,
                "progression_analysis": progression_analysis,
                "internal_analysis": internal_analysis
            })
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating resolution strategy: {e}")
            return {}
            
    async def _validate_resolution_strategy(
        self,
        strategy: Dict[str, Any],
        conflict_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a resolution strategy."""
        try:
            # Check strategy completeness
            completeness_check = await self._check_strategy_completeness(strategy)
            if not completeness_check["success"]:
                return completeness_check
                
            # Check resource feasibility
            feasibility_check = await self._check_strategy_feasibility(strategy, conflict_state)
            if not feasibility_check["success"]:
                return feasibility_check
                
            # Check stakeholder alignment
            alignment_check = await self._check_stakeholder_alignment(strategy, conflict_state)
            if not alignment_check["success"]:
                return alignment_check
                
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error validating resolution strategy: {e}")
            return {"success": False, "error": str(e)}
            
    async def _execute_resolution(
        self,
        conflict_state: Dict[str, Any],
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a resolution strategy."""
        try:
            # Initialize resolution tracking
            resolution_tracking = {
                "start_time": datetime.utcnow().isoformat(),
                "phases": [],
                "stakeholder_reactions": {},
                "resource_usage": {},
                "outcomes": {}
            }
            
            # Execute each phase of the strategy
            for phase in strategy.get("phases", []):
                phase_result = await self._execute_resolution_phase(phase, conflict_state)
                resolution_tracking["phases"].append(phase_result)
                
                # Update stakeholder reactions
                if "stakeholder_reactions" in phase_result:
                    resolution_tracking["stakeholder_reactions"].update(phase_result["stakeholder_reactions"])
                    
                # Update resource usage
                if "resource_usage" in phase_result:
                    resolution_tracking["resource_usage"].update(phase_result["resource_usage"])
                    
            # Calculate final outcomes
            final_outcomes = await self._calculate_final_outcomes(resolution_tracking, conflict_state)
            resolution_tracking["outcomes"] = final_outcomes
            
            # Record resolution completion
            resolution_tracking["end_time"] = datetime.utcnow().isoformat()
            
            return resolution_tracking
            
        except Exception as e:
            logger.error(f"Error executing resolution: {e}")
            return {"success": False, "error": str(e)}
            
    async def _execute_resolution_phase(
        self,
        phase: Dict[str, Any],
        conflict_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single phase of the resolution strategy."""
        try:
            # Initialize phase tracking
            phase_tracking = {
                "phase_name": phase.get("name"),
                "start_time": datetime.utcnow().isoformat(),
                "actions": [],
                "stakeholder_reactions": {},
                "resource_usage": {}
            }
            
            # Execute each action in the phase
            for action in phase.get("actions", []):
                action_result = await self._execute_resolution_action(action, conflict_state)
                phase_tracking["actions"].append(action_result)
                
                # Update stakeholder reactions
                if "stakeholder_reactions" in action_result:
                    phase_tracking["stakeholder_reactions"].update(action_result["stakeholder_reactions"])
                    
                # Update resource usage
                if "resource_usage" in action_result:
                    phase_tracking["resource_usage"].update(action_result["resource_usage"])
                    
            # Record phase completion
            phase_tracking["end_time"] = datetime.utcnow().isoformat()
            
            return phase_tracking
            
        except Exception as e:
            logger.error(f"Error executing resolution phase: {e}")
            return {"success": False, "error": str(e)}
            
    async def _execute_resolution_action(
        self,
        action: Dict[str, Any],
        conflict_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single resolution action."""
        try:
            # Initialize action tracking
            action_tracking = {
                "action_name": action.get("name"),
                "start_time": datetime.utcnow().isoformat(),
                "stakeholder_reactions": {},
                "resource_usage": {}
            }
            
            # Get action type and parameters
            action_type = action.get("type")
            parameters = action.get("parameters", {})
            
            # Execute action based on type
            if action_type == "negotiate":
                result = await self._execute_negotiation(parameters, conflict_state)
            elif action_type == "manipulate":
                result = await self._execute_manipulation(parameters, conflict_state)
            elif action_type == "resolve_internal":
                result = await self._execute_internal_resolution(parameters, conflict_state)
            else:
                result = {"success": False, "error": f"Unknown action type: {action_type}"}
                
            # Update action tracking
            action_tracking.update(result)
            action_tracking["end_time"] = datetime.utcnow().isoformat()
            
            return action_tracking
            
        except Exception as e:
            logger.error(f"Error executing resolution action: {e}")
            return {"success": False, "error": str(e)}
            
    async def _update_conflict_state(
        self,
        conflict_id: int,
        resolution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update the conflict state after resolution."""
        try:
            ctx = RunContextWrapper({
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
            async with get_db_connection_context() as conn:
                # Update conflict progress
                await update_conflict_progress(
                    ctx,
                    conflict_id,
                    resolution_result.get("outcomes", {}).get("progress", 0)
                )
                
                # Update stakeholder statuses
                for stakeholder_id, status in resolution_result.get("outcomes", {}).get("stakeholder_statuses", {}).items():
                    await update_stakeholder_status(
                        ctx,
                        conflict_id,
                        stakeholder_id,
                        status
                    )
                
                # Update player involvement
                if "player_outcome" in resolution_result.get("outcomes", {}):
                    await update_player_involvement(
                        ctx,
                        conflict_id,
                        resolution_result["outcomes"]["player_outcome"]
                    )
                
                # Log canonical event for resolution
                await canon.log_canonical_event(
                    ctx, conn,
                    f"Conflict {conflict_id} reached new phase with {resolution_result.get('outcomes', {}).get('progress', 0)}% progress",
                    tags=["conflict", "resolution", "progress"],
                    significance=6
                )
                
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error updating conflict state: {e}")
            return {"success": False, "error": str(e)}
            
    async def _calculate_final_outcomes(
        self,
        resolution_tracking: Dict[str, Any],
        conflict_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate final outcomes of the resolution."""
        try:
            # Calculate stakeholder outcomes
            stakeholder_outcomes = {}
            for stakeholder_id, reactions in resolution_tracking.get("stakeholder_reactions", {}).items():
                outcome = await self._calculate_stakeholder_outcome(stakeholder_id, reactions, conflict_state)
                stakeholder_outcomes[stakeholder_id] = outcome
                
            # Calculate player outcome
            player_outcome = await self._calculate_player_outcome(resolution_tracking, conflict_state)
            
            # Calculate overall progress
            overall_progress = await self._calculate_overall_progress_from_tracking(resolution_tracking)
            
            return {
                "stakeholder_statuses": stakeholder_outcomes,
                "player_outcome": player_outcome,
                "progress": overall_progress
            }
            
        except Exception as e:
            logger.error(f"Error calculating final outcomes: {e}")
            return {}
            
    async def _calculate_stakeholder_outcome(
        self,
        stakeholder_id: str,
        reactions: List[Dict[str, Any]],
        conflict_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate outcome for a specific stakeholder."""
        try:
            # Get stakeholder details
            stakeholders = conflict_state.get("conflict", {}).get("stakeholders", [])
            stakeholder = next((s for s in stakeholders if s["npc_id"] == stakeholder_id), None)
            if not stakeholder:
                return {}
                
            # Analyze reactions
            reaction_analysis = await self._analyze_stakeholder_reactions(reactions)
            
            # Calculate influence changes
            influence_changes = await self._calculate_influence_changes(stakeholder, reactions)
            
            # Calculate relationship changes
            relationship_changes = await self._calculate_relationship_changes(stakeholder, reactions)
            
            return {
                "reaction_analysis": reaction_analysis,
                "influence_changes": influence_changes,
                "relationship_changes": relationship_changes
            }
            
        except Exception as e:
            logger.error(f"Error calculating stakeholder outcome: {e}")
            return {}
            
    async def _calculate_player_outcome(
        self,
        resolution_tracking: Dict[str, Any],
        conflict_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate outcome for the player."""
        try:
            # Get player involvement
            player_involvement = conflict_state.get("player_analysis", {}).get("player_involvement", {})
            
            # Calculate resource changes
            resource_changes = await self._calculate_resource_changes(resolution_tracking)
            
            # Calculate influence changes
            influence_changes = await self._calculate_player_influence_changes(resolution_tracking)
            
            # Calculate relationship changes
            relationship_changes = await self._calculate_player_relationship_changes(resolution_tracking)
            
            return {
                "resource_changes": resource_changes,
                "influence_changes": influence_changes,
                "relationship_changes": relationship_changes
            }
            
        except Exception as e:
            logger.error(f"Error calculating player outcome: {e}")
            return {}
            
    async def _calculate_overall_progress_from_tracking(
        self,
        resolution_tracking: Dict[str, Any]
    ) -> float:
        """Calculate overall progress from resolution tracking."""
        try:
            # Get all phase results
            phases = resolution_tracking.get("phases", [])
            
            # Calculate progress from each phase
            phase_progress = []
            for phase in phases:
                if "outcomes" in phase:
                    phase_progress.append(phase["outcomes"].get("progress", 0))
                    
            # Calculate average progress
            if phase_progress:
                return sum(phase_progress) / len(phase_progress)
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating overall progress: {e}")
            return 0.0
            
    async def _analyze_stakeholder_reactions(self, reactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze stakeholder reactions to resolution actions."""
        try:
            # Group reactions by type
            reaction_types = {}
            for reaction in reactions:
                reaction_type = reaction.get("type")
                if reaction_type not in reaction_types:
                    reaction_types[reaction_type] = []
                reaction_types[reaction_type].append(reaction)
                
            # Analyze each reaction type
            analysis = {}
            for reaction_type, type_reactions in reaction_types.items():
                analysis[reaction_type] = await self._analyze_reaction_type(reaction_type, type_reactions)
                
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing stakeholder reactions: {e}")
            return {}
            
    async def _analyze_reaction_type(
        self,
        reaction_type: str,
        reactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze reactions of a specific type."""
        try:
            # Calculate average intensity
            intensities = [r.get("intensity", 0) for r in reactions]
            avg_intensity = sum(intensities) / len(intensities) if intensities else 0
            
            # Calculate sentiment
            sentiments = [r.get("sentiment", 0) for r in reactions]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
            # Calculate consistency
            consistency = await self._calculate_reaction_consistency(reactions)
            
            return {
                "average_intensity": avg_intensity,
                "average_sentiment": avg_sentiment,
                "consistency": consistency,
                "reaction_count": len(reactions)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing reaction type: {e}")
            return {}
            
    async def _calculate_reaction_consistency(self, reactions: List[Dict[str, Any]]) -> float:
        """Calculate consistency of reactions."""
        try:
            if not reactions:
                return 0.0
                
            # Calculate standard deviation of intensities
            intensities = [r.get("intensity", 0) for r in reactions]
            mean = sum(intensities) / len(intensities)
            variance = sum((i - mean) ** 2 for i in intensities) / len(intensities)
            std_dev = variance ** 0.5
            
            # Convert to consistency score (0-1)
            max_possible_std_dev = 1.0  # Assuming intensity is 0-1
            consistency = 1 - (std_dev / max_possible_std_dev)
            
            return max(0, min(1, consistency))
            
        except Exception as e:
            logger.error(f"Error calculating reaction consistency: {e}")
            return 0.0
            
    async def _calculate_influence_changes(
        self,
        stakeholder: Dict[str, Any],
        reactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate influence changes for a stakeholder."""
        try:
            # Get base influence
            base_influence = stakeholder.get("influence", 0)
            
            # Calculate influence changes from reactions
            influence_changes = []
            for reaction in reactions:
                if "influence_change" in reaction:
                    influence_changes.append(reaction["influence_change"])
                    
            # Calculate total change
            total_change = sum(influence_changes)
            
            # Calculate new influence
            new_influence = max(0, min(1, base_influence + total_change))
            
            return {
                "base_influence": base_influence,
                "total_change": total_change,
                "new_influence": new_influence
            }
            
        except Exception as e:
            logger.error(f"Error calculating influence changes: {e}")
            return {}
            
    async def _calculate_relationship_changes(
        self,
        stakeholder: Dict[str, Any],
        reactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate relationship changes for a stakeholder."""
        try:
            # Get base relationships
            base_relationships = stakeholder.get("relationships", {})
            
            # Calculate relationship changes from reactions
            relationship_changes = {}
            for reaction in reactions:
                if "relationship_changes" in reaction:
                    for target_id, change in reaction["relationship_changes"].items():
                        if target_id not in relationship_changes:
                            relationship_changes[target_id] = 0
                        relationship_changes[target_id] += change
                        
            # Calculate new relationships
            new_relationships = {}
            for target_id, base_value in base_relationships.items():
                change = relationship_changes.get(target_id, 0)
                new_relationships[target_id] = max(-1, min(1, base_value + change))
                
            return {
                "base_relationships": base_relationships,
                "relationship_changes": relationship_changes,
                "new_relationships": new_relationships
            }
            
        except Exception as e:
            logger.error(f"Error calculating relationship changes: {e}")
            return {}
            
    async def _calculate_resource_changes(self, resolution_tracking: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate resource changes from resolution tracking."""
        try:
            # Get all resource usage
            resource_usage = {}
            for phase in resolution_tracking.get("phases", []):
                if "resource_usage" in phase:
                    for resource_type, amount in phase["resource_usage"].items():
                        if resource_type not in resource_usage:
                            resource_usage[resource_type] = 0
                        resource_usage[resource_type] += amount
                        
            return resource_usage
            
        except Exception as e:
            logger.error(f"Error calculating resource changes: {e}")
            return {}
            
    async def _calculate_player_influence_changes(self, resolution_tracking: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate player influence changes from resolution tracking."""
        try:
            # Get all stakeholder reactions
            stakeholder_reactions = {}
            for phase in resolution_tracking.get("phases", []):
                if "stakeholder_reactions" in phase:
                    for stakeholder_id, reactions in phase["stakeholder_reactions"].items():
                        if stakeholder_id not in stakeholder_reactions:
                            stakeholder_reactions[stakeholder_id] = []
                        stakeholder_reactions[stakeholder_id].extend(reactions)
                        
            # Calculate influence changes
            influence_changes = {}
            for stakeholder_id, reactions in stakeholder_reactions.items():
                influence_changes[stakeholder_id] = await self._calculate_influence_changes_from_reactions(reactions)
                
            return influence_changes
            
        except Exception as e:
            logger.error(f"Error calculating player influence changes: {e}")
            return {}
            
    async def _calculate_player_relationship_changes(self, resolution_tracking: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate player relationship changes from resolution tracking."""
        try:
            # Get all stakeholder reactions
            stakeholder_reactions = {}
            for phase in resolution_tracking.get("phases", []):
                if "stakeholder_reactions" in phase:
                    for stakeholder_id, reactions in phase["stakeholder_reactions"].items():
                        if stakeholder_id not in stakeholder_reactions:
                            stakeholder_reactions[stakeholder_id] = []
                        stakeholder_reactions[stakeholder_id].extend(reactions)
                        
            # Calculate relationship changes
            relationship_changes = {}
            for stakeholder_id, reactions in stakeholder_reactions.items():
                relationship_changes[stakeholder_id] = await self._calculate_relationship_changes_from_reactions(reactions)
                
            return relationship_changes
            
        except Exception as e:
            logger.error(f"Error calculating player relationship changes: {e}")
            return {}
            
    async def _calculate_influence_changes_from_reactions(self, reactions: List[Dict[str, Any]]) -> float:
        """Calculate influence changes from reactions."""
        try:
            # Sum up influence changes
            total_change = 0
            for reaction in reactions:
                if "influence_change" in reaction:
                    total_change += reaction["influence_change"]
                    
            return total_change
            
        except Exception as e:
            logger.error(f"Error calculating influence changes from reactions: {e}")
            return 0.0
            
    async def _calculate_relationship_changes_from_reactions(self, reactions: List[Dict[str, Any]]) -> float:
        """Calculate relationship changes from reactions."""
        try:
            # Sum up relationship changes
            total_change = 0
            for reaction in reactions:
                if "relationship_change" in reaction:
                    total_change += reaction["relationship_change"]
                    
            return total_change
            
        except Exception as e:
            logger.error(f"Error calculating relationship changes from reactions: {e}")
            return 0.0
            
    async def _check_strategy_completeness(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a resolution strategy is complete."""
        try:
            # Check required fields
            required_fields = ["name", "description", "phases"]
            missing_fields = [field for field in required_fields if field not in strategy]
            
            if missing_fields:
                return {
                    "success": False,
                    "error": f"Strategy missing required fields: {', '.join(missing_fields)}"
                }
                
            # Check phases
            if not strategy["phases"]:
                return {
                    "success": False,
                    "error": "Strategy has no phases"
                }
                
            # Check each phase
            for phase in strategy["phases"]:
                phase_check = await self._check_phase_completeness(phase)
                if not phase_check["success"]:
                    return phase_check
                    
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error checking strategy completeness: {e}")
            return {"success": False, "error": str(e)}
            
    async def _check_phase_completeness(self, phase: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a phase is complete."""
        try:
            # Check required fields
            required_fields = ["name", "description", "actions"]
            missing_fields = [field for field in required_fields if field not in phase]
            
            if missing_fields:
                return {
                    "success": False,
                    "error": f"Phase missing required fields: {', '.join(missing_fields)}"
                }
                
            # Check actions
            if not phase["actions"]:
                return {
                    "success": False,
                    "error": "Phase has no actions"
                }
                
            # Check each action
            for action in phase["actions"]:
                action_check = await self._check_action_completeness(action)
                if not action_check["success"]:
                    return action_check
                    
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error checking phase completeness: {e}")
            return {"success": False, "error": str(e)}
            
    async def _check_action_completeness(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Check if an action is complete."""
        try:
            # Check required fields
            required_fields = ["name", "type", "parameters"]
            missing_fields = [field for field in required_fields if field not in action]
            
            if missing_fields:
                return {
                    "success": False,
                    "error": f"Action missing required fields: {', '.join(missing_fields)}"
                }
                
            # Check parameters based on type
            if action["type"] == "negotiate":
                if "target_stakeholder" not in action["parameters"]:
                    return {
                        "success": False,
                        "error": "Negotiation action missing target stakeholder"
                    }
            elif action["type"] == "manipulate":
                if "target_stakeholder" not in action["parameters"] or "method" not in action["parameters"]:
                    return {
                        "success": False,
                        "error": "Manipulation action missing required parameters"
                    }
            elif action["type"] == "resolve_internal":
                if "conflict_id" not in action["parameters"]:
                    return {
                        "success": False,
                        "error": "Internal resolution action missing conflict ID"
                    }
                    
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error checking action completeness: {e}")
            return {"success": False, "error": str(e)}
            
    async def _check_strategy_feasibility(
        self,
        strategy: Dict[str, Any],
        conflict_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if a resolution strategy is feasible."""
        try:
            # Get player resources
            player_resources = conflict_state.get("player_analysis", {}).get("resources", {})
            
            # Calculate required resources
            required_resources = await self._calculate_required_resources(strategy)
            
            # Check resource availability
            for resource_type, required_amount in required_resources.items():
                available_amount = player_resources.get(resource_type, 0)
                if available_amount < required_amount:
                    return {
                        "success": False,
                        "error": f"Insufficient {resource_type}: {available_amount} available, {required_amount} required"
                    }
                    
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error checking strategy feasibility: {e}")
            return {"success": False, "error": str(e)}
            
    async def _check_stakeholder_alignment(
        self,
        strategy: Dict[str, Any],
        conflict_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check stakeholder alignment with the strategy."""
        try:
            # Get stakeholder analysis
            stakeholder_analysis = conflict_state.get("stakeholder_analysis", {})
            
            # Check each phase
            for phase in strategy.get("phases", []):
                alignment_check = await self._check_phase_stakeholder_alignment(phase, stakeholder_analysis)
                if not alignment_check["success"]:
                    return alignment_check
                    
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error checking stakeholder alignment: {e}")
            return {"success": False, "error": str(e)}
            
    async def _check_phase_stakeholder_alignment(
        self,
        phase: Dict[str, Any],
        stakeholder_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check stakeholder alignment with a phase."""
        try:
            # Get stakeholder groups
            faction_groups = stakeholder_analysis.get("faction_groups", {})
            
            # Check each action
            for action in phase.get("actions", []):
                if action["type"] == "negotiate":
                    # Check if target stakeholder exists
                    target_id = action["parameters"].get("target_stakeholder")
                    if not any(target_id in group for group in faction_groups.values()):
                        return {
                            "success": False,
                            "error": f"Target stakeholder {target_id} not found"
                        }
                elif action["type"] == "manipulate":
                    # Check if target stakeholder exists and is manipulatable
                    target_id = action["parameters"].get("target_stakeholder")
                    if not any(target_id in group for group in faction_groups.values()):
                        return {
                            "success": False,
                            "error": f"Target stakeholder {target_id} not found"
                        }
                    # Check manipulation method feasibility
                    method = action["parameters"].get("method")
                    if not await self._check_manipulation_feasibility(method, target_id, stakeholder_analysis):
                        return {
                            "success": False,
                            "error": f"Manipulation method {method} not feasible for target {target_id}"
                        }
                        
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error checking phase stakeholder alignment: {e}")
            return {"success": False, "error": str(e)}
            
    async def _check_manipulation_feasibility(
        self,
        method: str,
        target_id: str,
        stakeholder_analysis: Dict[str, Any]
    ) -> bool:
        """Check if a manipulation method is feasible for a target."""
        try:
            # Get target stakeholder
            stakeholders = stakeholder_analysis.get("stakeholder_influence", {})
            target = stakeholders.get(target_id)
            if not target:
                return False
                
            # Check method feasibility based on target's characteristics
            if method == "blackmail":
                return target.get("has_secrets", False)
            elif method == "bribery":
                return target.get("is_corruptible", False)
            elif method == "threat":
                return target.get("has_vulnerabilities", False)
            elif method == "flattery":
                return target.get("is_vain", False)
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error checking manipulation feasibility: {e}")
            return False
            
    async def _calculate_required_resources(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate resources required for a strategy."""
        try:
            required_resources = {}
            
            # Calculate for each phase
            for phase in strategy.get("phases", []):
                phase_resources = await self._calculate_phase_resources(phase)
                for resource_type, amount in phase_resources.items():
                    if resource_type not in required_resources:
                        required_resources[resource_type] = 0
                    required_resources[resource_type] += amount
                    
            return required_resources
            
        except Exception as e:
            logger.error(f"Error calculating required resources: {e}")
            return {}
            
    async def _calculate_phase_resources(self, phase: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate resources required for a phase."""
        try:
            required_resources = {}
            
            # Calculate for each action
            for action in phase.get("actions", []):
                action_resources = await self._calculate_action_resources(action)
                for resource_type, amount in action_resources.items():
                    if resource_type not in required_resources:
                        required_resources[resource_type] = 0
                    required_resources[resource_type] += amount
                    
            return required_resources
            
        except Exception as e:
            logger.error(f"Error calculating phase resources: {e}")
            return {}
            
    async def _calculate_action_resources(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate resources required for an action."""
        try:
            required_resources = {}
            
            # Calculate based on action type
            if action["type"] == "negotiate":
                required_resources["influence"] = 1
                required_resources["time"] = 2
            elif action["type"] == "manipulate":
                method = action["parameters"].get("method")
                if method == "blackmail":
                    required_resources["influence"] = 2
                    required_resources["time"] = 3
                elif method == "bribery":
                    required_resources["money"] = 100
                    required_resources["influence"] = 1
                    required_resources["time"] = 2
                elif method == "threat":
                    required_resources["influence"] = 2
                    required_resources["time"] = 2
                elif method == "flattery":
                    required_resources["influence"] = 1
                    required_resources["time"] = 1
            elif action["type"] == "resolve_internal":
                required_resources["influence"] = 2
                required_resources["time"] = 3
                
            return required_resources
            
        except Exception as e:
            logger.error(f"Error calculating action resources: {e}")
            return {}
            
    async def _execute_negotiation(
        self,
        parameters: Dict[str, Any],
        conflict_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a negotiation action."""
        try:
            # Get target stakeholder
            target_id = parameters.get("target_stakeholder")
            stakeholders = conflict_state.get("conflict", {}).get("stakeholders", [])
            target = next((s for s in stakeholders if s["npc_id"] == target_id), None)
            if not target:
                return {"success": False, "error": "Target stakeholder not found"}
                
            # Get negotiation parameters
            negotiation_params = parameters.get("negotiation_params", {})
            
            # Calculate negotiation success
            success = await self._calculate_negotiation_success(target, negotiation_params, conflict_state)
            
            # Generate reactions
            reactions = await self._generate_negotiation_reactions(target, success, conflict_state)
            
            return {
                "success": success,
                "reactions": reactions,
                "stakeholder_reactions": {target_id: reactions},
                "resource_usage": {
                    "influence": 1,
                    "time": 2
                }
            }
            
        except Exception as e:
            logger.error(f"Error executing negotiation: {e}")
            return {"success": False, "error": str(e)}
            
    async def _execute_manipulation(
        self,
        parameters: Dict[str, Any],
        conflict_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a manipulation action."""
        try:
            # Get target stakeholder
            target_id = parameters.get("target_stakeholder")
            stakeholders = conflict_state.get("conflict", {}).get("stakeholders", [])
            target = next((s for s in stakeholders if s["npc_id"] == target_id), None)
            if not target:
                return {"success": False, "error": "Target stakeholder not found"}
                
            # Get manipulation method
            method = parameters.get("method")
            
            # Calculate manipulation success
            success = await self._calculate_manipulation_success(target, method, conflict_state)
            
            # Generate reactions
            reactions = await self._generate_manipulation_reactions(target, method, success, conflict_state)
            
            return {
                "success": success,
                "reactions": reactions,
                "stakeholder_reactions": {target_id: reactions},
                "resource_usage": await self._calculate_manipulation_resources(method)
            }
            
        except Exception as e:
            logger.error(f"Error executing manipulation: {e}")
            return {"success": False, "error": str(e)}
            
    async def _execute_internal_resolution(
        self,
        parameters: Dict[str, Any],
        conflict_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an internal conflict resolution action."""
        try:
            # Get internal conflict
            conflict_id = parameters.get("conflict_id")
            internal_conflicts = await get_internal_conflicts(RunContextWrapper(self.user_id, self.conversation_id), conflict_id)
            internal_conflict = next((c for c in internal_conflicts if c["struggle_id"] == conflict_id), None)
            if not internal_conflict:
                return {"success": False, "error": "Internal conflict not found"}
                
            # Calculate resolution success
            success = await self._calculate_internal_resolution_success(internal_conflict, conflict_state)
            
            # Generate reactions
            reactions = await self._generate_internal_resolution_reactions(internal_conflict, success, conflict_state)
            
            return {
                "success": success,
                "reactions": reactions,
                "stakeholder_reactions": reactions,
                "resource_usage": {
                    "influence": 2,
                    "time": 3
                }
            }
            
        except Exception as e:
            logger.error(f"Error executing internal resolution: {e}")
            return {"success": False, "error": str(e)}
            
    async def _calculate_negotiation_success(
        self,
        target: Dict[str, Any],
        negotiation_params: Dict[str, Any],
        conflict_state: Dict[str, Any]
    ) -> bool:
        """Calculate negotiation success probability."""
        try:
            # Get base success rate
            base_success = 0.5
            
            # Apply modifiers based on target characteristics
            if target.get("is_reasonable", False):
                base_success += 0.2
            if target.get("is_stubborn", False):
                base_success -= 0.2
                
            # Apply modifiers based on negotiation parameters
            if negotiation_params.get("has_leverage", False):
                base_success += 0.3
            if negotiation_params.get("has_threat", False):
                base_success -= 0.2
                
            # Apply modifiers based on relationship
            relationship = target.get("relationship_with_player", 0)
            base_success += relationship * 0.1
                
            # Calculate final success
            success = max(0, min(1, base_success))
            
            # Apply random factor
            return random.random() < success
            
        except Exception as e:
            logger.error(f"Error calculating negotiation success: {e}")
            return False
            
    async def _calculate_manipulation_success(
        self,
        target: Dict[str, Any],
        method: str,
        conflict_state: Dict[str, Any]
    ) -> bool:
        """Calculate manipulation success probability."""
        try:
            # Get base success rate
            base_success = 0.4
            
            # Apply modifiers based on method
            if method == "blackmail":
                if target.get("has_secrets", False):
                    base_success += 0.3
                if target.get("is_honest", False):
                    base_success -= 0.2
            elif method == "bribery":
                if target.get("is_corruptible", False):
                    base_success += 0.3
                if target.get("is_honest", False):
                    base_success -= 0.2
            elif method == "threat":
                if target.get("has_vulnerabilities", False):
                    base_success += 0.3
                if target.get("is_brave", False):
                    base_success -= 0.2
            elif method == "flattery":
                if target.get("is_vain", False):
                    base_success += 0.3
                if target.get("is_humble", False):
                    base_success -= 0.2
                    
            # Apply modifiers based on relationship
            relationship = target.get("relationship_with_player", 0)
            base_success += relationship * 0.1
                
            # Calculate final success
            success = max(0, min(1, base_success))
            
            # Apply random factor
            return random.random() < success
            
        except Exception as e:
            logger.error(f"Error calculating manipulation success: {e}")
            return False
            
    async def _calculate_internal_resolution_success(
        self,
        internal_conflict: Dict[str, Any],
        conflict_state: Dict[str, Any]
    ) -> bool:
        """Calculate internal conflict resolution success probability."""
        try:
            # Get base success rate
            base_success = 0.5
            
            # Apply modifiers based on conflict characteristics
            if internal_conflict.get("is_complex", False):
                base_success -= 0.2
            if internal_conflict.get("has_clear_sides", False):
                base_success += 0.2
                
            # Apply modifiers based on player influence
            player_influence = conflict_state.get("player_analysis", {}).get("influence", 0)
            base_success += player_influence * 0.3
                
            # Calculate final success
            success = max(0, min(1, base_success))
            
            # Apply random factor
            return random.random() < success
            
        except Exception as e:
            logger.error(f"Error calculating internal resolution success: {e}")
            return False
            
    async def _generate_negotiation_reactions(
        self,
        target: Dict[str, Any],
        success: bool,
        conflict_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate reactions to a negotiation attempt."""
        try:
            reactions = []
            
            # Generate target reaction
            target_reaction = {
                "type": "negotiation_response",
                "stakeholder_id": target["npc_id"],
                "success": success,
                "intensity": 0.7 if success else 0.3,
                "sentiment": 0.5 if success else -0.5,
                "influence_change": 0.1 if success else -0.1,
                "relationship_change": 0.1 if success else -0.1
            }
            reactions.append(target_reaction)
            
            # Generate allied stakeholder reactions
            for stakeholder in conflict_state.get("conflict", {}).get("stakeholders", []):
                if stakeholder["npc_id"] != target["npc_id"] and stakeholder.get("allied_with", target["npc_id"]):
                    ally_reaction = {
                        "type": "ally_response",
                        "stakeholder_id": stakeholder["npc_id"],
                        "success": success,
                        "intensity": 0.5,
                        "sentiment": 0.3 if success else -0.3,
                        "influence_change": 0.05 if success else -0.05,
                        "relationship_change": 0.05 if success else -0.05
                    }
                    reactions.append(ally_reaction)
                    
            return reactions
            
        except Exception as e:
            logger.error(f"Error generating negotiation reactions: {e}")
            return []
            
    async def _generate_manipulation_reactions(
        self,
        target: Dict[str, Any],
        method: str,
        success: bool,
        conflict_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate reactions to a manipulation attempt."""
        try:
            reactions = []
            
            # Generate target reaction
            target_reaction = {
                "type": "manipulation_response",
                "stakeholder_id": target["npc_id"],
                "method": method,
                "success": success,
                "intensity": 0.8 if success else 0.4,
                "sentiment": -0.7 if success else 0.3,
                "influence_change": -0.2 if success else 0.1,
                "relationship_change": -0.3 if success else 0.1
            }
            reactions.append(target_reaction)
            
            # Generate witness reactions
            for stakeholder in conflict_state.get("conflict", {}).get("stakeholders", []):
                if stakeholder["npc_id"] != target["npc_id"]:
                    witness_reaction = {
                        "type": "witness_response",
                        "stakeholder_id": stakeholder["npc_id"],
                        "method": method,
                        "success": success,
                        "intensity": 0.6,
                        "sentiment": -0.5 if success else 0.2,
                        "influence_change": -0.1 if success else 0.05,
                        "relationship_change": -0.2 if success else 0.05
                    }
                    reactions.append(witness_reaction)
                    
            return reactions
            
        except Exception as e:
            logger.error(f"Error generating manipulation reactions: {e}")
            return []
            
    async def _generate_internal_resolution_reactions(
        self,
        internal_conflict: Dict[str, Any],
        success: bool,
        conflict_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate reactions to an internal conflict resolution attempt."""
        try:
            reactions = []
            
            # Generate primary stakeholder reaction
            primary_reaction = {
                "type": "internal_resolution_response",
                "stakeholder_id": internal_conflict["primary_npc_id"],
                "success": success,
                "intensity": 0.7,
                "sentiment": 0.6 if success else -0.6,
                "influence_change": 0.2 if success else -0.1,
                "relationship_change": 0.2 if success else -0.1
            }
            reactions.append(primary_reaction)
            
            # Generate target stakeholder reaction
            target_reaction = {
                "type": "internal_resolution_response",
                "stakeholder_id": internal_conflict["target_npc_id"],
                "success": success,
                "intensity": 0.7,
                "sentiment": -0.6 if success else 0.6,
                "influence_change": -0.2 if success else 0.1,
                "relationship_change": -0.2 if success else 0.1
            }
            reactions.append(target_reaction)
            
            # Generate faction reactions
            faction_id = internal_conflict.get("faction_id")
            if faction_id:
                faction_reaction = {
                    "type": "faction_response",
                    "faction_id": faction_id,
                    "success": success,
                    "intensity": 0.6,
                    "sentiment": 0.4 if success else -0.4,
                    "influence_change": 0.1 if success else -0.05,
                    "relationship_change": 0.1 if success else -0.05
                }
                reactions.append(faction_reaction)
                
            return reactions
            
        except Exception as e:
            logger.error(f"Error generating internal resolution reactions: {e}")
            return []
            
    async def _calculate_manipulation_resources(self, method: str) -> Dict[str, Any]:
        """Calculate resources required for manipulation."""
        try:
            resources = {
                "influence": 1,
                "time": 2
            }
            
            # Add method-specific resources
            if method == "blackmail":
                resources["influence"] += 1
                resources["time"] += 1
            elif method == "bribery":
                resources["money"] = 100
            elif method == "threat":
                resources["influence"] += 1
            elif method == "flattery":
                resources["time"] += 1
                
            return resources
            
        except Exception as e:
            logger.error(f"Error calculating manipulation resources: {e}")
            return {}
