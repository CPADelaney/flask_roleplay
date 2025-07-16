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
import random
from lore.core import canon

from agents import function_tool, RunContextWrapper
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance
from db.connection import get_db_connection_context

from npcs.npc_relationship import NPCRelationshipManager
from logic.resource_management import ResourceManager

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
        try:
            governance = await get_central_governance(self.user_id, self.conversation_id)
            # Use CONFLICT_ANALYST as the agent type since CONFLICT_RESOLVER doesn't exist
            return await governance.create_agent(
                agent_type=AgentType.CONFLICT_ANALYST,
                agent_id="conflict_resolver",
                capabilities=["resolution_planning", "outcome_prediction", "stakeholder_management"]
            )
        except Exception as e:
            logger.warning(f"Could not create resolution agent via governance: {e}")
            # Return a mock agent object or None
            return None
        
    async def _create_stakeholder_agent(self):
        """Create the stakeholder agent for managing conflict stakeholders."""
        try:
            governance = await get_central_governance(self.user_id, self.conversation_id)
            # Use CONFLICT_ANALYST as the agent type
            return await governance.create_agent(
                agent_type=AgentType.CONFLICT_ANALYST,
                agent_id="stakeholder_manager",
                capabilities=["stakeholder_analysis", "motivation_tracking", "alliance_management"]
            )
        except Exception as e:
            logger.warning(f"Could not create stakeholder agent via governance: {e}")
            return None
            
    async def _create_strategy_agent(self):
        """Create the strategy agent for developing resolution strategies."""
        try:
            governance = await get_central_governance(self.user_id, self.conversation_id)
            # Use CONFLICT_ANALYST as the agent type
            return await governance.create_agent(
                agent_type=AgentType.CONFLICT_ANALYST,
                agent_id="strategy_planner",
                capabilities=["strategy_development", "risk_assessment", "resource_optimization"]
            )
        except Exception as e:
            logger.warning(f"Could not create strategy agent via governance: {e}")
            return None

    async def _calculate_faction_relationship(self, faction1_members: List[Dict[str, Any]], 
                                            faction2_members: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate the relationship between two factions based on their members."""
        try:
            # Calculate average relationships between faction members
            total_relationship = 0
            relationship_count = 0
            
            for member1 in faction1_members:
                for member2 in faction2_members:
                    # Get relationship between members
                    rel_manager = NPCRelationshipManager(
                        member1['npc_id'], 
                        self.user_id, 
                        self.conversation_id
                    )
                    relationship = await rel_manager.get_relationship_details('npc', member2['npc_id'])
                    
                    if relationship:
                        link_level = relationship.get('link_level', 0)
                        total_relationship += link_level
                        relationship_count += 1
            
            avg_relationship = total_relationship / relationship_count if relationship_count > 0 else 0
            
            # Determine relationship type based on average
            if avg_relationship > 50:
                rel_type = "allied"
            elif avg_relationship > 0:
                rel_type = "neutral"
            elif avg_relationship > -50:
                rel_type = "tense"
            else:
                rel_type = "hostile"
            
            return {
                "average_link_level": avg_relationship,
                "relationship_type": rel_type,
                "sample_size": relationship_count,
                "tension_level": max(0, -avg_relationship)  # Higher tension for negative relationships
            }
            
        except Exception as e:
            logger.error(f"Error calculating faction relationship: {e}")
            return {
                "average_link_level": 0,
                "relationship_type": "unknown",
                "sample_size": 0,
                "tension_level": 0
            }
    
    async def _calculate_stakeholder_influence(self, stakeholder: Dict[str, Any], 
                                             all_stakeholders: List[Dict[str, Any]]) -> float:
        """Calculate a stakeholder's influence based on various factors."""
        try:
            influence = 0.0
            
            # Base influence from involvement level
            influence += stakeholder.get('involvement_level', 0) * 10
            
            # Influence from dominance trait
            influence += stakeholder.get('dominance', 50) / 2
            
            # Influence from faction position
            faction_position = stakeholder.get('faction_position', '').lower()
            if 'leader' in faction_position:
                influence += 30
            elif 'commander' in faction_position or 'captain' in faction_position:
                influence += 20
            elif 'lieutenant' in faction_position:
                influence += 10
            
            # Influence from leadership ambition
            influence += stakeholder.get('leadership_ambition', 0) / 5
            
            # Influence from faction standing
            influence += stakeholder.get('faction_standing', 50) / 4
            
            # Influence from relationships with other stakeholders
            relationship_bonus = 0
            for other in all_stakeholders:
                if other['npc_id'] != stakeholder['npc_id']:
                    # Check if this stakeholder has alliances
                    alliances = stakeholder.get('alliances', {})
                    if isinstance(alliances, str):
                        try:
                            alliances = json.loads(alliances)
                        except:
                            alliances = {}
                    
                    if str(other['npc_id']) in alliances or other['npc_id'] in alliances:
                        relationship_bonus += 5
            
            influence += relationship_bonus
            
            # Cap influence at 100
            return min(100.0, influence)
            
        except Exception as e:
            logger.error(f"Error calculating stakeholder influence: {e}")
            return 0.0
    
    async def _calculate_player_influence(self, player_involvement: Dict[str, Any], 
                                        stakeholders: List[Dict[str, Any]]) -> float:
        """Calculate player's influence in the conflict."""
        try:
            influence = 0.0
            
            # Base influence from involvement level
            involvement_level = player_involvement.get('involvement_level', 'none')
            involvement_scores = {
                'none': 0,
                'observing': 10,
                'participating': 30,
                'leading': 50
            }
            influence += involvement_scores.get(involvement_level, 0)
            
            # Influence from resources committed
            money_committed = player_involvement.get('money_committed', 0)
            supplies_committed = player_involvement.get('supplies_committed', 0)
            influence_committed = player_involvement.get('influence_committed', 0)
            
            # Convert resources to influence (weighted)
            influence += money_committed / 100  # Money has less direct influence
            influence += supplies_committed / 10  # Supplies have moderate influence
            influence += influence_committed  # Direct influence conversion
            
            # Influence from faction alignment
            if player_involvement.get('faction') not in ['neutral', None]:
                influence += 10
            
            # Influence from being manipulated (negative)
            if player_involvement.get('manipulated_by'):
                influence -= 20
            
            # Influence from actions taken
            actions_taken = player_involvement.get('actions_taken', [])
            if isinstance(actions_taken, str):
                try:
                    actions_taken = json.loads(actions_taken)
                except:
                    actions_taken = []
            influence += len(actions_taken) * 5
            
            return max(0.0, min(100.0, influence))
            
        except Exception as e:
            logger.error(f"Error calculating player influence: {e}")
            return 0.0
    
    async def _analyze_player_resources(self, player_involvement: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze player's available resources."""
        try:
            # Get resource manager
            resource_manager = ResourceManager(self.user_id, self.conversation_id)
            
            # Get current resources
            current_resources = await resource_manager.get_resources()
            
            # Get committed resources
            committed = {
                'money': player_involvement.get('money_committed', 0),
                'supplies': player_involvement.get('supplies_committed', 0),
                'influence': player_involvement.get('influence_committed', 0)
            }
            
            # Calculate available resources
            available = {
                'money': current_resources.get('money', 0) - committed['money'],
                'supplies': current_resources.get('supplies', 0) - committed['supplies'],
                'influence': current_resources.get('influence', 0) - committed['influence']
            }
            
            # Analyze resource status
            resource_status = "abundant"
            if available['money'] < 50 or available['supplies'] < 10 or available['influence'] < 5:
                resource_status = "limited"
            elif available['money'] < 20 or available['supplies'] < 5 or available['influence'] < 2:
                resource_status = "scarce"
            elif any(v < 0 for v in available.values()):
                resource_status = "depleted"
            
            return {
                'current': current_resources,
                'committed': committed,
                'available': available,
                'status': resource_status,
                'can_commit_more': resource_status in ["abundant", "limited"]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing player resources: {e}")
            return {
                'current': {'money': 0, 'supplies': 0, 'influence': 0},
                'committed': {'money': 0, 'supplies': 0, 'influence': 0},
                'available': {'money': 0, 'supplies': 0, 'influence': 0},
                'status': 'unknown',
                'can_commit_more': False
            }
    
    async def _analyze_player_relationships(self, player_involvement: Dict[str, Any], 
                                          stakeholders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze player's relationships with stakeholders."""
        try:
            relationships = {}
            relationship_summary = {
                'allies': [],
                'enemies': [],
                'neutral': [],
                'manipulators': []
            }
            
            # Check who manipulated the player
            manipulated_by = player_involvement.get('manipulated_by')
            if manipulated_by:
                if isinstance(manipulated_by, str):
                    try:
                        manipulated_by = json.loads(manipulated_by)
                    except:
                        manipulated_by = {}
                manipulator_id = manipulated_by.get('npc_id')
                if manipulator_id:
                    relationship_summary['manipulators'].append(manipulator_id)
            
            # Get RelationshipIntegration instance
            rel_integration = RelationshipIntegration(self.user_id, self.conversation_id)
            
            # Analyze relationship with each stakeholder
            for stakeholder in stakeholders:
                npc_id = stakeholder['npc_id']
                
                # Get relationship details
                relationship = await rel_integration.get_relationship(
                    'player', self.user_id,
                    'npc', npc_id
                )
                
                if relationship:
                    link_level = relationship.get('link_level', 0)
                    link_type = relationship.get('link_type', 'neutral')
                    dynamics = relationship.get('dynamics', {})
                    
                    relationships[npc_id] = {
                        'npc_name': stakeholder.get('npc_name', f'NPC {npc_id}'),
                        'link_level': link_level,
                        'link_type': link_type,
                        'dynamics': dynamics,
                        'is_manipulator': npc_id in relationship_summary['manipulators']
                    }
                    
                    # Categorize based on link level
                    if link_level > 50:
                        relationship_summary['allies'].append(npc_id)
                    elif link_level < -50:
                        relationship_summary['enemies'].append(npc_id)
                    else:
                        relationship_summary['neutral'].append(npc_id)
                else:
                    # No existing relationship
                    relationships[npc_id] = {
                        'npc_name': stakeholder.get('npc_name', f'NPC {npc_id}'),
                        'link_level': 0,
                        'link_type': 'neutral',
                        'dynamics': {},
                        'is_manipulator': npc_id in relationship_summary['manipulators']
                    }
                    relationship_summary['neutral'].append(npc_id)
            
            return {
                'individual_relationships': relationships,
                'summary': relationship_summary,
                'total_allies': len(relationship_summary['allies']),
                'total_enemies': len(relationship_summary['enemies']),
                'being_manipulated': len(relationship_summary['manipulators']) > 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing player relationships: {e}")
            return {
                'individual_relationships': {},
                'summary': {
                    'allies': [],
                    'enemies': [],
                    'neutral': [],
                    'manipulators': []
                },
                'total_allies': 0,
                'total_enemies': 0,
                'being_manipulated': False
            }
    
    async def _analyze_path_progress(self, path: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze progress on a specific resolution path."""
        try:
            progress = path.get('progress', 0)
            is_completed = path.get('is_completed', False)
            difficulty = path.get('difficulty', 5)
            
            # Calculate completion rate
            if is_completed:
                completion_rate = 100
                status = 'completed'
            elif progress >= 75:
                completion_rate = progress
                status = 'nearly_complete'
            elif progress >= 50:
                completion_rate = progress
                status = 'progressing_well'
            elif progress >= 25:
                completion_rate = progress
                status = 'early_progress'
            elif progress > 0:
                completion_rate = progress
                status = 'just_started'
            else:
                completion_rate = 0
                status = 'not_started'
            
            # Estimate remaining effort based on difficulty
            if not is_completed:
                remaining_effort = (100 - progress) * (difficulty / 10)
            else:
                remaining_effort = 0
            
            # Analyze key challenges
            key_challenges = path.get('key_challenges', [])
            if isinstance(key_challenges, str):
                try:
                    key_challenges = json.loads(key_challenges)
                except:
                    key_challenges = []
            
            # Estimate challenges completed based on progress
            total_challenges = len(key_challenges)
            if total_challenges > 0:
                challenges_completed = int((progress / 100) * total_challenges)
                challenges_remaining = total_challenges - challenges_completed
            else:
                challenges_completed = 0
                challenges_remaining = 0
            
            return {
                'path_id': path.get('path_id'),
                'name': path.get('name'),
                'progress': progress,
                'completion_rate': completion_rate,
                'status': status,
                'is_completed': is_completed,
                'difficulty': difficulty,
                'remaining_effort': remaining_effort,
                'total_challenges': total_challenges,
                'challenges_completed': challenges_completed,
                'challenges_remaining': challenges_remaining,
                'approach_type': path.get('approach_type', 'standard')
            }
            
        except Exception as e:
            logger.error(f"Error analyzing path progress: {e}")
            return {
                'path_id': path.get('path_id', 'unknown'),
                'name': path.get('name', 'Unknown Path'),
                'progress': 0,
                'completion_rate': 0,
                'status': 'error',
                'is_completed': False,
                'difficulty': 5,
                'remaining_effort': 100,
                'total_challenges': 0,
                'challenges_completed': 0,
                'challenges_remaining': 0,
                'approach_type': 'unknown'
            }
    
    async def _calculate_overall_progress(self, conflict: Dict[str, Any], 
                                        resolution_paths: List[Dict[str, Any]]) -> float:
        """Calculate overall conflict progress based on all resolution paths."""
        try:
            if not resolution_paths:
                # If no paths, use conflict's direct progress
                return conflict.get('progress', 0)
            
            # Calculate weighted average of path progress
            total_progress = 0
            total_weight = 0
            
            for path in resolution_paths:
                progress = path.get('progress', 0)
                difficulty = path.get('difficulty', 5)
                
                # Weight by inverse difficulty (easier paths contribute more to overall progress)
                weight = 11 - difficulty  # So difficulty 1 = weight 10, difficulty 10 = weight 1
                
                total_progress += progress * weight
                total_weight += weight
            
            if total_weight > 0:
                weighted_average = total_progress / total_weight
            else:
                weighted_average = 0
            
            # Blend with conflict's stored progress (in case it's tracked separately)
            conflict_progress = conflict.get('progress', 0)
            
            # 70% from path progress, 30% from conflict progress
            overall_progress = (weighted_average * 0.7) + (conflict_progress * 0.3)
            
            # Apply phase modifiers
            phase = conflict.get('phase', 'brewing')
            if phase == 'brewing' and overall_progress > 30:
                overall_progress = min(overall_progress, 30)  # Cap at 30% in brewing
            elif phase == 'active' and overall_progress > 60:
                overall_progress = min(overall_progress, 60)  # Cap at 60% in active
            elif phase == 'climax' and overall_progress > 90:
                overall_progress = min(overall_progress, 90)  # Cap at 90% in climax
            
            return min(100.0, max(0.0, overall_progress))
            
        except Exception as e:
            logger.error(f"Error calculating overall progress: {e}")
            return 0.0
    
    async def _analyze_faction_internal_conflicts(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze internal conflicts within a faction."""
        try:
            # Group conflicts by phase
            phases = {
                'brewing': [],
                'active': [],
                'climax': [],
                'aftermath': [],
                'resolved': []
            }
            
            # Analyze each conflict
            conflict_analyses = []
            
            for conflict in conflicts:
                current_phase = conflict.get('current_phase', 'brewing')
                phases[current_phase].append(conflict)
                
                # Analyze conflict intensity
                progress = conflict.get('progress', 0)
                public_knowledge = conflict.get('public_knowledge', False)
                
                intensity = 'low'
                if progress > 75:
                    intensity = 'critical'
                elif progress > 50:
                    intensity = 'high'
                elif progress > 25:
                    intensity = 'moderate'
                
                # Determine likely outcome
                approach = conflict.get('approach', 'subtle')
                if approach in ['force', 'direct']:
                    likely_outcome = 'confrontation'
                elif approach in ['blackmail', 'sabotage']:
                    likely_outcome = 'betrayal'
                else:
                    likely_outcome = 'negotiation'
                
                conflict_analyses.append({
                    'struggle_id': conflict.get('struggle_id'),
                    'faction_id': conflict.get('faction_id'),
                    'primary_npc_id': conflict.get('primary_npc_id'),
                    'target_npc_id': conflict.get('target_npc_id'),
                    'prize': conflict.get('prize', 'unknown'),
                    'approach': approach,
                    'phase': current_phase,
                    'progress': progress,
                    'intensity': intensity,
                    'is_public': public_knowledge,
                    'likely_outcome': likely_outcome,
                    'description': conflict.get('description', '')
                })
            
            # Calculate faction stability based on internal conflicts
            stability_score = 100
            
            # Deduct for each conflict based on phase and intensity
            for analysis in conflict_analyses:
                if analysis['phase'] == 'resolved':
                    stability_score -= 5  # Past conflicts leave scars
                elif analysis['phase'] == 'aftermath':
                    stability_score -= 10
                elif analysis['phase'] == 'climax':
                    if analysis['intensity'] == 'critical':
                        stability_score -= 30
                    else:
                        stability_score -= 20
                elif analysis['phase'] == 'active':
                    if analysis['intensity'] == 'high':
                        stability_score -= 15
                    else:
                        stability_score -= 10
                else:  # brewing
                    stability_score -= 5
                
                # Public conflicts are more damaging
                if analysis['is_public']:
                    stability_score -= 5
            
            stability_score = max(0, stability_score)
            
            # Determine faction status
            if stability_score > 80:
                faction_status = 'stable'
            elif stability_score > 60:
                faction_status = 'tensions_present'
            elif stability_score > 40:
                faction_status = 'unstable'
            elif stability_score > 20:
                faction_status = 'crisis'
            else:
                faction_status = 'collapse_imminent'
            
            return {
                'total_conflicts': len(conflicts),
                'conflicts_by_phase': {k: len(v) for k, v in phases.items()},
                'active_struggles': len(phases['active']) + len(phases['climax']),
                'resolved_struggles': len(phases['resolved']),
                'conflict_analyses': conflict_analyses,
                'faction_stability_score': stability_score,
                'faction_status': faction_status,
                'most_intense_conflict': max(conflict_analyses, key=lambda x: x['progress']) if conflict_analyses else None
            }
            
        except Exception as e:
            logger.error(f"Error analyzing faction internal conflicts: {e}")
            return {
                'total_conflicts': 0,
                'conflicts_by_phase': {},
                'active_struggles': 0,
                'resolved_struggles': 0,
                'conflict_analyses': [],
                'faction_stability_score': 100,
                'faction_status': 'unknown',
                'most_intense_conflict': None
            }
        
    @with_governance(
        agent_type=AgentType.CONFLICT_ANALYST,
        action_type="resolve_conflict",
        action_description="Resolving conflict {conflict_id}",
        id_from_context=lambda ctx: "conflict_resolver"
    )
    
    async def resolve_conflict(
        self,
        ctx,  # Add this parameter
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
            conflict = await get_conflict_details(ctx, conflict_id)
            if not conflict:
                return {"success": False, "error": "Conflict not found"}
                
            # Get stakeholders
            stakeholders = await get_conflict_stakeholders(ctx, conflict_id)
            
            # Get player involvement
            player_involvement = await get_player_involvement(ctx, conflict_id)
            
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
            ctx = RunContextWrapper({
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            # Analyze stakeholder dynamics
            stakeholder_analysis = await self._analyze_stakeholder_dynamics(stakeholders)
            
            # Analyze player position
            player_analysis = await self._analyze_player_position(player_involvement, stakeholders)
            
            # Analyze conflict progression
            progression_analysis = await self._analyze_conflict_progression(conflict)
            
            # Analyze internal conflicts
            internal_conflicts = await get_internal_conflicts(ctx, conflict["conflict_id"])
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
            ctx = RunContextWrapper({
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            # Get resolution paths
            resolution_paths = await get_resolution_paths(ctx, conflict["conflict_id"])
            
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
            ctx = RunContextWrapper({
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
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
            ctx = RunContextWrapper({
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
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
            
            # Log action canonically
            async with get_db_connection_context() as conn:
                await canon.log_canonical_event(
                    ctx, conn,
                    f"Conflict resolution action '{action_type}' executed with {'success' if result.get('success') else 'failure'}",
                    tags=["conflict", "resolution", action_type],
                    significance=5 if result.get('success') else 4
                )
            
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
            ctx = RunContextWrapper({
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            # Get internal conflict
            conflict_id = parameters.get("conflict_id")
            internal_conflicts = await get_internal_conflicts(ctx, conflict_id)
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
