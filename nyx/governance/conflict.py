# nyx/governance/conflict.py
"""
Conflict creation and management.
"""
import logging
import json
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from db.connection import get_db_connection_context
from agents import RunContextWrapper

logger = logging.getLogger(__name__)


class ConflictGovernanceMixin:
    """Handles conflict-related governance functions."""
    
    async def create_conflict(self, conflict_data: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Create conflict through synthesizer"""
        if not self._initialized:
            await self.initialize()
        
        from logic.conflict_system.conflict_synthesizer import get_synthesizer
        
        logger.info(f"NYX: Creating conflict through synthesizer: {conflict_data.get('name', 'Unnamed')}")
        
        synthesizer = await get_synthesizer(self.user_id, self.conversation_id)
        
        # Map old conflict types to new slice-of-life types
        conflict_type = conflict_data.get('conflict_type', 'slice')
        if conflict_type in ['major', 'minor', 'standard']:
            # Convert old linear types to new types
            conflict_type = 'slice'  # Default to slice-of-life
        
        # Add Nyx governance metadata
        context = conflict_data.copy()
        context["governance_reason"] = reason
        context["created_by"] = "nyx_governance"
        
        result = await synthesizer.create_conflict(conflict_type, context)
        
        # Record in governance
        await self._record_narrative_event(
            event_type="conflict_created",
            details={
                "conflict_id": result.get("conflict_id"),
                "conflict_type": conflict_type,
                "reason": reason
            }
        )
        
        return result

    async def handle_agent_conflict(self, agent1_type: str, agent1_id: str, 
                                  agent2_type: str, agent2_id: str,
                                  conflict_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle conflicts between agents by making a governance decision.
        
        Args:
            agent1_type: Type of first agent
            agent1_id: ID of first agent
            agent2_type: Type of second agent
            agent2_id: ID of second agent
            conflict_details: Details of the conflict
            
        Returns:
            Resolution decision
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"NYX: Resolving conflict between {agent1_type}/{agent1_id} and {agent2_type}/{agent2_id}")

        # Analyze the conflict
        conflict_analysis = await self._analyze_agent_conflict(
            agent1_type, agent1_id, agent2_type, agent2_id, conflict_details
        )

        # Make a decision based on priorities and impact
        decision = await self._make_conflict_decision(conflict_analysis)

        # If the conflict involves world state changes, use LoreSystem
        if decision.get("requires_world_change"):
            ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
            
            for change in decision.get("world_changes", []):
                await self.lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type=change["entity_type"],
                    entity_identifier=change["identifier"],
                    updates=change["updates"],
                    reason=f"Conflict resolution: {decision.get('reasoning', 'Agent conflict')}"
                )

        # Import DirectiveType and DirectivePriority from constants
        from .constants import DirectiveType, DirectivePriority

        # Issue directives to the agents based on the decision
        if decision.get("agent1_directive"):
            await self.issue_directive(
                agent_type=agent1_type,
                agent_id=agent1_id,
                directive_type=DirectiveType.OVERRIDE,
                directive_data=decision["agent1_directive"],
                priority=DirectivePriority.HIGH
            )

        if decision.get("agent2_directive"):
            await self.issue_directive(
                agent_type=agent2_type,
                agent_id=agent2_id,
                directive_type=DirectiveType.OVERRIDE,
                directive_data=decision["agent2_directive"],
                priority=DirectivePriority.HIGH
            )

        return {
            "conflict_id": f"{agent1_type}_{agent1_id}_vs_{agent2_type}_{agent2_id}_{int(time.time())}",
            "decision": decision,
            "analysis": conflict_analysis,
            "timestamp": datetime.now().isoformat()
        }

    async def _analyze_agent_conflict(self, agent1_type: str, agent1_id: str,
                                    agent2_type: str, agent2_id: str,
                                    conflict_details: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a conflict between agents with dynamic scoring."""
        from .constants import AgentType
        
        analysis = {
            "agents": [
                {"type": agent1_type, "id": agent1_id},
                {"type": agent2_type, "id": agent2_id}
            ],
            "conflict_type": conflict_details.get("type", "unknown"),
            "severity": conflict_details.get("severity", 5),
            "narrative_impact": 0.0,
            "world_consistency_impact": 0.0,
            "player_experience_impact": 0.0,
            "resolution_difficulty": 0.0
        }
    
        # Dynamic scoring based on conflict type and context
        conflict_type = conflict_details.get("type", "unknown")
        
        # Base impact scores
        impact_matrix = {
            "narrative_contradiction": {
                "narrative_impact": 0.8,
                "world_consistency_impact": 0.6,
                "player_experience_impact": 0.4,
                "resolution_difficulty": 0.7
            },
            "resource_competition": {
                "narrative_impact": 0.3,
                "world_consistency_impact": 0.2,
                "player_experience_impact": 0.5,
                "resolution_difficulty": 0.4
            },
            "goal_conflict": {
                "narrative_impact": 0.5,
                "world_consistency_impact": 0.3,
                "player_experience_impact": 0.6,
                "resolution_difficulty": 0.5
            },
            "timing_conflict": {
                "narrative_impact": 0.4,
                "world_consistency_impact": 0.2,
                "player_experience_impact": 0.3,
                "resolution_difficulty": 0.3
            },
            "authority_conflict": {
                "narrative_impact": 0.6,
                "world_consistency_impact": 0.7,
                "player_experience_impact": 0.5,
                "resolution_difficulty": 0.8
            }
        }
        
        # Get base scores
        base_scores = impact_matrix.get(conflict_type, {
            "narrative_impact": 0.5,
            "world_consistency_impact": 0.5,
            "player_experience_impact": 0.5,
            "resolution_difficulty": 0.5
        })
        
        # Apply base scores
        for key, value in base_scores.items():
            analysis[key] = value
        
        # Adjust based on agent types and their importance
        agent_importance = {
            # World director has highest narrative authority in open-world mode
            AgentType.WORLD_DIRECTOR: 1.5,
            AgentType.UNIVERSAL_UPDATER: 1.3,
            AgentType.SCENE_MANAGER: 1.2,
            AgentType.CONFLICT_ANALYST: 1.1,
            AgentType.NPC: 0.9
        }
        
        # Calculate importance multiplier
        importance1 = agent_importance.get(agent1_type, 1.0)
        importance2 = agent_importance.get(agent2_type, 1.0)
        avg_importance = (importance1 + importance2) / 2
        
        # Scale impacts by importance
        analysis["narrative_impact"] *= avg_importance
        analysis["world_consistency_impact"] *= avg_importance
        
        # Adjust based on severity
        severity_multiplier = conflict_details.get("severity", 5) / 10.0
        for impact_type in ["narrative_impact", "world_consistency_impact", "player_experience_impact"]:
            analysis[impact_type] *= (0.5 + severity_multiplier)
        
        # Cap all values at 1.0
        for key in ["narrative_impact", "world_consistency_impact", "player_experience_impact", "resolution_difficulty"]:
            analysis[key] = min(1.0, analysis[key])
        
        # Add context about the conflict
        analysis["context"] = {
            "current_game_state": self.game_state,
            "recent_conflicts": len([c for c in self.coordination_history[-10:] if c.get("type") == "conflict"]),
            "agent_performance": {
                agent1_type: self.agent_performance.get(agent1_type, {}).get(agent1_id, {}),
                agent2_type: self.agent_performance.get(agent2_type, {}).get(agent2_id, {})
            }
        }
        
        return analysis

    async def _make_conflict_decision(self, conflict_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make a decision on how to resolve an agent conflict."""
        decision = {
            "resolution_type": "mediate",
            "reasoning": "",
            "requires_world_change": False,
            "world_changes": [],
            "agent1_directive": None,
            "agent2_directive": None,
            "coordination_required": []
        }
        
        # Analyze conflict type and severity
        conflict_type = conflict_analysis.get("conflict_type", "unknown")
        severity = conflict_analysis.get("severity", 5)
        
        # Get agent priorities from their types
        agent1 = conflict_analysis["agents"][0]
        agent2 = conflict_analysis["agents"][1]
        
        agent_priorities = {
            AgentType.WORLD_DIRECTOR: 10,
            AgentType.UNIVERSAL_UPDATER: 9,
            AgentType.CONFLICT_ANALYST: 8,
            AgentType.SCENE_MANAGER: 7,
            AgentType.NARRATIVE_CRAFTER: 6,
            AgentType.RELATIONSHIP_MANAGER: 5,
            AgentType.NPC: 4
        }
        
        priority1 = agent_priorities.get(agent1["type"], 1)
        priority2 = agent_priorities.get(agent2["type"], 1)
        
        # Decisions based on conflict analysis
        if conflict_analysis["narrative_impact"] > 0.8:
            # High narrative impact - prioritize story consistency
            if priority1 > priority2:
                winning_agent = agent1
                losing_agent = agent2
            else:
                winning_agent = agent2
                losing_agent = agent1
                
            decision["resolution_type"] = "narrative_priority"
            decision["reasoning"] = f"Prioritizing {winning_agent['type']} for narrative consistency"
            decision["agent1_directive"] = {
                "action": "proceed" if winning_agent == agent1 else "defer",
                "instruction": "Proceed with narrative-aligned action" if winning_agent == agent1 else "Defer to narrative requirements",
                "coordination_with": losing_agent["id"] if winning_agent == agent1 else None
            }
            decision["agent2_directive"] = {
                "action": "proceed" if winning_agent == agent2 else "defer",
                "instruction": "Proceed with narrative-aligned action" if winning_agent == agent2 else "Defer to narrative requirements",
                "coordination_with": losing_agent["id"] if winning_agent == agent2 else None
            }
            
        elif conflict_analysis["world_consistency_impact"] > 0.8:
            # High world consistency impact - enforce canon rules
            decision["resolution_type"] = "consistency_enforcement"
            decision["reasoning"] = "Enforcing world consistency through LoreSystem"
            decision["requires_world_change"] = True
            
            # Determine what world changes are needed
            if conflict_type == "resource_competition":
                decision["world_changes"] = [{
                    "entity_type": "Resources",
                    "action": "redistribute",
                    "details": "Redistribute resources to resolve competition"
                }]
            elif conflict_type == "narrative_contradiction":
                decision["world_changes"] = [{
                    "entity_type": "CanonicalEvents",
                    "action": "reconcile",
                    "details": "Reconcile contradictory narrative elements"
                }]
                
            # Both agents need to coordinate with LoreSystem
            decision["coordination_required"] = [AgentType.UNIVERSAL_UPDATER]
            
        elif severity >= 8:
            # High severity - requires multi-agent coordination
            decision["resolution_type"] = "coordinated_resolution"
            decision["reasoning"] = "High severity conflict requires coordinated response"
            
            # Bring in specialized agents
            if conflict_type == "resource_competition":
                decision["coordination_required"].append(AgentType.RESOURCE_OPTIMIZER)
            if conflict_type == "goal_conflict":
                decision["coordination_required"].append(AgentType.NARRATIVE_CRAFTER)
                
            # Set up coordination plan
            decision["agent1_directive"] = {
                "action": "coordinate",
                "instruction": "Participate in coordinated resolution",
                "wait_for": decision["coordination_required"]
            }
            decision["agent2_directive"] = {
                "action": "coordinate",
                "instruction": "Participate in coordinated resolution",
                "wait_for": decision["coordination_required"]
            }
            
        else:
            # Moderate conflicts - find compromise
            decision["resolution_type"] = "compromise"
            decision["reasoning"] = "Finding balanced solution through agent negotiation"
            
            # Analyze specific compromise based on conflict type
            if conflict_type == "timing_conflict":
                decision["agent1_directive"] = {
                    "action": "sequence",
                    "instruction": "Execute action first, then coordinate",
                    "notify_on_complete": agent2["id"]
                }
                decision["agent2_directive"] = {
                    "action": "wait",
                    "instruction": "Wait for agent1 completion, then proceed",
                    "wait_for": agent1["id"]
                }
            else:
                # General compromise
                decision["agent1_directive"] = {
                    "action": "modify",
                    "instruction": "Adjust approach to accommodate other agent",
                    "constraints": ["respect_agent2_goals"]
                }
                decision["agent2_directive"] = {
                    "action": "modify", 
                    "instruction": "Adjust approach to accommodate other agent",
                    "constraints": ["respect_agent1_goals"]
                }
        
        return decision
