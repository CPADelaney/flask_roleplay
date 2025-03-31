# logic/conflict_system/conflict_integration.py
"""
Integration module for Conflict System with Nyx governance.

This module provides classes and functions to properly integrate
the conflict system with Nyx central governance.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import asyncio
import asyncpg
from datetime import datetime

from agents import function_tool, RunContextWrapper
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance
from db.connection import get_db_connection_context

from logic.conflict_system.conflict_agents import (
    triage_agent, conflict_generation_agent, stakeholder_agent,
    manipulation_agent, resolution_agent, initialize_agents
)

logger = logging.getLogger(__name__)

class ConflictSystemIntegration:
    """
    Integration class for conflict system with Nyx governance.
    
    This class wraps the conflict system agents and provides 
    governance-compliant methods for permission checking, action reporting,
    and directive handling.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the conflict system integration."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.agents = None
        self.agent_id = "conflict_manager"
        self.is_initialized = False
        self.lore_system = None
        self.npc_system = None
        self.story_context = None
        self.db_dsn = None
        
    async def initialize(self):
        """Initialize the conflict system with all necessary systems."""
        if not self.is_initialized:
            self.agents = await initialize_agents()
            self.lore_system = await get_lore_system(self.user_id, self.conversation_id)
            self.npc_system = await get_npc_system(self.user_id, self.conversation_id)
            self.story_context = await self._get_story_context()
            self.is_initialized = True
            logger.info(f"Conflict system initialized for user {self.user_id}")
        return self
    
    async def _get_story_context(self) -> Dict[str, Any]:
        """Get current story context including NPCs and lore."""
        try:
            # Get active NPCs
            active_npcs = await self.npc_system.get_active_npcs()
            
            # Get relevant lore
            lore_context = await self.lore_system.get_narrative_elements(
                self.story_context.get('current_narrative_id')
            )
            
            # Get story progression
            story_progression = await self._get_story_progression()
            
            return {
                'active_npcs': active_npcs,
                'lore_context': lore_context,
                'story_progression': story_progression
            }
        except Exception as e:
            logger.error(f"Error getting story context: {e}")
            return {}
    
    async def check_permission(self, action_type: str, action_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if an action is permitted by Nyx governance.
        
        Args:
            action_type: Type of action being performed
            action_details: Details of the action
            
        Returns:
            Permission check result
        """
        governance = await get_central_governance(self.user_id, self.conversation_id)
        
        # Check permission with governance
        permission = await governance.check_action_permission(
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=self.agent_id,
            action_type=action_type,
            action_details=action_details
        )
        
        return permission
    
    async def report_action(self, action: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Report an action and its results to Nyx governance.
        
        Args:
            action: Information about the action performed
            result: Result of the action
            
        Returns:
            Action reporting result
        """
        governance = await get_central_governance(self.user_id, self.conversation_id)
        
        # Report action to governance
        report_result = await governance.process_agent_action_report(
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=self.agent_id,
            action=action,
            result=result
        )
        
        return report_result
    
    async def handle_directive(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a directive from Nyx governance.
        
        Args:
            directive: The directive to handle
            
        Returns:
            Directive handling result
        """
        await self.initialize()
        
        directive_type = directive.get("type")
        directive_data = directive.get("data", {})
        
        # Log directive receipt
        logger.info(f"Conflict system received directive: {directive_type}")
        logger.debug(f"Directive data: {json.dumps(directive_data, indent=2)}")
        
        result = {
            "success": False,
            "message": "Unhandled directive type",
            "directive_type": directive_type
        }
        
        if directive_type == DirectiveType.ACTION_REQUEST:
            # Handle action request directives
            result = await self._handle_action_directive(directive_data)
        elif directive_type == DirectiveType.SCENE_CHANGE:
            # Handle scene change directives
            result = await self._handle_scene_directive(directive_data)
        elif directive_type == DirectiveType.PROHIBITION:
            # Handle prohibition directives
            result = await self._handle_prohibition_directive(directive_data)
        elif directive_type == DirectiveType.INFORMATION:
            # Handle information directives
            result = await self._handle_information_directive(directive_data)
        
        # Return result of directive handling
        return result
        
    async def _handle_action_directive(self, directive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an action request directive."""
        action_type = directive_data.get("action_type")
        action_params = directive_data.get("parameters", {})
        
        if action_type == "generate_conflict":
            return await self.generate_conflict(action_params)
        elif action_type == "resolve_conflict":
            return await self.resolve_conflict(action_params)
        elif action_type == "update_stakeholders":
            return await self.update_stakeholders(action_params)
        elif action_type == "manage_manipulation":
            return await self.manage_manipulation(action_params)
        else:
            return {
                "success": False,
                "message": f"Conflict system doesn't know how to handle action type: {action_type}"
            }
    
    async def _handle_scene_directive(self, directive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a scene change directive."""
        # Extract scene details
        scene_type = directive_data.get("scene_type")
        location = directive_data.get("location")
        participants = directive_data.get("participants", [])
        
        # Log receipt of scene change
        logger.info(f"Conflict system aware of scene change to {location}")
        
        # Update any conflict data for this scene
        # This might involve checking if any conflicts are taking place in this location
        # Or adjusting conflict progression based on scene participants
        
        # Example: Check if any active conflicts involve this location
        # Implementation would depend on your conflict data storage
        
        return {
            "success": True,
            "message": "Conflict system is aware of scene change",
            "scene_type": scene_type,
            "location": location
        }
    
    async def _handle_prohibition_directive(self, directive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a prohibition directive."""
        # Extract prohibition details
        prohibited_action = directive_data.get("prohibited_action")
        reason = directive_data.get("reason")
        duration = directive_data.get("duration_minutes", 60)
        
        # Register this prohibition in the conflict system
        # This might involve temporarily disabling certain conflict types,
        # or preventing certain NPCs from getting involved in conflicts
        
        logger.info(f"Conflict system registering prohibition: {prohibited_action}")
        
        return {
            "success": True,
            "message": f"Prohibition registered: {prohibited_action}",
            "duration_minutes": duration
        }
        
    async def _handle_information_directive(self, directive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an information directive."""
        # Extract information details
        info_type = directive_data.get("info_type")
        info_content = directive_data.get("content")
        
        # Process the information based on its type
        if info_type == "npc_update":
            # Update NPC information for conflicts
            npc_id = directive_data.get("npc_id")
            logger.info(f"Updating conflict system with NPC changes for NPC {npc_id}")
        elif info_type == "lore_update":
            # Update lore-related conflict information
            logger.info(f"Updating conflict system with lore changes")
        elif info_type == "event_notification":
            # Process event notifications
            event_type = directive_data.get("event_type")
            logger.info(f"Conflict system notified of event: {event_type}")
        
        return {
            "success": True,
            "message": f"Information processed: {info_type}",
            "info_type": info_type
        }
    
    @with_governance(
        agent_type=AgentType.CONFLICT_ANALYST,
        action_type="generate_conflict",
        action_description="Generate a new conflict with stakeholders"
    )
    async def generate_conflict(self, conflict_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new conflict with enhanced integration."""
        await self.initialize()
        
        try:
            # Get story context
            story_context = await self._get_story_context()
            
            # Enhance conflict data with story context
            enhanced_data = {
                **conflict_data,
                'story_context': story_context,
                'active_npcs': story_context['active_npcs'],
                'lore_context': story_context['lore_context']
            }
            
            # Generate conflict with enhanced context
            generation_result = await self.agents["generation"].arun(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                conflict_type=enhanced_data.get("conflict_type", "interpersonal"),
                location=enhanced_data.get("location"),
                intensity=enhanced_data.get("intensity", "medium"),
                player_involvement=enhanced_data.get("player_involvement", "indirect"),
                duration=enhanced_data.get("duration", "medium"),
                topics=enhanced_data.get("topics", []),
                existing_npcs=enhanced_data.get("existing_npcs", []),
                story_context=story_context
            )
            
            # Generate stakeholders with enhanced context
            stakeholder_result = await self.agents["stakeholder"].arun(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                conflict_id=generation_result.get("conflict_id"),
                conflict_details=generation_result.get("conflict_details"),
                existing_npcs=enhanced_data.get("existing_npcs", []),
                required_roles=enhanced_data.get("required_roles", []),
                story_context=story_context
            )
            
            # Update lore with new conflict
            await self.lore_system.handle_narrative_event(
                self.run_ctx,
                f"New conflict generated: {generation_result.get('conflict_details', {}).get('name', 'Unnamed Conflict')}",
                affected_lore_ids=generation_result.get('affected_lore_ids', []),
                resolution_type="conflict_generation",
                impact_level="medium"
            )
            
            # Combine results
            combined_result = {
                "success": True,
                "conflict_id": generation_result.get("conflict_id"),
                "conflict_details": generation_result.get("conflict_details"),
                "stakeholders": stakeholder_result.get("stakeholders", []),
                "estimated_duration": generation_result.get("estimated_duration"),
                "player_hooks": generation_result.get("player_hooks", []),
                "story_context": story_context
            }
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Error generating conflict: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Error generating conflict: {str(e)}"
            }
    
    @with_governance(
        agent_type=AgentType.CONFLICT_ANALYST,
        action_type="resolve_conflict",
        action_description="Resolve an existing conflict"
    )
    async def resolve_conflict(self, resolution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a conflict with enhanced integration."""
        await self.initialize()
        
        try:
            # Get story context
            story_context = await self._get_story_context()
            
            # Enhance resolution data with story context
            enhanced_data = {
                **resolution_data,
                'story_context': story_context,
                'active_npcs': story_context['active_npcs'],
                'lore_context': story_context['lore_context']
            }
            
            # Call the conflict resolution agent with enhanced context
            resolution_result = await self.agents["resolution"].arun(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                conflict_id=enhanced_data.get("conflict_id"),
                resolution_type=enhanced_data.get("resolution_type", "compromise"),
                winner=enhanced_data.get("winner"),
                loser=enhanced_data.get("loser"),
                player_involvement=enhanced_data.get("player_involvement", False),
                player_action=enhanced_data.get("player_action"),
                resolution_details=enhanced_data.get("resolution_details"),
                story_context=story_context
            )
            
            # Update lore with conflict resolution
            await self.lore_system.handle_narrative_event(
                self.run_ctx,
                f"Conflict resolved: {resolution_result.get('conflict_details', {}).get('name', 'Unnamed Conflict')}",
                affected_lore_ids=resolution_result.get('affected_lore_ids', []),
                resolution_type=resolution_result.get('resolution_type', 'standard'),
                impact_level=resolution_result.get('impact_level', 'medium')
            )
            
            # Process consequences and rewards
            conflict_id = enhanced_data.get("conflict_id")
            conflict = await self.get_conflict_data(conflict_id)
            
            # Get player involvement
            player_involvement = conflict.get("player_involvement", {})
            involvement_level = player_involvement.get("involvement_level", "none")
            player_faction = player_involvement.get("faction", "neutral")
            
            # Get completed resolution paths
            completed_paths = resolution_result.get("completed_paths", [])
            if not completed_paths and "resolution_path" in resolution_result:
                completed_paths = [resolution_result["resolution_path"]]
            
            # Generate consequences including rewards
            consequences = generate_conflict_consequences(
                conflict.get("conflict_type", "standard"),
                "resolved",
                involvement_level,
                player_faction,
                completed_paths,
                story_context
            )
            
            # Process and apply rewards
            await self._apply_conflict_rewards(consequences)
            
            # Add consequences to the result
            resolution_result["consequences"] = consequences
            
            return {
                "success": True,
                "conflict_id": enhanced_data.get("conflict_id"),
                "resolution_type": resolution_result.get("resolution_type"),
                "resolution_details": resolution_result.get("resolution_details"),
                "aftermath": resolution_result.get("aftermath", {}),
                "relationship_changes": resolution_result.get("relationship_changes", []),
                "consequences": consequences,
                "story_context": story_context
            }
            
        except Exception as e:
            logger.error(f"Error resolving conflict: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Error resolving conflict: {str(e)}"
            }
    
    async def _apply_conflict_rewards(self, consequences: List[Dict[str, Any]]) -> None:
        """
        Apply conflict rewards to the player's inventory and status.
        
        Args:
            consequences: List of consequences including rewards
        """
        try:
            async with get_db_connection_context() as conn:
                # Begin transaction
                async with conn.transaction():
                    # Process each consequence
                    for consequence in consequences:
                        consequence_type = consequence.get("type")
                        
                        # Apply stat changes
                        if consequence_type == "player_stat" and "stat_changes" in consequence:
                            for stat_name, stat_change in consequence["stat_changes"].items():
                                # Update player stats
                                query = f"""
                                    UPDATE PlayerStats 
                                    SET {stat_name} = {stat_name} + $1
                                    WHERE user_id = $2 AND conversation_id = $3
                                """
                                await conn.execute(
                                    query, 
                                    stat_change, self.user_id, self.conversation_id
                                )
                                
                                logger.info(f"Updated player stat {stat_name} by {stat_change} for user {self.user_id}")
                        
                        # Add item rewards to inventory
                        elif consequence_type == "item_reward" and "item" in consequence:
                            item = consequence["item"]
                            
                            query = """
                                INSERT INTO PlayerInventory
                                (user_id, conversation_id, item_name, item_description, 
                                 item_category, item_properties, quantity, equipped)
                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            """
                            await conn.execute(
                                query,
                                self.user_id, 
                                self.conversation_id,
                                item["name"],
                                item["description"],
                                item.get("category", "conflict_reward"),
                                json.dumps({
                                    "rarity": item.get("rarity", "common"),
                                    "resolution_style": item.get("resolution_style", "neutral"),
                                    "source": "conflict_resolution"
                                }),
                                1,  # quantity
                                False  # not equipped by default
                            )
                            
                            logger.info(f"Added item {item['name']} to inventory for user {self.user_id}")
                        
                        # Add perks to player status
                        elif consequence_type == "perk_reward" and "perk" in consequence:
                            perk = consequence["perk"]
                            
                            # Check if perk already exists
                            query = """
                                SELECT perk_id FROM PlayerPerks
                                WHERE user_id = $1 AND conversation_id = $2 AND perk_name = $3
                            """
                            existing_perk = await conn.fetchrow(
                                query, 
                                self.user_id, self.conversation_id, perk["name"]
                            )
                            
                            if existing_perk:
                                # Perk exists, update tier if the new one is higher
                                update_query = """
                                    UPDATE PlayerPerks
                                    SET perk_tier = GREATEST(perk_tier, $1),
                                        perk_description = $2
                                    WHERE user_id = $3 AND conversation_id = $4 AND perk_name = $5
                                """
                                await conn.execute(
                                    update_query,
                                    perk.get("tier", 1),
                                    perk["description"],
                                    self.user_id,
                                    self.conversation_id,
                                    perk["name"]
                                )
                            else:
                                # New perk, insert it
                                insert_query = """
                                    INSERT INTO PlayerPerks
                                    (user_id, conversation_id, perk_name, perk_description, 
                                     perk_category, perk_tier, perk_properties)
                                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                                """
                                await conn.execute(
                                    insert_query,
                                    self.user_id,
                                    self.conversation_id,
                                    perk["name"],
                                    perk["description"],
                                    perk.get("category", "conflict_resolution"),
                                    perk.get("tier", 1),
                                    json.dumps({
                                        "resolution_style": perk.get("resolution_style", "neutral"),
                                        "source": "conflict_resolution"
                                    })
                                )
                            
                            logger.info(f"Added/updated perk {perk['name']} for user {self.user_id}")
                        
                        # Add special rewards
                        elif consequence_type == "special_reward" and "special_reward" in consequence:
                            special = consequence["special_reward"]
                            
                            # Special rewards are unique items with special effects
                            query = """
                                INSERT INTO PlayerSpecialRewards
                                (user_id, conversation_id, reward_name, reward_description, 
                                 reward_effect, reward_category, reward_properties, used)
                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            """
                            await conn.execute(
                                query,
                                self.user_id,
                                self.conversation_id,
                                special["name"],
                                special["description"],
                                special.get("effect", ""),
                                special.get("category", "unique_conflict_reward"),
                                json.dumps({
                                    "resolution_style": special.get("resolution_style", "neutral"),
                                    "source": "major_conflict_resolution"
                                }),
                                False  # not used yet
                            )
                            
                            logger.info(f"Added special reward {special['name']} for user {self.user_id}")
                        
                        # Apply NPC relationship changes
                        elif consequence_type == "npc_relationship" and consequence.get("npc_id"):
                            npc_id = consequence.get("npc_id")
                            relationship_change = consequence.get("relationship_change", {})
                            
                            # Update NPC relationship
                            for rel_type, change_amount in relationship_change.items():
                                # Skip if no change
                                if change_amount == 0:
                                    continue
                                    
                                query = f"""
                                    UPDATE NPCStats
                                    SET {rel_type} = {rel_type} + $1
                                    WHERE npc_id = $2 AND user_id = $3 AND conversation_id = $4
                                """
                                await conn.execute(
                                    query, 
                                    change_amount, npc_id, self.user_id, self.conversation_id
                                )
                                
                            logger.info(f"Updated relationship with NPC {npc_id} for user {self.user_id}")
                            
        except Exception as e:
            logger.error(f"Error applying conflict rewards: {str(e)}", exc_info=True)
            raise
    
    @with_governance(
        agent_type=AgentType.CONFLICT_ANALYST,
        action_type="update_stakeholders",
        action_description="Update conflict stakeholders"
    )
    async def update_stakeholders(self, stakeholder_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update stakeholders for an existing conflict.
        
        Args:
            stakeholder_data: Data containing stakeholder updates
            
        Returns:
            Update result
        """
        await self.initialize()
        
        try:
            # Call the stakeholder agent to update stakeholders
            stakeholder_result = await self.agents["stakeholder"].arun(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                conflict_id=stakeholder_data.get("conflict_id"),
                action="update",
                updates=stakeholder_data.get("updates", []),
                add_stakeholders=stakeholder_data.get("add_stakeholders", []),
                remove_stakeholders=stakeholder_data.get("remove_stakeholders", [])
            )
            
            return {
                "success": True,
                "conflict_id": stakeholder_data.get("conflict_id"),
                "updated_stakeholders": stakeholder_result.get("updated_stakeholders", []),
                "added_stakeholders": stakeholder_result.get("added_stakeholders", []),
                "removed_stakeholders": stakeholder_result.get("removed_stakeholders", [])
            }
            
        except Exception as e:
            logger.error(f"Error updating conflict stakeholders: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Error updating conflict stakeholders: {str(e)}"
            }
    
    @with_governance(
        agent_type=AgentType.CONFLICT_ANALYST,
        action_type="manage_manipulation",
        action_description="Manage NPC manipulation attempts"
    )
    async def manage_manipulation(self, manipulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage manipulation attempts in a conflict.
        
        Args:
            manipulation_data: Data about the manipulation attempt
            
        Returns:
            Manipulation result
        """
        await self.initialize()
        
        try:
            # Call the manipulation agent
            manipulation_result = await self.agents["manipulation"].arun(
                user_id=self.user_id,
                conversation_id=self.conversation_id,
                conflict_id=manipulation_data.get("conflict_id"),
                manipulator_id=manipulation_data.get("manipulator_id"),
                target_id=manipulation_data.get("target_id"),
                manipulation_type=manipulation_data.get("manipulation_type", "persuasion"),
                manipulation_goal=manipulation_data.get("manipulation_goal"),
                manipulation_details=manipulation_data.get("manipulation_details"),
                player_assisted=manipulation_data.get("player_assisted", False)
            )
            
            return {
                "success": True,
                "manipulation_id": manipulation_result.get("manipulation_id"),
                "success_rate": manipulation_result.get("success_rate"),
                "outcome": manipulation_result.get("outcome"),
                "target_reaction": manipulation_result.get("target_reaction"),
                "relationship_impact": manipulation_result.get("relationship_impact", {})
            }
            
        except Exception as e:
            logger.error(f"Error managing manipulation: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Error managing manipulation: {str(e)}"
            }
    
    async def process_request(self, request_type: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request through the conflict system with governance integration.
        
        Args:
            request_type: Type of request (generate, resolve, update, manipulate)
            request_data: Data needed for the request
            
        Returns:
            Request processing result
        """
        await self.initialize()
        
        # All methods below already have governance integration via @with_governance
        if request_type == "generate_conflict":
            return await self.generate_conflict(request_data)
        elif request_type == "resolve_conflict":
            return await self.resolve_conflict(request_data)
        elif request_type == "update_stakeholders":
            return await self.update_stakeholders(request_data)
        elif request_type == "manage_manipulation":
            return await self.manage_manipulation(request_data)
        else:
            return {
                "success": False,
                "message": f"Unknown request type: {request_type}"
            }

    @with_governance(
        agent_type=AgentType.CONFLICT_ANALYST,
        action_type="get_conflicts_by_location",
        action_description="Getting conflicts for location: {location}"
    )
    async def get_conflicts_by_location(self, location: str) -> List[Dict[str, Any]]:
        """
        Get all conflicts associated with a specific location.
        
        Args:
            location: Name of the location
            
        Returns:
            List of conflict data dictionaries
        """
        try:
            if not hasattr(self, 'agents') or self.agents is None:
                await self.initialize()
                
            async with get_db_connection_context() as conn:
                # First get location ID if we need it
                location_id = await conn.fetchval("""
                    SELECT id FROM Locations 
                    WHERE location_name = $1 AND user_id = $2 AND conversation_id = $3
                """, location, self.user_id, self.conversation_id)
                
                # Get conflicts directly associated with this location
                rows = await conn.fetch("""
                    SELECT c.*, cp.resolution_path, cp.completion_status
                    FROM Conflicts c
                    LEFT JOIN ConflictPaths cp ON c.id = cp.conflict_id 
                        AND cp.user_id = c.user_id 
                        AND cp.conversation_id = c.conversation_id
                    WHERE c.user_id = $1 AND c.conversation_id = $2
                    AND (c.location = $3 OR c.location_id = $4)
                    ORDER BY c.created_at DESC
                """, self.user_id, self.conversation_id, location, location_id)
                
                # Format the results
                conflicts = []
                for row in rows:
                    conflict_data = dict(row)
                    
                    # Get additional details if needed
                    conflict_details = await self._extract_conflict_details(conflict_data)
                    
                    # Add to result list
                    conflicts.append(conflict_details)
                
                return conflicts
                
        except Exception as e:
            logger.error(f"Error getting conflicts by location: {e}", exc_info=True)
            return []
            
    async def _extract_conflict_details(self, conflict_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and format conflict details from the database row.
        
        Args:
            conflict_data: Raw conflict data from database
            
        Returns:
            Formatted conflict details
        """
        # Basic details
        details = {
            "id": conflict_data.get("id"),
            "name": conflict_data.get("name", "Unnamed Conflict"),
            "type": conflict_data.get("conflict_type", "standard"),
            "status": conflict_data.get("status", "inactive"),
            "location": conflict_data.get("location", "Unknown"),
            "description": conflict_data.get("description", ""),
            "phase": conflict_data.get("phase", "brewing"),
            "faction_a": conflict_data.get("faction_a_name", ""),
            "faction_b": conflict_data.get("faction_b_name", ""),
            "resolution_path": conflict_data.get("resolution_path", ""),
            "completion_status": conflict_data.get("completion_status", "incomplete")
        }
        
        # Check for completion
        details["is_resolved"] = details["status"] == "resolved"
        
        # Progress calculation based on phase
        phases = ["brewing", "active", "climax", "resolved"]
        phase_idx = phases.index(details["phase"]) if details["phase"] in phases else 0
        details["progress"] = (phase_idx / (len(phases) - 1)) * 100  # As percentage
        
        return details

    async def resolve_conflict_with_lore(self, conflict_id: int) -> Dict[str, Any]:
        """Resolve conflict with lore system integration.
        
        Args:
            conflict_id: The ID of the conflict to resolve
            
        Returns:
            Dict[str, Any]: Dictionary containing resolution details and affected lore
        """
        try:
            # Get lore system instance
            lore_system = await get_lore_system(self.user_id, self.conversation_id)
            
            # Get conflict data
            conflict_data = await self.get_conflict_data(conflict_id)
            
            # Get stakeholder analysis
            stakeholders = await self.get_stakeholder_analysis(conflict_id)
            
            # Generate resolution through Nyx governance
            resolution = await self.generate_conflict_resolution(
                conflict_id,
                stakeholders=stakeholders
            )
            
            # Handle narrative event through lore system
            narrative_result = await lore_system.handle_narrative_event(
                self.run_ctx,
                f"Conflict resolution: {conflict_data['description']}",
                affected_lore_ids=conflict_data['affected_lore_ids'],
                resolution_type=resolution['type'],
                impact_level=resolution['impact_level']
            )
            
            # Update relationships based on resolution
            await self.update_relationships_from_resolution(
                conflict_id,
                resolution,
                stakeholders
            )
            
            # Combine results
            result = {
                'conflict_id': conflict_id,
                'resolution': resolution,
                'narrative_impact': narrative_result,
                'affected_stakeholders': stakeholders,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Add to memory system
            await self.memory_system.add_memory(
                self.run_ctx,
                f"Conflict resolution for conflict {conflict_id}",
                result,
                memory_type="conflict_resolution"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error resolving conflict {conflict_id}: {e}", exc_info=True)
            return {
                'error': str(e),
                'conflict_id': conflict_id,
                'status': 'failed'
            }

async def register_with_governance(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Register the conflict system with Nyx governance.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Registration result
    """
    try:
        # Get governance
        governance = await get_central_governance(user_id, conversation_id)
        
        # Create conflict system instance
        conflict_system = ConflictSystemIntegration(user_id, conversation_id)
        await conflict_system.initialize()
        
        # Register with governance
        await governance.governor.register_agent(
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_instance=conflict_system
        )
        
        # Store in local registry
        governance.registered_agents[AgentType.CONFLICT_ANALYST] = conflict_system
        
        logger.info("Conflict System registered with Nyx governance")
        
        return {
            "success": True,
            "message": "Conflict System successfully registered with governance"
        }
    except Exception as e:
        logger.error(f"Error registering conflict system: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"Failed to register Conflict System with governance: {str(e)}"
        }

# The enhanced integration class is retained as in the original file

# Add async initialization and proper error handling to the EnhancedConflictSystemIntegration class
class EnhancedConflictSystemIntegration:
    """
    Enhanced integration class for conflict system with Nyx governance.
    
    This class adds more sophisticated directive handling and better integration
    with other game systems.
    """
    
    async def handle_directive(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a directive from Nyx governance with enhanced capabilities.
        
        Args:
            directive: The directive to handle
            
        Returns:
            Directive handling result
        """
        directive_type = directive.get("type")
        directive_data = directive.get("data", {})
        
        # Enhanced directive handling logic
        if directive_type == DirectiveType.ACTION_REQUEST:
            return await self._handle_action_directive(directive_data)
        elif directive_type == DirectiveType.SCENE_CHANGE:
            return await self._handle_scene_directive(directive_data)
        elif directive_type == DirectiveType.PROHIBITION:
            return await self._handle_prohibition_directive(directive_data)
        elif directive_type == DirectiveType.INFORMATION:
            return await self._handle_information_directive(directive_data)
        else:
            return {
                "success": False,
                "message": f"Enhanced conflict system doesn't know how to handle this directive type: {directive_type}"
            }
    
    async def _handle_action_directive(self, directive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an action request directive with enhanced capabilities."""
        action_type = directive_data.get("action_type")
        action_params = directive_data.get("parameters", {})
        
        # Enhanced action handling
        if action_type == "generate_conflict":
            # Augment with additional context or parameters
            augmented_params = {**action_params}
            
            # Add contextual information like game time, season, etc. if needed
            augmented_params["enhanced"] = True
            
            return await self.generate_conflict(augmented_params)
        elif action_type == "resolve_conflict":
            return await self.resolve_conflict(action_params)
        elif action_type == "update_stakeholders":
            return await self.update_stakeholders(action_params)
        elif action_type == "manage_manipulation":
            return await self.manage_manipulation(action_params)
        else:
            return {
                "success": False,
                "message": "Conflict system doesn't know how to handle this directive type"
            }
    
    async def _handle_scene_directive(self, directive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a scene change directive with enhanced capabilities."""
        scene_type = directive_data.get("scene_type")
        location = directive_data.get("location")
        participants = directive_data.get("participants", [])
        
        # Enhanced processing for scene changes
        # You might automatically generate location-appropriate conflicts here
        
        # For example, automatically creating tensions in a crowded marketplace
        if scene_type == "public_space" and len(participants) > 3:
            # Create subtle background conflict
            conflict_data = {
                "conflict_type": "background",
                "location": location,
                "intensity": "low",
                "player_involvement": "observable",
                "duration": "short",
                "existing_npcs": participants
            }
            
            # Schedule or immediately create the conflict
            asyncio.create_task(self.generate_conflict(conflict_data))
        
        return {
            "success": True,
            "message": "Conflict system is aware of scene change",
            "scene_type": scene_type,
            "location": location
        }
    
    async def _handle_prohibition_directive(self, directive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a prohibition directive with enhanced capabilities."""
        prohibited_action = directive_data.get("prohibited_action")
        reason = directive_data.get("reason")
        duration = directive_data.get("duration_minutes", 60)
        
        # Enhanced prohibition handling
        # You might track and adjust conflicts based on prohibitions
        
        # For example, you could have the conflict system "rebel" against certain prohibitions
        # by creating tensions that push against boundaries
        if prohibited_action == "violence" and reason == "public_location":
            # Create subtle tension that tests the boundary
            tension_data = {
                "tension_type": "building_hostility",
                "location": directive_data.get("location"),
                "visible": False,
                "escalation_chance": 0.3
            }
            
            # Store this tension for later development
            # This would be implemented in your conflict tracking system
        
        return {
            "success": True,
            "message": f"Enhanced prohibition handling: {prohibited_action}",
            "duration_minutes": duration
        }
        
    async def _handle_information_directive(self, directive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an information directive with enhanced capabilities."""
        info_type = directive_data.get("info_type")
        info_content = directive_data.get("content")
        
        # Enhanced information handling
        if info_type == "npc_update":
            # Update NPC record and check if this affects active conflicts
            npc_id = directive_data.get("npc_id")
            update_type = directive_data.get("update_type")
            
            # Check for conflict impact
            # This would connect to your conflict tracking system
            if update_type == "mood_change" and directive_data.get("new_mood") == "angry":
                # This might trigger a conflict escalation
                pass
                
        elif info_type == "lore_update":
            # Apply lore changes to conflict generation parameters
            pass
            
        elif info_type == "world_event":
            # World events might trigger new conflicts
            event_type = directive_data.get("event_type")
            if event_type in ["festival", "disaster", "war"]:
                # These events might warrant special conflict generation
                pass
        
        return {
            "success": True,
            "message": f"Enhanced information handling: {info_type}"
        }
    
    @with_governance(
        agent_type=AgentType.CONFLICT_ANALYST,
        action_type="generate_conflict",
        action_description="Generate a new conflict with stakeholders"
    )
    async def generate_conflict(self, conflict_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a new conflict with stakeholders using enhanced capabilities.
        
        Args:
            conflict_data: Data needed to generate the conflict
            
        Returns:
            Generated conflict data
        """
        # Enhanced conflict generation
        # This would include added features like:
        # - Connection to lore system
        # - Integration with NPC personality models
        # - Game world conditions affecting conflict parameters
        
        if not hasattr(self, 'agents') or self.agents is None:
            await self.initialize()
        
        try:
            # Base implementation from ConflictSystemIntegration
            base_result = await super().generate_conflict(conflict_data)
            
            # Add enhanced data
            base_result["enhanced"] = True
            base_result["connected_systems"] = ["lore", "npc", "world_state"]
            
            # You might add additional processing here
            
            return base_result
            
        except Exception as e:
            logger.error(f"Error in enhanced conflict generation: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Error in enhanced conflict generation: {str(e)}"
            }
            
    # ... other enhanced methods would follow the same pattern ...
    
    async def process_request(self, request_type: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request through the conflict system with enhanced governance integration.
        
        Args:
            request_type: Type of request (generate, resolve, update, manipulate)
            request_data: Data needed for the request
            
        Returns:
            Request processing result
        """
        # Enhanced request processing
        if request_type == "generate_conflict":
            return await self.generate_conflict(request_data)
        elif request_type == "resolve_conflict":
            return await self.resolve_conflict(request_data)
        elif request_type == "update_stakeholders":
            return await self.update_stakeholders(request_data)
        elif request_type == "manage_manipulation":
            return await self.manage_manipulation(request_data)
        else:
            return {
                "success": False,
                "message": f"Unknown request type: {request_type}"
            }

async def register_enhanced_integration(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Register the enhanced conflict system with Nyx governance.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Registration result
    """
    try:
        # Get governance
        governance = await get_central_governance(user_id, conversation_id)
        
        # Create a class that inherits from both integration classes
        class NyxEnhancedConflictSystem(ConflictSystemIntegration, EnhancedConflictSystemIntegration):
            async def initialize(self):
                """Initialize both base and enhanced systems."""
                await ConflictSystemIntegration.initialize(self)
                # Any enhanced initialization would go here
                return self
        
        # Create and initialize conflict system integration
        enhanced_system = NyxEnhancedConflictSystem(user_id, conversation_id)
        await enhanced_system.initialize()
        
        # Register with governance
        await governance.governor.register_agent(
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_instance=enhanced_system
        )
        
        # Store in local registry
        governance.registered_agents[AgentType.CONFLICT_ANALYST] = enhanced_system
        
        # Register conflict system's ability to handle specific directive types
        await governance.governor.register_directive_handler(
            agent_type=AgentType.CONFLICT_ANALYST,
            directive_types=[
                DirectiveType.ACTION_REQUEST,
                DirectiveType.SCENE_CHANGE,
                DirectiveType.PROHIBITION,
                DirectiveType.INFORMATION
            ],
            handler=enhanced_system
        )
        
        # Announce the registration
        logging.info("Enhanced Conflict System registered with Nyx governance")
        
        return {
            "success": True,
            "message": "Enhanced Conflict System successfully registered with governance",
            "registered_directive_types": [
                DirectiveType.ACTION_REQUEST,
                DirectiveType.SCENE_CHANGE,
                DirectiveType.PROHIBITION,
                DirectiveType.INFORMATION
            ]
        }
    except Exception as e:
        logger.error(f"Error registering enhanced conflict system: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"Failed to register Enhanced Conflict System with governance: {str(e)}"
        }

@function_tool
async def register_enhanced_conflict_system_tool(ctx: RunContextWrapper) -> Dict[str, Any]:
    """Register the enhanced conflict system with governance."""
    user_id = ctx.agent_context.get("user_id")
    conversation_id = ctx.agent_context.get("conversation_id")
    
    if not user_id or not conversation_id:
        return {
            "success": False,
            "message": "Missing user_id or conversation_id in context"
        }
    
    return await register_enhanced_integration(user_id, conversation_id)
