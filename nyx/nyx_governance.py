# nyx/nyx_governance.py

"""
Central governance system for Nyx to control all NPC agents.
This module ensures Nyx has ultimate authority over all NPCs and their actions.
"""

import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import asyncpg

from agents import trace
from db.connection import get_db_connection
from memory.wrapper import MemorySystem
from nyx.nyx_memory_system import NyxMemorySystem
from utils.caching import CACHE_TTL, NPC_DIRECTIVE_CACHE

logger = logging.getLogger(__name__)

class DirectiveType:
    """Constants for directive types"""
    ACTION = "action"
    MOVEMENT = "movement"
    DIALOGUE = "dialogue"
    RELATIONSHIP = "relationship"
    EMOTION = "emotion"
    PROHIBITION = "prohibition"
    SCENE = "scene"
    OVERRIDE = "override"

class DirectivePriority:
    """Constants for directive priorities"""
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    CRITICAL = 10

class NyxGovernor:
    """
    Central governance system for Nyx to control all NPC agents.
    
    This class provides:
    1. Permission checking for all NPC actions
    2. Directive management
    3. Action reporting and monitoring
    4. Override capabilities
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the governance system.
        
        Args:
            user_id: The user/player ID
            conversation_id: The current conversation/scene ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.memory_system = None
        self.directive_cache = {}
        self.action_history = {}
        self.lock = asyncio.Lock()
    
    async def get_memory_system(self) -> NyxMemorySystem:
        """Lazy-load the memory system."""
        if self.memory_system is None:
            self.memory_system = NyxMemorySystem(self.user_id, self.conversation_id)
        return self.memory_system
    
    async def check_action_permission(
        self, 
        npc_id: int, 
        action_type: str, 
        action_details: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Check if an NPC is allowed to perform an action.
        
        Args:
            npc_id: ID of the NPC requesting permission
            action_type: Type of action (e.g., "dialogue", "movement", "attack")
            action_details: Details of the action
            context: Additional context information
        
        Returns:
            Dict with permission decision and possible override
        """
        with trace(workflow_name=f"NPC {npc_id} Permission Check"):
            # Get any directives for this NPC
            directives = await self.get_npc_directives(npc_id)
            
            # Default response - approve but track
            response = {
                "approved": True,
                "directive_applied": False,
                "override_action": None,
                "reasoning": "No applicable directives found",
                "tracking_id": await self._track_action_request(npc_id, action_type, action_details, context)
            }
            
            # Check for prohibitions first (directives that explicitly forbid actions)
            for directive in directives:
                if directive.get("type") == DirectiveType.PROHIBITION:
                    prohibited_actions = directive.get("prohibited_actions", [])
                    
                    if action_type in prohibited_actions or "*" in prohibited_actions:
                        response.update({
                            "approved": False,
                            "directive_applied": True,
                            "directive_id": directive.get("id"),
                            "reasoning": directive.get("reason", "Action prohibited by Nyx directive")
                        })
                        
                        # Check if there's an alternative action suggested
                        if "alternative_action" in directive:
                            response["override_action"] = directive["alternative_action"]
                        
                        return response
            
            # Check for override directives next
            for directive in directives:
                if directive.get("type") == DirectiveType.OVERRIDE:
                    # Check if this directive applies to the current action type
                    applies_to = directive.get("applies_to", [])
                    
                    if action_type in applies_to or "*" in applies_to:
                        # Apply the override
                        response.update({
                            "approved": False,  # Reject original action
                            "directive_applied": True,
                            "directive_id": directive.get("id"),
                            "override_action": directive.get("override_action"),
                            "reasoning": directive.get("reason", "Action overridden by Nyx directive")
                        })
                        
                        return response
            
            # Check for action-specific directives
            for directive in directives:
                if directive.get("type") == action_type:
                    # This is a specific directive for this action type
                    response.update({
                        "approved": True,  # Allow but modify
                        "directive_applied": True,
                        "directive_id": directive.get("id"),
                        "action_modifications": directive.get("modifications", {}),
                        "reasoning": directive.get("reason", "Action modified by Nyx directive")
                    })
                    
                    return response
            
            # If we get here, no specific directives applied
            return response
    
    async def get_npc_directives(self, npc_id: int) -> List[Dict[str, Any]]:
        """
        Get all active directives for an NPC.
        
        Args:
            npc_id: ID of the NPC
        
        Returns:
            List of active directives
        """
        # Check cache first
        cache_key = f"directives:{self.user_id}:{self.conversation_id}:{npc_id}"
        cached = NPC_DIRECTIVE_CACHE.get(cache_key)
        
        if cached:
            return cached
        
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT id, directive, priority, expires_at
                        FROM NyxNPCDirectives
                        WHERE user_id = $1
                          AND conversation_id = $2
                          AND npc_id = $3
                          AND expires_at > NOW()
                        ORDER BY priority DESC
                    """, self.user_id, self.conversation_id, npc_id)
                    
                    directives = []
                    for row in rows:
                        directive_data = json.loads(row["directive"])
                        directive_data["id"] = row["id"]
                        directive_data["priority"] = row["priority"]
                        directive_data["expires_at"] = row["expires_at"].isoformat()
                        directives.append(directive_data)
                    
                    # Cache the result
                    NPC_DIRECTIVE_CACHE.set(cache_key, directives, CACHE_TTL["directives"])
                    
                    return directives
        except Exception as e:
            logger.error(f"Error fetching directives for NPC {npc_id}: {e}")
            return []
    
    async def issue_directive(
        self,
        npc_id: int,
        directive_type: str,
        directive_data: Dict[str, Any],
        priority: int = DirectivePriority.MEDIUM,
        duration_minutes: int = 30,
        scene_id: str = None
    ) -> int:
        """
        Issue a new directive to an NPC.
        
        Args:
            npc_id: ID of the target NPC
            directive_type: Type of directive (use DirectiveType constants)
            directive_data: Directive details
            priority: Priority level (use DirectivePriority constants)
            duration_minutes: How long this directive should remain active
            scene_id: Optional ID of the scene this directive is part of
        
        Returns:
            ID of the created directive
        """
        try:
            directive = {
                "type": directive_type,
                "timestamp": datetime.now().isoformat(),
                **directive_data
            }
            
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        INSERT INTO NyxNPCDirectives (
                            user_id, conversation_id, npc_id, directive, 
                            expires_at, priority, scene_id
                        )
                        VALUES ($1, $2, $3, $4, NOW() + $5::INTERVAL, $6, $7)
                        RETURNING id
                    """, 
                        self.user_id, 
                        self.conversation_id, 
                        npc_id,
                        json.dumps(directive),
                        f"{duration_minutes} minutes",
                        priority,
                        scene_id
                    )
                    
                    directive_id = row["id"]
                    
                    # Invalidate cache
                    cache_key = f"directives:{self.user_id}:{self.conversation_id}:{npc_id}"
                    NPC_DIRECTIVE_CACHE.delete(cache_key)
                    
                    # Log directive
                    await self._log_directive(npc_id, directive_id, directive_type, directive_data)
                    
                    return directive_id
        except Exception as e:
            logger.error(f"Error issuing directive to NPC {npc_id}: {e}")
            return -1
    
    async def revoke_directive(self, directive_id: int) -> bool:
        """
        Revoke a directive immediately.
        
        Args:
            directive_id: ID of the directive to revoke
        
        Returns:
            True if successful, False otherwise
        """
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    # First, get directive details to invalidate cache
                    row = await conn.fetchrow("""
                        SELECT npc_id FROM NyxNPCDirectives
                        WHERE id = $1 AND user_id = $2 AND conversation_id = $3
                    """, directive_id, self.user_id, self.conversation_id)
                    
                    if not row:
                        return False
                    
                    npc_id = row["npc_id"]
                    
                    # Now expire the directive
                    await conn.execute("""
                        UPDATE NyxNPCDirectives
                        SET expires_at = NOW()
                        WHERE id = $1
                    """, directive_id)
                    
                    # Invalidate cache
                    cache_key = f"directives:{self.user_id}:{self.conversation_id}:{npc_id}"
                    NPC_DIRECTIVE_CACHE.delete(cache_key)
                    
                    return True
        except Exception as e:
            logger.error(f"Error revoking directive {directive_id}: {e}")
            return False
    
    async def process_npc_action_report(
        self,
        npc_id: int,
        action: Dict[str, Any],
        result: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process an action report from an NPC and determine if intervention is needed.
        
        Args:
            npc_id: ID of the NPC
            action: The action taken
            result: Result of the action
            context: Additional context
        
        Returns:
            Response with possible override
        """
        async with self.lock:
            # Record the action
            tracking_id = await self._track_completed_action(npc_id, action, result, context)
            
            # Create memory of this action
            memory_system = await self.get_memory_system()
            
            # Get NPC name
            npc_name = await self._get_npc_name(npc_id)
            
            # Add memory for Nyx
            await memory_system.add_memory(
                memory_text=f"{npc_name} performed: {action.get('description', 'unknown action')}",
                memory_type="observation",
                memory_scope="game",
                significance=min(abs(result.get("emotional_impact", 0)) + 4, 10),
                tags=["npc_action", f"npc_{npc_id}"],
                metadata={
                    "npc_id": npc_id,
                    "action": action,
                    "result": result,
                    "tracking_id": tracking_id
                }
            )
            
            # Determine if Nyx should intervene based on action severity
            intervention_needed = await self._check_if_intervention_needed(npc_id, action, result)
            
            if intervention_needed:
                # Create an override directive
                directive_id = await self.issue_directive(
                    npc_id=npc_id,
                    directive_type=DirectiveType.OVERRIDE,
                    directive_data={
                        "reason": intervention_needed.get("reason"),
                        "override_action": intervention_needed.get("override_action"),
                        "applies_to": ["*"],  # Apply to all action types
                        "source": "action_evaluation"
                    },
                    priority=DirectivePriority.HIGH,
                    duration_minutes=10
                )
                
                return {
                    "intervention": True,
                    "directive_id": directive_id,
                    "reason": intervention_needed.get("reason"),
                    "tracking_id": tracking_id
                }
            
            return {
                "intervention": False,
                "tracking_id": tracking_id
            }
    
    async def _check_if_intervention_needed(
        self,
        npc_id: int,
        action: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Determine if Nyx should intervene based on the action and its results.
        
        Args:
            npc_id: ID of the NPC
            action: The action taken
            result: Result of the action
            
        Returns:
            Intervention details or None if no intervention needed
        """
        # Rules-based evaluation system
        emotional_impact = result.get("emotional_impact", 0)
        action_type = action.get("type", "unknown")
        target = action.get("target", "unknown")
        description = action.get("description", "")
        
        # Get the current narrative context
        narrative_context = await self._get_current_narrative_context()
        current_arcs = narrative_context.get("active_arcs", [])
        
        # 1. Check for contradiction with narrative arcs
        for arc in current_arcs:
            # Check if this NPC has a specific role in the arc
            for npc_role in arc.get("npc_roles", []):
                if npc_role.get("npc_id") == npc_id:
                    required_relationship = npc_role.get("relationship")
                    required_action = npc_role.get("required_action")
                    
                    # Check for relationship contradiction
                    if required_relationship == "friendly" and action_type in ["mock", "attack", "threaten"]:
                        return {
                            "reason": f"Action contradicts NPC's friendly role in current narrative arc",
                            "override_action": {
                                "type": "talk",
                                "description": "speak in a more friendly manner",
                                "target": target
                            }
                        }
                    
                    # Check for action contradiction
                    if required_action == "help" and action_type == "leave":
                        return {
                            "reason": f"NPC is required to help in current narrative arc",
                            "override_action": {
                                "type": "assist",
                                "description": "offer assistance instead of leaving",
                                "target": target
                            }
                        }
        
        # 2. Check for excessive emotional impact
        if abs(emotional_impact) > 7:
            return {
                "reason": "Action has excessive emotional impact",
                "override_action": {
                    "type": action_type,
                    "description": description.replace("forcefully", "moderately").replace("aggressively", "assertively"),
                    "target": target
                }
            }
        
        # 3. Check for coordinated group actions
        if target in ["group", "player"] and action_type in ["attack", "threaten", "dominate"]:
            # Check if other NPCs are also targeting the same entity
            recent_actions = await self._get_recent_similar_actions(action_type, target)
            if len(recent_actions) >= 2:
                return {
                    "reason": "Too many NPCs performing similar actions against the same target",
                    "override_action": {
                        "type": "observe",
                        "description": "wait and observe instead of joining the others",
                        "target": "environment"
                    }
                }
        
        # No intervention needed
        return None
    
    async def _get_recent_similar_actions(self, action_type: str, target: str) -> List[Dict[str, Any]]:
        """Get recent similar actions by other NPCs."""
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT action_data
                        FROM NyxActionTracking
                        WHERE user_id = $1
                          AND conversation_id = $2
                          AND action_data->>'type' = $3
                          AND action_data->>'target' = $4
                          AND timestamp > NOW() - INTERVAL '5 minutes'
                        ORDER BY timestamp DESC
                    """, self.user_id, self.conversation_id, action_type, target)
                    
                    actions = [json.loads(row["action_data"]) for row in rows]
                    return actions
        except Exception as e:
            logger.error(f"Error getting recent similar actions: {e}")
            return []
    
    async def _track_action_request(
        self,
        npc_id: int,
        action_type: str,
        action_details: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> int:
        """Track an action request for monitoring purposes."""
        try:
            tracking_data = {
                "npc_id": npc_id,
                "action_type": action_type,
                "action_details": action_details,
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "status": "requested"
            }
            
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        INSERT INTO NyxActionTracking (
                            user_id, conversation_id, npc_id, 
                            action_type, action_data, status, timestamp
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, NOW())
                        RETURNING id
                    """, 
                        self.user_id, 
                        self.conversation_id, 
                        npc_id,
                        action_type,
                        json.dumps(action_details),
                        "requested"
                    )
                    
                    return row["id"]
        except Exception as e:
            logger.error(f"Error tracking action request: {e}")
            return -1
    
    async def _track_completed_action(
        self,
        npc_id: int,
        action: Dict[str, Any],
        result: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> int:
        """Track a completed action for monitoring purposes."""
        try:
            tracking_data = {
                "npc_id": npc_id,
                "action": action,
                "result": result,
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        INSERT INTO NyxActionTracking (
                            user_id, conversation_id, npc_id, 
                            action_type, action_data, result_data, status, timestamp
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                        RETURNING id
                    """, 
                        self.user_id, 
                        self.conversation_id, 
                        npc_id,
                        action.get("type", "unknown"),
                        json.dumps(action),
                        json.dumps(result),
                        "completed"
                    )
                    
                    return row["id"]
        except Exception as e:
            logger.error(f"Error tracking completed action: {e}")
            return -1
    
    async def _log_directive(
        self,
        npc_id: int,
        directive_id: int,
        directive_type: str,
        directive_data: Dict[str, Any]
    ):
        """Log a directive for monitoring purposes."""
        memory_system = await self.get_memory_system()
        
        npc_name = await self._get_npc_name(npc_id)
        
        # Create a memory of issuing this directive
        await memory_system.add_memory(
            memory_text=f"I issued a {directive_type} directive to {npc_name}",
            memory_type="observation",
            memory_scope="game",
            significance=6,
            tags=["directive", directive_type, f"npc_{npc_id}"],
            metadata={
                "directive_id": directive_id,
                "directive_type": directive_type,
                "directive_data": directive_data,
                "npc_id": npc_id
            }
        )
    
    async def _get_npc_name(self, npc_id: int) -> str:
        """Get the name of an NPC."""
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT npc_name FROM NPCStats
                        WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                    """, npc_id, self.user_id, self.conversation_id)
                    
                    if row:
                        return row["npc_name"]
                    return f"NPC {npc_id}"
        except Exception as e:
            logger.error(f"Error getting NPC name: {e}")
            return f"NPC {npc_id}"
    
    async def _get_current_narrative_context(self) -> Dict[str, Any]:
        """Get the current narrative context from the database."""
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT value FROM CurrentRoleplay 
                        WHERE user_id = $1 AND conversation_id = $2 AND key = 'NyxNarrativeArcs'
                    """, self.user_id, self.conversation_id)
                    
                    if row and row["value"]:
                        return json.loads(row["value"])
            return {}
        except Exception as e:
            logger.error(f"Error getting narrative context: {e}")
            return {}

    async def approve_group_interaction(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Approve or modify a requested group interaction.
        
        Args:
            request: Dictionary with npc_ids, context, and other details
            
        Returns:
            Dictionary with approval status and any modifications
        """
        # Default to approved
        result = {
            "approved": True,
            "reason": "Approved by Nyx"
        }
        
        try:
            # Get narrative context
            narrative_data = await self._get_current_narrative_context()
            
            # Check if interaction aligns with narrative goals
            if narrative_data.get("active_arcs"):
                current_arc = narrative_data["active_arcs"][0]
                
                # Extract needed NPCs for current arc
                arc_npcs = current_arc.get("required_npcs", [])
                requested_npcs = request.get("npc_ids", [])
                
                # If this interaction doesn't include NPCs needed for the current arc
                # and those NPCs are supposed to be in the current location
                # Consider rejecting or modifying
                if arc_npcs and not any(npc_id in requested_npcs for npc_id in arc_npcs):
                    current_location = request.get("context", {}).get("location")
                    
                    # Get where the required NPCs should be
                    required_npc_locations = await self._get_npc_locations(arc_npcs)
                    
                    # If required NPCs should be here but aren't included
                    if any(required_npc_locations.get(npc_id) == current_location for npc_id in arc_npcs):
                        # Modify instead of reject - add the required NPCs
                        modified_npcs = requested_npcs.copy()
                        for npc_id in arc_npcs:
                            if required_npc_locations.get(npc_id) == current_location and npc_id not in modified_npcs:
                                modified_npcs.append(npc_id)
                        
                        # Get NPC names for guidance
                        npc_names = await self._get_npc_names(arc_npcs)
                        
                        # Return modified context
                        modified_context = request.get("context", {}).copy()
                        modified_context["modified_by_nyx"] = True
                        modified_context["nyx_guidance"] = f"Ensure {npc_names} are involved to advance the current narrative arc."
                        
                        result["approved"] = True
                        result["modified_context"] = modified_context
                        result["modified_npc_ids"] = modified_npcs
                        result["reason"] = "Modified by Nyx to include required NPCs for narrative progression"
            
            # Add Nyx's current guidance
            result["nyx_guidance"] = await self._generate_interaction_guidance(
                request.get("npc_ids", []), 
                request.get("context", {})
            )
            
            return result
        except Exception as e:
            logger.error(f"Error in Nyx approval: {e}")
            return result  # Default to approved
    
    async def _get_npc_locations(self, npc_ids: List[int]) -> Dict[int, str]:
        """Get current locations for a list of NPCs."""
        locations = {}
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    for npc_id in npc_ids:
                        row = await conn.fetchrow("""
                            SELECT current_location FROM NPCStats
                            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                        """, self.user_id, self.conversation_id, npc_id)
                        
                        if row:
                            locations[npc_id] = row["current_location"]
            return locations
        except Exception as e:
            logger.error(f"Error getting NPC locations: {e}")
            return locations
    
    async def _get_npc_names(self, npc_ids: List[int]) -> str:
        """Get names of NPCs as a formatted string."""
        names = []
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    for npc_id in npc_ids:
                        row = await conn.fetchrow("""
                            SELECT npc_name FROM NPCStats
                            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                        """, self.user_id, self.conversation_id, npc_id)
                        
                        if row:
                            names.append(row["npc_name"])
            return ", ".join(names)
        except Exception as e:
            logger.error(f"Error getting NPC names: {e}")
            return ", ".join([f"NPC_{npc_id}" for npc_id in npc_ids])
    
    async def _generate_interaction_guidance(
        self,
        npc_ids: List[int],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate guidance for a group interaction."""
        # A more sophisticated implementation would use GPT here
        # This is a simplified version
        
        # Basic guidance structure
        guidance = {
            "primary_focus": "natural_interaction",
            "npc_specific_guidance": {},
            "relationship_emphasis": []
        }
        
        # Get narrative context
        narrative_data = await self._get_current_narrative_context()
        current_arc = None
        if narrative_data.get("active_arcs"):
            current_arc = narrative_data["active_arcs"][0]
        
        # If we have an active arc, use it for guidance
        if current_arc:
            guidance["primary_focus"] = current_arc.get("name", "natural_interaction")
            
            # Check each NPC's role in the arc
            for npc_id in npc_ids:
                for npc_role in current_arc.get("npc_roles", []):
                    if npc_role.get("npc_id") == npc_id:
                        guidance["npc_specific_guidance"][npc_id] = {
                            "role": npc_role.get("role", "supporting"),
                            "behavior": npc_role.get("behavior", "neutral"),
                            "relationship": npc_role.get("relationship", "neutral")
                        }
                        
                        # Note important relationships
                        if "target_relationship" in npc_role:
                            guidance["relationship_emphasis"].append({
                                "npc_id": npc_id,
                                "target_type": npc_role.get("target_type", "player"),
                                "target_id": npc_role.get("target_id", self.user_id),
                                "relationship": npc_role.get("target_relationship")
                            })
        
        return guidance
