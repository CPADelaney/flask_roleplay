# nyx/nyx_governance.py

"""
Unified governance system for Nyx to control all agents (NPCs and beyond).

This file merges:
- The "Ultimate governance system" (previously in nyx/ultimate_governor.py), which controlled NPCs, StoryDirector, ConflictAnalyst, etc.
- The "Central governance system for Nyx to control all NPC agents" (previously in nyx/nyx_governance.py), which focused on NPC management.

Now combined into a single class `NyxUnifiedGovernor`, ensuring:
1. Central authority over all agents (NPCs, story, specialized).
2. Permission checking for all actions (NPC or other).
3. Directive management for all agent types.
4. Action reporting, monitoring, and override capabilities.
5. NPC-specific and general-agent logic all in one place.
"""

import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

import asyncpg

# The agent trace utility (from the original references)
from agents import trace

# Database connection helper
from db.connection import get_db_connection

# Memory system references
from memory.wrapper import MemorySystem
from nyx.nyx_memory_system import NyxMemorySystem

# Caching utilities
from utils.caching import CACHE_TTL, NPC_DIRECTIVE_CACHE, AGENT_DIRECTIVE_CACHE

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# From nyx_governance.py (central NPC governance) -- now consolidated here
# -------------------------------------------------------------------------
class DirectiveType:
    """
    Constants for directive types.

    Originally defined in nyx_governance.py (for NPCs) and referenced
    in ultimate_governor.py. Retained as the single definition for all agents.
    """
    ACTION = "action"
    MOVEMENT = "movement"
    DIALOGUE = "dialogue"
    RELATIONSHIP = "relationship"
    EMOTION = "emotion"
    PROHIBITION = "prohibition"
    SCENE = "scene"
    OVERRIDE = "override"


class DirectivePriority:
    """
    Constants for directive priorities.

    Used in both NPC-only logic and general agent logic.
    """
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    CRITICAL = 10


# -------------------------------------------------------------------------
# From ultimate_governor.py: class AgentType
# -------------------------------------------------------------------------
class AgentType:
    """Constants for agent types."""
    NPC = "npc"
    STORY_DIRECTOR = "story_director"
    CONFLICT_ANALYST = "conflict_analyst"
    NARRATIVE_CRAFTER = "narrative_crafter"
    RESOURCE_OPTIMIZER = "resource_optimizer"
    RELATIONSHIP_MANAGER = "relationship_manager"
    ACTIVITY_ANALYZER = "activity_analyzer"
    SCENE_MANAGER = "scene_manager"
    UNIVERSAL_UPDATER = "universal_updater"
    MEMORY_MANAGER = "memory_manager"  # Add this new agent type

# -------------------------------------------------------------------------
# Unified Class: NyxUnifiedGovernor
# Combines the entire logic from both modules into one.
# -------------------------------------------------------------------------
class NyxUnifiedGovernor:
    """
    Unified governance system for Nyx to control all agents (NPC and non-NPC).

    This class merges:
    - The functionalities of the original NyxGovernor (NPC-focused).
    - The functionalities of the original NyxUltimateGovernor (all agent types).

    It provides:
      1. Central authority over all agents (NPCs, StoryDirector, etc.).
      2. Permission checking for all agent actions.
      3. Directive management across all systems.
      4. Action reporting and monitoring.
      5. Global override capabilities.
      6. Inter-agent communication management.
      7. NPC-specific intervention logic merged with general agent logic.
      8. Database table setup for directives and action tracking.
      9. Access to memory system for logging, reflection, and context.
    """

    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the unified governance system.

        Args:
            user_id: The user/player ID
            conversation_id: The current conversation/scene ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id

        # Memory system will be lazily loaded
        self.memory_system: Optional[NyxMemorySystem] = None

        # Directive caches (one for NPC, one for non-NPC)
        self.npc_directive_cache = {}
        self.agent_directive_cache = {}

        # Track action history
        self.action_history = {}

        # Lock for concurrency
        self.lock = asyncio.Lock()

        # Track agent instances for direct communication (non-NPC types, if any)
        self.registered_agents = {}

        # Sub-managers placeholders (carried over from ultimate_governor)
        self._story_manager = None
        self._resource_manager = None
        self._relationship_manager = None

    # ---------------------------------------------------------------------
    # Shared memory system logic
    # ---------------------------------------------------------------------
    async def get_memory_system(self) -> NyxMemorySystem:
        """
        Lazy-load the memory system.

        Copied from both modules (functionally identical).
        """
        if self.memory_system is None:
            self.memory_system = NyxMemorySystem(self.user_id, self.conversation_id)
        return self.memory_system

    # ---------------------------------------------------------------------
    # Agent registration logic (from ultimate_governor)
    # ---------------------------------------------------------------------
    async def register_agent(self, agent_type: str, agent_instance: Any) -> None:
        """
        Register an agent with the governance system.

        Args:
            agent_type: The type of agent (use AgentType constants)
            agent_instance: The agent instance
        """
        self.registered_agents[agent_type] = agent_instance
        logger.info(f"Registered agent of type {agent_type}")

    # ---------------------------------------------------------------------
    # PERMISSION CHECK
    # Combine: `check_action_permission` for NPCs (from NyxGovernor)
    # and non-NPC (from NyxUltimateGovernor).
    # ---------------------------------------------------------------------
    async def check_action_permission(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        action_type: str,
        action_details: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Check if an agent (NPC or otherwise) is allowed to perform an action.

        - If it's an NPC (AgentType.NPC), apply NPC logic (originally in NyxGovernor).
        - Otherwise, apply the logic from NyxUltimateGovernor.

        Returns:
            Dict with fields like:
              - approved (bool)
              - directive_applied (bool)
              - override_action (optional dict)
              - reasoning (str)
              - tracking_id (int)
        """

        with trace(workflow_name=f"Agent {agent_type} Permission Check"):
            if agent_type == AgentType.NPC:
                # NPC logic from original NyxGovernor.check_action_permission
                directives = await self.get_npc_directives(int(agent_id))

                response = {
                    "approved": True,
                    "directive_applied": False,
                    "override_action": None,
                    "reasoning": "No applicable directives found",
                    "tracking_id": await self._track_action_request_npc(
                        npc_id=int(agent_id),
                        action_type=action_type,
                        action_details=action_details,
                        context=context
                    )
                }

                # Check for prohibitions first
                for directive in directives:
                    if directive.get("type") == DirectiveType.PROHIBITION:
                        prohibited_actions = directive.get("prohibited_actions", [])
                        if action_type in prohibited_actions or "*" in prohibited_actions:
                            response.update({
                                "approved": False,
                                "directive_applied": True,
                                "directive_id": directive.get("id"),
                                "reasoning": directive.get(
                                    "reason",
                                    "Action prohibited by Nyx directive"
                                )
                            })
                            # If there's an alternative action
                            if "alternative_action" in directive:
                                response["override_action"] = directive["alternative_action"]
                            return response

                # Check for override directives
                for directive in directives:
                    if directive.get("type") == DirectiveType.OVERRIDE:
                        applies_to = directive.get("applies_to", [])
                        if action_type in applies_to or "*" in applies_to:
                            response.update({
                                "approved": False,
                                "directive_applied": True,
                                "directive_id": directive.get("id"),
                                "override_action": directive.get("override_action"),
                                "reasoning": directive.get("reason", "Action overridden by Nyx directive")
                            })
                            return response

                # Check for action-specific directives
                for directive in directives:
                    if directive.get("type") == action_type:
                        response.update({
                            "approved": True,
                            "directive_applied": True,
                            "directive_id": directive.get("id"),
                            "action_modifications": directive.get("modifications", {}),
                            "reasoning": directive.get("reason", "Action modified by Nyx directive")
                        })
                        return response

                # No specific directives
                return response

            else:
                # Non-NPC logic from original NyxUltimateGovernor.check_action_permission
                # Get agent directives
                directives = await self.get_agent_directives(agent_type, agent_id)

                response = {
                    "approved": True,
                    "directive_applied": False,
                    "override_action": None,
                    "reasoning": f"No applicable directives found for {agent_type}",
                    "tracking_id": await self._track_action_request_general(
                        agent_type=agent_type,
                        agent_id=agent_id,
                        action_type=action_type,
                        action_details=action_details,
                        context=context
                    )
                }

                # Check for prohibitions first
                for directive in directives:
                    if directive.get("type") == DirectiveType.PROHIBITION:
                        prohibited_actions = directive.get("prohibited_actions", [])
                        if action_type in prohibited_actions or "*" in prohibited_actions:
                            response.update({
                                "approved": False,
                                "directive_applied": True,
                                "directive_id": directive.get("id"),
                                "reasoning": directive.get(
                                    "reason",
                                    f"Action prohibited by Nyx directive for {agent_type}"
                                )
                            })
                            # Check alternative
                            if "alternative_action" in directive:
                                response["override_action"] = directive["alternative_action"]
                            return response

                # Check for override directives
                for directive in directives:
                    if directive.get("type") == DirectiveType.OVERRIDE:
                        applies_to = directive.get("applies_to", [])
                        if action_type in applies_to or "*" in applies_to:
                            response.update({
                                "approved": False,
                                "directive_applied": True,
                                "directive_id": directive.get("id"),
                                "override_action": directive.get("override_action"),
                                "reasoning": directive.get(
                                    "reason",
                                    f"Action overridden by Nyx directive for {agent_type}"
                                )
                            })
                            return response

                # Check for action-specific directives
                for directive in directives:
                    if directive.get("type") == action_type:
                        response.update({
                            "approved": True,
                            "directive_applied": True,
                            "directive_id": directive.get("id"),
                            "action_modifications": directive.get("modifications", {}),
                            "reasoning": directive.get(
                                "reason",
                                f"Action modified by Nyx directive for {agent_type}"
                            )
                        })
                        return response

                # No specific directives
                return response

    # ---------------------------------------------------------------------
    # GET DIRECTIVES
    # Combine: `get_npc_directives` (NPC) with `get_agent_directives` (others).
    # ---------------------------------------------------------------------
    async def get_npc_directives(self, npc_id: int) -> List[Dict[str, Any]]:
        """
        Get all active directives for an NPC.

        Originally from NyxGovernor (nyx_governance.py).
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

    async def get_agent_directives(
        self,
        agent_type: str,
        agent_id: Union[int, str]
    ) -> List[Dict[str, Any]]:
        """
        Get all active directives for a non-NPC agent.

        Originally from NyxUltimateGovernor (ultimate_governor.py).
        """
        # Check cache
        cache_key = f"directives:{self.user_id}:{self.conversation_id}:{agent_type}:{agent_id}"
        cached = AGENT_DIRECTIVE_CACHE.get(cache_key)
        if cached:
            return cached

        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT id, directive, priority, expires_at
                        FROM NyxAgentDirectives
                        WHERE user_id = $1
                          AND conversation_id = $2
                          AND agent_type = $3
                          AND agent_id = $4
                          AND expires_at > NOW()
                        ORDER BY priority DESC
                    """, self.user_id, self.conversation_id, agent_type, str(agent_id))

                    directives = []
                    for row in rows:
                        directive_data = json.loads(row["directive"])
                        directive_data["id"] = row["id"]
                        directive_data["priority"] = row["priority"]
                        directive_data["expires_at"] = row["expires_at"].isoformat()
                        directives.append(directive_data)

                    AGENT_DIRECTIVE_CACHE.set(cache_key, directives, CACHE_TTL["directives"])
                    return directives

        except Exception as e:
            logger.error(f"Error fetching directives for agent {agent_type}/{agent_id}: {e}")
            return []

    # ---------------------------------------------------------------------
    # ISSUE DIRECTIVE
    # Combine: `issue_directive` for NPC (NyxGovernor) and non-NPC (NyxUltimateGovernor).
    # ---------------------------------------------------------------------
    async def issue_directive(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        directive_type: str,
        directive_data: Dict[str, Any],
        priority: int = DirectivePriority.MEDIUM,
        duration_minutes: int = 30,
        scene_id: str = None
    ) -> int:
        """
        Issue a new directive to any agent (NPC or non-NPC).

        Combines:
          - NyxGovernor.issue_directive for NPCs
          - NyxUltimateGovernor.issue_directive for others
        """
        if agent_type == AgentType.NPC:
            # NPC path (was nyx_governance.py)
            try:
                npc_id = int(agent_id)
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
                        await self._log_directive_npc(npc_id, directive_id, directive_type, directive_data)
                        return directive_id

            except Exception as e:
                logger.error(f"Error issuing directive to NPC {agent_id}: {e}")
                return -1

        else:
            # Non-NPC path (was ultimate_governor)
            try:
                directive = {
                    "type": directive_type,
                    "timestamp": datetime.now().isoformat(),
                    **directive_data
                }
                async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                    async with pool.acquire() as conn:
                        row = await conn.fetchrow("""
                            INSERT INTO NyxAgentDirectives (
                                user_id, conversation_id, agent_type, agent_id, directive,
                                expires_at, priority, scene_id
                            )
                            VALUES ($1, $2, $3, $4, $5, NOW() + $6::INTERVAL, $7, $8)
                            RETURNING id
                        """,
                        self.user_id,
                        self.conversation_id,
                        agent_type,
                        str(agent_id),
                        json.dumps(directive),
                        f"{duration_minutes} minutes",
                        priority,
                        scene_id
                        )
                        directive_id = row["id"]

                        # Invalidate cache
                        cache_key = f"directives:{self.user_id}:{self.conversation_id}:{agent_type}:{agent_id}"
                        AGENT_DIRECTIVE_CACHE.delete(cache_key)

                        # Log directive
                        await self._log_directive_general(
                            agent_type, agent_id, directive_id, directive_type, directive_data
                        )
                        return directive_id

            except Exception as e:
                logger.error(f"Error issuing directive to agent {agent_type}/{agent_id}: {e}")
                return -1

    # ---------------------------------------------------------------------
    # REVOKE DIRECTIVE
    # Combine: `revoke_directive` from NyxGovernor (NPC) and from Ultimate.
    # ---------------------------------------------------------------------
    async def revoke_directive(self, directive_id: int, agent_type: str = None) -> bool:
        """
        Revoke a directive immediately.

        Combines logic for:
          - NPC directives (NyxNPCDirectives)
          - Non-NPC directives (NyxAgentDirectives)

        Args:
            directive_id: ID of the directive
            agent_type: Optional agent type to speed up lookups
        """
        if agent_type == AgentType.NPC:
            # NPC path from nyx_governance
            return await self._revoke_directive_npc(directive_id)

        # If not sure or it's non-NPC, handle ultimate approach
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    # 1) If agent_type is None or unknown, we attempt to see if it's an NPC directive
                    if agent_type is None:
                        # Try NPC first
                        row_npc = await conn.fetchrow("""
                            SELECT npc_id FROM NyxNPCDirectives
                            WHERE id = $1 AND user_id = $2 AND conversation_id = $3
                        """, directive_id, self.user_id, self.conversation_id)

                        if row_npc:
                            # It's an NPC directive
                            npc_id = row_npc["npc_id"]
                            return await self._revoke_directive_npc(directive_id)

                        # Otherwise check agent directives
                        row_agent = await conn.fetchrow("""
                            SELECT agent_type, agent_id FROM NyxAgentDirectives
                            WHERE id = $1 AND user_id = $2 AND conversation_id = $3
                        """, directive_id, self.user_id, self.conversation_id)
                        if not row_agent:
                            return False

                        found_agent_type = row_agent["agent_type"]
                        found_agent_id = row_agent["agent_id"]

                        # Revoke in agent directives
                        result = await conn.execute("""
                            UPDATE NyxAgentDirectives
                            SET expires_at = NOW()
                            WHERE id = $1
                        """, directive_id)
                        # Invalidate cache
                        cache_key = f"directives:{self.user_id}:{self.conversation_id}:{found_agent_type}:{found_agent_id}"
                        AGENT_DIRECTIVE_CACHE.delete(cache_key)
                        return True

                    else:
                        # We have an agent_type that is presumably non-NPC
                        row_agent = await conn.fetchrow("""
                            SELECT agent_type, agent_id FROM NyxAgentDirectives
                            WHERE id = $1 AND user_id = $2 AND conversation_id = $3
                        """, directive_id, self.user_id, self.conversation_id)

                        if not row_agent:
                            return False

                        found_agent_type = row_agent["agent_type"]
                        found_agent_id = row_agent["agent_id"]

                        # Expire it
                        await conn.execute("""
                            UPDATE NyxAgentDirectives
                            SET expires_at = NOW()
                            WHERE id = $1
                        """, directive_id)

                        # Invalidate cache
                        cache_key = f"directives:{self.user_id}:{self.conversation_id}:{found_agent_type}:{found_agent_id}"
                        AGENT_DIRECTIVE_CACHE.delete(cache_key)
                        return True

        except Exception as e:
            logger.error(f"Error revoking directive {directive_id}: {e}")
            return False

    async def _revoke_directive_npc(self, directive_id: int) -> bool:
        """
        Internal helper to revoke an NPC directive (from original NyxGovernor).
        """
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT npc_id FROM NyxNPCDirectives
                        WHERE id = $1 AND user_id = $2 AND conversation_id = $3
                    """, directive_id, self.user_id, self.conversation_id)
                    if not row:
                        return False
                    npc_id = row["npc_id"]

                    # Expire
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

    # ---------------------------------------------------------------------
    # PROCESS ACTION REPORT
    # Combine: NPC version (NyxGovernor.process_npc_action_report)
    #          and general version (NyxUltimateGovernor.process_agent_action_report).
    # ---------------------------------------------------------------------
    async def process_agent_action_report(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        action: Dict[str, Any],
        result: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process an action report from any agent (NPC or otherwise)
        and determine if intervention is needed.

        Merged:
          - If NPC, originally used `process_npc_action_report` from NyxGovernor.
          - If non-NPC, used `process_agent_action_report` from UltimateGovernor.
        """

        async with self.lock:
            if agent_type == AgentType.NPC:
                # NPC path (from NyxGovernor.process_npc_action_report)
                tracking_id = await self._track_completed_action_npc(int(agent_id), action, result, context)
                memory_system = await self.get_memory_system()

                npc_name = await self._get_npc_name(int(agent_id))
                await memory_system.add_memory(
                    memory_text=f"{npc_name} performed: {action.get('description', 'unknown action')}",
                    memory_type="observation",
                    memory_scope="game",
                    significance=min(abs(result.get("emotional_impact", 0)) + 4, 10),
                    tags=["npc_action", f"npc_{agent_id}"],
                    metadata={
                        "npc_id": agent_id,
                        "action": action,
                        "result": result,
                        "tracking_id": tracking_id
                    }
                )

                # Check if intervention needed
                intervention_needed = await self._check_if_intervention_needed_npc(int(agent_id), action, result)
                if intervention_needed:
                    # Issue override directive
                    directive_id = await self.issue_directive(
                        agent_type=AgentType.NPC,
                        agent_id=agent_id,
                        directive_type=DirectiveType.OVERRIDE,
                        directive_data={
                            "reason": intervention_needed.get("reason"),
                            "override_action": intervention_needed.get("override_action"),
                            "applies_to": ["*"],
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

            else:
                # Non-NPC path (from ultimate_governor.process_agent_action_report)
                tracking_id = await self._track_completed_action_general(agent_type, agent_id, action, result, context)
                memory_system = await self.get_memory_system()
                agent_identifier = await self._get_agent_identifier(agent_type, agent_id)

                # Add memory
                await memory_system.add_memory(
                    memory_text=f"{agent_identifier} performed: {action.get('description', 'unknown action')}",
                    memory_type="observation",
                    memory_scope="game",
                    significance=6,
                    tags=["agent_action", f"{agent_type}_{agent_id}"],
                    metadata={
                        "agent_type": agent_type,
                        "agent_id": agent_id,
                        "action": action,
                        "result": result,
                        "tracking_id": tracking_id
                    }
                )

                # Decide if we intervene
                intervention_needed = await self._check_if_intervention_needed_general(agent_type, agent_id, action, result)
                if intervention_needed:
                    directive_id = await self.issue_directive(
                        agent_type=agent_type,
                        agent_id=agent_id,
                        directive_type=DirectiveType.OVERRIDE,
                        directive_data={
                            "reason": intervention_needed.get("reason"),
                            "override_action": intervention_needed.get("override_action"),
                            "applies_to": ["*"],
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

    # ---------------------------------------------------------------------
    # NPC-SPECIFIC INTERVENTION LOGIC (from NyxGovernor)
    # ---------------------------------------------------------------------
    async def _check_if_intervention_needed_npc(
        self,
        npc_id: int,
        action: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Determine if Nyx should intervene based on NPC action and its results.

        Original logic from nyx_governance.py: `_check_if_intervention_needed`
        """
        emotional_impact = result.get("emotional_impact", 0)
        action_type = action.get("type", "unknown")
        target = action.get("target", "unknown")
        description = action.get("description", "")

        # Narrative context check
        narrative_context = await self._get_current_narrative_context()
        current_arcs = narrative_context.get("active_arcs", [])

        # 1. Contradiction with narrative arcs
        for arc in current_arcs:
            for npc_role in arc.get("npc_roles", []):
                if npc_role.get("npc_id") == npc_id:
                    required_relationship = npc_role.get("relationship")
                    required_action = npc_role.get("required_action")
                    # Relationship contradiction
                    if required_relationship == "friendly" and action_type in ["mock", "attack", "threaten"]:
                        return {
                            "reason": "Action contradicts NPC's friendly role in current narrative arc",
                            "override_action": {
                                "type": "talk",
                                "description": "speak in a more friendly manner",
                                "target": target
                            }
                        }
                    # Required action contradiction
                    if required_action == "help" and action_type == "leave":
                        return {
                            "reason": "NPC is required to help in current narrative arc",
                            "override_action": {
                                "type": "assist",
                                "description": "offer assistance instead of leaving",
                                "target": target
                            }
                        }

        # 2. Excessive emotional impact
        if abs(emotional_impact) > 7:
            return {
                "reason": "Action has excessive emotional impact",
                "override_action": {
                    "type": action_type,
                    "description": description.replace("forcefully", "moderately").replace("aggressively", "assertively"),
                    "target": target
                }
            }

        # 3. Coordinated group actions
        if target in ["group", "player"] and action_type in ["attack", "threaten", "dominate"]:
            # Check recent similar actions
            recent_actions = await self._get_recent_similar_actions_npc(action_type, target)
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

    async def _get_recent_similar_actions_npc(self, action_type: str, target: str) -> List[Dict[str, Any]]:
        """
        Get recent similar actions by other NPCs (from nyx_governance.py).
        """
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

    # Add to the _check_memory_manager_intervention method in NyxUnifiedGovernor class
    async def _check_memory_manager_intervention(self, agent_id, action, result):
        """
        Check if intervention is needed for memory manager actions.
        
        Args:
            agent_id: Agent ID
            action: Action performed
            result: Result of the action
        
        Returns:
            Intervention data or None
        """
        # Check for specific scenarios requiring intervention
        
        # 1. If creating potentially harmful memories
        if action.get("operation") == "remember" and "error" in result:
            return {
                "reason": f"Memory creation error: {result.get('error')}",
                "override_action": {
                    "type": "recover",
                    "description": "Recover from memory creation error",
                    "parameters": {"retry": True}
                }
            }
        
        # 2. If retrieving excessive memories
        if action.get("operation") == "recall" and len(result.get("memories", [])) > 20:
            return {
                "reason": "Excessive memory retrieval may impact performance",
                "override_action": {
                    "type": "optimize",
                    "description": "Limit memory retrieval to improve performance",
                    "parameters": {"max_memories": 20}
                }
            }
        
        # 3. If creating beliefs with extreme confidence
        if action.get("operation") == "create_belief" and action.get("confidence", 0) > 0.95:
            return {
                "reason": "Creating beliefs with excessive confidence",
                "override_action": {
                    "type": "moderate",
                    "description": "Reduce belief confidence to more reasonable level",
                    "parameters": {"max_confidence": 0.9}
                }
            }
        
        # No intervention needed
        return None

    async def _check_narrative_crafter_intervention(self, agent_id, action, result):
        """Check if intervention is needed for narrative crafter actions."""
        action_type = action.get("type", "unknown")
        
        # Check for lore inconsistencies
        if action_type == "generate_lore" and result.get("inconsistencies", []):
            return {
                "reason": "Inconsistencies detected in generated lore",
                "override_action": {
                    "type": "fix_inconsistencies",
                    "description": "Resolve detected lore inconsistencies",
                    "parameters": {"inconsistencies": result.get("inconsistencies", [])}
                }
            }
        
        # Check for unreasonable lore complexity
        if action_type == "generate_lore" and result.get("complexity_score", 0) > 8:
            return {
                "reason": "Generated lore is too complex for current context",
                "override_action": {
                    "type": "simplify_lore",
                    "description": "Simplify lore to more manageable complexity",
                    "parameters": {"max_complexity": 7}
                }
            }
        
        # Prevent overwhelming NPC knowledge
        if action_type == "integrate_lore_with_npcs" and result.get("average_knowledge_per_npc", 0) > 12:
            return {
                "reason": "Too much knowledge being assigned to NPCs",
                "override_action": {
                    "type": "limit_npc_knowledge",
                    "description": "Limit the amount of lore knowledge given to NPCs",
                    "parameters": {"max_knowledge_per_npc": 10}
                }
            }
        
        return None

    
    # ---------------------------------------------------------------------
    # GENERAL-AGENT INTERVENTION LOGIC (from ultimate_governor)
    # ---------------------------------------------------------------------
    async def _check_if_intervention_needed_general(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        action: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Determine if Nyx should intervene for non-NPC agents
        (story_director, conflict_analyst, etc.).

        Derived from `_check_if_intervention_needed` in ultimate_governor.py.
        """

        # For specialized agent types, check the relevant sub-method:
        intervention_rules = {
            AgentType.STORY_DIRECTOR: self._check_story_director_intervention,
            AgentType.CONFLICT_ANALYST: self._check_conflict_analyst_intervention,
            AgentType.NARRATIVE_CRAFTER: self._check_narrative_crafter_intervention,
            AgentType.RESOURCE_OPTIMIZER: self._check_resource_optimizer_intervention,
            AgentType.RELATIONSHIP_MANAGER: self._check_relationship_manager_intervention,
            AgentType.ACTIVITY_ANALYZER: self._check_activity_analyzer_intervention,
            AgentType.SCENE_MANAGER: self._check_scene_manager_intervention,
            AgentType.UNIVERSAL_UPDATER: self._check_universal_updater_intervention,
            AgentType.MEMORY_MANAGER: self._check_memory_manager_intervention
        }

        if agent_type in intervention_rules:
            return await intervention_rules[agent_type](agent_id, action, result)

        # Otherwise, generic logic
        return await self._check_generic_agent_intervention(agent_type, agent_id, action, result)

    # Sub-check methods (from ultimate_governor):
    async def _check_story_director_intervention(self, agent_id, action, result):
        action_type = action.get("type", "unknown")
        # Prevent excessively rapid narrative progression
        if action_type == "advance_narrative" and result.get("progression_rate", 0) > 0.3:
            return {
                "reason": "Narrative is advancing too quickly",
                "override_action": {
                    "type": "slow_progression",
                    "description": "Reduce the rate of narrative advancement",
                    "parameters": {"max_progression_rate": 0.2}
                }
            }
        # Prevent too many conflicts
        if action_type == "generate_conflict" and result.get("active_conflicts", 0) > 3:
            return {
                "reason": "Too many active conflicts would overwhelm the player",
                "override_action": {
                    "type": "delay_conflict",
                    "description": "Delay conflict generation until existing conflicts are resolved",
                    "parameters": {"max_active_conflicts": 3}
                }
            }
        return None

    async def _check_conflict_analyst_intervention(self, agent_id, action, result):
        return None

    async def _check_narrative_crafter_intervention(self, agent_id, action, result):
        return None

    async def _check_resource_optimizer_intervention(self, agent_id, action, result):
        # Prevent drastic resource changes
        if action.get("type") == "adjust_resources" and abs(result.get("money_change", 0)) > 500:
            return {
                "reason": "Resource adjustment is too extreme",
                "override_action": {
                    "type": "moderate_resource_change",
                    "description": "Apply a more moderate resource adjustment",
                    "parameters": {"max_change": 500}
                }
            }
        return None

    async def _check_relationship_manager_intervention(self, agent_id, action, result):
        return None

    async def _check_activity_analyzer_intervention(self, agent_id, action, result):
        return None

    async def _check_scene_manager_intervention(self, agent_id, action, result):
        return None

    async def _check_universal_updater_intervention(self, agent_id, action, result):
        return None

    async def _check_generic_agent_intervention(self, agent_type, agent_id, action, result):
        """
        Generic intervention logic for any agent type.
        """
        # Check for critical errors
        if result.get("error") and result.get("critical", False):
            return {
                "reason": f"Critical error in {agent_type} action: {result.get('error')}",
                "override_action": {
                    "type": "recover",
                    "description": "Attempt recovery from error state",
                    "parameters": {"reset_state": True}
                }
            }

        # Check for severe consequences
        if result.get("severity", 0) > 8:
            return {
                "reason": "Action has potentially severe consequences",
                "override_action": {
                    "type": "moderate",
                    "description": "Reduce severity of action outcome",
                    "parameters": {"max_severity": 7}
                }
            }
        return None

    # ---------------------------------------------------------------------
    # TRACKING ACTION REQUESTS/COMPLETION
    # We keep separate internal methods for NPC vs. non-NPC
    # since the original code used different inserts (npc_id vs agent_type).
    # ---------------------------------------------------------------------
    async def _track_action_request_npc(
        self,
        npc_id: int,
        action_type: str,
        action_details: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> int:
        """
        Track an action request for an NPC (original from NyxGovernor).
        """
        try:
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

    async def _track_completed_action_npc(
        self,
        npc_id: int,
        action: Dict[str, Any],
        result: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> int:
        """
        Track a completed NPC action (original from NyxGovernor).
        """
        try:
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

    async def _track_action_request_general(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        action_type: str,
        action_details: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> int:
        """
        Track an action request for non-NPC agents.
        (from ultimate_governor._track_action_request)
        """
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        INSERT INTO NyxActionTracking (
                            user_id, conversation_id, agent_type, agent_id,
                            action_type, action_data, status, timestamp
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                        RETURNING id
                    """,
                    self.user_id,
                    self.conversation_id,
                    agent_type,
                    str(agent_id),
                    action_type,
                    json.dumps(action_details),
                    "requested"
                    )
                    return row["id"]
        except Exception as e:
            logger.error(f"Error tracking action request: {e}")
            return -1

    async def _track_completed_action_general(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        action: Dict[str, Any],
        result: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> int:
        """
        Track a completed action for non-NPC agents.
        (from ultimate_governor._track_completed_action)
        """
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        INSERT INTO NyxActionTracking (
                            user_id, conversation_id, agent_type, agent_id,
                            action_type, action_data, result_data, status, timestamp
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                        RETURNING id
                    """,
                    self.user_id,
                    self.conversation_id,
                    agent_type,
                    str(agent_id),
                    action.get("type", "unknown"),
                    json.dumps(action),
                    json.dumps(result),
                    "completed"
                    )
                    return row["id"]
        except Exception as e:
            logger.error(f"Error tracking completed action: {e}")
            return -1

    # ---------------------------------------------------------------------
    # LOGGING DIRECTIVES
    # Keep separate methods for NPC vs. general, to preserve docstrings + calls.
    # ---------------------------------------------------------------------
    async def _log_directive_npc(
        self,
        npc_id: int,
        directive_id: int,
        directive_type: str,
        directive_data: Dict[str, Any]
    ):
        """
        NPC version (original from NyxGovernor._log_directive).
        """
        memory_system = await self.get_memory_system()
        npc_name = await self._get_npc_name(npc_id)
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

    async def _log_directive_general(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        directive_id: int,
        directive_type: str,
        directive_data: Dict[str, Any]
    ):
        """
        General agent version (original from ultimate_governor._log_directive).
        """
        memory_system = await self.get_memory_system()
        agent_identifier = await self._get_agent_identifier(agent_type, agent_id)
        await memory_system.add_memory(
            memory_text=f"I issued a {directive_type} directive to {agent_identifier}",
            memory_type="observation",
            memory_scope="game",
            significance=6,
            tags=["directive", directive_type, f"{agent_type}_{agent_id}"],
            metadata={
                "directive_id": directive_id,
                "directive_type": directive_type,
                "directive_data": directive_data,
                "agent_type": agent_type,
                "agent_id": agent_id
            }
        )

    # ---------------------------------------------------------------------
    # GETTING NPC NAME & NARRATIVE CONTEXT
    # (From NyxGovernor, with no changes)
    # ---------------------------------------------------------------------
    async def _get_npc_name(self, npc_id: int) -> str:
        """
        Get the name of an NPC.
        """
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
        """
        Get the current narrative context from the database.
        (From NyxGovernor).
        """
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

    # ---------------------------------------------------------------------
    # GETTING "AGENT IDENTIFIER" (from ultimate_governor)
    # For non-NPC agents, we produce a label. For NPC, we call `_get_npc_name`.
    # ---------------------------------------------------------------------
    async def _get_agent_identifier(self, agent_type: str, agent_id: Union[int, str]) -> str:
        """
        Get a readable identifier for any agent.

        If NPC, we use `_get_npc_name()`.
        If not NPC, format agent_type + agent_id.
        """
        if agent_type == AgentType.NPC:
            return await self._get_npc_name(int(agent_id))
        # else
        agent_type_formatted = agent_type.replace("_", " ").title()
        return f"{agent_type_formatted} Agent {agent_id}"

    # ---------------------------------------------------------------------
    # COORDINATE AGENTS (from ultimate_governor)
    # ---------------------------------------------------------------------
    async def coordinate_agents(
        self,
        action_type: str,
        primary_agent_type: str,
        action_details: Dict[str, Any],
        supporting_agents: List[str] = None
    ) -> Dict[str, Any]:
        """
        Coordinate multiple agents for a complex action.

        Originally from NyxUltimateGovernor. Now in the unified class.
        """
        supporting_agents = supporting_agents or []

        # First, check permission with primary agent
        primary_permission = await self.check_action_permission(
            primary_agent_type, "primary", action_type, action_details
        )
        if not primary_permission["approved"]:
            return {
                "success": False,
                "reason": primary_permission["reasoning"],
                "action_type": action_type
            }

        # Track the coordinated action
        tracking_id = await self._track_action_request_general(
            agent_type=primary_agent_type,
            agent_id="coordinated",
            action_type=action_type,
            action_details={**action_details, "supporting_agents": supporting_agents}
        )
        result = {
            "success": True,
            "action_type": action_type,
            "primary_agent": primary_agent_type,
            "supporting_agents": supporting_agents,
            "tracking_id": tracking_id,
            "agent_results": {}
        }

        # Execute the primary agent's action
        try:
            if primary_agent_type in self.registered_agents:
                primary_agent = self.registered_agents[primary_agent_type]
                primary_result = await self._execute_agent_action(
                    primary_agent, action_type, action_details
                )
            else:
                # Simulate
                primary_result = {
                    "status": "simulated",
                    "action_type": action_type,
                    "message": f"Simulated {primary_agent_type} response"
                }

            result["agent_results"][primary_agent_type] = primary_result

        except Exception as e:
            logger.error(f"Error executing primary agent action: {e}")
            result["agent_results"][primary_agent_type] = {
                "error": str(e),
                "action_type": action_type
            }
            result["success"] = False

        # Execute supporting agents
        for ag_type in supporting_agents:
            try:
                support_permission = await self.check_action_permission(
                    ag_type, "support", action_type, action_details
                )
                if not support_permission["approved"]:
                    result["agent_results"][ag_type] = {
                        "skipped": True,
                        "reason": support_permission["reasoning"]
                    }
                    continue

                if ag_type in self.registered_agents:
                    agent_inst = self.registered_agents[ag_type]
                    agent_result = await self._execute_agent_action(
                        agent_inst, action_type, action_details
                    )
                else:
                    agent_result = {
                        "status": "simulated",
                        "action_type": action_type,
                        "message": f"Simulated {ag_type} supporting response"
                    }
                result["agent_results"][ag_type] = agent_result

            except Exception as e:
                logger.error(f"Error executing supporting agent action: {e}")
                result["agent_results"][ag_type] = {
                    "error": str(e),
                    "action_type": action_type
                }

        # Track completed
        await self._track_completed_action_general(
            agent_type=primary_agent_type,
            agent_id="coordinated",
            action={**action_details, "supporting_agents": supporting_agents},
            result=result
        )
        return result

    async def _execute_agent_action(self, agent, action_type, action_details):
        """
        Placeholder for agent method calls. (from ultimate_governor)
        """
        return {
            "status": "executed",
            "action_type": action_type,
            "details": action_details
        }

    # ---------------------------------------------------------------------
    # BROADCAST TO ALL AGENTS (from ultimate_governor)
    # ---------------------------------------------------------------------
    async def broadcast_to_all_agents(self, message_type: str, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Broadcast a message to all registered agents. (non-NPC logic)

        NPCs typically aren't "registered" in the same sense, but if you store them
        in self.registered_agents, they'd get it too. This is from ultimate_governor.
        """
        results = {
            "message_type": message_type,
            "broadcast_time": datetime.now().isoformat(),
            "recipients": len(self.registered_agents),
            "responses": {}
        }

        # Log in memory
        memory_system = await self.get_memory_system()
        await memory_system.add_memory(
            memory_text=f"Broadcast message of type '{message_type}' to all agents",
            memory_type="system",
            memory_scope="game",
            significance=4,
            tags=["broadcast", message_type],
            metadata={"message_data": message_data}
        )

        # Broadcast
        for ag_type, agent_inst in self.registered_agents.items():
            try:
                # Check if agent can process message
                if hasattr(agent_inst, "can_process_message") and not agent_inst.can_process_message(message_type):
                    results["responses"][ag_type] = {
                        "skipped": True,
                        "reason": "Agent does not process this message type"
                    }
                    continue

                # Check permission
                permission = await self.check_action_permission(
                    ag_type, "system", "receive_broadcast",
                    {"message_type": message_type}
                )
                if not permission["approved"]:
                    results["responses"][ag_type] = {
                        "skipped": True,
                        "reason": permission["reasoning"]
                    }
                    continue

                # Send
                if hasattr(agent_inst, "process_broadcast"):
                    response = await agent_inst.process_broadcast(message_type, message_data)
                    results["responses"][ag_type] = response
                else:
                    results["responses"][ag_type] = {
                        "status": "no_handler", "message_type": message_type
                    }

            except Exception as e:
                logger.error(f"Error broadcasting to {ag_type}: {e}")
                results["responses"][ag_type] = {"error": str(e), "message_type": message_type}

        return results

    # ---------------------------------------------------------------------
    # GET NARRATIVE STATUS (from ultimate_governor)
    # ---------------------------------------------------------------------
    async def get_narrative_status(self) -> Dict[str, Any]:
        """
        Get the current status of the narrative as a whole. (from ultimate_governor)
        """
        memory_system = await self.get_memory_system()
        recent_memories = await memory_system.get_recent_memories(limit=5)

        # Attempt to get current narrative stage
        try:
            from logic.narrative_progression import get_current_narrative_stage
            narrative_stage = await get_current_narrative_stage(self.user_id, self.conversation_id)
        except Exception as e:
            logger.error(f"Error retrieving narrative stage: {e}")
            narrative_stage = None

        # Active conflicts
        active_conflicts = []
        try:
            from logic.conflict_system.conflict_manager import ConflictManager
            conflict_manager = ConflictManager(self.user_id, self.conversation_id)
            active_conflicts = await conflict_manager.get_active_conflicts()
        except Exception as e:
            logger.error(f"Error getting active conflicts: {e}")

        # Key NPCs
        key_npcs = []
        try:
            from story_agent.tools import get_key_npcs
            class ContextMock:
                def __init__(self, user_id, conversation_id):
                    self.context = {"user_id": user_id, "conversation_id": conversation_id}
            ctx_mock = ContextMock(self.user_id, self.conversation_id)
            key_npcs = await get_key_npcs(ctx_mock, limit=5)
        except Exception as e:
            logger.error(f"Error getting key NPCs: {e}")

        # Resource status
        resources = {}
        try:
            from logic.resource_management import ResourceManager
            resource_manager = ResourceManager(self.user_id, self.conversation_id)
            resources = await resource_manager.get_resources()
            vitals = await resource_manager.get_vitals()
            resources.update(vitals)
        except Exception as e:
            logger.error(f"Error getting resources: {e}")

        # Build the status
        directive_count = 0
        # We'll reuse the NPC directive call; 'all' isn't real, but we mimic the old call:
        try:
            directives_all = await self.get_npc_directives(npc_id="all")  # not truly supported, but for example
            directive_count = len(directives_all)
        except:
            pass

        return {
            "narrative_stage": {
                "name": narrative_stage.name if narrative_stage else "Unknown",
                "description": narrative_stage.description if narrative_stage else ""
            },
            "recent_memories": recent_memories,
            "active_conflicts": active_conflicts,
            "key_npcs": key_npcs,
            "resources": resources,
            "directive_count": directive_count,
            "timestamp": datetime.now().isoformat()
        }

    # ---------------------------------------------------------------------
    # GET/CREATE AGENT MEMORY (from ultimate_governor)
    # ---------------------------------------------------------------------
    async def get_agent_memory(self, agent_type: str, agent_id: Union[int, str], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get memories related to a specific agent. (from ultimate_governor)
        """
        memory_system = await self.get_memory_system()
        memories = await memory_system.retrieve_memories(
            query="",
            memory_types=["observation", "reflection", "abstraction"],
            scopes=["game"],
            limit=limit,
            context={"tags": [f"{agent_type}_{agent_id}"]}
        )
        return memories

    async def create_agent_memory(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        memory_text: str,
        significance: int = 5,
        tags: List[str] = None
    ) -> int:
        """
        Create a memory about an agent. (from ultimate_governor)
        """
        memory_system = await self.get_memory_system()
        tags = tags or []
        agent_tag = f"{agent_type}_{agent_id}"
        if agent_tag not in tags:
            tags.append(agent_tag)

        memory_id = await memory_system.add_memory(
            memory_text=memory_text,
            memory_type="observation",
            memory_scope="game",
            significance=significance,
            tags=tags,
            metadata={
                "agent_type": agent_type,
                "agent_id": agent_id,
                "created_at": datetime.now().isoformat()
            }
        )
        return memory_id

    async def generate_agent_reflection(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        topic: str = None
    ) -> Dict[str, Any]:
        """
        Generate a reflection about an agent's actions and behavior.
        (from ultimate_governor)
        """
        memory_system = await self.get_memory_system()
        agent_tag = f"{agent_type}_{agent_id}"
        memories = await memory_system.retrieve_memories(
            query="",
            memory_types=["observation"],
            scopes=["game"],
            limit=20,
            context={"tags": [agent_tag]}
        )
        if not memories:
            return {
                "reflection": f"I don't have sufficient memories about {agent_type} {agent_id} to form a reflection.",
                "confidence": 0.1,
                "agent_type": agent_type,
                "agent_id": agent_id
            }

        memory_texts = [m["memory_text"] for m in memories]
        from nyx.llm_integration import generate_reflection
        reflection_text = await generate_reflection(
            memory_texts=memory_texts,
            topic=topic or f"Behavior and actions of {agent_type} {agent_id}",
            context={"agent_type": agent_type, "agent_id": agent_id}
        )
        reflection_memory_id = await memory_system.add_memory(
            memory_text=reflection_text,
            memory_type="reflection",
            memory_scope="game",
            significance=6,
            tags=[agent_tag, "reflection"],
            metadata={
                "agent_type": agent_type,
                "agent_id": agent_id,
                "topic": topic,
                "created_at": datetime.now().isoformat()
            }
        )
        return {
            "reflection": reflection_text,
            "confidence": 0.7,
            "agent_type": agent_type,
            "agent_id": agent_id,
            "memory_id": reflection_memory_id
        }

    # ---------------------------------------------------------------------
    # setup_database_tables (from ultimate_governor)
    # ---------------------------------------------------------------------
    async def setup_database_tables(self):
        """
        Set up necessary database tables for the governance system.
        (from ultimate_governor)
        """
        async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
            async with pool.acquire() as conn:
                # For non-NPC directives
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS NyxAgentDirectives (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        conversation_id INTEGER NOT NULL,
                        agent_type VARCHAR(50) NOT NULL,
                        agent_id VARCHAR(50) NOT NULL,
                        directive JSONB NOT NULL,
                        priority INTEGER DEFAULT 5,
                        expires_at TIMESTAMP WITH TIME ZONE,
                        scene_id VARCHAR(50),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

                        CONSTRAINT agent_directives_user_conversation_fk
                            FOREIGN KEY (user_id, conversation_id)
                            REFERENCES conversations(user_id, id)
                            ON DELETE CASCADE
                    )
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_agent_directives_agent
                    ON NyxAgentDirectives(user_id, conversation_id, agent_type, agent_id)
                """)

                # For action tracking (covers both NPC and non-NPC)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS NyxActionTracking (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        conversation_id INTEGER NOT NULL,
                        agent_type VARCHAR(50),
                        agent_id VARCHAR(50),
                        npc_id INTEGER,
                        action_type VARCHAR(50),
                        action_data JSONB,
                        result_data JSONB,
                        status VARCHAR(20),
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

                        CONSTRAINT action_tracking_user_conversation_fk
                            FOREIGN KEY (user_id, conversation_id)
                            REFERENCES conversations(user_id, id)
                            ON DELETE CASCADE
                    )
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_action_tracking_agent
                    ON NyxActionTracking(user_id, conversation_id, agent_type, agent_id)
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_action_tracking_npc
                    ON NyxActionTracking(user_id, conversation_id, npc_id)
                """)

                # For NPC directives (from original NyxGovernor)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS NyxNPCDirectives (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        conversation_id INTEGER NOT NULL,
                        npc_id INTEGER NOT NULL,
                        directive JSONB NOT NULL,
                        expires_at TIMESTAMP WITH TIME ZONE,
                        priority INTEGER DEFAULT 5,
                        scene_id VARCHAR(50),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

                        CONSTRAINT npc_directives_user_conversation_fk
                            FOREIGN KEY (user_id, conversation_id)
                            REFERENCES conversations(user_id, id)
                            ON DELETE CASCADE
                    )
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_npc_directives_npc
                    ON NyxNPCDirectives(user_id, conversation_id, npc_id)
                """)
                logger.info("Created NyxUnifiedGovernor database tables")

    # ---------------------------------------------------------------------
    # Additional NPC-specific group interaction method (from NyxGovernor)
    # ---------------------------------------------------------------------
    async def approve_group_interaction(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Approve or modify a requested group interaction for NPCs.

        (from NyxGovernor.approve_group_interaction)
        """
        result = {
            "approved": True,
            "reason": "Approved by Nyx"
        }
        try:
            narrative_data = await self._get_current_narrative_context()
            if narrative_data.get("active_arcs"):
                current_arc = narrative_data["active_arcs"][0]
                arc_npcs = current_arc.get("required_npcs", [])
                requested_npcs = request.get("npc_ids", [])
                if arc_npcs and not any(npc_id in requested_npcs for npc_id in arc_npcs):
                    current_location = request.get("context", {}).get("location")
                    required_npc_locations = await self._get_npc_locations(arc_npcs)
                    if any(required_npc_locations.get(npc_id) == current_location for npc_id in arc_npcs):
                        modified_npcs = requested_npcs.copy()
                        for npc_id in arc_npcs:
                            if required_npc_locations.get(npc_id) == current_location and npc_id not in modified_npcs:
                                modified_npcs.append(npc_id)
                        npc_names = await self._get_npc_names(arc_npcs)
                        modified_context = request.get("context", {}).copy()
                        modified_context["modified_by_nyx"] = True
                        modified_context["nyx_guidance"] = (
                            f"Ensure {npc_names} are involved to advance the current narrative arc."
                        )
                        result["approved"] = True
                        result["modified_context"] = modified_context
                        result["modified_npc_ids"] = modified_npcs
                        result["reason"] = "Modified by Nyx to include required NPCs for narrative progression"

            result["nyx_guidance"] = await self._generate_interaction_guidance(
                request.get("npc_ids", []),
                request.get("context", {})
            )
            return result

        except Exception as e:
            logger.error(f"Error in Nyx approval: {e}")
            return result

    async def _get_npc_locations(self, npc_ids: List[int]) -> Dict[int, str]:
        """
        Get current locations for a list of NPCs.
        (from NyxGovernor)
        """
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
        """
        Get names of NPCs as a formatted string.
        (from NyxGovernor)
        """
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
        """
        Generate guidance for a group interaction. (from NyxGovernor)
        """
        guidance = {
            "primary_focus": "natural_interaction",
            "npc_specific_guidance": {},
            "relationship_emphasis": []
        }
        narrative_data = await self._get_current_narrative_context()
        current_arc = None
        if narrative_data.get("active_arcs"):
            current_arc = narrative_data["active_arcs"][0]

        if current_arc:
            guidance["primary_focus"] = current_arc.get("name", "natural_interaction")
            for npc_id in npc_ids:
                for npc_role in current_arc.get("npc_roles", []):
                    if npc_role.get("npc_id") == npc_id:
                        guidance["npc_specific_guidance"][npc_id] = {
                            "role": npc_role.get("role", "supporting"),
                            "behavior": npc_role.get("behavior", "neutral"),
                            "relationship": npc_role.get("relationship", "neutral")
                        }
                        if "target_relationship" in npc_role:
                            guidance["relationship_emphasis"].append({
                                "npc_id": npc_id,
                                "target_type": npc_role.get("target_type", "player"),
                                "target_id": npc_role.get("target_id", self.user_id),
                                "relationship": npc_role.get("target_relationship")
                            })
        return guidance
