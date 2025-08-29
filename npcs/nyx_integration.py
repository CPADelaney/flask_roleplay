# npcs/nyx_integration.py

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Core NPC systems
from npcs.npc_agent import NPCAgent
from npcs.npc_agent_system import NPCAgentSystem

# Nyx governance
from nyx.nyx_governance import NyxUnifiedGovernor, DirectiveType, DirectivePriority

# Memory governance bridge
from memory.memory_nyx_integration import get_memory_nyx_bridge

# DB helpers
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)


class NyxNPCBridge:
    """
    Central integration point between Nyx and all NPC agents.
    Provides:
    - Directive issuance (scene/single-NPC)
    - Governed memory operations (remember/recall/beliefs/maintenance)
    - Group event routing through Nyx filters
    """

    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        enable_governance: bool = True,
        enable_memory_governance: bool = True
    ):
        self.user_id = int(user_id)
        self.conversation_id = int(conversation_id)

        self.enable_governance = bool(enable_governance)
        self.enable_memory_governance = bool(enable_memory_governance)

        self.governor: Optional[NyxUnifiedGovernor] = None
        self.npc_system: Optional[NPCAgentSystem] = None
        self.memory_bridge = None

    # ----- Lazy getters -----

    async def _get_governor(self) -> NyxUnifiedGovernor:
        if self.governor is None:
            self.governor = NyxUnifiedGovernor(self.user_id, self.conversation_id)
        return self.governor

    async def get_npc_system(self) -> NPCAgentSystem:
        """Lazy-load the NPC agent system (initialize agents if needed)."""
        if self.npc_system is None:
            self.npc_system = NPCAgentSystem(self.user_id, self.conversation_id, None)
            try:
                await self.npc_system.initialize_agents()
            except Exception as e:
                logger.debug(f"[NyxBridge] initialize_agents failed (continuing): {e}")
        return self.npc_system

    async def get_memory_bridge(self):
        """Lazy-load the governed memory bridge."""
        if self.memory_bridge is None:
            self.memory_bridge = await get_memory_nyx_bridge(self.user_id, self.conversation_id)
        return self.memory_bridge

    # ----- Directive issuance -----

    async def issue_scene_directives(self, scene_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Issue directives to all NPCs involved in a scene.

        scene_plan:
        {
          "scene_id": str,
          "npc_directives": [
            {"npc_id": int, "directive": {...}}
          ]
        }
        """
        if not self.enable_governance:
            return {"enabled": False, "scene_id": scene_plan.get("scene_id"), "results": {}}

        gov = await self._get_governor()
        results: Dict[int, Dict[str, Any]] = {}
        issued_count = 0

        for item in scene_plan.get("npc_directives", []) or []:
            npc_id = item.get("npc_id")
            if not npc_id:
                continue

            directive_data = item.get("directive", {}) or {}
            try:
                directive_id = await gov.issue_directive(
                    npc_id=int(npc_id),
                    directive_type=DirectiveType.SCENE,
                    directive_data=directive_data,
                    priority=DirectivePriority.HIGH,
                    duration_minutes=int(directive_data.get("duration_minutes", 30)),
                    scene_id=str(scene_plan.get("scene_id", "unknown"))
                )
                ok = directive_id and directive_id > 0
                results[int(npc_id)] = {"directive_id": directive_id, "status": "issued" if ok else "failed"}
                if ok:
                    issued_count += 1
            except Exception as e:
                logger.debug(f"[NyxBridge] issue_directive failed for NPC {npc_id}: {e}")
                results[int(npc_id)] = {"directive_id": None, "status": "error", "error": str(e)}

        return {
            "scene_id": scene_plan.get("scene_id"),
            "directives_issued": issued_count,
            "results": results
        }

    async def issue_directive_to_npc(
        self,
        npc_id: int,
        directive_type: DirectiveType,
        directive_data: Dict[str, Any],
        priority: DirectivePriority = DirectivePriority.MEDIUM,
        duration_minutes: int = 30,
        scene_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Issue a single directive to one NPC."""
        if not self.enable_governance:
            return {"enabled": False, "status": "skipped"}

        try:
            gov = await self._get_governor()
            directive_id = await gov.issue_directive(
                npc_id=int(npc_id),
                directive_type=directive_type,
                directive_data=directive_data or {},
                priority=priority,
                duration_minutes=int(duration_minutes),
                scene_id=scene_id or "ad-hoc"
            )
            return {
                "npc_id": int(npc_id),
                "directive_id": directive_id,
                "status": "issued" if directive_id and directive_id > 0 else "failed"
            }
        except Exception as e:
            logger.debug(f"[NyxBridge] issue_directive_to_npc failed for NPC {npc_id}: {e}")
            return {"npc_id": int(npc_id), "status": "error", "error": str(e)}

    # ----- Group event processing -----

    async def process_group_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a world/game event through Nyx, optionally routing to NPC system.

        Returns:
          {
            "processed": bool,
            "reason": Optional[str],
            "affected_npcs": List[int],
            "npc_responses": List[Dict]
          }
        """
        try:
            from nyx.integrate import GameEventManager
        except Exception as e:
            logger.debug(f"[NyxBridge] GameEventManager import failed: {e}")
            return {"processed": False, "reason": "Nyx integration unavailable", "npc_responses": []}

        try:
            event_manager = GameEventManager(self.user_id, self.conversation_id)
            broadcast_result = await event_manager.broadcast_event(event_type, event_data)
        except Exception as e:
            logger.debug(f"[NyxBridge] broadcast_event failed: {e}")
            return {"processed": False, "reason": "broadcast_failed", "npc_responses": []}

        if broadcast_result.get("blocked_by_nyx", False):
            return {
                "processed": False,
                "reason": broadcast_result.get("reason", "Blocked by Nyx"),
                "npc_responses": [],
                "affected_npcs": []
            }

        affected_npcs = list(broadcast_result.get("aware_npcs", []) or [])
        if not affected_ncs := affected_npcs:
            return {"processed": True, "affected_npcs": [], "npc_responses": []}

        try:
            npc_system = await self.get_npc_system()
            player_action = {
                "type": "system_event",
                "description": f"Event: {event_type}",
                "event_data": event_data
            }
            result = await npc_system.handle_group_npc_interaction(
                affected_ncs,
                player_action,
                {"event_type": event_type, "from_nyx": True}
            )
            return {
                "processed": True,
                "affected_npcs": affected_ncs,
                "npc_responses": result.get("npc_responses", [])
            }
        except Exception as e:
            logger.debug(f"[NyxBridge] NPC system routing failed: {e}")
            return {
                "processed": True,
                "affected_npcs": affected_ncs,
                "npc_responses": [],
                "warning": "npc_system_routing_failed"
            }

    # ----- Governed memory operations -----

    async def remember_for_npc(
        self,
        npc_id: int,
        memory_text: str,
        importance: str = "medium",
        emotional: bool = True,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a memory for an NPC through Nyx governance if enabled, otherwise
        uses raw bridge (still governed in this module).
        """
        try:
            bridge = await self.get_memory_bridge()
        except Exception as e:
            logger.debug(f"[NyxBridge] memory bridge unavailable: {e}")
            return {"success": False, "error": "memory_bridge_unavailable"}

        # Add npc tag for traceability
        tags = list(tags or [])
        npc_tag = f"npc_{int(npc_id)}"
        if npc_tag not in tags:
            tags.append(npc_tag)

        try:
            return await bridge.remember(
                entity_type="npc",
                entity_id=int(npc_id),
                memory_text=str(memory_text),
                importance=str(importance),
                emotional=bool(emotional),
                tags=tags
            )
        except Exception as e:
            logger.debug(f"[NyxBridge] remember_for_npc failed for NPC {npc_id}: {e}")
            return {"success": False, "error": str(e)}

    async def recall_for_npc(
        self,
        npc_id: int,
        query: Optional[str] = None,
        context: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """Recall memories for an NPC through governed bridge."""
        try:
            bridge = await self.get_memory_bridge()
            return await bridge.recall(
                entity_type="npc",
                entity_id=int(npc_id),
                query=query,
                context=context,
                limit=int(limit)
            )
        except Exception as e:
            logger.debug(f"[NyxBridge] recall_for_npc failed for NPC {npc_id}: {e}")
            return {"memories": [], "error": str(e)}

    async def create_belief_for_npc(
        self,
        npc_id: int,
        belief_text: str,
        confidence: float = 0.7
    ) -> Dict[str, Any]:
        """Create a belief for an NPC through governed bridge."""
        try:
            bridge = await self.get_memory_bridge()
            return await bridge.create_belief(
                entity_type="npc",
                entity_id=int(npc_id),
                belief_text=str(belief_text),
                confidence=float(confidence)
            )
        except Exception as e:
            logger.debug(f"[NyxBridge] create_belief_for_npc failed for NPC {npc_id}: {e}")
            return {"success": False, "error": str(e)}

    async def get_beliefs_for_npc(
        self,
        npc_id: int,
        topic: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get beliefs for an NPC through governed bridge."""
        try:
            bridge = await self.get_memory_bridge()
            beliefs = await bridge.get_beliefs(entity_type="npc", entity_id=int(npc_id), topic=topic)
            return list(beliefs or [])
        except Exception as e:
            logger.debug(f"[NyxBridge] get_beliefs_for_npc failed for NPC {npc_id}: {e}")
            return []

    async def run_memory_maintenance_for_npc(self, npc_id: int) -> Dict[str, Any]:
        """Run memory maintenance for an NPC through governed bridge."""
        try:
            bridge = await self.get_memory_bridge()
            return await bridge.run_maintenance(entity_type="npc", entity_id=int(npc_id))
        except Exception as e:
            logger.debug(f"[NyxBridge] run_memory_maintenance_for_npc failed for NPC {npc_id}: {e}")
            return {"success": False, "error": str(e)}


class NPCMemoryAccess:
    """
    Governed memory accessor bound to a single NPC.
    Attach this to agents to replace direct memory calls with Nyx-governed calls.
    """

    def __init__(self, npc_id: int, user_id: int, conversation_id: int):
        self.npc_id = int(npc_id)
        self.user_id = int(user_id)
        self.conversation_id = int(conversation_id)
        self.bridge: Optional[NyxNPCBridge] = None

    async def get_bridge(self) -> NyxNPCBridge:
        if self.bridge is None:
            self.bridge = NyxNPCBridge(self.user_id, self.conversation_id)
        return self.bridge

    async def remember(
        self,
        memory_text: str,
        importance: str = "medium",
        emotional: bool = True,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        bridge = await self.get_bridge()
        return await bridge.remember_for_npc(
            npc_id=self.npc_id,
            memory_text=memory_text,
            importance=importance,
            emotional=bool(emotional),
            tags=tags
        )

    async def recall(
        self,
        query: Optional[str] = None,
        context: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        bridge = await self.get_bridge()
        return await bridge.recall_for_npc(
            npc_id=self.npc_id,
            query=query,
            context=context,
            limit=int(limit)
        )

    async def create_belief(self, belief_text: str, confidence: float = 0.7) -> Dict[str, Any]:
        bridge = await self.get_bridge()
        return await bridge.create_belief_for_npc(
            npc_id=self.npc_id,
            belief_text=belief_text,
            confidence=float(confidence)
        )

    async def get_beliefs(self, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        bridge = await self.get_bridge()
        return await bridge.get_beliefs_for_npc(npc_id=self.npc_id, topic=topic)

    async def run_maintenance(self) -> Dict[str, Any]:
        bridge = await self.get_bridge()
        return await bridge.run_memory_maintenance_for_npc(npc_id=self.npc_id)


def enhance_npc_with_memory_access(npc_agent: NPCAgent) -> NPCAgent:
    """
    Enhance an NPCAgent instance with governed memory access methods.

    Adds attributes:
      - npc_agent.memory_access
      - npc_agent.remember_with_governance(...)
      - npc_agent.recall_with_governance(...)
      - npc_agent.create_belief_with_governance(...)
      - npc_agent.get_beliefs_with_governance(...)
    """
    memory_access = NPCMemoryAccess(
        npc_id=npc_agent.npc_id,
        user_id=npc_agent.user_id,
        conversation_id=npc_agent.conversation_id
    )

    npc_agent.memory_access = memory_access

    async def remember_with_governance(memory_text, importance="medium", emotional=True, tags=None):
        return await memory_access.remember(
            memory_text=memory_text,
            importance=importance,
            emotional=emotional,
            tags=tags
        )

    async def recall_with_governance(query=None, context=None, limit=5):
        return await memory_access.recall(query=query, context=context, limit=limit)

    async def create_belief_with_governance(belief_text, confidence=0.7):
        return await memory_access.create_belief(belief_text=belief_text, confidence=confidence)

    async def get_beliefs_with_governance(topic=None):
        return await memory_access.get_beliefs(topic=topic)

    npc_agent.remember_with_governance = remember_with_governance  # type: ignore[attr-defined]
    npc_agent.recall_with_governance = recall_with_governance      # type: ignore[attr-defined]
    npc_agent.create_belief_with_governance = create_belief_with_governance  # type: ignore[attr-defined]
    npc_agent.get_beliefs_with_governance = get_beliefs_with_governance      # type: ignore[attr-defined]

    return npc_agent


async def create_npc_with_memory_governance(npc_id: int, user_id: int, conversation_id: int) -> NPCAgent:
    """
    Factory to create an NPCAgent with governed memory access.
    """
    agent = NPCAgent(int(npc_id), int(user_id), int(conversation_id))
    await agent.initialize()
    enhance_npc_with_memory_access(agent)
    return agent
