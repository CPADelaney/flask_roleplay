# npcs/nyx_integration.py

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from npcs.npc_agent import NPCAgent
from npcs.npc_agent_system import NPCAgentSystem
from nyx.nyx_governance import NyxGovernor, DirectiveType, DirectivePriority

logger = logging.getLogger(__name__)

class NyxNPCBridge:
    """
    Central integration point between Nyx and all NPC agents.
    This ensures Nyx has ultimate governance over all NPCs.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = NyxGovernor(user_id, conversation_id)
        self.npc_system = None
        
    async def get_npc_system(self) -> NPCAgentSystem:
        """Lazy-load the NPC system."""
        if self.npc_system is None:
            # You'll need to properly initialize this with your connection pool
            from db.connection import get_db_connection
            import asyncpg
            
            pool = await asyncpg.create_pool(dsn=get_db_connection())
            self.npc_system = NPCAgentSystem(self.user_id, self.conversation_id, pool)
            await self.npc_system.initialize_agents()
        return self.npc_system
    
    async def issue_scene_directives(self, scene_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Issue directives to all NPCs involved in a scene as planned by Nyx.
        
        Args:
            scene_plan: Nyx's plan for the scene
            
        Returns:
            Results of issuing directives
        """
        results = {}
        
        for npc_directive in scene_plan.get("npc_directives", []):
            npc_id = npc_directive.get("npc_id")
            if not npc_id:
                continue
                
            directive_id = await self.governor.issue_directive(
                npc_id=npc_id,
                directive_type=DirectiveType.SCENE,
                directive_data=npc_directive.get("directive", {}),
                priority=DirectivePriority.HIGH,
                duration_minutes=30,
                scene_id=scene_plan.get("scene_id", "unknown")
            )
            
            results[npc_id] = {
                "directive_id": directive_id,
                "status": "issued" if directive_id > 0 else "failed"
            }
        
        return {
            "scene_id": scene_plan.get("scene_id"),
            "directives_issued": sum(1 for r in results.values() if r["status"] == "issued"),
            "results": results
        }
    
    async def process_group_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an event that affects multiple NPCs, with Nyx's governance.
        
        Args:
            event_type: Type of event
            event_data: Event details
            
        Returns:
            Results of processing the event
        """
        npc_system = await self.get_npc_system()
        
        # Let Nyx filter and modify the event first
        from nyx.integrate import GameEventManager
        event_manager = GameEventManager(self.user_id, self.conversation_id)
        
        broadcast_result = await event_manager.broadcast_event(event_type, event_data)
        
        # Return early if Nyx blocked the event
        if broadcast_result.get("blocked_by_nyx", False):
            return {
                "processed": False,
                "reason": broadcast_result.get("reason", "Blocked by Nyx"),
                "npc_responses": []
            }
        
        # If event affects NPCs at a location, process through the NPCs there
        affected_npcs = broadcast_result.get("aware_npcs", [])
        
        if affected_npcs:
            # Convert event to player action format for NPC system
            player_action = {
                "type": "system_event",
                "description": f"Event: {event_type}",
                "event_data": event_data
            }
            
            # Let NPCs respond to the event through the normal pathway
            # which already has Nyx governance integration
            result = await npc_system.handle_group_npc_interaction(
                affected_npcs,
                player_action,
                {"event_type": event_type, "from_nyx": True}
            )
            
            return {
                "processed": True,
                "affected_npcs": affected_npcs,
                "npc_responses": result.get("npc_responses", [])
            }
        
        return {
            "processed": True,
            "affected_npcs": [],
            "npc_responses": []
        }
