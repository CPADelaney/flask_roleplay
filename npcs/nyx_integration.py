# npcs/nyx_integration.py

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from npcs.npc_agent import NPCAgent
from npcs.npc_agent_system import NPCAgentSystem
from nyx.nyx_governance import NyxUnifiedGovernor, DirectiveType, DirectivePriority  # Changed from NyxGovernor
from memory.memory_nyx_integration import get_memory_nyx_bridge
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

class NyxNPCBridge:
    """
    Central integration point between Nyx and all NPC agents.
    This ensures Nyx has ultimate governance over all NPCs.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = NyxUnifiedGovernor(user_id, conversation_id)  # Changed from NyxGovernor
        self.npc_system = None
        self.memory_bridge = None
        
    async def get_npc_system(self) -> NPCAgentSystem:
        """Lazy-load the NPC system."""
        if self.npc_system is None:
            # Updated to use async connection pool
            self.npc_system = NPCAgentSystem(self.user_id, self.conversation_id, None)
            await self.npc_system.initialize_agents()
        return self.npc_system
    
    async def get_memory_bridge(self):
        """Lazy-load the memory bridge."""
        if self.memory_bridge is None:
            self.memory_bridge = await get_memory_nyx_bridge(self.user_id, self.conversation_id)
        return self.memory_bridge
    
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
    
    # ----- Memory Integration Methods -----
    
    async def remember_for_npc(
        self,
        npc_id: int,
        memory_text: str,
        importance: str = "medium",
        emotional: bool = True,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a memory for an NPC with Nyx governance.
        
        Args:
            npc_id: ID of the NPC
            memory_text: The memory text to record
            importance: Importance level ("trivial", "low", "medium", "high", "critical")
            emotional: Whether to analyze emotional content
            tags: Optional tags for the memory
        """
        # Add NPC-specific tag if not present
        if tags is None:
            tags = []
        npc_tag = f"npc_{npc_id}"
        if npc_tag not in tags:
            tags.append(npc_tag)
        
        memory_bridge = await self.get_memory_bridge()
        return await memory_bridge.remember(
            entity_type="npc",
            entity_id=npc_id,
            memory_text=memory_text,
            importance=importance,
            emotional=emotional,
            tags=tags
        )
    
    async def recall_for_npc(
        self,
        npc_id: int,
        query: Optional[str] = None,
        context: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Recall memories for an NPC with Nyx governance.
        
        Args:
            npc_id: ID of the NPC
            query: Optional search query
            context: Current context that might influence recall
            limit: Maximum number of memories to return
        """
        memory_bridge = await self.get_memory_bridge()
        return await memory_bridge.recall(
            entity_type="npc",
            entity_id=npc_id,
            query=query,
            context=context,
            limit=limit
        )
    
    async def create_belief_for_npc(
        self,
        npc_id: int,
        belief_text: str,
        confidence: float = 0.7
    ) -> Dict[str, Any]:
        """
        Create a belief for an NPC with Nyx governance.
        
        Args:
            npc_id: ID of the NPC
            belief_text: The belief statement
            confidence: Confidence in this belief (0.0-1.0)
        """
        memory_bridge = await self.get_memory_bridge()
        return await memory_bridge.create_belief(
            entity_type="npc",
            entity_id=npc_id,
            belief_text=belief_text,
            confidence=confidence
        )
    
    async def get_beliefs_for_npc(
        self,
        npc_id: int,
        topic: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get beliefs for an NPC with Nyx governance.
        
        Args:
            npc_id: ID of the NPC
            topic: Optional topic filter
        """
        memory_bridge = await self.get_memory_bridge()
        return await memory_bridge.get_beliefs(
            entity_type="npc",
            entity_id=npc_id,
            topic=topic
        )
    
    async def run_memory_maintenance_for_npc(
        self,
        npc_id: int
    ) -> Dict[str, Any]:
        """
        Run memory maintenance for an NPC with Nyx governance.
        
        Args:
            npc_id: ID of the NPC
        """
        memory_bridge = await self.get_memory_bridge()
        return await memory_bridge.run_maintenance(
            entity_type="npc",
            entity_id=npc_id
        )


class NPCMemoryAccess:
    """
    Provides governed access to the memory system for a specific NPC.
    
    This class is used to enhance an NPC agent with memory capabilities
    that are governed by Nyx.
    """
    
    def __init__(self, npc_id: int, user_id: int, conversation_id: int):
        """
        Initialize memory access for an NPC.
        
        Args:
            npc_id: NPC ID
            user_id: User ID
            conversation_id: Conversation ID
        """
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.bridge = None
    
    async def get_bridge(self):
        """Lazy-load the Nyx-NPC bridge."""
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
        """
        Create a memory for this NPC through Nyx governance.
        
        Args:
            memory_text: The memory text to record
            importance: Importance level ("trivial", "low", "medium", "high", "critical")
            emotional: Whether to analyze emotional content
            tags: Optional tags for the memory
        """
        bridge = await self.get_bridge()
        return await bridge.remember_for_npc(
            npc_id=self.npc_id,
            memory_text=memory_text,
            importance=importance,
            emotional=emotional,
            tags=tags
        )
    
    async def recall(
        self,
        query: Optional[str] = None,
        context: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Recall memories for this NPC through Nyx governance.
        
        Args:
            query: Optional search query
            context: Current context that might influence recall
            limit: Maximum number of memories to return
        """
        bridge = await self.get_bridge()
        return await bridge.recall_for_npc(
            npc_id=self.npc_id,
            query=query,
            context=context,
            limit=limit
        )
    
    async def create_belief(
        self,
        belief_text: str,
        confidence: float = 0.7
    ) -> Dict[str, Any]:
        """
        Create a belief for this NPC through Nyx governance.
        
        Args:
            belief_text: The belief statement
            confidence: Confidence in this belief (0.0-1.0)
        """
        bridge = await self.get_bridge()
        return await bridge.create_belief_for_npc(
            npc_id=self.npc_id,
            belief_text=belief_text,
            confidence=confidence
        )
    
    async def get_beliefs(
        self,
        topic: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get beliefs for this NPC through Nyx governance.
        
        Args:
            topic: Optional topic filter
        """
        bridge = await self.get_bridge()
        return await bridge.get_beliefs_for_npc(
            npc_id=self.npc_id,
            topic=topic
        )
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """
        Run memory maintenance for this NPC through Nyx governance.
        """
        bridge = await self.get_bridge()
        return await bridge.run_memory_maintenance_for_npc(
            npc_id=self.npc_id
        )


# Function to update NPCAgent class to use governed memory access
def enhance_npc_with_memory_access(npc_agent: NPCAgent) -> NPCAgent:
    """
    Enhance an NPC agent with governed memory access.
    This replaces direct memory access with governed access through Nyx.
    
    Args:
        npc_agent: The NPC agent to enhance
    """
    # Create memory access for this NPC
    memory_access = NPCMemoryAccess(
        npc_id=npc_agent.npc_id,
        user_id=npc_agent.user_id,
        conversation_id=npc_agent.conversation_id
    )
    
    # Add memory access to the NPC agent
    npc_agent.memory_access = memory_access
    
    # Add accessor methods to maintain backward compatibility
    async def remember_with_governance(memory_text, importance="medium", emotional=True, tags=None):
        return await memory_access.remember(
            memory_text=memory_text,
            importance=importance,
            emotional=emotional,
            tags=tags
        )
    
    async def recall_with_governance(query=None, context=None, limit=5):
        return await memory_access.recall(
            query=query,
            context=context,
            limit=limit
        )
    
    async def create_belief_with_governance(belief_text, confidence=0.7):
        return await memory_access.create_belief(
            belief_text=belief_text,
            confidence=confidence
        )
    
    async def get_beliefs_with_governance(topic=None):
        return await memory_access.get_beliefs(topic=topic)
    
    # Attach the methods to the NPC agent
    npc_agent.remember_with_governance = remember_with_governance
    npc_agent.recall_with_governance = recall_with_governance
    npc_agent.create_belief_with_governance = create_belief_with_governance
    npc_agent.get_beliefs_with_governance = get_beliefs_with_governance
    
    # Return the enhanced agent
    return npc_agent


# Helper function to create an NPC with governed memory access
async def create_npc_with_memory_governance(npc_id: int, user_id: int, conversation_id: int) -> NPCAgent:
    """
    Create an NPC agent with governed memory access.
    
    Args:
        npc_id: NPC ID
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        NPC agent with governed memory access
    """
    from npcs.npc_agent import NPCAgent
    
    # Create the NPC agent
    npc_agent = NPCAgent(npc_id, user_id, conversation_id)
    await npc_agent.initialize()
    
    # Enhance it with governed memory access
    enhance_npc_with_memory_access(npc_agent)
    
    return npc_agent
