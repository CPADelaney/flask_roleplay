# nyx/integrate.py

import logging
from typing import Dict, List, Any, Optional
from nyx.nyx_agent import NyxAgent
from logic.npc_agents.agent_coordinator import NPCAgentCoordinator

class GameEventManager:
    """Manages events that both Nyx and NPCs should be aware of"""
    
    def __init__(self, user_id, conversation_id):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.nyx_agent = NyxAgent(user_id, conversation_id)
        self.npc_coordinator = NPCAgentCoordinator(user_id, conversation_id)
    
    async def broadcast_event(self, event_type, event_data):
        """Broadcast event to both Nyx and NPCs"""
        # Tell Nyx about the event
        await self.nyx_agent.process_game_event(event_type, event_data)
        
        # Tell NPCs about the event
        affected_npcs = event_data.get("affected_npcs")
        if not affected_npcs:
            # If no specific NPCs mentioned, determine who would know
            affected_npcs = await self._determine_aware_npcs(event_type, event_data)
        
        if affected_npcs:
            await self.npc_coordinator.batch_update_npcs(
                affected_npcs,
                "event_update",
                {"event_type": event_type, "event_data": event_data}
            )

import logging
from typing import Dict, List, Any, Optional
from nyx.nyx_agent import NyxAgent
from logic.npc_agents.agent_coordinator import NPCAgentCoordinator

class NyxNPCIntegrationManager:
    """Manages integration between Nyx and NPC agents"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.nyx_agent = NyxAgent(user_id, conversation_id)
        self.npc_coordinator = NPCAgentCoordinator(user_id, conversation_id)
    
    async def process_user_input(self, user_input: str, context: Dict[str, Any]):
        """Process user input through both systems in an integrated way"""
        # First get Nyx's response
        nyx_response = await self.nyx_agent.process_input(user_input, context)
        
        # Extract NPC guidance from Nyx's response
        npc_guidance = self._extract_npc_guidance(nyx_response)
        
        # Add Nyx's guidance to context
        npc_context = context.copy()
        npc_context["nyx_guidance"] = npc_guidance
        
        # Get NPC responses with Nyx's guidance
        npc_responses = []
        if "responding_npcs" in npc_guidance:
            npc_responses = await self.npc_coordinator.handle_player_action(
                {"description": user_input, "type": "talk"},
                npc_context,
                npc_guidance["responding_npcs"]
            )
        
        # Combine responses
        combined_response = self._combine_responses(nyx_response, npc_responses)
        return combined_response


async def transition_scene_with_npcs(
    self,
    user_id: int,
    conversation_id: int,
    new_location: str,
    transition_context: Dict[str, Any]
):
    """Manage scene transition with NPC movement"""
    # First, handle Nyx's scene transition
    scene_result = await self.transition_scene(
        user_id, conversation_id, new_location, transition_context
    )
    
    # Get NPCs that should move to the new location
    npc_coordinator = NPCAgentCoordinator(user_id, conversation_id)
    npcs_to_move = transition_context.get("accompanying_npcs", [])
    
    # Move NPCs to the new location
    await npc_coordinator.batch_update_npcs(
        npcs_to_move,
        "location_change",
        {"new_location": new_location}
    )
    
    # Generate emotions and memory for NPCs about the location change
    await npc_coordinator.batch_update_npcs(
        npcs_to_move,
        "memory_update",
        {"memory_text": f"I moved to {new_location} with the player", 
         "tags": ["location_change"]}
    )
    
    return scene_result


import logging
from typing import Dict, Any
from nyx.nyx_agent import NyxAgent
from logic.npc_agents.agent_coordinator import NPCAgentCoordinator

async def run_joint_memory_maintenance(user_id: int, conversation_id: int):
    """Maintain both Nyx and NPC memories with coordination"""
    # Run Nyx memory maintenance
    nyx_agent = NyxAgent(user_id, conversation_id)
    nyx_result = await nyx_agent.run_memory_maintenance()
    
    # Run NPC memory maintenance
    npc_coordinator = NPCAgentCoordinator(user_id, conversation_id)
    npc_ids = await npc_coordinator.load_agents()
    
    # Create reflections based on Nyx's insights
    if "reflections" in nyx_result:
        for reflection in nyx_result["reflections"]:
            await npc_coordinator.batch_update_npcs(
                npc_ids,
                "belief_update",
                {"belief_text": f"Nyx thinks: {reflection}",
                 "confidence": 0.8,
                 "topic": "nyx_perspective"}
            )
