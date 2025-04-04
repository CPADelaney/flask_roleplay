# npcs/belief_system_integration.py

import logging
import asyncio
import random
from typing import Dict, List, Any, Optional, Union, Tuple
import json

from npcs.npc_belief_formation import NPCBeliefFormation
from npcs.npc_agent import NPCAgent
from data.npc_dal import NPCDataAccess
from memory.wrapper import MemorySystem
from nyx.nyx_governance import AgentType, DirectiveType
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NPCBeliefSystemIntegration:
    """
    Integrates the belief formation system with the NPC architecture.
    This ensures NPCs automatically form beliefs based on experiences and
    have their own subjective perspectives rather than omniscient knowledge.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the belief system integration.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.memory_system = None
        self.belief_formation_systems = {}  # Map of NPC ID to belief formation systems
        
    async def initialize(self):
        """Initialize the system."""
        self.memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
    
    def get_belief_system_for_npc(self, npc_id: int) -> NPCBeliefFormation:
        """
        Get or create a belief formation system for an NPC.
        
        Args:
            npc_id: NPC ID
            
        Returns:
            NPCBeliefFormation instance for this NPC
        """
        if npc_id not in self.belief_formation_systems:
            self.belief_formation_systems[npc_id] = NPCBeliefFormation(
                self.user_id, self.conversation_id, npc_id
            )
        return self.belief_formation_systems[npc_id]
    
    @with_governance(
        agent_type=AgentType.NPC,
        action_type="process_event_for_beliefs",
        action_description="Processing event for NPC belief formation",
        id_from_context=lambda ctx: "belief_system"
    )
    async def process_event_for_beliefs(
        self, 
        ctx,
        event_text: str,
        event_type: str,
        npc_ids: List[int],
        factuality: float = 1.0
    ) -> Dict[str, Any]:
        """
        Process a game event to generate beliefs for multiple NPCs.
        Different NPCs may interpret the same event differently.
        
        Args:
            event_text: Description of the event
            event_type: Type of event
            npc_ids: List of NPC IDs who witnessed the event
            factuality: Base factuality level for beliefs
            
        Returns:
            Information about the beliefs formed
        """
        if not self.memory_system:
            await self.initialize()
        
        results = {
            "event_processed": True,
            "event_text": event_text,
            "event_type": event_type,
            "npc_beliefs": {}
        }
        
        # Process the event for each NPC
        for npc_id in npc_ids:
            belief_system = self.get_belief_system_for_npc(npc_id)
            await belief_system.initialize()
            
            # Each NPC can form a slightly different belief
            # Vary factuality by up to ±20% to create differences
            npc_factuality = max(0.1, min(1.0, factuality + random.uniform(-0.2, 0.2)))
            
            try:
                belief_result = await belief_system.form_subjective_belief_from_observation(
                    observation=event_text,
                    factuality=npc_factuality
                )
                
                if "error" not in belief_result:
                    # Store the memory of the event for this NPC
                    await self.memory_system.remember(
                        entity_type="npc",
                        entity_id=npc_id,
                        memory_text=event_text,
                        importance="medium",
                        tags=["event", event_type],
                        emotional=True
                    )
                    
                results["npc_beliefs"][npc_id] = belief_result
            except Exception as e:
                logger.error(f"Error forming belief for NPC {npc_id}: {e}")
                results["npc_beliefs"][npc_id] = {"error": str(e)}
        
        return results
    
    @with_governance(
        agent_type=AgentType.NPC,
        action_type="process_conversation_for_beliefs",
        action_description="Processing conversation for NPC belief formation",
        id_from_context=lambda ctx: "belief_system"
    )
    async def process_conversation_for_beliefs(
        self, 
        ctx,
        conversation_text: str,
        speaker_id: Union[int, str],
        listener_id: int,
        topic: str = "general",
        credibility: float = 0.7
    ) -> Dict[str, Any]:
        """
        Process a conversation to generate beliefs for the listening NPC.
        
        Args:
            conversation_text: The content of what was said
            speaker_id: ID of the speaker (NPC ID or 'player')
            listener_id: NPC ID of the listener
            topic: Topic of conversation 
            credibility: How credible the speaker is (0.0-1.0)
            
        Returns:
            Information about the beliefs formed
        """
        if not self.memory_system:
            await self.initialize()
        
        belief_system = self.get_belief_system_for_npc(listener_id)
        await belief_system.initialize()
        
        # Factuality is influenced by speaker credibility
        factuality = credibility
        
        # If the speaker is an NPC, check their relationship with the listener
        if isinstance(speaker_id, int):
            # Get relationship level
            from npcs.npc_relationship import NPCRelationshipManager
            relationship_manager = NPCRelationshipManager(listener_id, self.user_id, self.conversation_id)
            relationship = await relationship_manager.get_relationship_with_npc(speaker_id)
            
            trust_level = relationship.get("trust", 50) / 100.0
            
            # Adjust factuality based on trust
            factuality = factuality * (0.3 + (trust_level * 0.7))
        
        try:
            belief_result = await belief_system.form_subjective_belief_from_observation(
                observation=conversation_text,
                factuality=factuality
            )
            
            if "error" not in belief_result:
                # Store the memory of the conversation
                speaker_description = "player" if speaker_id == "player" else f"NPC {speaker_id}"
                
                await self.memory_system.remember(
                    entity_type="npc",
                    entity_id=listener_id,
                    memory_text=f"{speaker_description} said: {conversation_text}",
                    importance="medium",
                    tags=["conversation", topic],
                    emotional=True
                )
            
            return {
                "conversation_processed": True,
                "belief_formed": "error" not in belief_result,
                "belief_details": belief_result,
                "speaker_id": speaker_id,
                "listener_id": listener_id,
                "topic": topic
            }
            
        except Exception as e:
            logger.error(f"Error forming belief from conversation for NPC {listener_id}: {e}")
            return {
                "conversation_processed": False,
                "error": str(e),
                "speaker_id": speaker_id,
                "listener_id": listener_id
            }
    
    @with_governance(
        agent_type=AgentType.NPC,
        action_type="form_narrative_from_recent_events",
        action_description="Forming a narrative belief from recent events for NPC {npc_id}",
        id_from_context=lambda ctx: f"npc_{ctx.npc_id}"
    )
    async def form_narrative_from_recent_events(
        self, 
        ctx,
        npc_id: int,
        max_events: int = 5,
        topic_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Form a narrative belief connecting recent events for an NPC.
        
        Args:
            npc_id: NPC ID
            max_events: Maximum number of events to include
            topic_filter: Optional topic to filter events by
            
        Returns:
            Information about the narrative belief formed
        """
        if not self.memory_system:
            await self.initialize()
        
        belief_system = self.get_belief_system_for_npc(npc_id)
        await belief_system.initialize()
        
        # Get recent memories
        query = topic_filter if topic_filter else ""
        tags = [topic_filter] if topic_filter else ["event"]
        
        memory_result = await self.memory_system.recall(
            entity_type="npc",
            entity_id=npc_id,
            query=query,
            limit=max_events
        )
        
        memories = memory_result.get("memories", [])
        
        if len(memories) < 2:
            return {
                "narrative_formed": False,
                "reason": "Not enough memories to form a narrative",
                "memories_found": len(memories)
            }
        
        # Form narrative
        try:
            narrative_result = await belief_system.form_narrative_from_memories(
                related_memories=memories
            )
            
            return {
                "narrative_formed": "error" not in narrative_result,
                "narrative_details": narrative_result,
                "memories_used": len(memories),
                "npc_id": npc_id
            }
            
        except Exception as e:
            logger.error(f"Error forming narrative belief for NPC {npc_id}: {e}")
            return {
                "narrative_formed": False,
                "error": str(e),
                "npc_id": npc_id
            }
    
    @with_governance(
        agent_type=AgentType.NPC,
        action_type="process_cultural_topic_for_beliefs",
        action_description="Forming cultural beliefs about topic for NPC {npc_id}",
        id_from_context=lambda ctx: f"npc_{ctx.npc_id}"
    )
    async def process_cultural_topic_for_beliefs(
        self, 
        ctx,
        npc_id: int,
        topic: str
    ) -> Dict[str, Any]:
        """
        Form a culturally influenced belief about a topic.
        
        Args:
            npc_id: NPC ID
            topic: The subject to form a cultural belief about
            
        Returns:
            Information about the cultural belief formed
        """
        if not self.memory_system:
            await self.initialize()
        
        belief_system = self.get_belief_system_for_npc(npc_id)
        await belief_system.initialize()
        
        try:
            belief_result = await belief_system.form_culturally_influenced_belief(
                subject=topic
            )
            
            return {
                "cultural_belief_formed": "error" not in belief_result,
                "belief_details": belief_result,
                "npc_id": npc_id,
                "topic": topic
            }
            
        except Exception as e:
            logger.error(f"Error forming cultural belief for NPC {npc_id}: {e}")
            return {
                "cultural_belief_formed": False,
                "error": str(e),
                "npc_id": npc_id,
                "topic": topic
            }
    
    async def update_beliefs_on_knowledge_discovery(
        self,
        npc_id: int,
        knowledge_type: str,
        knowledge_id: int
    ) -> Dict[str, Any]:
        """
        Update existing beliefs when an NPC discovers new knowledge.
        
        Args:
            npc_id: NPC ID
            knowledge_type: Type of knowledge (lore_type)
            knowledge_id: ID of knowledge
            
        Returns:
            Information about updated beliefs
        """
        if not self.memory_system:
            await self.initialize()
        
        belief_system = self.get_belief_system_for_npc(npc_id)
        await belief_system.initialize()
        
        try:
            result = await belief_system.reevaluate_beliefs_based_on_new_knowledge(
                knowledge_type=knowledge_type,
                knowledge_id=knowledge_id
            )
            
            return {
                "beliefs_updated": result.get("updated", 0) > 0,
                "update_details": result,
                "npc_id": npc_id,
                "knowledge_type": knowledge_type,
                "knowledge_id": knowledge_id
            }
            
        except Exception as e:
            logger.error(f"Error updating beliefs for NPC {npc_id}: {e}")
            return {
                "beliefs_updated": False,
                "error": str(e),
                "npc_id": npc_id
            }
    
    async def add_opposing_beliefs(
        self,
        npc_id: int,
        chance: float = 0.3
    ) -> Dict[str, Any]:
        """
        Add opposing beliefs to some of an NPC's existing beliefs.
        This creates more realistic, contradictory belief systems.
        
        Args:
            npc_id: NPC ID
            chance: Chance of creating an opposing belief (0.0-1.0)
            
        Returns:
            Information about opposing beliefs created
        """
        if not self.memory_system:
            await self.initialize()
        
        belief_system = self.get_belief_system_for_npc(npc_id)
        await belief_system.initialize()
        
        # Get all NPC's beliefs
        beliefs = await self.memory_system.get_beliefs(
            entity_type="npc",
            entity_id=npc_id
        )
        
        if not beliefs:
            return {
                "opposing_beliefs_added": False,
                "reason": "No existing beliefs found",
                "npc_id": npc_id
            }
        
        results = {
            "opposing_beliefs_added": 0,
            "belief_pairs": [],
            "npc_id": npc_id
        }
        
        # Select a random subset of beliefs to create opposites for
        for belief in beliefs:
            if random.random() < chance:
                belief_id = belief.get("id")
                if not belief_id:
                    continue
                    
                try:
                    opposing_result = await belief_system.create_opposing_belief(belief_id)
                    
                    if "error" not in opposing_result:
                        results["opposing_beliefs_added"] += 1
                        results["belief_pairs"].append({
                            "original": belief.get("belief", ""),
                            "opposing": opposing_result.get("belief_text", "")
                        })
                except Exception as e:
                    logger.error(f"Error creating opposing belief for NPC {npc_id}: {e}")
        
        return results

# Decorator to enhance an NPC agent with belief system
def enhance_npc_with_belief_system(npc_agent: NPCAgent) -> NPCAgent:
    """
    Enhance an NPC agent with belief system capabilities.
    
    Args:
        npc_agent: NPC agent to enhance
        
    Returns:
        Enhanced NPC agent
    """
    user_id = npc_agent.user_id
    conversation_id = npc_agent.conversation_id
    npc_id = npc_agent.npc_id
    
    # Create integration system
    integration = NPCBeliefSystemIntegration(user_id, conversation_id)
    belief_system = integration.get_belief_system_for_npc(npc_id)
    
    # Add it to the agent
    npc_agent.belief_system = belief_system
    npc_agent.belief_integration = integration
    
    # Add convenience methods
    async def form_belief_about(observation: str, factuality: float = 1.0):
        await belief_system.initialize()
        return await belief_system.form_subjective_belief_from_observation(
            observation=observation,
            factuality=factuality
        )
    
    async def form_narrative_about(memories: List[Dict[str, Any]]):
        await belief_system.initialize()
        return await belief_system.form_narrative_from_memories(
            related_memories=memories
        )
    
    async def form_cultural_belief_about(subject: str):
        await belief_system.initialize()
        return await belief_system.form_culturally_influenced_belief(
            subject=subject
        )
    
    # Attach methods to agent
    npc_agent.form_belief_about = form_belief_about
    npc_agent.form_narrative_about = form_narrative_about
    npc_agent.form_cultural_belief_about = form_cultural_belief_about
    
    return npc_agent
