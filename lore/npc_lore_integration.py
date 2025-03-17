# lore/npc_lore_integration.py

import logging
import asyncio
import random
from typing import Dict, List, Any, Optional, Tuple
import json

from db.connection import get_db_connection
from lore.lore_manager import LoreManager
from memory.wrapper import MemorySystem

# Import belief system
from npcs.npc_belief_formation import NPCBeliefFormation
from npcs.belief_system_integration import NPCBeliefSystemIntegration

# Import Nyx governance
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting

class NPCLoreIntegration:
    """
    Integrates lore into NPC agent behavior and memory systems with Nyx governance oversight.
    """
    
    def __init__(self, user_id: int, conversation_id: int, npc_id: int = None):
        """
        Initialize the NPC lore integration.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            npc_id: Optional NPC ID for specific NPC integration
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.npc_id = npc_id
        self.lore_manager = LoreManager(user_id, conversation_id)
        self.governor = None
        
        # Initialize belief systems
        self.belief_integration = NPCBeliefSystemIntegration(user_id, conversation_id)
        self.belief_system = None
        if npc_id:
            self.belief_system = self.belief_integration.get_belief_system_for_npc(npc_id)
    
    async def initialize_governance(self):
        """Initialize Nyx governance connection"""
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
        return self.governor
    
    async def initialize_belief_system(self):
        """Initialize belief system if needed"""
        if self.npc_id and self.belief_system:
            await self.belief_system.initialize()
        await self.belief_integration.initialize()
    
    @with_governance(
        agent_type=AgentType.NPC,
        action_type="initialize_npc_lore_knowledge",
        action_description="Initializing lore knowledge for NPC {npc_id}",
        id_from_context=lambda ctx: f"npc_{ctx.npc_id}"
    )
    async def initialize_npc_lore_knowledge(self, ctx, npc_id: int, 
                                          cultural_background: str,
                                          faction_affiliations: List[str]) -> Dict[str, Any]:
        """
        Initialize an NPC's knowledge of lore based on their background with Nyx governance oversight.
        
        Args:
            npc_id: ID of the NPC
            cultural_background: Cultural background of the NPC
            faction_affiliations: List of faction names the NPC is affiliated with
            
        Returns:
            Dictionary of knowledge granted
        """
        # Update NPC's cultural background and faction memberships
        await self.lore_manager.update_npc_lore_knowledge(
            npc_id,
            cultural_background=cultural_background,
            faction_memberships=faction_affiliations
        )
        
        # Track knowledge granted
        knowledge_granted = {
            "world_lore": [],
            "cultural_elements": [],
            "factions": [],
            "historical_events": []
        }
        
        # Get IDs of the factions the NPC is affiliated with
        faction_ids = []
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                for faction_name in faction_affiliations:
                    faction_id = await conn.fetchval("""
                        SELECT id FROM Factions
                        WHERE name = $1
                    """, faction_name)
                    
                    if faction_id:
                        faction_ids.append(faction_id)
        
        # 1. Give knowledge of the NPC's own factions
        for faction_id in faction_ids:
            # Knowledge level 7-9 for own factions (high knowledge)
            knowledge_level = 7 + min(2, len(faction_affiliations))
            
            await self.lore_manager.add_lore_knowledge(
                "npc", npc_id,
                "Factions", faction_id,
                knowledge_level=knowledge_level
            )
            
            # Add to tracking
            faction_data = await self.lore_manager.get_lore_by_id("Factions", faction_id)
            if faction_data:
                knowledge_granted["factions"].append({
                    "id": faction_id,
                    "name": faction_data.get("name"),
                    "knowledge_level": knowledge_level
                })
        
        # 2. Give cultural knowledge based on background
        cultural_elements = await self.lore_manager.get_relevant_lore(
            cultural_background,
            lore_types=["CulturalElements"],
            min_relevance=0.6,
            limit=10
        )
        
        for element in cultural_elements:
            # Knowledge level 5-8 for cultural elements
            practiced_by = element.get("practiced_by", [])
            # Higher knowledge if it's practiced by their culture or factions
            knowledge_level = 5
            
            if any(affiliation in practiced_by for affiliation in faction_affiliations):
                knowledge_level += 2
            
            if cultural_background in practiced_by:
                knowledge_level += 1
            
            await self.lore_manager.add_lore_knowledge(
                "npc", npc_id,
                "CulturalElements", element["id"],
                knowledge_level=knowledge_level
            )
            
            # Add to tracking
            knowledge_granted["cultural_elements"].append({
                "id": element["id"],
                "name": element.get("name"),
                "knowledge_level": knowledge_level
            })
        
        # 3. Give knowledge of world lore relevant to their background
        background_query = f"{cultural_background} {' '.join(faction_affiliations)}"
        world_lore = await self.lore_manager.get_relevant_lore(
            background_query,
            lore_types=["WorldLore"],
            min_relevance=0.5,
            limit=5
        )
        
        for lore in world_lore:
            # Knowledge level 3-6 for general world lore
            # Base knowledge level depends on lore significance
            significance = lore.get("significance", 5)
            knowledge_level = min(3 + (significance // 3), 6)
            
            await self.lore_manager.add_lore_knowledge(
                "npc", npc_id,
                "WorldLore", lore["id"],
                knowledge_level=knowledge_level
            )
            
            # Add to tracking
            knowledge_granted["world_lore"].append({
                "id": lore["id"],
                "name": lore.get("name"),
                "category": lore.get("category"),
                "knowledge_level": knowledge_level
            })
        
        # 4. Give knowledge of historical events related to their factions
        for faction_id in faction_ids:
            # Get historical events involving this faction
            async with self.lore_manager.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    events = await conn.fetch("""
                        SELECT id, name, significance FROM HistoricalEvents
                        WHERE $1 = ANY(participating_factions)
                    """, str(faction_id))  # Note: may need to adjust based on how faction IDs are stored
                    
                    for event in events:
                        # Knowledge level based on significance
                        significance = event["significance"]
                        knowledge_level = min(4 + (significance // 2), 8)
                        
                        await self.lore_manager.add_lore_knowledge(
                            "npc", npc_id,
                            "HistoricalEvents", event["id"],
                            knowledge_level=knowledge_level
                        )
                        
                        # Add to tracking
                        knowledge_granted["historical_events"].append({
                            "id": event["id"],
                            "name": event["name"],
                            "knowledge_level": knowledge_level
                        })
        
        # Create memories for the most significant knowledge
        await self._create_lore_memories(npc_id, knowledge_granted)
        
        return knowledge_granted
    
    @with_governance(
        agent_type=AgentType.NPC,
        action_type="get_lore_relevant_to_npc_decision",
        action_description="Getting relevant lore for NPC {npc_id} decision making",
        id_from_context=lambda ctx: f"npc_{ctx.npc_id}"
    )
    async def get_lore_relevant_to_npc_decision(self, ctx, npc_id: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get lore that's relevant to an NPC's current decision-making context with Nyx governance oversight.
        
        Args:
            npc_id: ID of the NPC
            context: Current decision context
            
        Returns:
            Dictionary of relevant lore
        """
        # Extract context information
        location = context.get("location", "")
        player_input = context.get("player_input", "")
        mentioned_npcs = context.get("mentioned_npcs", [])
        
        # Combine into a search query
        query_text = f"{location} {player_input} {' '.join(mentioned_npcs)}"
        
        # Get lore known by this NPC and relevant to current context
        relevant_lore = await self.lore_manager.generate_available_lore_for_context(
            query_text,
            entity_type="npc",
            entity_id=npc_id,
            limit=5
        )
        
        # Organize by type
        organized_lore = {
            "faction_lore": [],
            "cultural_lore": [],
            "historical_lore": [],
            "world_lore": [],
            "location_lore": []
        }
        
        for lore in relevant_lore:
            lore_type = lore["lore_type"]
            
            if lore_type == "Factions":
                organized_lore["faction_lore"].append(lore)
            elif lore_type == "CulturalElements":
                organized_lore["cultural_lore"].append(lore)
            elif lore_type == "HistoricalEvents":
                organized_lore["historical_lore"].append(lore)
            elif lore_type == "WorldLore":
                organized_lore["world_lore"].append(lore)
            elif lore_type == "LocationLore":
                organized_lore["location_lore"].append(lore)
        
        # If we're in a specific location, check for location-specific lore
        if location:
            # Get location ID
            async with self.lore_manager.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    location_id = await conn.fetchval("""
                        SELECT id FROM Locations
                        WHERE location_name = $1 AND user_id = $2 AND conversation_id = $3
                    """, location, self.user_id, self.conversation_id)
                    
                    if location_id:
                        # Get this location's lore
                        location_with_lore = await self.lore_manager.get_location_with_lore(location_id)
                        
                        if location_with_lore:
                            # Check if NPC knows about this location's lore
                            knowledge_level = 0
                            
                            knowledge = await conn.fetchval("""
                                SELECT knowledge_level FROM LoreKnowledge
                                WHERE entity_type = 'npc' AND entity_id = $1
                                AND lore_type = 'LocationLore' AND lore_id = $2
                            """, npc_id, location_id)
                            
                            if knowledge:
                                knowledge_level = knowledge
                            
                            # Add location lore based on knowledge level
                            if knowledge_level > 0:
                                # Adjust what's shared based on knowledge level
                                if "lore_founding_story" in location_with_lore and knowledge_level >= 3:
                                    organized_lore["location_lore"].append({
                                        "type": "founding_story",
                                        "content": location_with_lore["lore_founding_story"],
                                        "knowledge_level": knowledge_level
                                    })
                                
                                if "lore_local_legends" in location_with_lore and knowledge_level >= 2:
                                    organized_lore["location_lore"].append({
                                        "type": "local_legends",
                                        "content": location_with_lore["lore_local_legends"],
                                        "knowledge_level": knowledge_level
                                    })
                                
                                if "lore_historical_significance" in location_with_lore and knowledge_level >= 4:
                                    organized_lore["location_lore"].append({
                                        "type": "historical_significance",
                                        "content": location_with_lore["lore_historical_significance"],
                                        "knowledge_level": knowledge_level
                                    })
        
        return organized_lore
    
    @with_governance(
        agent_type=AgentType.NPC,
        action_type="process_npc_lore_interaction",
        action_description="Processing lore interaction for NPC {npc_id}",
        id_from_context=lambda ctx: f"npc_{ctx.npc_id}"
    )
    async def process_npc_lore_interaction(self, ctx, npc_id: int, player_input: str) -> Dict[str, Any]:
        """
        Process a potential lore interaction between the player and an NPC with Nyx governance oversight.
        Determines if the player is asking about lore and what the NPC knows.
        
        Args:
            npc_id: ID of the NPC
            player_input: The player's input text
            
        Returns:
            Lore response information if relevant
        """
        # Check if the player is asking about lore/knowledge
        lore_keywords = [
            "tell me about", "what do you know about", "what's the history of",
            "who are", "what are", "history", "tell me the story of", "legend",
            "myth", "how was", "founded", "created", "origin", "where did", "why do"
        ]
        
        is_lore_question = any(keyword in player_input.lower() for keyword in lore_keywords)
        
        if not is_lore_question:
            # Not a lore-focused interaction
            return {"is_lore_interaction": False}
        
        # Determine what the player is asking about
        relevant_lore = await self.lore_manager.get_relevant_lore(
            player_input,
            min_relevance=0.6,
            limit=3
        )
        
        if not relevant_lore:
            # No matching lore found
            return {
                "is_lore_interaction": True,
                "has_knowledge": False,
                "response_type": "no_knowledge",
                "message": "I don't know much about that, I'm afraid."
            }
        
        # Check if the NPC knows about this lore
        top_lore = relevant_lore[0]
        lore_type = top_lore["lore_type"]
        lore_id = top_lore["id"]
        
        # Check NPC's knowledge level
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                knowledge = await conn.fetchrow("""
                    SELECT knowledge_level, is_secret FROM LoreKnowledge
                    WHERE entity_type = 'npc' AND entity_id = $1
                    AND lore_type = $2 AND lore_id = $3
                """, npc_id, lore_type, lore_id)
        
        if not knowledge:
            # NPC doesn't know about this
            return {
                "is_lore_interaction": True,
                "has_knowledge": False,
                "response_type": "no_knowledge",
                "lore_type": lore_type,
                "lore_name": top_lore.get("name", "that topic"),
                "message": f"I don't know anything about {top_lore.get('name', 'that topic')}."
            }
        
        # NPC knows about this - formulate a response based on knowledge level
        knowledge_level = knowledge["knowledge_level"]
        is_secret = knowledge["is_secret"]
        
        # Get NPC's personality to influence response style
        npc_personality = await self._get_npc_personality(npc_id)
        
        # Generate response based on knowledge level and personality
        response = await self._generate_lore_response(
            top_lore, knowledge_level, is_secret, npc_personality
        )
        
        # Check if sharing this lore should create a discovery for the player
        should_grant_player_knowledge = not is_secret and knowledge_level >= 3
        
        result = {
            "is_lore_interaction": True,
            "has_knowledge": True,
            "knowledge_level": knowledge_level,
            "is_secret": is_secret,
            "lore_type": lore_type,
            "lore_id": lore_id,
            "lore_name": top_lore.get("name", "that topic"),
            "response": response,
            "should_grant_player_knowledge": should_grant_player_knowledge
        }
        
        if should_grant_player_knowledge:
            # Calculate player knowledge level (slightly less than NPC's)
            player_knowledge_level = max(2, knowledge_level - 2)
            
            result["player_knowledge_level"] = player_knowledge_level
            
            # The calling code should handle the actual granting of knowledge
            # This just indicates that it should happen
        
        return result
    
    @with_governance(
        agent_type=AgentType.NPC,
        action_type="process_npc_lore_discovery",
        action_description="Processing potential lore discoveries for NPC {npc_id}",
        id_from_context=lambda ctx: f"npc_{ctx.npc_id}"
    )
    async def process_npc_lore_discovery(self, ctx, npc_id: int, location_id: int = None) -> Dict[str, Any]:
        """
        Process potential lore discoveries based on NPC location with Nyx governance oversight.
        
        Args:
            npc_id: ID of the NPC
            location_id: Optional location ID
            
        Returns:
            Discovery results
        """
        # If no location provided, get NPC's current location
        if location_id is None:
            async with self.lore_manager.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    # Get NPC's current location
                    location_name = await conn.fetchval("""
                        SELECT current_location FROM NPCStats
                        WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                    """, npc_id, self.user_id, self.conversation_id)
                    
                    if location_name:
                        # Get location ID
                        location_id = await conn.fetchval("""
                            SELECT id FROM Locations
                            WHERE location_name = $1 AND user_id = $2 AND conversation_id = $3
                        """, location_name, self.user_id, self.conversation_id)
        
        if not location_id:
            return {"discoveries": []}
        
        # Check for lore discovery opportunities at this location
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                opportunities = await conn.fetch("""
                    SELECT * FROM LoreDiscoveryOpportunities
                    WHERE location_id = $1 AND NOT discovered
                """, location_id)
                
                if not opportunities:
                    return {"discoveries": []}
                
                # Process each opportunity
                discoveries = []
                
                for opp in opportunities:
                    # Check difficulty against NPC traits
                    difficulty = opp["difficulty"]
                    
                    # Get NPC's relevant traits
                    npc_data = await conn.fetchrow("""
                        SELECT dominance, personality_traits FROM NPCStats
                        WHERE npc_id = $1
                    """, npc_id)
                    
                    if not npc_data:
                        continue
                    
                    # Process traits
                    traits = npc_data["personality_traits"]
                    if isinstance(traits, str):
                        try:
                            import json
                            traits = json.loads(traits)
                        except:
                            traits = []
                    
                    # Check for traits that help with discovery
                    discovery_traits = ["observant", "curious", "inquisitive", "scholarly"]
                    trait_bonus = sum(2 for trait in discovery_traits if trait in traits)
                    
                    # Calculate discovery chance
                    dominance = npc_data["dominance"] or 50
                    discovery_score = dominance // 10 + trait_bonus
                    
                    # Higher score makes easier discovery
                    if discovery_score >= difficulty:
                        # Discovery successful!
                        
                        # Get the lore
                        lore_data = await self.lore_manager.get_lore_by_id(
                            opp["lore_type"], opp["lore_id"]
                        )
                        
                        if lore_data:
                            # Grant knowledge to the NPC
                            knowledge_level = min(7, 10 - difficulty + trait_bonus)
                            
                            await self.lore_manager.add_lore_knowledge(
                                "npc", npc_id,
                                opp["lore_type"], opp["lore_id"],
                                knowledge_level=knowledge_level
                            )
                            
                            # Mark as discovered
                            await conn.execute("""
                                UPDATE LoreDiscoveryOpportunities
                                SET discovered = TRUE
                                WHERE id = $1
                            """, opp["id"])
                            
                            # Add to discoveries list
                            discoveries.append({
                                "lore_type": opp["lore_type"],
                                "lore_id": opp["lore_id"],
                                "lore_name": lore_data.get("name", "Unknown"),
                                "knowledge_level": knowledge_level,
                                "discovery_method": opp["discovery_method"]
                            })
                            
                            # Create memory for the NPC
                            await self._create_discovery_memory(
                                npc_id, lore_data, knowledge_level, opp["discovery_method"]
                            )
                
                return {"discoveries": discoveries}
    
    async def _create_lore_memories(self, npc_id: int, knowledge_granted: Dict[str, List[Dict[str, Any]]]):
        """
        Create memories for the most significant lore knowledge.
        
        Args:
            npc_id: ID of the NPC
            knowledge_granted: Dictionary of granted knowledge
        """
        # Get memory system for this NPC
        memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        
        # Get belief system for this NPC
        belief_system = self.belief_integration.get_belief_system_for_npc(npc_id)
        await belief_system.initialize()
        
        # Create memories for faction knowledge
        for faction in knowledge_granted["factions"]:
            if faction["knowledge_level"] >= 7:
                # Fetch complete faction data
                faction_data = await self.lore_manager.get_lore_by_id("Factions", faction["id"])
                
                if faction_data:
                    # Create memory for faction affiliation
                    memory_text = f"I am affiliated with {faction_data['name']}, a {faction_data['type']} faction. "
                    
                    # Add factional values and goals if knowledge is high
                    if faction["knowledge_level"] >= 8:
                        values = faction_data.get("values", [])
                        goals = faction_data.get("goals", [])
                        
                        if values:
                            memory_text += f"We value {', '.join(values[:3])}. "
                        
                        if goals:
                            memory_text += f"Our goals include {', '.join(goals[:2])}."
                    
                    # Store the memory
                    await memory_system.remember(
                        entity_type="npc",
                        entity_id=npc_id,
                        memory_text=memory_text,
                        importance="high",
                        tags=["lore", "faction", "identity"]
                    )
                    
                    # Form a belief about the faction (high factuality since it's their own faction)
                    await belief_system.form_subjective_belief_from_observation(
                        observation=f"{faction_data['name']} is a faction that represents my interests and values.",
                        factuality=0.9
                    )
        
        # Create memories for cultural knowledge
        significant_culture = [c for c in knowledge_granted["cultural_elements"] if c["knowledge_level"] >= 6]
        if significant_culture:
            # Only create memories for the most significant cultural elements
            for culture in significant_culture[:3]:
                culture_data = await self.lore_manager.get_lore_by_id("CulturalElements", culture["id"])
                
                if culture_data:
                    memory_text = f"I observe the {culture_data['name']}, which is a {culture_data['type']}. "
                    memory_text += culture_data.get("description", "")[:100]
                    
                    # Store the memory
                    await memory_system.remember(
                        entity_type="npc",
                        entity_id=npc_id,
                        memory_text=memory_text,
                        importance="medium",
                        tags=["lore", "culture", "beliefs"]
                    )
                    
                    # Form a cultural belief (medium-high factuality)
                    if culture_data.get("type") in ["religion", "faith", "belief system"]:
                        await belief_system.form_culturally_influenced_belief(
                            subject="values"
                        )
                    else:
                        await belief_system.form_culturally_influenced_belief(
                            subject="society"
                        )
        
        # Create memories for significant historical events
        significant_events = [e for e in knowledge_granted["historical_events"] if e["knowledge_level"] >= 7]
        if significant_events:
            # Only create memories for the most significant events
            for event in significant_events[:2]:
                event_data = await self.lore_manager.get_lore_by_id("HistoricalEvents", event["id"])
                
                if event_data:
                    memory_text = f"I remember the {event_data['name']} which happened {event_data.get('date_description', 'in the past')}. "
                    memory_text += event_data.get("description", "")[:100]
                    
                    # Store the memory
                    await memory_system.remember(
                        entity_type="npc",
                        entity_id=npc_id,
                        memory_text=memory_text,
                        importance="medium",
                        tags=["lore", "history", "event"]
                    )
                    
                    # Form a belief about the historical event (with medium factuality)
                    # Historical interpretations often have bias
                    await belief_system.form_subjective_belief_from_observation(
                        observation=f"The {event_data['name']} had a significant impact on our history.",
                        factuality=0.7  # Allow for some personal interpretation
                    )
    
    async def _create_discovery_memory(self, npc_id: int, lore_data: Dict[str, Any],
                                     knowledge_level: int, discovery_method: str):
        """
        Create a memory for the NPC about discovering lore.
        Also forms subjective beliefs based on the discovery.
        
        Args:
            npc_id: ID of the NPC
            lore_data: The discovered lore
            knowledge_level: Level of knowledge gained
            discovery_method: How it was discovered
        """
        # Get memory system
        memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        
        # Get belief system for this NPC
        belief_system = self.belief_integration.get_belief_system_for_npc(npc_id)
        await belief_system.initialize()
        
        # Create discovery memory
        lore_name = lore_data.get("name", "something")
        lore_type = lore_data["lore_type"]
        
        # Format the memory based on discovery method
        if discovery_method == "observation":
            memory_text = f"I noticed {lore_name} while observing my surroundings."
        elif discovery_method == "conversation":
            memory_text = f"I learned about {lore_name} during a conversation."
        elif discovery_method == "investigation":
            memory_text = f"I discovered information about {lore_name} by investigating."
        elif discovery_method == "book":
            memory_text = f"I read about {lore_name} in a book."
        else:
            memory_text = f"I learned about {lore_name}."
        
        # Add details based on knowledge level
        if knowledge_level >= 5:
            # Add a brief description for higher knowledge levels
            description = lore_data.get("description", "")
            if description and len(description) > 150:
                description = description[:150] + "..."
            memory_text += f" {description}"
        
        # Create the memory
        await memory_system.remember(
            entity_type="npc",
            entity_id=npc_id,
            memory_text=memory_text,
            importance="medium" if knowledge_level >= 6 else "low",
            tags=["lore", "discovery", lore_type.lower()]
        )
        
        # Now form subjective belief about this discovery
        # Factuality depends on knowledge level and discovery method
        base_factuality = min(knowledge_level / 10.0, 0.9)  # Max 0.9 factuality
        
        # Adjust factuality based on discovery method
        method_factuality_map = {
            "observation": 0.8,  # Direct observation is fairly reliable
            "investigation": 0.7,  # Investigation can have bias
            "conversation": 0.6,  # Conversations might contain misinformation
            "book": 0.75,        # Books are somewhat reliable but may have bias
            "rumor": 0.4,        # Rumors are often unreliable
            "default": 0.6       # Default for unknown methods
        }
        
        method_factuality = method_factuality_map.get(discovery_method, method_factuality_map["default"])
        
        # Combine knowledge level and method reliability
        factuality = (base_factuality * 0.7) + (method_factuality * 0.3)
        
        # Create belief from the discovery with calculated factuality
        lore_description = lore_data.get("description", f"Information about {lore_name}")
        
        # Trim description to keep observation concise
        if len(lore_description) > 200:
            lore_description = lore_description[:200] + "..."
        
        await belief_system.form_subjective_belief_from_observation(
            observation=lore_description,
            factuality=factuality
        )
        
        # For significant discoveries, also update any existing related beliefs
        if knowledge_level >= 6:
            await self.belief_integration.update_beliefs_on_knowledge_discovery(
                npc_id=npc_id,
                knowledge_type=lore_type,
                knowledge_id=lore_data["id"]
            )
        
        # For very significant discoveries (knowledge_level >= 8), form a narrative
        # that connects this discovery to other memories
        if knowledge_level >= 8:
            await self.belief_integration.form_narrative_from_recent_events(
                npc_id=npc_id,
                max_events=4  # Include a few recent memories
            )
    
    async def _get_npc_personality(self, npc_id: int) -> Dict[str, Any]:
        """
        Get NPC personality traits and other relevant attributes.
        Uses cached data when possible for improved performance.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Dictionary with personality information
        """
        try:
            # Create a simple cache key
            cache_key = f"npc_personality:{npc_id}:{self.user_id}:{self.conversation_id}"
            
            # Check if there's an in-memory personality cache in the class
            if not hasattr(self.__class__, '_personality_cache'):
                self.__class__._personality_cache = {}
            
            # Check for cached data (with 5-minute expiration)
            if cache_key in self.__class__._personality_cache:
                cached_data, timestamp = self.__class__._personality_cache[cache_key]
                from datetime import datetime
                if (datetime.now() - timestamp).total_seconds() < 300:  # 5 minutes TTL
                    return cached_data
            
            # Get NPC data using the more efficient NPCDataAccess
            from data.npc_dal import NPCDataAccess
            
            npc_data = await NPCDataAccess.get_npc_details(
                npc_id=npc_id,
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            if not npc_data:
                return {"traits": {}, "dominance": 50, "archetypes": [], "alignment": "neutral"}
            
            # Extract personality
            personality = npc_data.get("personality", {})
            
            # Parse the personality if it's a string
            if isinstance(personality, str):
                try:
                    personality = json.loads(personality)
                except json.JSONDecodeError:
                    logging.warning(f"Failed to parse personality JSON for NPC {npc_id}")
                    personality = {}
            
            # Ensure personality format is consistent
            if not isinstance(personality, dict):
                personality = {}
            
            # Create structured personality data
            result = {
                "traits": {},
                "dominance": npc_data.get("dominance", 50),
                "archetypes": [],
                "alignment": "neutral"
            }
            
            # Extract traits with their values
            traits = personality.get("traits", {})
            if isinstance(traits, dict):
                result["traits"] = traits
            elif isinstance(traits, list):
                # Convert list format to dict format
                for trait in traits:
                    if isinstance(trait, dict) and "name" in trait and "value" in trait:
                        result["traits"][trait["name"]] = trait["value"]
                    elif isinstance(trait, str):
                        # Assume default strength if only trait name provided
                        result["traits"][trait] = 7
            
            # Get archetypes from archetype field or fallback to backstory
            archetypes = npc_data.get("archetypes", [])
            if isinstance(archetypes, list):
                result["archetypes"] = archetypes[:3]  # Limit to top 3
            elif isinstance(archetypes, dict) and "primary" in archetypes:
                # Handle structured archetype format
                result["archetypes"] = [archetypes["primary"]]
                if "secondary" in archetypes:
                    result["archetypes"].append(archetypes["secondary"])
                
            # Get alignment
            psychological_profile = npc_data.get("psychological_profile", {})
            if isinstance(psychological_profile, dict):
                result["alignment"] = psychological_profile.get("alignment", "neutral")
                
                # Add additional temperament data if available
                if "temperament" in psychological_profile:
                    result["temperament"] = psychological_profile["temperament"]
            
            # Cache the result with timestamp
            from datetime import datetime
            self.__class__._personality_cache[cache_key] = (result, datetime.now())
            
            return result
        except Exception as e:
            logging.error(f"Error getting NPC personality: {e}", exc_info=True)
            return {"traits": {}, "dominance": 50, "archetypes": [], "alignment": "neutral"}
    
    async def _generate_lore_response(self, lore: Dict[str, Any], knowledge_level: int,
                                    is_secret: bool, personality: Dict[str, Any]) -> str:
        """
        Generate an appropriate lore response based on knowledge level and personality.
        
        Args:
            lore: Lore data
            knowledge_level: NPC's knowledge level (1-10)
            is_secret: Whether this is secret knowledge
            personality: NPC's personality data
            
        Returns:
            Formatted response text
        """
        # Base response components
        name = lore.get("name", "that")
        description = lore.get("description", "")
        lore_type = lore["lore_type"]
        
        # Truncate description based on knowledge level
        if knowledge_level <= 3:
            if len(description) > 100:
                description = description[:100] + "... that's about all I know."
        elif knowledge_level <= 6:
            if len(description) > 200:
                description = description[:200] + "... that's what I know about it."
        
        # Add personality styling
        traits = personality.get("traits", [])
        dominance = personality.get("dominance", 50)
        cruelty = personality.get("cruelty", 50)
        
        # Determine tone based on personality
        tone = "neutral"
        if "arrogant" in traits or "boastful" in traits or dominance > 70:
            tone = "confident"
        elif "shy" in traits or "timid" in traits or dominance < 30:
            tone = "hesitant"
        elif "scholarly" in traits or "intellectual" in traits:
            tone = "scholarly"
        elif "secretive" in traits or is_secret:
            tone = "secretive"
        
        # Format intro based on tone and knowledge level
        intro = ""
        if tone == "confident":
            intro = f"Of course I know about {name}. "
            if knowledge_level > 7:
                intro = f"I know everything worth knowing about {name}. "
        elif tone == "hesitant":
            intro = f"I think I know a bit about {name}... "
            if knowledge_level < 5:
                intro = f"I'm not sure, but I believe {name} is... "
        elif tone == "scholarly":
            intro = f"According to what I've studied, {name} "
            if knowledge_level > 7:
                intro = f"Having extensively researched {name}, I can tell you that "
        elif tone == "secretive":
            if is_secret:
                intro = f"I shouldn't really talk about {name}, but... "
            else:
                intro = f"Not many know this about {name}, but "
        else:  # neutral
            intro = f"About {name}? "
            if knowledge_level > 6:
                intro = f"I know quite a bit about {name}. "
        
        # Add specific details based on lore type
        if lore_type == "Factions":
            faction_type = lore.get("type", "group")
            values = lore.get("values", [])
            goals = lore.get("goals", [])
            
            faction_details = f"They're a {faction_type}. "
            
            if values and knowledge_level >= 5:
                faction_details += f"They value {', '.join(values[:2])}. "
            
            if goals and knowledge_level >= 6:
                faction_details += f"Their goals include {', '.join(goals[:1])}. "
            
            return intro + faction_details + description
        
        elif lore_type == "CulturalElements":
            element_type = lore.get("type", "tradition")
            practiced_by = lore.get("practiced_by", [])
            
            cultural_details = f"It's a {element_type}. "
            
            if practiced_by and knowledge_level >= 4:
                if len(practiced_by) > 1:
                    cultural_details += f"It's practiced by {', '.join(practiced_by[:2])}. "
                else:
                    cultural_details += f"It's practiced by {practiced_by[0]}. "
            
            return intro + cultural_details + description
        
        elif lore_type == "HistoricalEvents":
            date = lore.get("date_description", "sometime in the past")
            consequences = lore.get("consequences", [])
            
            event_details = f"It happened {date}. "
            
            if consequences and knowledge_level >= 5:
                event_details += f"It led to {consequences[0]}. "
            
            return intro + event_details + description
        
        # Default for other lore types
        return intro + description
    
    @with_governance(
        agent_type=AgentType.NPC,
        action_type="get_npc_speech_patterns",
        action_description="Getting speech patterns for NPC {npc_id}",
        id_from_context=lambda ctx: f"npc_{ctx.npc_id}"
    )
    async def get_npc_speech_patterns(self, ctx, npc_id: int) -> Dict[str, Any]:
        """
        Get speech patterns for an NPC based on their cultural background, dialects, and language knowledge.
        This is used to style NPC dialogue with appropriate dialectal features.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Dictionary with speech pattern information including:
            - primary_language: Main language spoken by the NPC
            - dialect: Information about the NPC's dialect
            - dialect_features: Specific features of the dialect (accent, vocabulary, grammar, etc.)
            - formality_level: The NPC's typical level of formality
            - cultural_expressions: Cultural-specific phrases or expressions
        """
        try:
            # Get the LoreSystem instance
            from lore.lore_system import LoreSystem
            lore_system = LoreSystem.get_instance(self.user_id, self.conversation_id)
            await lore_system.initialize()
            
            # Get the NPC's complete cultural attributes
            cultural_attrs = await lore_system.get_npc_cultural_attributes(npc_id)
            
            if not cultural_attrs or "error" in cultural_attrs:
                return {
                    "primary_language": None,
                    "dialect": None,
                    "dialect_features": {},
                    "formality_level": "neutral",
                    "cultural_expressions": []
                }
            
            # Get NPC details for personality traits
            npc_personality = await self._get_npc_personality(npc_id)
            
            # Initialize speech pattern data
            speech_patterns = {
                "primary_language": None,
                "dialect": None,
                "dialect_features": {},
                "formality_level": "neutral",  # Default formality
                "cultural_expressions": []
            }
            
            # Set primary language (if any languages are known)
            languages = cultural_attrs.get("languages", [])
            if languages:
                # Sort by fluency (highest first)
                languages.sort(key=lambda x: x.get("fluency", 0), reverse=True)
                primary_language = languages[0]
                
                speech_patterns["primary_language"] = {
                    "id": primary_language.get("id"),
                    "name": primary_language.get("name"),
                    "fluency": primary_language.get("fluency", 7),
                    "writing_system": primary_language.get("writing_system")
                }
                
                # Get formality levels from primary language
                formality_levels = primary_language.get("formality_levels", [])
                
                # Set dialect information if available
                if cultural_attrs.get("primary_dialect"):
                    speech_patterns["dialect"] = cultural_attrs["primary_dialect"]
                    speech_patterns["dialect_features"] = cultural_attrs.get("dialect_features", {})
                
                # Extract cultural expressions from common phrases if available
                common_phrases = primary_language.get("common_phrases", {})
                if common_phrases and isinstance(common_phrases, dict):
                    expressions = []
                    for category, phrases in common_phrases.items():
                        if isinstance(phrases, list):
                            expressions.extend(phrases[:3])  # Limit to 3 phrases per category
                    
                    speech_patterns["cultural_expressions"] = expressions
            
            # Determine formality level based on personality and cultural background
            personality_traits = npc_personality.get("personality_traits", [])
            
            # Default to neutral formality
            formality = "neutral"
            
            # Check personality traits for formality indicators
            formal_traits = ["proper", "pompous", "dignified", "aristocratic", "strict", "refined"]
            casual_traits = ["casual", "relaxed", "easygoing", "informal", "blunt", "direct"]
            
            if any(trait.lower() in formal_traits for trait in personality_traits):
                formality = "formal"
            elif any(trait.lower() in casual_traits for trait in personality_traits):
                formality = "casual"
            
            # Cultural norms can override personality
            cultural_norms = cultural_attrs.get("cultural_norms_followed", [])
            for norm in cultural_norms:
                if norm.get("category", "").lower() in ["speech", "communication", "greetings"]:
                    if "formal" in norm.get("description", "").lower():
                        formality = "formal"
                        break
                    elif "casual" in norm.get("description", "").lower() or "informal" in norm.get("description", "").lower():
                        formality = "casual"
                        break
            
            speech_patterns["formality_level"] = formality
            
            return speech_patterns
            
        except Exception as e:
            logging.error(f"Error getting NPC speech patterns: {e}")
            return {
                "primary_language": None,
                "dialect": None,
                "dialect_features": {},
                "formality_level": "neutral",
                "cultural_expressions": [],
                "error": str(e)
            }
    
    async def apply_dialect_to_text(self, ctx, npc_id: int, text: str, intensity: str = 'medium') -> str:
        """
        Apply dialect features to a text based on NPC's dialect.
        Uses optimized regex matching and caching for better performance.
        
        Args:
            ctx: Governance context
            npc_id: ID of the NPC
            text: Original text
            intensity: Intensity of dialect application ('light', 'medium', 'strong')
            
        Returns:
            Modified text with dialect features applied
        """
        if not text or not text.strip():
            return text
        
        try:
            # Get the LoreSystem instance
            from lore.lore_system import LoreSystem
            lore_system = LoreSystem.get_instance(self.user_id, self.conversation_id)
            await lore_system.initialize()
            
            # Create a simple cache key for speech patterns
            cache_key = f"npc_speech:{npc_id}:{self.user_id}:{self.conversation_id}"
            
            # Check if there's an in-memory speech pattern cache in the class
            if not hasattr(self.__class__, '_speech_pattern_cache'):
                self.__class__._speech_pattern_cache = {}
            
            # Get the dialect information to use
            speech_patterns = None
            
            # Check for cached speech pattern data (with 5-minute expiration)
            dialect_id = None
            if cache_key in self.__class__._speech_pattern_cache:
                cached_data, timestamp = self.__class__._speech_pattern_cache[cache_key]
                from datetime import datetime
                if (datetime.now() - timestamp).total_seconds() < 300:  # 5 minutes TTL
                    speech_patterns = cached_data
                    if speech_patterns and speech_patterns.get('dialect'):
                        dialect_id = speech_patterns['dialect'].get('id')
            
            # If not cached, get speech patterns and cache them
            if not speech_patterns:
                speech_patterns = await lore_system.get_npc_speech_patterns(npc_id)
                
                # Cache the speech patterns
                from datetime import datetime
                self.__class__._speech_pattern_cache[cache_key] = (speech_patterns, datetime.now())
                
                if speech_patterns and speech_patterns.get('primary_dialect'):
                    dialect_id = speech_patterns['primary_dialect'].get('id')
            
            # If no dialect found, return original text
            if not dialect_id:
                return text
            
            # Apply the dialect to the text with optimized regex in lore_system
            modified_text = await lore_system.apply_dialect_to_text(
                text=text,
                dialect_id=dialect_id,
                intensity=intensity,
                npc_id=npc_id
            )
            
            # Report action to governance
            await ctx.report_action(
                action="applied_dialect_to_text",
                details={
                    "npc_id": npc_id,
                    "dialect_id": dialect_id,
                    "intensity": intensity,
                    "text_length": len(text)
                }
            )
            
            return modified_text
        except Exception as e:
            logging.error(f"Error applying dialect to text: {e}", exc_info=True)
            return text  # Return original text on error
    
    async def register_with_nyx_governance(self):
        """Register with Nyx governance system."""
        await self.initialize_governance()
        
        # Register this integration with governance
        await self.governor.register_agent(
            agent_type=AgentType.NPC,
            agent_id=self.npc_id or "npc_lore_integration",
            agent_instance=self
        )
        
        logging.info(f"NPCLoreIntegration registered with Nyx governance for user {self.user_id}, conversation {self.conversation_id}")
    
    async def get_npc_faction_knowledge(self, npc_id: int) -> List[Dict[str, Any]]:
        """
        Get knowledge about factions that this NPC has.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            List of known factions with knowledge level
        """
        # Get all lore knowledge for this NPC
        all_knowledge = await self.lore_manager.get_entity_lore_knowledge("npc", npc_id)
        
        # Filter to just faction knowledge
        faction_knowledge = [
            k for k in all_knowledge
            if k.get("lore_type") == "Factions"
        ]
        
        return faction_knowledge
    
    async def process_npc_lore_discovery(self, npc_id: int, location_id: int = None) -> Dict[str, Any]:
        """
        Process potential lore discoveries based on NPC location.
        
        Args:
            npc_id: ID of the NPC
            location_id: Optional location ID
            
        Returns:
            Discovery results
        """
        # If no location provided, get NPC's current location
        if location_id is None:
            async with self.lore_manager.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    # Get NPC's current location
                    location_name = await conn.fetchval("""
                        SELECT current_location FROM NPCStats
                        WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                    """, npc_id, self.user_id, self.conversation_id)
                    
                    if location_name:
                        # Get location ID
                        location_id = await conn.fetchval("""
                            SELECT id FROM Locations
                            WHERE location_name = $1 AND user_id = $2 AND conversation_id = $3
                        """, location_name, self.user_id, self.conversation_id)
        
        if not location_id:
            return {"discoveries": []}
        
        # Check for lore discovery opportunities at this location
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                opportunities = await conn.fetch("""
                    SELECT * FROM LoreDiscoveryOpportunities
                    WHERE location_id = $1 AND NOT discovered
                """, location_id)
                
                if not opportunities:
                    return {"discoveries": []}
                
                # Process each opportunity
                discoveries = []
                
                for opp in opportunities:
                    # Check difficulty against NPC traits
                    difficulty = opp["difficulty"]
                    
                    # Get NPC's relevant traits
                    npc_data = await conn.fetchrow("""
                        SELECT dominance, personality_traits FROM NPCStats
                        WHERE npc_id = $1
                    """, npc_id)
                    
                    if not npc_data:
                        continue
                    
                    # Process traits
                    traits = npc_data["personality_traits"]
                    if isinstance(traits, str):
                        try:
                            import json
                            traits = json.loads(traits)
                        except:
                            traits = []
                    
                    # Check for traits that help with discovery
                    discovery_traits = ["observant", "curious", "inquisitive", "scholarly"]
                    trait_bonus = sum(2 for trait in discovery_traits if trait in traits)
                    
                    # Calculate discovery chance
                    dominance = npc_data["dominance"] or 50
                    discovery_score = dominance // 10 + trait_bonus
                    
                    # Higher score makes easier discovery
                    if discovery_score >= difficulty:
                        # Discovery successful!
                        
                        # Get the lore
                        lore_data = await self.lore_manager.get_lore_by_id(
                            opp["lore_type"], opp["lore_id"]
                        )
                        
                        if lore_data:
                            # Grant knowledge to the NPC
                            knowledge_level = min(7, 10 - difficulty + trait_bonus)
                            
                            await self.lore_manager.add_lore_knowledge(
                                "npc", npc_id,
                                opp["lore_type"], opp["lore_id"],
                                knowledge_level=knowledge_level
                            )
                            
                            # Mark as discovered
                            await conn.execute("""
                                UPDATE LoreDiscoveryOpportunities
                                SET discovered = TRUE
                                WHERE id = $1
                            """, opp["id"])
                            
                            # Add to discoveries list
                            discoveries.append({
                                "lore_type": opp["lore_type"],
                                "lore_id": opp["lore_id"],
                                "lore_name": lore_data.get("name", "Unknown"),
                                "knowledge_level": knowledge_level,
                                "discovery_method": opp["discovery_method"]
                            })
                            
                            # Create memory for the NPC
                            await self._create_discovery_memory(
                                npc_id, lore_data, knowledge_level, opp["discovery_method"]
                            )
                
                return {"discoveries": discoveries}


class DMAgentLoreIntegration:
    """
    Integrates lore into the DM agent (Nyx) behavior and decision-making.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the DM agent lore integration.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_manager = LoreManager(user_id, conversation_id)
    
    async def add_lore_to_dm_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add relevant lore to the DM agent's context for better responses.
        
        Args:
            context: Current context for the DM agent
            
        Returns:
            Enhanced context with lore
        """
        # Create a copy of the context to modify
        enhanced_context = context.copy()
        
        # Extract search terms from context
        location = context.get("location", "")
        player_input = context.get("player_input", "")
        current_npcs = context.get("current_npcs", [])
        
        # Build search query
        npc_names = [npc.get("npc_name", "") for npc in current_npcs]
        search_query = f"{location} {player_input} {' '.join(npc_names)}"
        
        # Get relevant lore for this context
        relevant_lore = await self.lore_manager.get_relevant_lore(
            search_query,
            min_relevance=0.65,
            limit=5
        )
        
        # Add lore to context in an organized way
        lore_by_type = {
            "WorldLore": [],
            "Factions": [],
            "CulturalElements": [],
            "HistoricalEvents": [],
            "GeographicRegions": [],
            "LocationLore": []
        }
        
        for lore in relevant_lore:
            lore_type = lore["lore_type"]
            if lore_type in lore_by_type:
                # Remove the embedding field if present
                if "embedding" in lore:
                    del lore["embedding"]
                lore_by_type[lore_type].append(lore)
        
        # Add specific location lore if we're in a known location
        if location:
            try:
                async with self.lore_manager.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        # Get location ID
                        location_id = await conn.fetchval("""
                            SELECT id FROM Locations
                            WHERE location_name = $1 AND user_id = $2 AND conversation_id = $3
                        """, location, self.user_id, self.conversation_id)
                        
                        if location_id:
                            # Get this location's lore
                            location_with_lore = await self.lore_manager.get_location_with_lore(location_id)
                            if location_with_lore:
                                lore_by_type["LocationLore"] = [location_with_lore]
            except Exception as e:
                logging.error(f"Error getting location lore: {e}")
        
        # Add to context
        enhanced_context["lore"] = lore_by_type
        
        # Add faction relationships if factions are relevant
        faction_lore = lore_by_type.get("Factions", [])
        if faction_lore:
            faction_relationships = {}
            for faction in faction_lore:
                try:
                    faction_id = faction["id"]
                    relationships = await self.lore_manager.get_faction_relationships(faction_id)
                    faction_relationships[faction["name"]] = relationships
                except Exception as e:
                    logging.error(f"Error getting faction relationships: {e}")
            
            enhanced_context["faction_relationships"] = faction_relationships
        
        # Add cultural context if relevant
        cultural_lore = lore_by_type.get("CulturalElements", [])
        if cultural_lore:
            enhanced_context["cultural_context"] = cultural_lore
        
        return enhanced_context
    
    async def check_for_lore_based_events(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if any lore-based events should trigger in the current context.
        
        Args:
            current_context: Current game context
            
        Returns:
            Event triggers if any should occur
        """
        # Get current location
        location = current_context.get("location", "")
        if not location:
            return {"events": []}
        
        # Check for NPCs present
        current_npcs = current_context.get("current_npcs", [])
        npc_ids = [npc.get("npc_id") for npc in current_npcs if "npc_id" in npc]
        
        # Get current time info
        time_of_day = current_context.get("time_of_day", "")
        day = current_context.get("day", "")
        
        # Events to return
        events = []
        
        # Check for location-triggered events
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get location ID
                location_id = await conn.fetchval("""
                    SELECT id FROM Locations
                    WHERE location_name = $1 AND user_id = $2 AND conversation_id = $3
                """, location, self.user_id, self.conversation_id)
                
                if location_id:
                    # Check for lore discovery opportunities at this location
                    opportunities = await conn.fetch("""
                        SELECT * FROM LoreDiscoveryOpportunities
                        WHERE location_id = $1 AND NOT discovered
                    """, location_id)
                    
                    for opp in opportunities:
                        # Get the lore details
                        lore_data = await self.lore_manager.get_lore_by_id(
                            opp["lore_type"], opp["lore_id"]
                        )
                        
                        if lore_data:
                            # Only trigger if relatively important
                            significance = lore_data.get("significance", 5)
                            
                            if significance >= 6:
                                events.append({
                                    "type": "lore_discovery_opportunity",
                                    "lore_type": opp["lore_type"],
                                    "lore_id": opp["lore_id"],
                                    "lore_name": lore_data.get("name", "Unknown"),
                                    "discovery_method": opp["discovery_method"],
                                    "difficulty": opp["difficulty"]
                                })
                
                # Check for faction presence/activity at this location/time
                if location_id:
                    # Get factions associated with this location
                    location_factions = await conn.fetch("""
                        SELECT id, name, type FROM Factions
                        WHERE id IN (
                            SELECT source_id FROM LoreConnections
                            WHERE target_type = 'LocationLore' AND target_id = $1
                            AND source_type = 'Factions'
                            AND connection_type = 'influences'
                        )
                    """, location_id)
                    
                    for faction in location_factions:
                        # Get connection details
                        connection = await conn.fetchrow("""
                            SELECT strength, description FROM LoreConnections
                            WHERE source_type = 'Factions' AND source_id = $1
                            AND target_type = 'LocationLore' AND target_id = $2
                            AND connection_type = 'influences'
                        """, faction["id"], location_id)
                        
                        if connection and connection["strength"] >= 7:
                            # Strong faction presence - might trigger an event
                            events.append({
                                "type": "faction_presence",
                                "faction_id": faction["id"],
                                "faction_name": faction["name"],
                                "faction_type": faction["type"],
                                "strength": connection["strength"],
                                "description": connection["description"]
                            })
                
                # Check for historical event commemorations/anniversaries
                if day:
                    # Get any events commemorated on this day
                    historical_events = await conn.fetch("""
                        SELECT id, name, date_description, description, commemorated_by
                        FROM HistoricalEvents
                        WHERE commemorated_by IS NOT NULL
                    """)
                    
                    for event in historical_events:
                        # Check if this event should trigger on this day
                        # This is simplified - you'd want a more sophisticated date matching system
                        commemorated_by = event["commemorated_by"]
                        
                        # Simple check - if day number is in the commemoration description
                        if day in commemorated_by:
                            events.append({
                                "type": "historical_commemoration",
                                "event_id": event["id"],
                                "event_name": event["name"],
                                "date_description": event["date_description"],
                                "description": event["description"],
                                "commemoration": commemorated_by
                            })
        
        return {"events": events}
    
    async def generate_lore_based_conflict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a conflict based on the game's lore.
        
        Args:
            context: Current game context
            
        Returns:
            Conflict data
        """
        # Extract context information
        location = context.get("location", "")
        
        # Get factions that might be involved
        relevant_factions = []
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get location-based factions
                if location:
                    # Get location ID
                    location_id = await conn.fetchval("""
                        SELECT id FROM Locations
                        WHERE location_name = $1 AND user_id = $2 AND conversation_id = $3
                    """, location, self.user_id, self.conversation_id)
                    
                    if location_id:
                        # Get factions that influence this location
                        location_factions = await conn.fetch("""
                            SELECT f.id, f.name, f.type, f.description
                            FROM Factions f
                            JOIN LoreConnections lc ON f.id = lc.source_id
                            WHERE lc.source_type = 'Factions'
                            AND lc.target_type = 'LocationLore' AND lc.target_id = $1
                            AND lc.connection_type = 'influences'
                        """, location_id)
                        
                        for faction in location_factions:
                            relevant_factions.append(dict(faction))
                
                # If we don't have enough factions, get some that have conflicts with each other
                if len(relevant_factions) < 2:
                    rival_factions = await conn.fetch("""
                        SELECT DISTINCT f.id, f.name, f.type, f.description
                        FROM Factions f
                        JOIN LoreConnections lc ON f.id = lc.source_id
                        WHERE lc.source_type = 'Factions'
                        AND lc.target_type = 'Factions'
                        AND lc.connection_type = 'conflicts_with'
                        LIMIT 5
                    """)
                    
                    for faction in rival_factions:
                        if not any(f["id"] == faction["id"] for f in relevant_factions):
                            relevant_factions.append(dict(faction))
        
        # Need at least two factions for a conflict
        if len(relevant_factions) < 2:
            return {"error": "Not enough factions for conflict generation"}
        
        # Select two factions that are most likely to be in conflict
        faction_pairs = []
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check for existing conflicts between factions
                for i, faction1 in enumerate(relevant_factions):
                    for faction2 in relevant_factions[i+1:]:
                        # Check if these factions have a conflict relationship
                        connection = await conn.fetchrow("""
                            SELECT strength, description FROM LoreConnections
                            WHERE (source_type = 'Factions' AND source_id = $1
                                AND target_type = 'Factions' AND target_id = $2)
                            OR (source_type = 'Factions' AND source_id = $2
                                AND target_type = 'Factions' AND target_id = $1)
                            AND connection_type = 'conflicts_with'
                        """, faction1["id"], faction2["id"])
                        
                        if connection:
                            # These factions have a conflict
                            faction_pairs.append({
                                "faction_a": faction1,
                                "faction_b": faction2,
                                "conflict_strength": connection["strength"],
                                "conflict_description": connection["description"]
                            })
                        else:
                            # No explicit conflict, but still potential rivals
                            faction_pairs.append({
                                "faction_a": faction1,
                                "faction_b": faction2,
                                "conflict_strength": 3,  # Low default strength
                                "conflict_description": "Potential disagreement"
                            })
        
        # If no pairs found, use the first two factions
        if not faction_pairs:
            faction_pairs = [{
                "faction_a": relevant_factions[0],
                "faction_b": relevant_factions[1],
                "conflict_strength": 3,
                "conflict_description": "Emerging tensions"
            }]
        
        # Sort by conflict strength to get the most intense conflicts first
        faction_pairs.sort(key=lambda x: x["conflict_strength"], reverse=True)
        
        # Choose the top pair
        chosen_pair = faction_pairs[0]
        
        # Generate conflict data
        conflict_type = "standard"
        if chosen_pair["conflict_strength"] >= 8:
            conflict_type = "major"
        elif chosen_pair["conflict_strength"] >= 5:
            conflict_type = "minor"
        
        faction_a = chosen_pair["faction_a"]
        faction_b = chosen_pair["faction_b"]
        
        # Create conflict data
        conflict_data = {
            "conflict_type": conflict_type,
            "faction_a_name": faction_a["name"],
            "faction_b_name": faction_b["name"],
            "description": f"Conflict between {faction_a['name']} and {faction_b['name']}. {chosen_pair['conflict_description']}",
            "brewing_description": f"Tensions are brewing between {faction_a['name']} and {faction_b['name']}.",
            "active_description": f"Open conflict has erupted between {faction_a['name']} and {faction_b['name']}.",
            "climax_description": f"The conflict between {faction_a['name']} and {faction_b['name']} has reached a critical point.",
            "resolution_description": f"The conflict between {faction_a['name']} and {faction_b['name']} has been resolved.",
            "estimated_duration": self._get_duration_for_conflict_type(conflict_type),
            "resources_required": self._get_resources_for_conflict_type(conflict_type),
            "lore_based": True
        }
        
        return conflict_data
    
    def _get_duration_for_conflict_type(self, conflict_type: str) -> int:
        """Get estimated duration based on conflict type."""
        import random
        
        if conflict_type == "major":
            return random.randint(14, 21)  # 2-3 weeks
        elif conflict_type == "minor":
            return random.randint(4, 10)   # 4-10 days
        else:  # standard
            return random.randint(2, 5)    # 2-5 days
    
    def _get_resources_for_conflict_type(self, conflict_type: str) -> Dict[str, int]:
        """Get required resources based on conflict type."""
        if conflict_type == "major":
            return {"money": 500, "supplies": 15, "influence": 50}
        elif conflict_type == "minor":
            return {"money": 200, "supplies": 8, "influence": 25}
        else:  # standard
            return {"money": 100, "supplies": 5, "influence": 10}
    
    async def get_associated_npcs_for_lore(self, lore_type: str, lore_id: int) -> List[Dict[str, Any]]:
        """
        Get NPCs associated with a piece of lore.
        
        Args:
            lore_type: Type of lore
            lore_id: ID of lore
            
        Returns:
            List of associated NPCs with knowledge level
        """
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get NPCs that know about this lore
                knowledge_rows = await conn.fetch("""
                    SELECT lk.entity_id as npc_id, lk.knowledge_level, ns.npc_name
                    FROM LoreKnowledge lk
                    JOIN NPCStats ns ON lk.entity_id = ns.npc_id
                    WHERE lk.entity_type = 'npc'
                    AND lk.lore_type = $1 AND lk.lore_id = $2
                    AND ns.user_id = $3 AND ns.conversation_id = $4
                    ORDER BY lk.knowledge_level DESC
                """, lore_type, lore_id, self.user_id, self.conversation_id)
                
                return [dict(row) for row in knowledge_rows]
    
    async def format_lore_for_dm_prompt(self) -> str:
        """
        Format key lore for inclusion in the DM agent's system prompt.
        
        Returns:
            Formatted lore text for the prompt
        """
        lore_sections = []
        
        # Add cosmology/world basics
        world_lore = await self.lore_manager.get_world_lore_by_category("cosmology")
        if world_lore:
            lore_sections.append("# World Cosmology")
            for lore in world_lore[:2]:  # Limit to avoid prompt bloat
                lore_sections.append(f"## {lore.get('name', 'Unknown')}")
                lore_sections.append(lore.get('description', ''))
        
        # Add major factions
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get major factions
                factions = await conn.fetch("""
                    SELECT id, name, type, description
                    FROM Factions
                    ORDER BY id
                    LIMIT 5
                """)
                
                if factions:
                    lore_sections.append("# Major Factions")
                    for faction in factions:
                        lore_sections.append(f"## {faction['name']} ({faction['type']})")
                        lore_sections.append(faction['description'])
                
                # Add cultural elements
                cultures = await conn.fetch("""
                    SELECT id, name, type, description
                    FROM CulturalElements
                    ORDER BY significance DESC
                    LIMIT 3
                """)
                
                if cultures:
                    lore_sections.append("# Cultural Elements")
                    for culture in cultures:
                        lore_sections.append(f"## {culture['name']} ({culture['type']})")
                        lore_sections.append(culture['description'])
                
                # Add major historical events
                events = await conn.fetch("""
                    SELECT id, name, date_description, description
                    FROM HistoricalEvents
                    ORDER BY significance DESC
                    LIMIT 3
                """)
                
                if events:
                    lore_sections.append("# Historical Events")
                    for event in events:
                        lore_sections.append(f"## {event['name']} ({event['date_description']})")
                        lore_sections.append(event['description'])
        
        return "\n\n".join(lore_sections)
