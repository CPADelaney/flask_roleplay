# lore/lore_integration.py

import logging
import asyncio
from typing import Dict, List, Any, Optional
from agents import trace

# Nyx governance integration
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance_permission, with_action_reporting, with_governance

from lore.dynamic_lore_generator import DynamicLoreGenerator
from lore.lore_manager import LoreManager
from lore.npc_lore_integration import NPCLoreIntegration
from logic.conflict_system.conflict_integration import ConflictSystemIntegration
from db.connection import get_db_connection

class LoreIntegrationSystem:
    """
    Integrates lore with other game systems with full Nyx governance oversight.
    
    Systems includes:
    - NPC knowledge and behavior
    - Quests and narrative
    - Conflicts and faction dynamics
    - Environment and location descriptions
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the lore integration system."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_manager = LoreManager(user_id, conversation_id)
        self.npc_lore = NPCLoreIntegration(user_id, conversation_id)
        self.lore_generator = DynamicLoreGenerator(user_id, conversation_id)
        self.governor = None
        
    async def initialize_governance(self):
        """Initialize Nyx governance connection"""
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
        return self.governor

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="initialize_game_lore",
        action_description="Initializing comprehensive lore for environment: {environment_desc}",
        id_from_context=lambda ctx: "lore_integration"
    )
    async def initialize_game_lore(self, environment_desc: str) -> Dict[str, Any]:
        """
        Initialize comprehensive lore for a new game with Nyx governance oversight.
        
        Args:
            environment_desc: Description of the game environment
            
        Returns:
            Dict of all generated lore
        """
        # Create trace for monitoring
        with trace(
            workflow_name="Lore Initialization",
            trace_id=f"lore-init-{self.conversation_id}-{int(datetime.now().timestamp())}",
            group_id=f"user-{self.user_id}"
        ):
            # Generate complete interconnected lore
            lore = await self.lore_generator.generate_complete_lore(environment_desc)
            
            # Store the lore in compact form for narrative context
            await self._store_lore_summary(lore)
            
            # Integrate with memory system
            await self._integrate_with_memory_system(lore)
            
            return lore

    async def get_comprehensive_location_context(self, location_name: str, context: str = "public") -> Dict[str, Any]:
        """
        Get comprehensive lore context for a specific location.
        
        Args:
            location_name: Name of the location
            context: Context type (public, private, dm)
            
        Returns:
            Dictionary with location lore
        """
        # Get location details from the database
        location_details = {}
        location_id = None
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get location from Locations table
                location_row = await conn.fetchrow("""
                    SELECT * FROM Locations 
                    WHERE location_name = $1 AND user_id = $2 AND conversation_id = $3
                """, location_name, self.user_id, self.conversation_id)
                
                if location_row:
                    location_details = dict(location_row)
                    location_id = location_details.get("id")
                    
                    # Get location lore if exists
                    location_lore = await conn.fetchrow("""
                        SELECT * FROM LocationLore
                        WHERE location_id = $1
                    """, location_id)
                    
                    if location_lore:
                        lore_details = dict(location_lore)
                        # Add lore fields to location details
                        for key, value in lore_details.items():
                            if key not in ["id", "user_id", "conversation_id", "location_id", "embedding"]:
                                location_details[f"lore_{key}"] = value
        
        # Get cultural context
        cultural_info = await self._get_cultural_context_for_location(location_name)
        
        # Get religious context
        religion_info = await self._get_religious_context_for_location(location_name)
        
        # Get political context
        political_info = await self._get_political_context_for_location(location_name)
        
        return {
            "location": location_details,
            "cultural_context": cultural_info,
            "religious_context": religion_info,
            "political_context": political_info
        }

    async def _get_cultural_context_for_location(self, location_name: str) -> Dict[str, Any]:
        """Get cultural context for a location."""
        cultural_elements = []
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get cultural elements practiced at this location
                rows = await conn.fetch("""
                    SELECT ce.* 
                    FROM CulturalElements ce
                    JOIN LoreConnections lc ON ce.id = lc.source_id
                    JOIN Locations l ON lc.target_id = l.id
                    WHERE l.location_name = $1
                    AND ce.user_id = $2 AND ce.conversation_id = $3
                    AND lc.source_type = 'CulturalElements'
                    AND lc.target_type = 'Locations'
                    AND lc.connection_type = 'practiced_at'
                """, location_name, self.user_id, self.conversation_id)
                
                if rows:
                    cultural_elements = [dict(row) for row in rows]
        
        return {
            "elements": cultural_elements,
            "count": len(cultural_elements)
        }

    async def _get_religious_context_for_location(self, location_name: str) -> Dict[str, Any]:
        """Get religious context for a location."""
        religious_elements = []
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get religious elements specific to this location
                rows = await conn.fetch("""
                    SELECT ce.* 
                    FROM CulturalElements ce
                    JOIN LoreConnections lc ON ce.id = lc.source_id
                    JOIN Locations l ON lc.target_id = l.id
                    WHERE l.location_name = $1
                    AND ce.user_id = $2 AND ce.conversation_id = $3
                    AND ce.element_type = 'religion'
                    AND lc.source_type = 'CulturalElements'
                    AND lc.target_type = 'Locations'
                    AND lc.connection_type = 'practiced_at'
                """, location_name, self.user_id, self.conversation_id)
                
                if rows:
                    religious_elements = [dict(row) for row in rows]
        
        return {
            "elements": religious_elements,
            "count": len(religious_elements)
        }

    async def _get_political_context_for_location(self, location_name: str) -> Dict[str, Any]:
        """Get political context for a location."""
        ruling_factions = []
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get factions ruling or influencing this location
                rows = await conn.fetch("""
                    SELECT f.*, lc.strength as influence_level, lc.description as influence_description 
                    FROM Factions f
                    JOIN LoreConnections lc ON f.id = lc.source_id
                    JOIN Locations l ON lc.target_id = l.id
                    WHERE l.location_name = $1
                    AND f.user_id = $2 AND f.conversation_id = $3
                    AND lc.source_type = 'Factions'
                    AND lc.target_type = 'Locations'
                    AND (lc.connection_type = 'controls' OR lc.connection_type = 'influences')
                """, location_name, self.user_id, self.conversation_id)
                
                if rows:
                    ruling_factions = [dict(row) for row in rows]
        
        return {
            "ruling_factions": ruling_factions,
            "count": len(ruling_factions)
        }

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="integrate_lore_with_npcs",
        action_description="Integrating lore with {len(npc_ids)} NPCs",
        id_from_context=lambda ctx: "lore_npc_integration"
    )
    async def integrate_lore_with_npcs(self, npc_ids: List[int]) -> Dict[str, Any]:
        """
        Integrate lore with NPCs by giving them appropriate knowledge with Nyx governance oversight.
        
        Args:
            npc_ids: List of NPC IDs to integrate lore with
            
        Returns:
            Dict of integration results
        """
        results = {}
        
        for npc_id in npc_ids:
            # Get NPC details
            npc_details = await self._get_npc_details(npc_id)
            if not npc_details:
                continue
                
            # Determine affiliations and background
            cultural_background = npc_details.get("archetypes", ["resident"])[0]
            affiliations = npc_details.get("affiliations", [])
            
            # Initialize NPC knowledge
            knowledge_granted = await self.npc_lore.initialize_npc_lore_knowledge(
                npc_id,
                cultural_background,
                affiliations
            )
            
            results[npc_id] = {
                "name": npc_details.get("npc_name", f"NPC #{npc_id}"),
                "knowledge_granted": knowledge_granted
            }
        
        return results

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_npc_lore_response",
        action_description="Generating lore-based response for NPC {npc_id}",
        id_from_context=lambda ctx: "npc_lore_response"
    )
    async def generate_npc_lore_response(self, npc_id: int, player_input: str) -> Dict[str, Any]:
        """
        Generate a lore-based response from an NPC based on player question with Nyx governance oversight.
        
        Args:
            npc_id: ID of the NPC
            player_input: Player's question or prompt
            
        Returns:
            NPC's lore-based response
        """
        # Process the potential lore interaction
        interaction = await self.npc_lore.process_npc_lore_interaction(
            npc_id, 
            player_input
        )
        
        # If this is a lore interaction and the NPC has knowledge
        if interaction.get("is_lore_interaction", False) and interaction.get("has_knowledge", False):
            # If this should grant player knowledge
            if interaction.get("should_grant_player_knowledge", False):
                await self.lore_manager.discover_lore(
                    interaction["lore_type"],
                    interaction["lore_id"],
                    "player",  # entity_type
                    0,         # entity_id for player is 0
                    interaction["player_knowledge_level"]
                )
            
            return {
                "response": interaction["response"],
                "lore_shared": True,
                "knowledge_level": interaction.get("knowledge_level", 0)
            }
        else:
            # Not a lore interaction or NPC doesn't know
            return {
                "response": interaction.get("message", "I don't know much about that."),
                "lore_shared": False
            }

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="enhance_context_with_lore",
        action_description="Enhancing context with relevant lore",
        id_from_context=lambda ctx: "lore_context_enhancement"
    )
    async def enhance_gpt_context_with_lore(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance GPT context with relevant lore for better responses with Nyx governance oversight.
        
        Args:
            context: Current context dict for GPT
            
        Returns:
            Enhanced context with relevant lore
        """
        # Create a copy of the context
        enhanced_context = context.copy()
        
        # Extract query information
        location = context.get("location", "")
        player_input = context.get("player_input", "")
        current_npcs = context.get("current_npcs", [])
        
        # Build search query
        npc_names = [npc.get("npc_name", "") for npc in current_npcs]
        search_query = f"{location} {player_input} {' '.join(npc_names)}"
        
        # Get relevant lore
        relevant_lore = await self.lore_manager.get_relevant_lore(
            search_query,
            min_relevance=0.7,
            limit=5
        )
        
        # Add lore to context
        enhanced_context["relevant_lore"] = relevant_lore
        
        # Get active conflicts for additional context
        conflict_integration = ConflictSystemIntegration(self.user_id, self.conversation_id)
        active_conflicts = await conflict_integration.get_active_conflicts()
        
        # Add conflict data to context
        enhanced_context["active_conflicts"] = active_conflicts
        
        # Get location lore if in a known location
        if location:
            location_lore = await self._get_location_lore(location)
            if location_lore:
                enhanced_context["location_lore"] = location_lore
        
        return enhanced_context

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_scene_with_lore",
        action_description="Generating scene with lore for location: {location}",
        id_from_context=lambda ctx: "lore_scene_generation"
    )
    async def generate_scene_description_with_lore(self, location: str) -> Dict[str, Any]:
        """
        Generate a scene description enhanced with relevant lore with Nyx governance oversight.
        
        Args:
            location: Current location name
            
        Returns:
            Enhanced scene description
        """
        # Get location details
        location_details = await self._get_location_details(location)
        base_description = location_details.get("description", f"You are at {location}")
        
        # Get location lore
        location_lore = await self._get_location_lore(location)
        
        # Get factions that control or influence this location
        controlling_factions = await self._get_location_factions(location)
        
        # Get cultural elements practiced in this area
        cultural_elements = await self._get_location_cultural_elements(location)
        
        # Compile enhanced description
        result = {
            "base_description": base_description,
            "lore_elements": {
                "hidden_secrets": location_lore.get("hidden_secrets", []),
                "local_legends": location_lore.get("local_legends", []),
                "controlling_factions": controlling_factions,
                "cultural_elements": cultural_elements
            }
        }
        
        # Generate a complete description using a prompt
        from logic.chatgpt_integration import get_chatgpt_response
        
        prompt = f"""
        Generate an atmospheric scene description for this location that subtly incorporates lore:
        
        Location: {location}
        Base Description: {base_description}
        
        Controlling Factions: {", ".join([f["name"] for f in controlling_factions]) if controlling_factions else "None"}
        
        Cultural Elements: {", ".join([e["name"] for e in cultural_elements]) if cultural_elements else "None"}
        
        Local Legends: {", ".join(location_lore.get("local_legends", ["None"]))[:150]}
        
        Write a rich, sensory description (200-300 words) that:
        1. Establishes the physical space and atmosphere
        2. Subtly hints at faction influence and cultural elements
        3. Potentially alludes to hidden history or secrets
        4. Feels immersive and authentic to the setting
        
        The description should feel natural, not like an exposition dump.
        """
        
        response = await get_chatgpt_response(
            self.conversation_id,
            system_prompt="You are an AI that specializes in atmospheric scene descriptions that subtly incorporate worldbuilding.",
            user_prompt=prompt
        )
        
        if isinstance(response, dict) and "response" in response:
            result["enhanced_description"] = response["response"]
        else:
            result["enhanced_description"] = str(response)
        
        return result

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="update_lore_after_event",
        action_description="Updating lore after narrative event: {event_description}",
        id_from_context=lambda ctx: "lore_event_update"
    )
    async def update_lore_after_narrative_event(self, event_description: str) -> Dict[str, Any]:
        """
        Update world lore after a significant narrative event with Nyx governance oversight.
        
        Args:
            event_description: Description of the narrative event
            
        Returns:
            Dict of lore updates
        """
        return await self.lore_generator.evolve_lore_with_event(event_description)

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_quest_lore_context",
        action_description="Getting lore context for quest {quest_id}",
        id_from_context=lambda ctx: "quest_lore_context"
    )
    async def get_quest_lore_context(self, quest_id: int) -> Dict[str, Any]:
        """
        Get lore context relevant to a specific quest with Nyx governance oversight.
        
        Args:
            quest_id: Quest ID to get context for
            
        Returns:
            Dict of lore context for the quest
        """
        # Get quest details
        quest_details = await self._get_quest_details(quest_id)
        
        if not quest_details:
            return {"error": "Quest not found"}
        
        quest_name = quest_details.get("quest_name", f"Quest #{quest_id}")
        quest_giver = quest_details.get("quest_giver", "")
        
        # Build search query
        search_query = f"{quest_name} {quest_giver}"
        
        # Get relevant lore
        relevant_lore = await self.lore_manager.get_relevant_lore(
            search_query,
            min_relevance=0.6,
            limit=10
        )
        
        # Get quest specific lore from cache
        from utils.caching import LORE_CACHE
        quest_lore_key = f"quest_lore:{self.user_id}:{self.conversation_id}:{quest_id}"
        quest_lore = LORE_CACHE.get(quest_lore_key) or {}
        
        # Return combined context
        return {
            "quest_details": quest_details,
            "quest_lore": quest_lore,
            "relevant_lore": relevant_lore
        }

    async def _store_lore_summary(self, lore: Dict[str, Any]) -> None:
        """
        Store a compact summary of the generated lore for quick reference with Nyx integration.
        
        Args:
            lore: Complete lore dictionary
        """
        # Initialize governance if needed
        await self.initialize_governance()
        
        try:
            import json
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Create a summary by taking key elements
            summary = {
                "world_overview": lore.get("world_lore", {}).get("cosmology", "")[:500],
                "faction_count": len(lore.get("factions", [])),
                "major_factions": [f["name"] for f in lore.get("factions", [])[:3]],
                "cultural_elements": [c["name"] for c in lore.get("cultural_elements", [])[:5]],
                "major_locations": [l["name"] for l in lore.get("locations", [])[:5]]
            }
            
            # Store in CurrentRoleplay
            cursor.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES (%s, %s, 'LoreSummary', %s)
                ON CONFLICT (user_id, conversation_id, key) DO UPDATE
                SET value = EXCLUDED.value
            """, (self.user_id, self.conversation_id, json.dumps(summary)))
            
            conn.commit()
            
            # Report action to governance
            await self.governor.process_agent_action_report(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="lore_integration",
                action={
                    "type": "store_lore_summary",
                    "description": "Stored compact lore summary for reference"
                },
                result={
                    "summary_size": len(json.dumps(summary)),
                    "faction_count": summary["faction_count"]
                }
            )
            
        except Exception as e:
            logging.error(f"Error storing lore summary: {e}")
        finally:
            cursor.close()
            conn.close()
            
    async def _integrate_with_memory_system(self, lore: Dict[str, Any]) -> None:
        """
        Integrate generated lore with Nyx's memory system.
        
        Args:
            lore: The generated lore
        """
        # Initialize governance if needed
        await self.initialize_governance()
        
        # Get memory system from Nyx
        memory_system = await self.governor.get_memory_system()
        
        # Store foundation lore in memory
        if "world_lore" in lore:
            for key, value in lore["world_lore"].items():
                if isinstance(value, str):
                    await memory_system.add_memory(
                        memory_text=f"World lore - {key}: {value[:100]}...",
                        memory_type="lore",
                        memory_scope="game",
                        significance=7,
                        tags=["lore", "world", key]
                    )
        
        # Store factions in memory
        if "factions" in lore:
            for faction in lore["factions"]:
                name = faction.get('name', 'Unknown faction')
                desc = faction.get('description', '')[:100] + '...'
                await memory_system.add_memory(
                    memory_text=f"Faction: {name} - {desc}",
                    memory_type="lore",
                    memory_scope="game",
                    significance=6,
                    tags=["lore", "faction", name]
                )
        
        # Store cultural elements in memory
        if "cultural_elements" in lore:
            for element in lore["cultural_elements"]:
                name = element.get('name', 'Unknown culture')
                desc = element.get('description', '')[:100] + '...'
                await memory_system.add_memory(
                    memory_text=f"Cultural element: {name} - {desc}",
                    memory_type="lore",
                    memory_scope="game",
                    significance=6,
                    tags=["lore", "culture", name]
                )
        
        # Store historical events in memory
        if "historical_events" in lore:
            for event in lore["historical_events"]:
                name = event.get('name', 'Unknown event')
                desc = event.get('description', '')[:100] + '...'
                await memory_system.add_memory(
                    memory_text=f"Historical event: {name} - {desc}",
                    memory_type="lore",
                    memory_scope="game",
                    significance=6,
                    tags=["lore", "history", name]
                )
        
        # Store locations in memory
        if "locations" in lore:
            for location in lore["locations"]:
                name = location.get('name', 'Unknown location')
                desc = location.get('description', '')[:100] + '...'
                await memory_system.add_memory(
                    memory_text=f"Location: {name} - {desc}",
                    memory_type="lore",
                    memory_scope="game",
                    significance=6,
                    tags=["lore", "location", name]
                )
                
        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_integration",
            action={
                "type": "integrate_with_memory",
                "description": "Integrated lore with memory system"
            },
            result={
                "memory_entries_added": (
                    len(lore.get("world_lore", {})) +
                    len(lore.get("factions", [])) +
                    len(lore.get("cultural_elements", [])) +
                    len(lore.get("historical_events", [])) +
                    len(lore.get("locations", []))
                )
            }
        )

    async def _get_npc_details(self, npc_id: int) -> Dict[str, Any]:
        """Get details for an NPC from the database."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT npc_name, archetypes, affiliations
                FROM NPCStats
                WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
            """, (npc_id, self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            if not row:
                return {}
            
            # Try to parse JSON fields
            try:
                archetypes = json.loads(row[1]) if row[1] else []
            except:
                archetypes = []
                
            try:
                affiliations = json.loads(row[2]) if row[2] else []
            except:
                affiliations = []
            
            return {
                "npc_name": row[0],
                "archetypes": archetypes,
                "affiliations": affiliations
            }
        except Exception as e:
            logging.error(f"Error getting NPC details: {e}")
            return {}
        finally:
            cursor.close()
            conn.close()
    
    async def _get_location_details(self, location_name: str) -> Dict[str, Any]:
        """Get details for a location from the database."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, location_name, description
                FROM Locations
                WHERE location_name=%s AND user_id=%s AND conversation_id=%s
            """, (location_name, self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            if not row:
                return {"description": f"You are at {location_name}"}
            
            return {
                "location_id": row[0],
                "location_name": row[1],
                "description": row[2]
            }
        except Exception as e:
            logging.error(f"Error getting location details: {e}")
            return {"description": f"You are at {location_name}"}
        finally:
            cursor.close()
            conn.close()
    
    async def _get_location_lore(self, location_name: str) -> Dict[str, Any]:
        """Get lore for a location."""
        location_details = await self._get_location_details(location_name)
        location_id = location_details.get("location_id")
        
        if not location_id:
            return {}
        
        # Get lore from LocationLore table
        location_with_lore = await self.lore_manager.get_location_with_lore(location_id)
        return location_with_lore or {}
    
    async def _get_location_factions(self, location_name: str) -> List[Dict[str, Any]]:
        """Get factions that control or influence a location."""
        location_details = await self._get_location_details(location_name)
        location_id = location_details.get("location_id")
        
        if not location_id:
            return []
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT f.id, f.name, f.type, lc.connection_type
                FROM Factions f
                JOIN LoreConnections lc ON f.id = lc.source_id
                WHERE lc.source_type = 'Factions'
                AND lc.target_type = 'LocationLore'
                AND lc.target_id = %s
            """, (location_id,))
            
            columns = [col[0] for col in cursor.description]
            result = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            return result
        except Exception as e:
            logging.error(f"Error getting location factions: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    async def _get_location_cultural_elements(self, location_name: str) -> List[Dict[str, Any]]:
        """Get cultural elements practiced at a location."""
        # This requires inference since we don't directly store
        # which cultural elements are practiced at specific locations
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get location controlling factions
            location_details = await self._get_location_details(location_name)
            location_id = location_details.get("location_id")
            
            if not location_id:
                return []
            
            # First get factions that control this location
            cursor.execute("""
                SELECT source_id FROM LoreConnections
                WHERE target_type = 'LocationLore'
                AND target_id = %s
                AND source_type = 'Factions'
            """, (location_id,))
            
            faction_ids = [row[0] for row in cursor.fetchall()]
            
            if not faction_ids:
                return []
            
            # Get cultural elements practiced by these factions
            placeholders = ', '.join(['%s'] * len(faction_ids))
            query = f"""
                SELECT DISTINCT ce.id, ce.name, ce.type, ce.description
                FROM CulturalElements ce
                JOIN LoreConnections lc ON ce.id = lc.target_id
                WHERE lc.source_type = 'Factions'
                AND lc.source_id IN ({placeholders})
                AND lc.target_type = 'CulturalElements'
                AND lc.connection_type = 'practices'
            """
            
            cursor.execute(query, faction_ids)
            
            columns = [col[0] for col in cursor.description]
            result = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            return result
        except Exception as e:
            logging.error(f"Error getting location cultural elements: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    async def _get_quest_details(self, quest_id: int) -> Dict[str, Any]:
        """Get details for a quest from the database."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT quest_name, status, progress_detail, quest_giver, reward
                FROM Quests
                WHERE quest_id=%s AND user_id=%s AND conversation_id=%s
            """, (quest_id, self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            if not row:
                return {}
            
            return {
                "quest_id": quest_id,
                "quest_name": row[0],
                "status": row[1],
                "progress_detail": row[2],
                "quest_giver": row[3],
                "reward": row[4]
            }
        except Exception as e:
            logging.error(f"Error getting quest details: {e}")
            return {}
        finally:
            cursor.close()
            conn.close()
