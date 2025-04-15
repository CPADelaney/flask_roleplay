# lore/setting_analyzer.py

import logging
import json
from typing import Dict, Any, List
from db.connection import get_db_connection_context

# Import Nyx governance
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType
from nyx.governance_helpers import with_governance, with_governance_permission

class SettingAnalyzer:
    """
    Analyzes the setting's NPC data to feed into an agentic tool 
    that generates organizations, with full Nyx governance integration.
    """

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = None

    async def initialize_governance(self):
        """Initialize Nyx governance integration"""
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
        return self.governor

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="aggregate_npc_data",
        action_description="Aggregating NPC data for setting analysis",
        id_from_context=lambda ctx: "setting_analyzer"
    )
    async def aggregate_npc_data(self, ctx) -> Dict[str, Any]:
        """
        Collect all NPC data (likes, hobbies, archetypes, affiliations) into a unified format
        with Nyx governance oversight.
        
        Returns:
            Dictionary with aggregated NPC data
        """
        # Initialize empty collections
        all_npcs = []
        all_archetypes, all_likes, all_hobbies, all_affiliations, all_locations = (
            set(), set(), set(), set(), set()
        )
        
        async with get_db_connection_context() as conn:
            # Fetch NPC data using async query
            rows = await conn.fetch("""
                SELECT npc_id, npc_name, archetypes, likes, dislikes, 
                       hobbies, affiliations, personality_traits, 
                       current_location, archetype_summary
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2
            """, self.user_id, self.conversation_id)
            
            # Process rows into a structured dict
            for row in rows:
                npc_id = row['npc_id']
                npc_name = row['npc_name']
                archetypes_json = row['archetypes']
                likes_json = row['likes']
                dislikes_json = row['dislikes']
                hobbies_json = row['hobbies']
                affiliations_json = row['affiliations']
                personality_json = row['personality_traits']
                current_location = row['current_location']
                archetype_summary = row['archetype_summary']

                # Safely load JSON fields
                def safe_load(s):
                    try:
                        return json.loads(s) if s else []
                    except:
                        return []

                archetypes = safe_load(archetypes_json)
                likes = safe_load(likes_json)
                dislikes = safe_load(dislikes_json)
                hobbies = safe_load(hobbies_json)
                affiliations = safe_load(affiliations_json)
                personality_traits = safe_load(personality_json)

                # Update sets
                all_archetypes.update(archetypes)
                all_likes.update(likes)
                all_hobbies.update(hobbies)
                all_affiliations.update(affiliations)
                if current_location:
                    all_locations.add(current_location)
                
                all_npcs.append({
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "archetypes": archetypes,
                    "likes": likes,
                    "dislikes": dislikes,
                    "hobbies": hobbies,
                    "affiliations": affiliations,
                    "personality_traits": personality_traits,
                    "current_location": current_location,
                    "archetype_summary": archetype_summary
                })

        # Get the environment desc and setting name from DB
        setting_desc, setting_name = await self._get_current_setting_info()

        return {
            "setting_name": setting_name,
            "setting_description": setting_desc,
            "npcs": all_npcs,
            "aggregated": {
                "archetypes": list(all_archetypes),
                "likes": list(all_likes),
                "hobbies": list(all_hobbies),
                "affiliations": list(all_affiliations),
                "locations": list(all_locations),
            }
        }

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="analyze_setting_demographics",
        action_description="Analyzing setting demographics and social structure",
        id_from_context=lambda ctx: "setting_analyzer"
    )
    async def analyze_setting_demographics(self, ctx) -> Dict[str, Any]:
        """
        Analyze the demographics and social structure of the setting with Nyx governance oversight.
        
        Returns:
            Dictionary with demographic analysis
        """
        # First get the aggregated NPC data
        npc_data = await self.aggregate_npc_data(ctx)
        
        # Group NPCs by location
        locations = {}
        for npc in npc_data["npcs"]:
            location = npc.get("current_location")
            if location:
                if location not in locations:
                    locations[location] = []
                locations[location].append(npc)
        
        # Count archetypes
        archetype_counts = {}
        for npc in npc_data["npcs"]:
            for archetype in npc.get("archetypes", []):
                if archetype not in archetype_counts:
                    archetype_counts[archetype] = 0
                archetype_counts[archetype] += 1
        
        # Count affiliations
        affiliation_counts = {}
        for npc in npc_data["npcs"]:
            for affiliation in npc.get("affiliations", []):
                if affiliation not in affiliation_counts:
                    affiliation_counts[affiliation] = 0
                affiliation_counts[affiliation] += 1
        
        return {
            "setting_name": npc_data["setting_name"],
            "total_npcs": len(npc_data["npcs"]),
            "locations": {
                location: len(npcs) for location, npcs in locations.items()
            },
            "archetype_distribution": archetype_counts,
            "affiliation_distribution": affiliation_counts,
            "setting_description": npc_data["setting_description"]
        }

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_organizations",
        action_description="Generating organizations based on setting analysis",
        id_from_context=lambda ctx: "setting_analyzer"
    )
    async def generate_organizations(self, ctx) -> Dict[str, Any]:
        """
        Generate organizations based on setting analysis with Nyx governance oversight.
        
        This method will:
        1. Analyze the setting data
        2. Send the data to an LLM for organization generation
        3. Return structured organization data
        
        Returns:
            Dictionary with generated organizations
        """
        # First, analyze the setting
        setting_data = await self.analyze_setting_demographics(ctx)
        
        # This would call an LLM via an agent system
        # For now, we'll just import and call a generic function
        from lore.lore_agents import analyze_setting
        
        organizations = await analyze_setting(
            ctx,
            setting_data["setting_description"],
            setting_data["npcs"] if "npcs" in setting_data else []
        )
        
        return {
            "setting_name": setting_data["setting_name"],
            "organizations": organizations,
            "organization_count": sum(len(category) for category in organizations.values() if isinstance(category, list))
        }

    async def _get_current_setting_info(self):
        """
        Helper to fetch the current setting name and environment desc from the DB.
        """
        setting_desc = "A setting with no description."
        setting_name = "The Setting"
        
        async with get_db_connection_context() as conn:
            # Get environment description
            row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'EnvironmentDesc'
            """, self.user_id, self.conversation_id)
            if row:
                setting_desc = row['value'] or setting_desc

            # Get setting name
            row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentSetting'
            """, self.user_id, self.conversation_id)
            if row:
                setting_name = row['value'] or setting_name

        return setting_desc, setting_name
        
    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await self.initialize_governance()
        
        # Register this analyzer with governance
        await self.governor.register_agent(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="setting_analyzer",
            agent_instance=self
        )
        
        # Issue a directive for setting analysis
        await self.governor.issue_directive(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="setting_analyzer",
            directive_type="ACTION",
            directive_data={
                "instruction": "Analyze setting data to generate coherent organizations.",
                "scope": "setting"
            },
            priority=5,  # Medium priority
            duration_minutes=24*60  # 24 hours
        )
        
        logging.info(f"SettingAnalyzer registered with Nyx governance for user {self.user_id}, conversation {self.conversation_id}")
