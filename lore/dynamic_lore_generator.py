# lore/dynamic_lore_generator.py

import logging
from typing import Dict, List, Any, Optional
from agents.run_context import RunContextWrapper

from lore.lore_manager import LoreManager
from lore.lore_tools import (
    generate_foundation_lore,
    generate_factions,
    generate_cultural_elements,
    generate_historical_events,
    generate_locations,
    generate_quest_hooks
)

class DynamicLoreGenerator:
    """
    A refactored class that uses the Agents SDK function tools (sub-agents)
    rather than direct ChatCompletion calls for generating comprehensive lore.
    
    - Each method corresponds to a particular lore step:
      1) initialize_world_lore  -> foundation lore
      2) generate_factions      -> faction creation
      3) generate_cultural_elements -> cultural traditions, taboos, etc.
      4) generate_historical_events -> major events shaping the world
      5) generate_locations     -> important places
      6) generate_quest_hooks   -> potential quests
    - generate_complete_lore orchestrates them all in sequence.
    
    The built-in 'LoreManager' is used to store everything in your DB.
    The actual DB schema can differ from this example, so adapt as needed.
    """

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_manager = LoreManager(user_id, conversation_id)

    async def initialize_world_lore(self, environment_desc: str) -> Dict[str, Any]:
        """
        Initialize core foundation lore (cosmology, magic, world history, etc.)
        and store it in the DB. The output is stored under 'WorldLore' with relevant tags.
        
        :param environment_desc: Short textual description of the environment
        :return: Dict containing the five fields from the FoundationLoreOutput
        """
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })

        # Call the agentic function tool
        foundation_data = await generate_foundation_lore(run_ctx, environment_desc)

        # For example: { 'cosmology': "...", 'magic_system': "...", ... }
        for category, desc in foundation_data.items():
            # e.g. "category" might be "cosmology", "magic_system", ...
            # We store each chunk as a separate record in WorldLore
            await self.lore_manager.add_world_lore(
                name=f"{category.title()} of {self.get_setting_name()}",
                category=category,
                description=desc,
                significance=8,
                tags=[category, "foundation", "world_building"]
            )

        return foundation_data

    async def generate_factions(self, environment_desc: str, world_lore: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate 3-5 distinct factions referencing the environment description
        and possibly 'social_structure' from the foundation data.
        
        :param environment_desc: Text describing environment or setting
        :param world_lore: The dictionary from initialize_world_lore
        :return: A list of faction dictionaries
        """
        # Typically we want the 'social_structure' from foundation_data
        social_structure = world_lore.get("social_structure", "")

        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })

        # Produce a list of faction dicts via our sub-agent
        factions_data = await generate_factions(run_ctx, environment_desc, social_structure)

        # Something like [ {name, type, description, ...}, {...}, ... ]
        # Store each in the DB
        for faction in factions_data:
            try:
                faction_id = await self.lore_manager.add_faction(
                    name=faction["name"],
                    faction_type=faction["type"],
                    description=faction["description"],
                    values=faction["values"],
                    goals=faction["goals"],
                    headquarters=faction.get("headquarters"),
                    rivals=faction.get("rivals", []),
                    allies=faction.get("allies", []),
                    hierarchy_type=faction.get("hierarchy_type")
                )
                # Possibly store connections between factions (like rivalry)
                # if "rivals" or "allies" references other generated factions.
            except Exception as e:
                logging.error(f"Error storing faction '{faction['name']}': {e}")

        return factions_data

    async def generate_cultural_elements(self, environment_desc: str, factions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate cultural elements referencing environment + the names of the existing factions.
        """
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        faction_names = ", ".join([f.get("name", "Unknown") for f in factions])

        cultural_data = await generate_cultural_elements(run_ctx, environment_desc, faction_names)

        # Now store them
        for element in cultural_data:
            try:
                element_id = await self.lore_manager.add_cultural_element(
                    name=element["name"],
                    element_type=element["type"],
                    description=element["description"],
                    practiced_by=element["practiced_by"],
                    significance=element["significance"],
                    historical_origin=element.get("historical_origin", "")
                )
            except Exception as e:
                logging.error(f"Error storing cultural element '{element.get('name','unknown')}': {e}")

        return cultural_data

    async def generate_historical_events(self, environment_desc: str, foundation_data: Dict[str, Any], factions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate 5-7 major historical events referencing environment + existing 'world_history' + faction names.
        """
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        # We can feed in the previously generated 'world_history'
        world_history = foundation_data.get("world_history", "")
        faction_names = ", ".join([f.get("name","Unknown") for f in factions])

        events_data = await generate_historical_events(run_ctx, environment_desc, world_history, faction_names)

        # Then store them
        for event in events_data:
            try:
                event_id = await self.lore_manager.add_historical_event(
                    name=event["name"],
                    description=event["description"],
                    date_description=event["date_description"],
                    significance=event["significance"],
                    participating_factions=event["participating_factions"],
                    consequences=event["consequences"]
                )
                # Possibly store connections if each event references multiple factions
            except Exception as e:
                logging.error(f"Error storing historical event '{event.get('name','Unknown')}': {e}")

        return events_data

    async def generate_locations(self, environment_desc: str, factions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate 5-8 significant locations referencing environment + faction names.
        The sub-agent returns a list of location dicts with keys like name, description, type, controlling_faction, etc.
        """
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        faction_names = ", ".join([f.get("name","Unknown") for f in factions])

        locations_data = await generate_locations(run_ctx, environment_desc, faction_names)

        # Store each location
        for loc in locations_data:
            try:
                # Suppose we have a separate 'Locations' table for the main record:
                location_row_id = await self.lore_manager.create_location_record(
                    name=loc["name"],
                    description=loc["description"],
                    location_type=loc["type"]
                )

                # Then attach location lore if needed
                controlling_faction = loc.get("controlling_faction")
                hidden_secrets = loc.get("hidden_secrets", [])
                founding_story = f"Founded as a {loc['type']}."  # or some logic

                # The add_location_lore might update or insert a row in 'LocationLore' table
                await self.lore_manager.add_location_lore(
                    location_id=location_row_id,
                    founding_story=founding_story,
                    hidden_secrets=hidden_secrets,
                    local_legends=[],
                    historical_significance=loc.get("strategic_importance", "")
                )

                # Also record controlling_faction if you want to create a LoreConnection
                if controlling_faction:
                    await self.lore_manager.connect_faction_to_location(location_row_id, controlling_faction)
            except Exception as e:
                logging.error(f"Error storing location '{loc.get('name','Unknown')}': {e}")

        return locations_data

    async def generate_quest_hooks(self, factions: List[Dict[str, Any]], locations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate 5-7 quest hooks referencing existing factions + location names.
        The sub-agent returns an array of quest dicts with keys:
          quest_name, quest_giver, location, description, objectives, rewards, difficulty, lore_significance
        """
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        faction_names = ", ".join([f.get("name","Unknown") for f in factions])
        location_names = ", ".join([l.get("name","Unknown") for l in locations])

        quests_data = await generate_quest_hooks(run_ctx, faction_names, location_names)

        # Store them in your Quests table or similar
        for quest in quests_data:
            try:
                # Suppose there's a method in LoreManager or a separate system for quest insertion
                quest_id = await self.lore_manager.add_quest_record(
                    quest_name=quest["quest_name"],
                    quest_giver=quest["quest_giver"],
                    location=quest["location"],
                    description=quest["description"],
                    difficulty=quest["difficulty"],
                    objectives=quest["objectives"],
                    rewards=quest["rewards"],
                    lore_significance=quest["lore_significance"]
                )
            except Exception as e:
                logging.error(f"Error storing quest hook '{quest.get('quest_name','Unknown')}': {e}")

        return quests_data

    async def generate_complete_lore(self, environment_desc: str) -> Dict[str, Any]:
        """
        A single method that calls each sub-step in a specific order:
          1) Foundation
          2) Factions
          3) Cultural
          4) Historical
          5) Locations
          6) Quests
        Then returns a dictionary combining all results.
        
        :param environment_desc: High-level environment description
        :return: A dictionary with keys:
                 "world_lore", "factions", "cultural_elements", 
                 "historical_events", "locations", "quests"
        """
        # 1) Foundation lore
        foundation_data = await self.initialize_world_lore(environment_desc)

        # 2) Factions referencing 'social_structure' from foundation_data
        factions_data = await self.generate_factions(environment_desc, foundation_data)

        # 3) Cultural elements referencing environment + factions
        cultural_data = await self.generate_cultural_elements(environment_desc, factions_data)

        # 4) Historical events referencing environment + foundation_data + factions
        historical_data = await self.generate_historical_events(environment_desc, foundation_data, factions_data)

        # 5) Locations referencing environment + factions
        locations_data = await self.generate_locations(environment_desc, factions_data)

        # 6) Quest hooks referencing factions + locations
        quests_data = await self.generate_quest_hooks(factions_data, locations_data)

        return {
            "world_lore": foundation_data,
            "factions": factions_data,
            "cultural_elements": cultural_data,
            "historical_events": historical_data,
            "locations": locations_data,
            "quests": quests_data
        }

    def get_setting_name(self) -> str:
        """
        Example method to fetch the current setting name from the DB or 
        from 'CurrentRoleplay' table. You can adapt it to your actual logic.
        """
        try:
            # Suppose you store 'CurrentSetting' in 'CurrentRoleplay'
            import asyncpg
            from db.connection import get_db_connection

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=%s AND conversation_id=%s AND key='CurrentSetting'
                LIMIT 1
            """, (self.user_id, self.conversation_id))
            row = cursor.fetchone()
            cursor.close()
            conn.close()

            if row:
                return row[0]
            else:
                return "The Setting"
        except:
            return "The Setting"
