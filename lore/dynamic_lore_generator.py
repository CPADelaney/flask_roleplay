# lore/dynamic_lore_generator.py

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional

from logic.chatgpt_integration import get_chatgpt_response
from lore.lore_manager import LoreManager
from db.connection import get_db_connection
from utils.caching import LORE_CACHE
from lore.setting_analyzer import SettingAnalyzer

class DynamicLoreGenerator:
    """
    Generates dynamic lore based on game setting using GPT.
    Creates interconnected lore elements that evolve with the game world.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_manager = LoreManager(user_id, conversation_id)
    
    async def initialize_world_lore(self, environment_desc: str) -> Dict[str, Any]:
        """
        Initialize the core world lore based on environment description.
        
        Args:
            environment_desc: Description of the game environment
            
        Returns:
            Dict of generated foundational lore
        """
        # Get the setting details to build our prompt
        prompt = f"""
        Generate cohesive foundational world lore for this environment:
        
        {environment_desc}
        
        Create these interconnected lore elements as a JSON object:
        
        1. "cosmology": An origin story/creation myth that fits the setting (1-2 paragraphs)
        2. "magic_system": How supernatural forces work in this world (if applicable)
        3. "world_history": Brief timeline of 3-5 major historical events
        4. "calendar_system": How time is measured (years, months, special dates)
        5. "social_structure": Class systems or hierarchies that dominate society
        
        Make everything internally consistent and subtly incorporate femdom themes where appropriate.
        Format as a valid JSON object where each key contains a detailed description.
        """
        
        # Get GPT response
        response = await get_chatgpt_response(
            self.conversation_id,
            system_prompt="You are a worldbuilding expert who creates rich, interconnected lore for fantasy settings.",
            user_prompt=prompt
        )
        
        # Extract lore data from response
        lore_data = self._extract_json_from_response(response)
        
        # Store foundation lore elements in database
        for category, description in lore_data.items():
            await self.lore_manager.add_world_lore(
                name=f"{category.title()} of {self.get_setting_name()}",
                category=category,
                description=description,
                significance=8,  # High significance for foundation lore
                tags=[category, "foundation", "world_building"]
            )
        
        return lore_data

    
    async def generate_factions_from_setting_analysis(self, environment_desc: str, world_lore: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Generate factions based on a comprehensive analysis of the setting and NPCs.
        
        Args:
            environment_desc: Description of the game environment
            world_lore: Previously established world lore
            
        Returns:
            List of generated factions
        """
        # Use setting analyzer to get organization data
        analyzer = SettingAnalyzer(self.user_id, self.conversation_id)
        organization_data = await analyzer.analyze_setting_for_organizations()
        
        # Collect all organizations from different categories
        all_organizations = []
        
        for category in ["academic", "athletic", "social", "professional", "cultural", "political", "other"]:
            if category in organization_data:
                for org in organization_data[category]:
                    org["category"] = category  # Add category to the organization data
                    all_organizations.append(org)
        
        # Store each organization as a faction
        faction_ids = []
        
        for org in all_organizations:
            # Skip if it doesn't have the required fields
            if not all(field in org for field in ["name", "description"]):
                continue
            
            # Create the faction
            faction_id = await self.lore_manager.add_faction(
                name=org["name"],
                faction_type=org.get("type", org["category"]),  # Use type or fall back to category
                description=org["description"],
                values=org.get("values", []),  # May not be present in the analysis
                goals=org.get("goals", []),    # May not be present in the analysis
                headquarters=org.get("gathering_location", ""),
                hierarchy_type=org.get("hierarchy", "")
            )
            
            # Store additional metadata
            metadata = {
                "membership_basis": org.get("membership_basis", ""),
                "source": "setting_analysis",
                "category": org["category"]
            }
            
            await self._store_faction_metadata(faction_id, metadata)
            
            # Store the organization with its ID for later connection creation
            faction_ids.append((faction_id, org))
        
        # Create connections between factions based on categories and logic
        for i, (faction_id, org) in enumerate(faction_ids):
            # Set up some default connections based on categories
            if org["category"] == "academic":
                # Academic departments are connected to their parent institution
                for j, (other_id, other_org) in enumerate(faction_ids):
                    if i != j and other_org["category"] == "academic":
                        # If this is a department and the other is an institution
                        if "department" in org.get("type", "").lower() and "institution" in other_org.get("type", "").lower():
                            await self.lore_manager.add_lore_connection(
                                source_type="Factions",
                                source_id=faction_id,
                                target_type="Factions",
                                target_id=other_id,
                                connection_type="part_of",
                                description=f"{org['name']} is part of {other_org['name']}",
                                strength=9
                            )
            
            elif org["category"] == "athletic":
                # Athletic teams might be rivals
                for j, (other_id, other_org) in enumerate(faction_ids):
                    if i != j and other_org["category"] == "athletic" and "team" in other_org.get("type", "").lower():
                        # 33% chance of being rivals
                        if random.random() < 0.33:
                            await self.lore_manager.add_lore_connection(
                                source_type="Factions",
                                source_id=faction_id,
                                target_type="Factions",
                                target_id=other_id,
                                connection_type="rivals_with",
                                description=f"{org['name']} has a rivalry with {other_org['name']}",
                                strength=7
                            )
        
        # Convert to the format expected by the rest of the system
        factions_data = []
        for faction_id, org in faction_ids:
            faction_data = {
                "name": org["name"],
                "type": org.get("type", org["category"]),
                "description": org["description"],
                "values": org.get("values", []),
                "goals": org.get("goals", []),
                "headquarters": org.get("gathering_location", ""),
                "hierarchy_type": org.get("hierarchy", ""),
                "membership_basis": org.get("membership_basis", ""),
                "category": org["category"]
            }
            factions_data.append(faction_data)
        
        return factions_data

    async def generate_factions(self, environment_desc: str, world_lore: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Generate factions based on environment and established world lore.
        
        Args:
            environment_desc: Description of the game environment
            world_lore: Previously established world lore
            
        Returns:
            List of generated factions
        """
        social_structure = world_lore.get("social_structure", "")
        world_history = world_lore.get("world_history", "")
        
        prompt = f"""
        Generate 3-5 distinct factions for this setting:
        
        Environment: {environment_desc}
        
        Social Structure: {social_structure}
        
        World History: {world_history}
        
        For each faction, provide this information as a JSON array of objects:
        1. "name": Faction name
        2. "type": Type of faction (political, religious, criminal, etc.)
        3. "description": Brief description (2-3 sentences)
        4. "values": Array of 3-4 core values
        5. "goals": Array of 2-3 major goals
        6. "headquarters": Main base or gathering place
        7. "rivals": Names of rival factions (can reference other generated factions)
        8. "allies": Names of allied factions (can reference other generated factions)
        9. "hierarchy_type": How the faction is organized internally 
        
        Make at least one faction have a primarily female leadership with dominance themes.
        Create naturally interconnected relationships between factions.
        """
        
        # Get GPT response
        response = await get_chatgpt_response(
            self.conversation_id,
            system_prompt="You are a worldbuilding expert specializing in creating complex faction dynamics.",
            user_prompt=prompt
        )
        
        # Extract faction data from response
        factions_data = self._extract_json_from_response(response)
        
        # Ensure we have a list of factions
        if not isinstance(factions_data, list):
            factions_data = [factions_data]
        
        # Store factions in database and create connections
        faction_ids = []
        for faction in factions_data:
            faction_id = await self.lore_manager.add_faction(
                name=faction["name"],
                faction_type=faction["type"],
                description=faction["description"],
                values=faction["values"],
                goals=faction["goals"],
                headquarters=faction.get("headquarters"),
                founding_story=faction.get("founding_story", ""),
                rivals=faction.get("rivals", []),
                allies=faction.get("allies", []),
                hierarchy_type=faction.get("hierarchy_type", "")
            )
            faction_ids.append((faction_id, faction))
        
        # Create connections between factions
        for faction_id, faction in faction_ids:
            # Process rivals
            for rival_name in faction.get("rivals", []):
                for rival_id, rival_data in faction_ids:
                    if rival_data["name"] == rival_name:
                        await self.lore_manager.add_lore_connection(
                            source_type="Factions",
                            source_id=faction_id,
                            target_type="Factions",
                            target_id=rival_id,
                            connection_type="conflicts_with",
                            description=f"{faction['name']} has a rivalry with {rival_name}",
                            strength=7
                        )
            
            # Process allies
            for ally_name in faction.get("allies", []):
                for ally_id, ally_data in faction_ids:
                    if ally_data["name"] == ally_name:
                        await self.lore_manager.add_lore_connection(
                            source_type="Factions",
                            source_id=faction_id,
                            target_type="Factions",
                            target_id=ally_id,
                            connection_type="allied_with",
                            description=f"{faction['name']} is allied with {ally_name}",
                            strength=7
                        )
        
        return factions_data
    
    async def generate_cultural_elements(self, environment_desc: str, factions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate cultural elements based on environment and established factions.
        
        Args:
            environment_desc: Description of the game environment
            factions: Previously generated factions
            
        Returns:
            List of generated cultural elements
        """
        faction_names = [faction["name"] for faction in factions]
        
        prompt = f"""
        Generate 4-7 unique cultural elements (traditions, customs, taboos) for this setting:
        
        Environment: {environment_desc}
        
        Factions: {', '.join(faction_names)}
        
        For each cultural element, provide this information as a JSON array of objects:
        1. "name": Name of the tradition/custom/ritual/taboo
        2. "type": Type of cultural element (tradition, ritual, taboo, etc.)
        3. "description": Detailed description (2-3 sentences)
        4. "practiced_by": Array of groups that practice this (can be faction names or "universal")
        5. "historical_origin": Brief origin story
        6. "significance": Number from 1-10 indicating social importance
        
        Include at least:
        - One universal tradition embraced by most factions
        - One taboo that's considered inappropriate in the society
        - One ritual with dominant/submissive connotations
        - One custom associated with a specific faction
        
        Make each element feel authentic to the world and interconnected.
        """
        
        # Get GPT response
        response = await get_chatgpt_response(
            self.conversation_id,
            system_prompt="You are a cultural anthropologist specializing in creating authentic customs and traditions.",
            user_prompt=prompt
        )
        
        # Extract cultural element data from response
        cultural_data = self._extract_json_from_response(response)
        
        # Ensure we have a list of cultural elements
        if not isinstance(cultural_data, list):
            cultural_data = [cultural_data]
        
        # Store cultural elements in database
        for element in cultural_data:
            element_id = await self.lore_manager.add_cultural_element(
                name=element["name"],
                element_type=element["type"],
                description=element["description"],
                practiced_by=element["practiced_by"],
                significance=element["significance"],
                historical_origin=element.get("historical_origin", "")
            )
            
            # Create connections to factions that practice this
            for faction_name in element.get("practiced_by", []):
                if faction_name != "universal" and faction_name in faction_names:
                    for faction in factions:
                        if faction["name"] == faction_name:
                            # Find faction ID (not ideal - would be better to have IDs from generate_factions)
                            faction_rows = await self._find_faction_by_name(faction_name)
                            if faction_rows:
                                faction_id = faction_rows[0]["id"]
                                await self.lore_manager.add_lore_connection(
                                    source_type="Factions",
                                    source_id=faction_id,
                                    target_type="CulturalElements",
                                    target_id=element_id,
                                    connection_type="practices",
                                    description=f"{faction_name} observes the {element['name']}",
                                    strength=8
                                )
        
        return cultural_data
    
    async def generate_historical_events(self, environment_desc: str, world_lore: Dict[str, str], factions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate significant historical events that shaped the world.
        
        Args:
            environment_desc: Description of the game environment
            world_lore: Previously established world lore
            factions: Previously generated factions
            
        Returns:
            List of generated historical events
        """
        world_history = world_lore.get("world_history", "")
        faction_names = [faction["name"] for faction in factions]
        
        prompt = f"""
        Generate 5-7 significant historical events that shaped this world:
        
        Environment: {environment_desc}
        
        Existing World History: {world_history}
        
        Factions: {', '.join(faction_names)}
        
        For each historical event, provide this information as a JSON array of objects:
        1. "name": Name of the historical event
        2. "date_description": When it happened (e.g., "200 years ago", "during the Blood Moon")
        3. "description": Detailed description (3-4 sentences)
        4. "participating_factions": Array of factions involved (can reference established factions)
        5. "consequences": Array of 2-3 major lasting consequences
        6. "significance": Number from 1-10 indicating historical impact
        
        Include events that:
        - Established or changed power dynamics
        - Created lasting rivalries or alliances
        - Shaped current cultural elements
        - Include a mix of recent and ancient history
        
        Ensure events have clear connections to the established factions and cultural elements.
        """
        
        # Get GPT response
        response = await get_chatgpt_response(
            self.conversation_id,
            system_prompt="You are a historical expert specializing in creating meaningful historical narratives.",
            user_prompt=prompt
        )
        
        # Extract historical event data from response
        events_data = self._extract_json_from_response(response)
        
        # Ensure we have a list of events
        if not isinstance(events_data, list):
            events_data = [events_data]
        
        # Store historical events in database
        for event in events_data:
            event_id = await self.lore_manager.add_historical_event(
                name=event["name"],
                description=event["description"],
                date_description=event["date_description"],
                significance=event["significance"],
                participating_factions=event.get("participating_factions", []),
                consequences=event.get("consequences", [])
            )
            
            # Create connections to participating factions
            for faction_name in event.get("participating_factions", []):
                if faction_name in faction_names:
                    # Find faction ID 
                    faction_rows = await self._find_faction_by_name(faction_name)
                    if faction_rows:
                        faction_id = faction_rows[0]["id"]
                        await self.lore_manager.add_lore_connection(
                            source_type="HistoricalEvents",
                            source_id=event_id,
                            target_type="Factions",
                            target_id=faction_id,
                            connection_type="involves",
                            description=f"{event['name']} involved {faction_name}",
                            strength=6
                        )
        
        return events_data
    
    async def generate_locations(self, environment_desc: str, factions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate significant locations within the game world.
        
        Args:
            environment_desc: Description of the game environment
            factions: Previously generated factions
            
        Returns:
            List of generated locations
        """
        faction_names = [faction["name"] for faction in factions]
        
        prompt = f"""
        Generate 5-8 significant locations within this world:
        
        Environment: {environment_desc}
        
        Factions: {', '.join(faction_names)}
        
        For each location, provide this information as a JSON array of objects:
        1. "name": Name of the location
        2. "description": Detailed description (3-4 sentences)
        3. "type": Type of location (settlement, landmark, institution, etc.)
        4. "controlling_faction": Which faction controls it (can reference established factions)
        5. "notable_features": Array of 2-3 distinctive features
        6. "hidden_secrets": Array of 1-2 secrets or mysteries about this place
        7. "strategic_importance": Brief explanation of why this location matters
        
        Include diverse locations:
        - At least one major settlement
        - At least one mysterious or magical location
        - At least one location with historical significance
        - At least one headquarters or stronghold of a faction
        
        Make locations feel authentic to the world and provide opportunities for intrigue.
        """
        
        # Get GPT response
        response = await get_chatgpt_response(
            self.conversation_id,
            system_prompt="You are a worldbuilding expert specializing in creating memorable and distinctive locations.",
            user_prompt=prompt
        )
        
        # Extract location data from response
        locations_data = self._extract_json_from_response(response)
        
        # Ensure we have a list of locations
        if not isinstance(locations_data, list):
            locations_data = [locations_data]
        
        # Store locations in database
        for location in locations_data:
            # First add to Locations table (this is from game code, not lore system)
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO Locations (user_id, conversation_id, location_name, description)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (self.user_id, self.conversation_id, location["name"], location["description"]))
                
                location_id = cursor.fetchone()[0]
                conn.commit()
                
                # Then add to LocationLore table
                await self.lore_manager.add_location_lore(
                    location_id=location_id,
                    founding_story=location.get("founding_story", f"Founded as a {location['type']}"),
                    hidden_secrets=location.get("hidden_secrets", []),
                    local_legends=location.get("local_legends", []),
                    historical_significance=location.get("historical_significance", location.get("strategic_importance", ""))
                )
                
                # Create connections to controlling faction
                controlling_faction = location.get("controlling_faction")
                if controlling_faction in faction_names:
                    # Find faction ID
                    faction_rows = await self._find_faction_by_name(controlling_faction)
                    if faction_rows:
                        faction_id = faction_rows[0]["id"]
                        await self.lore_manager.add_lore_connection(
                            source_type="Factions",
                            source_id=faction_id,
                            target_type="LocationLore",
                            target_id=location_id,
                            connection_type="controls",
                            description=f"{controlling_faction} controls {location['name']}",
                            strength=9
                        )
            finally:
                cursor.close()
                conn.close()
        
        return locations_data
    
    async def generate_quest_hooks(self, factions: List[Dict[str, Any]], locations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate potential quest hooks tied to lore elements.
        
        Args:
            factions: Previously generated factions
            locations: Previously generated locations
            
        Returns:
            List of generated quest hooks
        """
        faction_names = [faction["name"] for faction in factions]
        location_names = [location["name"] for location in locations]
        
        prompt = f"""
        Generate 5-7 engaging quest hooks tied to the established lore:
        
        Factions: {', '.join(faction_names)}
        
        Locations: {', '.join(location_names)}
        
        For each quest hook, provide this information as a JSON array of objects:
        1. "quest_name": Catchy name for the quest
        2. "quest_giver": Who offers this quest (can be a faction or specific NPC)
        3. "location": Where this quest primarily takes place
        4. "description": Detailed description (3-4 sentences)
        5. "objectives": Array of 2-3 major objectives
        6. "rewards": Potential rewards for completion
        7. "difficulty": Easy, Medium, Hard, or Very Hard
        8. "lore_significance": How this quest connects to the wider lore (1-2 sentences)
        
        Include diverse quest types:
        - At least one faction-related power struggle
        - At least one mysterious artifact or secret
        - At least one morally ambiguous situation
        - A mix of difficulties
        
        Make quests feel integrated with established lore and provide opportunities for character development.
        """
        
        # Get GPT response
        response = await get_chatgpt_response(
            self.conversation_id,
            system_prompt="You are a quest designer specializing in creating engaging story hooks that connect to world lore.",
            user_prompt=prompt
        )
        
        # Extract quest data from response
        quests_data = self._extract_json_from_response(response)
        
        # Ensure we have a list of quests
        if not isinstance(quests_data, list):
            quests_data = [quests_data]
        
        # Store quests in database (using Quests table from game code)
        for quest in quests_data:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO Quests (
                        user_id, conversation_id, quest_name, status, progress_detail, 
                        quest_giver, reward
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING quest_id
                """, (
                    self.user_id, 
                    self.conversation_id, 
                    quest["quest_name"], 
                    "Available",  # Initial status
                    quest["description"],
                    quest["quest_giver"],
                    quest["rewards"]
                ))
                
                quest_id = cursor.fetchone()[0]
                conn.commit()
                
                # Store lore connection information
                quest_lore_key = f"quest_lore:{self.user_id}:{self.conversation_id}:{quest_id}"
                LORE_CACHE.set(quest_lore_key, {
                    "lore_significance": quest.get("lore_significance", ""),
                    "objectives": quest.get("objectives", []),
                    "difficulty": quest.get("difficulty", "Medium"),
                    "location": quest.get("location", "")
                }, 3600)  # Cache for 1 hour
                
            finally:
                cursor.close()
                conn.close()
        
        return quests_data
    
    async def generate_complete_lore(self, environment_desc: str) -> Dict[str, Any]:
        """
        Generate a complete, interconnected set of lore for a game world.
        
        Args:
            environment_desc: Description of the game environment
            
        Returns:
            Dict containing all generated lore elements
        """
        # Generate foundation lore
        world_lore = await self.initialize_world_lore(environment_desc)
        
        # Generate factions based on setting analysis
        factions = await self.generate_factions_from_setting_analysis(environment_desc, world_lore)
        
        # Generate cultural elements
        cultural_elements = await self.generate_cultural_elements(environment_desc, factions)
        
        # Generate historical events
        historical_events = await self.generate_historical_events(environment_desc, world_lore, factions)
        
        # Generate locations
        locations = await self.generate_locations_from_setting_analysis(environment_desc, factions)
        
        # Generate quest hooks
        quests = await self.generate_quest_hooks(factions, locations)
        
        # Return all generated lore
        return {
            "world_lore": world_lore,
            "factions": factions,
            "cultural_elements": cultural_elements,
            "historical_events": historical_events,
            "locations": locations,
            "quests": quests
        }
    
    # Add this method to be consistent with the setting analysis approach
    async def generate_locations_from_setting_analysis(self, environment_desc: str, factions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate significant locations within the game world, focusing on faction gathering places.
        
        Args:
            environment_desc: Description of the game environment
            factions: Previously generated factions
            
        Returns:
            List of generated locations
        """
        # Get existing locations from NPCs
        analyzer = SettingAnalyzer(self.user_id, self.conversation_id)
        data = await analyzer.aggregate_npc_data()
        existing_locations = data.get("aggregated", {}).get("locations", [])
        
        # Get faction gathering places
        faction_locations = [faction.get("headquarters", "") for faction in factions if "headquarters" in faction]
        
        # Combine all location names for the prompt
        all_mentioned_locations = list(set(existing_locations + faction_locations))
        mentioned_locations_text = ", ".join(all_mentioned_locations) if all_mentioned_locations else "None explicitly mentioned"
        
        # Create list of faction names for reference
        faction_names = [faction["name"] for faction in factions]
        faction_names_text = ", ".join(faction_names)
        
        prompt = f"""
        Generate 5-8 significant locations within this world:
        
        Environment: {environment_desc}
        
        Factions: {faction_names_text}
        
        Existing Locations Mentioned: {mentioned_locations_text}
        
        For each location, provide this information as a JSON array of objects:
        1. "name": Name of the location (USE EXACTLY the names already mentioned when applicable)
        2. "description": Detailed description (3-4 sentences)
        3. "type": Type of location (settlement, landmark, institution, etc.)
        4. "controlling_faction": Which faction controls it (reference established factions when applicable)
        5. "notable_features": Array of 2-3 distinctive features
        6. "hidden_secrets": Array of 1-2 secrets or mysteries about this place
        7. "strategic_importance": Brief explanation of why this location matters
        
        Include a diverse mix of locations:
        - At least one major gathering place for each main faction type (academic, athletic, social, etc.)
        - At least one mysterious or hidden location
        - At least one public space where different groups interact
        - At least one exclusive location with restricted access
        
        Make locations feel authentic to the world and provide opportunities for intrigue.
        """
        
        # Get GPT response
        response = await get_chatgpt_response(
            self.conversation_id,
            system_prompt="You are a worldbuilding expert specializing in creating memorable and distinctive locations.",
            user_prompt=prompt
        )
        
        # Extract location data from response
        locations_data = self._extract_json_from_response(response)
        
        # Ensure we have a list of locations
        if not isinstance(locations_data, list):
            locations_data = [locations_data]
        
        # Store locations in database
        for location in locations_data:
            # Skip if it doesn't have the required fields
            if not all(field in location for field in ["name", "description"]):
                continue
                
            # First add to Locations table (this is from game code, not lore system)
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO Locations (user_id, conversation_id, location_name, description)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (self.user_id, self.conversation_id, location["name"], location["description"]))
                
                location_id = cursor.fetchone()[0]
                conn.commit()
                
                # Then add to LocationLore table
                await self.lore_manager.add_location_lore(
                    location_id=location_id,
                    founding_story=location.get("founding_story", f"Founded as a {location['type']}"),
                    hidden_secrets=location.get("hidden_secrets", []),
                    local_legends=location.get("local_legends", []),
                    historical_significance=location.get("historical_significance", location.get("strategic_importance", ""))
                )
                
                # Create connections to controlling faction
                controlling_faction = location.get("controlling_faction")
                if controlling_faction:
                    # Find matching faction
                    for faction in factions:
                        if faction["name"] == controlling_faction:
                            # Find faction ID (would be better to have IDs directly)
                            faction_rows = await self._find_faction_by_name(controlling_faction)
                            if faction_rows:
                                faction_id = faction_rows[0]["id"]
                                await self.lore_manager.add_lore_connection(
                                    source_type="Factions",
                                    source_id=faction_id,
                                    target_type="LocationLore",
                                    target_id=location_id,
                                    connection_type="controls",
                                    description=f"{controlling_faction} controls {location['name']}",
                                    strength=9
                                )
            finally:
                cursor.close()
                conn.close()
        
        return locations_data
    async def evolve_lore_with_event(self, event_description: str) -> Dict[str, Any]:
        """
        Evolve existing lore based on a new major event.
        
        Args:
            event_description: Description of the new event that impacts the world
            
        Returns:
            Dict of lore updates 
        """
        # Get relevant existing lore
        world_lore_items = await self.lore_manager.get_world_lore_by_category("world_history")
        factions = await self._get_all_factions()
        
        faction_names = [faction.get("name", "Unknown") for faction in factions]
        
        prompt = f"""
        A major event has occurred in the game world:
        
        {event_description}
        
        How does this event impact the existing lore? Generate lore updates as a JSON object with these keys:
        
        1. "new_historical_event": Details of this event as a new historical record
        2. "faction_updates": Array of objects showing how factions are affected
        3. "power_shifts": Any changes in faction influence or territory
        4. "new_cultural_elements": Any new traditions or customs arising from this event
        5. "quest_opportunities": New potential quest hooks resulting from this event
        
        Existing factions: {', '.join(faction_names)}
        
        Make all updates feel consistent with existing lore while allowing the world to evolve naturally.
        """
        
        # Get GPT response
        response = await get_chatgpt_response(
            self.conversation_id,
            system_prompt="You are a lore master who evolves game worlds in response to major events.",
            user_prompt=prompt
        )
        
        # Extract lore updates from response
        updates = self._extract_json_from_response(response)
        
        # Apply the updates to the database
        try:
            # Add new historical event
            new_event = updates.get("new_historical_event", {})
            if new_event:
                event_id = await self.lore_manager.add_historical_event(
                    name=new_event.get("name", "Unnamed Event"),
                    description=new_event.get("description", ""),
                    date_description="Recent event",
                    significance=new_event.get("significance", 8),
                    participating_factions=new_event.get("participating_factions", []),
                    consequences=new_event.get("consequences", [])
                )
            
            # Process faction updates
            faction_updates = updates.get("faction_updates", [])
            for update in faction_updates:
                faction_name = update.get("faction_name")
                if faction_name:
                    faction_rows = await self._find_faction_by_name(faction_name)
                    if faction_rows:
                        faction_id = faction_rows[0]["id"]
                        
                        # Create a memory entry about this change
                        await self._create_faction_update_memory(
                            faction_id, 
                            update.get("update_description", f"Affected by: {event_description}")
                        )
            
            # Add new cultural elements
            new_cultural_elements = updates.get("new_cultural_elements", [])
            for element in new_cultural_elements:
                if isinstance(element, dict) and "name" in element:
                    await self.lore_manager.add_cultural_element(
                        name=element["name"],
                        element_type=element.get("type", "custom"),
                        description=element.get("description", ""),
                        practiced_by=element.get("practiced_by", []),
                        significance=element.get("significance", 6),
                        historical_origin=f"Arose from: {event_description}"
                    )
        
        except Exception as e:
            logging.error(f"Error evolving lore: {e}")
            return {"error": str(e)}
        
        return updates
    
    async def _create_faction_update_memory(self, faction_id: int, update_text: str) -> None:
        """Create a memory/history entry for a faction update."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO FactionHistory (
                    user_id, conversation_id, faction_id, event_text, timestamp
                ) VALUES (%s, %s, %s, %s, NOW())
            """, (self.user_id, self.conversation_id, faction_id, update_text))
            
            conn.commit()
        except Exception as e:
            logging.error(f"Error creating faction update memory: {e}")
        finally:
            cursor.close()
            conn.close()
    
    async def _find_faction_by_name(self, faction_name: str) -> List[Dict[str, Any]]:
        """Find a faction by name in the database."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Try approximate matching to be more robust
            cursor.execute("""
                SELECT id, name FROM Factions
                WHERE LOWER(name) LIKE LOWER(%s)
                LIMIT 1
            """, (f"%{faction_name}%",))
            
            columns = [col[0] for col in cursor.description]
            result = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            return result
        finally:
            cursor.close()
            conn.close()
    
    async def _get_all_factions(self) -> List[Dict[str, Any]]:
        """Retrieve all factions from the database."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, name, type, description FROM Factions
            """)
            
            columns = [col[0] for col in cursor.description]
            result = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            return result
        finally:
            cursor.close()
            conn.close()
    
    def get_setting_name(self) -> str:
        """Get the current setting name from the conversation."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=%s AND conversation_id=%s AND key='CurrentSetting'
                LIMIT 1
            """, (self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            if row:
                return row[0]
            else:
                return "The Setting"
        except Exception:
            return "The Setting"
        finally:
            cursor.close()
            conn.close()
    
    def _extract_json_from_response(self, response):
        """Extract JSON from GPT response."""
        try:
            if isinstance(response, dict):
                if "function_args" in response:
                    return response["function_args"]
                elif "response" in response:
                    response_text = response["response"]
                else:
                    response_text = str(response)
            else:
                response_text = str(response)
            
            # Try to find JSON in the response
            try:
                # First try to parse the entire response
                return json.loads(response_text)
            except json.JSONDecodeError:
                # If that fails, look for JSON between code blocks
                import re
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if match:
                    return json.loads(match.group(1))
                
                # Try finding the first { and last }
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(response_text[start:end])
            
            logging.warning(f"Failed to extract JSON from response: {response_text[:200]}...")
            return {}
            
        except Exception as e:
            logging.error(f"Error extracting JSON from response: {e}")
            return {}
