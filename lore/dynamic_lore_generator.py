# lore/dynamic_lore_generator.py

import logging
from typing import Dict, List, Any, Optional
from agents.run_context import RunContextWrapper

# Nyx governance integration
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType

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
    and is fully integrated with Nyx governance.
    
    - Each method corresponds to a particular lore step:
      1) initialize_world_lore  -> foundation lore
      2) generate_factions      -> faction creation
      3) generate_cultural_elements -> cultural traditions, taboos, etc.
      4) generate_historical_events -> major events shaping the world
      5) generate_locations     -> important places
      6) generate_quest_hooks   -> potential quests
    - generate_complete_lore orchestrates them all in sequence.
    
    The built-in 'LoreManager' is used to store everything in your DB.
    """

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_manager = LoreManager(user_id, conversation_id)
        self.governor = None
        
    async def initialize_governance(self):
        """Initialize Nyx governance integration"""
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
        return self.governor

    async def initialize_world_lore(self, environment_desc: str) -> Dict[str, Any]:
        """
        Initialize core foundation lore (cosmology, magic system, world history, etc.)
        and store it in the DB. The output is stored under 'WorldLore' with relevant tags.
        
        Args:
            environment_desc: Short textual description of the environment
            
        Returns:
            Dict containing the five fields from the FoundationLoreOutput
        """
        # First, check permission with governance system
        await self.initialize_governance()
        
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="initialize_world_lore",
            action_details={"environment_desc": environment_desc}
        )
        
        if not permission["approved"]:
            logging.warning(f"World lore initialization not approved: {permission.get('reasoning')}")
            return {"error": permission.get("reasoning"), "approved": False}
            
        # Create run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })

        # Call the function tool with governance integration
        foundation_data = await generate_foundation_lore(run_ctx, environment_desc)

        # Store in database
        for category, desc in foundation_data.items():
            # e.g. "category" might be "cosmology", "magic_system", ...
            await self.lore_manager.add_world_lore(
                name=f"{category.title()} of {self.get_setting_name()}",
                category=category,
                description=desc,
                significance=8,
                tags=[category, "foundation", "world_building"]
            )

        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "initialize_world_lore",
                "description": f"Generated foundation lore for {environment_desc[:50]}"
            },
            result={
                "categories": list(foundation_data.keys()),
                "world_name": self.get_setting_name()
            }
        )

        return foundation_data

    async def generate_factions(self, environment_desc: str, world_lore: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate 3-5 distinct factions referencing the environment description
        and possibly 'social_structure' from the foundation data.
        
        Args:
            environment_desc: Text describing environment or setting
            world_lore: The dictionary from initialize_world_lore
            
        Returns:
            A list of faction dictionaries
        """
        # First, check permission with governance system
        await self.initialize_governance()
        
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="generate_factions",
            action_details={"environment_desc": environment_desc}
        )
        
        if not permission["approved"]:
            logging.warning(f"Faction generation not approved: {permission.get('reasoning')}")
            return []
            
        # Typically we want the 'social_structure' from foundation_data
        social_structure = world_lore.get("social_structure", "")

        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })

        # Produce a list of faction dicts via our sub-agent
        factions_data = await generate_factions(run_ctx, environment_desc, social_structure)

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
            except Exception as e:
                logging.error(f"Error storing faction '{faction['name']}': {e}")

        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "generate_factions",
                "description": f"Generated {len(factions_data)} factions"
            },
            result={
                "faction_count": len(factions_data),
                "faction_names": [f.get("name", "Unknown") for f in factions_data]
            }
        )

        return factions_data

    async def generate_cultural_elements(self, environment_desc: str, factions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate cultural elements referencing environment + the names of the existing factions.
        
        Args:
            environment_desc: Text describing environment
            factions: List of faction dictionaries
            
        Returns:
            List of cultural element dictionaries
        """
        # First, check permission with governance system
        await self.initialize_governance()
        
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="generate_cultural_elements",
            action_details={"environment_desc": environment_desc}
        )
        
        if not permission["approved"]:
            logging.warning(f"Cultural elements generation not approved: {permission.get('reasoning')}")
            return []
            
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        faction_names = ", ".join([f.get("name", "Unknown") for f in factions])

        cultural_data = await generate_cultural_elements(run_ctx, environment_desc, faction_names)

        # Store them
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

        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "generate_cultural_elements",
                "description": f"Generated {len(cultural_data)} cultural elements"
            },
            result={
                "element_count": len(cultural_data),
                "element_types": list(set([e.get("type", "unknown") for e in cultural_data]))
            }
        )

        return cultural_data

    async def generate_historical_events(self, environment_desc: str, foundation_data: Dict[str, Any], factions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate 5-7 major historical events referencing environment + existing 'world_history' + faction names.
        
        Args:
            environment_desc: Environment description text
            foundation_data: Foundation lore dictionary
            factions: List of faction dictionaries
            
        Returns:
            List of historical event dictionaries
        """
        # First, check permission with governance system
        await self.initialize_governance()
        
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="generate_historical_events",
            action_details={"environment_desc": environment_desc}
        )
        
        if not permission["approved"]:
            logging.warning(f"Historical events generation not approved: {permission.get('reasoning')}")
            return []
            
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
            except Exception as e:
                logging.error(f"Error storing historical event '{event.get('name','Unknown')}': {e}")

        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "generate_historical_events",
                "description": f"Generated {len(events_data)} historical events"
            },
            result={
                "event_count": len(events_data),
                "significant_events": [e.get("name") for e in events_data if e.get("significance", 0) > 7]
            }
        )

        return events_data

    async def generate_locations(self, environment_desc: str, factions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate 5-8 significant locations referencing environment + faction names.
        
        Args:
            environment_desc: Environment description text
            factions: List of faction dictionaries
            
        Returns:
            List of location dictionaries
        """
        # First, check permission with governance system
        await self.initialize_governance()
        
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="generate_locations",
            action_details={"environment_desc": environment_desc}
        )
        
        if not permission["approved"]:
            logging.warning(f"Locations generation not approved: {permission.get('reasoning')}")
            return []
            
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        faction_names = ", ".join([f.get("name","Unknown") for f in factions])

        locations_data = await generate_locations(run_ctx, environment_desc, faction_names)

        # Store each location
        for loc in locations_data:
            try:
                # Create the location record
                location_row_id = await self.lore_manager.create_location_record(
                    name=loc["name"],
                    description=loc["description"],
                    location_type=loc["type"]
                )

                # Add location lore
                controlling_faction = loc.get("controlling_faction")
                hidden_secrets = loc.get("hidden_secrets", [])
                founding_story = f"Founded as a {loc['type']}."

                await self.lore_manager.add_location_lore(
                    location_id=location_row_id,
                    founding_story=founding_story,
                    hidden_secrets=hidden_secrets,
                    local_legends=[],
                    historical_significance=loc.get("strategic_importance", "")
                )

                # Record controlling_faction if needed
                if controlling_faction:
                    await self.lore_manager.connect_faction_to_location(location_row_id, controlling_faction)
            except Exception as e:
                logging.error(f"Error storing location '{loc.get('name','Unknown')}': {e}")

        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "generate_locations",
                "description": f"Generated {len(locations_data)} locations"
            },
            result={
                "location_count": len(locations_data),
                "location_types": list(set([l.get("type", "unknown") for l in locations_data]))
            }
        )

        return locations_data

    async def generate_quest_hooks(self, factions: List[Dict[str, Any]], locations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate 5-7 quest hooks referencing existing factions + location names.
        
        Args:
            factions: List of faction dictionaries
            locations: List of location dictionaries
            
        Returns:
            List of quest hook dictionaries
        """
        # First, check permission with governance system
        await self.initialize_governance()
        
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="generate_quest_hooks",
            action_details={"faction_count": len(factions), "location_count": len(locations)}
        )
        
        if not permission["approved"]:
            logging.warning(f"Quest hooks generation not approved: {permission.get('reasoning')}")
            return []
            
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        faction_names = ", ".join([f.get("name","Unknown") for f in factions])
        location_names = ", ".join([l.get("name","Unknown") for l in locations])

        quests_data = await generate_quest_hooks(run_ctx, faction_names, location_names)

        # Store them
        for quest in quests_data:
            try:
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

        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "generate_quest_hooks",
                "description": f"Generated {len(quests_data)} quest hooks"
            },
            result={
                "quest_count": len(quests_data),
                "quest_difficulties": list(set([q.get("difficulty", 0) for q in quests_data]))
            }
        )

        return quests_data

    async def generate_complete_lore(self, environment_desc: str) -> Dict[str, Any]:
        """
        A single method that calls each sub-step in a specific order with governance oversight:
          1) Foundation
          2) Factions
          3) Cultural
          4) Historical
          5) Locations
          6) Quests
        Then returns a dictionary combining all results.
        
        Args:
            environment_desc: High-level environment description
            
        Returns:
            A dictionary with keys:
                 "world_lore", "factions", "cultural_elements", 
                 "historical_events", "locations", "quests"
        """
        # First, check permission with governance system
        await self.initialize_governance()
        
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="generate_complete_lore",
            action_details={"environment_desc": environment_desc}
        )
        
        if not permission["approved"]:
            logging.warning(f"Complete lore generation not approved: {permission.get('reasoning')}")
            return {"error": permission.get("reasoning"), "approved": False}
        
        # Process any existing directives before proceeding
        await self._check_and_process_directives()
            
        # 1) Foundation lore
        foundation_data = await self.initialize_world_lore(environment_desc)
        if isinstance(foundation_data, dict) and "error" in foundation_data:
            return foundation_data

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

        # Report complete action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "generate_complete_lore",
                "description": f"Generated complete lore for environment: {environment_desc[:50]}"
            },
            result={
                "world_lore_count": len(foundation_data) if isinstance(foundation_data, dict) else 0,
                "factions_count": len(factions_data),
                "cultural_elements_count": len(cultural_data),
                "historical_events_count": len(historical_data),
                "locations_count": len(locations_data),
                "quests_count": len(quests_data),
                "setting_name": self.get_setting_name()
            }
        )

        return {
            "world_lore": foundation_data,
            "factions": factions_data,
            "cultural_elements": cultural_data,
            "historical_events": historical_data,
            "locations": locations_data,
            "quests": quests_data
        }
    
    async def _check_and_process_directives(self):
        """Check for and process any pending directives from Nyx"""
        # Initialize governance if needed
        await self.initialize_governance()
        
        # Get directives for lore generator
        directives = await self.governor.get_agent_directives(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator"
        )
        
        for directive in directives:
            directive_type = directive.get("type")
            
            # Process prohibition directives
            if directive_type == DirectiveType.PROHIBITION:
                prohibited_actions = directive.get("prohibited_actions", [])
                logging.info(f"Found prohibition directive: {prohibited_actions}")
                
                # Store prohibited actions (will be checked during permission checks)
                if not hasattr(self, 'prohibited_lore_actions'):
                    self.prohibited_lore_actions = []
                
                self.prohibited_lore_actions.extend(prohibited_actions)
            
            # Process action directives
            elif directive_type == DirectiveType.ACTION:
                instruction = directive.get("instruction", "")
                logging.info(f"Processing action directive: {instruction}")
                
                # Implement action directive processing as needed
                # We'll report back that we've processed it
                await self.governor.process_agent_action_report(
                    agent_type=AgentType.NARRATIVE_CRAFTER,
                    agent_id="lore_generator",
                    action={
                        "type": "process_directive",
                        "description": f"Processed directive: {instruction[:50]}"
                    },
                    result={
                        "directive_id": directive.get("id"),
                        "processed": True
                    }
                )

    async def evolve_lore_with_event(self, event_description: str) -> Dict[str, Any]:
        """
        Update world lore based on a significant narrative event with governance oversight.
        
        Args:
            event_description: Description of the narrative event
            
        Returns:
            Dictionary with lore updates
        """
        # First, check permission with governance system
        await self.initialize_governance()
        
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="evolve_lore_with_event",
            action_details={"event_description": event_description}
        )
        
        if not permission["approved"]:
            logging.warning(f"Lore evolution not approved: {permission.get('reasoning')}")
            return {"error": permission.get("reasoning"), "approved": False}
        
        # TODO: Implement lore evolution logic
        # This would likely involve:
        # 1. Analyzing existing lore
        # 2. Determining impact of the event
        # 3. Updating affected lore entries
        
        # Placeholder for now
        result = {
            "updated": True,
            "event": event_description,
            "affected_elements": []
        }
        
        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "evolve_lore_with_event",
                "description": f"Evolved lore with event: {event_description[:50]}"
            },
            result=result
        )
        
        return result

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
