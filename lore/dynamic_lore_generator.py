# lore/dynamic_lore_generator.py

import logging
from typing import Dict, List, Any, Optional, Set
from agents.run_context import RunContextWrapper
from datetime import datetime
from .lore_cache_manager import LoreCacheManager
from .base_manager import BaseManager
from .resource_manager import resource_manager

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

import json
import copy
import asyncio

logger = logging.getLogger(__name__)

class DynamicLoreGenerator(BaseManager):
    """Generator for dynamic lore with resource management support."""
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        max_size_mb: float = 100,
        redis_url: Optional[str] = None
    ):
        super().__init__(user_id, conversation_id, max_size_mb, redis_url)
        self.dynamic_data = {}
        self.generation_stats = {}
        self.performance_metrics = {}
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_manager = LoreManager(user_id, conversation_id)
        self.governor = None
        self.resource_manager = resource_manager
        self.evolution_agent = None
        self.trigger_agent = None
        self.planning_agent = None
        self.evolution_history = []
        self.active_triggers = set()
        self.evolution_queue = asyncio.Queue()
        
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
        
        # Evolve lore using the implemented evolution logic
        evolved_lore = await self._evolve_lore(
            {"event": event_description},
            {"type": "event_evolution"}
        )
        
        result = {
            "updated": True,
            "event": event_description,
            "affected_elements": evolved_lore.get("affected_elements", []),
            "evolution_history": evolved_lore.get("evolution_history", {}),
            "future_implications": evolved_lore.get("future_implications", {})
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

    async def _evolve_lore(
        self,
        lore: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evolve lore based on context and world state."""
        try:
            # Analyze potential evolution triggers
            triggers = await self._analyze_evolution_triggers(lore, context)
            
            if not triggers:
                logger.info("No evolution triggers found")
                return lore
                
            # Generate evolution plan
            evolution_plan = await self._generate_evolution_plan(
                lore,
                triggers,
                context
            )
            
            # Apply evolution
            evolved_lore = await self._apply_evolution(
                lore,
                evolution_plan,
                context
            )
            
            # Validate evolved lore
            validated_lore = await self._validate_evolution(
                evolved_lore,
                lore,
                context
            )
            
            # Enhance evolved lore
            enhanced_lore = await self._enhance_evolution(
                validated_lore,
                lore,
                context
            )
            
            return enhanced_lore
            
        except Exception as e:
            logger.error(f"Failed to evolve lore: {str(e)}")
            raise
            
    async def _analyze_evolution_triggers(
        self,
        lore: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze potential evolution triggers."""
        try:
            triggers = []
            
            # Check time-based triggers
            time_triggers = await self._check_time_triggers(lore, context)
            triggers.extend(time_triggers)
            
            # Check event-based triggers
            event_triggers = await self._check_event_triggers(lore, context)
            triggers.extend(event_triggers)
            
            # Check state-based triggers
            state_triggers = await self._check_state_triggers(lore, context)
            triggers.extend(state_triggers)
            
            # Check relationship-based triggers
            relationship_triggers = await self._check_relationship_triggers(
                lore,
                context
            )
            triggers.extend(relationship_triggers)
            
            return triggers
            
        except Exception as e:
            logger.error(f"Failed to analyze evolution triggers: {str(e)}")
            raise
            
    async def _generate_evolution_plan(
        self,
        lore: Dict[str, Any],
        triggers: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate evolution plan."""
        try:
            # Build prompt for evolution plan
            prompt = f"""
            Generate an evolution plan for the lore:
            
            Current Lore:
            {json.dumps(lore, indent=2)}
            
            Evolution Triggers:
            {json.dumps(triggers, indent=2)}
            
            Context:
            {json.dumps(context, indent=2)}
            
            Generate a plan with:
            1. Evolution steps
            2. Dependencies between steps
            3. Constraints and requirements
            4. Timeline and order
            5. Impact analysis
            6. Risk assessment
            7. Quality metrics
            
            Return as JSON with fields:
            - steps: List of evolution steps
            - dependencies: Map of step dependencies
            - constraints: List of constraints
            - timeline: Ordered list of steps
            - impact_analysis: Analysis of potential impacts
            - risks: List of potential risks
            - quality_metrics: Expected quality metrics
            """
            
            # Generate plan using LLM
            plan = await self._generate_with_llm(prompt, context)
            
            # Validate plan
            validated_plan = await self._validate_evolution_plan(
                plan,
                lore,
                context
            )
            
            # Enhance plan
            enhanced_plan = await self._enhance_evolution_plan(
                validated_plan,
                lore,
                context
            )
            
            return enhanced_plan
            
        except Exception as e:
            logger.error(f"Failed to generate evolution plan: {str(e)}")
            raise
            
    async def _apply_evolution(
        self,
        lore: Dict[str, Any],
        plan: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply evolution plan to lore."""
        try:
            evolved_lore = copy.deepcopy(lore)
            
            # Apply each step in order
            for step in plan["timeline"]:
                # Check dependencies
                if not await self._check_dependencies(step, plan, evolved_lore):
                    logger.warning(f"Skipping step {step['id']}: dependencies not met")
                    continue
                    
                # Apply step
                evolved_lore = await self._apply_evolution_step(
                    evolved_lore,
                    step,
                    context
                )
                
                # Update relationships
                evolved_lore = await self._update_relationships(
                    evolved_lore,
                    step,
                    context
                )
                
                # Update metadata
                evolved_lore = await self._update_evolution_metadata(
                    evolved_lore,
                    step,
                    context
                )
                
            return evolved_lore
            
        except Exception as e:
            logger.error(f"Failed to apply evolution: {str(e)}")
            raise
            
    async def _validate_evolution(
        self,
        evolved_lore: Dict[str, Any],
        original_lore: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate evolved lore."""
        try:
            # Check consistency
            if not await self._check_evolution_consistency(
                evolved_lore,
                original_lore,
                context
            ):
                raise ValueError("Evolution consistency check failed")
                
            # Check coherence
            if not await self._check_evolution_coherence(
                evolved_lore,
                original_lore,
                context
            ):
                raise ValueError("Evolution coherence check failed")
                
            # Check quality
            if not await self._check_evolution_quality(
                evolved_lore,
                original_lore,
                context
            ):
                raise ValueError("Evolution quality check failed")
                
            # Fix any issues found
            fixed_lore = await self._fix_evolution_issues(
                evolved_lore,
                original_lore,
                context
            )
            
            return fixed_lore
            
        except Exception as e:
            logger.error(f"Failed to validate evolution: {str(e)}")
            raise
            
    async def _enhance_evolution(
        self,
        evolved_lore: Dict[str, Any],
        original_lore: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance evolved lore with additional information."""
        try:
            enhanced_lore = copy.deepcopy(evolved_lore)
            
            # Add evolution history
            enhanced_lore["evolution_history"] = await self._generate_evolution_history(
                evolved_lore,
                original_lore,
                context
            )
            
            # Add future implications
            enhanced_lore["future_implications"] = await self._analyze_future_implications(
                evolved_lore,
                original_lore,
                context
            )
            
            # Add evolution metrics
            enhanced_lore["evolution_metrics"] = await self._calculate_evolution_metrics(
                evolved_lore,
                original_lore,
                context
            )
            
            return enhanced_lore
            
        except Exception as e:
            logger.error(f"Failed to enhance evolution: {str(e)}")
            raise

    async def start(self):
        """Start the dynamic generator and resource management."""
        await super().start()
        await self.resource_manager.start()
    
    async def stop(self):
        """Stop the dynamic generator and cleanup resources."""
        await super().stop()
        await self.resource_manager.stop()
    
    async def get_dynamic_data(
        self,
        dynamic_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get dynamic data from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('dynamic', dynamic_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting dynamic data: {e}")
            return None
    
    async def set_dynamic_data(
        self,
        dynamic_id: str,
        data: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set dynamic data in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('dynamic', dynamic_id, data, tags)
        except Exception as e:
            logger.error(f"Error setting dynamic data: {e}")
            return False
    
    async def invalidate_dynamic_data(
        self,
        dynamic_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate dynamic data cache."""
        try:
            await self.invalidate_cached_data('dynamic', dynamic_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating dynamic data: {e}")
    
    async def get_generation_stats(
        self,
        dynamic_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get generation stats from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('generation_stats', dynamic_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting generation stats: {e}")
            return None
    
    async def set_generation_stats(
        self,
        dynamic_id: str,
        stats: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set generation stats in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('generation_stats', dynamic_id, stats, tags)
        except Exception as e:
            logger.error(f"Error setting generation stats: {e}")
            return False
    
    async def invalidate_generation_stats(
        self,
        dynamic_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate generation stats cache."""
        try:
            await self.invalidate_cached_data('generation_stats', dynamic_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating generation stats: {e}")
    
    async def get_performance_metrics(
        self,
        dynamic_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get performance metrics from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('performance_metrics', dynamic_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return None
    
    async def set_performance_metrics(
        self,
        dynamic_id: str,
        metrics: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set performance metrics in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('performance_metrics', dynamic_id, metrics, tags)
        except Exception as e:
            logger.error(f"Error setting performance metrics: {e}")
            return False
    
    async def invalidate_performance_metrics(
        self,
        dynamic_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate performance metrics cache."""
        try:
            await self.invalidate_cached_data('performance_metrics', dynamic_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating performance metrics: {e}")
    
    async def get_generation_patterns(
        self,
        dynamic_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get generation patterns from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('generation_patterns', dynamic_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting generation patterns: {e}")
            return None
    
    async def set_generation_patterns(
        self,
        dynamic_id: str,
        patterns: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set generation patterns in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('generation_patterns', dynamic_id, patterns, tags)
        except Exception as e:
            logger.error(f"Error setting generation patterns: {e}")
            return False
    
    async def invalidate_generation_patterns(
        self,
        dynamic_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate generation patterns cache."""
        try:
            await self.invalidate_cached_data('generation_patterns', dynamic_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating generation patterns: {e}")
    
    async def get_resource_usage(
        self,
        dynamic_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get resource usage from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('resource_usage', dynamic_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return None
    
    async def set_resource_usage(
        self,
        dynamic_id: str,
        usage: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set resource usage in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('resource_usage', dynamic_id, usage, tags)
        except Exception as e:
            logger.error(f"Error setting resource usage: {e}")
            return False
    
    async def invalidate_resource_usage(
        self,
        dynamic_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate resource usage cache."""
        try:
            await self.invalidate_cached_data('resource_usage', dynamic_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating resource usage: {e}")
    
    async def get_active_tasks(
        self,
        dynamic_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get active tasks from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('active_tasks', dynamic_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting active tasks: {e}")
            return None
    
    async def set_active_tasks(
        self,
        dynamic_id: str,
        tasks: List[Dict[str, Any]],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set active tasks in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('active_tasks', dynamic_id, tasks, tags)
        except Exception as e:
            logger.error(f"Error setting active tasks: {e}")
            return False
    
    async def invalidate_active_tasks(
        self,
        dynamic_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate active tasks cache."""
        try:
            await self.invalidate_cached_data('active_tasks', dynamic_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating active tasks: {e}")
    
    async def get_error_states(
        self,
        dynamic_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get error states from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('error_states', dynamic_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting error states: {e}")
            return None
    
    async def set_error_states(
        self,
        dynamic_id: str,
        states: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set error states in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('error_states', dynamic_id, states, tags)
        except Exception as e:
            logger.error(f"Error setting error states: {e}")
            return False
    
    async def invalidate_error_states(
        self,
        dynamic_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate error states cache."""
        try:
            await self.invalidate_cached_data('error_states', dynamic_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating error states: {e}")
    
    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        try:
            return await self.resource_manager.get_resource_stats()
        except Exception as e:
            logger.error(f"Error getting resource stats: {e}")
            return {}
    
    async def optimize_resources(self):
        """Optimize resource usage."""
        try:
            await self.resource_manager._optimize_resource_usage('memory')
        except Exception as e:
            logger.error(f"Error optimizing resources: {e}")
    
    async def cleanup_resources(self):
        """Clean up unused resources."""
        try:
            await self.resource_manager._cleanup_all_resources()
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")

    async def initialize_agents(self):
        """Initialize the evolution system agents."""
        if not self.evolution_agent:
            self.evolution_agent = await self._create_evolution_agent()
        if not self.trigger_agent:
            self.trigger_agent = await self._create_trigger_agent()
        if not self.planning_agent:
            self.planning_agent = await self._create_planning_agent()
        return True

    async def _create_evolution_agent(self):
        """Create the evolution agent for managing lore changes."""
        return {
            "type": "evolution_agent",
            "capabilities": [
                "analyze_lore_impact",
                "generate_evolution_plan",
                "apply_evolution_changes",
                "validate_evolution_results"
            ],
            "state": {
                "active_evolutions": [],
                "evolution_history": [],
                "impact_metrics": {}
            }
        }

    async def _create_trigger_agent(self):
        """Create the trigger agent for detecting evolution triggers."""
        return {
            "type": "trigger_agent",
            "capabilities": [
                "detect_evolution_triggers",
                "analyze_trigger_impact",
                "prioritize_triggers",
                "validate_triggers"
            ],
            "state": {
                "active_triggers": set(),
                "trigger_history": [],
                "trigger_metrics": {}
            }
        }

    async def _create_planning_agent(self):
        """Create the planning agent for evolution planning."""
        return {
            "type": "planning_agent",
            "capabilities": [
                "generate_evolution_plans",
                "optimize_evolution_paths",
                "validate_evolution_plans",
                "adjust_evolution_parameters"
            ],
            "state": {
                "active_plans": [],
                "plan_history": [],
                "plan_metrics": {}
            }
        }

    async def _evolve_lore(self, trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evolve the lore based on triggers and current state.
        
        Args:
            trigger_data: Data about what triggered the evolution
            
        Returns:
            Evolution result with changes and impact
        """
        await self.initialize_agents()
        
        # Check permission with governance
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_evolution",
            action_type="evolve_lore",
            action_details=trigger_data
        )
        
        if not permission["approved"]:
            return {"error": permission.get("reasoning"), "approved": False}
        
        # Analyze trigger impact
        impact_analysis = await self._analyze_trigger_impact(trigger_data)
        
        # Generate evolution plan
        evolution_plan = await self._generate_evolution_plan(impact_analysis)
        
        # Apply evolution changes
        evolution_result = await self._apply_evolution_changes(evolution_plan)
        
        # Validate results
        validation_result = await self._validate_evolution_results(evolution_result)
        
        # Update evolution history
        self.evolution_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "trigger": trigger_data,
            "impact": impact_analysis,
            "plan": evolution_plan,
            "result": evolution_result,
            "validation": validation_result
        })
        
        # Report to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_evolution",
            action={
                "type": "evolve_lore",
                "description": f"Evolved lore based on trigger: {trigger_data.get('type')}"
            },
            result={
                "impact_level": impact_analysis.get("impact_level"),
                "changes_made": len(evolution_result.get("changes", [])),
                "validation_status": validation_result.get("status")
            }
        )
        
        return evolution_result

    async def _analyze_trigger_impact(self, trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the impact of a trigger on the lore."""
        impact_analysis = {
            "impact_level": 0,
            "affected_components": [],
            "potential_changes": [],
            "risk_assessment": {},
            "opportunity_assessment": {}
        }
        
        # Get current lore state
        current_lore = await self.lore_manager.get_world_lore()
        
        # Analyze impact on different components
        for component_type in ["locations", "npcs", "factions", "events", "artifacts"]:
            components = current_lore.get(component_type, [])
            affected = []
            
            for component in components:
                # Check if component is affected by trigger
                if self._is_component_affected(component, trigger_data):
                    affected.append(component)
                    impact_analysis["impact_level"] += 1
            
            if affected:
                impact_analysis["affected_components"].append({
                    "type": component_type,
                    "components": affected
                })
        
        # Generate potential changes
        impact_analysis["potential_changes"] = await self._generate_potential_changes(
            impact_analysis["affected_components"],
            trigger_data
        )
        
        # Assess risks and opportunities
        impact_analysis["risk_assessment"] = await self._assess_evolution_risks(
            impact_analysis["potential_changes"]
        )
        impact_analysis["opportunity_assessment"] = await self._assess_evolution_opportunities(
            impact_analysis["potential_changes"]
        )
        
        return impact_analysis

    async def _generate_evolution_plan(self, impact_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a plan for evolving the lore."""
        evolution_plan = {
            "phases": [],
            "dependencies": [],
            "timeline": [],
            "resource_requirements": {},
            "risk_mitigation": []
        }
        
        # Generate phases based on impact analysis
        for component in impact_analysis["affected_components"]:
            phase = await self._create_evolution_phase(component)
            evolution_plan["phases"].append(phase)
        
        # Determine dependencies between phases
        evolution_plan["dependencies"] = await self._determine_phase_dependencies(
            evolution_plan["phases"]
        )
        
        # Create timeline
        evolution_plan["timeline"] = await self._create_evolution_timeline(
            evolution_plan["phases"],
            evolution_plan["dependencies"]
        )
        
        # Calculate resource requirements
        evolution_plan["resource_requirements"] = await self._calculate_resource_requirements(
            evolution_plan["phases"]
        )
        
        # Generate risk mitigation strategies
        evolution_plan["risk_mitigation"] = await self._generate_risk_mitigation(
            impact_analysis["risk_assessment"]
        )
        
        return evolution_plan

    async def _apply_evolution_changes(self, evolution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the evolution changes according to the plan."""
        evolution_result = {
            "changes": [],
            "success": True,
            "errors": [],
            "metrics": {}
        }
        
        # Apply each phase in the timeline
        for phase in evolution_plan["timeline"]:
            try:
                phase_result = await self._apply_evolution_phase(phase)
                evolution_result["changes"].extend(phase_result.get("changes", []))
                
                if not phase_result.get("success", False):
                    evolution_result["errors"].append({
                        "phase": phase["id"],
                        "error": phase_result.get("error")
                    })
                    evolution_result["success"] = False
                
                # Update metrics
                evolution_result["metrics"][phase["id"]] = phase_result.get("metrics", {})
                
            except Exception as e:
                evolution_result["errors"].append({
                    "phase": phase["id"],
                    "error": str(e)
                })
                evolution_result["success"] = False
        
        return evolution_result

    async def _validate_evolution_results(self, evolution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the results of the evolution."""
        validation_result = {
            "status": "success",
            "issues": [],
            "metrics": {},
            "recommendations": []
        }
        
        # Validate each change
        for change in evolution_result["changes"]:
            change_validation = await self._validate_evolution_change(change)
            
            if not change_validation.get("valid", False):
                validation_result["status"] = "partial"
                validation_result["issues"].append({
                    "change_id": change.get("id"),
                    "issues": change_validation.get("issues", [])
                })
        
        # Calculate validation metrics
        validation_result["metrics"] = await self._calculate_validation_metrics(
            evolution_result["changes"]
        )
        
        # Generate recommendations if needed
        if validation_result["status"] != "success":
            validation_result["recommendations"] = await self._generate_validation_recommendations(
                validation_result["issues"]
            )
        
        return validation_result

    async def _analyze_evolution_triggers(self) -> List[Dict[str, Any]]:
        """Analyze and detect evolution triggers."""
        triggers = []
        
        # Get current lore state
        current_lore = await self.lore_manager.get_world_lore()
        
        # Check for various trigger types
        triggers.extend(await self._check_narrative_triggers(current_lore))
        triggers.extend(await self._check_character_triggers(current_lore))
        triggers.extend(await self._check_environment_triggers(current_lore))
        triggers.extend(await self._check_conflict_triggers(current_lore))
        
        # Prioritize triggers
        prioritized_triggers = await self._prioritize_triggers(triggers)
        
        # Update active triggers
        self.active_triggers = {t["id"] for t in prioritized_triggers}
        
        return prioritized_triggers

    async def _check_narrative_triggers(self, current_lore: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for narrative-based evolution triggers."""
        triggers = []
        
        # Check story progression
        story_progression = current_lore.get("story_progression", {})
        if story_progression.get("milestone_reached"):
            triggers.append({
                "id": f"narrative_{datetime.utcnow().timestamp()}",
                "type": "narrative_milestone",
                "priority": "high",
                "data": {
                    "milestone": story_progression["current_milestone"],
                    "impact_level": story_progression.get("impact_level", 1)
                }
            })
        
        # Check plot consistency
        plot_issues = await self._check_plot_consistency(current_lore)
        if plot_issues:
            triggers.append({
                "id": f"plot_{datetime.utcnow().timestamp()}",
                "type": "plot_inconsistency",
                "priority": "medium",
                "data": {
                    "issues": plot_issues,
                    "affected_components": self._get_affected_components(plot_issues)
                }
            })
        
        return triggers

    async def _check_character_triggers(self, current_lore: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for character-based evolution triggers."""
        triggers = []
        
        # Check character development
        for npc in current_lore.get("npcs", []):
            if npc.get("development_milestone_reached"):
                triggers.append({
                    "id": f"character_{npc['id']}_{datetime.utcnow().timestamp()}",
                    "type": "character_development",
                    "priority": "medium",
                    "data": {
                        "npc_id": npc["id"],
                        "milestone": npc["development_milestone"],
                        "impact_level": npc.get("development_impact", 1)
                    }
                })
        
        # Check relationship changes
        relationship_changes = await self._check_relationship_changes(current_lore)
        if relationship_changes:
            triggers.append({
                "id": f"relationships_{datetime.utcnow().timestamp()}",
                "type": "relationship_change",
                "priority": "medium",
                "data": {
                    "changes": relationship_changes,
                    "affected_npcs": self._get_affected_npcs(relationship_changes)
                }
            })
        
        return triggers

    async def _check_environment_triggers(self, current_lore: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for environment-based evolution triggers."""
        triggers = []
        
        # Check location changes
        for location in current_lore.get("locations", []):
            if location.get("significant_change"):
                triggers.append({
                    "id": f"location_{location['id']}_{datetime.utcnow().timestamp()}",
                    "type": "location_change",
                    "priority": "medium",
                    "data": {
                        "location_id": location["id"],
                        "change_type": location["change_type"],
                        "impact_level": location.get("change_impact", 1)
                    }
                })
        
        # Check world events
        world_events = await self._check_world_events(current_lore)
        if world_events:
            triggers.append({
                "id": f"world_{datetime.utcnow().timestamp()}",
                "type": "world_event",
                "priority": "high",
                "data": {
                    "events": world_events,
                    "global_impact": self._calculate_global_impact(world_events)
                }
            })
        
        return triggers

    async def _check_conflict_triggers(self, current_lore: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for conflict-based evolution triggers."""
        triggers = []
        
        # Check active conflicts
        for conflict in current_lore.get("conflicts", []):
            if conflict.get("resolution_phase_reached"):
                triggers.append({
                    "id": f"conflict_{conflict['id']}_{datetime.utcnow().timestamp()}",
                    "type": "conflict_resolution",
                    "priority": "high",
                    "data": {
                        "conflict_id": conflict["id"],
                        "resolution_phase": conflict["resolution_phase"],
                        "impact_level": conflict.get("resolution_impact", 1)
                    }
                })
        
        # Check faction changes
        faction_changes = await self._check_faction_changes(current_lore)
        if faction_changes:
            triggers.append({
                "id": f"faction_{datetime.utcnow().timestamp()}",
                "type": "faction_change",
                "priority": "medium",
                "data": {
                    "changes": faction_changes,
                    "affected_factions": self._get_affected_factions(faction_changes)
                }
            })
        
        return triggers

    async def _prioritize_triggers(self, triggers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize triggers based on various factors."""
        prioritized = []
        
        # Sort triggers by priority and impact
        for trigger in triggers:
            priority_score = self._calculate_trigger_priority(trigger)
            trigger["priority_score"] = priority_score
            prioritized.append(trigger)
        
        # Sort by priority score
        prioritized.sort(key=lambda x: x["priority_score"], reverse=True)
        
        return prioritized

    def _calculate_trigger_priority(self, trigger: Dict[str, Any]) -> float:
        """Calculate priority score for a trigger."""
        priority_weights = {
            "high": 3.0,
            "medium": 2.0,
            "low": 1.0
        }
        
        base_priority = priority_weights.get(trigger.get("priority", "low"), 1.0)
        impact_level = trigger.get("data", {}).get("impact_level", 1)
        
        # Additional factors
        time_factor = 1.0  # Could be adjusted based on time since last evolution
        dependency_factor = 1.0  # Could be adjusted based on dependencies
        
        return base_priority * impact_level * time_factor * dependency_factor

# Create a singleton instance for easy access
dynamic_generator = DynamicLoreGenerator()
